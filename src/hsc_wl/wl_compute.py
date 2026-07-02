"""Core dsigma weak-lensing computation utilities.

This module provides reusable building blocks for HSC weak-lensing analysis.
It is fully self-contained (no dependency on ``initial.py``) and can be
imported from any script or notebook in the project.

The top-level entry point is :func:`run_pipeline`, which orchestrates the
``load source → load prepared lens/random → precompute → stack per bin →
save`` stages.  Each stage is also exposed as a standalone function so it
can be run individually from an interactive ``# %%`` cell.
"""

import datetime
import glob
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits as fits_io
from astropy.table import Table
from dsigma.helpers import dsigma_table
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density
from dsigma.surveys import hsc as hsc_survey

from hsc_wl.config import CorrectionConfig, SourceConfig, WLConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path / column helpers
# ---------------------------------------------------------------------------


def find_one(path_or_pattern, description):
    """Return the first existing path matching *path_or_pattern*.

    Parameters
    ----------
    path_or_pattern : str or Path
        A literal path or a glob pattern (containing ``*``, ``?``, ``[``, or ``]``).
    description : str
        Human-readable description used in the ``FileNotFoundError`` message.

    Returns
    -------
    Path
        The first existing match.

    Raises
    ------
    FileNotFoundError
        If no match is found.
    """
    paths = (
        sorted(glob.glob(str(path_or_pattern)))
        if any(ch in str(path_or_pattern) for ch in "*?[]")
        else [str(path_or_pattern)]
    )
    for p in paths:
        if os.path.exists(p):
            return Path(p)
    raise FileNotFoundError(f"Could not find {description}: {path_or_pattern}")


def pick_column(cols, candidates):
    """Return the first column name from *candidates* that exists in *cols*."""
    for c in candidates:
        if c in cols:
            return c
    return None


def pick_required_column(cols, candidates, description):
    """Like :func:`pick_column` but raises on failure."""
    col = pick_column(cols, candidates)
    if col is None:
        raise KeyError(f"Could not find {description}. Tried: {', '.join(candidates)}")
    return col


# ---------------------------------------------------------------------------
# Jackknife helpers
# ---------------------------------------------------------------------------


def assign_jackknife_fields_with_fallback(
    table_l, table_r, n_jk_requested, distance_threshold=1.0
):
    """Assign jackknife fields, reducing *n_jk* or relaxing connectivity as needed.

    Parameters
    ----------
    table_l : astropy.table.Table
        Lens table (after ``precompute``).
    table_r : astropy.table.Table
        Random table (after ``precompute``).
    n_jk_requested : int
        Desired number of jackknife regions.
    distance_threshold : float, optional
        Initial distance threshold in degrees (default ``1.0``).

    Returns
    -------
    centers : array
        Jackknife field centres.
    n_jk_use : int
        Actual number of jackknife fields used.

    Raises
    ------
    ValueError
        If the input tables are too small.
    RuntimeError
        If no valid jackknife configuration can be found.
    """
    if len(table_l) < 2:
        raise ValueError(
            f"Need at least 2 lenses after precompute filtering for jackknife; got {len(table_l)}"
        )

    weights = np.sum(table_l["sum 1"], axis=1)
    n_positive_weight = int(np.sum(weights > 0))
    if n_positive_weight < 2:
        raise ValueError(
            "Need at least 2 lenses with positive jackknife weights after filtering; "
            f"got {n_positive_weight}"
        )

    n_jk_start = min(int(n_jk_requested), len(table_l), n_positive_weight)
    last_error = None
    distance_thresholds = [float(distance_threshold)]
    while distance_thresholds[-1] < 180.0:
        next_threshold = min(distance_thresholds[-1] * 2.0, 180.0)
        if next_threshold == distance_thresholds[-1]:
            break
        distance_thresholds.append(next_threshold)

    for n_jk_try in range(n_jk_start, 1, -1):
        for distance_threshold_try in distance_thresholds:
            try:
                centers = compute_jackknife_fields(
                    table_l,
                    n_jk_try,
                    distance_threshold=distance_threshold_try,
                    weights=weights,
                )
                compute_jackknife_fields(table_r, centers)
                if n_jk_try < n_jk_requested or distance_threshold_try != float(
                    distance_threshold
                ):
                    logger.info(
                        "[jackknife] requested n_jk=%d, using n_jk=%d, "
                        "distance_threshold=%.1f deg",
                        n_jk_requested,
                        n_jk_try,
                        distance_threshold_try,
                    )
                return centers, n_jk_try
            except ValueError as err:
                err_text = str(err)
                last_error = err
                if (
                    "larger sample than population" in err_text
                    or "0 sample(s)" in err_text
                ):
                    continue
                raise
            except RuntimeError as err:
                last_error = err
                continue

    raise RuntimeError(
        "Could not assign jackknife fields with n_jk >= 2 after filtering. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Source catalogue loading
# ---------------------------------------------------------------------------

_DEFAULT_RP_MIN = 0.10
_DEFAULT_RP_MAX = 20.0
_DEFAULT_RP_NBINS = 11


def load_and_prepare_source(
    source_file,
    source_version="Y3",
    source_survey="hsc",
    source_nz_file=None,
    source_calib_file=None,
):
    """Load and prepare the source (and optional calibration / n(z)) catalogues.

    This function encapsulates the full source-loading logic: column
    auto-detection, ``dsigma_table()`` conversion, selection-bias
    correction, b-mode mask filtering, and tomographic redshift handling.
    """
    src_file = find_one(source_file, "source catalog")
    calib_file = (
        find_one(source_calib_file, "calibration catalog")
        if source_calib_file
        else None
    )

    tomography = source_version in ("Y3", "PDR3", "S19A")
    nz_file = find_one(source_nz_file, "n(z) file") if tomography else None

    logger.info("[load] sources: %s", src_file)
    if calib_file is not None:
        logger.info("[load] calibration: %s", calib_file)

    table_s = Table.read(src_file)
    table_c = Table.read(calib_file) if calib_file is not None else None

    source_cols = table_s.colnames
    source_ra_col = pick_required_column(
        source_cols, ["i_ra", "RA", "ra"], "source right ascension column"
    )
    source_dec_col = pick_required_column(
        source_cols, ["i_dec", "Dec", "DEC", "dec"], "source declination column"
    )
    source_e1_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_e1", "e_1", "e1"],
        "source e_1 column",
    )
    source_e2_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_e2", "e_2", "e2"],
        "source e_2 column",
    )
    source_w_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_derived_weight", "weight", "w"],
        "source weight column",
    )
    source_m_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_derived_shear_bias_m", "m_corr", "m"],
        "source shear bias column",
    )
    source_e_rms_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_derived_rms_e", "e_rms"],
        "source e_rms column",
    )
    source_r2_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_resolution", "resolution", "R_2"],
        "source resolution column",
    )

    dsigma_table_kwargs = dict(
        ra=source_ra_col,
        dec=source_dec_col,
        e_1=source_e1_col,
        e_2=source_e2_col,
        w=source_w_col,
        m=source_m_col,
        e_rms=source_e_rms_col,
        R_2=source_r2_col,
    )

    if source_version in ("Y3", "PDR3", "S19A"):
        source_mag_col = pick_required_column(
            source_cols,
            ["i_apertureflux_10_mag", "aperture_mag", "mag_A"],
            "source aperture magnitude column",
        )
        dsigma_table_kwargs["mag_A"] = source_mag_col

    if "b_mode_mask" in source_cols:
        table_s = table_s[table_s["b_mode_mask"] == 1]

    if tomography:
        source_zbin_col = pick_required_column(
            source_cols, ["hsc_y3_zbin", "z_bin"], "source redshift-bin column"
        )
        dsigma_table_kwargs["z_bin"] = source_zbin_col
    else:
        source_z_col = pick_required_column(
            source_cols, ["z", "photoz_best"], "source photo-z column"
        )
        dsigma_table_kwargs["z"] = source_z_col

    z_low_col = pick_column(source_cols, ["z_low", "photoz_err68_min"])
    if z_low_col:
        dsigma_table_kwargs["z_low"] = z_low_col
    elif source_version == "Y1":
        dsigma_table_kwargs["z_low"] = dsigma_table_kwargs.get("z", "z")

    table_s = dsigma_table(
        table_s,
        "source",
        survey=source_survey.upper(),
        version=source_version,
        **dsigma_table_kwargs,
    )

    if source_version == "Y1":
        table_s["e_2"] = -table_s["e_2"]

    table_s["m_sel"] = hsc_survey.multiplicative_selection_bias(
        table_s, version=source_version
    )

    if tomography:
        table_s = table_s[table_s["z_bin"] > 0]
        table_s["z_bin"] = table_s["z_bin"] - 1

        logger.info("[load] n(z): %s", nz_file)
        table_n = Table.read(nz_file)
        table_n.rename_column("Z_MID", "z")
        table_n["n"] = np.column_stack([table_n[f"BIN{i + 1}"] for i in range(4)])
        table_n.keep_columns(["z", "n"])

        table_s["z"] = np.sum(table_n["z"][:, np.newaxis] * table_n["n"], axis=0)[
            table_s["z_bin"]
        ]
    else:
        table_n = None

    rp_bins_default = np.logspace(
        np.log10(_DEFAULT_RP_MIN), np.log10(_DEFAULT_RP_MAX), _DEFAULT_RP_NBINS + 1
    )

    return table_s, table_c, table_n, rp_bins_default


# ---------------------------------------------------------------------------
# Per-bin lensing profile computation
# ---------------------------------------------------------------------------

_DEFAULT_CORRECTIONS = {
    "photo_z_dilution_correction": False,
    "boost_correction": False,
    "scalar_shear_response_correction": True,
    "matrix_shear_response_correction": False,
    "shear_responsivity_correction": True,
    "random_subtraction": True,
    "selection_bias_correction": True,
}


def precompute_catalogs(
    table_l,
    table_r,
    table_s,
    table_c,
    table_n,
    rp_bins,
    comoving=False,
    lens_source_cut=0.1,
    n_jobs=12,
):
    """Run dsigma precompute and filter out objects without any nearby sources."""
    orig_len_l = len(table_l)
    modified_l = False
    if orig_len_l > 0 and np.all(table_l["z"] == table_l["z"][0]):
        table_l.add_row(table_l[0])
        table_l["z"][-1] = table_l["z"][0] + 1e-4
        modified_l = True

    logger.info("[precompute] lenses (%d rows)", len(table_l))
    precompute(
        table_l,
        table_s,
        rp_bins,
        table_c=table_c,
        cosmology=Planck18,
        comoving=comoving,
        table_n=table_n,
        lens_source_cut=lens_source_cut,
        progress_bar=True,
        n_jobs=n_jobs,
    )
    if modified_l:
        table_l = table_l[:orig_len_l]

    orig_len_r = len(table_r)
    modified_r = False
    if orig_len_r > 0 and np.all(table_r["z"] == table_r["z"][0]):
        table_r.add_row(table_r[0])
        table_r["z"][-1] = table_r["z"][0] + 1e-4
        modified_r = True

    logger.info("[precompute] randoms (%d rows)", len(table_r))
    precompute(
        table_r,
        table_s,
        rp_bins,
        table_c=table_c,
        cosmology=Planck18,
        comoving=comoving,
        table_n=table_n,
        lens_source_cut=lens_source_cut,
        progress_bar=True,
        n_jobs=n_jobs,
    )
    if modified_r:
        table_r = table_r[:orig_len_r]

    table_l = table_l[np.sum(table_l["sum 1"], axis=1) > 0]
    table_r = table_r[np.sum(table_r["sum 1"], axis=1) > 0]
    return table_l, table_r


def compute_single_bin_profile(
    table_l,
    table_r,
    table_s,
    table_c,
    table_n,
    rp_bins,
    z_bins,
    n_jackknife=100,
    n_jobs=12,
    comoving=False,
    lens_source_cut=0.1,
    corrections=None,
    source_version="Y3",
):
    """Compute a stacked ΔΣ profile for a single lens bin.

    Runs the full dsigma pipeline for **one** lens / random pair:
    ``precompute`` → zero-weight filtering → jackknife assignment →
    redshift-bin masking → ``excess_surface_density`` + jackknife
    resampling.

    Parameters
    ----------
    table_l : astropy.table.Table
        Lens table (already converted via ``dsigma_table``).
    table_r : astropy.table.Table
        Random table (already converted via ``dsigma_table``).
    table_s : astropy.table.Table
        Source table (from :func:`load_and_prepare_source`).
    table_c : astropy.table.Table or None
        Calibration table.
    table_n : astropy.table.Table or None
        Redshift distribution table.
    rp_bins : array-like
        Projected-radius bin edges (Mpc).
    z_bins : array-like
        Two-element array ``[z_lo, z_hi]`` defining the lens redshift cut.
    n_jackknife : int, optional
        Requested number of jackknife regions (default ``100``).
    n_jobs : int, optional
        Number of parallel workers for ``precompute`` (default ``12``).
    comoving : bool, optional
        Use comoving coordinates (default ``False``).
    lens_source_cut : float, optional
        Minimum lens-source redshift separation (default ``0.1``).
    corrections : dict or None, optional
        Override individual correction flags (see ``_DEFAULT_CORRECTIONS``).
    source_version : {'Y3', 'Y1'}, optional
        Source catalogue version, used only for logging (default ``'Y3'``).

    Returns
    -------
    dict
        ``'result_table'``, ``'jk_cov'``, ``'n_lens'``, ``'z_median'``.
    """
    rp_bins = np.asarray(rp_bins, dtype=float)
    z_bins = np.asarray(z_bins, dtype=float)

    corr = _DEFAULT_CORRECTIONS.copy()
    if corrections is not None:
        corr.update(corrections)

    if "sum 1" not in table_l.colnames:
        table_l, table_r = precompute_catalogs(
            table_l,
            table_r,
            table_s,
            table_c,
            table_n,
            rp_bins,
            comoving=comoving,
            lens_source_cut=lens_source_cut,
            n_jobs=n_jobs,
        )

    logger.info("[jackknife] fields")
    _centers, _n_jk_use = assign_jackknife_fields_with_fallback(
        table_l, table_r, n_jackknife
    )

    lo, hi = z_bins[0], z_bins[1]
    mask_l = (lo <= table_l["z"]) & (table_l["z"] < hi)
    mask_r = (lo <= table_r["z"]) & (table_r["z"] < hi)

    table_l_bin = table_l[mask_l]
    table_r_bin = table_r[mask_r]

    n_lens = len(table_l_bin)
    z_median = float(np.median(table_l_bin["z"])) if n_lens > 0 else np.nan

    kwargs = corr.copy()
    kwargs["return_table"] = True
    kwargs["table_r"] = table_r_bin

    logger.info(
        "[stack] ΔΣ  z=[%.2f, %.2f)  n_lens=%d  z_median=%.3f",
        lo,
        hi,
        n_lens,
        z_median,
    )

    result = excess_surface_density(table_l_bin, **kwargs)

    kwargs["return_table"] = False
    cov = jackknife_resampling(
        excess_surface_density,
        table_l_bin,
        **kwargs,
    )
    result["ds_err"] = np.sqrt(np.diag(cov))

    return {
        "result_table": result,
        "jk_cov": cov,
        "n_lens": n_lens,
        "z_median": z_median,
    }


# ---------------------------------------------------------------------------
# Stage result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceData:
    """Output of :func:`load_source`."""

    table_s: Table
    table_c: Table | None
    table_n: Table | None
    rp_bins: np.ndarray
    version: str


@dataclass(frozen=True, slots=True)
class PreparedTables:
    """Output of :func:`load_prepared_tables`."""

    lens: Table
    random: Table
    bin_metadata: list


@dataclass(frozen=True, slots=True)
class PrecomputedTables:
    """Output of :func:`precompute_global`."""

    lens: Table
    random: Table
    bin_metadata: list


@dataclass(frozen=True, slots=True)
class BinProfile:
    """Output of :func:`stack_one_bin`."""

    bin_id: int
    bin_name: str
    result_table: Table
    jk_cov: np.ndarray
    n_lens: int
    z_median: float


# ---------------------------------------------------------------------------
# Root / path resolution
# ---------------------------------------------------------------------------


def _find_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    current_dir = Path.cwd().resolve()
    while True:
        if not current_dir or current_dir == current_dir.parent:
            break
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError("Could not find the project root.")


def default_source_path(version: str, root: Path, kind: str) -> Path:
    """Return the default source / n(z) / calib file path for *version*.

    *kind* is one of ``"source"``, ``"nz"``, ``"calib"``.
    """
    root = Path(root)
    if version in ("Y3", "PDR3", "S19A"):
        if kind == "source":
            return root / "data/hsc_y3.fits"
        if kind == "nz":
            return root / "data/nz.fits"
    elif version == "Y1":
        if kind == "source":
            return (
                root / "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_source.fits"
            )
        if kind == "calib":
            return (
                root / "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_calib.fits"
            )
    raise ValueError(f"No default {kind} file for source version {version!r}.")


def _resolve_source_files(
    src: SourceConfig, root: Path
) -> tuple[Path, Path | None, Path | None]:
    source_file = (
        Path(src.file)
        if src.file is not None
        else default_source_path(src.version, root, "source")
    )
    nz_file = (
        Path(src.nz_file)
        if src.nz_file is not None
        else (
            default_source_path(src.version, root, "nz")
            if src.version in ("Y3", "PDR3", "S19A")
            else None
        )
    )
    calib_file = (
        Path(src.calib_file)
        if src.calib_file is not None
        else (
            default_source_path(src.version, root, "calib")
            if src.version == "Y1"
            else None
        )
    )
    return source_file, nz_file, calib_file


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def load_source(cfg: WLConfig, root: Path | None = None) -> SourceData:
    """Stage 1: load and prepare the source catalogue.

    The result is independent of the lens catalog and can be reused across
    runs that share the same :class:`SourceConfig`.
    """
    root = _find_root(root)
    src = cfg.source
    source_file, nz_file, calib_file = _resolve_source_files(src, root)

    table_s, table_c, table_n, rp_bins_default = load_and_prepare_source(
        source_file=str(source_file),
        source_version=src.version,
        source_survey=src.survey,
        source_nz_file=str(nz_file) if nz_file is not None else None,
        source_calib_file=str(calib_file) if calib_file is not None else None,
    )
    return SourceData(
        table_s=table_s,
        table_c=table_c,
        table_n=table_n,
        rp_bins=rp_bins_default,
        version=src.version,
    )


def load_prepared_tables(cfg: WLConfig, root: Path | None = None) -> PreparedTables:
    """Stage 2: read the unified lens/random FITS written by the prepare stage."""
    root = _find_root(root)
    save_root = cfg.resolved_save_root(root)
    lens_file = save_root / "prepare" / f"{cfg.label}_lenses.fits"
    random_file = save_root / "prepare" / f"{cfg.label}_randoms.fits"

    if not lens_file.exists():
        raise FileNotFoundError(f"Unified lens file not found: {lens_file}")
    if not random_file.exists():
        raise FileNotFoundError(f"Unified random file not found: {random_file}")

    logger.info("loading global lens and random files...")
    global_table_l_raw = Table.read(str(lens_file))
    global_table_r_raw = Table.read(str(random_file))

    bin_meta_json = global_table_l_raw.meta.get("BIN_META", "[]")
    try:
        bin_metadata = json.loads(bin_meta_json)
    except json.JSONDecodeError:
        bin_metadata = []
        logger.warning("BIN_META header could not be parsed.")

    if not bin_metadata:
        unique_bids = np.unique(global_table_l_raw["bin_id"]).tolist()
        bin_metadata = [{"bin_id": bid, "bin_name": f"bin{bid}"} for bid in unique_bids]

    return PreparedTables(
        lens=global_table_l_raw, random=global_table_r_raw, bin_metadata=bin_metadata
    )


def precompute_global(
    prepared: PreparedTables,
    source: SourceData,
    cfg: WLConfig,
) -> PrecomputedTables:
    """Stage 3: convert to dsigma format and run precompute for all bins at once."""
    rp_bins = np.logspace(
        np.log10(cfg.rp.rp_min), np.log10(cfg.rp.rp_max), cfg.rp.n_bins + 1
    )

    global_table_l = dsigma_table(
        prepared.lens, "lens", z="z", ra="ra", dec="dec", w_sys=1.0
    )
    if "bin_id" not in global_table_l.colnames:
        global_table_l["bin_id"] = prepared.lens["bin_id"]

    global_table_r = dsigma_table(
        prepared.random, "lens", z="z", ra="ra", dec="dec", w_sys=1.0
    )
    if "bin_id" not in global_table_r.colnames:
        global_table_r["bin_id"] = prepared.random["bin_id"]

    logger.info(
        "[precompute] global lenses (%d), randoms (%d)",
        len(global_table_l),
        len(global_table_r),
    )
    global_table_l, global_table_r = precompute_catalogs(
        global_table_l,
        global_table_r,
        source.table_s,
        source.table_c,
        source.table_n,
        rp_bins,
        cfg.comoving,
        cfg.lens_source_cut,
        cfg.n_jobs,
    )
    logger.info("[precompute] global precompute finished")

    return PrecomputedTables(
        lens=global_table_l,
        random=global_table_r,
        bin_metadata=prepared.bin_metadata,
    )


def stack_one_bin(
    pre: PrecomputedTables,
    source: SourceData,
    cfg: WLConfig,
    bin_meta: dict,
    rp_bins: np.ndarray,
) -> BinProfile | None:
    """Stage 4 (per bin): jackknife + stack + jackknife covariance."""
    bid = bin_meta["bin_id"]
    bin_name = bin_meta.get("bin_name", f"bin{bid}")

    table_l = pre.lens[pre.lens["bin_id"] == bid]
    table_r = pre.random[pre.random["bin_id"] == bid]

    logger.info(
        "--- Processing %s (ID: %s): %d lenses ---", bin_name, bid, len(table_l)
    )
    if len(table_l) == 0:
        logger.info("Skipping %s: no valid lenses after precompute.", bin_name)
        return None

    logger.info("[jackknife] fields")
    assign_jackknife_fields_with_fallback(table_l, table_r, cfg.n_jackknife)

    z_lo, z_hi = cfg.lens.redshift_range
    mask_l = (z_lo <= table_l["z"]) & (table_l["z"] < z_hi)
    mask_r = (z_lo <= table_r["z"]) & (table_r["z"] < z_hi)

    n_lens = int(np.sum(mask_l))
    z_median = float(np.median(table_l["z"][mask_l])) if n_lens > 0 else np.nan

    corr = CorrectionConfig().to_dsigma_kwargs()
    corr.update(cfg.corrections.to_dsigma_kwargs())

    kwargs = corr.copy()
    kwargs["return_table"] = True
    kwargs["table_r"] = table_r[mask_r]

    logger.info(
        "[stack] ΔΣ  z=[%.2f, %.2f)  n_lens=%d  z_median=%.3f",
        z_lo,
        z_hi,
        n_lens,
        z_median,
    )
    result = excess_surface_density(table_l[mask_l], **kwargs)

    kwargs["return_table"] = False
    cov = jackknife_resampling(excess_surface_density, table_l[mask_l], **kwargs)
    result["ds_err"] = np.sqrt(np.diag(cov))

    return BinProfile(
        bin_id=bid,
        bin_name=bin_name,
        result_table=result,
        jk_cov=cov,
        n_lens=n_lens,
        z_median=z_median,
    )


def stack_per_bin(
    pre: PrecomputedTables,
    source: SourceData,
    cfg: WLConfig,
) -> list[BinProfile]:
    """Stage 4: stack ΔΣ for every bin in ``bin_metadata``."""
    rp_bins = np.logspace(
        np.log10(cfg.rp.rp_min), np.log10(cfg.rp.rp_max), cfg.rp.n_bins + 1
    )
    profiles: list[BinProfile] = []
    for bin_meta in pre.bin_metadata:
        prof = stack_one_bin(pre, source, cfg, bin_meta, rp_bins)
        if prof is not None:
            profiles.append(prof)
    return profiles


def save_profiles(
    profiles: list[BinProfile],
    cfg: WLConfig,
    root: Path | None = None,
) -> list[Path]:
    """Stage 5: write one FITS per bin under ``<save_root>/<source_version>/dsigma/``."""
    root = _find_root(root)
    save_root = cfg.resolved_save_root(root)
    savepath = save_root / cfg.source.version / "dsigma"
    savepath.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for prof in profiles:
        out_fits = (
            savepath
            / f"{cfg.source.survey.lower()}_{cfg.lens_survey}_{prof.bin_name}.fits"
        )
        hdul = fits_io.HDUList(
            [
                fits_io.PrimaryHDU(),
                fits_io.BinTableHDU(prof.result_table, name="PROFILE"),
                fits_io.ImageHDU(prof.jk_cov, name="JK_COV"),
            ]
        )
        hdul.writeto(out_fits, overwrite=True)
        logger.info("  wrote: %s", out_fits)
        written.append(out_fits)
    return written


# ---------------------------------------------------------------------------
# Manifest (reproducibility record)
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _config_to_serialisable(obj):
    """Recursively convert a dataclass / tuple / Path into JSON-safe types."""
    if hasattr(obj, "__dataclass_fields__"):
        return {
            k: _config_to_serialisable(getattr(obj, k))
            for k in obj.__dataclass_fields__
        }
    if isinstance(obj, (list, tuple)):
        return [_config_to_serialisable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_manifest(
    cfg: WLConfig,
    profiles: list[BinProfile],
    root: Path | None = None,
) -> Path:
    """Write ``<save_root>/manifest.json`` describing this run."""
    root = _find_root(root)
    save_root = cfg.resolved_save_root(root)
    save_root.mkdir(parents=True, exist_ok=True)

    import astropy
    import dsigma

    manifest = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "versions": {
            "dsigma": getattr(dsigma, "__version__", "unknown"),
            "astropy": astropy.__version__,
        },
        "config": _config_to_serialisable(cfg),
        "bins": [
            {
                "bin_id": p.bin_id,
                "bin_name": p.bin_name,
                "n_lens": p.n_lens,
                "z_median": p.z_median,
            }
            for p in profiles
        ],
    }
    manifest_path = save_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info("wrote manifest: %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    cfg: WLConfig,
    root: Path | None = None,
    *,
    write_manifest_file: bool = True,
) -> list[BinProfile]:
    """Run the full weak-lensing pipeline for *cfg*.

    Stages: ``load_source`` → ``load_prepared_tables`` →
    ``precompute_global`` → ``stack_per_bin`` → ``save_profiles``
    (→ ``write_manifest``).

    Parameters
    ----------
    cfg : WLConfig
        Fully-specified run configuration (typically an entry of
        :data:`hsc_wl.config.RUN_REGISTRY`).
    root : Path or None
        Project root.  ``None`` auto-detects via ``pyproject.toml``.
    write_manifest_file : bool
        If ``True`` (default) write ``manifest.json`` next to the outputs.

    Returns
    -------
    list[BinProfile]
        One profile per non-empty bin.
    """
    root = _find_root(root)
    logger.info("=== run_pipeline: %s ===", cfg.label)

    source = load_source(cfg, root)
    prepared = load_prepared_tables(cfg, root)
    pre = precompute_global(prepared, source, cfg)
    profiles = stack_per_bin(pre, source, cfg)
    save_profiles(profiles, cfg, root)
    if write_manifest_file:
        write_manifest(cfg, profiles, root)

    logger.info("[done] %s: %d bins", cfg.label, len(profiles))
    return profiles
