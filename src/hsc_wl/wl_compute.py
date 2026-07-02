"""Core dsigma weak-lensing computation utilities.

This module provides reusable building blocks for HSC weak-lensing analysis.
It is fully self-contained (no dependency on ``initial.py``) and can be
imported from any script or notebook in the project.
"""

import glob
import logging
import os
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18
from astropy.table import Table
from dsigma.helpers import dsigma_table
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density
from dsigma.surveys import hsc as hsc_survey

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
    """Return the first column name from *candidates* that exists in *cols*.

    Parameters
    ----------
    cols : list[str]
        Available column names.
    candidates : list[str]
        Preferred names, tried in order.

    Returns
    -------
    str or None
        The first match, or ``None`` if nothing matches.
    """
    for c in candidates:
        if c in cols:
            return c
    return None


def pick_required_column(cols, candidates, description):
    """Like :func:`pick_column` but raises on failure.

    Parameters
    ----------
    cols : list[str]
        Available column names.
    candidates : list[str]
        Preferred names, tried in order.
    description : str
        Human-readable label for the error message.

    Returns
    -------
    str
        The first match.

    Raises
    ------
    KeyError
        If none of *candidates* is found.
    """
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

    # Use only lenses with non-zero lens-source pair counts for clustering weights.
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

# Default rp bin parameters (matching Y3 defaults in run_hsc_wl.py).
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

    This function encapsulates the full source-loading logic that was
    previously embedded in ``run_analysis``: column auto-detection,
    ``dsigma_table()`` conversion, selection-bias correction, b-mode mask
    filtering, and tomographic redshift handling.

    Parameters
    ----------
    source_file : str or Path
        Path to the source catalogue FITS file.
    source_version : {'Y3', 'Y1'}, optional
        Source catalogue version (default ``'Y3'``).
    source_survey : str, optional
        Survey name passed to ``dsigma_table`` (default ``'hsc'``).
    source_nz_file : str or Path or None, optional
        Path to the n(z) FITS file.  Required when *source_version* is ``'Y3'``
        (tomographic mode).
    source_calib_file : str or Path or None, optional
        Path to the calibration catalogue (used for Y1/S16A).

    Returns
    -------
    table_s : astropy.table.Table
        Prepared source table.
    table_c : astropy.table.Table or None
        Calibration table (``None`` when not applicable).
    table_n : astropy.table.Table or None
        Redshift distribution table (``None`` for non-tomographic runs).
    rp_bins_default : numpy.ndarray
        Default projected-radius bin edges (log-spaced, Mpc).
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

    # ---- auto-detect column names ----
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

    # mag_A is only used/required for Y3+ selection bias correction
    if source_version in ("Y3", "PDR3", "S19A"):
        source_mag_col = pick_required_column(
            source_cols,
            ["i_apertureflux_10_mag", "aperture_mag", "mag_A"],
            "source aperture magnitude column",
        )
        dsigma_table_kwargs["mag_A"] = source_mag_col

    # Apply b-mode mask if available
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

    # dsigma sets 'z_low': 'photoz_err68_min' by default for S16A/Y1, so we
    # must override it if our catalog uses a different name.
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

    # Re-flip e_2 for Y1 if it was already in standard format
    if source_version == "Y1":
        table_s["e_2"] = -table_s["e_2"]

    table_s["m_sel"] = hsc_survey.multiplicative_selection_bias(
        table_s, version=source_version
    )

    if tomography:
        # Remove galaxies with bimodal P(z)'s.
        table_s = table_s[table_s["z_bin"] > 0]
        # dsigma expects the first redshift bin to be 0, not 1.
        table_s["z_bin"] = table_s["z_bin"] - 1

        logger.info("[load] n(z): %s", nz_file)
        table_n = Table.read(nz_file)
        table_n.rename_column("Z_MID", "z")
        table_n["n"] = np.column_stack([table_n[f"BIN{i + 1}"] for i in range(4)])
        table_n.keep_columns(["z", "n"])

        # Assign each galaxy in the source catalog the mean redshift of the bin.
        # This is only used to determine which lens-source pairs to use.
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
    # Ensure all redshifts in table_l are not completely identical (which happens if there's only 1 lens)
    # to prevent scipy/dsigma cubic interpolation error: ValueError("Expect x to not have duplicates")
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

    # Ensure all redshifts in table_r are not completely identical to prevent SciPy interpolation duplicate x error
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

    # Drop lenses / randoms without any nearby source.
    table_l = table_l[np.sum(table_l["sum 1"], axis=1) > 0]
    table_r = table_r[np.sum(table_r["sum 1"], axis=1) > 0]
    return table_l, table_r


_DEFAULT_CORRECTIONS = {
    "photo_z_dilution_correction": False,
    "boost_correction": False,
    "scalar_shear_response_correction": True,
    "matrix_shear_response_correction": False,
    "shear_responsivity_correction": True,
    "random_subtraction": True,
    "selection_bias_correction": True,
}


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

    This function runs the full dsigma pipeline for **one** lens / random
    pair: ``precompute`` → zero-weight filtering → jackknife assignment →
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
        Override individual correction flags.  Keys are the same as those
        accepted by ``excess_surface_density`` (see
        ``_DEFAULT_CORRECTIONS``).  Any key not supplied falls back to the
        default.
    source_version : {'Y3', 'Y1'}, optional
        Source catalogue version, used only for logging (default ``'Y3'``).

    Returns
    -------
    dict
        ``'result_table'`` – astropy Table with the ΔΣ profile,
        ``'jk_cov'`` – jackknife covariance matrix,
        ``'n_lens'`` – number of lenses in the redshift bin,
        ``'z_median'`` – median lens redshift.
    """
    rp_bins = np.asarray(rp_bins, dtype=float)
    z_bins = np.asarray(z_bins, dtype=float)

    corr = _DEFAULT_CORRECTIONS.copy()
    if corrections is not None:
        corr.update(corrections)

    # ---- precompute ----
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

    # ---- jackknife ----
    logger.info("[jackknife] fields")
    _centers, _n_jk_use = assign_jackknife_fields_with_fallback(
        table_l, table_r, n_jackknife
    )

    # ---- redshift bin mask ----
    lo, hi = z_bins[0], z_bins[1]
    mask_l = (lo <= table_l["z"]) & (table_l["z"] < hi)
    mask_r = (lo <= table_r["z"]) & (table_r["z"] < hi)

    table_l_bin = table_l[mask_l]
    table_r_bin = table_r[mask_r]

    n_lens = len(table_l_bin)
    z_median = float(np.median(table_l_bin["z"])) if n_lens > 0 else np.nan

    # ---- stacking ----
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


def run_wl_analysis(
    run_label,
    run_profiles,
    source_version="Y3",
    njobs=12,
    comoving=False,
    lens_source_cut=0.1,
    n_jackknife=100,
    lens_survey="hsc",
    lens_rpmin=0.10,
    lens_rpmax=20.0,
    lens_n_rpbins=11,
    lens_linlog="log",
    lens_z_col="z",
    lens_ra_col="ra",
    lens_dec_col="dec",
    source_file=None,
    source_nz_file=None,
    source_calib_file=None,
    source_survey="hsc",
    corrections=None,
    root_path=None,
):
    """Run the lensing profile computation and write output FITS files."""
    if root_path is None:
        current_dir = Path.cwd().resolve()
        marker = "pyproject.toml"
        while True:
            if not current_dir or current_dir == current_dir.parent:
                break
            if (current_dir / marker).exists():
                root_path = current_dir
                break
            current_dir = current_dir.parent
        if root_path is None:
            raise FileNotFoundError("Could not find the project root.")

    # Assign default files if not provided
    if source_file is None:
        if source_version == "Y3":
            source_file = str(root_path / "data/hsc_y3.fits")
        elif source_version == "Y1":
            source_file = str(root_path / "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_source.fits")
        else:
            raise ValueError(f"Unsupported source_version: {source_version}")

    if source_nz_file is None and source_version == "Y3":
        source_nz_file = str(root_path / "data/nz.fits")

    if source_calib_file is None and source_version == "Y1":
        source_calib_file = str(root_path / "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_calib.fits")

    run_paths = run_profiles[run_label]
    save_root = run_paths["save_root"]

    lens_file = Path(save_root) / f"prepare/{run_label}_lenses.fits"
    random_file = Path(save_root) / f"prepare/{run_label}_randoms.fits"

    if not lens_file.exists():
        raise FileNotFoundError(f"Unified lens file not found: {lens_file}")
    if not random_file.exists():
        raise FileNotFoundError(f"Unified random file not found: {random_file}")

    savepath = Path(save_root) / source_version / "dsigma"
    savepath.mkdir(parents=True, exist_ok=True)

    table_s, table_c, table_n, _ = load_and_prepare_source(
        source_file=source_file,
        source_version=source_version,
        source_survey=source_survey,
        source_nz_file=source_nz_file,
        source_calib_file=source_calib_file,
    )

    rp_bins = np.logspace(np.log10(lens_rpmin), np.log10(lens_rpmax), lens_n_rpbins + 1)

    logger.info("loading global lens and random files...")
    global_table_l_raw = Table.read(str(lens_file))
    global_table_r_raw = Table.read(str(random_file))

    global_table_l = dsigma_table(
        global_table_l_raw,
        "lens",
        z=lens_z_col,
        ra=lens_ra_col,
        dec=lens_dec_col,
        w_sys=1.0,
    )
    if "bin_id" not in global_table_l.colnames:
        global_table_l["bin_id"] = global_table_l_raw["bin_id"]

    global_table_r = dsigma_table(
        global_table_r_raw, "lens", z="z", ra="ra", dec="dec", w_sys=1.0
    )
    if "bin_id" not in global_table_r.colnames:
        global_table_r["bin_id"] = global_table_r_raw["bin_id"]

    logger.info(
        f"[precompute] global lenses ({len(global_table_l)}), randoms ({len(global_table_r)})"
    )
    global_table_l, global_table_r = precompute_catalogs(
        global_table_l,
        global_table_r,
        table_s,
        table_c,
        table_n,
        rp_bins,
        comoving,
        lens_source_cut,
        njobs,
    )
    logger.info("[precompute] global precompute finished")

    import json
    bin_meta_json = global_table_l_raw.meta.get("BIN_META", "[]")
    try:
        bin_metadata = json.loads(bin_meta_json)
    except json.JSONDecodeError:
        bin_metadata = []
        logger.warning("BIN_META header could not be parsed.")

    if not bin_metadata:
        unique_bids = np.unique(global_table_l["bin_id"])
        bin_metadata = [{"bin_id": bid, "bin_name": f"bin{bid}"} for bid in unique_bids]

    for bmeta in bin_metadata:
        bid = bmeta["bin_id"]
        bin_name = bmeta.get("bin_name", f"bin{bid}")
        logger.info(f"--- Processing {bin_name} (ID: {bid}) ---")

        table_l = global_table_l[global_table_l["bin_id"] == bid]
        table_r = global_table_r[global_table_r["bin_id"] == bid]

        logger.info(f"{bin_name} valid lenses: {len(table_l)}, randoms: {len(table_r)}")

        if len(table_l) == 0:
            logger.info(f"Skipping {bin_name}: no valid lenses after precompute.")
            continue

        logger.info("[jackknife] fields")
        centers, n_jk_use = assign_jackknife_fields_with_fallback(
            table_l, table_r, n_jackknife
        )

        logger.info("[stack] ΔΣ by lens z-bin")
        z_bins = np.array(run_paths["lens_z_bins"])
        lo, hi = z_bins[0], z_bins[1]
        mL = (lo <= table_l["z"]) & (table_l["z"] < hi)
        mR = (lo <= table_r["z"]) & (table_r["z"] < hi)

        corr = {
            "photo_z_dilution_correction": False,
            "boost_correction": False,
            "scalar_shear_response_correction": True,
            "matrix_shear_response_correction": False,
            "shear_responsivity_correction": True,
            "random_subtraction": True,
            "selection_bias_correction": True,
        }
        if corrections is not None:
            corr.update(corrections)

        kwargs = corr.copy()
        kwargs["return_table"] = True
        kwargs["table_r"] = table_r[mR]

        result = excess_surface_density(table_l[mL], **kwargs)
        kwargs["return_table"] = False
        cov = jackknife_resampling(
            excess_surface_density,
            table_l[mL],
            **kwargs,
        )
        result["ds_err"] = np.sqrt(np.diag(cov))

        out_fits = (
            savepath / f"{source_survey.lower()}_{lens_survey or 'lenses'}_{bin_name}.fits"
        )

        from astropy.io import fits
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(result, name="PROFILE"),
                fits.ImageHDU(cov, name="JK_COV"),
            ]
        )
        hdul.writeto(out_fits, overwrite=True)
        logger.info(f"  wrote: {out_fits}")

    logger.info("[done]")

