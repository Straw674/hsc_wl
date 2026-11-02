"""Configuration objects and registries for the HSC weak-lensing pipeline.

All run-time configuration is expressed as immutable :class:`~dataclasses.dataclass`
objects (``frozen=True``).

- :data:`LENS_REGISTRY` maps catalog identifiers to :class:`LensCatalogConfig` objects.
- :data:`RUN_REGISTRY` maps run labels (e.g. ``"cosine_4bin"``, ``"rz_diff_1bin"``)
  to fully-specified :class:`WLConfig` objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ColumnMapping",
    "LensCatalogConfig",
    "SourceConfig",
    "BinningConfig",
    "CorrectionConfig",
    "RPConfig",
    "WLConfig",
    "LENS_REGISTRY",
    "RUN_REGISTRY",
    "resolve_binning",
    "resolve_config",
    "resolve_run_dir",
    "get_latest_cluster_catalog",
    "replace",
]


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def get_latest_cluster_catalog(
    cluster_dir: str | Path = "/Users/xinq/cluster_finder/output/cluster",
) -> str:
    """Find the latest timestamped cluster catalog parquet file."""
    import glob

    pattern = str(Path(cluster_dir) / "cluster_catalog_*.parquet")
    files = glob.glob(pattern)
    if not files:
        return "/Users/xinq/cluster_finder/output/cluster/cluster_catalog_20260603_085729.parquet"
    return max(files, key=lambda p: Path(p).stat().st_mtime)


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """Column-name mapping for a lens catalog."""

    col_rank: str
    ra: str
    dec: str
    z: str
    z_spec: Optional[str] = None


@dataclass(frozen=True, slots=True)
class LensCatalogConfig:
    """Specification of where to find a lens catalog and how to interpret it."""

    label: str
    lens_path: str
    columns: ColumnMapping
    redshift_range: tuple[float, float]
    top_counts_factor: float = 1.0
    lens_format: Optional[str] = None
    fields: Optional[tuple[str, ...]] = None
    survey_overlap: Optional[str] = None
    ra_range: Optional[tuple[float, float]] = None
    dec_range: Optional[tuple[float, float]] = None
    area_deg2: Optional[float] = None
    redshift_type: Literal["photoz", "specz"] = "photoz"
    random_path: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Source (background galaxy) catalog configuration."""

    version: Literal["Y1", "Y3"] = "Y3"
    survey: str = "hsc"
    file: Optional[str] = None
    nz_file: Optional[str] = None
    calib_file: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BinningConfig:
    """How lenses are partitioned into bins."""

    mode: Literal["edges", "top_counts", "top_n"] = "top_counts"
    top_counts: tuple[int, ...] = (53, 196, 660, 1159)
    top_n: int = 500
    edges_richness: tuple[float, ...] = (6.0, 10.0, 20.0, 35.0, 120.0)
    edges_mass: tuple[float, ...] = (10.63, 10.8, 11.0, 11.2, 11.6)
    selection_order: Literal["asc", "desc"] = "desc"


@dataclass(frozen=True, slots=True)
class CorrectionConfig:
    """Toggles for the dsigma correction flags."""

    photo_z_dilution: bool = False
    boost: bool = False
    scalar_shear_response: bool = True
    matrix_shear_response: bool = False
    shear_responsivity: bool = True
    random_subtraction: bool = True
    selection_bias: bool = True

    def to_dsigma_kwargs(self) -> dict[str, bool]:
        """Return the dict of long-form flag names expected by dsigma."""
        return {
            "photo_z_dilution_correction": self.photo_z_dilution,
            "boost_correction": self.boost,
            "scalar_shear_response_correction": self.scalar_shear_response,
            "matrix_shear_response_correction": self.matrix_shear_response,
            "shear_responsivity_correction": self.shear_responsivity,
            "random_subtraction": self.random_subtraction,
            "selection_bias_correction": self.selection_bias,
        }


@dataclass(frozen=True, slots=True)
class RPConfig:
    """Projected-radius binning configuration."""

    rp_min: float = 0.10
    rp_max: float = 20.0
    n_bins: int = 11
    linlog: str = "log"


@dataclass(frozen=True, slots=True)
class WLConfig:
    """A fully-specified weak-lensing run."""

    label: str
    lens: LensCatalogConfig
    source: SourceConfig = field(default_factory=SourceConfig)
    binning: BinningConfig = field(default_factory=BinningConfig)
    corrections: CorrectionConfig = field(default_factory=CorrectionConfig)
    rp: RPConfig = field(default_factory=RPConfig)
    save_root: str = "output/{label}"
    lens_survey: str = "hsc"
    n_jackknife: int = 100
    n_jobs: int = 12
    comoving: bool = False
    lens_source_cut: float = 0.1
    random_multiplier: int = 20
    rng_seed: Optional[int] = None
    make_plots: bool = True
    redshift_type: Literal["photoz", "specz"] = "photoz"

    def resolved_save_root(self, root: Path | str | None = None) -> Path:
        """Return ``save_root`` as an absolute path under *root*."""
        rendered = self.save_root.format(label=self.label)
        base = Path(root) if root is not None else Path.cwd()
        return base / rendered if not Path(rendered).is_absolute() else Path(rendered)

    def prepare_dir(self, root: Path | str | None = None) -> Path:
        """Return directory for lens/random prepare outputs."""
        return self.resolved_save_root(root) / "prepare"

    def dsigma_dir(self, root: Path | str | None = None) -> Path:
        """Return directory for dsigma profile outputs."""
        base = self.resolved_save_root(root)
        return (
            base / "y1" / "dsigma" if self.source.version == "Y1" else base / "dsigma"
        )

    def fit_dir(self, root: Path | str | None = None) -> Path:
        """Return directory for scatter fit products."""
        base = self.resolved_save_root(root)
        return base / "y1" / "fit" if self.source.version == "Y1" else base / "fit"

    def pkl_dir(self, root: Path | str | None = None) -> Path:
        """Return directory for pickled summary products."""
        base = self.resolved_save_root(root)
        return base / "y1" / "pkl" if self.source.version == "Y1" else base / "pkl"


# ---------------------------------------------------------------------------
# Path & Binning Resolution Helpers
# ---------------------------------------------------------------------------


def resolve_run_dir(
    label: str,
    sub: Literal["prepare", "dsigma", "fit", "pkl"] = "dsigma",
    root: Path | str | None = None,
    version: str = "Y3",
) -> Path:
    """Resolve standard output subdirectory for a run label."""
    if label in RUN_REGISTRY:
        cfg = RUN_REGISTRY[label]
        if version != "Y3" or cfg.source.version != version:
            cfg = replace(cfg, source=replace(cfg.source, version=version))
        if sub == "prepare":
            return cfg.prepare_dir(root)
        if sub == "dsigma":
            return cfg.dsigma_dir(root)
        if sub == "fit":
            return cfg.fit_dir(root)
        if sub == "pkl":
            return cfg.pkl_dir(root)
        return cfg.resolved_save_root(root)

    # Fallback parsing for ad-hoc "{catalog_id}_{nbins}" labels
    catalog_id, nbins = label.rsplit("_", 1)
    base = (Path(root) if root else Path.cwd()) / f"output/{catalog_id}/{nbins}"
    if version == "Y1" and sub != "prepare":
        return base / "y1" / sub
    return base / sub


def resolve_binning(binning: BinningConfig, factor: float) -> BinningConfig:
    """Apply the per-catalog ``top_counts_factor`` scaling."""
    if factor == 1.0:
        return binning
    if binning.mode == "top_counts":
        scaled = tuple(int(round(c * factor)) for c in binning.top_counts)
        if any(c <= 0 for c in scaled):
            raise ValueError(
                f"Scaled top_counts must be positive (factor={factor}, "
                f"raw={binning.top_counts}, scaled={scaled})."
            )
        return replace(binning, top_counts=scaled)
    if binning.mode == "top_n":
        scaled_n = max(1, int(round(binning.top_n * factor)))
        return replace(binning, top_n=scaled_n)
    return binning


def resolve_config(cfg: WLConfig, root: Path | None = None) -> WLConfig:
    """Resolve effective survey coverage, area, and volume factor."""
    effective_redshift_type = (
        cfg.redshift_type if cfg.redshift_type != "photoz" else cfg.lens.redshift_type
    )
    try:
        from hsc_wl.coverage import resolve_area_and_factor

        area, factor = resolve_area_and_factor(cfg.lens, root)
        new_lens = replace(
            cfg.lens,
            area_deg2=area,
            top_counts_factor=factor,
            redshift_type=effective_redshift_type,
        )
        return replace(cfg, lens=new_lens, redshift_type=effective_redshift_type)
    except (FileNotFoundError, ImportError) as exc:
        logger.warning(
            "Could not resolve area/factor for %s (%s); falling back to static factor=%.6f",
            cfg.label,
            exc,
            cfg.lens.top_counts_factor,
        )
        new_lens = replace(cfg.lens, redshift_type=effective_redshift_type)
        return replace(cfg, lens=new_lens, redshift_type=effective_redshift_type)


# ---------------------------------------------------------------------------
# Footprint Inference & Binning Presets
# ---------------------------------------------------------------------------

_BOX_HECTOMAP = {"ra_range": (210.0, 250.0), "dec_range": (42.0, 44.5)}
_Z_RANGE = (0.19, 0.52)

_DEFAULT_BINNING_4BIN = BinningConfig(
    mode="top_counts",
    top_counts=(53, 196, 660, 1159),
    edges_richness=(6.0, 10.0, 20.0, 35.0, 120.0),
    edges_mass=(10.63, 10.8, 11.0, 11.2, 11.6),
)
_DEFAULT_BINNING_1BIN = BinningConfig(
    mode="top_n",
    top_counts=(53, 196, 660, 1159),
    top_n=320,
    edges_richness=(6.0, 10.0, 20.0, 35.0, 120.0),
    edges_mass=(10.63, 10.8, 11.0, 11.2, 11.6),
)


def _infer_footprint(
    label: str,
    fields: Optional[tuple[str, ...]],
    survey_overlap: Optional[str],
) -> tuple[Optional[tuple[str, ...]], Optional[str]]:
    """Determine effective survey field and overlap for a catalog label."""
    if fields is not None or survey_overlap is not None:
        return fields, survey_overlap
    label_lower = label.lower()
    is_s16a = label_lower.startswith(("logm_s16a", "redm_s16a", "forced"))
    is_hecto = "hecto" in label_lower or label_lower.startswith(
        ("redm_pdr3", "cosine", "pls", "cca", "regression", "ideal")
    )
    if is_hecto and is_s16a:
        return ("HECTOMAP",), "s16a"
    if is_hecto:
        return ("HECTOMAP",), None
    if is_s16a:
        return None, "s16a"
    return None, None


# ---------------------------------------------------------------------------
# Reusable Column Mappings & Random Catalogs
# ---------------------------------------------------------------------------

_RAND_HECTOMAP = "data/random_hectomap.fits"
_RAND_S16A = "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits"
_RAND_Y3 = "data/random_y3_mask.fits"

_COLS_REDM = ColumnMapping(
    col_rank="lambda", ra="ra", dec="dec", z="z_lambda", z_spec="z_spec"
)
_COLS_R16 = ColumnMapping(
    col_rank="lambda", ra="RAJ2000", dec="DEJ2000", z="zlambda", z_spec="zspec"
)
_COLS_LOGM = ColumnMapping(
    col_rank="logm_50_100", ra="ra", dec="dec", z="z_best", z_spec="z_spec"
)
_COLS_FORCED = ColumnMapping(
    col_rank="lam", ra="ra", dec="dec", z="z_best", z_spec="z_spec"
)
_COLS_CAMIRA = ColumnMapping(
    col_rank="N_mem", ra="RA", dec="Dec", z="z_cl", z_spec="z_spec"
)
_COLS_CF_STANDARD = ColumnMapping(
    col_rank="true_richness",
    ra="ra",
    dec="dec",
    z="z_cl",
    z_spec="specz",
)
_COLS_CCA1 = ColumnMapping(
    col_rank="cca1", ra="ra", dec="dec", z="z_cl", z_spec="specz"
)
_COLS_CCA2 = ColumnMapping(
    col_rank="cca2", ra="ra", dec="dec", z="z_cl", z_spec="specz"
)
_COLS_RICHNESS_Z = ColumnMapping(col_rank="richness", ra="ra", dec="dec", z="z")
_COLS_WH24 = ColumnMapping(col_rank="lam500", ra="ra", dec="dec", z="z")
_COLS_MASS_Z = ColumnMapping(col_rank="mass", ra="ra", dec="dec", z="z")
_COLS_YANG21_MASS = ColumnMapping(col_rank="log_m_h", ra="ra", dec="dec", z="z")


def _lens(
    label: str,
    lens_path: str,
    columns: ColumnMapping,
    random_path: Optional[str] = None,
    *,
    redshift_range: tuple[float, float] = _Z_RANGE,
    lens_format: Optional[str] = None,
    fields: Optional[tuple[str, ...]] = None,
    survey_overlap: Optional[str] = None,
    ra_range: Optional[tuple[float, float]] = None,
    dec_range: Optional[tuple[float, float]] = None,
    redshift_type: Literal["photoz", "specz"] = "photoz",
    top_counts_factor: float = 1.0,
) -> LensCatalogConfig:
    eff_fields, eff_overlap = _infer_footprint(label, fields, survey_overlap)
    return LensCatalogConfig(
        label=label,
        lens_path=lens_path,
        columns=columns,
        redshift_range=redshift_range,
        fields=eff_fields,
        survey_overlap=eff_overlap,
        top_counts_factor=top_counts_factor,
        lens_format=lens_format,
        ra_range=ra_range,
        dec_range=dec_range,
        redshift_type=redshift_type,
        random_path=random_path,
    )


# ---------------------------------------------------------------------------
# Lens Catalog Definitions (LENS_REGISTRY)
# ---------------------------------------------------------------------------

_PATH_REDM_RUN = "/Users/xinq/redmapper_HSC/output/redmapper_run"
_PATH_CF_FALL = "/Users/xinq/cluster_finder/output/fall_hectomap_spring/rz_diff"

LENS_REGISTRY: dict[str, LensCatalogConfig] = {
    # RedMapper PDR3 variants (HectoMap)
    "redm_pdr3_3band_fixed": _lens(
        "redm_pdr3_3band_fixed",
        f"{_PATH_REDM_RUN}/new_run_no_mask/run/hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit",
        _COLS_REDM,
        _RAND_HECTOMAP,
    ),
    "redm_pdr3_5band_free": _lens(
        "redm_pdr3_5band_free",
        f"{_PATH_REDM_RUN}/new_run_5bands_offdiag/run/hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit",
        _COLS_REDM,
        _RAND_HECTOMAP,
    ),
    "redm_pdr3_3band_free": _lens(
        "redm_pdr3_3band_free",
        f"{_PATH_REDM_RUN}/new_run_free_offdiag/run/hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit",
        _COLS_REDM,
        _RAND_HECTOMAP,
    ),
    # RedMapper S16a (Full & HectoMap)
    "redm_s16a": _lens(
        "redm_s16a",
        "/Users/xinq/redmapper_HSC/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits",
        _COLS_REDM,
        _RAND_S16A,
    ),
    "redm_s16a_hectomap": _lens(
        "redm_s16a_hectomap",
        "/Users/xinq/redmapper_HSC/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits",
        _COLS_REDM,
        _RAND_HECTOMAP,
        **_BOX_HECTOMAP,
    ),
    # S16a Stellar Mass Selection
    "logm_s16a": _lens(
        "logm_s16a",
        "/Users/xinq/research/HSC_massive/s16a_massive_logm_11.2.fits",
        _COLS_LOGM,
        _RAND_S16A,
    ),
    "logm_s16a_hectomap": _lens(
        "logm_s16a_hectomap",
        "/Users/xinq/research/HSC_massive/s16a_massive_logm_11.2.fits",
        _COLS_LOGM,
        _RAND_HECTOMAP,
        **_BOX_HECTOMAP,
    ),
    # Forced-Richness S16a
    "forced": _lens(
        "forced",
        "/Users/xinq/redmapper_HSC/output/s16a_massive_logm_11.2_forced_results.fits",
        _COLS_FORCED,
        _RAND_S16A,
    ),
    "forced_hectomap": _lens(
        "forced_hectomap",
        "/Users/xinq/redmapper_HSC/output/s16a_massive_logm_11.2_forced_results.fits",
        _COLS_FORCED,
        _RAND_HECTOMAP,
        **_BOX_HECTOMAP,
    ),
    # CAMIRA S23b Wide
    "camira": _lens(
        "camira",
        "data/camira_s23b_wide_sm_v3.dat",
        _COLS_CAMIRA,
        _RAND_Y3,
        lens_format="ascii",
    ),
    "camira_hectomap": _lens(
        "camira_hectomap",
        "data/camira_s23b_wide_sm_v3.dat",
        _COLS_CAMIRA,
        _RAND_HECTOMAP,
        lens_format="ascii",
        **_BOX_HECTOMAP,
    ),
    # RedMapper SDSS R16
    "redm_r16": _lens(
        "redm_r16",
        "data/R16/R16_cluster_catalog_bin.fit",
        _COLS_R16,
        _RAND_Y3,
    ),
    "redm_r16_hectomap": _lens(
        "redm_r16_hectomap",
        "data/R16/R16_cluster_catalog_bin.fit",
        _COLS_R16,
        _RAND_HECTOMAP,
        **_BOX_HECTOMAP,
    ),
    # Cluster Finders (HectoMap & Full Y3)
    "cosine": _lens(
        "cosine",
        get_latest_cluster_catalog(),
        _COLS_CF_STANDARD,
        _RAND_HECTOMAP,
        lens_format="parquet",
    ),
    "pls": _lens(
        "pls",
        "/Users/xinq/cluster_finder/output/pls/pls_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_HECTOMAP,
        lens_format="parquet",
    ),
    "amico": _lens(
        "amico",
        get_latest_cluster_catalog(
            "/Users/xinq/cluster_finder/output/fall_hectomap_spring/amico/cluster"
        ),
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "amico_hectomap": _lens(
        "amico_hectomap",
        get_latest_cluster_catalog(
            "/Users/xinq/cluster_finder/output/fall_hectomap_spring/amico/cluster"
        ),
        _COLS_CF_STANDARD,
        _RAND_HECTOMAP,
        lens_format="parquet",
        **_BOX_HECTOMAP,
    ),
    "regression": _lens(
        "regression",
        "/Users/xinq/cluster_finder/output/regression/regression_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_HECTOMAP,
        lens_format="parquet",
    ),
    "cca1": _lens(
        "cca1",
        "/Users/xinq/cluster_finder/output/cca/cca1_cluster_catalog.parquet",
        _COLS_CCA1,
        _RAND_HECTOMAP,
        lens_format="parquet",
    ),
    "cca1_specz": _lens(
        "cca1_specz",
        "/Users/xinq/cluster_finder/output/cca/cca1_cluster_catalog.parquet",
        _COLS_CCA1,
        _RAND_HECTOMAP,
        lens_format="parquet",
        redshift_type="specz",
    ),
    "cca2": _lens(
        "cca2",
        "/Users/xinq/cluster_finder/output/cca/cca2_cluster_catalog.parquet",
        _COLS_CCA2,
        _RAND_HECTOMAP,
        lens_format="parquet",
    ),
    "cca2_specz": _lens(
        "cca2_specz",
        "/Users/xinq/cluster_finder/output/cca/cca2_cluster_catalog.parquet",
        _COLS_CCA2,
        _RAND_HECTOMAP,
        lens_format="parquet",
        redshift_type="specz",
    ),
    # 2D r-z Difference Profiles
    "rz_diff_preset": _lens(
        "rz_diff_preset",
        f"{_PATH_CF_FALL}/preset/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_preset_lum": _lens(
        "rz_diff_preset_lum",
        f"{_PATH_CF_FALL}/preset/rz_diff_lum_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff": _lens(
        "rz_diff",
        f"{_PATH_CF_FALL}/baseline/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_lum": _lens(
        "rz_diff_lum",
        f"{_PATH_CF_FALL}/baseline/rz_diff_lum_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_two_box_match_recovery": _lens(
        "rz_diff_two_box_match_recovery",
        f"{_PATH_CF_FALL}/two_box_match_recovery/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_single_box": _lens(
        "rz_diff_single_box",
        f"{_PATH_CF_FALL}/single_box_wl/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_single_box_lum": _lens(
        "rz_diff_single_box_lum",
        f"{_PATH_CF_FALL}/single_box_wl/rz_diff_lum_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_single_box_match_recovery": _lens(
        "rz_diff_single_box_match_recovery",
        f"{_PATH_CF_FALL}/single_box_match_recovery/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_two_box_red": _lens(
        "rz_diff_two_box_red",
        f"{_PATH_CF_FALL}/two_box_red_wl/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_two_box_red_lum": _lens(
        "rz_diff_two_box_red_lum",
        f"{_PATH_CF_FALL}/two_box_red_wl/rz_diff_lum_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_two_box_red_match_recovery": _lens(
        "rz_diff_two_box_red_match_recovery",
        f"{_PATH_CF_FALL}/two_box_red_match_recovery/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_two_box_red_redmapper_matching": _lens(
        "rz_diff_two_box_red_redmapper_matching",
        f"{_PATH_CF_FALL}/two_box_red_redmapper_matching/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_two_box_red_camira_matching": _lens(
        "rz_diff_two_box_red_camira_matching",
        f"{_PATH_CF_FALL}/two_box_red_camira_matching/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_six_param": _lens(
        "rz_diff_six_param",
        f"{_PATH_CF_FALL}/six_param_wl/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_six_param_match_recovery": _lens(
        "rz_diff_six_param_match_recovery",
        f"{_PATH_CF_FALL}/six_param_match_recovery/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_mag_2_p_5": _lens(
        "rz_diff_mag_2_p_5",
        f"{_PATH_CF_FALL}/mag_2_p_5/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    "rz_diff_mag_3_p_0": _lens(
        "rz_diff_mag_3_p_0",
        f"{_PATH_CF_FALL}/mag_3_p_0/rz_diff_cluster_catalog.parquet",
        _COLS_CF_STANDARD,
        _RAND_Y3,
        lens_format="parquet",
    ),
    # External DESI Legacy Surveys
    "wh20": _lens(
        "wh20", "data/external_clusters/wh20.parquet", _COLS_RICHNESS_Z, _RAND_Y3
    ),
    "wh24": _lens("wh24", "data/external_clusters/wh24.parquet", _COLS_WH24, _RAND_Y3),
    "zou21": _lens(
        "zou21", "data/external_clusters/zou21.parquet", _COLS_RICHNESS_Z, _RAND_Y3
    ),
    "clumpr_mass": _lens(
        "clumpr_mass",
        "data/external_clusters/clumpr.parquet",
        _COLS_MASS_Z,
        _RAND_Y3,
    ),
    "clumpr_richness": _lens(
        "clumpr_richness",
        "data/external_clusters/clumpr.parquet",
        _COLS_RICHNESS_Z,
        _RAND_Y3,
    ),
    "yang21_mass": _lens(
        "yang21_mass",
        "data/external_clusters/yang21.parquet",
        _COLS_YANG21_MASS,
        _RAND_Y3,
    ),
    "yang21_richness": _lens(
        "yang21_richness",
        "data/external_clusters/yang21.parquet",
        _COLS_RICHNESS_Z,
        _RAND_Y3,
    ),
}

# ---------------------------------------------------------------------------
# Construct RUN_REGISTRY from LENS_REGISTRY
# ---------------------------------------------------------------------------


def _make_wl_config(
    label: str,
    lens_cfg: LensCatalogConfig,
    binning: BinningConfig,
    save_root: str,
) -> WLConfig:
    return WLConfig(
        label=label,
        lens=replace(lens_cfg, label=label),
        source=SourceConfig(),
        binning=binning,
        save_root=save_root,
        redshift_type=lens_cfg.redshift_type,
    )


RUN_REGISTRY: dict[str, WLConfig] = {}

# Programmatically generate paired 4bin and 1bin configurations
for cat_id, lens_cfg in LENS_REGISTRY.items():
    RUN_REGISTRY[f"{cat_id}_4bin"] = _make_wl_config(
        f"{cat_id}_4bin",
        lens_cfg,
        _DEFAULT_BINNING_4BIN,
        f"output/{cat_id}/4bin",
    )
    RUN_REGISTRY[f"{cat_id}_1bin"] = _make_wl_config(
        f"{cat_id}_1bin",
        lens_cfg,
        _DEFAULT_BINNING_1BIN,
        f"output/{cat_id}/1bin",
    )

# Theoretical upper limits (MDPL2 simulation and Colossus halo model)
_IDEAL_SPECS = [
    (
        "ideal_mdpl2_1bin",
        "output/ideal_mdpl2/1bin/prepare/ideal_mdpl2_1bin_lenses.fits",
        _DEFAULT_BINNING_1BIN,
        "output/ideal_mdpl2/1bin",
    ),
    (
        "ideal_mdpl2_4bin",
        "output/ideal_mdpl2/4bin/prepare/ideal_mdpl2_4bin_lenses.fits",
        _DEFAULT_BINNING_4BIN,
        "output/ideal_mdpl2/4bin",
    ),
    (
        "ideal_colossus_1bin",
        "output/ideal_colossus/1bin/prepare/ideal_colossus_1bin_lenses.fits",
        _DEFAULT_BINNING_1BIN,
        "output/ideal_colossus/1bin",
    ),
    (
        "ideal_colossus_4bin",
        "output/ideal_colossus/4bin/prepare/ideal_colossus_4bin_lenses.fits",
        _DEFAULT_BINNING_4BIN,
        "output/ideal_colossus/4bin",
    ),
]

for label, pth, bin_cfg, sroot in _IDEAL_SPECS:
    ideal_lens = _lens(label, pth, _COLS_CF_STANDARD, _RAND_HECTOMAP)
    RUN_REGISTRY[label] = _make_wl_config(label, ideal_lens, bin_cfg, sroot)
