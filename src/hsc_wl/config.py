"""Configuration objects for the HSC weak-lensing pipeline.

All run-time configuration is expressed as :class:`~dataclasses.dataclass`
objects (``frozen=True`` so they are immutable value objects).  A single
:data:`RUN_REGISTRY` maps a run label to a fully-specified
:class:`WLConfig`, replacing the previous pair of parallel dicts
(``CATALOG_SOURCES`` and ``RUN_PROFILES``).

Typical usage::

    from hsc_wl.config import RUN_REGISTRY
    cfg = RUN_REGISTRY["cosine_4bin"]
    run_pipeline(cfg)

To override a single parameter without mutating the registry, use
:func:`dataclasses.replace`::

    from dataclasses import replace
    cfg = replace(RUN_REGISTRY["cosine_1bin"], n_jackknife=50)

Run labels follow the convention ``{catalog_id}_{nbins}`` where:

- ``catalog_id`` encodes both the lens catalog and its sky footprint.
  Catalogs that naturally cover the full survey area (e.g. s16a redMapper,
  CAMIRA, SDSS r16 redMaPPer) have an optional ``_hectomap`` suffix for the
  HectoMAP sub-region (RA 200-250 / Dec 42-44.5; or RA 210-250 / Dec 42-44.5 for
  SDSS r16).  Catalogs that are inherently confined to a single footprint
  (pdr3 redMapper, COSINE) carry no footprint suffix.  A further ``_s16a``
  sub-variant additionally restricts the footprint to the s16a random survey
  area (s16a random ∩ HectoMAP box ∩ Y3 mask), enabling fair comparison
  across all lens catalogs on the same sky patch.
- ``nbins`` is either ``1bin`` (single top-N bin, ``top_n`` mode) or
  ``4bin`` (four richness/mass bins, ``top_counts`` mode).

Output is written to ``output/{catalog_id}/{nbins}/``.
"""

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
    "RUN_REGISTRY",
    "resolve_binning",
    "resolve_config",
    "get_latest_cluster_catalog",
    "replace",
]


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def get_latest_cluster_catalog(
    cluster_dir="/Users/xinq/cluster_finder/output/cluster",
):
    """Find the latest timestamped cluster catalog parquet file."""
    import glob

    pattern = str(Path(cluster_dir) / "cluster_catalog_*.parquet")
    files = glob.glob(pattern)
    if not files:
        return "/Users/xinq/cluster_finder/output/cluster/cluster_catalog_20260603_085729.parquet"
    return max(files, key=lambda p: Path(p).stat().st_mtime)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """Column-name mapping for a lens catalog."""

    col_rank: str
    ra: str
    dec: str
    z: str


@dataclass(frozen=True, slots=True)
class LensCatalogConfig:
    """Where to find a lens catalog and how to interpret it.

    The ``lens_path`` / ``random_path`` may be absolute or relative to the
    project root (resolved at run time).  ``lens_format`` selects a reader;
    ``None`` lets ``astropy`` auto-detect.
    """

    label: str
    lens_path: str
    random_path: str
    columns: ColumnMapping
    redshift_range: tuple[float, float]
    top_counts_factor: float = 1.0
    lens_format: Optional[str] = None
    random_format: Optional[str] = None
    ra_range: Optional[tuple[float, float]] = None
    dec_range: Optional[tuple[float, float]] = None
    area_deg2: Optional[float] = None


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Source (background galaxy) catalog configuration.

    When ``file`` is ``None`` the default path for the chosen ``version``
    is resolved at run time (see :func:`hsc_wl.wl_compute.default_source_path`).
    """

    version: Literal["Y1", "Y3"] = "Y3"
    survey: str = "hsc"
    file: Optional[str] = None
    nz_file: Optional[str] = None
    calib_file: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BinningConfig:
    """How lenses are partitioned into bins.

    ``mode`` is one of ``"edges"``, ``"top_counts"``, ``"top_n"``.
    The ``top_counts`` / ``top_n`` fields hold the *raw* user-specified
    values; the per-catalog ``top_counts_factor`` is applied separately by
    :func:`resolve_binning`.
    """

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

    def to_dsigma_kwargs(self) -> dict:
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
    """Projected-radius binning."""

    rp_min: float = 0.10
    rp_max: float = 20.0
    n_bins: int = 11
    linlog: str = "log"


@dataclass(frozen=True, slots=True)
class WLConfig:
    """A fully-specified weak-lensing run.

    One object per run label; collected in :data:`RUN_REGISTRY`.  Pass it
    to :func:`hsc_wl.wl_compute.run_pipeline` (full pipeline) or
    :func:`hsc_wl.prepare.run_prepare_pipeline` (prepare stage only).
    """

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

    def resolved_save_root(self, root: Path) -> Path:
        """Return ``save_root`` as an absolute path under *root*."""
        rendered = self.save_root.format(label=self.label)
        if Path(rendered).is_absolute():
            return Path(rendered)
        return Path(root) / rendered


# ---------------------------------------------------------------------------
# Binning resolution (apply top_counts_factor)
# ---------------------------------------------------------------------------


def resolve_binning(binning: BinningConfig, factor: float) -> BinningConfig:
    """Apply the per-catalog ``top_counts_factor`` scaling.

    Returns a new :class:`BinningConfig` with the *effective* (scaled)
    ``top_counts`` (for ``top_counts`` mode) or ``top_n`` (for ``top_n``
    mode).  No-op for ``edges`` mode or when ``factor`` is ``1.0``.
    """
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
    """Resolve ``area_deg2`` and ``top_counts_factor`` from sky coverage.

    Computes the effective area (lens-random footprint ∩ Y3 FDFC mask) and
    the volume factor relative to the s16a reference (z 0.19–0.52,
    s16a-random ∩ Y3 area).  Returns a **new** :class:`WLConfig` with the
    resolved ``LensCatalogConfig``; the original registry entry is unchanged.

    Falls back to the static ``top_counts_factor`` (default 1.0) if the
    mask or random files are unavailable.
    """
    try:
        from hsc_wl.coverage import resolve_area_and_factor

        area, factor = resolve_area_and_factor(cfg.lens, root)
        new_lens = replace(cfg.lens, area_deg2=area, top_counts_factor=factor)
        return replace(cfg, lens=new_lens)
    except (FileNotFoundError, ImportError) as exc:
        logger.warning(
            "Could not resolve area/factor for %s (%s); "
            "falling back to static top_counts_factor=%.6f",
            cfg.label,
            exc,
            cfg.lens.top_counts_factor,
        )
        return cfg


# ---------------------------------------------------------------------------
# Run registry – shared constants
# ---------------------------------------------------------------------------

_EDGES_RICHNESS = (6.0, 10.0, 20.0, 35.0, 120.0)
_EDGES_MASS = (10.63, 10.8, 11.0, 11.2, 11.6)
_TOP_COUNTS_4BIN = (53, 196, 660, 1159)

# 4-bin mode: lenses are split into four richness/mass bins by top_counts.
_DEFAULT_BINNING_4BIN = BinningConfig(
    mode="top_counts",
    top_counts=_TOP_COUNTS_4BIN,
    edges_richness=_EDGES_RICHNESS,
    edges_mass=_EDGES_MASS,
)
# 1-bin mode: all lenses treated as a single top-N sample.
# top_n=320 in reference volume corresponds to exactly top_n=100 in the HectoMAP volume.
_DEFAULT_BINNING_1BIN = BinningConfig(
    mode="top_n",
    top_counts=_TOP_COUNTS_4BIN,
    top_n=320,
    edges_richness=_EDGES_RICHNESS,
    edges_mass=_EDGES_MASS,
)

# ---------------------------------------------------------------------------
# Lens catalog paths
# ---------------------------------------------------------------------------

# redMapper PDR3 – three photometric variants
_PATH_REDM_PDR3_3BAND_FIXED = (
    "/Users/xinq/redmapper_HSC/output/redmapper_run/new_run_no_mask/run/"
    "hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit"
)
_PATH_REDM_PDR3_5BAND_FREE = (
    "/Users/xinq/redmapper_HSC/output/redmapper_run/new_run_5bands_offdiag/run/"
    "hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit"
)
_PATH_REDM_PDR3_3BAND_FREE = (
    "/Users/xinq/redmapper_HSC/output/redmapper_run/new_run_free_offdiag/run/"
    "hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit"
)

# redMapper S16a
_PATH_REDM_S16A = (
    "/Users/xinq/redmapper_HSC/data/reference/redmapper_s16a/"
    "redmapper_hsc_s16a_cluster_bsm.fits"
)

# Stellar-mass selected S16a galaxies
_PATH_LOGM_S16A = "/Users/xinq/redmapper_HSC/data/reference/s16a_massive_logm_11.2.fits"

# Forced-richness S16a massive galaxies
_PATH_FORCED = (
    "/Users/xinq/redmapper_HSC/output/s16a_massive_logm_11.2_forced_results.fits"
)

# CAMIRA S23b wide
_PATH_CAMIRA = "data/camira_s23b_wide_sm_v3.dat"

# AMICO cluster finder
_PATH_AMICO = get_latest_cluster_catalog(
    "/Users/xinq/cluster_finder/output/amico/cluster"
)

# PLS cluster finder (2D CoG PLS decomposition + Cylinder NMS)
_PATH_PLS = "/Users/xinq/cluster_finder/output/pls/pls_cluster_catalog_no_nms.parquet"

# Direct 2D r-z profile subtraction cluster catalog (no model, no NMS)
_PATH_RZ_DIFF = (
    "/Users/xinq/cluster_finder/output/rz_diff/rz_diff_cluster_catalog.parquet"
)

# Linear regression against WL mass (ElasticNet on 2D differential profiles, no NMS)
_PATH_REGRESSION = (
    "/Users/xinq/cluster_finder/output/regression/regression_cluster_catalog.parquet"
)

# CCA cluster finders (2D CoG CCA decomposition, cca1 & cca2 ranking, no NMS)
_PATH_CCA1 = "/Users/xinq/cluster_finder/output/cca/cca1_cluster_catalog.parquet"
_PATH_CCA2 = "/Users/xinq/cluster_finder/output/cca/cca2_cluster_catalog.parquet"
_PATH_CCA = _PATH_CCA1

# redMapper SDSS R16 cluster catalog
_PATH_R16 = "data/R16/R16_cluster_catalog_bin.fit"

# Ideal theoretical upper limit catalog paths (MDPL2 simulation and Colossus halo model)
_PATH_IDEAL_MDPL2_1BIN = "output/ideal_mdpl2/1bin/prepare/ideal_mdpl2_1bin_lenses.fits"
_PATH_IDEAL_MDPL2_4BIN = "output/ideal_mdpl2/4bin/prepare/ideal_mdpl2_4bin_lenses.fits"
_PATH_IDEAL_COLOSSUS_1BIN = (
    "output/ideal_colossus/1bin/prepare/ideal_colossus_1bin_lenses.fits"
)
_PATH_IDEAL_COLOSSUS_4BIN = (
    "output/ideal_colossus/4bin/prepare/ideal_colossus_4bin_lenses.fits"
)

# ---------------------------------------------------------------------------
# Random catalog paths
# ---------------------------------------------------------------------------

_RAND_HECTOMAP = "data/random_hectomap.fits"
_RAND_S16A = "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits"
_RAND_Y3 = "data/random_y3_mask.fits"

# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

_COLS_REDM = {"col_rank": "lambda", "ra": "ra", "dec": "dec", "z": "z_lambda"}
_COLS_R16 = {"col_rank": "lambda", "ra": "RAJ2000", "dec": "DEJ2000", "z": "zlambda"}
_COLS_LOGM = {"col_rank": "logm_50_100", "ra": "ra", "dec": "dec", "z": "z_best"}
_COLS_FORCED = {"col_rank": "lam", "ra": "ra", "dec": "dec", "z": "z_best"}
_COLS_CAMIRA = {"col_rank": "N_mem", "ra": "RA", "dec": "Dec", "z": "z_cl"}
_COLS_COSINE = {"col_rank": "true_richness", "ra": "ra", "dec": "dec", "z": "z_cl"}
_COLS_AMICO = {"col_rank": "true_richness", "ra": "ra", "dec": "dec", "z": "z_cl"}
_COLS_PLS = {"col_rank": "true_richness", "ra": "ra", "dec": "dec", "z": "z_cl"}
_COLS_RZ_DIFF = {"col_rank": "true_richness", "ra": "ra", "dec": "dec", "z": "z_cl"}
_COLS_REGRESSION = {"col_rank": "true_richness", "ra": "ra", "dec": "dec", "z": "z_cl"}
_COLS_CCA = {"col_rank": "true_richness", "ra": "ra", "dec": "dec", "z": "z_cl"}
_COLS_CCA1 = _COLS_CCA
_COLS_CCA2 = _COLS_CCA

# ---------------------------------------------------------------------------
# Sky footprint boxes  (unpacked as **kwargs into _cfg)
# ---------------------------------------------------------------------------

# HectoMap sub-region: RA 210-250 deg, Dec 42-44.5 deg
_BOX_HECTOMAP = {"ra_range": (210, 250), "dec_range": (42, 44.5)}
_BOX_R16_HECTOMAP = _BOX_HECTOMAP

_BOX_AMICO = _BOX_HECTOMAP

# Reference redshift range shared by all catalogs
_Z_RANGE = (0.19, 0.52)


# ---------------------------------------------------------------------------
# Compact WLConfig constructor
# ---------------------------------------------------------------------------


def _cfg(
    label,
    lens_path,
    random_path,
    *,
    columns,
    redshift_range,
    top_counts_factor=1.0,
    lens_format=None,
    random_format=None,
    ra_range=None,
    dec_range=None,
    source=None,
    binning=None,
    save_root=None,
):
    """Compact constructor for :class:`WLConfig` registry entries.

    ``save_root`` defaults to ``output/{catalog_id}/{nbins}`` where
    ``catalog_id`` and ``nbins`` are derived by splitting *label* at its
    last underscore (e.g. ``"redm_s16a_4bin"`` → ``"redm_s16a"`` /
    ``"4bin"``).
    """
    if save_root is None:
        catalog_id, nbins = label.rsplit("_", 1)
        save_root = f"output/{catalog_id}/{nbins}"
    return WLConfig(
        label=label,
        lens=LensCatalogConfig(
            label=label,
            lens_path=lens_path,
            random_path=random_path,
            columns=ColumnMapping(**columns),
            redshift_range=tuple(redshift_range),
            top_counts_factor=top_counts_factor,
            lens_format=lens_format,
            random_format=random_format,
            ra_range=tuple(ra_range) if ra_range else None,
            dec_range=tuple(dec_range) if dec_range else None,
        ),
        source=source or SourceConfig(),
        binning=binning or _DEFAULT_BINNING_4BIN,
        save_root=save_root,
    )


# ---------------------------------------------------------------------------
# Run registry
#
# Label convention : {catalog_id}_{nbins}
# Output convention: output/{catalog_id}/{nbins}/
#
# Catalog IDs
#   redm_pdr3_3band_fixed        – redMapper PDR3, 3-band colours, fixed off-diag cov
#   redm_pdr3_3band_fixed_s16a   – same, restricted to s16a random ∩ HectoMAP box
#   redm_pdr3_5band_free         – redMapper PDR3, 5-band colours, free off-diag cov
#   redm_pdr3_5band_free_s16a    – same, restricted to s16a random ∩ HectoMAP box
#   redm_pdr3_3band_free         – redMapper PDR3, 3-band colours, free off-diag cov
#   redm_pdr3_3band_free_s16a    – same, restricted to s16a random ∩ HectoMAP box
#   redm_s16a                    – redMapper S16a, full footprint
#   redm_s16a_hectomap           – redMapper S16a, HectoMap sub-region
#   logm_s16a                    – S16a massive galaxies ranked by log stellar mass
#   logm_s16a_hectomap           – same, HectoMap sub-region
#   forced                       – S16a massive galaxies with forced redMapper richness
#   forced_hectomap              – same, HectoMap sub-region
#   camira                       – CAMIRA S23b wide, full footprint
#   camira_hectomap              – CAMIRA S23b wide, HectoMap sub-region
#   camira_hecto_s16a            – CAMIRA, HectoMAP box ∩ s16a random footprint
#   redm_r16                     – redMapper SDSS R16, full footprint (Y3 mask)
#   redm_r16_hectomap            – redMapper SDSS R16, HectoMap sub-region (RA 210-250)
#   redm_r16_hecto_s16a          – redMapper SDSS R16, HectoMAP box ∩ s16a random footprint
#   cosine                       – COSINE cluster finder (natural HectoMap footprint)
#   cosine_s16a                  – same, restricted to s16a random ∩ HectoMAP box
#   pls                          – PLS cluster finder (2D CoG PLS decomposition + Cylinder NMS)
#   pls_s16a                     – same, restricted to s16a random ∩ HectoMAP box
#   rz_diff                      – Direct 2D r-z profile subtraction cluster catalog (no model, no NMS)
#   rz_diff_s16a                 – same, restricted to s16a random ∩ HectoMAP box
#   regression                   – Linear regression against WL mass (ElasticNet, no NMS)
#   regression_s16a              – same, restricted to s16a random ∩ HectoMAP box
#   cca / cca1                   – CCA1 cluster finder (Primary WL mass mode, cca1 ranking, no NMS)
#   cca_s16a / cca1_s16a         – same, restricted to s16a random ∩ HectoMAP box
#   cca2                         – CCA2 cluster finder (Morphology/concentration mode, cca2 ranking, no NMS)
#   cca2_s16a                    – same, restricted to s16a random ∩ HectoMAP box
#   amico                        – AMICO cluster finder (RA 215-250 / Dec 42.2-44.5)
#   ideal_mdpl2                  – Theoretical upper limit (MDPL2 simulation central halos, sigma=0)
#   ideal_colossus               – Theoretical upper limit (Colossus analytical halo model, sigma=0)
# ---------------------------------------------------------------------------

RUN_REGISTRY: dict[str, WLConfig] = {
    # -----------------------------------------------------------------------
    # redMapper PDR3 – three photometric variants
    # All three catalogs are naturally confined to the HectoMap footprint.
    # The _s16a variants further restrict the footprint to the s16a random
    # survey area (s16a random ∩ HectoMAP box ∩ Y3 mask) for direct
    # comparison with s16a-based and COSINE catalogs.
    # -----------------------------------------------------------------------
    # 3-band colours, fixed off-diagonal covariance (previously: pdr3_redm_hsc_no_mask)
    "redm_pdr3_3band_fixed_4bin": _cfg(
        "redm_pdr3_3band_fixed_4bin",
        _PATH_REDM_PDR3_3BAND_FIXED,
        _RAND_HECTOMAP,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "redm_pdr3_3band_fixed_1bin": _cfg(
        "redm_pdr3_3band_fixed_1bin",
        _PATH_REDM_PDR3_3BAND_FIXED,
        _RAND_HECTOMAP,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # 3-band fixed, s16a footprint ∩ HectoMAP box
    "redm_pdr3_3band_fixed_s16a_4bin": _cfg(
        "redm_pdr3_3band_fixed_s16a_4bin",
        _PATH_REDM_PDR3_3BAND_FIXED,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "redm_pdr3_3band_fixed_s16a_1bin": _cfg(
        "redm_pdr3_3band_fixed_s16a_1bin",
        _PATH_REDM_PDR3_3BAND_FIXED,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # 5-band colours, free off-diagonal covariance (previously: pdr3_redm_hsc_5bands_offdiag)
    "redm_pdr3_5band_free_4bin": _cfg(
        "redm_pdr3_5band_free_4bin",
        _PATH_REDM_PDR3_5BAND_FREE,
        _RAND_HECTOMAP,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "redm_pdr3_5band_free_1bin": _cfg(
        "redm_pdr3_5band_free_1bin",
        _PATH_REDM_PDR3_5BAND_FREE,
        _RAND_HECTOMAP,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # 5-band free, s16a footprint ∩ HectoMAP box
    "redm_pdr3_5band_free_s16a_4bin": _cfg(
        "redm_pdr3_5band_free_s16a_4bin",
        _PATH_REDM_PDR3_5BAND_FREE,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "redm_pdr3_5band_free_s16a_1bin": _cfg(
        "redm_pdr3_5band_free_s16a_1bin",
        _PATH_REDM_PDR3_5BAND_FREE,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # 3-band colours, free off-diagonal covariance (previously: pdr3_redm_hsc_free_offdiag)
    "redm_pdr3_3band_free_4bin": _cfg(
        "redm_pdr3_3band_free_4bin",
        _PATH_REDM_PDR3_3BAND_FREE,
        _RAND_HECTOMAP,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "redm_pdr3_3band_free_1bin": _cfg(
        "redm_pdr3_3band_free_1bin",
        _PATH_REDM_PDR3_3BAND_FREE,
        _RAND_HECTOMAP,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # 3-band free, s16a footprint ∩ HectoMAP box
    "redm_pdr3_3band_free_s16a_4bin": _cfg(
        "redm_pdr3_3band_free_s16a_4bin",
        _PATH_REDM_PDR3_3BAND_FREE,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "redm_pdr3_3band_free_s16a_1bin": _cfg(
        "redm_pdr3_3band_free_s16a_1bin",
        _PATH_REDM_PDR3_3BAND_FREE,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # redMapper S16a – full footprint and HectoMap sub-region
    # -----------------------------------------------------------------------
    # Full footprint (previously: s16a_redm_hsc / s16a_redm_hsc_topn)
    "redm_s16a_4bin": _cfg(
        "redm_s16a_4bin",
        _PATH_REDM_S16A,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "redm_s16a_1bin": _cfg(
        "redm_s16a_1bin",
        _PATH_REDM_S16A,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # HectoMap sub-region (previously: s16a_redm_hsc_hectomap / s16a_redm_hsc_topn_hectomap)
    "redm_s16a_hectomap_4bin": _cfg(
        "redm_s16a_hectomap_4bin",
        _PATH_REDM_S16A,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "redm_s16a_hectomap_1bin": _cfg(
        "redm_s16a_hectomap_1bin",
        _PATH_REDM_S16A,
        _RAND_S16A,
        columns=_COLS_REDM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # Stellar-mass selected S16a galaxies (log M ranking)
    # -----------------------------------------------------------------------
    # Full footprint (previously: s16a_logm_50_100 / s16a_logm_50_100_topn)
    "logm_s16a_4bin": _cfg(
        "logm_s16a_4bin",
        _PATH_LOGM_S16A,
        _RAND_S16A,
        columns=_COLS_LOGM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "logm_s16a_1bin": _cfg(
        "logm_s16a_1bin",
        _PATH_LOGM_S16A,
        _RAND_S16A,
        columns=_COLS_LOGM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # HectoMap sub-region (previously: s16a_logm_50_100_hectomap / s16a_logm_50_100_topn_hectomap)
    "logm_s16a_hectomap_4bin": _cfg(
        "logm_s16a_hectomap_4bin",
        _PATH_LOGM_S16A,
        _RAND_S16A,
        columns=_COLS_LOGM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "logm_s16a_hectomap_1bin": _cfg(
        "logm_s16a_hectomap_1bin",
        _PATH_LOGM_S16A,
        _RAND_S16A,
        columns=_COLS_LOGM,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # Forced-richness S16a massive galaxies
    # -----------------------------------------------------------------------
    # Full footprint (previously: forced)
    "forced_4bin": _cfg(
        "forced_4bin",
        _PATH_FORCED,
        _RAND_S16A,
        columns=_COLS_FORCED,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "forced_1bin": _cfg(
        "forced_1bin",
        _PATH_FORCED,
        _RAND_S16A,
        columns=_COLS_FORCED,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # HectoMap sub-region
    "forced_hectomap_4bin": _cfg(
        "forced_hectomap_4bin",
        _PATH_FORCED,
        _RAND_S16A,
        columns=_COLS_FORCED,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "forced_hectomap_1bin": _cfg(
        "forced_hectomap_1bin",
        _PATH_FORCED,
        _RAND_S16A,
        columns=_COLS_FORCED,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # CAMIRA S23b wide – full footprint and HectoMap sub-region
    # -----------------------------------------------------------------------
    # Full footprint (previously: camira_4bin / camira)
    "camira_4bin": _cfg(
        "camira_4bin",
        _PATH_CAMIRA,
        _RAND_Y3,
        columns=_COLS_CAMIRA,
        redshift_range=_Z_RANGE,
        lens_format="pandas_dat",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "camira_1bin": _cfg(
        "camira_1bin",
        _PATH_CAMIRA,
        _RAND_Y3,
        columns=_COLS_CAMIRA,
        redshift_range=_Z_RANGE,
        lens_format="pandas_dat",
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # HectoMap sub-region (previously: camira_4bin_hectomap / camira_hectomap)
    "camira_hectomap_4bin": _cfg(
        "camira_hectomap_4bin",
        _PATH_CAMIRA,
        _RAND_HECTOMAP,
        columns=_COLS_CAMIRA,
        redshift_range=_Z_RANGE,
        lens_format="pandas_dat",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "camira_hectomap_1bin": _cfg(
        "camira_hectomap_1bin",
        _PATH_CAMIRA,
        _RAND_HECTOMAP,
        columns=_COLS_CAMIRA,
        redshift_range=_Z_RANGE,
        lens_format="pandas_dat",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # HectoMAP box ∩ s16a random footprint – for direct comparison with
    # s16a-based and COSINE catalogs on the same sky patch.
    "camira_hecto_s16a_4bin": _cfg(
        "camira_hecto_s16a_4bin",
        _PATH_CAMIRA,
        _RAND_S16A,
        columns=_COLS_CAMIRA,
        redshift_range=_Z_RANGE,
        lens_format="pandas_dat",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "camira_hecto_s16a_1bin": _cfg(
        "camira_hecto_s16a_1bin",
        _PATH_CAMIRA,
        _RAND_S16A,
        columns=_COLS_CAMIRA,
        redshift_range=_Z_RANGE,
        lens_format="pandas_dat",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # redMapper SDSS R16 – full footprint (Y3 mask) and HectoMap sub-regions
    # -----------------------------------------------------------------------
    # Full footprint (Y3 mask)
    "redm_r16_4bin": _cfg(
        "redm_r16_4bin",
        _PATH_R16,
        _RAND_Y3,
        columns=_COLS_R16,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "redm_r16_1bin": _cfg(
        "redm_r16_1bin",
        _PATH_R16,
        _RAND_Y3,
        columns=_COLS_R16,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # HectoMap sub-region
    "redm_r16_hectomap_4bin": _cfg(
        "redm_r16_hectomap_4bin",
        _PATH_R16,
        _RAND_HECTOMAP,
        columns=_COLS_R16,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "redm_r16_hectomap_1bin": _cfg(
        "redm_r16_hectomap_1bin",
        _PATH_R16,
        _RAND_HECTOMAP,
        columns=_COLS_R16,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # HectoMAP box ∩ s16a random footprint
    "redm_r16_hecto_s16a_4bin": _cfg(
        "redm_r16_hecto_s16a_4bin",
        _PATH_R16,
        _RAND_S16A,
        columns=_COLS_R16,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_R16_HECTOMAP,
    ),
    "redm_r16_hecto_s16a_1bin": _cfg(
        "redm_r16_hecto_s16a_1bin",
        _PATH_R16,
        _RAND_S16A,
        columns=_COLS_R16,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_R16_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # COSINE cluster finder (natural HectoMap footprint)
    # -----------------------------------------------------------------------
    # (previously: cosine_4bin / cosine)
    "cosine_4bin": _cfg(
        "cosine_4bin",
        get_latest_cluster_catalog(),
        _RAND_HECTOMAP,
        columns=_COLS_COSINE,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "cosine_1bin": _cfg(
        "cosine_1bin",
        get_latest_cluster_catalog(),
        _RAND_HECTOMAP,
        columns=_COLS_COSINE,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # s16a footprint ∩ HectoMAP box – for direct comparison with s16a-based
    # and PDR3 catalogs on the same sky patch.
    "cosine_s16a_4bin": _cfg(
        "cosine_s16a_4bin",
        get_latest_cluster_catalog(),
        _RAND_S16A,
        columns=_COLS_COSINE,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "cosine_s16a_1bin": _cfg(
        "cosine_s16a_1bin",
        get_latest_cluster_catalog(),
        _RAND_S16A,
        columns=_COLS_COSINE,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # PLS cluster finder (2D CoG PLS decomposition + Cylinder NMS)
    # -----------------------------------------------------------------------
    "pls_4bin": _cfg(
        "pls_4bin",
        _PATH_PLS,
        _RAND_HECTOMAP,
        columns=_COLS_PLS,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "pls_1bin": _cfg(
        "pls_1bin",
        _PATH_PLS,
        _RAND_HECTOMAP,
        columns=_COLS_PLS,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # s16a footprint ∩ HectoMAP box – for direct comparison with s16a-based
    # and PDR3 catalogs on the same sky patch.
    "pls_s16a_4bin": _cfg(
        "pls_s16a_4bin",
        _PATH_PLS,
        _RAND_S16A,
        columns=_COLS_PLS,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "pls_s16a_1bin": _cfg(
        "pls_s16a_1bin",
        _PATH_PLS,
        _RAND_S16A,
        columns=_COLS_PLS,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # Direct 2D r-z profile subtraction cluster catalog (no model, no NMS)
    # -----------------------------------------------------------------------
    "rz_diff_4bin": _cfg(
        "rz_diff_4bin",
        _PATH_RZ_DIFF,
        _RAND_HECTOMAP,
        columns=_COLS_RZ_DIFF,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "rz_diff_1bin": _cfg(
        "rz_diff_1bin",
        _PATH_RZ_DIFF,
        _RAND_HECTOMAP,
        columns=_COLS_RZ_DIFF,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # s16a footprint ∩ HectoMAP box – for direct comparison with s16a-based
    # and PDR3 catalogs on the same sky patch.
    "rz_diff_s16a_4bin": _cfg(
        "rz_diff_s16a_4bin",
        _PATH_RZ_DIFF,
        _RAND_S16A,
        columns=_COLS_RZ_DIFF,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "rz_diff_s16a_1bin": _cfg(
        "rz_diff_s16a_1bin",
        _PATH_RZ_DIFF,
        _RAND_S16A,
        columns=_COLS_RZ_DIFF,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # Linear regression against WL mass (ElasticNet on 2D differential profiles, no NMS)
    # -----------------------------------------------------------------------
    "regression_4bin": _cfg(
        "regression_4bin",
        _PATH_REGRESSION,
        _RAND_HECTOMAP,
        columns=_COLS_REGRESSION,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "regression_1bin": _cfg(
        "regression_1bin",
        _PATH_REGRESSION,
        _RAND_HECTOMAP,
        columns=_COLS_REGRESSION,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # s16a footprint ∩ HectoMAP box – for direct comparison with s16a-based
    # and PDR3 catalogs on the same sky patch.
    "regression_s16a_4bin": _cfg(
        "regression_s16a_4bin",
        _PATH_REGRESSION,
        _RAND_S16A,
        columns=_COLS_REGRESSION,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "regression_s16a_1bin": _cfg(
        "regression_s16a_1bin",
        _PATH_REGRESSION,
        _RAND_S16A,
        columns=_COLS_REGRESSION,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # CCA1 cluster finder (2D CoG CCA decomposition, cca1 ranking, no NMS)
    # -----------------------------------------------------------------------
    "cca_4bin": _cfg(
        "cca_4bin",
        _PATH_CCA1,
        _RAND_HECTOMAP,
        columns=_COLS_CCA1,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "cca_1bin": _cfg(
        "cca_1bin",
        _PATH_CCA1,
        _RAND_HECTOMAP,
        columns=_COLS_CCA1,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
    ),
    "cca1_4bin": _cfg(
        "cca1_4bin",
        _PATH_CCA1,
        _RAND_HECTOMAP,
        columns=_COLS_CCA1,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "cca1_1bin": _cfg(
        "cca1_1bin",
        _PATH_CCA1,
        _RAND_HECTOMAP,
        columns=_COLS_CCA1,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
    ),
    # s16a footprint ∩ HectoMAP box – for direct comparison with s16a-based
    # and PDR3 catalogs on the same sky patch.
    "cca_s16a_4bin": _cfg(
        "cca_s16a_4bin",
        _PATH_CCA1,
        _RAND_S16A,
        columns=_COLS_CCA1,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "cca_s16a_1bin": _cfg(
        "cca_s16a_1bin",
        _PATH_CCA1,
        _RAND_S16A,
        columns=_COLS_CCA1,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    "cca1_s16a_4bin": _cfg(
        "cca1_s16a_4bin",
        _PATH_CCA1,
        _RAND_S16A,
        columns=_COLS_CCA1,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "cca1_s16a_1bin": _cfg(
        "cca1_s16a_1bin",
        _PATH_CCA1,
        _RAND_S16A,
        columns=_COLS_CCA1,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # CCA2 cluster finder (2D CoG CCA decomposition, cca2 ranking, no NMS)
    # -----------------------------------------------------------------------
    "cca2_4bin": _cfg(
        "cca2_4bin",
        _PATH_CCA2,
        _RAND_HECTOMAP,
        columns=_COLS_CCA2,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "cca2_1bin": _cfg(
        "cca2_1bin",
        _PATH_CCA2,
        _RAND_HECTOMAP,
        columns=_COLS_CCA2,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
    ),
    "cca2_s16a_4bin": _cfg(
        "cca2_s16a_4bin",
        _PATH_CCA2,
        _RAND_S16A,
        columns=_COLS_CCA2,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "cca2_s16a_1bin": _cfg(
        "cca2_s16a_1bin",
        _PATH_CCA2,
        _RAND_S16A,
        columns=_COLS_CCA2,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # AMICO cluster finder
    # -----------------------------------------------------------------------
    "amico_4bin": _cfg(
        "amico_4bin",
        _PATH_AMICO,
        _RAND_HECTOMAP,
        columns=_COLS_AMICO,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
        **_BOX_HECTOMAP,
    ),
    "amico_1bin": _cfg(
        "amico_1bin",
        _PATH_AMICO,
        _RAND_HECTOMAP,
        columns=_COLS_AMICO,
        redshift_range=_Z_RANGE,
        lens_format="parquet",
        binning=_DEFAULT_BINNING_1BIN,
        **_BOX_HECTOMAP,
    ),
    # -----------------------------------------------------------------------
    # Ideal Theoretical Upper Limit (N-body MDPL2 Simulation, sigma=0)
    # -----------------------------------------------------------------------
    "ideal_mdpl2_1bin": _cfg(
        "ideal_mdpl2_1bin",
        _PATH_IDEAL_MDPL2_1BIN,
        _RAND_HECTOMAP,
        columns=_COLS_COSINE,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    "ideal_mdpl2_4bin": _cfg(
        "ideal_mdpl2_4bin",
        _PATH_IDEAL_MDPL2_4BIN,
        _RAND_HECTOMAP,
        columns=_COLS_COSINE,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
    # -----------------------------------------------------------------------
    # Ideal Theoretical Upper Limit (Colossus Analytical Halo Model, sigma=0)
    # -----------------------------------------------------------------------
    "ideal_colossus_1bin": _cfg(
        "ideal_colossus_1bin",
        _PATH_IDEAL_COLOSSUS_1BIN,
        _RAND_HECTOMAP,
        columns=_COLS_COSINE,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_1BIN,
    ),
    "ideal_colossus_4bin": _cfg(
        "ideal_colossus_4bin",
        _PATH_IDEAL_COLOSSUS_4BIN,
        _RAND_HECTOMAP,
        columns=_COLS_COSINE,
        redshift_range=_Z_RANGE,
        binning=_DEFAULT_BINNING_4BIN,
    ),
}
