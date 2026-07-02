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
    cfg = replace(RUN_REGISTRY["cosine"], n_jackknife=50)
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
    The ``top_counts`` / ``edges_*`` fields hold the *raw* user-specified
    values; the per-catalog ``top_counts_factor`` is applied separately by
    :func:`resolve_binning`.
    """

    mode: Literal["edges", "top_counts", "top_n"] = "top_counts"
    top_counts: tuple[int, ...] = (53, 196, 660, 1159)
    top_n: int = 800
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
    ``top_counts``.  No-op for non-``top_counts`` modes or when ``factor``
    is ``1.0``.
    """
    if binning.mode != "top_counts" or factor == 1.0:
        return binning
    scaled = tuple(int(round(c * factor)) for c in binning.top_counts)
    if any(c <= 0 for c in scaled):
        raise ValueError(
            f"Scaled top_counts must be positive (factor={factor}, "
            f"raw={binning.top_counts}, scaled={scaled})."
        )
    return replace(binning, top_counts=scaled)


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
# Run registry
# ---------------------------------------------------------------------------

_EDGES_RICHNESS = (6.0, 10.0, 20.0, 35.0, 120.0)
_EDGES_MASS = (10.63, 10.8, 11.0, 11.2, 11.6)
_TOP_COUNTS_TOPN = (53, 196, 660, 1159)

_DEFAULT_BINNING_4BIN = BinningConfig(
    mode="top_counts",
    top_counts=_TOP_COUNTS_TOPN,
    edges_richness=_EDGES_RICHNESS,
    edges_mass=_EDGES_MASS,
)
_DEFAULT_BINNING_TOPN = BinningConfig(
    mode="top_n",
    top_counts=_TOP_COUNTS_TOPN,
    top_n=800,
    edges_richness=_EDGES_RICHNESS,
    edges_mass=_EDGES_MASS,
)


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
    """Compact constructor for :class:`WLConfig` registry entries."""
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
        save_root=save_root or f"output/{label}",
    )


RUN_REGISTRY: dict[str, WLConfig] = {
    "pdr3_redm_hsc_no_mask": _cfg(
        "pdr3_redm_hsc_no_mask",
        "/Users/xinq/redmapper_HSC/output/redmapper_run/new_run_no_mask/run/hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit",
        "data/random_hectomap.fits",
        columns={
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
        redshift_range=(0.19, 0.52),
    ),
    "pdr3_redm_hsc_5bands_offdiag": _cfg(
        "pdr3_redm_hsc_5bands_offdiag",
        "/Users/xinq/redmapper_HSC/output/redmapper_run/new_run_5bands_offdiag/run/hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit",
        "data/random_hectomap.fits",
        columns={
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
        redshift_range=(0.19, 0.52),
    ),
    "pdr3_redm_hsc_free_offdiag": _cfg(
        "pdr3_redm_hsc_free_offdiag",
        "/Users/xinq/redmapper_HSC/output/redmapper_run/new_run_free_offdiag/run/hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit",
        "data/random_hectomap.fits",
        columns={
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
        redshift_range=(0.19, 0.52),
    ),
    "s16a_redm_hsc": _cfg(
        "s16a_redm_hsc",
        "/Users/xinq/redmapper_HSC/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits",
        "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        columns={
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
        redshift_range=(0.19, 0.52),
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "s16a_logm_50_100": _cfg(
        "s16a_logm_50_100",
        "/Users/xinq/redmapper_HSC/data/reference/s16a_massive_logm_11.2.fits",
        "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        columns={
            "col_rank": "logm_50_100",
            "ra": "ra",
            "dec": "dec",
            "z": "z_best",
        },
        redshift_range=(0.19, 0.52),
    ),
    "s16a_redm_hsc_topn": _cfg(
        "s16a_redm_hsc_topn",
        "/Users/xinq/redmapper_HSC/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits",
        "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        columns={
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
        redshift_range=(0.19, 0.52),
        binning=_DEFAULT_BINNING_TOPN,
    ),
    "s16a_logm_50_100_topn": _cfg(
        "s16a_logm_50_100_topn",
        "/Users/xinq/redmapper_HSC/data/reference/s16a_massive_logm_11.2.fits",
        "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        columns={
            "col_rank": "logm_50_100",
            "ra": "ra",
            "dec": "dec",
            "z": "z_best",
        },
        redshift_range=(0.19, 0.52),
        binning=_DEFAULT_BINNING_TOPN,
    ),
    "forced": _cfg(
        "forced",
        "/Users/xinq/redmapper_HSC/output/s16a_massive_logm_11.2_forced_results.fits",
        "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        columns={
            "col_rank": "lam",
            "ra": "ra",
            "dec": "dec",
            "z": "z_best",
        },
        redshift_range=(0.19, 0.52),
    ),
    "camira": _cfg(
        "camira",
        "data/camira_s23b_wide_sm_v3.dat",
        "data/random_y3_mask.fits",
        columns={
            "col_rank": "N_mem",
            "ra": "RA",
            "dec": "Dec",
            "z": "z_cl",
        },
        redshift_range=(0.19, 0.52),
        lens_format="pandas_dat",
        binning=_DEFAULT_BINNING_TOPN,
    ),
    "camira_4bin": _cfg(
        "camira_4bin",
        "data/camira_s23b_wide_sm_v3.dat",
        "data/random_y3_mask.fits",
        columns={
            "col_rank": "N_mem",
            "ra": "RA",
            "dec": "Dec",
            "z": "z_cl",
        },
        redshift_range=(0.19, 0.52),
        lens_format="pandas_dat",
        binning=_DEFAULT_BINNING_4BIN,
    ),
    "cosine": _cfg(
        "cosine",
        get_latest_cluster_catalog(),
        "data/random_hectomap.fits",
        columns={
            "col_rank": "true_richness",
            "ra": "ra",
            "dec": "dec",
            "z": "z_cl",
        },
        redshift_range=(0.19, 0.52),
        lens_format="parquet",
        binning=_DEFAULT_BINNING_TOPN,
    ),
    "cosine_4bin": _cfg(
        "cosine_4bin",
        get_latest_cluster_catalog(),
        "data/random_hectomap.fits",
        columns={
            "col_rank": "true_richness",
            "ra": "ra",
            "dec": "dec",
            "z": "z_cl",
        },
        redshift_range=(0.19, 0.52),
        lens_format="parquet",
        binning=_DEFAULT_BINNING_4BIN,
    ),
}
