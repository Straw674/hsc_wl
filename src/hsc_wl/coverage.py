"""Sky-coverage, HealSparse mask, and comoving-volume utilities for HSC weak lensing.

Every lens catalog configuration is associated with an effective boolean HealSparse
mask derived from the HSC Y3 shape catalog mask (NSIDE=8192, bit-packed).
The mask defines:
1. The exact source-galaxy coverage area (A_eff).
2. The spatial boundary for filtering lens galaxies.
3. The uniform spatial distribution from which random catalogs are dynamically sampled.

The *volume factor* (``top_counts_factor``) scales the per-bin lens count so
that catalogs covering different sky areas and redshift shells select a
comparable comoving number density. It is defined relative to a fixed
reference:

    factor = (A_eff / A_ref) * (V_shell(z) / V_shell(z_ref))

where:
* ``A_eff``: effective area of the config's mask in deg^2.
* ``A_ref``: reference area (S16A survey footprint ∩ Y3 8192 mask, ~137.83 deg^2).
* ``V_shell(z)``: full-sky comoving volume between ``z_min`` and ``z_max``.
* ``z_ref``: reference redshift range (0.19, 0.52).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import healpy as hp
import healsparse as hsp
import numpy as np
from astropy.cosmology import Planck18
from astropy.table import Table

if TYPE_CHECKING:
    from hsc_wl.config import LensCatalogConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NSIDE: int = 8192
NSIDE_COVERAGE: int = 32

#: Reference redshift range (s16a baseline).
REFERENCE_Z_RANGE: tuple[float, float] = (0.19, 0.52)

#: Y3 shape catalog mask path relative to project root.
Y3_MASK_PATH: str = "data/mask/hsc_y3_mask_nside8192.hs"

#: S16A reference random catalog (defines the S16A survey area).
REFERENCE_RANDOM_PATH: str = (
    "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits"
)

CANONICAL_FIELDS: tuple[str, ...] = (
    "GAMA09H",
    "WIDE12H",
    "GAMA15H",
    "VVDS",
    "XMM",
    "HECTOMAP",
)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "SPRING": ("GAMA09H", "WIDE12H", "GAMA15H"),
    "AUTUMN": ("XMM", "VVDS"),
    "NORTH": ("HECTOMAP",),
    "ALL": CANONICAL_FIELDS,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (pyproject.toml).")


def _resolve_path(path_value: str, root: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else root / p


def expand_field_names(fields: Sequence[str] | None) -> tuple[str, ...]:
    """Expand field aliases to canonical field names, preserving order without duplicates.

    Parameters
    ----------
    fields : Sequence[str] or None
        List or tuple of field names or aliases (e.g. ``["SPRING"]``, ``["HECTOMAP"]``).
        ``None`` expands to all survey fields.

    Returns
    -------
    tuple[str, ...]
        Tuple of canonical field names.
    """
    if fields is None:
        return CANONICAL_FIELDS

    expanded: list[str] = []
    for f in fields:
        f_upper = f.strip().upper()
        if f_upper in FIELD_ALIASES:
            for sub_f in FIELD_ALIASES[f_upper]:
                if sub_f not in expanded:
                    expanded.append(sub_f)
        elif f_upper in CANONICAL_FIELDS:
            if f_upper not in expanded:
                expanded.append(f_upper)
        else:
            valid_names = sorted(set(CANONICAL_FIELDS) | set(FIELD_ALIASES.keys()))
            raise ValueError(
                f"Unknown field '{f}'. Valid fields and aliases: {valid_names}"
            )

    return tuple(expanded)


def in_field(ra: np.ndarray, dec: np.ndarray, field_name: str) -> np.ndarray:
    """Return boolean mask for coordinates falling inside a canonical HSC field.

    Parameters
    ----------
    ra : np.ndarray
        Right ascension array in degrees.
    dec : np.ndarray
        Declination array in degrees.
    field_name : str
        Canonical field name (e.g. ``'HECTOMAP'``, ``'GAMA09H'``).

    Returns
    -------
    np.ndarray
        Boolean array.
    """
    f_upper = field_name.strip().upper()
    if f_upper == "HECTOMAP":
        return dec > 30.0

    dec_eq = dec < 10.0
    if f_upper == "XMM":
        return dec_eq & (ra >= 25.0) & (ra <= 45.0)
    if f_upper == "VVDS":
        return dec_eq & ((ra >= 320.0) | (ra <= 15.0))
    if f_upper == "GAMA09H":
        return dec_eq & (ra >= 125.0) & (ra < 153.5)
    if f_upper == "WIDE12H":
        return dec_eq & (ra >= 153.5) & (ra < 203.0)
    if f_upper == "GAMA15H":
        return dec_eq & (ra >= 203.0) & (ra <= 235.0)

    raise ValueError(f"Unknown canonical field: {field_name}")


def load_y3_mask(root: Path | None = None) -> hsp.HealSparseMap:
    """Load the master HSC Y3 shape catalog HealSparse mask (NSIDE=8192).

    Parameters
    ----------
    root : Path or None
        Project root directory.

    Returns
    -------
    hsp.HealSparseMap
        Master boolean HealSparse map.
    """
    root_dir = _find_root(root)
    mask_path = root_dir / Y3_MASK_PATH
    if not mask_path.exists():
        raise FileNotFoundError(f"HSC Y3 8192 mask not found: {mask_path}")
    return hsp.HealSparseMap.read(str(mask_path))


def load_s16a_pixset_1024(root: Path | None = None) -> set[int]:
    """Load the S16A survey footprint as a unique pixel set at NSIDE=1024 (NESTED).

    Parameters
    ----------
    root : Path or None
        Project root directory.

    Returns
    -------
    set[int]
        Unique nested HEALPix pixel indices at NSIDE=1024.
    """
    root_dir = _find_root(root)
    rand_path = _resolve_path(REFERENCE_RANDOM_PATH, root_dir)
    if not rand_path.exists():
        raise FileNotFoundError(f"S16A reference random catalog not found: {rand_path}")

    t = Table.read(str(rand_path))
    ra = np.asarray(t["ra"], dtype=np.float64)
    dec = np.asarray(t["dec"], dtype=np.float64)
    pix = hp.ang2pix(1024, ra, dec, nest=True, lonlat=True)
    return set(np.unique(pix).tolist())


# ---------------------------------------------------------------------------
# Mask construction & effective area
# ---------------------------------------------------------------------------


def build_config_mask(
    lens_cfg: LensCatalogConfig,
    root: Path | None = None,
) -> hsp.HealSparseMap:
    """Build the configuration-specific HealSparse mask.

    Applies the specified field selection, survey footprint overlap (e.g. S16A),
    and optional box range cuts to the Y3 shape catalog mask.

    Parameters
    ----------
    lens_cfg : LensCatalogConfig
        Lens configuration (specifies ``fields``, ``survey_overlap``,
        ``ra_range``, ``dec_range``).
    root : Path or None
        Project root directory.

    Returns
    -------
    hsp.HealSparseMap
        Filtered boolean HealSparse map.
    """
    root_dir = _find_root(root)
    base_mask = load_y3_mask(root_dir)
    valid_pixels = base_mask.valid_pixels
    nside_sparse = base_mask.nside_sparse

    ra_pix, dec_pix = hp.pix2ang(nside_sparse, valid_pixels, nest=True, lonlat=True)

    # 1. Field selection
    canonical_fields = expand_field_names(lens_cfg.fields)
    keep = np.zeros(len(valid_pixels), dtype=bool)
    for field_name in canonical_fields:
        keep |= in_field(ra_pix, dec_pix, field_name)

    # 2. S16A survey overlap
    if lens_cfg.survey_overlap is not None:
        overlap_key = lens_cfg.survey_overlap.strip().lower()
        if overlap_key == "s16a":
            s16a_pix_1024 = load_s16a_pixset_1024(root_dir)
            # In NESTED ordering, dividing by (8192 / 1024)**2 = 64 maps to parent NSIDE=1024
            parent_1024 = valid_pixels // 64
            keep &= np.isin(parent_1024, list(s16a_pix_1024))
        else:
            raise ValueError(f"Unsupported survey_overlap: {lens_cfg.survey_overlap}")

    # 3. Optional box cuts
    if lens_cfg.ra_range is not None:
        keep &= (ra_pix >= lens_cfg.ra_range[0]) & (ra_pix <= lens_cfg.ra_range[1])
    if lens_cfg.dec_range is not None:
        keep &= (dec_pix >= lens_cfg.dec_range[0]) & (dec_pix <= lens_cfg.dec_range[1])

    selected_pixels = valid_pixels[keep]

    config_mask = hsp.HealSparseMap.make_empty(
        nside_coverage=base_mask.nside_coverage,
        nside_sparse=nside_sparse,
        dtype=bool,
        bit_packed=True,
        sentinel=False,
    )
    if len(selected_pixels) > 0:
        config_mask.update_values_pix(
            selected_pixels, np.ones(len(selected_pixels), dtype=bool)
        )

    return config_mask


def effective_area_deg2(
    lens_cfg: LensCatalogConfig,
    root: Path | None = None,
) -> float:
    """Compute effective geometric area (deg^2) directly from the config mask.

    Parameters
    ----------
    lens_cfg : LensCatalogConfig
        Lens configuration.
    root : Path or None
        Project root directory.

    Returns
    -------
    float
        Effective area in deg^2.
    """
    config_mask = build_config_mask(lens_cfg, root)
    pix_area = hp.nside2pixarea(config_mask.nside_sparse, degrees=True)
    return float(len(config_mask.valid_pixels) * pix_area)


def reference_area_deg2(root: Path | None = None) -> float:
    """Reference area: S16A survey footprint ∩ Y3 8192 mask (~137.83 deg^2).

    Serves as the baseline (factor = 1.0) for the reference redshift shell (0.19, 0.52).
    """
    from hsc_wl.config import ColumnMapping, LensCatalogConfig

    ref_cfg = LensCatalogConfig(
        label="ref_s16a",
        lens_path="",
        columns=ColumnMapping(col_rank="", ra="", dec="", z=""),
        redshift_range=REFERENCE_Z_RANGE,
        survey_overlap="s16a",
    )
    return effective_area_deg2(ref_cfg, root)


# ---------------------------------------------------------------------------
# Comoving volume & volume factor
# ---------------------------------------------------------------------------


def comoving_volume_shell_gpc3(
    z_min: float,
    z_max: float,
    cosmo: Planck18 = Planck18,
) -> float:
    """Full-sky comoving volume between *z_min* and *z_max* in Gpc^3."""
    return float(
        (cosmo.comoving_volume(z_max) - cosmo.comoving_volume(z_min)).to_value(u.Gpc**3)
    )


def _volume_ratio(
    z_range: tuple[float, float],
    ref_z_range: tuple[float, float] = REFERENCE_Z_RANGE,
) -> float:
    """V_shell(z_range) / V_shell(ref_z_range)."""
    return comoving_volume_shell_gpc3(*z_range) / comoving_volume_shell_gpc3(
        *ref_z_range
    )


def volume_factor(
    area: float,
    z_range: tuple[float, float],
    root: Path | None = None,
    ref_z_range: tuple[float, float] = REFERENCE_Z_RANGE,
) -> float:
    """Compute volume factor relative to the S16A reference.

    factor = (area / A_ref) * (V(z) / V(z_ref))
    """
    ref_area = reference_area_deg2(root)
    return (area / ref_area) * _volume_ratio(z_range, ref_z_range)


def resolve_area_and_factor(
    lens_cfg: LensCatalogConfig,
    root: Path | None = None,
) -> tuple[float, float]:
    """Compute the effective area and volume factor for a lens configuration.

    Returns
    -------
    tuple[float, float]
        ``(area_deg2, top_counts_factor)``.
    """
    area = effective_area_deg2(lens_cfg, root)
    factor = volume_factor(area, lens_cfg.redshift_range, root)
    logger.info(
        "[coverage] %s: area=%.4f deg2  z=%s  factor=%.6f",
        lens_cfg.label,
        area,
        lens_cfg.redshift_range,
        factor,
    )
    return area, factor


# ---------------------------------------------------------------------------
# Filtering & random sampling
# ---------------------------------------------------------------------------


def filter_lens_by_mask(
    lens: Table,
    mask: hsp.HealSparseMap,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> Table:
    """Remove lens objects that fall outside the given HealSparse mask.

    Parameters
    ----------
    lens : Table
        Lens catalog.
    mask : hsp.HealSparseMap
        Config mask at NSIDE=8192.
    ra_col : str
        Right ascension column name.
    dec_col : str
        Declination column name.

    Returns
    -------
    Table
        Filtered copy containing only objects inside the mask.
    """
    ra = np.asarray(lens[ra_col], dtype=np.float64)
    dec = np.asarray(lens[dec_col], dtype=np.float64)
    inside = mask.get_values_pos(ra, dec, lonlat=True)

    n_total = len(lens)
    n_outside = int(np.sum(~inside))
    logger.info(
        "[coverage] mask filter: %d / %d lenses outside mask removed (%.1f%%)",
        n_outside,
        n_total,
        100.0 * n_outside / n_total if n_total > 0 else 0.0,
    )
    return lens[inside]


def sample_random_from_mask(
    mask: hsp.HealSparseMap,
    n_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample (RA, Dec) coordinates strictly within the valid pixels of a HealSparse mask.

    Uses pixel selection with disk jitter and rejection sampling to guarantee that
    100% of generated points reside within valid mask pixels.

    Parameters
    ----------
    mask : hsp.HealSparseMap
        Boolean HealSparse map.
    n_points : int
        Number of random coordinates to generate.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of (RA, Dec) in degrees.
    """
    if rng is None:
        rng = np.random.default_rng()

    valid_pixels = mask.valid_pixels
    if len(valid_pixels) == 0:
        raise ValueError("Cannot sample points from an empty mask.")

    ra_out = np.empty(n_points, dtype=np.float64)
    dec_out = np.empty(n_points, dtype=np.float64)
    n_filled = 0
    nside_sparse = mask.nside_sparse
    pixrad = hp.max_pixrad(nside_sparse)

    while n_filled < n_points:
        n_needed = n_points - n_filled
        n_batch = int(n_needed * 1.1) + 100
        sampled_pix = rng.choice(valid_pixels, size=n_batch, replace=True)

        theta, phi = hp.pix2ang(nside_sparse, sampled_pix, nest=True)
        r = pixrad * np.sqrt(rng.uniform(0.0, 1.0, n_batch))
        ang = rng.uniform(0.0, 2.0 * np.pi, n_batch)
        dtheta = r * np.cos(ang)
        dphi = r * np.sin(ang) / np.sin(np.clip(theta, 1e-3, np.pi - 1e-3))

        theta_rand = np.clip(theta + dtheta, 1e-6, np.pi - 1e-6)
        phi_rand = np.mod(phi + dphi, 2.0 * np.pi)

        ra_cand = np.degrees(phi_rand)
        dec_cand = 90.0 - np.degrees(theta_rand)

        valid = mask.get_values_pos(ra_cand, dec_cand, lonlat=True)
        n_good = int(np.sum(valid))
        if n_good > 0:
            n_take = min(n_good, n_needed)
            ra_out[n_filled : n_filled + n_take] = ra_cand[valid][:n_take]
            dec_out[n_filled : n_filled + n_take] = dec_cand[valid][:n_take]
            n_filled += n_take

    return ra_out, dec_out


# ---------------------------------------------------------------------------
# Diagnostics & inspection utilities
# ---------------------------------------------------------------------------


def y3_mask_area_deg2(root: Path | None = None) -> float:
    """Total area of the master Y3 shape mask in deg^2."""
    mask = load_y3_mask(root)
    pix_area = hp.nside2pixarea(mask.nside_sparse, degrees=True)
    return float(len(mask.valid_pixels) * pix_area)


def field_areas_deg2(root: Path | None = None) -> dict[str, float]:
    """Compute effective area per canonical field and total in deg^2."""
    mask = load_y3_mask(root)
    valid_pixels = mask.valid_pixels
    pix_area = hp.nside2pixarea(mask.nside_sparse, degrees=True)
    ra_pix, dec_pix = hp.pix2ang(
        mask.nside_sparse, valid_pixels, nest=True, lonlat=True
    )

    areas: dict[str, float] = {}
    for f in CANONICAL_FIELDS:
        cnt = int(np.sum(in_field(ra_pix, dec_pix, f)))
        areas[f] = cnt * pix_area
    areas["Total"] = len(valid_pixels) * pix_area
    return areas
