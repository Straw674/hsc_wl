"""Sky-coverage and comoving-volume utilities for fair inter-lens comparison.

The *volume factor* (``top_counts_factor``) scales the per-bin lens count so
that catalogs covering different sky areas and redshift shells select a
comparable comoving number density.  It is defined relative to a fixed
reference:

    factor = (A_eff / A_ref) * (V_shell(z) / V_shell(z_ref))

where:

* ``A_eff`` — effective area = Y3 (s19a FDFC) mask pixel count × pixel area,
  i.e. the area is read directly from the mask file without any random-catalog
  intermediary, avoiding Poisson shot-noise from sparse random points.
* ``V_shell(z)`` — full-sky comoving volume between ``z_min`` and ``z_max``.
* Reference: ``z_ref = (0.19, 0.52)``, ``A_ref`` = s16a-wide random ∩ Y3 mask
  (s19a FDFC).  The s16a configs therefore have ``factor = 1`` by construction.

Design notes:
- No lru_cache: mask loading is fast (~0.1 s) and caching complicates
  interactive workflows where the mask file may change between runs.
- RA/Dec box cuts: when a lens config defines ``ra_range`` / ``dec_range``
  the random catalog is clipped to that box before pixelising.  The
  effective area is then ``box ∩ lens-random footprint ∩ Y3 mask``, i.e.
  the intersection of the rectangular cut, the lens survey's actual
  coverage, and the Y3 source-coverage mask.  Configs without box cuts
  use the full random footprint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table

from hsc_wl.config import LensCatalogConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NSIDE = 1024

#: Reference redshift range (s16a baseline).
REFERENCE_Z_RANGE: tuple[float, float] = (0.19, 0.52)

#: Y3 (s19a full-depth-full-colour) mask, relative to project root.
Y3_MASK_PATH = "data/mask/s19a_fdfc_hp_contarea_izy-gt-5_trimmed.fits"

#: s16a wide-field random catalog — defines the reference sky area.
REFERENCE_RANDOM_PATH = (
    "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits"
)


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


def _y3_mask_pixset(root: Path, nside: int = NSIDE) -> np.ndarray:
    """Load the Y3 (s19a FDFC) mask as a sorted HEALPix pixel index array.

    The mask file is a healsparse-style boolean map (RING ordering) stored
    as a ``(12288, 1024)`` bool array in a BinTable HDU.
    """
    path = root / Y3_MASK_PATH
    if not path.exists():
        raise FileNotFoundError(f"Y3 mask not found: {path}")
    hdu = fits.open(path)[1]
    flat = hdu.data[hdu.data.names[0]].ravel()
    pix = np.where(flat)[0].astype(np.int64)
    logger.info(
        "[coverage] Y3 mask loaded: %d pixels (%.3f deg2)",
        len(pix),
        len(pix) * hp.nside2pixarea(nside, degrees=True),
    )
    return pix


def _pixset_from_radec(
    ra: np.ndarray, dec: np.ndarray, nside: int = NSIDE
) -> np.ndarray:
    """Build a sorted unique pixel set from RA/Dec arrays."""
    ipix = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra), nest=False)
    return np.unique(ipix)


def _random_pixset(
    random_path: str,
    nside: int = NSIDE,
    ra_range: tuple[float, float] | None = None,
    dec_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Load a random catalog's footprint as a sorted pixel set.

    When *ra_range* / *dec_range* are given the random points are first
    clipped to the rectangular box; the returned pixel set then reflects
    the box-limited lens-survey footprint (still to be intersected with
    the Y3 source mask downstream).
    """
    t = Table.read(random_path)
    ra_col = next((c for c in t.colnames if c.lower() == "ra"), "ra")
    dec_col = next((c for c in t.colnames if c.lower() == "dec"), "dec")
    ra = np.asarray(t[ra_col], float)
    dec = np.asarray(t[dec_col], float)
    keep = np.ones(len(ra), dtype=bool)
    if ra_range is not None:
        keep &= (ra >= ra_range[0]) & (ra <= ra_range[1])
    if dec_range is not None:
        keep &= (dec >= dec_range[0]) & (dec <= dec_range[1])
    ra = ra[keep]
    dec = dec[keep]
    pix = _pixset_from_radec(ra, dec, nside)
    logger.info(
        "[coverage] random footprint: %s  %d pixels (%.4f deg2)%s",
        Path(random_path).name,
        len(pix),
        len(pix) * hp.nside2pixarea(nside, degrees=True),
        f"  box=[RA {ra_range}, Dec {dec_range}]" if ra_range or dec_range else "",
    )
    return pix


# ---------------------------------------------------------------------------
# Reference area (s16a random ∩ Y3 mask)
# ---------------------------------------------------------------------------


def _reference_area_deg2(root: Path, nside: int = NSIDE) -> float:
    """Reference area = s16a-wide random ∩ Y3 mask (s19a FDFC).

    This is the effective source-covered area of the s16a wide-field
    survey, which serves as the ``factor = 1`` baseline.
    """
    rand_path = _resolve_path(REFERENCE_RANDOM_PATH, root)
    rand_pix = _random_pixset(str(rand_path), nside)
    y3_pix = _y3_mask_pixset(root, nside)
    overlap = np.intersect1d(rand_pix, y3_pix, assume_unique=True)
    area = len(overlap) * hp.nside2pixarea(nside, degrees=True)
    logger.info("[coverage] reference area (s16a random ∩ Y3) = %.4f deg2", area)
    return area


# ---------------------------------------------------------------------------
# Comoving volume
# ---------------------------------------------------------------------------


def comoving_volume_shell_gpc3(z_min: float, z_max: float, cosmo=Planck18) -> float:
    """Full-sky comoving volume between *z_min* and *z_max* in Gpc^3."""
    return (cosmo.comoving_volume(z_max) - cosmo.comoving_volume(z_min)).to_value(
        u.Gpc**3
    )


def _volume_ratio(
    z_range: tuple[float, float],
    ref_z_range: tuple[float, float] = REFERENCE_Z_RANGE,
) -> float:
    """V_shell(z_range) / V_shell(ref_z_range)."""
    return comoving_volume_shell_gpc3(*z_range) / comoving_volume_shell_gpc3(
        *ref_z_range
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def effective_area_deg2(
    lens_cfg: LensCatalogConfig,
    root: Path | None = None,
    nside: int = NSIDE,
) -> float:
    """Effective area = box ∩ lens-random footprint ∩ Y3 mask.

    The lens footprint is determined by pixelising the random catalog
    associated with ``lens_cfg``.  When ``lens_cfg.ra_range`` /
    ``lens_cfg.dec_range`` are set the random points are first clipped to
    that rectangular box.  The (box-limited) lens footprint is then
    intersected with the Y3 (s19a FDFC) source-coverage mask, yielding the
    true lens+source overlap area.

    Parameters
    ----------
    lens_cfg : LensCatalogConfig
        Lens configuration (provides ``random_path``, ``ra_range``,
        ``dec_range``).
    root : Path or None
        Project root for resolving relative paths.
    nside : int
        HEALPix resolution (default 1024 to match the Y3 mask).

    Returns
    -------
    float
        Effective area in deg^2.
    """
    root = _find_root(root)
    rand_path = _resolve_path(lens_cfg.random_path, root)
    rand_pix = _random_pixset(
        str(rand_path),
        nside,
        ra_range=lens_cfg.ra_range,
        dec_range=lens_cfg.dec_range,
    )
    y3_pix = _y3_mask_pixset(root, nside)
    overlap = np.intersect1d(rand_pix, y3_pix, assume_unique=True)
    area = len(overlap) * hp.nside2pixarea(nside, degrees=True)
    if lens_cfg.ra_range or lens_cfg.dec_range:
        logger.info(
            "[coverage] box ∩ random ∩ Y3 = %d pixels (%.4f deg2)",
            len(overlap),
            area,
        )
    return area


def volume_factor(
    area: float,
    z_range: tuple[float, float],
    root: Path | None = None,
    ref_z_range: tuple[float, float] = REFERENCE_Z_RANGE,
    nside: int = NSIDE,
) -> float:
    """Volume factor relative to the s16a reference.

    ``factor = (area / A_ref) * (V(z) / V(z_ref))``

    ``A_ref`` is the s16a-wide random ∩ Y3 mask area, computed from data.
    """
    root = _find_root(root)
    ref_area = _reference_area_deg2(root, nside)
    return (area / ref_area) * _volume_ratio(z_range, ref_z_range)


def resolve_area_and_factor(
    lens_cfg: LensCatalogConfig,
    root: Path | None = None,
    nside: int = NSIDE,
) -> tuple[float, float]:
    """Compute the effective area and volume factor for a lens config.

    Returns ``(area_deg2, factor)``.
    """
    root = _find_root(root)
    area = effective_area_deg2(lens_cfg, root, nside)
    factor = volume_factor(area, lens_cfg.redshift_range, root, nside=nside)
    logger.info(
        "[coverage] %s: area=%.4f deg2  z=%s  factor=%.6f",
        lens_cfg.label,
        area,
        lens_cfg.redshift_range,
        factor,
    )
    return area, factor


def filter_lens_by_mask(
    lens: "Table",
    root: Path | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
    nside: int = NSIDE,
) -> "Table":
    """Remove lens objects that fall outside the Y3 (s19a FDFC) mask.

    Lenses outside the Y3 source footprint cannot contribute to ΔΣ because
    there are no background source galaxies near them.  Keeping them in the
    pool biases top-N / top-counts selection when the lens catalog extends
    beyond the source survey footprint (e.g. CAMIRA from S23B, or redMaPPer
    runs without a spatial mask).

    Parameters
    ----------
    lens : astropy.table.Table
        Lens catalog.  Must contain RA/Dec columns.
    root : Path or None
        Project root for resolving the mask path.
    ra_col, dec_col : str
        Column names for right ascension and declination (degrees).
    nside : int
        HEALPix resolution — must match the mask file (default 1024).

    Returns
    -------
    astropy.table.Table
        Filtered copy of *lens* containing only objects inside the mask.
    """
    root = _find_root(root)
    y3_pix = _y3_mask_pixset(root, nside)
    y3_pix_set = set(y3_pix.tolist())

    ra = np.asarray(lens[ra_col], float)
    dec = np.asarray(lens[dec_col], float)
    ipix = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra), nest=False)
    inside = np.isin(ipix, list(y3_pix_set))

    n_total = len(lens)
    n_outside = int(np.sum(~inside))
    logger.info(
        "[coverage] mask filter: %d / %d lenses outside Y3 mask removed (%.1f%%)",
        n_outside,
        n_total,
        100.0 * n_outside / n_total if n_total > 0 else 0.0,
    )
    return lens[inside]


# ---------------------------------------------------------------------------
# Mask / source-footprint inspection utilities
# ---------------------------------------------------------------------------


def y3_mask_area_deg2(root: Path | None = None, nside: int = NSIDE) -> float:
    """Total area of the Y3 (s19a FDFC) mask in deg^2."""
    root = _find_root(root)
    pix = _y3_mask_pixset(root, nside)
    return len(pix) * hp.nside2pixarea(nside, degrees=True)


def _y3_mask_regions(root: Path, nside: int = NSIDE) -> list[tuple[int, float, int]]:
    """Identify disjoint connected regions in the Y3 mask via union-find.

    Returns a list of ``(region_index, area_deg2, n_pixels)`` sorted by
    descending area.
    """
    pix = _y3_mask_pixset(root, nside)
    pix_area = hp.nside2pixarea(nside, degrees=True)
    pix_set = set(pix.tolist())

    parent = {p: p for p in pix.tolist()}

    def find(x: int) -> int:
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for p in pix.tolist():
        for n in hp.get_all_neighbours(nside, p, nest=False):
            if n >= 0 and n in pix_set:
                union(p, n)

    from collections import Counter

    counts = Counter(find(p) for p in pix.tolist())
    regions = [
        (i, cnt * pix_area, cnt) for i, (_root, cnt) in enumerate(counts.most_common())
    ]
    logger.info(
        "[coverage] Y3 mask has %d disjoint regions (total %.4f deg2)",
        len(regions),
        sum(r[1] for r in regions),
    )
    return regions


def y3_mask_regions_deg2(
    root: Path | None = None, nside: int = NSIDE
) -> list[tuple[int, float, int]]:
    """Return the disjoint regions of the Y3 mask as ``(index, area_deg2, n_pixels)``.

    Sorted by descending area.  Useful for verifying that a lens catalog
    covering only a subset of regions gets the correct summed area.
    """
    root = _find_root(root)
    return _y3_mask_regions(root, nside)


def source_footprint_area_deg2(
    source_file: str | Path,
    root: Path | None = None,
    nside: int = NSIDE,
) -> float:
    """Area of the source catalog's pixel footprint in deg^2.

    This pixelises the source-galaxy RA/Dec and counts unique HEALPix
    pixels.  It is *not* the same as :func:`y3_mask_area_deg2` because
    stochastic sampling means some mask pixels contain no source galaxies.
    """
    root = _find_root(root)
    path = _resolve_path(str(source_file), root)
    t = Table.read(str(path))
    ra_col = next((c for c in t.colnames if c.lower() in ("ra", "i_ra")), "ra")
    dec_col = next((c for c in t.colnames if c.lower() in ("dec", "i_dec")), "dec")
    ra = np.asarray(t[ra_col], float)
    dec = np.asarray(t[dec_col], float)
    pix = _pixset_from_radec(ra, dec, nside)
    area = len(pix) * hp.nside2pixarea(nside, degrees=True)
    logger.info(
        "[coverage] source footprint %s: %d pixels (%.4f deg2)",
        Path(source_file).name,
        len(pix),
        area,
    )
    return area
