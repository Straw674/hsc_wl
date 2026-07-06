"""Generate a uniform random catalog over the Y3 (s19a FDFC) mask.

The output ``data/random_y3_mask.fits`` covers the full source/shape catalog
footprint and is used as the lens random for full-area lens configs (e.g.
unfiltered CAMIRA) where no survey-specific random catalog exists.

Method:
    1. Load the Y3 mask HEALPix pixels (NSIDE=1024).
    2. Uniformly select N pixels (with replacement).
    3. Place a random point inside each selected pixel (pixel centre + disk
       jitter of radius ``hp.max_pixrad``).
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
while (
    project_root != project_root.parent
    and not (project_root / "pyproject.toml").exists()
):
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.table import Table

NSIDE = 1024
N_POINTS = 20_000_000
RNG_SEED = 42
OUT_PATH = project_root / "data" / "random_y3_mask.fits"
MASK_PATH = (
    project_root / "data" / "mask" / "s19a_fdfc_hp_contarea_izy-gt-5_trimmed.fits"
)


# %% [Stage 1: Load Y3 mask]
hdu = fits.open(MASK_PATH)[1]
flat = hdu.data[hdu.data.names[0]].ravel()
mask_pix = np.where(flat)[0].astype(np.int64)
pixarea = hp.nside2pixarea(NSIDE, degrees=True)
print(f"Y3 mask: {len(mask_pix)} pixels, {len(mask_pix) * pixarea:.4f} deg2")


# %% [Stage 2: Generate random points]
rng = np.random.default_rng(RNG_SEED)
selected_pix = rng.choice(mask_pix, size=N_POINTS, replace=True)

theta, phi = hp.pix2ang(NSIDE, selected_pix, nest=False)

pixrad = hp.max_pixrad(NSIDE)
r = pixrad * np.sqrt(rng.uniform(0, 1, N_POINTS))
ang_offset = rng.uniform(0, 2 * np.pi, N_POINTS)
dtheta = r * np.cos(ang_offset)
dphi = r * np.sin(ang_offset) / np.sin(np.clip(theta, 1e-3, np.pi - 1e-3))

theta_rand = np.clip(theta + dtheta, 1e-6, np.pi - 1e-6)
phi_rand = np.mod(phi + dphi, 2 * np.pi)

ra = np.degrees(phi_rand)
dec = 90.0 - np.degrees(theta_rand)

verify_pix = hp.ang2pix(NSIDE, theta_rand, phi_rand, nest=False)
frac_in_mask = np.mean(np.isin(verify_pix, mask_pix))
print(f"Points in mask: {frac_in_mask * 100:.2f}%")
print(f"RA range: {ra.min():.4f} - {ra.max():.4f}")
print(f"Dec range: {dec.min():.4f} - {dec.max():.4f}")


# %% [Stage 3: Write FITS]
t = Table()
t["object_id"] = np.arange(N_POINTS, dtype=np.int64)
t["ra"] = ra.astype(np.float64)
t["dec"] = dec.astype(np.float64)
t.write(str(OUT_PATH), overwrite=True)
print(f"Wrote {N_POINTS} random points to {OUT_PATH}")
