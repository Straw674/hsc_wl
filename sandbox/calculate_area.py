# %%
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None

while True:
    if not current_dir or current_dir == current_dir.parent:
        print("Error: pyproject.toml not found in parent directories.")
        break
    if (current_dir / marker).exists():
        root_path = current_dir
        break
    else:
        current_dir = current_dir.parent

if root_path:
    root_path_str = str(root_path)
    if root_path_str not in sys.path:
        sys.path.append(root_path_str)
    from initial import *

import healpy as hp
import numpy as np
from astropy.table import Table

# CONFIGURATION
MASK_PATH = root_path / 'data' / 'mask' / 's19a_fdfc_hp_contarea_izy-gt-5_trimmed.fits'
RA_MIN = 200
RA_MAX = 250
DEC_MIN = 42
DEC_MAX = 44.5

# %%
print(f"Reading mask from {MASK_PATH}...")
t = Table.read(MASK_PATH)
mask = t['T'].data.flatten()
n_pixels = len(mask)
nside = hp.npix2nside(n_pixels)
print(f"Total pixels: {n_pixels}")
print(f"Calculated Nside: {nside}")

# Generate coords
print("Generating pixel coordinates...")
theta, phi = hp.pix2ang(nside, np.arange(n_pixels))
ra = np.degrees(phi)
dec = 90.0 - np.degrees(theta)

# Filter by region
print("Filtering by given RA and Dec range...")
in_region = (ra >= RA_MIN) & (ra <= RA_MAX) & (dec >= DEC_MIN) & (dec <= DEC_MAX)

# Get area
pix_area = hp.nside2pixarea(nside, degrees=True)
valid_pixels = mask[in_region]
num_valid = np.sum(valid_pixels)
area_sq_deg = num_valid * pix_area

print(f"Region: RA {RA_MIN}-{RA_MAX}, Dec {DEC_MIN}-{DEC_MAX}")
print(f"Number of valid pixels in region: {num_valid}")
print(f"Total area in region: {area_sq_deg:.4f} sq deg")
