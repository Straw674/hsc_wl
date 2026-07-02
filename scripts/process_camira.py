# %%
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

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

import pandas as pd
import numpy as np
import healpy as hp
from astropy.io import fits

# %%
# CONFIGURATION
IN_CATALOG = "data/camira_s23b_wide_sm_v3.dat"
OUT_CATALOG = "data/camira_s23b_wide_sm_v3_filtered.dat"
MASK_FILE = "data/mask/s19a_fdfc_hp_contarea_izy-gt-5_trimmed.fits"

RA_MIN, RA_MAX = 210.0, 250.0
DEC_MIN, DEC_MAX = 42.0, 44.5

# %%
# EXECUTION
print(f"Reading catalog from {IN_CATALOG}")
with open(IN_CATALOG, 'r') as f:
    header_line = f.readline().strip()
    header = header_line.lstrip('#').split()

df = pd.read_csv(IN_CATALOG, sep=r'\s+', comment='#', names=header)
print(f"Original catalog size: {len(df)}")

# Filter RA/Dec
df_filtered = df[(df['RA'] >= RA_MIN) & (df['RA'] <= RA_MAX) & (df['Dec'] >= DEC_MIN) & (df['Dec'] <= DEC_MAX)]
print(f"Size after RA/Dec cut: {len(df_filtered)}")

# Read Mask
print(f"Reading mask from {MASK_FILE}")
hdul = fits.open(MASK_FILE)
mask_data = hdul[1].data['T'].flatten()
nside = hdul[1].header['NSIDE']
print(f"Mask NSIDE: {nside}")

# Calculate healpix indices for the catalog
phi = np.radians(df_filtered['RA'])
theta = np.radians(90.0 - df_filtered['Dec'])

pix = hp.ang2pix(nside, theta, phi, nest=False)

# Keep only those where mask_data is True
valid_mask = mask_data[pix]
df_final = df_filtered[valid_mask]

print(f"Size after mask cut: {len(df_final)}")

# Save to new file, keeping the same format
with open(OUT_CATALOG, 'w') as f:
    f.write("# " + " ".join(header) + "\n")
df_final.to_csv(OUT_CATALOG, sep=' ', index=False, header=False, mode='a')
print(f"Saved to {OUT_CATALOG}")
