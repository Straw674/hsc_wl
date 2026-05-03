# %%
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        logger.error("Error: pyproject.toml not found in parent directories.")
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        logger.info(f"Project root found: {root_path}")  # Confirm the path found
        break
    else:
        current_dir = current_dir.parent

if root_path:
    root_path_str = str(root_path)

    if root_path_str not in sys.path:
        sys.path.append(root_path_str)

    try:
        from initial import *
    except ModuleNotFoundError as e:
        logger.error(f"Error importing 'initial': {e}")

else:
    print("Could not proceed without finding the project root.")


# %%
# List available pkl files in the jianbing results directory
results_dir = root_path / "libs/jianbing/data/results"
pkl_files = list(results_dir.glob("*.pkl"))
print(f"Available pkl files in {results_dir.relative_to(root_path)}:")
for i, f in enumerate(pkl_files):
    print(f"[{i}] {f.name}")


# %%
PKL_RELATIVE_PATH = root_path / "libs/jianbing/data/results/topn_galaxies.pkl"

with PKL_RELATIVE_PATH.open("rb") as f:
    data = pickle.load(f)
    print(f"File loaded: {PKL_RELATIVE_PATH}")
    for key, value in data.items():
        print(f"  {key}: {len(value)} rows.")


# %%
TABLE_KEY = "logm_50_100"

if TABLE_KEY in data:
    table = data[TABLE_KEY]
    print(f"--- Table: {TABLE_KEY} ---")
    print(f"Number of rows: {len(table)}")
    print(f"Columns: {table.colnames}")
else:
    print(f"Error: Key '{TABLE_KEY}' not found in the loaded data.")


# %%
def export_table_to_fits(
    pkl_path: Path,
    table_key: str,
    output_dir: Path,
) -> list[Path]:
    with pkl_path.open("rb") as handle:
        sum_data = pickle.load(handle)

    if table_key not in sum_data:
        raise KeyError(f"{table_key} not found in {pkl_path.name}")

    table = sum_data[table_key]
    r_mpc = np.asarray(table.meta.get("r_mpc", []), dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []

    for row_idx, row in enumerate(table):
        ds = np.asarray(row["dsigma"], dtype=float)
        ds_err = np.asarray(row["dsig_err_jk"], dtype=float)
        cov = np.asarray(row["dsig_cov_jk"], dtype=float)

        n_radial = min(len(r_mpc), len(ds), len(ds_err))

        # Create the profile table for the first extension
        profile_table = Table()
        profile_table["rp"] = r_mpc[:n_radial]
        profile_table["ds"] = ds[:n_radial]
        profile_table["ds_err"] = ds_err[:n_radial]

        lens_idx = row_idx
        out_path = output_dir / f"hsc_hsc_lens{lens_idx}_lens.fits"

        # Create the HDU list with the profile and the full covariance matrix
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(profile_table, name="PROFILE"),
                fits.ImageHDU(cov[:n_radial, :n_radial], name="JK_COV"),
            ]
        )
        hdul.writeto(out_path, overwrite=True)

        output_files.append(out_path)
        print(f"Wrote {len(profile_table)} bins -> {out_path}")

    print(f"\nInput file: {pkl_path}")
    print(f"Target key: {table_key}")
    print(f"Total files written: {len(output_files)}")
    return output_files


# %%
OUTPUT_LABEL = "huang2022_logm_50_100"
output_path = root_path / "output" / OUTPUT_LABEL / "dsigma"

# Uncomment the following line to run the export
fits_paths = export_table_to_fits(PKL_RELATIVE_PATH, TABLE_KEY, output_path)
