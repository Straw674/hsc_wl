# %%
import logging
import sys
from pathlib import Path

import h5py

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None

while True:
    if not current_dir or current_dir == current_dir.parent:
        logger.error("Error: pyproject.toml not found in parent directories.")
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        logger.info(f"Project root found: {root_path}")
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
hdf5_path = root_path / "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium.hdf5"
print(f"Loading HDF5 file: {hdf5_path}")

# If the HDF5 file contains multiple datasets (e.g., 'calib', 'random', 'source'),
# a 'path' argument must be specified; otherwise, astropy reads the first one by default.


with h5py.File(hdf5_path, "r") as f:
    keys = list(f.keys())
    print(f"HDF5 file contains the following datasets (Keys): {keys}")

    for key in keys:
        print(f"\n--- Analyzing dataset: {key} ---")
        # Read a specific dataset
        data_subset = Table.read(hdf5_path, path=key)
        print(f"Table length: {len(data_subset)}")
        data_subset.info()
        print("\nData preview (first 3 rows):")
        print(data_subset[:3])


# %%
# Export HDF5 datasets to FITS files
output_dir = root_path / "data" / "s16a_weak_lensing_hdf"
output_dir.mkdir(parents=True, exist_ok=True)

try:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        for key in keys:
            print(f"Exporting dataset '{key}' to FITS...")
            data_to_save = Table.read(hdf5_path, path=key)
            fits_filename = output_dir / f"s16a_weak_lensing_medium_{key}.fits"
            data_to_save.write(fits_filename, overwrite=True)
            print(f"Saved to: {fits_filename}")

except ImportError:
    print("h5py is not installed. Cannot export datasets.")
except Exception as e:
    print(f"An error occurred during export: {e}")
