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
# PKL_RELATIVE_PATH = root_path / "libs/jianbing/data/results/topn_galaxies_sum.pkl"
PKL_RELATIVE_PATH = root_path / "libs/jianbing/data/results/topn_clusters.pkl"
TABLE_KEY = "redm_hsc_lambda"
OUTPUT_LABEL = "huang2022_redm_hsc"


def export_table_to_csvs(
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

        n_radial = min(len(r_mpc), len(ds), len(ds_err))
        out_df = pd.DataFrame(
            {
                "rp": r_mpc[:n_radial],
                "ds": ds[:n_radial],
                "ds_err": ds_err[:n_radial],
            }
        )

        lens_idx = row_idx
        out_path = output_dir / f"hsc_hsc_lens{lens_idx}_lens.csv"
        out_df.to_csv(out_path, index=False)
        output_files.append(out_path)
        print(f"Wrote {len(out_df)} rows -> {out_path}")

    print(f"Input file: {pkl_path}")
    print(f"Target key: {table_key}")
    print(f"Total files written: {len(output_files)}")
    return output_files


# %%

with open(PKL_RELATIVE_PATH, "rb") as f:
    data = pickle.load(f)
    print(f"Keys in the loaded data: {list(data.keys())}")


# %%
pkl_path = root_path / PKL_RELATIVE_PATH
output_path = root_path / "output" / OUTPUT_LABEL / "dsigma"
csv_paths = export_table_to_csvs(pkl_path, TABLE_KEY, output_path)
