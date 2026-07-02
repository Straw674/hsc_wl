# %%
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        print("Error: pyproject.toml not found in parent directories.")
        # Handle the error appropriately, maybe raise an exception or exit
        # For now, just break to avoid infinite loop if marker is truly missing
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

    try:
        from initial import *
    except ModuleNotFoundError as e:
        print(f"Error importing 'initial': {e}")

    from hsc_wl.prepare import (
        build_bin_slices,
        get_binning_settings,
        read_dat_to_pandas,
        resolve_path,
        summarize_bin_boundaries,
        prepare_lens_random_tables,
        get_latest_cluster_catalog,
        run_prepare_pipeline,
    )
    from hsc_wl.config import CATALOG_SOURCES


else:
    print("Could not proceed without finding the project root.")

# %%
# =========================
# User-editable parameters
# =========================

SOURCE = "pdr3_redm_hsc"  # Choose from CATALOG_SOURCES keys

RANDOM_MULTIPLIER = 20

# Path can be absolute or relative to the project root.
# CATALOG_SOURCES is now imported from src.config

# Binning mode:
# - "edges": use COL_RANK_EDGES as left-closed right-open intervals.
# - "top_counts": sort by col_rank and split sequentially by TOP_COUNTS.
# - "top_n": sort by col_rank and select top TOP_N objects as a single bin.
BINNING_MODE = "top_counts"  # "edges", "top_counts", or "top_n"

# -----------------------------------------------------------------------
# EDGES

# Used only when BINNING_MODE == "edges".
COL_RANK_EDGES_RICHNESS = [6.0, 10.0, 20.0, 35.0, 120.0]
COL_RANK_EDGES_MASS = [10.63, 10.8, 11.0, 11.2, 11.6]

# -----------------------------------------------------------------------
# TOP_COUNTS

# Used only when BINNING_MODE == "top_counts".
# Example: [x1, x2, x3, x4] means pick top x1 first, then top x2 from remaining, etc.
# TOP_COUNTS = [50, 197, 662, 1165]

# exactly same as the number in topn paper
TOP_COUNTS = [53, 196, 660, 1159]


# "desc": larger col_rank is better (top first); "asc": smaller col_rank is better.
TOP_SELECTION_ORDER = "desc"

# -----------------------------------------------------------------------
# TOP_N

# Used only when BINNING_MODE == "top_n".
TOP_N = 800

# -----------------------------------------------------------------------

# Set to an integer for reproducibility (e.g., 42), or None for random seed.
RNG_SEED = None

# Whether to draw alignment plots for each bin.
MAKE_PLOTS = True


# %%
def run_pipeline(source_name):
    run_prepare_pipeline(
        source_name=source_name,
        catalog_sources=CATALOG_SOURCES,
        binning_mode=BINNING_MODE,
        top_counts=TOP_COUNTS,
        top_n=TOP_N,
        col_rank_edges_mass=COL_RANK_EDGES_MASS,
        col_rank_edges_richness=COL_RANK_EDGES_RICHNESS,
        top_selection_order=TOP_SELECTION_ORDER,
        random_multiplier=RANDOM_MULTIPLIER,
        rng_seed=RNG_SEED,
        make_plots=MAKE_PLOTS,
        root_path=root_path,
    )


if __name__ == "__main__":
    run_pipeline(SOURCE)
