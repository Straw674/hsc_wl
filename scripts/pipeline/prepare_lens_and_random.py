# %% [Initialization]

import sys
from pathlib import Path

# Dynamically locate the project root using pyproject.toml as a marker
root_path = Path(__file__).resolve().parent
while root_path != root_path.parent and not (root_path / "pyproject.toml").exists():
    root_path = root_path.parent

if not (root_path / "pyproject.toml").exists():
    raise RuntimeError(
        "Could not find project root (containing pyproject.toml) in any parent directory."
    )

if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from hsc_wl.config import CATALOG_SOURCES
from hsc_wl.prepare import run_prepare_pipeline
from initial import *

# %%
# ---------- Local Functions ----------


def run_pipeline(source_name):
    """Run the prepare pipeline for a single catalog source."""
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


# %% [Global Configuration]

# Catalog source key (must exist in CATALOG_SOURCES).
SOURCE = "pdr3_redm_hsc"

# Random-sample multiplier w.r.t. the number of lenses.
RANDOM_MULTIPLIER = 20

# Binning mode:
# - "edges": use COL_RANK_EDGES_* as left-closed right-open intervals.
# - "top_counts": sort by col_rank and split sequentially by TOP_COUNTS.
# - "top_n": sort by col_rank and keep top TOP_N as a single bin.
BINNING_MODE = "top_counts"

# EDGES mode parameters
COL_RANK_EDGES_RICHNESS = [6.0, 10.0, 20.0, 35.0, 120.0]
COL_RANK_EDGES_MASS = [10.63, 10.8, 11.0, 11.2, 11.6]

# TOP_COUNTS mode parameters
# Pick top counts sequentially; chosen to match the topN paper.
TOP_COUNTS = [53, 196, 660, 1159]

# "desc": larger col_rank is better (top first); "asc": smaller is better.
TOP_SELECTION_ORDER = "desc"

# TOP_N mode parameters
TOP_N = 800

# Reproducibility: integer seed or None for random.
RNG_SEED = None

# Whether to draw alignment plots for each bin.
MAKE_PLOTS = True


# %% [Stage 1: Prepare lens and random catalogs]

run_pipeline(SOURCE)
