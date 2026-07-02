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

from hsc_wl.config import CATALOG_SOURCES, RUN_PROFILES
from hsc_wl.prepare import run_prepare_pipeline
from hsc_wl.wl_compute import run_wl_analysis
from initial import *

# %%
# ---------- Local Functions ----------


def prepare_cosine(source_name, binning_mode):
    """Prepare lens & random catalogs for a cosine run."""
    run_prepare_pipeline(
        source_name=source_name,
        catalog_sources=CATALOG_SOURCES,
        binning_mode=binning_mode,
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


def compute_cosine(run_label):
    """Compute the lensing profile for a cosine run."""
    run_wl_analysis(
        run_label=run_label,
        run_profiles=RUN_PROFILES,
        source_version=SOURCE_VERSION,
        njobs=NJOBS,
        comoving=COMOVING,
        lens_source_cut=LENS_SOURCE_CUT,
        n_jackknife=NJACKKNIFE,
        lens_survey=LENS_SURVEY,
        lens_rpmin=LENS_RPMIN,
        lens_rpmax=LENS_RPMAX,
        lens_n_rpbins=LENS_N_RPBINS,
        lens_linlog=LENS_LINLOG,
        lens_z_col=LENS_Z_COL,
        lens_ra_col=LENS_RA_COL,
        lens_dec_col=LENS_DEC_COL,
        source_file=None,
        source_nz_file=None,
        source_calib_file=None,
        source_survey=SOURCE_SURVEY,
        corrections=None,
        root_path=root_path,
    )


# %% [Global Configuration]

# Source catalog version: "Y3" (PDR3/S19A) or "Y1" (S16A/Y1)
SOURCE_VERSION = "Y3"
SOURCE_SURVEY = "hsc"

# Misc
NJOBS = 12
COMOVING = False
LENS_SOURCE_CUT = 0.1
NJACKKNIFE = 100

# Lens
LENS_SURVEY = "hsc"
LENS_RPMIN = 0.10
LENS_RPMAX = 20.0
LENS_N_RPBINS = 11
LENS_LINLOG = "log"
LENS_Z_COL = "z"
LENS_RA_COL = "ra"
LENS_DEC_COL = "dec"

# Prepare (shared by both cosine runs)
TOP_COUNTS = [53, 196, 660, 1159]
TOP_N = 800
COL_RANK_EDGES_MASS = [10.63, 10.8, 11.0, 11.2, 11.6]
COL_RANK_EDGES_RICHNESS = [6.0, 10.0, 20.0, 35.0, 120.0]
TOP_SELECTION_ORDER = "desc"
RANDOM_MULTIPLIER = 20
RNG_SEED = None
# Plots disabled in automated runs to avoid blocking / unused figures.
MAKE_PLOTS = False


# %% [Stage 1: Prepare lens & random for "cosine" (top_n, single bin)]

SOURCE_NAME = "cosine"
BINNING_MODE = "top_n"

prepare_cosine(SOURCE_NAME, BINNING_MODE)


# %% [Stage 2: Prepare lens & random for "cosine_4bin" (top_counts, 4 bins)]

SOURCE_NAME = "cosine_4bin"
BINNING_MODE = "top_counts"

prepare_cosine(SOURCE_NAME, BINNING_MODE)


# %% [Stage 3: Compute lensing profile for "cosine"]

RUN_LABEL = "cosine"

compute_cosine(RUN_LABEL)


# %% [Stage 4: Compute lensing profile for "cosine_4bin"]

RUN_LABEL = "cosine_4bin"

compute_cosine(RUN_LABEL)
