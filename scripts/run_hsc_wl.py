# %%
import logging
import sys
from pathlib import Path

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
    logger.error("Could not proceed without finding the project root.")

# %%
from dsigma.helpers import dsigma_table
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density
from dsigma.surveys import hsc as hsc_survey
from hsc_wl.wl_compute import (
    assign_jackknife_fields_with_fallback,
    find_one,
    pick_column,
    pick_required_column,
)
from hsc_wl.config import RUN_PROFILES


# ---------- Runtime Settings ----------
# Switch this label before each run when using profile-based YAML config.
RUN_PROFILE_LABEL = "pdr3_redm_hsc"

# Source catalog version: "Y3" (PDR3/S19A) or "Y1" (S16A/Y1)
SOURCE_VERSION = "Y3"

# ---------- Misc ----------
NJOBS = 12
COMOVING = False
LENS_SOURCE_CUT = 0.1
VERBOSE = True
NJACKKNIFE = 100

# ---------- Lens ----------
LENS_SURVEY = "hsc"

LENS_RPMIN = 0.10
LENS_RPMAX = 20.0
LENS_N_RPBINS = 11
LENS_LINLOG = "log"

# Column names in lens catalog
LENS_Z_COL = "z"
LENS_RA_COL = "ra"
LENS_DEC_COL = "dec"

# ---------- Source Configuration ----------
if SOURCE_VERSION == "Y3":
    TOMOGRAPHY = True
    PHOTO_Z_DILUTION_CORRECTION = False
    SOURCE_FILE = "/Users/xinq/dev/repos/hsc_wl/data/hsc_y3.fits"
    SOURCE_NZ_FILE = "/Users/xinq/dev/repos/hsc_wl/data/nz.fits"
    SOURCE_CALIB_FILE = None
    SOURCE_SURVEY = "hsc"
elif SOURCE_VERSION == "Y1":
    TOMOGRAPHY = False
    PHOTO_Z_DILUTION_CORRECTION = True
    # Path to the S16A medium source catalog
    SOURCE_FILE = "/Users/xinq/dev/repos/hsc_wl/data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_source.fits"
    SOURCE_NZ_FILE = None
    SOURCE_CALIB_FILE = "/Users/xinq/dev/repos/hsc_wl/data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_calib.fits"
    SOURCE_SURVEY = "hsc"
    SOURCE_Z_COL = "z"
else:
    raise ValueError(f"Unsupported SOURCE_VERSION: {SOURCE_VERSION}")


# ---------- Paths ----------
# RUN_PROFILES is now imported from src.config

# ---------- Corrections ----------
SCALAR_SHEAR_RESPONSE_CORRECTION = True
MATRIX_SHEAR_RESPONSE_CORRECTION = False
SHEAR_RESPONSIVITY_CORRECTION = True
SELECTION_BIAS_CORRECTION = True
RANDOM_SUBTRACTION = True
BOOST_CORRECTION = False


# %%


from hsc_wl.wl_compute import run_wl_analysis


# ---------- core ----------
def run_analysis(run_label=RUN_PROFILE_LABEL):
    """Run dsigma analysis by delegating to hsc_wl.wl_compute.run_wl_analysis."""
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
        source_file=SOURCE_FILE,
        source_nz_file=SOURCE_NZ_FILE,
        source_calib_file=SOURCE_CALIB_FILE,
        source_survey=SOURCE_SURVEY,
        corrections={
            "photo_z_dilution_correction": PHOTO_Z_DILUTION_CORRECTION,
            "boost_correction": BOOST_CORRECTION,
            "scalar_shear_response_correction": SCALAR_SHEAR_RESPONSE_CORRECTION,
            "matrix_shear_response_correction": MATRIX_SHEAR_RESPONSE_CORRECTION,
            "shear_responsivity_correction": SHEAR_RESPONSIVITY_CORRECTION,
            "random_subtraction": RANDOM_SUBTRACTION,
            "selection_bias_correction": SELECTION_BIAS_CORRECTION,
        },
        root_path=root_path,
    )


# %%
# ---------- Execution ----------
if __name__ == "__main__":
    run_analysis(run_label=RUN_PROFILE_LABEL)
