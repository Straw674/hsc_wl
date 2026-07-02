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

from hsc_wl.config import RUN_PROFILES
from hsc_wl.wl_compute import run_wl_analysis
from initial import *

# %%
# ---------- Local Functions ----------


def resolve_source_settings(version):
    """Return version-dependent source catalog settings.

    Parameters
    ----------
    version : str
        Source catalog version: "Y3" (PDR3/S19A) or "Y1" (S16A).

    Returns
    -------
    dict
        Mapping with keys: source_file, source_nz_file, source_calib_file,
        source_survey, photo_z_dilution_correction.
    """
    if version == "Y3":
        return {
            "source_file": str(root_path / "data/hsc_y3.fits"),
            "source_nz_file": str(root_path / "data/nz.fits"),
            "source_calib_file": None,
            "source_survey": "hsc",
            "photo_z_dilution_correction": False,
        }
    if version == "Y1":
        return {
            "source_file": str(
                root_path
                / "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_source.fits"
            ),
            "source_nz_file": None,
            "source_calib_file": str(
                root_path
                / "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_calib.fits"
            ),
            "source_survey": "hsc",
            "photo_z_dilution_correction": True,
        }
    raise ValueError(f"Unsupported SOURCE_VERSION: {version}")


def build_corrections(source_settings):
    """Assemble the dsigma corrections dict from global toggles."""
    return {
        "photo_z_dilution_correction": source_settings["photo_z_dilution_correction"],
        "boost_correction": BOOST_CORRECTION,
        "scalar_shear_response_correction": SCALAR_SHEAR_RESPONSE_CORRECTION,
        "matrix_shear_response_correction": MATRIX_SHEAR_RESPONSE_CORRECTION,
        "shear_responsivity_correction": SHEAR_RESPONSIVITY_CORRECTION,
        "random_subtraction": RANDOM_SUBTRACTION,
        "selection_bias_correction": SELECTION_BIAS_CORRECTION,
    }


def run_analysis(run_label):
    """Run the dsigma weak-lensing analysis for the given run label."""
    source_settings = resolve_source_settings(SOURCE_VERSION)
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
        source_file=source_settings["source_file"],
        source_nz_file=source_settings["source_nz_file"],
        source_calib_file=source_settings["source_calib_file"],
        source_survey=source_settings["source_survey"],
        corrections=build_corrections(source_settings),
        root_path=root_path,
    )


# %% [Global Configuration]

# Switch this label before each run when using profile-based YAML config.
RUN_PROFILE_LABEL = "pdr3_redm_hsc"

# Source catalog version: "Y3" (PDR3/S19A) or "Y1" (S16A/Y1)
SOURCE_VERSION = "Y3"

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

# Column names in lens catalog
LENS_Z_COL = "z"
LENS_RA_COL = "ra"
LENS_DEC_COL = "dec"

# Corrections toggles (photo_z_dilution_correction is set per SOURCE_VERSION)
SCALAR_SHEAR_RESPONSE_CORRECTION = True
MATRIX_SHEAR_RESPONSE_CORRECTION = False
SHEAR_RESPONSIVITY_CORRECTION = True
SELECTION_BIAS_CORRECTION = True
RANDOM_SUBTRACTION = True
BOOST_CORRECTION = False


# %% [Stage 1: Run dsigma weak-lensing analysis]

run_analysis(RUN_PROFILE_LABEL)
