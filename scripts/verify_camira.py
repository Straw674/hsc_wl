import sys
from pathlib import Path

# Setup root path
current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None
while True:
    if not current_dir or current_dir == current_dir.parent:
        break
    if (current_dir / marker).exists():
        root_path = current_dir
        break
    current_dir = current_dir.parent

if root_path and str(root_path) not in sys.path:
    sys.path.append(str(root_path))
    from initial import *

from hsc_wl.prepare import run_prepare_pipeline
from hsc_wl.wl_compute import run_wl_analysis
from hsc_wl.config import CATALOG_SOURCES, RUN_PROFILES

def run():
    print("--- Running prepare for camira (top_n) ---")
    run_prepare_pipeline(
        source_name="camira",
        catalog_sources=CATALOG_SOURCES,
        binning_mode="top_n",
        top_counts=[53, 196, 660, 1159],
        top_n=800,
        col_rank_edges_mass=[10.63, 10.8, 11.0, 11.2, 11.6],
        col_rank_edges_richness=[6.0, 10.0, 20.0, 35.0, 120.0],
        top_selection_order="desc",
        random_multiplier=20,
        rng_seed=None,
        make_plots=False,
        root_path=root_path,
    )

    print("--- Running wl_compute for camira ---")
    run_wl_analysis(
        run_label="camira",
        run_profiles=RUN_PROFILES,
        source_version="Y3",
        njobs=12,
        comoving=False,
        lens_source_cut=0.1,
        n_jackknife=100,
        lens_survey="hsc",
        lens_rpmin=0.10,
        lens_rpmax=20.0,
        lens_n_rpbins=11,
        lens_linlog="log",
        lens_z_col="z",
        lens_ra_col="ra",
        lens_dec_col="dec",
        source_file="/Users/xinq/dev/repos/hsc_wl/data/hsc_y3.fits",
        source_nz_file="/Users/xinq/dev/repos/hsc_wl/data/nz.fits",
        source_calib_file=None,
        source_survey="hsc",
        corrections={
            "photo_z_dilution_correction": False,
            "boost_correction": False,
            "scalar_shear_response_correction": True,
            "matrix_shear_response_correction": False,
            "shear_responsivity_correction": True,
            "random_subtraction": True,
            "selection_bias_correction": True,
        },
        root_path=root_path,
    )

if __name__ == "__main__":
    run()
