# %%
import sys
from pathlib import Path

# Locate project root and initialize
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
    try:
        from initial import *
    except ModuleNotFoundError as e:
        print(f"Error importing 'initial': {e}")

from hsc_wl.prepare import run_prepare_pipeline
from hsc_wl.wl_compute import run_wl_analysis
from hsc_wl.config import CATALOG_SOURCES, RUN_PROFILES

# %%
# ---------- Configuration ----------
# We define configs here, but mostly import them from the main script definitions.


# %%
# ---------- Execution ----------
def main():
    print("=========================================================")
    print("Starting Automated Weak Lensing Analysis for Cosine Catalogs")
    print("=========================================================\n")

    # 1. Run prepare for cosine (top_n mode -> 1 bin)
    print("--- Step 1: Preparing Lens & Random for 'cosine' (top_n) ---")
    run_prepare_pipeline(
        source_name="cosine",
        catalog_sources=CATALOG_SOURCES,
        binning_mode="top_n",
        top_counts=[53, 196, 660, 1159],  # Default top counts
        top_n=800,  # 1bin top_n size
        col_rank_edges_mass=[10.63, 10.8, 11.0, 11.2, 11.6],
        col_rank_edges_richness=[6.0, 10.0, 20.0, 35.0, 120.0],
        top_selection_order="desc",
        random_multiplier=20,
        rng_seed=None,
        make_plots=False,  # Set to False to avoid blocking or creating unused figures in auto run
        root_path=root_path,
    )

    # 2. Run prepare for cosine_4bin (top_counts mode -> 4 bins)
    print("\n--- Step 2: Preparing Lens & Random for 'cosine_4bin' (top_counts) ---")
    run_prepare_pipeline(
        source_name="cosine_4bin",
        catalog_sources=CATALOG_SOURCES,
        binning_mode="top_counts",
        top_counts=[53, 196, 660, 1159],  # 4bin top counts
        top_n=800,
        col_rank_edges_mass=[10.63, 10.8, 11.0, 11.2, 11.6],
        col_rank_edges_richness=[6.0, 10.0, 20.0, 35.0, 120.0],
        top_selection_order="desc",
        random_multiplier=20,
        rng_seed=None,
        make_plots=False,  # Set to False to avoid blocking or creating unused figures in auto run
        root_path=root_path,
    )

    # 3. Run hsc_wl for cosine
    print("\n--- Step 3: Computing Lensing Profile for 'cosine' ---")
    run_wl_analysis(
        run_label="cosine",
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
        source_file=None,
        source_nz_file=None,
        source_calib_file=None,
        source_survey="hsc",
        corrections=None,
        root_path=root_path,
    )

    # 4. Run hsc_wl for cosine_4bin
    print("\n--- Step 4: Computing Lensing Profile for 'cosine_4bin' ---")
    run_wl_analysis(
        run_label="cosine_4bin",
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
        source_file=None,
        source_nz_file=None,
        source_calib_file=None,
        source_survey="hsc",
        corrections=None,
        root_path=root_path,
    )

    print("\n=========================================================")
    print("Pipeline completed successfully!")
    print("=========================================================")


# %%
if __name__ == "__main__":
    main()
