#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fit_custom_scatter.py

Calculate best-fit scatter and output summary results directly from
pre-computed lens FITS files (with PROFILE and JK_COV extensions)
and pre-generated simulation models.
"""

import sys
from pathlib import Path
import pickle
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

# ---------- project root setup ----------
current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None

while True:
    if not current_dir or current_dir == current_dir.parent:
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
    from initial import *
else:
    print("Error: Could not find project root.")
    sys.exit(1)

# ---------- runtime settings ----------
LABEL = "s16a"

# Path to simulation data (relative to root_path)
# SIM_PATH = "libs/jianbing/data/simulation/sim_merge_all_dsig.fits"
SIM_PATH = "libs/jianbing/data/simulation/sim_mdpl2_cen_dsig.fits"

# List your 4 FITS files in order (relative to root_path).
# bin_id=1 corresponds to the 1st file (richest/most massive).
FITS_FILES = [
    f"output/{LABEL}/dsigma/hsc_hsc_lens0_lens.fits",
    f"output/{LABEL}/dsigma/hsc_hsc_lens1_lens.fits",
    f"output/{LABEL}/dsigma/hsc_hsc_lens2_lens.fits",
    f"output/{LABEL}/dsigma/hsc_hsc_lens3_lens.fits",
]

OUTPUT_PKL = "output/custom_samples_sum.pkl"
VIS_PLOT_DIR = "output/plots"


# ---------- core ----------
def main():
    from jianbing import scatter

    abs_sim_path = root_path / SIM_PATH
    sim_cat = Table.read(abs_sim_path)
    print(f"Loaded simulation model templates from: {abs_sim_path}")

    obs = Table()
    bin_ids, ds_list, ds_err_list, jk_cov_list = [], [], [], []
    rp_mpc = None

    for i, rel_path in enumerate(FITS_FILES):
        bin_id = i + 1
        abs_path = root_path / rel_path
        with fits.open(abs_path) as hdul:
            prof_data = hdul[1].data
            rp, ds, ds_err = prof_data["rp"], prof_data["ds"], prof_data["ds_err"]
            cov_data = hdul[2].data

            if rp_mpc is None:
                rp_mpc = rp
            else:
                assert np.allclose(rp_mpc, rp), (
                    f"Error: rp bins in {abs_path} mismatch."
                )

            bin_ids.append(bin_id)
            ds_list.append(ds)
            ds_err_list.append(ds_err)
            jk_cov_list.append(cov_data)

    obs["bin_id"] = bin_ids
    obs["dsigma"] = ds_list
    obs["dsig_err_jk"] = ds_err_list
    obs["dsig_err_bt"] = ds_err_list
    obs["dsig_cov_jk"] = jk_cov_list
    obs["dsig_cov_bt"] = jk_cov_list
    obs.meta["r_mpc"] = rp_mpc

    print("Observed profiles packaged successfully. Fitting scatter...")
    custom_sum = scatter.compare_model_dsigma(
        obs, sim_cat, model_err=False, poly=True, verbose=True
    )

    abs_out_pkl = root_path / OUTPUT_PKL
    with open(abs_out_pkl, "wb") as f:
        pickle.dump({"custom_sample": custom_sum}, f)
    print(f"Final summary table saved to: {abs_out_pkl}")

    try:
        from jianbing import visual

        print("\nGenerating standard jianbing visualization...")
        fig_visual = visual.sum_plot_topn(
            custom_sum, label="Custom Sample", cov_type="jk", show_bin=True
        )
        abs_vis_dir = root_path / VIS_PLOT_DIR
        abs_vis_dir.mkdir(parents=True, exist_ok=True)
        vis_plot_path = abs_vis_dir / f"{LABEL}_scatter.png"
        fig_visual.savefig(vis_plot_path, bbox_inches="tight")
        print(f"Jianbing Standard QA plot saved to: {vis_plot_path}")
    except Exception as e:
        print(f"\nSkipping standard visualization due to error: {e}")


if __name__ == "__main__":
    main()
