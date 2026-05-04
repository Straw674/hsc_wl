"""
fit_custom_scatter.py

Calculate best-fit scatter and output summary results directly from
pre-computed lens FITS files (with PROFILE and JK_COV extensions)
and pre-generated simulation models.
"""

# %%
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from jianbing import scatter, visual

plt.rcParams["mathtext.fontset"] = "stix"

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

# %% ---------- runtime settings ----------
LABEL = "s16a"
VERSION = "Y3"  # "Y1" or "Y3"

# Path to simulation data (relative to root_path)
# SIM_PATH = "libs/jianbing/data/simulation/sim_merge_all_dsig.fits"
SIM_PATH = "libs/jianbing/data/simulation/sim_mdpl2_cen_dsig.fits"

# List your 4 FITS files in order (relative to root_path).
# bin_id=1 corresponds to the 1st file (richest/most massive).
FITS_FILES = [
    f"output/{LABEL}/{VERSION}/dsigma/hsc_hsc_lens0.fits",
    f"output/{LABEL}/{VERSION}/dsigma/hsc_hsc_lens1.fits",
    f"output/{LABEL}/{VERSION}/dsigma/hsc_hsc_lens2.fits",
    f"output/{LABEL}/{VERSION}/dsigma/hsc_hsc_lens3.fits",
]

OUTPUT_PKL = f"output/{LABEL}/{VERSION}/pkl/{LABEL}_{VERSION}_sum.pkl"

# %% ---------- load simulation model ----------

abs_sim_path = root_path / SIM_PATH
sim_cat = Table.read(abs_sim_path)
print(f"Loaded simulation model templates from: {abs_sim_path}")

# %% ---------- load observed profiles ----------
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
            assert np.allclose(rp_mpc, rp), f"Error: rp bins in {abs_path} mismatch."

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

print("Observed profiles packaged successfully.")

# %% ---------- fitting scatter ----------
print("Fitting scatter using jianbing.scatter...")
custom_sum = scatter.compare_model_dsigma(
    obs, sim_cat, model_err=False, poly=True, verbose=True
)

# Output summary of scatter and chi2 for each bin
print("\nScatter Fitting Results Summary (JK):")
for row in custom_sum:
    bin_id = row["bin_id"]
    sig_med = row["sig_med_jk"]
    sig_err = row["sig_err_jk"]
    min_chi2 = np.nanmin(row["chi2_jk"])
    # Number of radial bins (degrees of freedom)
    dof = len(obs.meta["r_mpc"])
    print(
        f"  Bin {bin_id}: Scatter = {sig_med:.3f} +/- {sig_err:.3f}, "
        f"Min Chi2 = {min_chi2:.3f} (DoF = {dof})"
    )


# %% ---------- save results ----------
abs_out_pkl = root_path / OUTPUT_PKL
abs_out_pkl.parent.mkdir(parents=True, exist_ok=True)
with open(abs_out_pkl, "wb") as f:
    pickle.dump({"custom_sample": custom_sum}, f)
print(f"Final summary table saved to: {abs_out_pkl}")

# %% ---------- visualization ----------

print("\nGenerating standard jianbing visualization...")
fig_visual = visual.sum_plot_topn(
    custom_sum, label="Custom Sample", cov_type="jk", show_bin=True
)
