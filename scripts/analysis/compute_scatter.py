# %% [Initialization]
import sys
from pathlib import Path

# Dynamically locate the project root using pyproject.toml as a marker
project_root = Path(__file__).resolve().parent
while (
    project_root != project_root.parent
    and not (project_root / "pyproject.toml").exists()
):
    project_root = project_root.parent

if not (project_root / "pyproject.toml").exists():
    raise RuntimeError(
        "Could not find project root (containing pyproject.toml) in any parent directory."
    )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pickle  # noqa: E402

from jianbing import scatter, visual  # noqa: E402

from initial import *  # noqa: F401, F403

# %% Local Functions


def load_simulation_model(root_path, sim_path):
    """Load the simulation model templates FITS file."""
    abs_sim_path = root_path / sim_path
    sim_cat = Table.read(abs_sim_path)
    print(f"Loaded simulation model templates from: {abs_sim_path}")
    return sim_cat


def load_observed_profiles(root_path, fits_files):
    """Load observed dsigma FITS files into a single table.

    Parameters
    ----------
    root_path : Path
        Project root path.
    fits_files : list[str]
        Relative paths for bin0..binN (richest bin first).

    Returns
    -------
    Table
        Observed profile table with bin_id, dsigma, errors and covariance,
        and ``meta['r_mpc']`` set to the radial bins.
    """
    obs = Table()
    bin_ids, ds_list, ds_err_list, jk_cov_list = [], [], [], []
    rp_mpc = None

    print(f"Loading {len(fits_files)} observed dsigma profile(s)...")
    for i, rel_path in enumerate(fits_files):
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

    print(f"Observed profiles ({len(bin_ids)} bins) packaged successfully.")
    return obs


def fit_scatter(obs, sim_cat):
    """Run jianbing scatter fitting on the observed profiles."""
    print("Fitting scatter using jianbing.scatter...")
    return scatter.compare_model_dsigma(
        obs, sim_cat, model_err=False, poly=True, verbose=True
    )


def summarize_scatter_results(custom_sum, r_mpc):
    """Print a per-bin summary of scatter, error, chi2 and DoF."""
    print("\nScatter Fitting Results Summary (JK):")
    dof = len(r_mpc)
    for row in custom_sum:
        bin_id = row["bin_id"]
        sig_med = row["sig_med_jk"]
        sig_err = row["sig_err_jk"]
        min_chi2 = np.nanmin(row["chi2_jk"])
        print(
            f"  Bin {bin_id}: Scatter = {sig_med:.3f} +/- {sig_err:.3f}, "
            f"Min Chi2 = {min_chi2:.3f} (DoF = {dof})"
        )


def save_summary_pkl(custom_sum, output_pkl_path):
    """Pickle the summary table under the 'custom_sample' key."""
    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump({"custom_sample": custom_sum}, f)
    print(f"Final summary table saved to: {output_pkl_path}")


def visualize_summary(custom_sum):
    """Generate the standard jianbing summary visualization."""
    print("\nGenerating standard jianbing visualization...")
    return visual.sum_plot_topn(
        custom_sum, label="Custom Sample", cov_type="jk", show_bin=True
    )


# %% Global Configuration

# For all available labels, refer to `RUN_REGISTRY` in `src/hsc_wl/config.py`.
LABEL = "cca2_4bin"  # Supports 3bin or 4bin configurations
VERSION = "Y3"  # "Y1" or "Y3"

# Parse catalog_id and nbins from the unified run label
catalog_id, nbins = LABEL.rsplit("_", 1)
if nbins not in ("3bin", "4bin"):
    raise ValueError(
        f"Scatter fitting requires a 3bin or 4bin configuration, got {nbins}"
    )

# Path to simulation data (relative to project_root)
SIM_PATH = "libs/jianbing/data/simulation/sim_mdpl2_cen_dsig.fits"

# Observed dsigma FITS files in order (bin_id=1 corresponds to bin0 / richest bin).
# Automatically detects available bin FITS files (e.g. 3 bins for redm_r16, 4 bins for standard runs).
_dsigma_dir = project_root / f"output/{catalog_id}/{nbins}/{VERSION}/dsigma"
FITS_FILES = sorted(
    [str(p.relative_to(project_root)) for p in _dsigma_dir.glob("hsc_hsc_bin*.fits")],
    key=lambda s: int(Path(s).stem.replace("hsc_hsc_bin", "")),
)
if not FITS_FILES:
    raise FileNotFoundError(f"No dsigma FITS files found in {_dsigma_dir}")
if len(FITS_FILES) > 4:
    raise ValueError(
        f"Simulation model templates only support up to 4 bins, but found {len(FITS_FILES)}."
    )

OUTPUT_PKL = (
    project_root
    / f"output/{catalog_id}/{nbins}/{VERSION}/pkl/{catalog_id}_{nbins}_{VERSION}_sum.pkl"
)

plt.rcParams["mathtext.fontset"] = "stix"


# %% [Stage 1: Load simulation model]
sim_cat = load_simulation_model(project_root, SIM_PATH)


# %% [Stage 2: Load observed profiles]
obs = load_observed_profiles(project_root, FITS_FILES)


# %% [Stage 3: Fit scatter]
custom_sum = fit_scatter(obs, sim_cat)
summarize_scatter_results(custom_sum, obs.meta["r_mpc"])


# %% [Stage 4: Save results]
save_summary_pkl(custom_sum, OUTPUT_PKL)


# %% [Stage 5: Visualize]
_ = visualize_summary(custom_sum)
