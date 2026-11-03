# %% [Initialization]
import sys
from pathlib import Path
from typing import Any

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

import scipy.optimize as optimize
from astropy.table import Table
from colossus.cosmology import cosmology

from hsc_wl.profile_scatter import (
    build_profile_scatter_model,
    compute_derived_quantities_and_errors,
    compute_stacked_profile_dsigma,
    fit_profile_scatter_map,
    load_dsigma_profile,
)
from initial import *

# %% Local Functions


def generate_mock_realizations(
    ds_true: np.ndarray,
    cov: np.ndarray,
    n_mocks: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic DeltaSigma profiles from true model and covariance.

    Parameters
    ----------
    ds_true : ndarray of shape (N_r,)
        True model DeltaSigma profile in Colossus units.
    cov : ndarray of shape (N_r, N_r)
        Measurement covariance matrix in Colossus units.
    n_mocks : int
        Number of synthetic realizations to generate.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_mocks, N_r)
        Synthetic noisy DeltaSigma realizations.
    """
    evals, evecs = np.linalg.eigh(cov)
    evals_clipped = np.maximum(evals, 1e-12 * np.max(evals))
    l_factor = evecs @ np.diag(np.sqrt(evals_clipped))

    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal((n_mocks, len(ds_true)))
    correlated_noise = white_noise @ l_factor.T
    return ds_true[None, :] + correlated_noise


def fit_scatter_fixed_nuisance(
    model_state: dict[str, Any],
    ds_obs_col: np.ndarray,
    cov_inv_col: np.ndarray,
    bin_index: int = 0,
    alpha_c_fixed: float = 1.0,
    alpha_b_fixed: float = 1.0,
    f_mis: float = 0.25,
    sigma_r: float = 100.0,
) -> dict[str, Any]:
    """Fit only scatter sigma while keeping concentration and bias fixed.

    Parameters
    ----------
    model_state : dict
        Precomputed background model state.
    ds_obs_col : ndarray
        Observed or mock DeltaSigma in Colossus units.
    cov_inv_col : ndarray
        Inverse covariance matrix in Colossus units.
    bin_index : int, default=0
        Bin index to evaluate.
    alpha_c_fixed, alpha_b_fixed : float, default=1.0
        Fixed concentration and bias scaling factors.
    f_mis, sigma_r : float
        Fixed miscentering parameters.

    Returns
    -------
    dict
        Fitted scatter, estimated error, chi2, and derived quantities.
    """

    def neg_log_lik(theta: np.ndarray) -> float:
        s = float(theta[0])
        try:
            mod = compute_stacked_profile_dsigma(
                scatter=s,
                alpha_c=alpha_c_fixed,
                alpha_b=alpha_b_fixed,
                f_mis=f_mis,
                sigma_R=sigma_r,
                bin_index=bin_index,
                model_state=model_state,
            )
            diff = ds_obs_col - mod
            return 0.5 * float(diff @ cov_inv_col @ diff)
        except Exception:
            return 1e30

    res = optimize.minimize(
        neg_log_lik,
        x0=[0.35],
        method="L-BFGS-B",
        bounds=[(0.05, 1.50)],
        options={"ftol": 1e-8, "maxiter": 100},
    )
    s_best = float(res.x[0])

    # 1D finite difference curvature for error
    eps = 1e-3
    s_step = max(s_best * eps, 1e-4)
    f0 = neg_log_lik(np.array([s_best]))
    fp = neg_log_lik(np.array([s_best + s_step]))
    fm = neg_log_lik(np.array([s_best - s_step]))
    d2 = (fp - 2.0 * f0 + fm) / (s_step**2)
    s_err = float(np.sqrt(1.0 / d2)) if d2 > 0 else np.nan

    # Derived mean logM
    cov_params_pseudo = np.array(
        [
            [s_err**2 if np.isfinite(s_err) else 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    derived, _ = compute_derived_quantities_and_errors(
        scatter=s_best,
        alpha_c=alpha_c_fixed,
        alpha_b=alpha_b_fixed,
        cov_params=cov_params_pseudo,
        bin_index=bin_index,
        model_state=model_state,
    )

    return {
        "scatter": s_best,
        "scatter_err": s_err,
        "alpha_c": alpha_c_fixed,
        "alpha_b": alpha_b_fixed,
        "chi2": 2.0 * f0,
        "success": bool(res.success),
        **derived,
    }


def execute_e2e_recovery_test(
    model_state: dict[str, Any],
    rp_mpc: np.ndarray,
    cov_col: np.ndarray,
    cov_inv_col: np.ndarray,
    factor: float,
    true_scatter: float,
    true_alpha_c: float,
    true_alpha_b: float,
    f_mis: float,
    sigma_r: float,
    bin_index: int,
    n_mocks: int,
    seed: int,
) -> dict[str, Any]:
    """Execute complete end-to-end recovery test suite across noiseless and noisy mocks."""
    # 1. Forward model true profile
    ds_col_true = compute_stacked_profile_dsigma(
        scatter=true_scatter,
        alpha_c=true_alpha_c,
        alpha_b=true_alpha_b,
        f_mis=f_mis,
        sigma_R=sigma_r,
        bin_index=bin_index,
        model_state=model_state,
    )
    ds_phys_true = ds_col_true / factor

    # 2. Noiseless validation
    fit_noiseless_fixed = fit_scatter_fixed_nuisance(
        model_state=model_state,
        ds_obs_col=ds_col_true,
        cov_inv_col=cov_inv_col,
        bin_index=bin_index,
        alpha_c_fixed=true_alpha_c,
        alpha_b_fixed=true_alpha_b,
        f_mis=f_mis,
        sigma_r=sigma_r,
    )
    fit_noiseless_free = fit_profile_scatter_map(
        model_state=model_state,
        ds_obs_colossus=ds_col_true,
        cov_inv_colossus=cov_inv_col,
        bin_index=bin_index,
        f_mis=f_mis,
        sigma_r=sigma_r,
    )

    # 3. Generate noisy mock realizations
    mock_profiles = generate_mock_realizations(
        ds_true=ds_col_true,
        cov=cov_col,
        n_mocks=n_mocks,
        seed=seed,
    )

    # 4. Fit all noisy realizations
    fixed_results = []
    free_results = []
    for i in range(n_mocks):
        res_fix = fit_scatter_fixed_nuisance(
            model_state=model_state,
            ds_obs_col=mock_profiles[i],
            cov_inv_col=cov_inv_col,
            bin_index=bin_index,
            alpha_c_fixed=true_alpha_c,
            alpha_b_fixed=true_alpha_b,
            f_mis=f_mis,
            sigma_r=sigma_r,
        )
        res_free = fit_profile_scatter_map(
            model_state=model_state,
            ds_obs_colossus=mock_profiles[i],
            cov_inv_colossus=cov_inv_col,
            bin_index=bin_index,
            f_mis=f_mis,
            sigma_r=sigma_r,
        )
        fixed_results.append(res_fix)
        free_results.append(res_free)

    return {
        "true_scatter": true_scatter,
        "true_alpha_c": true_alpha_c,
        "true_alpha_b": true_alpha_b,
        "ds_phys_true": ds_phys_true,
        "mock_phys_profiles": mock_profiles / factor,
        "fit_noiseless_fixed": fit_noiseless_fixed,
        "fit_noiseless_free": fit_noiseless_free,
        "fixed_results": fixed_results,
        "free_results": free_results,
    }


def plot_e2e_recovery_summary(
    recovery_data: dict[str, Any],
    rp_mpc: np.ndarray,
    ds_phys_err: np.ndarray,
    out_plot_path: Path,
) -> None:
    """Create three-panel diagnostic plot summarizing the e2e scatter recovery."""
    true_s = recovery_data["true_scatter"]
    ds_true = recovery_data["ds_phys_true"]
    mock_phys = recovery_data["mock_phys_profiles"]
    fixed_res = recovery_data["fixed_results"]
    free_res = recovery_data["free_results"]

    s_fixed = np.array([r["scatter"] for r in fixed_res])
    s_free = np.array([r["scatter"] for r in free_res])
    logm_fixed = np.array([r["mean_logm"] for r in fixed_res])
    logm_free = np.array([r["mean_logm"] for r in free_res])

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # Panel 1: Profiles
    ax = axes[0]
    ax.fill_between(
        rp_mpc,
        np.percentile(mock_phys, 16, axis=0),
        np.percentile(mock_phys, 84, axis=0),
        color="lightgray",
        alpha=0.6,
        label=r"Mock $1\sigma$ Envelopes",
    )
    ax.errorbar(
        rp_mpc,
        mock_phys[0],
        yerr=ds_phys_err,
        fmt="o",
        color="black",
        markersize=3.5,
        capsize=2.0,
        label="Example Realization",
    )
    ax.plot(
        rp_mpc,
        ds_true,
        "-",
        color="crimson",
        linewidth=2.0,
        label=rf"Truth ($\sigma={true_s:.2f}$)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R_p\ [\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\Delta\Sigma\ [\mathrm{M_\odot/pc^2}]$")
    ax.set_title("Weak Lensing Profiles", fontweight="normal")
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(fontsize=8.5)

    # Panel 2: Recovered Scatter Distributions
    ax = axes[1]
    bins_s = np.linspace(0.05, 0.70, 16)
    ax.hist(
        s_fixed,
        bins=bins_s,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color="forestgreen",
        label=rf"Fixed $\alpha$ ($\langle\hat{{\sigma}}\rangle={np.mean(s_fixed):.3f}\pm{np.std(s_fixed):.3f}$)",
    )
    ax.hist(
        s_free,
        bins=bins_s,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="royalblue",
        label=rf"Free $\alpha$ ($\langle\hat{{\sigma}}\rangle={np.mean(s_free):.3f}\pm{np.std(s_free):.3f}$)",
    )
    ax.axvline(
        true_s,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=rf"Truth ($\sigma={true_s:.2f}$)",
    )
    ax.set_xlabel(r"Recovered Scatter $\hat{\sigma}$")
    ax.set_ylabel("Posterior Density")
    ax.set_title("Scatter Recovery Distribution", fontweight="normal")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=8.0)

    # Panel 3: Scatter vs Mean Mass Degeneracy
    ax = axes[2]
    ax.scatter(
        s_fixed,
        logm_fixed,
        color="forestgreen",
        alpha=0.8,
        s=30,
        label="Fixed Concentration/Bias",
    )
    ax.scatter(
        s_free,
        logm_free,
        color="royalblue",
        alpha=0.8,
        s=30,
        marker="^",
        label="Free Concentration/Bias",
    )
    true_logm = recovery_data["fit_noiseless_fixed"]["mean_logm"]
    ax.axvline(true_s, color="crimson", linestyle="--", alpha=0.7)
    ax.axhline(true_logm, color="crimson", linestyle="--", alpha=0.7)
    ax.scatter(
        [true_s],
        [true_logm],
        color="red",
        marker="*",
        s=120,
        zorder=5,
        label="Truth",
    )
    ax.set_xlabel(r"Recovered Scatter $\hat{\sigma}$")
    ax.set_ylabel(r"$\langle \log_{10} M_{\rm 200m} \rangle$")
    ax.set_title(r"$\sigma$ vs. Effective Mean Mass", fontweight="normal")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=8.0)

    plt.tight_layout()
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved e2e recovery plot to: {out_plot_path}")
    plt.show()
    plt.close(fig)


# %% Global Configuration

TRUE_SCATTER = 0.35
TRUE_ALPHA_C = 1.0
TRUE_ALPHA_B = 1.0
FIXED_F_MIS = 0.25
FIXED_SIGMA_R = 100.0  # kpc/h
COSMOLOGY_NAME = "planck18"
MASS_DEF = "200m"

# Number of synthetic mock datasets
N_MOCKS = 25
RANDOM_SEED = 42

# HSC Lens configuration
LENS_BIN_COUNTS = [3000]
LENS_AREA_DEG2 = 439.4937
Z_MIN = 0.19
Z_MAX = 0.52

# Base reference FITS file for realistic radii, lens redshift, and covariance
REF_FITS_PATH = (
    root_path / "output/redm_pdr3_5band_free_s16a/1bin/dsigma/hsc_hsc_bin0.fits"
)
OUTPUT_PLOT_PATH = root_path / "output/plots_for_agents/test_scatter_recovery_e2e.png"

# %% [Stage 1: Build Model State and Run End-to-End Recovery]

cosmo = cosmology.setCosmology(COSMOLOGY_NAME)
h_val = cosmo.h
flux_factor = 1e6 / h_val

rp_mpc_ref, _, ds_phys_err_ref, cov_phys_ref, z_lens_ref = load_dsigma_profile(
    str(REF_FITS_PATH)
)
rp_kpc_h_ref = rp_mpc_ref * 1000.0 * h_val

cov_colossus = cov_phys_ref * flux_factor**2
cov_inv_colossus = np.linalg.pinv(cov_colossus)

logging.info("Building profile scatter model state for e2e test...")
e2e_model_state = build_profile_scatter_model(
    rp_kpc_h=rp_kpc_h_ref,
    z_lens=z_lens_ref,
    bin_counts=LENS_BIN_COUNTS,
    area_sq_deg=LENS_AREA_DEG2,
    z_min=Z_MIN,
    z_max=Z_MAX,
    cosmology_name=COSMOLOGY_NAME,
    mass_def=MASS_DEF,
)

logging.info(f"Executing recovery experiment for true scatter = {TRUE_SCATTER:.3f}...")
recovery_results = execute_e2e_recovery_test(
    model_state=e2e_model_state,
    rp_mpc=rp_mpc_ref,
    cov_col=cov_colossus,
    cov_inv_col=cov_inv_colossus,
    factor=flux_factor,
    true_scatter=TRUE_SCATTER,
    true_alpha_c=TRUE_ALPHA_C,
    true_alpha_b=TRUE_ALPHA_B,
    f_mis=FIXED_F_MIS,
    sigma_r=FIXED_SIGMA_R,
    bin_index=0,
    n_mocks=N_MOCKS,
    seed=RANDOM_SEED,
)

# Print Summary Diagnostics
print("\n" + "=" * 70)
print(f"E2E SCATTER RECOVERY BENCHMARK (Truth: sigma = {TRUE_SCATTER:.3f})")
print("=" * 70)

res_nl_fix = recovery_results["fit_noiseless_fixed"]
res_nl_free = recovery_results["fit_noiseless_free"]
print(f"[Noiseless Recovery (Fixed alpha)]: sigma = {res_nl_fix['scatter']:.4f}")
print(
    f"[Noiseless Recovery (Free alpha)]:  sigma = {res_nl_free['scatter']:.4f}, "
    f"alpha_c = {res_nl_free['alpha_c']:.3f}, alpha_b = {res_nl_free['alpha_b']:.3f}"
)

s_fix_arr = np.array([r["scatter"] for r in recovery_results["fixed_results"]])
s_free_arr = np.array([r["scatter"] for r in recovery_results["free_results"]])
print("-" * 70)
print(
    f"[Noisy Mocks (Fixed alpha, N={N_MOCKS})]: "
    f"mean(sigma) = {np.mean(s_fix_arr):.3f} +/- {np.std(s_fix_arr):.3f} "
    f"(Bias: {np.mean(s_fix_arr) - TRUE_SCATTER:+.3f})"
)
print(
    f"[Noisy Mocks (Free alpha,  N={N_MOCKS})]: "
    f"mean(sigma) = {np.mean(s_free_arr):.3f} +/- {np.std(s_free_arr):.3f} "
    f"(Bias: {np.mean(s_free_arr) - TRUE_SCATTER:+.3f})"
)
print("=" * 70 + "\n")

# %% [Stage 2: Plot Diagnostic Summary]

plot_e2e_recovery_summary(
    recovery_data=recovery_results,
    rp_mpc=rp_mpc_ref,
    ds_phys_err=ds_phys_err_ref,
    out_plot_path=OUTPUT_PLOT_PATH,
)
