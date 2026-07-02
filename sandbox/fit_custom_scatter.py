# %% Initialization
import sys
from pathlib import Path

# Find project root dynamically
_proj_root = Path(__file__).resolve().parent
while not (_proj_root / "pyproject.toml").exists():
    _proj_root = _proj_root.parent
    if _proj_root == _proj_root.parent:
        raise RuntimeError("Could not find pyproject.toml")
sys.path.append(str(_proj_root))

from initial import *
from hsc_wl.scatter_fit import (
    build_scatter_model,
    compute_2halo_base_dsigma,
    compute_mass_distribution,
    compute_stacked_dsigma,
    compute_survey_number_density,
    convert_dsigma_to_colossus_units,
    fit_scatter_map,
)

# %% Global Configuration

DSIGMA_FITS = "output/cosine/Y3/dsigma/hsc_hsc_bin0.fits"

# Cosmologies: 'planck18-only', 'planck18', 'planck15-only', 'planck15', 'planck13-only', 'planck13',
# 'WMAP9-only', 'WMAP9-ML', 'WMAP9', 'WMAP7-only', 'WMAP7-ML', 'WMAP7', 'WMAP5-only', 'WMAP5-ML', 'WMAP5',
# 'WMAP3-ML', 'WMAP3', 'WMAP1-ML', 'WMAP1', 'illustris', 'bolshoi', 'multidark-planck', 'millennium', 'EdS', 'powerlaw'
COSMOLOGY_NAME = "planck18"

# Mass definitions: '200m', '200c', '500c', 'vir', etc. (m: mean density, c: critical density)
MASS_DEF = "200m"

# Concentration models: 'bullock01', 'duffy08', 'klypin11', 'prada12', 'bhattacharya13', 'dutton14',
# 'diemer15_orig', 'diemer15', 'klypin16_m', 'klypin16_nu', 'ludlow16', 'child18', 'diemer19', 'ishiyama21'
CONC_MODEL = "diemer19"

# Bias models: 'cole89', 'jing98', 'sheth01', 'seljak04', 'pillepich10', 'tinker10', 'bhattacharya11', 'comparat17'
BIAS_MODEL = "tinker10"

# Mass function models: 'press74', 'sheth99' (fof only), 'jenkins01', 'reed03', 'warren06', 'reed07', 'tinker08',
# 'crocce10', 'bhattacharya11', 'courtin11', 'angulo12', 'watson13', 'bocquet16', 'despali16',
# 'rodriguezpuebla16', 'comparat17', 'diemer20', 'seppi20', 'yung24', 'yung25', 'fernandezgarcia26', 'fiorilli26'
MASS_FUNC_MODEL = "tinker08"

# Survey Configuration
AREA_SQ_DEG = 51.4198
Z_MIN = 0.1
Z_MAX = 0.8
N_OBJ = 924


# %% Execution

# Initialization and Data Loading
from colossus.cosmology import cosmology

cosmo = cosmology.setCosmology(COSMOLOGY_NAME)
h = cosmo.h

n_obs = compute_survey_number_density(AREA_SQ_DEG, Z_MIN, Z_MAX, N_OBJ, COSMOLOGY_NAME)

dsigma_path = Path(_proj_root) / DSIGMA_FITS
t = Table.read(dsigma_path)

rp_mpc = t["rp"]
ds_data = t["ds"]
ds_err = t["ds_err"]

# Load jackknife covariance matrix from HDU[2]
with fits.open(dsigma_path) as hdul:
    jk_cov = hdul["JK_COV"].data.copy()
logging.info(f"Loaded JK covariance matrix: shape={jk_cov.shape}")

z_lens = np.nanmedian(t["z_l"]) if "z_l" in t.colnames else 0.3
logging.info(f"Using lens redshift z_lens = {z_lens:.4f}")

# Convert to Colossus internal units
rp_kpc_h, ds_colossus, ds_err_colossus, jk_cov_colossus, jk_cov_inv_colossus = (
    convert_dsigma_to_colossus_units(rp_mpc, ds_data, ds_err, jk_cov, h)
)


# %% Build Model State

model_state = build_scatter_model(
    rp_kpc_h,
    z_lens,
    n_obs,
    COSMOLOGY_NAME,
    MASS_DEF,
    CONC_MODEL,
    BIAS_MODEL,
    MASS_FUNC_MODEL,
)


# %% Run MAP Fit

map_result = fit_scatter_map(model_state, ds_colossus, jk_cov_inv_colossus)

map_scatter = map_result["scatter"]
map_f_mis = map_result["f_mis"]
map_sigma_R = map_result["sigma_R"]
map_mean_logm = map_result["mean_logm"]

logging.info("MAP Result:")
logging.info(f"scatter   = {map_scatter:.3f}")
logging.info(f"f_mis     = {map_f_mis:.3f}")
logging.info(f"sigma_R   = {map_sigma_R:.1f}")
logging.info(f"chi2/dof  = {map_result['chi2_reduced']:.3f}")
logging.info(f"Derived <logM> = {map_mean_logm:.3f}")


# %% Fit Visualization
best_fit_model = compute_stacked_dsigma(
    map_scatter, map_f_mis, map_sigma_R, model_state
)
# Centered model for comparison
centered_model = compute_stacked_dsigma(map_scatter, 0.0, 0.0, model_state)

best_fit_phys = best_fit_model * h / 1e6
centered_phys = centered_model * h / 1e6

# Smooth line generation
rp_smooth_mpc = np.logspace(np.log10(np.min(rp_mpc)), np.log10(np.max(rp_mpc)), 40)
rp_smooth_kpc_h = rp_smooth_mpc * 1000.0 * h

ds_xi_smooth = compute_2halo_base_dsigma(rp_smooth_kpc_h, z_lens)

ds_smooth_colossus, ds_1h_c, ds_1h_m, ds_2h = compute_stacked_dsigma(
    map_scatter,
    map_f_mis,
    map_sigma_R,
    model_state,
    rp_eval=rp_smooth_kpc_h,
    ds_xi_eval=ds_xi_smooth,
    return_components=True,
)

ds_smooth_phys = ds_smooth_colossus * h / 1e6
ds_1h_c_phys = ds_1h_c * h / 1e6
ds_1h_m_phys = ds_1h_m * h / 1e6
ds_2h_phys = ds_2h * h / 1e6

fig, ax = plt.subplots(figsize=(7, 6))
ax.errorbar(rp_mpc, ds_data, yerr=ds_err, fmt="o", color="black", label="Data")

ax.plot(
    rp_smooth_mpc,
    ds_smooth_phys,
    "-",
    color="red",
    linewidth=2,
    label="Total Best Fit",
)

# Plot components
ax.plot(rp_smooth_mpc, ds_1h_c_phys, "--", color="blue", label="1-halo (Centered)")
if map_f_mis > 0.0:
    ax.plot(
        rp_smooth_mpc, ds_1h_m_phys, "--", color="orange", label="1-halo (Miscentered)"
    )
ax.plot(rp_smooth_mpc, ds_2h_phys, ":", color="green", label="2-halo Term")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(bottom=1e0)  # avoid plotting negative issues on log scale if any
ax.set_xlabel(r"$R_p \ [\mathrm{Mpc}]$")
ax.set_ylabel(r"$\Delta\Sigma \ [\mathrm{M_\odot / pc^2}]$")

fit_summary = "\n".join(
    [
        rf"$\sigma = {map_scatter:.2f}$",
        rf"$f_{{\rm mis}} = {map_f_mis:.2f}$",
        rf"$\sigma_R = {map_sigma_R:.1f}\ \mathrm{{kpc}}/h$",
        rf"$\chi^2/\nu = {map_result['chi2_reduced']:.3f}$",
        rf"$\langle \log M \rangle = {map_mean_logm:.3f}$",
    ]
)
ax.text(
    0.05,
    0.05,
    fit_summary,
    transform=ax.transAxes,
    fontsize=10,
    va="bottom",
    ha="left",
    bbox=dict(
        boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.7"
    ),
)

ax.legend(loc="upper right", fontsize=10)
ax.set_title("Stacked Lensing MAP Fit Decomposition")
plt.tight_layout()
plt.show()
