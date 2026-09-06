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

from colossus.cosmology import cosmology  # noqa: E402

from hsc_wl.scatter_fit import (  # noqa: E402
    build_scatter_model,
    compute_2halo_base_dsigma,
    compute_stacked_dsigma,
    compute_survey_number_density,
    convert_dsigma_to_colossus_units,
    fit_scatter_map,
)
from initial import *  # noqa: F401,F403

# %% Local Functions


def load_dsigma_data(root_path, dsigma_fits):
    """Load rp, ds, ds_err, jackknife covariance, and lens redshift from FITS."""
    dsigma_path = root_path / dsigma_fits
    t = Table.read(dsigma_path)

    rp_mpc = t["rp"]
    ds_data = t["ds"]
    ds_err = t["ds_err"]

    with fits.open(dsigma_path) as hdul:
        jk_cov = hdul["JK_COV"].data.copy()
    logging.info(f"Loaded JK covariance matrix: shape={jk_cov.shape}")

    z_lens = np.nanmedian(t["z_l"]) if "z_l" in t.colnames else 0.3
    logging.info(f"Using lens redshift z_lens = {z_lens:.4f}")
    return rp_mpc, ds_data, ds_err, jk_cov, z_lens


def log_map_result(map_result):
    """Print the MAP fit results to the logger."""
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


def plot_map_fit_decomposition(
    rp_mpc,
    ds_data,
    ds_err,
    rp_smooth_mpc,
    ds_smooth_phys,
    ds_1h_c_phys,
    ds_1h_m_phys,
    ds_2h_phys,
    map_result,
    map_scatter,
    map_f_mis,
    map_sigma_R,
    map_mean_logm,
    output_path,
):
    """Plot the MAP fit decomposition and save the figure."""
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

    ax.plot(rp_smooth_mpc, ds_1h_c_phys, "--", color="blue", label="1-halo (Centered)")
    if map_f_mis > 0.0:
        ax.plot(
            rp_smooth_mpc,
            ds_1h_m_phys,
            "--",
            color="orange",
            label="1-halo (Miscentered)",
        )
    ax.plot(rp_smooth_mpc, ds_2h_phys, ":", color="green", label="2-halo Term")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e0)
    ax.set_xlabel(r"$R_p\ [\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\Delta\Sigma\ [\mathrm{M_\odot / pc^2}]$")

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
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %% Global Configuration

# For all available labels, refer to `RUN_REGISTRY` in `src/hsc_wl/config.py`.
LABEL = "cca2_1bin"  # Must match a run label in RUN_REGISTRY
VERSION = "Y3"  # "Y1" or "Y3"
BIN_INDEX = 0  # Index of the lens bin to fit (0, 1, 2, 3...)

# Cosmologies: 'planck18-only', 'planck18', 'planck15-only', 'planck15', 'planck13-only', 'planck13',
# 'WMAP9-only', 'WMAP9-ML', 'WMAP9', 'WMAP7-only', 'WMAP7-ML', 'WMAP7', 'WMAP5-only', 'WMAP5-ML', 'WMAP5',
# 'WMAP3-ML', 'WMAP3', 'WMAP1-ML', 'WMAP1', 'illustris', 'bolshoi', 'multidark-planck', 'millennium', 'EdS', 'powerlaw'
COSMOLOGY_NAME = "planck18"

# Mass definitions: '200m', '200c', '500c', 'vir', etc.
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

# Fallback survey configuration if config or prepared catalog is missing
AREA_SQ_DEG = 51.4198
Z_MIN = 0.1
Z_MAX = 0.8
N_OBJ = 924

OUTPUT_FIG = project_root / "output/plots_for_agents/fit_custom_scatter.png"


# %% [Stage 1: Load data and setup cosmology]
cosmo = cosmology.setCosmology(COSMOLOGY_NAME)
h = cosmo.h

# Parse catalog_id and nbins from the unified run label
catalog_id, nbins = LABEL.rsplit("_", 1)
DSIGMA_FITS = (
    f"output/{catalog_id}/{nbins}/{VERSION}/dsigma/hsc_hsc_bin{BIN_INDEX}.fits"
)

# Resolve run configuration for sky area and redshift range
from hsc_wl.config import RUN_REGISTRY, resolve_config

cfg = RUN_REGISTRY[LABEL]
resolved_cfg = resolve_config(cfg, project_root)

if resolved_cfg.lens.area_deg2 is not None:
    AREA_SQ_DEG = resolved_cfg.lens.area_deg2
Z_MIN, Z_MAX = resolved_cfg.lens.redshift_range

# Read prepared lenses FITS table to get the correct N_OBJ
lenses_path = (
    project_root
    / f"output/{catalog_id}/{nbins}/prepare/{catalog_id}_{nbins}_lenses.fits"
)
if lenses_path.exists():
    lenses_table = Table.read(lenses_path)
    N_OBJ = len(lenses_table)
    logging.info(f"Loaded prepared lenses from {lenses_path}, N_OBJ = {N_OBJ}")
else:
    logging.info(
        f"Prepared lenses not found at {lenses_path}, using fallback N_OBJ = {N_OBJ}"
    )

n_obs = compute_survey_number_density(AREA_SQ_DEG, Z_MIN, Z_MAX, N_OBJ, COSMOLOGY_NAME)

rp_mpc, ds_data, ds_err, jk_cov, z_lens = load_dsigma_data(project_root, DSIGMA_FITS)

rp_kpc_h, ds_colossus, ds_err_colossus, jk_cov_colossus, jk_cov_inv_colossus = (
    convert_dsigma_to_colossus_units(rp_mpc, ds_data, ds_err, jk_cov, h)
)


# %% [Stage 2: Build model state]
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


# %% [Stage 3: Run MAP fit]
map_result = fit_scatter_map(model_state, ds_colossus, jk_cov_inv_colossus)
log_map_result(map_result)

map_scatter = map_result["scatter"]
map_f_mis = map_result["f_mis"]
map_sigma_R = map_result["sigma_R"]
map_mean_logm = map_result["mean_logm"]


# %% [Stage 4: Visualize fit decomposition]
best_fit_model = compute_stacked_dsigma(
    map_scatter, map_f_mis, map_sigma_R, model_state
)
best_fit_phys = best_fit_model * h / 1e6

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

plot_map_fit_decomposition(
    rp_mpc=rp_mpc,
    ds_data=ds_data,
    ds_err=ds_err,
    rp_smooth_mpc=rp_smooth_mpc,
    ds_smooth_phys=ds_smooth_phys,
    ds_1h_c_phys=ds_1h_c_phys,
    ds_1h_m_phys=ds_1h_m_phys,
    ds_2h_phys=ds_2h_phys,
    map_result=map_result,
    map_scatter=map_scatter,
    map_f_mis=map_f_mis,
    map_sigma_R=map_sigma_R,
    map_mean_logm=map_mean_logm,
    output_path=OUTPUT_FIG,
)
