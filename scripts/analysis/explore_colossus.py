# %% [Initialization]
import sys
import warnings
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

import scipy.optimize as optimize  # noqa: E402
from colossus.cosmology import cosmology  # noqa: E402
from colossus.halo import profile_nfw, profile_outer  # noqa: E402

from initial import *  # noqa: F401,F403

# %% Local Functions


def load_dsigma_data(root_path, dsigma_fits_path):
    """Load ΔΣ data and jackknife-free error from a dsigma FITS file.

    Returns
    -------
    rp_mpc, ds_data, ds_err, z_lens : np.ndarray, np.ndarray, np.ndarray, float
    """
    t = Table.read(root_path / dsigma_fits_path)
    rp_mpc = t["rp"]
    ds_data = t["ds"]
    ds_err = t["ds_err"]
    z_lens = np.nanmedian(t["z_l"]) if "z_l" in t.colnames else 0.3
    logging.info(f"Using lens redshift z_lens = {z_lens:.4f}")
    return rp_mpc, ds_data, ds_err, z_lens


def convert_to_colossus_units(rp_mpc, ds_data, ds_err, h):
    """Convert Mpc / M_sun/pc^2 to kpc/h / h*M_sun/kpc^2.

    Parameters
    ----------
    h : float
        Hubble constant in units of 100 km/s/Mpc.
    """
    rp_kpc_h = rp_mpc * 1000.0 * h
    ds_colossus = ds_data * 1e6 / h
    ds_err_colossus = ds_err * 1e6 / h
    return rp_kpc_h, ds_colossus, ds_err_colossus


def build_dsigma_model(M, c, bias, z_lens, mass_def):
    """Build an NFW + 2-halo model and return a callable evaluating deltaSigma.

    The returned function evaluates deltaSigma exactly (no interpolation) to
    avoid negative correlation function issues. The 2-halo term is built from
    the matter correlation function with a linear halo bias.
    """
    outer_term = profile_outer.OuterTermCorrelationFunction(z=z_lens, bias=bias)
    p = profile_nfw.NFWProfile(
        M=M, c=c, z=z_lens, mdef=mass_def, outer_terms=[outer_term]
    )

    def _eval(r_kpc_h):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return p.deltaSigma(
                r_kpc_h, interpolate=False, interpolate_surface_density=False
            )

    return _eval, p


def dsigma_curve(r_kpc_h, M, c, bias, z_lens, mass_def):
    """Wrapper function compatible with ``scipy.optimize.curve_fit``."""
    eval_fn, _ = build_dsigma_model(M, c, bias, z_lens, mass_def)
    return eval_fn(r_kpc_h)


def fit_nfw_2halo(rp_kpc_h, ds_colossus, ds_err_colossus, z_lens, mass_def, h):
    """Fit M, c, and 2-halo bias via ``curve_fit`` (TRF, bounded).

    Returns
    -------
    popt, pcov, chi2, ndof : np.ndarray, np.ndarray, float, int
    """
    logging.info("Starting NFW + 2-Halo fit...")

    p0 = [1e13 * h, 5.0, 1.0]
    bounds = ([1e10, 0.1, 0.0], [1e16, 30.0, 10.0])

    def _model(r_kpc_h, M, c, bias):
        return dsigma_curve(r_kpc_h, M, c, bias, z_lens, mass_def)

    popt, pcov = optimize.curve_fit(
        _model,
        rp_kpc_h,
        ds_colossus,
        sigma=ds_err_colossus,
        absolute_sigma=True,
        p0=p0,
        bounds=bounds,
        method="trf",
    )

    best_fit_M, best_fit_c, best_fit_bias = popt
    ds_fit = _model(rp_kpc_h, best_fit_M, best_fit_c, best_fit_bias)
    chi2 = np.sum(((ds_colossus - ds_fit) / ds_err_colossus) ** 2)
    ndof = len(rp_kpc_h) - len(popt)

    logging.info(f"Best fit M ({mass_def}): {best_fit_M:.2e} M_sun/h")
    logging.info(f"Best fit c: {best_fit_c:.2f}")
    logging.info(f"Best fit bias: {best_fit_bias:.2f}")
    logging.info(f"Reduced chi2: {chi2 / ndof:.2f}")

    return popt, pcov, chi2, ndof


def plot_nfw_2halo_fit(
    rp_mpc,
    ds_data,
    ds_err,
    rp_smooth_mpc,
    ds_smooth,
    ds_1h,
    ds_2h,
    best_fit_M,
    best_fit_c,
    best_fit_bias,
    h,
    mass_def,
    output_path,
):
    """Plot the data, total fit, and 1-/2-halo decomposition and save figure."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.errorbar(
        rp_mpc, ds_data, yerr=ds_err, fmt="o", color="black", label="Data (dsigma)"
    )

    ax.plot(
        rp_smooth_mpc,
        ds_smooth,
        "-",
        color="red",
        label=(
            f"Total Fit\n"
            rf"$M_{{{mass_def}}}={best_fit_M / h:.1e}\,M_\odot$"
            f"\n$c={best_fit_c:.1f}$\n$b={best_fit_bias:.2f}$"
        ),
    )
    ax.plot(rp_smooth_mpc, ds_1h, "--", color="blue", alpha=0.7, label="1-Halo Term")
    ax.plot(rp_smooth_mpc, ds_2h, ":", color="green", alpha=0.7, label="2-Halo Term")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R_p\ [\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\Delta\Sigma\ [\mathrm{M_\odot / pc^2}]$")
    ax.legend(loc="upper right", fontsize="small")
    ax.set_title("Colossus NFW + 2-Halo Fit")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def evaluate_best_fit_components(
    best_fit_M, best_fit_c, best_fit_bias, z_lens, mass_def, rp_smooth_kpc_h, h
):
    """Evaluate total / 1-halo / 2-halo ΔΣ on a smooth radial grid (M_sun/pc^2)."""
    _, p_best = build_dsigma_model(
        best_fit_M, best_fit_c, best_fit_bias, z_lens, mass_def
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds_smooth_colossus = p_best.deltaSigma(
            rp_smooth_kpc_h, interpolate=False, interpolate_surface_density=False
        )
    ds_smooth = ds_smooth_colossus * h / 1e6

    p_1halo = profile_nfw.NFWProfile(
        M=best_fit_M, c=best_fit_c, z=z_lens, mdef=mass_def
    )
    ds_1h_colossus = p_1halo.deltaSigma(rp_smooth_kpc_h)
    ds_1h = ds_1h_colossus * h / 1e6

    ds_2h = ds_smooth - ds_1h
    return ds_smooth, ds_1h, ds_2h


# %% Global Configuration

DSIGMA_FITS = "output/cosine/Y3/dsigma/hsc_hsc_lens0.fits"
COSMOLOGY_NAME = "planck18"
MASS_DEF = "200m"

OUTPUT_FIG = project_root / "output/plots_for_agents/explore_colossus.png"


# %% [Stage 1: Load data and setup cosmology]
cosmo = cosmology.setCosmology(COSMOLOGY_NAME)
h = cosmo.h

rp_mpc, ds_data, ds_err, z_lens = load_dsigma_data(project_root, DSIGMA_FITS)
rp_kpc_h, ds_colossus, ds_err_colossus = convert_to_colossus_units(
    rp_mpc, ds_data, ds_err, h
)


# %% [Stage 2: Fit NFW + 2-halo model]
popt, pcov, chi2, ndof = fit_nfw_2halo(
    rp_kpc_h, ds_colossus, ds_err_colossus, z_lens, MASS_DEF, h
)
best_fit_M, best_fit_c, best_fit_bias = popt


# %% [Stage 3: Evaluate model components]
rp_smooth_mpc = np.logspace(np.log10(np.min(rp_mpc)), np.log10(np.max(rp_mpc)), 40)
rp_smooth_kpc_h = rp_smooth_mpc * 1000.0 * h

ds_smooth, ds_1h, ds_2h = evaluate_best_fit_components(
    best_fit_M, best_fit_c, best_fit_bias, z_lens, MASS_DEF, rp_smooth_kpc_h, h
)


# %% [Stage 4: Plot fit decomposition]
plot_nfw_2halo_fit(
    rp_mpc=rp_mpc,
    ds_data=ds_data,
    ds_err=ds_err,
    rp_smooth_mpc=rp_smooth_mpc,
    ds_smooth=ds_smooth,
    ds_1h=ds_1h,
    ds_2h=ds_2h,
    best_fit_M=best_fit_M,
    best_fit_c=best_fit_c,
    best_fit_bias=best_fit_bias,
    h=h,
    mass_def=MASS_DEF,
    output_path=OUTPUT_FIG,
)
