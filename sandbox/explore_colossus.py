# %%
import sys
import warnings
from pathlib import Path


import scipy.optimize as optimize
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw, profile_outer

# Find project root dynamically
_proj_root = Path(__file__).resolve().parent
while not (_proj_root / "pyproject.toml").exists():
    _proj_root = _proj_root.parent
    if _proj_root == _proj_root.parent:
        raise RuntimeError("Could not find pyproject.toml")
sys.path.append(str(_proj_root))

from initial import *  # noqa

# %% Configuration
DSIGMA_FITS = "output/cosine/Y3/dsigma/hsc_hsc_lens0.fits"
COSMOLOGY_NAME = "planck18"
MASS_DEF = "200m"

# %% Execution
# 1. Setup cosmology
cosmo = cosmology.setCosmology(COSMOLOGY_NAME)
h = cosmo.h

# 2. Load the data
t = Table.read(Path(_proj_root) / DSIGMA_FITS)

# Extract data (dsigma usually outputs physical Mpc and M_sun / pc^2)
rp_mpc = t["rp"]
ds_data = t["ds"]
ds_err = t["ds_err"]

z_lens = np.nanmedian(t["z_l"]) if "z_l" in t.colnames else 0.3
logging.info(f"Using lens redshift z_lens = {z_lens:.4f}")

# 3. Unit Conversions for Colossus
# rp_mpc -> kpc/h
# ds (M_sun/pc^2) -> ds (h M_sun/kpc^2)
rp_kpc_h = rp_mpc * 1000.0 * h
ds_colossus = ds_data * 1e6 / h
ds_err_colossus = ds_err * 1e6 / h

# 4. Perform the Fit using a custom `curve_fit` wrapper
# Colossus's built-in `.fit()` doesn't handle negative outer term densities well.
# By wrapping the model calculation, we can fit for M, c, and a 2-halo bias explicitly,
# and we can limit `max_r_interpolate` to avoid correlation function negativity.


def dsigma_model(r_kpc_h, M, c, bias):
    """
    Evaluate NFW + 2-halo model.
    M: Halo mass in M_sun/h
    c: Concentration
    bias: Linear halo bias for the 2-halo term
    """
    # Define the 2-halo term from matter correlation function
    outer_term = profile_outer.OuterTermCorrelationFunction(z=z_lens, bias=bias)

    # Initialize the NFW profile with the outer term attached
    p = profile_nfw.NFWProfile(
        M=M, c=c, z=z_lens, mdef=MASS_DEF, outer_terms=[outer_term]
    )

    # Evaluate deltaSigma exactly without interpolation to avoid negative correlation function issues.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.deltaSigma(
            r_kpc_h, interpolate=False, interpolate_surface_density=False
        )


logging.info("Starting NFW + 2-Halo fit...")

# Initial guesses: M = 1e13 M_sun/h, c = 5.0, bias = 1.0
p0 = [1e13 * h, 5.0, 1.0]

# Bounds: Mass (1e10 - 1e16), concentration (0.1 - 30), bias (0 - 10)
bounds = ([1e10, 0.1, 0.0], [1e16, 30.0, 10.0])

popt, pcov = optimize.curve_fit(
    dsigma_model,
    rp_kpc_h,
    ds_colossus,
    sigma=ds_err_colossus,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds,
    method="trf",
)

best_fit_M, best_fit_c, best_fit_bias = popt

# Calculate Chi2
ds_fit = dsigma_model(rp_kpc_h, best_fit_M, best_fit_c, best_fit_bias)
chi2 = np.sum(((ds_colossus - ds_fit) / ds_err_colossus) ** 2)
ndof = len(rp_kpc_h) - len(popt)

logging.info(f"Best fit M ({MASS_DEF}): {best_fit_M:.2e} M_sun/h")
logging.info(f"Best fit c: {best_fit_c:.2f}")
logging.info(f"Best fit bias: {best_fit_bias:.2f}")
logging.info(f"Reduced chi2: {chi2 / ndof:.2f}")

# Re-initialize profile with best fit parameters to generate models for plotting
outer_best = profile_outer.OuterTermCorrelationFunction(z=z_lens, bias=best_fit_bias)
p_best = profile_nfw.NFWProfile(
    M=best_fit_M, c=best_fit_c, z=z_lens, mdef=MASS_DEF, outer_terms=[outer_best]
)

# %% Plotting the results
fig, ax = plt.subplots(figsize=(6, 5))

# Data
ax.errorbar(rp_mpc, ds_data, yerr=ds_err, fmt="o", color="black", label="Data (dsigma)")

# Smooth Model
rp_smooth_mpc = np.logspace(np.log10(np.min(rp_mpc)), np.log10(np.max(rp_mpc)), 40)
rp_smooth_kpc_h = rp_smooth_mpc * 1000.0 * h

# Evaluate total signal
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ds_smooth_colossus = p_best.deltaSigma(
        rp_smooth_kpc_h, interpolate=False, interpolate_surface_density=False
    )
ds_smooth = ds_smooth_colossus * h / 1e6

# Evaluate 1-halo term only
p_1halo = profile_nfw.NFWProfile(M=best_fit_M, c=best_fit_c, z=z_lens, mdef=MASS_DEF)
ds_1h_colossus = p_1halo.deltaSigma(rp_smooth_kpc_h)
ds_1h = ds_1h_colossus * h / 1e6

# Evaluate 2-halo term only (Total - 1Halo)
ds_2h = ds_smooth - ds_1h

ax.plot(
    rp_smooth_mpc,
    ds_smooth,
    "-",
    color="red",
    label=f"Total Fit\n$M_{{{MASS_DEF}}}={best_fit_M / h:.1e}\,M_\odot$\n$c={best_fit_c:.1f}$\n$b={best_fit_bias:.2f}$",
)
ax.plot(rp_smooth_mpc, ds_1h, "--", color="blue", alpha=0.7, label="1-Halo Term")
ax.plot(rp_smooth_mpc, ds_2h, ":", color="green", alpha=0.7, label="2-Halo Term")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$R_p \ [\mathrm{Mpc}]$")
ax.set_ylabel(r"$\Delta\Sigma \ [\mathrm{M_\odot / pc^2}]$")
ax.legend(loc="upper right", fontsize="small")
ax.set_title("Colossus NFW + 2-Halo Fit")

plt.show()
