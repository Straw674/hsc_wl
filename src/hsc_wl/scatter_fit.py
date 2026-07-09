"""Scatter fitting module for HSC weak lensing analysis.

Provides pure computation functions for stacked lensing signal modelling
with log-normal mass–observable scatter, miscentering, and abundance
matching.  A convenience ``build_scatter_model`` function pre-computes
all expensive grids and splines that are reused during fitting.

All units follow the Colossus convention (kpc/h, M_sun h/kpc^2) unless
otherwise noted.
"""

import logging
import warnings

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp
import scipy.optimize as optimize
from colossus.cosmology import cosmology
from colossus.halo import concentration, profile_nfw, profile_outer
from colossus.lss import bias, mass_function
from scipy.optimize import root_scalar
from scipy.special import erfc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------


def compute_2halo_base_dsigma(rp_eval, z_lens):
    """Compute the 2-halo base ΔΣ from the matter correlation function.

    Manually integrates the outer correlation function profile so that
    Colossus does *not* internally adjust the NFW parameters when
    ``outer_terms`` are passed with a fixed mass.

    Parameters
    ----------
    rp_eval : array_like
        Projected radii in kpc/h where the signal is evaluated.
    z_lens : float
        Lens redshift.

    Returns
    -------
    ds_xi : ndarray
        2-halo base ΔΣ evaluated at *rp_eval* (unit-bias, in Colossus
        internal units).
    """
    p_outer = profile_outer.OuterTermCorrelationFunction(z=z_lens, bias=1.0)
    r_grid = np.logspace(0, 5, 200)  # 1 kpc/h to 100 Mpc/h
    rho_outer = p_outer.density(r_grid)
    rho_outer[rho_outer < 0] = 0.0  # avoid negative density at BAO scales
    rho_spline = interp.InterpolatedUnivariateSpline(r_grid, rho_outer, k=1, ext=1)

    def _calc_sigma_outer(R):
        def integrand(r):
            return rho_spline(r) * r / np.sqrt(r**2 - R**2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=integrate.IntegrationWarning)
            val, _ = integrate.quad(integrand, R, 80000.0, limit=200)
        return 2.0 * val

    sigma_xi = np.array([_calc_sigma_outer(R) for R in rp_eval])
    x_sigma_xi_spline = interp.InterpolatedUnivariateSpline(
        rp_eval, rp_eval * sigma_xi, k=1, ext=1
    )

    ds_xi_out = np.zeros_like(rp_eval)
    for i, R in enumerate(rp_eval):
        if i == 0:
            sigma_bar = sigma_xi[0]
        else:
            sigma_bar = (2.0 / R**2) * x_sigma_xi_spline.integral(rp_eval[0], R) + (
                rp_eval[0] ** 2 / R**2
            ) * sigma_xi[0]
        ds_xi_out[i] = sigma_bar - sigma_xi[i]

    return ds_xi_out


def compute_mass_distribution(scatter, th_spline, dndlnm_spline, logm_grid_dense):
    """Return normalised weights for the true mass distribution.

    Given the mass–observable relation (MOR) scatter and the abundance
    matching threshold, compute the weight of each mass bin.

    Parameters
    ----------
    scatter : float
        Standard deviation of log10(M) in the MOR.
    th_spline : callable
        Spline mapping scatter → abundance-matching threshold.
    dndlnm_spline : callable
        Spline of dn/dlnM evaluated on *logm_grid_dense*.
    logm_grid_dense : array_like
        Dense grid of log10(M) values.

    Returns
    -------
    weights : ndarray
        Normalised weights (sum ≈ 1) for each mass bin.
    """
    th = float(th_spline(scatter))

    weights = (
        dndlnm_spline(logm_grid_dense)
        * 0.5
        * erfc((th - logm_grid_dense) / (np.sqrt(2) * scatter))
    )
    dlogm = logm_grid_dense[1] - logm_grid_dense[0]
    weights /= np.sum(weights * dlogm)

    return weights * dlogm


def fast_sigma_mis_gamma(R_bins, sigma_R, log_sigma_spline, R_min, R_max):
    """Compute the miscentered surface density using a Gamma(shape=2) kernel.

    Parameters
    ----------
    R_bins : array_like
        Target radii where miscentered Σ is evaluated (kpc/h).
    sigma_R : float
        Gamma scale parameter for the miscentering offset (kpc/h).
    log_sigma_spline : callable
        Spline of log(Σ) vs log(R).
    R_min, R_max : float
        Valid range of the spline.

    Returns
    -------
    sigma_mis : ndarray
        Miscentered surface density at *R_bins*.
    """
    theta = np.linspace(0, np.pi, 50)
    R_mis = np.linspace(0, 10 * sigma_R, 75)

    R_grid, R_mis_grid, theta_grid = np.meshgrid(R_bins, R_mis, theta, indexing="ij")
    arg = np.sqrt(
        R_grid**2 + R_mis_grid**2 + 2 * R_grid * R_mis_grid * np.cos(theta_grid)
    )

    mask = (arg >= R_min) & (arg <= R_max)
    sigma_eval = np.zeros_like(arg)
    sigma_eval[mask] = np.exp(log_sigma_spline(np.log(arg[mask])))
    sigma_eval[arg < R_min] = np.exp(log_sigma_spline(np.log(R_min)))

    # Integrate over theta
    sigma_mis_R_Rmis = np.trapz(sigma_eval, theta, axis=2) / np.pi

    # Integrate over R_mis (Gamma shape=2 weighting)
    # PDF = (R_mis / sigma_R**2) * exp(-R_mis / sigma_R)
    P_Rmis = (R_mis / sigma_R**2) * np.exp(-R_mis / sigma_R)
    norm = np.trapz(P_Rmis, R_mis)
    if norm > 0:
        P_Rmis = P_Rmis / norm

    integrand = sigma_mis_R_Rmis * P_Rmis[None, :]
    return np.trapz(integrand, R_mis, axis=1)


def compute_stacked_dsigma(
    scatter,
    f_mis,
    sigma_R,
    model_state,
    rp_eval=None,
    ds_xi_eval=None,
    return_components=False,
):
    """Compute the stacked ΔΣ with miscentering and abundance matching.

    Parameters
    ----------
    scatter : float
        Standard deviation of log10(M) in the MOR.
    f_mis : float
        Fraction of miscentered halos.
    sigma_R : float
        Gamma scale parameter of the miscentering offset (kpc/h).
    model_state : dict
        Pre-computed model state returned by ``build_scatter_model``.
    rp_eval : array_like, optional
        Projected radii (kpc/h).  Defaults to ``model_state['rp_mcmc']``.
    ds_xi_eval : array_like, optional
        2-halo base ΔΣ at *rp_eval*.  Defaults to ``model_state['ds_xi']``.
    return_components : bool, optional
        If *True*, also return individual 1-halo and 2-halo components.

    Returns
    -------
    ds_total : ndarray
        Total stacked ΔΣ.
    (ds_1h_centered, ds_1h_mis, ds_2h) : tuple of ndarray
        Returned only when *return_components* is True.
    """
    norm_weights = compute_mass_distribution(
        scatter,
        model_state["th_spline"],
        model_state["dndlnm_spline"],
        model_state["logm_grid_dense"],
    )

    if rp_eval is None:
        rp_eval = model_state["rp_mcmc"]
        rp_eval_dense = model_state["rp_dense_mcmc"]
        ds_xi_eval = model_state["ds_xi"]
        use_grid = True
    else:
        rp_eval_dense = np.logspace(-1, np.log10(np.max(rp_eval) * 10), 200)
        use_grid = False

    # 1. Compute stacked 1-halo ΔΣ(R), 2-halo ΔΣ(R), and Σ(R)
    if use_grid:
        sigma_1h_dense = np.dot(norm_weights, model_state["sigma_1h_grid"])
        ds_1h_centered = np.dot(norm_weights, model_state["ds_1h_grid"])
        ds_2h_stack = np.sum(norm_weights * model_state["b_grid_dense"]) * ds_xi_eval
    else:
        sigma_1h_dense = np.zeros_like(rp_eval_dense)
        ds_1h_centered = np.zeros_like(rp_eval)
        ds_2h_stack = np.zeros_like(rp_eval)
        logm_grid_dense = model_state["logm_grid_dense"]
        c_spline = model_state["c_spline"]
        b_spline = model_state["b_spline"]
        z_lens = model_state["z_lens"]
        mass_def = model_state["mass_def"]
        for logm, w in zip(logm_grid_dense, norm_weights):
            M_h = 10**logm
            c = c_spline(logm)
            b = b_spline(logm)
            p = profile_nfw.NFWProfile(M=M_h, c=c, z=z_lens, mdef=mass_def)
            sigma_1h_dense += w * p.surfaceDensity(rp_eval_dense)
            ds_1h_centered += w * p.deltaSigma(rp_eval)
            ds_2h_stack += w * b * ds_xi_eval

    # 2. Apply miscentering to Σ(R)
    if f_mis > 0.0 and sigma_R > 0.0:
        log_sigma_spline = interp.InterpolatedUnivariateSpline(
            np.log(rp_eval_dense), np.log(sigma_1h_dense), ext=1
        )
        sigma_1h_mis = fast_sigma_mis_gamma(
            rp_eval_dense,
            sigma_R,
            log_sigma_spline,
            rp_eval_dense[0],
            rp_eval_dense[-1],
        )

        # 3. Compute miscentered ΔΣ from miscentered Σ
        x_sigma_spline = interp.InterpolatedUnivariateSpline(
            rp_eval_dense, rp_eval_dense * sigma_1h_mis, k=1
        )
        sigma_mis_spline = interp.InterpolatedUnivariateSpline(
            rp_eval_dense, sigma_1h_mis, k=1
        )

        ds_1h_mis = np.zeros_like(rp_eval)
        for i, R in enumerate(rp_eval):
            integral_val = x_sigma_spline.integral(0.0, R)
            sigma_bar = (2.0 / R**2) * integral_val
            sigma_R_val = sigma_mis_spline(R)
            ds_1h_mis[i] = sigma_bar - sigma_R_val

        ds_1h_final = (1.0 - f_mis) * ds_1h_centered + f_mis * ds_1h_mis
    else:
        ds_1h_final = ds_1h_centered
        ds_1h_mis = np.zeros_like(rp_eval)

    ds_total = ds_1h_final + ds_2h_stack

    if return_components:
        return (
            ds_total,
            (1.0 - f_mis) * ds_1h_centered,
            f_mis * ds_1h_mis,
            ds_2h_stack,
        )
    return ds_total


# ---------------------------------------------------------------------------
# Unit conversion utilities
# ---------------------------------------------------------------------------


def convert_dsigma_to_colossus_units(rp_mpc, ds_data, ds_err, jk_cov, h):
    """Convert dsigma output units to Colossus internal units.

    dsigma units:  physical Mpc, M_sun/pc^2
    Colossus units: kpc/h, M_sun h/kpc^2

    Parameters
    ----------
    rp_mpc : array_like
        Projected radii in physical Mpc.
    ds_data : array_like
        ΔΣ values in M_sun/pc^2.
    ds_err : array_like
        ΔΣ errors in M_sun/pc^2.
    jk_cov : ndarray
        Jackknife covariance matrix in (M_sun/pc^2)^2.
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    rp_kpc_h : ndarray
    ds_colossus : ndarray
    ds_err_colossus : ndarray
    jk_cov_colossus : ndarray
    jk_cov_inv_colossus : ndarray
    """
    rp_kpc_h = np.asarray(rp_mpc) * 1000.0 * h
    factor = 1e6 / h
    ds_colossus = np.asarray(ds_data) * factor
    ds_err_colossus = np.asarray(ds_err) * factor
    jk_cov_colossus = np.asarray(jk_cov) * factor**2
    jk_cov_inv_colossus = np.linalg.inv(jk_cov_colossus)
    return rp_kpc_h, ds_colossus, ds_err_colossus, jk_cov_colossus, jk_cov_inv_colossus


def compute_survey_number_density(
    area_sq_deg, z_min, z_max, n_obj, cosmology_name="planck18"
):
    """Compute comoving survey volume and observed number density.

    Parameters
    ----------
    area_sq_deg : float
        Survey area in square degrees.
    z_min, z_max : float
        Redshift range of the lens sample.
    n_obj : float
        Number of objects in the sample.
    cosmology_name : str, optional
        Colossus cosmology name (default ``'planck18'``).

    Returns
    -------
    n_obs : float
        Comoving number density in (h/Mpc)^3.
    """
    cosmo = cosmology.setCosmology(cosmology_name)
    d_c_min = cosmo.comovingDistance(z_min=0.0, z_max=z_min)
    d_c_max = cosmo.comovingDistance(z_min=0.0, z_max=z_max)
    V_sphere = (4.0 / 3.0) * np.pi * (d_c_max**3 - d_c_min**3)
    f_sky = area_sq_deg / 41253.0
    V_survey = V_sphere * f_sky
    n_obs = n_obj / V_survey
    logger.info("Survey Volume: %.2e (Mpc/h)^3", V_survey)
    logger.info("Observed Number Density n_obs: %.2e (h/Mpc)^3", n_obs)
    return n_obs


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_scatter_model(
    rp_kpc_h,
    z_lens,
    n_obs,
    cosmology_name="planck18",
    mass_def="200m",
    conc_model="diemer19",
    bias_model="tinker10",
    mass_func_model="tinker08",
):
    """Pre-compute all grids, splines, and signals needed for scatter fitting.

    This is the single expensive setup call.  The returned *model_state*
    dictionary is passed to ``compute_stacked_dsigma`` and ``fit_scatter_map``.

    Parameters
    ----------
    rp_kpc_h : array_like
        Projected radii of the data in kpc/h.
    z_lens : float
        Lens redshift.
    n_obs : float
        Observed comoving number density (h/Mpc)^3.
    cosmology_name : str
        Colossus cosmology identifier.
    mass_def : str
        Halo mass definition (e.g. ``'200m'``).
    conc_model, bias_model, mass_func_model : str
        Model names for concentration, bias, and mass function.

    Returns
    -------
    model_state : dict
        Dictionary containing all pre-computed data.
    """
    cosmo = cosmology.setCosmology(cosmology_name)
    h = cosmo.h

    # --- Concentration, bias, HMF splines ---
    logger.info("Pre-computing concentration, bias, and HMF splines...")
    logm_grid = np.linspace(12.0, 16.0, 100)
    c_grid = [
        concentration.concentration(10**m, mass_def, z_lens, model=conc_model)
        for m in logm_grid
    ]
    b_grid = [
        bias.haloBias(10**m, model=bias_model, z=z_lens, mdef=mass_def)
        for m in logm_grid
    ]
    dndlnm_grid = [
        mass_function.massFunction(
            10**m, z_lens, mdef=mass_def, model=mass_func_model, q_out="dndlnM"
        )
        for m in logm_grid
    ]

    c_spline = interp.InterpolatedUnivariateSpline(logm_grid, c_grid)
    b_spline = interp.InterpolatedUnivariateSpline(logm_grid, b_grid)
    dndlnm_spline_obj = interp.InterpolatedUnivariateSpline(logm_grid, dndlnm_grid)

    # --- 2-halo base signal ---
    logger.info("Pre-computing matter correlation function 2-halo base signal...")
    ds_xi = compute_2halo_base_dsigma(rp_kpc_h, z_lens)

    # --- Abundance matching threshold spline ---
    logger.info("Pre-computing Abundance Matching threshold spline...")

    # Safely cap n_obs if it exceeds the maximum possible integrated HMF density
    func_max = lambda m: dndlnm_spline_obj(m) * np.log(10)
    n_pred_max, _ = integrate.quad(func_max, 13.0, 16.0, limit=100)
    if n_obs >= n_pred_max:
        logger.warning(
            "Observed number density n_obs (%.2e) exceeds maximum HMF density (%.2e). Capping at %.2e.",
            n_obs,
            n_pred_max,
            n_pred_max * 0.95,
        )
        n_obs = n_pred_max * 0.95

    scatter_grid = np.linspace(0.01, 5.0, 100)
    th_grid = []
    for s in scatter_grid:

        def _diff(th, _s=s):
            func = lambda m: (
                dndlnm_spline_obj(m)
                * np.log(10)
                * 0.5
                * erfc((th - m) / (np.sqrt(2) * _s))
            )
            n_pred, _ = integrate.quad(func, 13.0, 16.0, limit=100)
            return n_pred - n_obs

        th_grid.append(root_scalar(_diff, bracket=[-50.0, 50.0]).root)

    th_spline = interp.InterpolatedUnivariateSpline(scatter_grid, th_grid, k=3, ext=3)

    # --- NFW surface-density and ΔΣ grids ---
    logger.info("Pre-computing NFW grids...")
    logm_grid_dense = np.linspace(12.0, 16.0, 100)
    b_grid_dense = b_spline(logm_grid_dense)
    rp_dense_mcmc = np.logspace(-1, np.log10(np.max(rp_kpc_h) * 10), 200)

    sigma_1h_grid = np.zeros((len(logm_grid_dense), len(rp_dense_mcmc)))
    ds_1h_grid = np.zeros((len(logm_grid_dense), len(rp_kpc_h)))
    for i, logm in enumerate(logm_grid_dense):
        p = profile_nfw.NFWProfile(
            M=10**logm, c=c_spline(logm), z=z_lens, mdef=mass_def
        )
        sigma_1h_grid[i] = p.surfaceDensity(rp_dense_mcmc)
        ds_1h_grid[i] = p.deltaSigma(rp_kpc_h)

    return {
        "h": h,
        "z_lens": z_lens,
        "n_obs": n_obs,
        "cosmo": cosmo,
        "c_spline": c_spline,
        "b_spline": b_spline,
        "dndlnm_spline": dndlnm_spline_obj,
        "th_spline": th_spline,
        "ds_xi": ds_xi,
        "logm_grid_dense": logm_grid_dense,
        "b_grid_dense": b_grid_dense,
        "rp_mcmc": np.asarray(rp_kpc_h),
        "rp_dense_mcmc": rp_dense_mcmc,
        "sigma_1h_grid": sigma_1h_grid,
        "ds_1h_grid": ds_1h_grid,
        "mass_def": mass_def,
    }


# ---------------------------------------------------------------------------
# MAP fitting
# ---------------------------------------------------------------------------


def fit_scatter_map(model_state, ds_colossus, jk_cov_inv_colossus):
    """Fast MAP estimation of scatter, f_mis, sigma_R.

    Uses ``scipy.optimize.minimize`` (L-BFGS-B) with multiple random
    starting points to mitigate local-minima issues.

    Parameters
    ----------
    model_state : dict
        Pre-computed model state from ``build_scatter_model``.
    ds_colossus : array_like
        Observed ΔΣ in Colossus units (M_sun h / kpc^2).
    jk_cov_inv_colossus : ndarray
        Inverse jackknife covariance in Colossus units.

    Returns
    -------
    result : dict
        Best-fit parameters and diagnostics::

            {
                'scatter': float,
                'f_mis': float,
                'sigma_R': float,
                'chi2_reduced': float,
                'mean_logm': float,
                'n_dof': int,
                'success': bool,
            }
    """
    ds_obs = np.asarray(ds_colossus)
    n_data = len(ds_obs)
    n_params = 3
    n_dof = n_data - n_params

    bounds = [(0.01, 5.0), (0.01, 1.0), (1.0, 5000.0)]

    def neg_log_likelihood(theta):
        scatter, f_mis, sigma_R = theta
        try:
            model = compute_stacked_dsigma(scatter, f_mis, sigma_R, model_state)
            residual = ds_obs - model
            return 0.5 * residual @ jk_cov_inv_colossus @ residual
        except Exception:
            import traceback

            logger.error(f"Fit crashed at theta={theta}:\n{traceback.format_exc()}")
            return 1e30

    # Multiple starting points to avoid local minima
    starting_points = [
        [0.2, 0.2, 200.0],
        [0.1, 0.05, 50.0],
        [0.4, 0.3, 400.0],
        [0.3, 0.1, 100.0],
        [0.5, 0.5, 300.0],
    ]

    def _run_single_opt(x0):
        try:
            return optimize.minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-6},
            )
        except Exception:
            return None

    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=-1)(
            delayed(_run_single_opt)(x0) for x0 in starting_points
        )
    except ImportError:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(_run_single_opt, starting_points))

    best_result = None
    best_fun = np.inf

    for res in results:
        if res is not None and res.fun < best_fun:
            best_fun = res.fun
            best_result = res

    if best_result is None:
        logger.error("All MAP optimisation attempts failed.")
        return {
            "scatter": np.nan,
            "f_mis": np.nan,
            "sigma_R": np.nan,
            "chi2_reduced": np.nan,
            "mean_logm": np.nan,
            "n_dof": n_dof,
            "success": False,
        }

    best_scatter, best_f_mis, best_sigma_R = best_result.x
    chi2_red = 2.0 * best_result.fun / n_dof if n_dof > 0 else np.nan

    # Derived <logM>
    weights = compute_mass_distribution(
        best_scatter,
        model_state["th_spline"],
        model_state["dndlnm_spline"],
        model_state["logm_grid_dense"],
    )
    mean_logm = np.sum(model_state["logm_grid_dense"] * weights)

    logger.info("MAP Result:")
    logger.info("  scatter   = %.3f", best_scatter)
    logger.info("  f_mis     = %.3f", best_f_mis)
    logger.info("  sigma_R   = %.1f", best_sigma_R)
    logger.info("  chi2/dof  = %.3f", chi2_red)
    logger.info("  <logM>    = %.3f", mean_logm)

    return {
        "scatter": best_scatter,
        "f_mis": best_f_mis,
        "sigma_R": best_sigma_R,
        "chi2_reduced": chi2_red,
        "mean_logm": mean_logm,
        "n_dof": n_dof,
        "success": best_result.success,
    }
