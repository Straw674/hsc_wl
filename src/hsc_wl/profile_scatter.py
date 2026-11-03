"""Profile scatter fitting module for HSC weak lensing analysis.

Provides pure functional modeling of stacked lensing signals with log-normal
mass-observable scatter, abundance matching mass distribution (1-bin or 4-bin),
flexible NFW concentration scaling, flexible large-scale halo bias scaling,
and miscentering convolution.
"""

import logging
import warnings
from typing import Any

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp
import scipy.optimize as optimize
from astropy.io import fits
from astropy.table import Table
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_so
from colossus.lss import bias, mass_function
from scipy.optimize import root_scalar
from scipy.special import erfc

from hsc_wl.scatter_fit import compute_2halo_base_dsigma, fast_sigma_mis_gamma

logger = logging.getLogger(__name__)


def compute_survey_volume(
    area_sq_deg: float,
    z_min: float,
    z_max: float,
    cosmology_name: str = "planck18",
) -> float:
    """Compute comoving survey volume in (Mpc/h)^3.

    Parameters
    ----------
    area_sq_deg : float
        Survey area in square degrees.
    z_min, z_max : float
        Redshift range of the lens sample.
    cosmology_name : str
        Colossus cosmology identifier.

    Returns
    -------
    float
        Comoving volume in (Mpc/h)^3.
    """
    cosmo = cosmology.setCosmology(cosmology_name)
    d_c_min = cosmo.comovingDistance(z_min=0.0, z_max=z_min)
    d_c_max = cosmo.comovingDistance(z_min=0.0, z_max=z_max)
    v_sphere = (4.0 / 3.0) * np.pi * (d_c_max**3 - d_c_min**3)
    f_sky = area_sq_deg / 41253.0
    return v_sphere * f_sky


def build_abundance_matching_splines(
    cum_n_obs_list: list[float],
    hmf_spline: Any,
    scatter_grid: np.ndarray,
) -> list[interp.InterpolatedUnivariateSpline]:
    """Build threshold splines for abundance matching across cumulative densities.

    Parameters
    ----------
    cum_n_obs_list : list of float
        Target cumulative number densities in (h/Mpc)^3.
    hmf_spline : callable
        Spline mapping log10(M) to dn/dlnM.
    scatter_grid : ndarray
        1D grid of scatter values.

    Returns
    -------
    list of InterpolatedUnivariateSpline
        Splines mapping scatter -> log10(M_threshold).
    """
    th_splines = []
    for n_obs in cum_n_obs_list:
        th_grid = []
        for s in scatter_grid:

            def _diff(th: float, _s: float = s, _n: float = n_obs) -> float:
                func = lambda m: (
                    hmf_spline(m)
                    * np.log(10)
                    * 0.5
                    * erfc((th - m) / (np.sqrt(2) * _s))
                )
                n_pred, _ = integrate.quad(func, 11.0, 16.5, limit=50)
                return n_pred - _n

            root = root_scalar(_diff, bracket=[-10.0, 25.0]).root
            th_grid.append(root)
        th_splines.append(
            interp.InterpolatedUnivariateSpline(scatter_grid, th_grid, k=3)
        )
    return th_splines


def compute_mass_weights(
    scatter: float,
    bin_index: int,
    th_splines: list[interp.InterpolatedUnivariateSpline],
    m_dense: np.ndarray,
    dndlnm_dense: np.ndarray,
    dlogm: float,
) -> np.ndarray:
    """Compute normalized mass distribution weights for a 1-bin or 4-bin selection.

    Parameters
    ----------
    scatter : float
        Log-normal mass-observable scatter sigma.
    bin_index : int
        Index of the bin (0 for top bin or 1-bin mode, 1..3 for subsequent bins).
    th_splines : list of InterpolatedUnivariateSpline
        Precomputed threshold splines for cumulative boundaries.
    m_dense : ndarray
        Grid of log10(M_200m).
    dndlnm_dense : ndarray
        HMF dn/dlnM on m_dense.
    dlogm : float
        Spacing of m_dense.

    Returns
    -------
    ndarray
        Normalized weights summing to 1.
    """
    if bin_index == 0:
        th = float(th_splines[0](scatter))
        selection = 0.5 * erfc((th - m_dense) / (np.sqrt(2) * scatter))
    else:
        th_low = float(th_splines[bin_index](scatter))
        th_high = float(th_splines[bin_index - 1](scatter))
        selection = 0.5 * erfc((th_low - m_dense) / (np.sqrt(2) * scatter)) - (
            0.5 * erfc((th_high - m_dense) / (np.sqrt(2) * scatter))
        )
        selection = np.maximum(selection, 0.0)

    weights = dndlnm_dense * selection * dlogm
    norm = np.sum(weights)
    if norm <= 0.0:
        return np.ones_like(weights) / len(weights)
    return weights / norm


def compute_vectorized_nfw_profiles(
    r200m_dense: np.ndarray,
    c_arr: np.ndarray,
    rho_m: float,
    rp_dense: np.ndarray,
    rp_kpc_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectorized NFW surface density and DeltaSigma grids across mass.

    Parameters
    ----------
    r200m_dense : ndarray
        Array of R_200m in kpc/h for each mass bin.
    c_arr : ndarray
        Array of concentration c_200m for each mass bin.
    rho_m : float
        Mean matter density in M_sun h^2 / kpc^3.
    rp_dense : ndarray
        Dense projected radial grid (kpc/h) for miscentering convolution.
    rp_kpc_h : ndarray
        Projected radial grid (kpc/h) of observational data points.

    Returns
    -------
    sigma_grid : ndarray of shape (N_mass, N_rp_dense)
        Surface density profiles.
    ds_grid : ndarray of shape (N_mass, N_rp_kpc_h)
        Differential surface density profiles.
    """
    rs = r200m_dense / c_arr
    c_factor = (c_arr**3) / (np.log(1.0 + c_arr) - c_arr / (1.0 + c_arr))
    rhos = (200.0 / 3.0) * rho_m * c_factor

    # Evaluate surface density on rp_dense
    x_dense = rp_dense[None, :] / rs[:, None]
    f_dense = np.zeros_like(x_dense)

    mask_low = x_dense < 0.999
    x_l = x_dense[mask_low]
    f_dense[mask_low] = (1.0 / (x_l**2 - 1.0)) * (
        1.0
        - (2.0 / np.sqrt(1.0 - x_l**2)) * np.arctanh(np.sqrt((1.0 - x_l) / (1.0 + x_l)))
    )

    mask_mid = (x_dense >= 0.999) & (x_dense <= 1.001)
    f_dense[mask_mid] = 1.0 / 3.0

    mask_high = x_dense > 1.001
    x_h = x_dense[mask_high]
    f_dense[mask_high] = (1.0 / (x_h**2 - 1.0)) * (
        1.0
        - (2.0 / np.sqrt(x_h**2 - 1.0)) * np.arctan(np.sqrt((x_h - 1.0) / (x_h + 1.0)))
    )
    sigma_grid = 2.0 * rs[:, None] * rhos[:, None] * f_dense

    # Evaluate DeltaSigma on rp_kpc_h
    x_data = rp_kpc_h[None, :] / rs[:, None]
    g_data = np.zeros_like(x_data)
    f_data = np.zeros_like(x_data)

    mask_dl = x_data < 0.999
    xd_l = x_data[mask_dl]
    f_data[mask_dl] = (1.0 / (xd_l**2 - 1.0)) * (
        1.0
        - (2.0 / np.sqrt(1.0 - xd_l**2))
        * np.arctanh(np.sqrt((1.0 - xd_l) / (1.0 + xd_l)))
    )
    g_data[mask_dl] = (2.0 / xd_l**2) * (
        (2.0 / np.sqrt(1.0 - xd_l**2))
        * np.arctanh(np.sqrt((1.0 - xd_l) / (1.0 + xd_l)))
        + np.log(xd_l / 2.0)
    ) - f_data[mask_dl]

    mask_dm = (x_data >= 0.999) & (x_data <= 1.001)
    f_data[mask_dm] = 1.0 / 3.0
    g_data[mask_dm] = 2.0 * (1.0 + np.log(0.5)) - 1.0 / 3.0

    mask_dh = x_data > 1.001
    xd_h = x_data[mask_dh]
    f_data[mask_dh] = (1.0 / (xd_h**2 - 1.0)) * (
        1.0
        - (2.0 / np.sqrt(xd_h**2 - 1.0))
        * np.arctan(np.sqrt((xd_h - 1.0) / (xd_h + 1.0)))
    )
    g_data[mask_dh] = (2.0 / xd_h**2) * (
        (2.0 / np.sqrt(xd_h**2 - 1.0)) * np.arctan(np.sqrt((xd_h - 1.0) / (xd_h + 1.0)))
        + np.log(xd_h / 2.0)
    ) - f_data[mask_dh]
    ds_grid = 2.0 * rs[:, None] * rhos[:, None] * g_data

    return sigma_grid, ds_grid


def compute_stacked_profile_dsigma(
    scatter: float,
    alpha_c: float,
    alpha_b: float,
    f_mis: float,
    sigma_R: float,
    bin_index: int,
    model_state: dict[str, Any],
    return_components: bool = False,
) -> tuple[np.ndarray, ...]:
    """Compute model DeltaSigma with abundance matching, scaling, and miscentering.

    Parameters
    ----------
    scatter : float
        Scatter sigma in log10(M).
    alpha_c : float
        Concentration scaling factor c / c_Diemer19.
    alpha_b : float
        Halo bias scaling factor b / b_Tinker10.
    f_mis : float
        Miscentered cluster fraction.
    sigma_R : float
        Miscentering scale parameter (kpc/h).
    bin_index : int
        Index of the bin to compute.
    model_state : dict
        Precomputed state dictionary from build_profile_scatter_model.
    return_components : bool
        If True, return (ds_tot, ds_1h_cen, ds_1h_mis, ds_2h).

    Returns
    -------
    ndarray or tuple of ndarray
        Modeled DeltaSigma in Colossus units (M_sun h / kpc^2).
    """
    weights = compute_mass_weights(
        scatter=scatter,
        bin_index=bin_index,
        th_splines=model_state["th_splines"],
        m_dense=model_state["m_dense"],
        dndlnm_dense=model_state["dndlnm_dense"],
        dlogm=model_state["dlogm"],
    )

    c_arr = np.clip(alpha_c * model_state["c_base_dense"], 0.5, 30.0)
    sigma_grid, ds_grid = compute_vectorized_nfw_profiles(
        r200m_dense=model_state["r200m_dense"],
        c_arr=c_arr,
        rho_m=model_state["rho_m"],
        rp_dense=model_state["rp_dense"],
        rp_kpc_h=model_state["rp_kpc_h"],
    )

    sigma_1h_dense = np.dot(weights, sigma_grid)
    ds_1h_cen = np.dot(weights, ds_grid)
    ds_2h = (
        alpha_b * np.sum(weights * model_state["b_base_dense"]) * model_state["ds_xi"]
    )

    if f_mis > 0.0 and sigma_R > 0.0:
        rp_dense = model_state["rp_dense"]
        rp_kpc_h = model_state["rp_kpc_h"]
        log_spl = interp.InterpolatedUnivariateSpline(
            np.log(rp_dense),
            np.log(np.maximum(sigma_1h_dense, 1e-12)),
            ext=1,
        )
        sigma_1h_mis = fast_sigma_mis_gamma(
            rp_dense, sigma_R, log_spl, rp_dense[0], rp_dense[-1]
        )
        x_sig_spl = interp.InterpolatedUnivariateSpline(
            rp_dense, rp_dense * sigma_1h_mis, k=1
        )
        sig_mis_spl = interp.InterpolatedUnivariateSpline(rp_dense, sigma_1h_mis, k=1)
        ds_1h_mis = np.zeros_like(rp_kpc_h)
        for i, r_val in enumerate(rp_kpc_h):
            sigma_bar = (2.0 / r_val**2) * x_sig_spl.integral(0.0, r_val)
            ds_1h_mis[i] = sigma_bar - sig_mis_spl(r_val)
        ds_1h_final = (1.0 - f_mis) * ds_1h_cen + f_mis * ds_1h_mis
    else:
        ds_1h_final = ds_1h_cen
        ds_1h_mis = np.zeros_like(ds_1h_cen)

    ds_total = ds_1h_final + ds_2h

    if return_components:
        return (
            ds_total,
            (1.0 - f_mis) * ds_1h_cen,
            f_mis * ds_1h_mis,
            ds_2h,
        )
    return ds_total


def build_profile_scatter_model(
    rp_kpc_h: np.ndarray,
    z_lens: float,
    bin_counts: list[int],
    area_sq_deg: float,
    z_min: float = 0.19,
    z_max: float = 0.52,
    cosmology_name: str = "planck18",
    mass_def: str = "200m",
    conc_model: str = "diemer19",
    bias_model: str = "tinker10",
    mass_func_model: str = "tinker08",
) -> dict[str, Any]:
    """Precompute all grids, splines, and background states for profile scatter fitting.

    Parameters
    ----------
    rp_kpc_h : ndarray
        Projected radii of data points in kpc/h.
    z_lens : float
        Mean/median lens redshift.
    bin_counts : list of int
        Lens count for each bin [N_0, ...] or [N_0, N_1, N_2, N_3].
    area_sq_deg : float
        Survey sky area in square degrees.
    z_min, z_max : float
        Lens redshift bounds for survey volume calculation.
    cosmology_name : str
        Colossus cosmology identifier.
    mass_def, conc_model, bias_model, mass_func_model : str
        Model specifications for Colossus halo functions.

    Returns
    -------
    dict
        State dictionary containing all precomputed grids and splines.
    """
    cosmo = cosmology.setCosmology(cosmology_name)
    h = cosmo.h
    rho_m = cosmo.rho_m(z_lens)

    v_survey = compute_survey_volume(area_sq_deg, z_min, z_max, cosmology_name)
    cum_counts = np.cumsum(bin_counts)
    cum_n_obs = [count / v_survey for count in cum_counts]

    logger.info("Survey volume: %.2e (Mpc/h)^3", v_survey)
    logger.info("Cumulative number densities: %s", cum_n_obs)

    # Precompute HMF, concentration, and bias splines
    logm_grid = np.linspace(11.0, 16.5, 120)
    c_grid = np.array(
        [
            concentration.concentration(10**m, mass_def, z_lens, model=conc_model)
            for m in logm_grid
        ]
    )
    b_grid = np.array(
        [
            bias.haloBias(10**m, model=bias_model, z=z_lens, mdef=mass_def)
            for m in logm_grid
        ]
    )
    dndlnm_grid = np.array(
        [
            mass_function.massFunction(
                10**m, z_lens, mdef=mass_def, model=mass_func_model, q_out="dndlnM"
            )
            for m in logm_grid
        ]
    )

    c_spline = interp.InterpolatedUnivariateSpline(logm_grid, c_grid)
    b_spline = interp.InterpolatedUnivariateSpline(logm_grid, b_grid)
    hmf_spline = interp.InterpolatedUnivariateSpline(logm_grid, dndlnm_grid)

    # Precompute matter correlation function 2-halo base signal
    logger.info("Precomputing 2-halo base DeltaSigma...")
    ds_xi = compute_2halo_base_dsigma(rp_kpc_h, z_lens)

    # Precompute abundance matching threshold splines
    logger.info("Precomputing abundance matching threshold splines...")
    scatter_grid = np.linspace(0.02, 2.0, 50)
    th_splines = build_abundance_matching_splines(cum_n_obs, hmf_spline, scatter_grid)

    # Dense mass grid and dense projected radial grid
    m_dense = np.linspace(12.5, 15.5, 60)
    dlogm = m_dense[1] - m_dense[0]
    dndlnm_dense = hmf_spline(m_dense)
    c_base_dense = c_spline(m_dense)
    b_base_dense = b_spline(m_dense)

    r200m_dense = np.array([mass_so.M_to_R(10**m, z_lens, mass_def) for m in m_dense])
    rp_dense = np.logspace(-1, np.log10(np.max(rp_kpc_h) * 10), 120)

    return {
        "h": h,
        "z_lens": z_lens,
        "rho_m": rho_m,
        "v_survey": v_survey,
        "cum_n_obs": cum_n_obs,
        "th_splines": th_splines,
        "m_dense": m_dense,
        "dlogm": dlogm,
        "dndlnm_dense": dndlnm_dense,
        "c_base_dense": c_base_dense,
        "b_base_dense": b_base_dense,
        "r200m_dense": r200m_dense,
        "rp_kpc_h": np.asarray(rp_kpc_h),
        "rp_dense": rp_dense,
        "ds_xi": ds_xi,
    }


def compute_numerical_hessian(
    func: Any,
    x0: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Compute numerical Hessian matrix using finite central differences.

    Parameters
    ----------
    func : callable
        Function f(x) -> scalar.
    x0 : ndarray
        Evaluation point.
    eps : float
        Relative step size.

    Returns
    -------
    ndarray of shape (N, N)
        Estimated Hessian matrix.
    """
    n = len(x0)
    hessian = np.zeros((n, n))
    steps = np.maximum(np.abs(x0) * eps, 1e-4)

    f0 = func(x0)
    for i in range(n):
        x_plus = x0.copy()
        x_plus[i] += steps[i]
        f_plus = func(x_plus)

        x_minus = x0.copy()
        x_minus[i] -= steps[i]
        f_minus = func(x_minus)

        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)

        for j in range(i + 1, n):
            x_pp = x0.copy()
            x_pp[i] += steps[i]
            x_pp[j] += steps[j]

            x_pm = x0.copy()
            x_pm[i] += steps[i]
            x_pm[j] -= steps[j]

            x_mp = x0.copy()
            x_mp[i] -= steps[i]
            x_mp[j] += steps[j]

            x_mm = x0.copy()
            x_mm[i] -= steps[i]
            x_mm[j] -= steps[j]

            hessian[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (
                4.0 * steps[i] * steps[j]
            )
            hessian[j, i] = hessian[i, j]

    return hessian


def fit_profile_scatter_map(
    model_state: dict[str, Any],
    ds_obs_colossus: np.ndarray,
    cov_inv_colossus: np.ndarray,
    bin_index: int,
    f_mis: float = 0.25,
    sigma_r: float = 100.0,
) -> dict[str, Any]:
    """Run multi-start MAP optimization of scatter, alpha_c, alpha_b with fixed miscentering.

    Parameters
    ----------
    model_state : dict
        Precomputed model state from build_profile_scatter_model.
    ds_obs_colossus : ndarray
        Observed DeltaSigma in Colossus units.
    cov_inv_colossus : ndarray
        Inverse covariance matrix in Colossus units.
    bin_index : int
        Index of the bin being fitted.
    f_mis : float, default=0.25
        Fixed miscentered cluster fraction.
    sigma_r : float, default=100.0
        Fixed miscentering scale sigma_R in kpc/h.

    Returns
    -------
    dict
        Fitted parameters, 1-sigma errors, derived physical quantities, and diagnostics.
    """
    n_data = len(ds_obs_colossus)
    n_params = 3
    n_dof = n_data - n_params

    bounds = [
        (0.05, 1.5),  # scatter
        (0.2, 5.0),  # alpha_c
        (0.1, 5.0),  # alpha_b
    ]

    def neg_log_likelihood(theta: np.ndarray) -> float:
        scatter, alpha_c, alpha_b = theta
        try:
            model = compute_stacked_profile_dsigma(
                scatter=scatter,
                alpha_c=alpha_c,
                alpha_b=alpha_b,
                f_mis=f_mis,
                sigma_R=sigma_r,
                bin_index=bin_index,
                model_state=model_state,
            )
            residual = ds_obs_colossus - model
            return 0.5 * float(residual @ cov_inv_colossus @ residual)
        except Exception:
            return 1e30

    starting_points = [
        [0.35, 1.0, 1.0],
        [0.30, 1.2, 0.8],
        [0.40, 0.9, 0.7],
        [0.25, 1.1, 1.1],
    ]

    best_result = None
    best_fun = np.inf

    for x0 in starting_points:
        try:
            res = optimize.minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 120, "ftol": 1e-7},
            )
            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res
        except Exception as e:
            logger.warning("Optimization run failed for x0=%s: %s", x0, e)

    if best_result is None or not np.isfinite(best_fun):
        logger.error("All MAP optimization attempts failed.")
        return {
            "scatter": np.nan,
            "scatter_err": np.nan,
            "alpha_c": np.nan,
            "alpha_c_err": np.nan,
            "alpha_b": np.nan,
            "alpha_b_err": np.nan,
            "f_mis": f_mis,
            "sigma_R": sigma_r,
            "chi2": np.nan,
            "chi2_reduced": np.nan,
            "n_dof": n_dof,
            "mean_logm": np.nan,
            "mean_c": np.nan,
            "mean_b": np.nan,
            "success": False,
        }

    p_best = best_result.x
    scatter_best, ac_best, ab_best = p_best

    # Compute pure data chi2
    model_best = compute_stacked_profile_dsigma(
        scatter=scatter_best,
        alpha_c=ac_best,
        alpha_b=ab_best,
        f_mis=f_mis,
        sigma_R=sigma_r,
        bin_index=bin_index,
        model_state=model_state,
    )
    res_best = ds_obs_colossus - model_best
    chi2 = float(res_best @ cov_inv_colossus @ res_best)
    chi2_red = chi2 / n_dof if n_dof > 0 else np.nan

    # Calculate parameter errors via numerical Hessian of negative log-likelihood
    try:
        hess = compute_numerical_hessian(neg_log_likelihood, p_best)
        cov_params = np.linalg.pinv(hess)
        diag = np.diag(cov_params)
        errors = np.where(diag > 0, np.sqrt(diag), np.nan)
    except Exception as e:
        logger.warning("Hessian inversion failed: %s", e)
        errors = np.full(3, np.nan)

    # Compute derived physical quantities <logM>, <c>, <b>
    weights = compute_mass_weights(
        scatter=scatter_best,
        bin_index=bin_index,
        th_splines=model_state["th_splines"],
        m_dense=model_state["m_dense"],
        dndlnm_dense=model_state["dndlnm_dense"],
        dlogm=model_state["dlogm"],
    )
    mean_logm = float(np.sum(model_state["m_dense"] * weights))
    mean_c = float(ac_best * np.sum(model_state["c_base_dense"] * weights))
    mean_b = float(ab_best * np.sum(model_state["b_base_dense"] * weights))

    return {
        "scatter": float(scatter_best),
        "scatter_err": float(errors[0]),
        "alpha_c": float(ac_best),
        "alpha_c_err": float(errors[1]),
        "alpha_b": float(ab_best),
        "alpha_b_err": float(errors[2]),
        "f_mis": float(f_mis),
        "sigma_R": float(sigma_r),
        "chi2": float(chi2),
        "chi2_reduced": float(chi2_red),
        "n_dof": int(n_dof),
        "mean_logm": mean_logm,
        "mean_c": mean_c,
        "mean_b": mean_b,
        "success": bool(best_result.success),
    }


def load_dsigma_profile(
    fits_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load radial bins, dsigma, errors, covariance, and redshift from FITS file.

    Parameters
    ----------
    fits_path : str
        Path to dsigma FITS file.

    Returns
    -------
    rp_mpc : ndarray
    ds_data : ndarray
    ds_err : ndarray
    cov_data : ndarray
    z_lens : float
    """
    t = Table.read(fits_path)
    rp_mpc = np.asarray(t["rp"], dtype=float)
    ds_data = np.asarray(t["ds"], dtype=float)
    ds_err = (
        np.asarray(t["ds_err"], dtype=float)
        if "ds_err" in t.colnames
        else np.zeros_like(ds_data)
    )

    with fits.open(fits_path) as hdul:
        cov_data = hdul["JK_COV"].data.copy()

    z_lens = float(np.nanmedian(t["z_l"])) if "z_l" in t.colnames else 0.35
    return rp_mpc, ds_data, ds_err, cov_data, z_lens
