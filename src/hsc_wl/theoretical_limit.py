"""Theoretical upper limit and zero-scatter weak lensing profiles.

This module provides utilities to load or compute the ideal theoretical
maximum stacked weak-lensing signal (zero mass-observable scatter,
zero miscentering: sigma = 0, f_mis = 0) from:
1. N-body cosmological simulations (MDPL2 / SMDPL);
2. Analytical halo model (Colossus Tinker HMF + NFW 1-halo + 2-halo).

Supports both 1-bin (cumulative top-N) and 4-bin (differential richness/mass)
configurations.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d

from hsc_wl.scatter_fit import (
    build_scatter_model,
    compute_2halo_base_dsigma,
    compute_stacked_dsigma,
    compute_survey_number_density,
)

logger = logging.getLogger(__name__)

__all__ = [
    "load_simulation_zero_scatter",
    "compute_colossus_zero_scatter",
    "get_theoretical_upper_limit",
    "export_theoretical_limit_outputs",
    "ensure_theoretical_limit_outputs",
]


def load_simulation_zero_scatter(
    root_path: Path,
    nbins: Literal["1bin", "4bin"] = "1bin",
    top_n: int = 500,
    sim_rel_path: str = "libs/jianbing/data/simulation/sim_mdpl2_cen_dsig.fits",
    rp_eval: np.ndarray | None = None,
) -> list[Table]:
    """Load the zero-scatter (sigma = 0) profile from N-body simulation data.

    Parameters
    ----------
    root_path : Path
        Project root path.
    nbins : {"1bin", "4bin"}, default "1bin"
        Number of bins: "1bin" for cumulative top-N, "4bin" for 4 differential bins.
    top_n : int, default 500
        Number of top objects for 1-bin mode (default 500 matching S16a/HectoMAP standard).
    sim_rel_path : str
        Relative path to the simulation templates FITS file.
    rp_eval : np.ndarray or None
        Optional radial grid (physical Mpc) to interpolate onto.

    Returns
    -------
    list of astropy.table.Table
        List of result tables (1 table for 1bin, 4 tables for 4bin), each containing
        columns ``["rp", "ds", "ds_err"]`` in physical units (Mpc, M_sun/pc^2).
    """
    sim_path = Path(root_path) / sim_rel_path
    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_path}")

    sim_table = Table.read(sim_path)
    zero_scatter_rows = sim_table[sim_table["scatter"] == 0.0]

    if len(zero_scatter_rows) == 0:
        # Fallback to minimum available scatter
        min_scatter = np.min(sim_table["scatter"])
        logger.warning(
            "Exact scatter=0.0 not found in %s; using min scatter=%.2f",
            sim_path.name,
            min_scatter,
        )
        zero_scatter_rows = sim_table[sim_table["scatter"] == min_scatter]

    zero_scatter_rows.sort("bin")

    if nbins == "4bin":
        tables = []
        for b_idx in range(4):
            bin_row = zero_scatter_rows[zero_scatter_rows["bin"] == b_idx]
            if len(bin_row) == 0:
                raise ValueError(
                    f"Bin {b_idx} not found in simulation zero-scatter rows."
                )
            rp = np.asarray(bin_row["r_mpc"][0], dtype=float)
            ds = np.asarray(bin_row["dsig"][0], dtype=float)
            ds_err = np.asarray(bin_row["dsig_err"][0], dtype=float)

            if rp_eval is not None:
                f_ds = interp1d(rp, ds, kind="cubic", fill_value="extrapolate")
                f_err = interp1d(rp, ds_err, kind="cubic", fill_value="extrapolate")
                rp_out = np.asarray(rp_eval, dtype=float)
                ds_out = f_ds(rp_out)
                ds_err_out = np.clip(f_err(rp_out), 0.0, None)
            else:
                rp_out, ds_out, ds_err_out = rp, ds, ds_err

            tbl = Table({"rp": rp_out, "ds": ds_out, "ds_err": ds_err_out})
            tbl.meta["label"] = f"MDPL2 (sigma=0, bin{b_idx})"
            tables.append(tbl)
        return tables

    # 1-bin mode: Cumulative Top-N weighting across simulation bins
    # Bin 0: rank 0..1334 (HSC 0..47)
    # Bin 1: rank 1334..6277 (HSC 47..221)
    # Bin 2: rank 6277..22523 (HSC 221..793)
    # Bin 3: rank 22523..50869 (HSC 793..1791)
    hsc_bounds = [0] + [int(row["hsc_n_upper"]) for row in zero_scatter_rows]
    r_sim = np.asarray(zero_scatter_rows["r_mpc"][0], dtype=float)

    # Compute weights for each differential bin contributing to top_n
    weights = []
    ds_bins = []
    err_bins = []

    remaining = top_n
    for b_idx in range(len(zero_scatter_rows)):
        bin_cap = hsc_bounds[b_idx + 1] - hsc_bounds[b_idx]
        take = min(remaining, bin_cap)
        if take <= 0:
            break
        weights.append(take)
        ds_bins.append(np.asarray(zero_scatter_rows[b_idx]["dsig"], dtype=float))
        err_bins.append(np.asarray(zero_scatter_rows[b_idx]["dsig_err"], dtype=float))
        remaining -= take

    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError(f"Invalid top_n={top_n}")

    w_arr = np.array(weights)[:, None] / total_weight
    ds_cum = np.sum(w_arr * np.array(ds_bins), axis=0)
    ds_err_cum = np.sqrt(np.sum((w_arr**2) * (np.array(err_bins) ** 2), axis=0))

    if rp_eval is not None:
        f_ds = interp1d(r_sim, ds_cum, kind="cubic", fill_value="extrapolate")
        f_err = interp1d(r_sim, ds_err_cum, kind="cubic", fill_value="extrapolate")
        rp_out = np.asarray(rp_eval, dtype=float)
        ds_out = f_ds(rp_out)
        ds_err_out = np.clip(f_err(rp_out), 0.0, None)
    else:
        rp_out, ds_out, ds_err_out = r_sim, ds_cum, ds_err_cum

    tbl = Table({"rp": rp_out, "ds": ds_out, "ds_err": ds_err_out})
    tbl.meta["label"] = f"MDPL2 Top-{top_n} (sigma=0)"
    return [tbl]


def compute_colossus_zero_scatter(
    rp_eval: np.ndarray,
    nbins: Literal["1bin", "4bin"] = "1bin",
    area_deg2: float = 170.0,
    z_min: float = 0.19,
    z_max: float = 0.52,
    z_lens: float = 0.35,
    top_n: int = 500,
    top_counts: tuple[int, ...] = (53, 196, 660, 1159),
    cosmology_name: str = "planck18",
) -> list[Table]:
    """Compute the analytical Halo Model theoretical upper limit (sigma=0, f_mis=0).

    Supports both 1-bin (cumulative top-N) and 4-bin (differential counts).

    Parameters
    ----------
    rp_eval : np.ndarray
        Projected radii (physical Mpc) to evaluate on.
    nbins : {"1bin", "4bin"}, default "1bin"
        Binning mode.
    area_deg2 : float, default 170.0
        Survey area in square degrees.
    z_min, z_max : float
        Redshift range of the sample.
    z_lens : float, default 0.35
        Effective lens redshift.
    top_n : int, default 500
        Number of top objects in the volume (1-bin mode).
    top_counts : tuple of int, default (53, 196, 660, 1159)
        Counts for 4-bin mode.
    cosmology_name : str, default "planck18"
        Colossus cosmology model.

    Returns
    -------
    list of astropy.table.Table
        List of Tables containing ``["rp", "ds", "ds_err"]``.
    """
    from colossus.cosmology import cosmology
    from colossus.halo import concentration, profile_nfw
    from colossus.lss import bias, mass_function

    cosmo = cosmology.setCosmology(cosmology_name)
    h = cosmo.h

    rp_mpc = np.asarray(rp_eval, dtype=float)
    rp_kpc_h = rp_mpc * 1000.0 * h

    if nbins == "1bin":
        n_obs = compute_survey_number_density(
            area_sq_deg=area_deg2,
            z_min=z_min,
            z_max=z_max,
            n_obj=top_n,
            cosmology_name=cosmology_name,
        )

        model_state = build_scatter_model(
            rp_kpc_h=rp_kpc_h,
            z_lens=z_lens,
            n_obs=n_obs,
            cosmology_name=cosmology_name,
        )

        ds_colossus = compute_stacked_dsigma(
            scatter=0.001,
            f_mis=0.0,
            sigma_R=0.0,
            model_state=model_state,
        )

        ds_phys = ds_colossus / (1e6 / h)
        ds_err = np.zeros_like(ds_phys)

        tbl = Table({"rp": rp_mpc, "ds": ds_phys, "ds_err": ds_err})
        tbl.meta["label"] = f"Colossus Halo Model Top-{top_n} (sigma=0)"
        return [tbl]

    # 4-bin differential calculation
    cum_counts = np.cumsum(top_counts)
    cum_n_obs = [
        compute_survey_number_density(area_deg2, z_min, z_max, c, cosmology_name)
        for c in cum_counts
    ]

    logm_grid = np.linspace(12.0, 16.5, 200)
    c_grid = [
        concentration.concentration(10**m, "200m", z_lens, model="diemer19")
        for m in logm_grid
    ]
    b_grid = [
        bias.haloBias(10**m, model="tinker10", z=z_lens, mdef="200m") for m in logm_grid
    ]
    dndlnm_grid = [
        mass_function.massFunction(
            10**m, z_lens, mdef="200m", model="tinker08", q_out="dndlnM"
        )
        for m in logm_grid
    ]

    c_spline = interp1d(logm_grid, c_grid, kind="cubic", fill_value="extrapolate")
    b_spline = interp1d(logm_grid, b_grid, kind="cubic", fill_value="extrapolate")
    dndlnm_spline = interp1d(
        logm_grid, dndlnm_grid, kind="cubic", fill_value="extrapolate"
    )

    logm_dense = np.linspace(12.0, 16.5, 500)
    dlogm = logm_dense[1] - logm_dense[0]
    dndlogm = dndlnm_spline(logm_dense) * np.log(10)
    n_cum_dense = np.cumsum(dndlogm[::-1] * dlogm)[::-1]

    th_logm = [16.5]
    for n_target in cum_n_obs:
        idx = int(np.argmin(np.abs(n_cum_dense - n_target)))
        th_logm.append(float(logm_dense[idx]))

    ds_xi = compute_2halo_base_dsigma(rp_kpc_h, z_lens)
    tables = []

    for b_idx in range(4):
        m_hi, m_lo = th_logm[b_idx], th_logm[b_idx + 1]
        mask = (logm_dense >= m_lo) & (logm_dense <= m_hi)
        m_bin = logm_dense[mask]
        w_bin = dndlogm[mask] * dlogm
        w_bin /= np.sum(w_bin)

        ds_1h = np.zeros_like(rp_kpc_h)
        ds_2h = np.zeros_like(rp_kpc_h)
        for m, w in zip(m_bin, w_bin):
            c = float(c_spline(m))
            b = float(b_spline(m))
            p = profile_nfw.NFWProfile(M=10**m, c=c, z=z_lens, mdef="200m")
            ds_1h += w * p.deltaSigma(rp_kpc_h)
            ds_2h += w * b * ds_xi

        ds_tot_phys = (ds_1h + ds_2h) / (1e6 / h)
        ds_err = np.zeros_like(ds_tot_phys)

        tbl = Table({"rp": rp_mpc, "ds": ds_tot_phys, "ds_err": ds_err})
        tbl.meta["label"] = f"Colossus Halo Model Bin {b_idx} (sigma=0)"
        tables.append(tbl)

    return tables


def get_theoretical_upper_limit(
    root_path: Path,
    nbins: Literal["1bin", "4bin"] = "1bin",
    top_n: int = 100,
    source: Literal["simulation", "colossus"] = "simulation",
    rp_eval: np.ndarray | None = None,
    area_deg2: float = 50.8824,
    z_min: float = 0.19,
    z_max: float = 0.52,
    z_lens: float = 0.35,
) -> list[Table]:
    """Unified entry point to retrieve theoretical upper limit profiles.

    Parameters
    ----------
    root_path : Path
        Project root path.
    nbins : {"1bin", "4bin"}, default "1bin"
        Binning mode.
    top_n : int, default 100
        Top N count for 1-bin mode.
    source : {"simulation", "colossus"}, default "simulation"
        Source of the upper limit curve: N-body simulation or analytical Halo Model.
    rp_eval : np.ndarray or None
        Projected radial grid (Mpc) to interpolate or evaluate onto.
    area_deg2, z_min, z_max, z_lens : float
        Cosmological volume and redshift parameters. For simulation mode in 1bin,
        area_deg2 is used to scale top_n to match comoving number density against
        the MDPL2 calibration volume (S16A area = 137.9 deg^2).

    Returns
    -------
    list of astropy.table.Table
        Per-bin theoretical limit tables with columns ``["rp", "ds", "ds_err"]``.
    """
    if source == "simulation":
        sim_top_n = top_n
        if nbins == "1bin" and area_deg2 is not None:
            sim_area = 137.9  # S16A calibration area of sim_mdpl2_cen_dsig.fits
            sim_top_n = int(round(top_n * (sim_area / area_deg2)))
            logger.info(
                "Scaling MDPL2 top_n from %d (area=%.2f) to %d (calib_area=%.2f) for density matching",
                top_n,
                area_deg2,
                sim_top_n,
                sim_area,
            )
        tbls = load_simulation_zero_scatter(
            root_path=root_path,
            nbins=nbins,
            top_n=sim_top_n,
            rp_eval=rp_eval,
        )
        if nbins == "1bin":
            for t in tbls:
                t.meta["label"] = f"MDPL2 Top-{top_n} (sigma=0)"
        return tbls
    elif source == "colossus":
        if rp_eval is None:
            # Default to standard HSC radial grid (0.1 to 20 Mpc)
            rp_eval = np.logspace(np.log10(0.1), np.log10(20.0), 20)
        return compute_colossus_zero_scatter(
            rp_eval=rp_eval,
            nbins=nbins,
            area_deg2=area_deg2,
            z_min=z_min,
            z_max=z_max,
            z_lens=z_lens,
            top_n=top_n,
        )
    else:
        raise ValueError(
            f"Unknown source: {source!r}. Must be 'simulation' or 'colossus'."
        )


def export_theoretical_limit_outputs(
    root_path: Path,
    catalog_id: str = "ideal_mdpl2",
    nbins: Literal["1bin", "4bin"] = "1bin",
    version: str = "Y3",
    top_n: int = 100,
    area_deg2: float = 50.8824,
    rp_min: float = 0.10,
    rp_max: float = 20.0,
    n_rp_bins: int = 11,
    overwrite: bool = True,
) -> list[Path]:
    """Export theoretical upper limit profiles to standard output FITS files.

    Writes FITS files with HDU 'PROFILE' and 'JK_COV' under:
    ``output/{catalog_id}/{nbins}/{version}/dsigma/hsc_hsc_bin{i}.fits``.

    Parameters
    ----------
    root_path : Path
        Project root path.
    catalog_id : str
        Catalog identifier, e.g. "ideal_mdpl2" or "ideal_colossus".
    nbins : {"1bin", "4bin"}
        Binning mode.
    version : str
        Source version (e.g. "Y3", "Y1").
    top_n : int, default 100
        Top-N count for 1-bin mode.
    area_deg2 : float, default 50.8824
        Survey area in square degrees.
    rp_min, rp_max, n_rp_bins : float, int
        Radial binning matching RPConfig.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    list of Path
        List of written FITS file paths.
    """
    from astropy.io import fits as fits_io

    source = "simulation" if "mdpl" in catalog_id.lower() else "colossus"
    rp_edges = np.logspace(np.log10(rp_min), np.log10(rp_max), n_rp_bins + 1)
    rp_lo = rp_edges[:-1]
    rp_hi = rp_edges[1:]
    rp_mid = np.sqrt(rp_lo * rp_hi)

    tables = get_theoretical_upper_limit(
        root_path=root_path,
        nbins=nbins,
        top_n=top_n,
        source=source,
        rp_eval=rp_mid,
        area_deg2=area_deg2,
    )

    out_dir = Path(root_path) / f"output/{catalog_id}/{nbins}/{version}/dsigma"
    out_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for i, tbl in enumerate(tables):
        out_table = Table()
        out_table["rp_min"] = rp_lo
        out_table["rp_max"] = rp_hi
        out_table["rp"] = rp_mid
        out_table["ds"] = np.asarray(tbl["ds"], dtype=float)
        out_table["ds_err"] = (
            np.asarray(tbl["ds_err"], dtype=float)
            if "ds_err" in tbl.colnames
            else np.zeros_like(out_table["ds"])
        )
        out_table["ds_raw"] = out_table["ds"]
        out_table["ds_r"] = np.zeros_like(out_table["ds"])
        out_table["z_l"] = np.full_like(out_table["ds"], 0.35)

        cov = np.diag(out_table["ds_err"] ** 2)

        hdul = fits_io.HDUList(
            [
                fits_io.PrimaryHDU(),
                fits_io.BinTableHDU(out_table, name="PROFILE"),
                fits_io.ImageHDU(cov, name="JK_COV"),
            ]
        )
        out_path = out_dir / f"hsc_hsc_bin{i}.fits"
        hdul.writeto(out_path, overwrite=overwrite)
        written_paths.append(out_path)

    return written_paths


def ensure_theoretical_limit_outputs(root_path: Path, overwrite: bool = False) -> None:
    """Ensure standard theoretical upper limit FITS and prepare tables exist."""
    top_counts_4bin = (53, 196, 660, 1159)
    total_4bin = sum(top_counts_4bin)

    for cat_id in ["ideal_mdpl2", "ideal_colossus"]:
        for nbins in ["1bin", "4bin"]:
            # 1. Ensure prepare lens table exists so scripts like fit_custom_scatter work seamlessly
            prep_dir = Path(root_path) / f"output/{cat_id}/{nbins}/prepare"
            prep_dir.mkdir(parents=True, exist_ok=True)
            lens_fits = prep_dir / f"{cat_id}_{nbins}_lenses.fits"
            rand_fits = prep_dir / f"{cat_id}_{nbins}_randoms.fits"

            n_obj = 100 if nbins == "1bin" else total_4bin
            if overwrite or not lens_fits.exists():
                t_lens = Table()
                t_lens["ra"] = np.linspace(200.0, 250.0, n_obj)
                t_lens["dec"] = np.linspace(42.0, 44.5, n_obj)
                t_lens["z"] = np.full(n_obj, 0.35)
                t_lens["wsys"] = np.ones(n_obj, dtype=float)
                if nbins == "1bin":
                    t_lens["bin_id"] = np.zeros(n_obj, dtype=int)
                else:
                    b_ids = []
                    for b_idx, c in enumerate(top_counts_4bin):
                        b_ids.extend([b_idx] * c)
                    t_lens["bin_id"] = np.array(b_ids)
                t_lens.write(lens_fits, overwrite=True)
                t_lens.write(rand_fits, overwrite=True)

            # 2. Ensure dsigma outputs exist
            for ver in ["Y3", "Y1"]:
                out_dir = Path(root_path) / f"output/{cat_id}/{nbins}/{ver}/dsigma"
                if overwrite or not (out_dir / "hsc_hsc_bin0.fits").exists():
                    export_theoretical_limit_outputs(
                        root_path=root_path,
                        catalog_id=cat_id,
                        nbins=nbins,
                        version=ver,
                        top_n=100,
                        area_deg2=50.8824,
                        overwrite=True,
                    )
