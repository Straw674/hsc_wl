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
    compute_stacked_dsigma,
    compute_survey_number_density,
)

logger = logging.getLogger(__name__)

__all__ = [
    "load_simulation_zero_scatter",
    "compute_colossus_zero_scatter",
    "get_theoretical_upper_limit",
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
    area_deg2: float = 170.0,
    z_min: float = 0.19,
    z_max: float = 0.52,
    z_lens: float = 0.35,
    top_n: int = 500,
    cosmology_name: str = "planck18",
) -> Table:
    """Compute the analytical Halo Model theoretical upper limit (sigma=0, f_mis=0).

    Uses Colossus with Tinker HMF, Diemer concentration, and linear matter
    correlation 2-halo term.

    Parameters
    ----------
    rp_eval : np.ndarray
        Projected radii (physical Mpc) to evaluate on.
    area_deg2 : float, default 170.0
        Survey area in square degrees.
    z_min, z_max : float
        Redshift range of the sample.
    z_lens : float, default 0.35
        Effective lens redshift.
    top_n : int, default 500
        Number of top objects in the volume.
    cosmology_name : str, default "planck18"
        Colossus cosmology model.

    Returns
    -------
    astropy.table.Table
        Table containing ``["rp", "ds", "ds_err"]`` (ds_err is 0.0 for analytic model).
    """
    from colossus.cosmology import cosmology

    cosmo = cosmology.setCosmology(cosmology_name)
    h = cosmo.h

    n_obs = compute_survey_number_density(
        area_sq_deg=area_deg2,
        z_min=z_min,
        z_max=z_max,
        n_obj=top_n,
        cosmology_name=cosmology_name,
    )

    rp_mpc = np.asarray(rp_eval, dtype=float)
    rp_kpc_h = rp_mpc * 1000.0 * h

    model_state = build_scatter_model(
        rp_kpc_h=rp_kpc_h,
        z_lens=z_lens,
        n_obs=n_obs,
        cosmology_name=cosmology_name,
    )

    # Evaluate at quasi-zero scatter (1e-3) and zero miscentering
    ds_colossus = compute_stacked_dsigma(
        scatter=0.001,
        f_mis=0.0,
        sigma_R=0.0,
        model_state=model_state,
    )

    # Convert from Colossus units (h M_sun / kpc^2) to physical (M_sun / pc^2)
    ds_phys = ds_colossus / (1e6 / h)
    ds_err = np.zeros_like(ds_phys)

    tbl = Table({"rp": rp_mpc, "ds": ds_phys, "ds_err": ds_err})
    tbl.meta["label"] = f"Colossus Halo Model Top-{top_n} (sigma=0)"
    return tbl


def get_theoretical_upper_limit(
    root_path: Path,
    nbins: Literal["1bin", "4bin"] = "1bin",
    top_n: int = 500,
    source: Literal["simulation", "colossus"] = "simulation",
    rp_eval: np.ndarray | None = None,
    area_deg2: float = 170.0,
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
    top_n : int, default 500
        Top N count for 1-bin mode.
    source : {"simulation", "colossus"}, default "simulation"
        Source of the upper limit curve: N-body simulation or analytical Halo Model.
    rp_eval : np.ndarray or None
        Projected radial grid (Mpc) to interpolate or evaluate onto.
    area_deg2, z_min, z_max, z_lens : float
        Cosmological volume and redshift parameters used for Colossus.

    Returns
    -------
    list of astropy.table.Table
        Per-bin theoretical limit tables with columns ``["rp", "ds", "ds_err"]``.
    """
    if source == "simulation":
        return load_simulation_zero_scatter(
            root_path=root_path,
            nbins=nbins,
            top_n=top_n,
            rp_eval=rp_eval,
        )
    elif source == "colossus":
        if rp_eval is None:
            # Default to standard HSC radial grid (0.1 to 20 Mpc)
            rp_eval = np.logspace(np.log10(0.1), np.log10(20.0), 20)
        tbl = compute_colossus_zero_scatter(
            rp_eval=rp_eval,
            area_deg2=area_deg2,
            z_min=z_min,
            z_max=z_max,
            z_lens=z_lens,
            top_n=top_n,
        )
        return [tbl]
    else:
        raise ValueError(
            f"Unknown source: {source!r}. Must be 'simulation' or 'colossus'."
        )
