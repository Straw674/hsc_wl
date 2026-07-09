"""Top-level pipeline for evaluating a cluster catalog via weak lensing.

This module orchestrates the full ``prepare → dsigma → scatter-fit``
pipeline and exposes a single :func:`evaluate_cluster_catalog` function
that the cluster-finder optimisation loop can call.

Typical usage from the cluster-finder repository::

    from hsc_wl.pipeline import (
        evaluate_cluster_catalog,
        load_random_catalog,
        load_source_catalog,
    )

    # One-time setup (expensive – cache the results)
    source_data = load_source_catalog()
    random_table = load_random_catalog("data/random_hectomap.fits")

    # Per-iteration call (fast)
    metrics = evaluate_cluster_catalog(
        cluster_df,
        source_data=source_data,
        random_table=random_table,
        catalog_config={...},
        ...
    )
    loss = metrics["scatter"]
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table
from dsigma.helpers import dsigma_table

from hsc_wl.config import (
    BinningConfig,
    ColumnMapping,
    LensCatalogConfig,
    WLConfig,
    resolve_binning,
)
from hsc_wl.prepare import prepare_lens_random_tables
from hsc_wl.scatter_fit import (
    build_scatter_model,
    compute_survey_number_density,
    convert_dsigma_to_colossus_units,
    fit_scatter_map,
)
from hsc_wl.wl_compute import (
    compute_single_bin_profile,
    load_and_prepare_source,
)

logger = logging.getLogger(__name__)


def lens_config_from_dict(d: dict) -> LensCatalogConfig:
    """Build a :class:`LensCatalogConfig` from a legacy ``CATALOG_SOURCES`` dict.

    This adapter preserves backwards compatibility for callers (e.g. the
    cluster-finder optimisation loop) that still pass a plain dict.
    """
    cols = d["columns"]
    rr = d.get("redshift_range")
    return LensCatalogConfig(
        label=d["label"],
        lens_path=d["lens_path"],
        random_path=d["random_path"],
        columns=ColumnMapping(
            col_rank=cols["col_rank"],
            ra=cols["ra"],
            dec=cols["dec"],
            z=cols["z"],
        ),
        redshift_range=tuple(rr) if rr else (0.0, 1.0),
        top_counts_factor=d.get("top_counts_factor", 1.0),
        lens_format=d.get("lens_format"),
        random_format=d.get("random_format"),
        ra_range=tuple(d["ra_range"]) if d.get("ra_range") else None,
        dec_range=tuple(d["dec_range"]) if d.get("dec_range") else None,
    )


# ---------------------------------------------------------------------------
# One-time loaders (cache across optimisation iterations)
# ---------------------------------------------------------------------------

# Default Y3 source catalog paths
_DEFAULT_SOURCE_FILE = "/Users/xinq/dev/repos/hsc_wl/data/hsc_y3.fits"
_DEFAULT_SOURCE_NZ_FILE = "/Users/xinq/dev/repos/hsc_wl/data/nz.fits"


def load_source_catalog(
    source_file=_DEFAULT_SOURCE_FILE,
    source_version="Y3",
    source_survey="hsc",
    source_nz_file=_DEFAULT_SOURCE_NZ_FILE,
    source_calib_file=None,
):
    """Load and prepare the source catalog (call once, reuse across iterations).

    Parameters
    ----------
    source_file : str or Path
        Path to the source catalog FITS file.
    source_version : str
        Source catalog version (``'Y3'`` or ``'Y1'``).
    source_survey : str
        Survey name for dsigma (default ``'hsc'``).
    source_nz_file : str or Path or None
        Path to the n(z) file (required for Y3 tomography).
    source_calib_file : str or Path or None
        Path to the calibration catalog (for Y1/S16A).

    Returns
    -------
    dict
        Contains ``'table_s'``, ``'table_c'``, ``'table_n'``,
        ``'rp_bins'``, ``'source_version'``.
    """
    table_s, table_c, table_n, rp_bins = load_and_prepare_source(
        source_file=source_file,
        source_version=source_version,
        source_survey=source_survey,
        source_nz_file=source_nz_file,
        source_calib_file=source_calib_file,
    )
    logger.info("Source catalog loaded: %d sources", len(table_s))
    return {
        "table_s": table_s,
        "table_c": table_c,
        "table_n": table_n,
        "rp_bins": rp_bins,
        "source_version": source_version,
    }


def load_simulation_catalog(
    sim_file="/Users/xinq/dev/repos/hsc_wl/libs/jianbing/data/simulation/sim_mdpl2_cen_dsig.fits",
):
    """Load the simulation catalog (call once, reuse across iterations)."""
    table = Table.read(sim_file)
    logger.info("Simulation catalog loaded: %d profiles", len(table))
    return table


def load_random_catalog(random_file):
    """Load the random-point catalog (call once, reuse across iterations).

    Parameters
    ----------
    random_file : str or Path
        Path to the random catalog FITS file.

    Returns
    -------
    astropy.table.Table
        Random-point catalog.
    """
    table = Table.read(random_file)
    logger.info("Random catalog loaded: %d points", len(table))
    return table


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def compute_cluster_catalog_dsigma(
    cluster_df,
    source_data,
    random_table,
    catalog_config,
    *,
    sim_cat=None,
    skip_map_fit=False,
    # Binning
    binning_mode="top_n",
    top_n=500,
    top_selection_order="desc",
    top_counts=None,
    col_rank_edges_richness=None,
    col_rank_edges_mass=None,
    random_multiplier=20,
    rng_seed=None,
    # WL computation
    rp_min=0.10,
    rp_max=20.0,
    n_rp_bins=11,
    n_jackknife=100,
    n_jobs=12,
    comoving=False,
    lens_source_cut=0.1,
    corrections=None,
    # Scatter fitting
    cosmology_name="planck18",
    mass_def="200m",
    conc_model="diemer19",
    bias_model="tinker10",
    mass_func_model="tinker08",
    area_sq_deg=72.0,
    z_min=0.1,
    z_max=0.6,
):
    """Run the full weak-lensing evaluation pipeline on a cluster catalog.

    This is the main entry point for the cluster-finder optimisation loop.
    It executes: ``prepare`` → ``dsigma`` → ``scatter fit (MAP)`` and
    returns a metrics dictionary.

    Parameters
    ----------
    cluster_df : pandas.DataFrame
        Cluster catalog with columns matching ``catalog_config['columns']``.
    source_data : dict
        Pre-loaded source catalog from :func:`load_source_catalog`.
    random_table : astropy.table.Table
        Pre-loaded random catalog from :func:`load_random_catalog`.
    catalog_config : dict
        Same structure as a ``CATALOG_SOURCES`` entry in
        ``prepare_lens_and_random.py``.  Required keys: ``'label'``,
        ``'columns'`` (with ``'col_rank'``, ``'ra'``, ``'dec'``, ``'z'``).
        Optional: ``'redshift_range'``, ``'ra_range'``, ``'dec_range'``,
        ``'top_counts_factor'``.

    Returns
    -------
    dict
        Metrics from the MAP scatter fit::

            {
                'scatter': float,
                'f_mis': float,
                'sigma_R': float,
                'chi2_reduced': float,
                'mean_logm': float,
                'n_dof': int,
                'success': bool,
                'n_lens': int,
                'z_median': float,
            }
    """
    # ------------------------------------------------------------------
    # Step 1: Prepare lens/random tables (in-memory)
    # ------------------------------------------------------------------
    lens_catalog = Table.from_pandas(cluster_df)

    lens_cfg = lens_config_from_dict(catalog_config)

    try:
        from hsc_wl.coverage import resolve_area_and_factor

        _, top_counts_factor = resolve_area_and_factor(lens_cfg)
    except (FileNotFoundError, ImportError):
        top_counts_factor = catalog_config.get("top_counts_factor", 1.0)

    binning = resolve_binning(
        BinningConfig(
            mode=binning_mode,
            top_counts=tuple(top_counts or ()),
            top_n=top_n,
            edges_richness=tuple(col_rank_edges_richness or ()),
            edges_mass=tuple(col_rank_edges_mass or ()),
            selection_order=top_selection_order,
        ),
        top_counts_factor,
    )

    bin_results = prepare_lens_random_tables(
        lens_catalog=lens_catalog,
        random_catalog=random_table,
        catalog_config=lens_cfg,
        binning=binning,
        random_multiplier=random_multiplier,
        rng_seed=rng_seed,
    )

    if not bin_results:
        logger.error("No valid objects left to prepare.")
        return {
            "scatter": np.nan,
            "f_mis": np.nan,
            "sigma_R": np.nan,
            "chi2_reduced": np.nan,
            "mean_logm": np.nan,
            "n_dof": 0,
            "success": False,
            "n_lens": 0,
            "z_median": np.nan,
        }

    global_lens_table = bin_results["global_lens_table"]
    global_random_table = bin_results["global_random_table"]
    bin_metadata = bin_results["bin_metadata"]

    table_s = source_data["table_s"]
    table_c = source_data["table_c"]
    table_n = source_data["table_n"]
    source_version = source_data["source_version"]
    rp_bins = np.logspace(np.log10(rp_min), np.log10(rp_max), n_rp_bins + 1)
    z_bins = np.array([z_min, z_max])

    if sim_cat is not None:
        from jianbing import scatter

        from hsc_wl.wl_compute import precompute_catalogs

        obs = Table()
        bin_ids, ds_list, ds_err_list, jk_cov_list = [], [], [], []
        rp_mpc = None

        total_n_lens = 0
        z_median_sum = 0.0

        global_table_l = dsigma_table(
            global_lens_table, "lens", z="z", ra="ra", dec="dec", w_sys=1.0
        )
        global_table_r = dsigma_table(
            global_random_table, "lens", z="z", ra="ra", dec="dec", w_sys=1.0
        )

        if "bin_id" not in global_table_l.colnames:
            global_table_l["bin_id"] = global_lens_table["bin_id"]
        if "bin_id" not in global_table_r.colnames:
            global_table_r["bin_id"] = global_random_table["bin_id"]

        logger.info(
            "Running global precompute across all %d objects...", len(global_table_l)
        )
        global_table_l, global_table_r = precompute_catalogs(
            global_table_l,
            global_table_r,
            table_s,
            table_c,
            table_n,
            rp_bins,
            comoving,
            lens_source_cut,
            n_jobs,
        )
        logger.info("Global precompute completed.")

        for bmeta in bin_metadata:
            bid = bmeta["bin_id"]
            n_lens = bmeta["n_lens"]
            total_n_lens += n_lens

            # Extract this bin's precomputed rows
            table_l = global_table_l[global_table_l["bin_id"] == bid]
            table_r = global_table_r[global_table_r["bin_id"] == bid]

            profile_result = compute_single_bin_profile(
                table_l=table_l,
                table_r=table_r,
                table_s=table_s,
                table_c=table_c,
                table_n=table_n,
                rp_bins=rp_bins,
                z_bins=z_bins,
                n_jackknife=n_jackknife,
                n_jobs=n_jobs,
                comoving=comoving,
                lens_source_cut=lens_source_cut,
                corrections=corrections,
                source_version=source_version,
            )

            result_table = profile_result["result_table"]
            z_median_sum += profile_result["z_median"] * n_lens

            if rp_mpc is None:
                rp_mpc = np.asarray(result_table["rp"])

            bin_ids.append(bid + 1)
            ds_list.append(np.asarray(result_table["ds"]))
            ds_err_list.append(np.asarray(result_table["ds_err"]))
            jk_cov_list.append(profile_result["jk_cov"])

        # Inject zero-signal profiles for completely empty bins
        expected_n_bins = len(top_counts) if top_counts else 4
        while len(bin_ids) < expected_n_bins:
            missing_bin_id = len(bin_ids) + 1
            logger.warning(
                "Bin %d is completely empty. Injecting zero signal for penalty.",
                missing_bin_id,
            )
            bin_ids.append(missing_bin_id)
            ds_list.append(np.zeros(len(rp_mpc)))
            # Use a realistic typical error bar (5.0) instead of borrowing potentially huge errors
            # to ensure empty bins correctly and robustly reflect in a large chi2 penalty.
            ds_err_list.append(np.full(len(rp_mpc), 5.0))
            jk_cov_list.append(np.diag(np.full(len(rp_mpc), 25.0)))

        obs["bin_id"] = bin_ids
        obs["dsigma"] = ds_list
        obs["dsig_err_jk"] = ds_err_list
        obs["dsig_err_bt"] = ds_err_list
        obs["dsig_cov_jk"] = jk_cov_list
        obs["dsig_cov_bt"] = jk_cov_list
        obs.meta["r_mpc"] = rp_mpc

        custom_sum = scatter.compare_model_dsigma(
            obs, sim_cat, model_err=False, poly=True, verbose=False
        )

        sum_min_chi2 = 0.0
        dof_per_bin = len(rp_mpc)
        total_dof = 0
        sig_med_list = []

        for row in custom_sum:
            sig_med = row["sig_med_jk"]
            if not np.isnan(sig_med):
                sig_med_list.append(sig_med)

            min_chi2 = np.nanmin(row["chi2_jk"])
            if not np.isnan(min_chi2):
                sum_min_chi2 += min_chi2
                total_dof += dof_per_bin

        combined_scatter = np.mean(sig_med_list) if len(sig_med_list) > 0 else np.nan
        combined_chi2_reduced = sum_min_chi2 / total_dof if total_dof > 0 else np.nan
        avg_z_median = z_median_sum / total_n_lens if total_n_lens > 0 else np.nan

        logger.info(
            "Multi-bin pipeline complete: scatter=%.3f, chi2/dof=%.3f",
            combined_scatter,
            combined_chi2_reduced,
        )

        return {
            "scatter": combined_scatter,
            "chi2_reduced": combined_chi2_reduced,
            "f_mis": 0.0,
            "sigma_R": np.nan,
            "mean_logm": np.nan,
            "n_dof": total_dof,
            "success": not np.isnan(combined_scatter)
            and not np.isnan(combined_chi2_reduced),
            "n_lens": total_n_lens,
            "z_median": avg_z_median,
            "ds_list": ds_list,
            "ds_err_list": ds_err_list,
            "rp_mpc": rp_mpc,
        }

    # ==================================================================
    # Legacy Single-bin Path (colossus + fit_scatter_map)
    # ==================================================================
    # Use the first bin (the single top-N bin in the typical use case)
    bmeta = bin_metadata[0]
    n_lens = bmeta["n_lens"]
    bid = bmeta["bin_id"]

    lens_table = global_lens_table[global_lens_table["bin_id"] == bid]
    rand_table = global_random_table[global_random_table["bin_id"] == bid]

    logger.info(
        "Prepared bin '%s': %d lenses, %d randoms",
        bmeta["bin_name"],
        n_lens,
        len(rand_table),
    )

    # Convert lens/random to dsigma format
    table_l = dsigma_table(lens_table, "lens", z="z", ra="ra", dec="dec", w_sys=1.0)
    table_r = dsigma_table(rand_table, "lens", z="z", ra="ra", dec="dec", w_sys=1.0)

    profile_result = compute_single_bin_profile(
        table_l=table_l,
        table_r=table_r,
        table_s=table_s,
        table_c=table_c,
        table_n=table_n,
        rp_bins=rp_bins,
        z_bins=z_bins,
        n_jackknife=n_jackknife,
        n_jobs=n_jobs,
        comoving=comoving,
        lens_source_cut=lens_source_cut,
        corrections=corrections,
        source_version=source_version,
    )

    result_table = profile_result["result_table"]
    jk_cov = profile_result["jk_cov"]
    z_median = profile_result["z_median"]

    rp_mpc = np.asarray(result_table["rp"])
    ds_data = np.asarray(result_table["ds"])
    ds_err = np.asarray(result_table["ds_err"])

    if n_lens < top_n:
        penalty_factor = float(n_lens) / top_n
        logger.warning(
            "n_lens (%d) < top_n (%d). Diluting signal by %.3f to penalize.",
            n_lens,
            top_n,
            penalty_factor,
        )
        ds_data = ds_data * penalty_factor
        ds_err = ds_err * np.sqrt(penalty_factor)
        jk_cov = jk_cov * penalty_factor

        # Override n_lens to top_n so abundance matching uses the expected number density
        n_lens = top_n

    logger.info(
        "ΔΣ computed: %d radial bins, z_median=%.3f",
        len(rp_mpc),
        z_median,
    )

    # ------------------------------------------------------------------
    # Step 3: Scatter fitting (MAP)
    # ------------------------------------------------------------------

    if skip_map_fit:
        weights = 1.0 / (ds_err**2)
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            mean_ds = float(np.sum(weights * ds_data) / sum_weights)
            snr = float(mean_ds * np.sqrt(sum_weights))
        else:
            mean_ds = np.nan
            snr = np.nan

        map_result = {
            "scatter": np.nan,
            "f_mis": 0.0,
            "sigma_R": 0.0,
            "chi2_reduced": 1.0,
            "mean_logm": np.nan,
            "mean_ds": mean_ds,
            "snr": snr,
            "n_lens": n_lens,
            "z_median": z_median,
            "success": not np.isnan(snr),
            "n_dof": len(ds_data),
            "ds_data": ds_data,
            "ds_err": ds_err,
            "rp_mpc": rp_mpc,
        }
        logger.info(
            "Pipeline complete (fast mode): mean_ds=%.3f, snr=%.3f", mean_ds, snr
        )
        return map_result

    # N_OBJ is auto-computed from the actual lens count
    n_obs = compute_survey_number_density(
        area_sq_deg, z_min, z_max, n_lens, cosmology_name
    )

    from colossus.cosmology import cosmology as colossus_cosmo

    cosmo = colossus_cosmo.setCosmology(cosmology_name)
    h = cosmo.h

    rp_kpc_h, ds_colossus, ds_err_colossus, jk_cov_colossus, jk_cov_inv_colossus = (
        convert_dsigma_to_colossus_units(rp_mpc, ds_data, ds_err, jk_cov, h)
    )

    # Filter to only fit within 3 Mpc/h (3000 kpc/h)
    mask = rp_kpc_h <= 3000.0
    rp_kpc_h = rp_kpc_h[mask]
    ds_colossus = ds_colossus[mask]
    ds_err_colossus = ds_err_colossus[mask]
    jk_cov_colossus = jk_cov_colossus[mask][:, mask]
    jk_cov_inv_colossus = np.linalg.inv(jk_cov_colossus)

    z_lens = z_median if np.isfinite(z_median) else 0.3

    model_state = build_scatter_model(
        rp_kpc_h=rp_kpc_h,
        z_lens=z_lens,
        n_obs=n_obs,
        cosmology_name=cosmology_name,
        mass_def=mass_def,
        conc_model=conc_model,
        bias_model=bias_model,
        mass_func_model=mass_func_model,
    )

    map_result = fit_scatter_map(model_state, ds_colossus, jk_cov_inv_colossus)

    # Augment the result with pipeline metadata
    map_result["n_lens"] = n_lens
    map_result["z_median"] = z_median

    logger.info(
        "Pipeline complete: scatter=%.3f, f_mis=%.3f, sigma_R=%.1f, chi2/dof=%.3f",
        map_result["scatter"],
        map_result["f_mis"],
        map_result["sigma_R"],
        map_result["chi2_reduced"],
    )

    return map_result
