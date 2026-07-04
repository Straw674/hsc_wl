"""Reusable data-preparation utilities for the HSC weak-lensing pipeline.

This module contains pure-computation functions extracted from
``scripts/prepare_lens_and_random.py``.  It has **no** dependency on
``initial.py`` and can be imported from anywhere.

Binning modes and lens readers are dispatched through small lookup
tables (``_BINNERS``, ``_LENS_READERS``) so adding a new mode is a
one-line change.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from astropy.table import Table, vstack

from hsc_wl.config import (
    BinningConfig,
    LensCatalogConfig,
    WLConfig,
    get_latest_cluster_catalog,
    resolve_binning,
    resolve_config,
)
from hsc_wl.coverage import filter_lens_by_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_path(path_value, root_path):
    """Return an absolute ``Path`` for *path_value*.

    If *path_value* is already absolute it is returned as-is; otherwise it is
    resolved relative to *root_path*.
    """
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return Path(root_path) / path_obj


# ---------------------------------------------------------------------------
# Format readers (dispatch table)
# ---------------------------------------------------------------------------


def read_dat_to_pandas(path):
    """Read a whitespace/csv file into a :class:`pandas.DataFrame`.

    If the first non-empty line starts with ``#`` the remainder of that line
    is used as column names.  Otherwise ``pandas`` infers the format.
    """
    with open(path, encoding="utf-8") as file:
        first_nonempty_line = next(
            (line.strip() for line in file if line.strip()),
            "",
        )

    if first_nonempty_line.startswith("#"):
        columns = first_nonempty_line.lstrip("#").split()
        df = pd.read_csv(
            path,
            comment="#",
            sep=r"\s+",
            names=columns,
            header=None,
            engine="python",
        )
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    return df


_LENS_READERS: dict[str, Callable[[Path], Table]] = {
    "pandas_dat": lambda p: Table.from_pandas(read_dat_to_pandas(str(p))),
}


def read_lens_catalog(path: Path, fmt: str | None) -> Table:
    """Read a lens catalog, dispatching on *fmt* (``None`` => auto-detect)."""
    if fmt is None:
        return Table.read(path)
    reader = _LENS_READERS.get(fmt)
    if reader is not None:
        return reader(path)
    return Table.read(path, format=fmt)


# ---------------------------------------------------------------------------
# Binning (dispatch table)
# ---------------------------------------------------------------------------


def _bin_by_edges(lens, col_rank, binning: BinningConfig) -> list[tuple]:
    edges = (
        binning.edges_mass
        if col_rank.lower().startswith("logm")
        else binning.edges_richness
    )
    if len(edges) < 2:
        raise ValueError("edges must contain at least two values.")
    slices = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        mask = (lens[col_rank] >= low) & (lens[col_rank] < high)
        bin_desc = f"{col_rank} in [{low}, {high})"
        slices.append((f"bin{i}", lens[mask], low, high, bin_desc))
    return slices


def _bin_by_top_n(lens, col_rank, binning: BinningConfig) -> list[tuple]:
    reverse = binning.selection_order == "desc"
    order_idx = np.argsort(np.asarray(lens[col_rank]))
    if reverse:
        order_idx = order_idx[::-1]
    sorted_lens = lens[order_idx]
    n_take = min(binning.top_n, len(sorted_lens))
    bin_desc = (
        f"top_n={binning.top_n}, rank top {n_take} by {col_rank} "
        f"{binning.selection_order}"
    )
    return [("bin0", sorted_lens[:n_take], None, None, bin_desc)]


def _bin_by_top_counts(lens, col_rank, binning: BinningConfig) -> list[tuple]:
    reverse = binning.selection_order == "desc"
    order_idx = np.argsort(np.asarray(lens[col_rank]))
    if reverse:
        order_idx = order_idx[::-1]
    sorted_lens = lens[order_idx]
    cursor = 0
    slices = []
    for i, count in enumerate(binning.top_counts):
        next_cursor = min(cursor + count, len(sorted_lens))
        lens_bin = sorted_lens[cursor:next_cursor]
        bin_desc = (
            f"top_counts={count}, rank [{cursor}, {next_cursor}) by {col_rank} "
            f"{binning.selection_order}"
        )
        slices.append((f"bin{i}", lens_bin, None, None, bin_desc))
        cursor = next_cursor
    return slices


_BINNERS: dict[str, Callable] = {
    "edges": _bin_by_edges,
    "top_counts": _bin_by_top_counts,
    "top_n": _bin_by_top_n,
}


def build_bin_slices(lens, col_rank, binning: BinningConfig) -> list[tuple]:
    """Partition *lens* into bin slices according to *binning.mode*.

    Returns
    -------
    list[tuple]
        Each element is ``(bin_name, lens_bin, low_edge, high_edge, bin_desc)``.
    """
    if binning.mode not in _BINNERS:
        raise ValueError(
            f"binning.mode must be one of {list(_BINNERS)}, got {binning.mode!r}."
        )
    return _BINNERS[binning.mode](lens, col_rank, binning)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


def summarize_bin_boundaries(lens_bin, col_rank, low_edge, high_edge, binning_mode):
    """Return a human-readable string describing the rank boundaries of a bin."""
    if len(lens_bin) == 0:
        return "lower=NA, upper=NA"

    if binning_mode == "edges":
        return f"lower={low_edge}, upper={high_edge}"

    rank_vals = np.asarray(lens_bin[col_rank])
    return f"lower={float(np.min(rank_vals)):.6g}, upper={float(np.max(rank_vals)):.6g}"


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


def apply_lens_quality_filters(lens: Table) -> Table:
    """Apply the standard quality cuts used by all catalogs.

    Each cut is optional: if the relevant column is missing it is skipped
    with a warning (the catalog is from a different survey).
    """
    if "bsm_s18a" in lens.colnames:
        lens = lens[lens["bsm_s18a"] > 0]
        logger.info("Applied bsm_s18a > 0 mask: %d objects remain.", len(lens))
    else:
        logger.warning(
            "'bsm_s18a' column not found in lens catalog; "
            "expected if the catalog is not from s16a_redm_hsc."
        )

    if "logm_cmod" in lens.colnames:
        lens = lens[lens["logm_cmod"] >= 11.2]
        logger.info("Applied logm_cmod >= 11.2 mask: %d objects remain.", len(lens))
    else:
        logger.warning(
            "'logm_cmod' column not found in lens catalog; "
            "expected if the catalog is not from s16a_redm_hsc."
        )

    return lens


def apply_lens_range_filters(lens: Table, cfg: LensCatalogConfig) -> Table:
    """Apply finite-rank, redshift, RA and Dec range cuts from *cfg*."""
    col_rank = cfg.columns.col_rank
    if col_rank in lens.colnames:
        lens = lens[np.isfinite(lens[col_rank])]
        logger.info(
            "Applied finite mask on '%s': %d objects remain.", col_rank, len(lens)
        )
    else:
        logger.warning(
            "col_rank '%s' not found in lens catalog; this may cause issues.",
            col_rank,
        )

    col_z, col_ra, col_dec = cfg.columns.z, cfg.columns.ra, cfg.columns.dec

    z_min, z_max = cfg.redshift_range
    lens = lens[(lens[col_z] >= z_min) & (lens[col_z] <= z_max)]
    logger.info(
        "Applied redshift mask %s on '%s': %d objects remain.",
        cfg.redshift_range,
        col_z,
        len(lens),
    )

    if cfg.ra_range is not None:
        ra_min, ra_max = cfg.ra_range
        lens = lens[(lens[col_ra] >= ra_min) & (lens[col_ra] <= ra_max)]
        logger.info(
            "Applied RA mask %s on '%s': %d objects remain.",
            cfg.ra_range,
            col_ra,
            len(lens),
        )

    if cfg.dec_range is not None:
        dec_min, dec_max = cfg.dec_range
        lens = lens[(lens[col_dec] >= dec_min) & (lens[col_dec] <= dec_max)]
        logger.info(
            "Applied Dec mask %s on '%s': %d objects remain.",
            cfg.dec_range,
            col_dec,
            len(lens),
        )

    return lens


# ---------------------------------------------------------------------------
# Core pipeline (in-memory, no I/O)
# ---------------------------------------------------------------------------


def prepare_lens_random_tables(
    lens_catalog: Table,
    random_catalog: Table,
    catalog_config: LensCatalogConfig,
    binning: BinningConfig,
    random_multiplier: int = 20,
    rng_seed: int | None = None,
):
    """Run the preparation pipeline and return in-memory tables.

    This mirrors :func:`run_prepare_pipeline` but performs **no** file I/O or
    plotting.  Returns structured results that callers can persist or
    visualise as needed.

    Parameters
    ----------
    lens_catalog : astropy.table.Table
        Raw lens catalog (already loaded).
    random_catalog : astropy.table.Table
        Random-points catalog.
    catalog_config : LensCatalogConfig
        Lens catalog configuration (columns, ranges, factor).
    binning : BinningConfig
        Effective binning (already scaled by ``top_counts_factor``).
    random_multiplier : int, optional
        Number of random points per lens object (default 20).
    rng_seed : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    dict or None
        Keys: ``global_lens_table``, ``global_random_table``,
        ``bin_metadata``.  ``None`` when no valid objects remain.
    """
    lens = apply_lens_quality_filters(lens_catalog.copy())
    lens = apply_lens_range_filters(lens, catalog_config)
    lens = filter_lens_by_mask(
        lens,
        ra_col=catalog_config.columns.ra,
        dec_col=catalog_config.columns.dec,
    )

    if len(random_catalog) == 0:
        raise ValueError("No random points found in the input catalog.")

    # Find random RA/Dec column names
    random_ra_col = next(
        (c for c in random_catalog.colnames if c.lower() == "ra"), "ra"
    )
    random_dec_col = next(
        (c for c in random_catalog.colnames if c.lower() == "dec"), "dec"
    )

    # Filter random catalog by RA/Dec box and mask footprint
    random_catalog = random_catalog.copy()
    if catalog_config.ra_range is not None:
        ra_min, ra_max = catalog_config.ra_range
        random_catalog = random_catalog[
            (random_catalog[random_ra_col] >= ra_min)
            & (random_catalog[random_ra_col] <= ra_max)
        ]
    if catalog_config.dec_range is not None:
        dec_min, dec_max = catalog_config.dec_range
        random_catalog = random_catalog[
            (random_catalog[random_dec_col] >= dec_min)
            & (random_catalog[random_dec_col] <= dec_max)
        ]

    random_catalog = filter_lens_by_mask(
        random_catalog,
        ra_col=random_ra_col,
        dec_col=random_dec_col,
    )

    if len(random_catalog) == 0:
        raise ValueError(
            f"No random points left after applying RA/Dec cuts {catalog_config.ra_range}/{catalog_config.dec_range} and mask."
        )

    col_rank = catalog_config.columns.col_rank
    bin_slices = build_bin_slices(lens, col_rank, binning)

    rng = np.random.default_rng(rng_seed)

    lens_tables = []
    random_tables = []
    bin_metadata = []

    for i, (bin_name, lens_bin, low_edge, high_edge, bin_desc) in enumerate(bin_slices):
        n_bin = len(lens_bin)
        if n_bin == 0:
            logger.info("%s has 0 objects; skipping. (%s)", bin_name, bin_desc)
            continue

        boundary_text = summarize_bin_boundaries(
            lens_bin, col_rank, low_edge, high_edge, binning.mode
        )
        logger.info(
            "%s (%s) -> N_total=%d, boundary: %s",
            bin_name,
            bin_desc,
            n_bin,
            boundary_text,
        )

        lens_out = Table()
        lens_out["ra"] = lens_bin[catalog_config.columns.ra]
        lens_out["dec"] = lens_bin[catalog_config.columns.dec]
        lens_out["z"] = lens_bin[catalog_config.columns.z]
        lens_out["wsys"] = np.ones(n_bin, dtype=float)
        lens_out["bin_id"] = i

        n_random = n_bin * int(random_multiplier)
        replace_ra_dec = n_random > len(random_catalog)
        rand_idx = rng.choice(
            len(random_catalog), size=n_random, replace=replace_ra_dec
        )
        z_idx = rng.choice(n_bin, size=n_random, replace=True)

        random_out = Table()
        random_out["ra"] = random_catalog[random_ra_col][rand_idx]
        random_out["dec"] = random_catalog[random_dec_col][rand_idx]
        random_out["z"] = lens_bin[catalog_config.columns.z][z_idx]
        random_out["wsys"] = np.ones(n_random, dtype=float)
        random_out["bin_id"] = i

        lens_tables.append(lens_out)
        random_tables.append(random_out)
        bin_metadata.append(
            {
                "bin_id": i,
                "bin_name": bin_name,
                "n_lens": n_bin,
                "bin_desc": bin_desc,
                "boundary_text": boundary_text,
            }
        )

    if not lens_tables:
        return None

    return {
        "global_lens_table": vstack(lens_tables),
        "global_random_table": vstack(random_tables),
        "bin_metadata": bin_metadata,
    }


# ---------------------------------------------------------------------------
# Dynamic Catalog and Preparation pipeline functions
# ---------------------------------------------------------------------------


def gaussian_kde_1d(np_mod, values, grid=None, num_points=256, bandwidth=None):
    """Simple Gaussian KDE implementation without scipy."""
    samples = np_mod.asarray(values, dtype=float)
    samples = samples[np_mod.isfinite(samples)]
    if samples.size == 0:
        return np_mod.array([]), np_mod.array([])

    if grid is None:
        data_min = float(np_mod.min(samples))
        data_max = float(np_mod.max(samples))
        if data_min == data_max:
            pad = 1.0 if data_min == 0.0 else abs(data_min) * 0.1
            grid = np_mod.linspace(data_min - pad, data_max + pad, num_points)
        else:
            pad = max((data_max - data_min) * 0.15, 1e-3)
            grid = np_mod.linspace(data_min - pad, data_max + pad, num_points)
    else:
        grid = np_mod.asarray(grid, dtype=float)

    if samples.size == 1:
        width = max(abs(samples[0]) * 0.1, 1e-3)
        density = np_mod.exp(-0.5 * ((grid - samples[0]) / width) ** 2)
        density /= width * np_mod.sqrt(2.0 * np_mod.pi)
        return grid, density

    if bandwidth is None:
        std = float(np_mod.std(samples, ddof=1))
        if not np_mod.isfinite(std) or std <= 0.0:
            std = float(np_mod.std(samples))
        bandwidth = 1.06 * std * (samples.size ** (-1.0 / 5.0))
        if not np_mod.isfinite(bandwidth) or bandwidth <= 0.0:
            span = float(np_mod.max(samples) - np_mod.min(samples))
            bandwidth = max(span / 25.0, 1e-3)

    diff = (grid[:, None] - samples[None, :]) / bandwidth
    density = np_mod.exp(-0.5 * diff**2).sum(axis=1)
    density /= samples.size * bandwidth * np_mod.sqrt(2.0 * np_mod.pi)
    return grid, density


def show_alignment_plot(
    plt,
    lens_out,
    random_out,
    lens_label,
    ra_col,
    dec_col,
    z_col,
    bin_name,
    low_edge,
    high_edge,
    bin_desc=None,
):
    """Draw and show alignment plots for validation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    lens_count = len(lens_out)
    random_count = len(random_out)

    ra_min = min(np.min(lens_out[ra_col]), np.min(random_out[ra_col]))
    ra_max = max(np.max(lens_out[ra_col]), np.max(random_out[ra_col]))
    dec_min = min(np.min(lens_out[dec_col]), np.min(random_out[dec_col]))
    dec_max = max(np.max(lens_out[dec_col]), np.max(random_out[dec_col]))

    ra_pad = (ra_max - ra_min) * 0.05 if ra_max != ra_min else 1.0
    dec_pad = (dec_max - dec_min) * 0.05 if dec_max != dec_min else 1.0

    plot_ra_lim = (ra_min - ra_pad, ra_max + ra_pad)
    plot_dec_lim = (dec_min - dec_pad, dec_max + dec_pad)

    axes[0].scatter(
        lens_out[ra_col],
        lens_out[dec_col],
        s=8,
        alpha=0.35,
        color="tab:blue",
        edgecolors="none",
    )
    axes[0].set_title(f"{lens_label} footprint (N={lens_count})")
    axes[0].set_xlabel("RA")
    axes[0].set_ylabel("Dec")
    axes[0].set_xlim(plot_ra_lim)
    axes[0].set_ylim(plot_dec_lim)

    h2 = axes[1].hexbin(
        random_out[ra_col],
        random_out[dec_col],
        gridsize=45,
        extent=(ra_min, ra_max, dec_min, dec_max),
        mincnt=1,
        cmap="Blues",
    )
    axes[1].set_title(f"Random footprint (N={random_count})")
    axes[1].set_xlabel("RA")
    axes[1].set_ylabel("Dec")
    axes[1].set_xlim(plot_ra_lim)
    axes[1].set_ylim(plot_dec_lim)
    fig.colorbar(h2, ax=axes[1], label="Counts")

    z_min = float(min(np.min(lens_out[z_col]), np.min(random_out[z_col])))
    z_max = float(max(np.max(lens_out[z_col]), np.max(random_out[z_col])))
    if z_min == z_max:
        z_grid = np.linspace(z_min - 1e-3, z_max + 1e-3, 256)
    else:
        z_grid = np.linspace(z_min, z_max, 256)

    lens_z_grid, lens_z_density = gaussian_kde_1d(np, lens_out[z_col], grid=z_grid)
    random_z_grid, random_z_density = gaussian_kde_1d(
        np, random_out[z_col], grid=z_grid
    )

    axes[2].plot(
        lens_z_grid, lens_z_density, color="tab:blue", lw=2, label=f"{lens_label} z"
    )
    axes[2].fill_between(lens_z_grid, lens_z_density, color="tab:blue", alpha=0.2)
    axes[2].plot(
        random_z_grid, random_z_density, color="tab:orange", lw=2, label="Random z"
    )
    axes[2].fill_between(
        random_z_grid, random_z_density, color="tab:orange", alpha=0.15
    )
    axes[2].set_title("z distribution (KDE)")
    axes[2].set_xlabel("z")
    axes[2].set_ylabel("Density")
    axes[2].legend()

    if bin_desc is None:
        title_high = "+inf" if high_edge == float("inf") else f"{high_edge}"
        fig.suptitle(f"{bin_name} lambda in [{low_edge}, {title_high})")
        fig.tight_layout()
        plt.close(fig)
        return
    else:
        fig.suptitle(f"{bin_name} {bin_desc}")
    fig.tight_layout()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level prepare entry point
# ---------------------------------------------------------------------------


def _find_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    current_dir = Path.cwd().resolve()
    while True:
        if not current_dir or current_dir == current_dir.parent:
            break
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent
    raise FileNotFoundError("Could not find the project root.")


def run_prepare_pipeline(cfg: WLConfig, root: Path | None = None):
    """Run the lens and random preparation pipeline and write catalogs to disk.

    Reads the raw lens/random catalogs described by ``cfg.lens``, applies the
    quality and range filters, bins them according to ``cfg.binning`` (scaled
    by the catalog's ``top_counts_factor``), and writes the unified
    ``<label>_lenses.fits`` / ``<label>_randoms.fits`` under
    ``<save_root>/prepare/``.
    """
    root = _find_root(root)
    cfg = resolve_config(cfg, root)
    lens_cfg = cfg.lens

    lens_path = resolve_path(lens_cfg.lens_path, root)
    random_path = resolve_path(lens_cfg.random_path, root)

    lens = read_lens_catalog(lens_path, lens_cfg.lens_format)
    if lens_cfg.random_format:
        random = Table.read(random_path, format=lens_cfg.random_format)
    else:
        random = Table.read(random_path)

    lens = apply_lens_quality_filters(lens)
    lens = apply_lens_range_filters(lens, lens_cfg)
    lens = filter_lens_by_mask(
        lens,
        root=root,
        ra_col=lens_cfg.columns.ra,
        dec_col=lens_cfg.columns.dec,
    )

    col_rank = lens_cfg.columns.col_rank
    print("-" * 80)
    print(f"Using source: {cfg.label}")
    print(f"Column used for ranking: {col_rank}")
    print("-" * 80)
    print(f"Redshift range: {lens_cfg.redshift_range}")
    print(
        f"Effective area: {lens_cfg.area_deg2:.4f} deg2"
        if lens_cfg.area_deg2
        else "Effective area: N/A"
    )
    print(f"Lens file: {lens_path}")
    print(f"Random file: {random_path}")
    print(f"Lens columns: {lens.colnames}")
    print(f"Random columns: {random.colnames}")

    binning = resolve_binning(cfg.binning, lens_cfg.top_counts_factor)

    print("-" * 80)
    if binning.mode == "edges":
        print(f"Binning mode=edges, edges={binning.edges_richness}")
    elif binning.mode == "top_n":
        print(
            f"Binning mode=top_n, top_n={cfg.binning.top_n}, "
            f"top_counts_factor={lens_cfg.top_counts_factor}, "
            f"top_n(scaled)={binning.top_n}, "
            f"selection_order={binning.selection_order}"
        )
    else:
        print(
            f"Binning mode=top_counts, top_counts(raw)={cfg.binning.top_counts}, "
            f"top_counts_factor={lens_cfg.top_counts_factor}, "
            f"top_counts(scaled)={binning.top_counts}, "
            f"selection_order={binning.selection_order}"
        )

    save_root = cfg.resolved_save_root(root)
    output_dir = save_root / "prepare"
    output_dir.mkdir(parents=True, exist_ok=True)

    prep_result = prepare_lens_random_tables(
        lens_catalog=lens,
        random_catalog=random,
        catalog_config=lens_cfg,
        binning=binning,
        random_multiplier=cfg.random_multiplier,
        rng_seed=cfg.rng_seed,
    )

    if not prep_result:
        print("No valid objects left to prepare.")
        return

    global_lens_table = prep_result["global_lens_table"]
    global_random_table = prep_result["global_random_table"]
    bin_metadata = prep_result["bin_metadata"]

    metadata_json = json.dumps(bin_metadata)
    global_lens_table.meta["BIN_META"] = metadata_json
    global_random_table.meta["BIN_META"] = metadata_json

    lens_file = output_dir / f"{lens_cfg.label}_lenses.fits"
    random_file = output_dir / f"{lens_cfg.label}_randoms.fits"

    global_lens_table.write(lens_file, overwrite=True)
    global_random_table.write(random_file, overwrite=True)

    if cfg.make_plots:
        import matplotlib.pyplot as plt

        for bmeta in bin_metadata:
            bid = bmeta["bin_id"]
            lens_bin = global_lens_table[global_lens_table["bin_id"] == bid]
            rand_bin = global_random_table[global_random_table["bin_id"] == bid]
            if len(lens_bin) > 0:
                show_alignment_plot(
                    plt=plt,
                    lens_out=lens_bin,
                    random_out=rand_bin,
                    lens_label=lens_cfg.label,
                    ra_col="ra",
                    dec_col="dec",
                    z_col="z",
                    bin_name=bmeta["bin_name"],
                    low_edge=None,
                    high_edge=None,
                    bin_desc=bmeta["bin_desc"],
                )

    print("\n" + "=" * 30)
    print(f"Lenses saved to: {lens_file}")
    print(f"Randoms saved to: {random_file}")
