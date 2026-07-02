"""Reusable data-preparation utilities for the HSC weak-lensing pipeline.

This module contains pure-computation functions extracted from
``scripts/prepare_lens_and_random.py``.  It has **no** dependency on
``initial.py`` and can be imported from anywhere.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_path(path_value, root_path):
    """Return an absolute ``Path`` for *path_value*.

    If *path_value* is already absolute it is returned as-is; otherwise it is
    resolved relative to *root_path*.

    Parameters
    ----------
    path_value : str or Path
        File-system path (absolute or relative).
    root_path : str or Path
        Project root directory used to resolve relative paths.

    Returns
    -------
    Path
    """
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return Path(root_path) / path_obj


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------


def read_dat_to_pandas(path):
    """Read a whitespace/csv file into a :class:`pandas.DataFrame`.

    If the first non-empty line starts with ``#`` the remainder of that line
    is used as column names.  Otherwise ``pandas`` infers the format.

    Parameters
    ----------
    path : str or Path
        Path to the input file.

    Returns
    -------
    pandas.DataFrame
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


# ---------------------------------------------------------------------------
# Binning configuration
# ---------------------------------------------------------------------------


def get_binning_settings(
    source_name,
    top_counts_factor,
    *,
    binning_mode,
    col_rank_edges_mass,
    col_rank_edges_richness,
    top_counts,
    top_selection_order,
    top_n,
):
    """Build a binning-settings dictionary from explicit parameters.

    All configuration values that were previously read from module-level
    globals are now required keyword arguments.

    Parameters
    ----------
    source_name : str
        Catalog source key (used to choose between mass / richness edges).
    top_counts_factor : float
        Multiplicative factor applied to *top_counts* entries.
    binning_mode : str
        One of ``"edges"``, ``"top_counts"``, or ``"top_n"``.
    col_rank_edges_mass : list[float]
        Edge values when ranking by mass.
    col_rank_edges_richness : list[float]
        Edge values when ranking by richness.
    top_counts : list[int]
        Per-bin counts for ``top_counts`` mode.
    top_selection_order : str
        ``"asc"`` or ``"desc"``.
    top_n : int
        Number of top objects for ``top_n`` mode.

    Returns
    -------
    dict
        Keys: ``mode``, ``col_rank_edges``, ``top_counts``,
        ``top_selection_order``, and optionally ``top_n``.
    """
    if binning_mode not in {"edges", "top_counts", "top_n"}:
        raise ValueError("binning_mode must be 'edges', 'top_counts', or 'top_n'.")

    if binning_mode == "edges":
        if source_name.endswith("mass"):
            col_rank_edges = col_rank_edges_mass
        else:
            col_rank_edges = col_rank_edges_richness

        if len(col_rank_edges) < 2:
            raise ValueError("col_rank_edges must contain at least two values.")

        return {
            "mode": "edges",
            "col_rank_edges": col_rank_edges,
            "top_counts": None,
            "top_selection_order": None,
        }

    if top_selection_order not in {"asc", "desc"}:
        raise ValueError("top_selection_order must be 'asc' or 'desc'.")

    if binning_mode == "top_n":
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive integer.")
        return {
            "mode": "top_n",
            "col_rank_edges": None,
            "top_counts": None,
            "top_n": int(round(top_n * top_counts_factor)),
            "top_selection_order": top_selection_order,
        }

    if not top_counts:
        raise ValueError("top_counts must be non-empty when binning_mode='top_counts'.")

    if any((not isinstance(c, int)) or c <= 0 for c in top_counts):
        raise ValueError("top_counts must contain only positive integers.")

    if not np.isfinite(top_counts_factor) or top_counts_factor <= 0:
        raise ValueError("top_counts_factor must be a positive finite number.")

    scaled_top_counts = [int(round(c * top_counts_factor)) for c in top_counts]
    if any(c <= 0 for c in scaled_top_counts):
        raise ValueError(
            "Scaled top_counts must be positive. Increase top_counts_factor."
        )

    return {
        "mode": "top_counts",
        "col_rank_edges": None,
        "top_counts": scaled_top_counts,
        "top_selection_order": top_selection_order,
    }


# ---------------------------------------------------------------------------
# Bin-slice construction
# ---------------------------------------------------------------------------


def build_bin_slices(
    lens,
    col_rank,
    *,
    binning_mode,
    col_rank_edges=None,
    top_counts=None,
    top_n=None,
    top_selection_order=None,
):
    """Partition *lens* into bin slices according to *binning_mode*.

    Parameters
    ----------
    lens : astropy.table.Table
        Lens catalog (already filtered).
    col_rank : str
        Column name used for ranking / binning.
    binning_mode : str
        ``"edges"``, ``"top_counts"``, or ``"top_n"``.
    col_rank_edges : list[float] or None
        Bin edges for ``edges`` mode.
    top_counts : list[int] or None
        Per-bin counts for ``top_counts`` mode.
    top_n : int or None
        Total count for ``top_n`` mode.
    top_selection_order : str or None
        ``"asc"`` or ``"desc"``.

    Returns
    -------
    list[tuple]
        Each element is ``(bin_name, lens_bin, low_edge, high_edge, bin_desc)``.
    """
    if binning_mode == "edges":
        if col_rank_edges is None or len(col_rank_edges) < 2:
            raise ValueError("col_rank_edges must contain at least two values.")

        bin_slices = []
        for i in range(len(col_rank_edges) - 1):
            low_edge = col_rank_edges[i]
            high_edge = col_rank_edges[i + 1]
            lens_mask = (lens[col_rank] >= low_edge) & (lens[col_rank] < high_edge)
            lens_bin = lens[lens_mask]
            bin_name = f"bin{i}"
            bin_desc = f"{col_rank} in [{low_edge}, {high_edge})"
            bin_slices.append((bin_name, lens_bin, low_edge, high_edge, bin_desc))
        return bin_slices

    if binning_mode == "top_n":
        if top_selection_order not in {"asc", "desc"}:
            raise ValueError("top_selection_order must be 'asc' or 'desc'.")

        reverse = top_selection_order == "desc"
        order_idx = np.argsort(np.asarray(lens[col_rank]))
        if reverse:
            order_idx = order_idx[::-1]

        sorted_lens = lens[order_idx]
        next_cursor = min(top_n, len(sorted_lens))
        lens_bin = sorted_lens[:next_cursor]

        rank_window = f"top {next_cursor} by {col_rank} {top_selection_order}"
        bin_desc = f"top_n={top_n}, {rank_window}"

        return [("bin0", lens_bin, None, None, bin_desc)]

    if binning_mode == "top_counts":
        if not top_counts:
            raise ValueError(
                "top_counts must be non-empty when binning_mode='top_counts'."
            )

        if any((not isinstance(c, int)) or c <= 0 for c in top_counts):
            raise ValueError("top_counts must contain only positive integers.")

        if top_selection_order not in {"asc", "desc"}:
            raise ValueError("top_selection_order must be 'asc' or 'desc'.")

        reverse = top_selection_order == "desc"
        order_idx = np.argsort(np.asarray(lens[col_rank]))
        if reverse:
            order_idx = order_idx[::-1]

        sorted_lens = lens[order_idx]
        cursor = 0
        raw_bins = []

        for count in top_counts:
            next_cursor = min(cursor + count, len(sorted_lens))
            lens_bin = sorted_lens[cursor:next_cursor]
            rank_window = (
                f"rank index [{cursor}, {next_cursor}) by {col_rank} "
                f"{top_selection_order}"
            )
            bin_desc = f"top_counts={count}, {rank_window}"
            raw_bins.append((lens_bin, None, None, bin_desc))
            cursor = next_cursor

        # Present bins in their natural selection order (richest/most massive first)
        ordered_bins = raw_bins


        bin_slices = []
        for i, (lens_bin, low_edge, high_edge, bin_desc) in enumerate(
            ordered_bins, start=0
        ):
            bin_name = f"bin{i}"
            bin_slices.append((bin_name, lens_bin, low_edge, high_edge, bin_desc))

        return bin_slices

    raise ValueError("binning_mode must be 'edges', 'top_counts', or 'top_n'.")


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


def summarize_bin_boundaries(lens_bin, col_rank, low_edge, high_edge, binning_mode):
    """Return a human-readable string describing the rank boundaries of a bin.

    Parameters
    ----------
    lens_bin : astropy.table.Table
        Lens rows in this bin.
    col_rank : str
        Ranking column name.
    low_edge, high_edge : float or None
        Explicit edges (used in ``edges`` mode).
    binning_mode : str
        ``"edges"``, ``"top_counts"``, or ``"top_n"``.

    Returns
    -------
    str
    """
    if len(lens_bin) == 0:
        return "lower=NA, upper=NA"

    if binning_mode == "edges":
        return f"lower={low_edge}, upper={high_edge}"

    rank_vals = np.asarray(lens_bin[col_rank])
    rank_min = float(np.min(rank_vals))
    rank_max = float(np.max(rank_vals))
    return f"lower={rank_min:.6g}, upper={rank_max:.6g}"


# ---------------------------------------------------------------------------
# Core pipeline (in-memory, no I/O)
# ---------------------------------------------------------------------------


def prepare_lens_random_tables(
    lens_catalog,
    random_catalog,
    catalog_config,
    binning_settings,
    random_multiplier=20,
    rng_seed=None,
):
    """Run the preparation pipeline and return in-memory tables.

    This function mirrors the logic of ``run_pipeline()`` in the script but
    performs **no** file I/O or plotting.  It returns structured results that
    callers can persist or visualise as needed.

    Parameters
    ----------
    lens_catalog : astropy.table.Table
        Raw lens catalog (already loaded).
    random_catalog : astropy.table.Table
        Random-points catalog.
    catalog_config : dict
        Same structure as a ``CATALOG_SOURCES`` entry.  Required keys:
        ``label``, ``columns`` (with sub-keys ``col_rank``, ``ra``, ``dec``,
        ``z``), ``redshift_range``.  Optional: ``ra_range``, ``dec_range``.
    binning_settings : dict
        As returned by :func:`get_binning_settings`.
    random_multiplier : int, optional
        Number of random points per lens object (default 20).
    rng_seed : int or None, optional
        Seed for the random number generator (default ``None``).

    Returns
    -------
    list[dict]
        Each element has keys: ``bin_name``, ``lens_table``
        (:class:`~astropy.table.Table`), ``random_table``
        (:class:`~astropy.table.Table`), ``n_lens``, ``bin_desc``.
    """
    lens = lens_catalog.copy()

    # --- quality filters ---------------------------------------------------
    if "bsm_s18a" in lens.colnames:
        mask_bsm = lens["bsm_s18a"] > 0
        lens = lens[mask_bsm]
        logger.info("Applied bsm_s18a > 0 mask: %d objects remain.", len(lens))
    else:
        logger.warning(
            "'bsm_s18a' column not found in lens catalog; "
            "expected if the catalog is not from s16a_redm_hsc."
        )

    if "logm_cmod" in lens.colnames:
        mask_logm = lens["logm_cmod"] >= 11.2
        lens = lens[mask_logm]
        logger.info(
            "Applied logm_cmod >= 11.2 mask: %d objects remain.", len(lens)
        )
    else:
        logger.warning(
            "'logm_cmod' column not found in lens catalog; "
            "expected if the catalog is not from s16a_redm_hsc."
        )

    # --- column short-hands ------------------------------------------------
    col_rank = catalog_config["columns"]["col_rank"]
    col_ra = catalog_config["columns"]["ra"]
    col_dec = catalog_config["columns"]["dec"]
    col_z = catalog_config["columns"]["z"]

    # --- finite col_rank ---------------------------------------------------
    if col_rank in lens.colnames:
        mask_rank_finite = np.isfinite(lens[col_rank])
        lens = lens[mask_rank_finite]
        logger.info(
            "Applied finite mask on '%s': %d objects remain.",
            col_rank,
            len(lens),
        )
    else:
        logger.warning(
            "col_rank '%s' not found in lens catalog; "
            "this may cause issues with binning.",
            col_rank,
        )

    # --- redshift range ----------------------------------------------------
    redshift_range = catalog_config.get("redshift_range")
    if redshift_range is not None:
        z_min, z_max = redshift_range
        mask_z = (lens[col_z] >= z_min) & (lens[col_z] <= z_max)
        lens = lens[mask_z]
        logger.info(
            "Applied redshift mask %s on '%s': %d objects remain.",
            redshift_range,
            col_z,
            len(lens),
        )

    # --- RA range ----------------------------------------------------------
    ra_range = catalog_config.get("ra_range")
    if ra_range is not None:
        ra_min, ra_max = ra_range
        mask_ra = (lens[col_ra] >= ra_min) & (lens[col_ra] <= ra_max)
        lens = lens[mask_ra]
        logger.info(
            "Applied RA mask %s on '%s': %d objects remain.",
            ra_range,
            col_ra,
            len(lens),
        )

    # --- Dec range ---------------------------------------------------------
    dec_range = catalog_config.get("dec_range")
    if dec_range is not None:
        dec_min, dec_max = dec_range
        mask_dec = (lens[col_dec] >= dec_min) & (lens[col_dec] <= dec_max)
        lens = lens[mask_dec]
        logger.info(
            "Applied Dec mask %s on '%s': %d objects remain.",
            dec_range,
            col_dec,
            len(lens),
        )

    # --- build bin slices --------------------------------------------------
    if len(random_catalog) == 0:
        raise ValueError("No random points found in the input catalog.")

    bin_slices = build_bin_slices(
        lens=lens,
        col_rank=col_rank,
        binning_mode=binning_settings["mode"],
        col_rank_edges=binning_settings["col_rank_edges"],
        top_counts=binning_settings["top_counts"],
        top_n=binning_settings.get("top_n"),
        top_selection_order=binning_settings["top_selection_order"],
    )

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
            lens_bin=lens_bin,
            col_rank=col_rank,
            low_edge=low_edge,
            high_edge=high_edge,
            binning_mode=binning_settings["mode"],
        )
        logger.info(
            "%s (%s) -> N_total=%d, boundary: %s",
            bin_name,
            bin_desc,
            n_bin,
            boundary_text,
        )

        # --- build lens output table ---------------------------------------
        lens_out = Table()
        lens_out["ra"] = lens_bin[col_ra]
        lens_out["dec"] = lens_bin[col_dec]
        lens_out["z"] = lens_bin[col_z]
        lens_out["wsys"] = np.ones(n_bin, dtype=float)
        lens_out["bin_id"] = i

        # --- build random output table -------------------------------------
        n_random = n_bin * int(random_multiplier)
        replace_ra_dec = n_random > len(random_catalog)
        rand_idx = rng.choice(
            len(random_catalog), size=n_random, replace=replace_ra_dec
        )
        z_idx = rng.choice(n_bin, size=n_random, replace=True)

        random_ra_col = next(
            (c for c in random_catalog.colnames if c.lower() == "ra"), "ra"
        )
        random_dec_col = next(
            (c for c in random_catalog.colnames if c.lower() == "dec"), "dec"
        )

        random_out = Table()
        random_out["ra"] = random_catalog[random_ra_col][rand_idx]
        random_out["dec"] = random_catalog[random_dec_col][rand_idx]
        random_out["z"] = lens_bin[col_z][z_idx]
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

    from astropy.table import vstack

    if not lens_tables:
        return None

    global_lens_table = vstack(lens_tables)
    global_random_table = vstack(random_tables)

    return {
        "global_lens_table": global_lens_table,
        "global_random_table": global_random_table,
        "bin_metadata": bin_metadata,
    }


# ---------------------------------------------------------------------------
# Dynamic Catalog and Preparation pipeline functions
# ---------------------------------------------------------------------------


def get_latest_cluster_catalog(cluster_dir="/Users/xinq/cluster_finder/output/cluster"):
    """Find the latest timestamped cluster catalog parquet file.

    Parameters
    ----------
    cluster_dir : str or Path
        Directory where cluster catalogs are generated.

    Returns
    -------
    str
        Path to the latest cluster catalog parquet file.
    """
    import glob

    pattern = str(Path(cluster_dir) / "cluster_catalog_*.parquet")
    files = glob.glob(pattern)
    if not files:
        # Fallback default if directory doesn't exist or is empty
        return "/Users/xinq/cluster_finder/output/cluster/cluster_catalog_20260603_085729.parquet"
    return max(files, key=lambda p: Path(p).stat().st_mtime)


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

    # Determine plot limits from data
    ra_min = min(np.min(lens_out[ra_col]), np.min(random_out[ra_col]))
    ra_max = max(np.max(lens_out[ra_col]), np.max(random_out[ra_col]))
    dec_min = min(np.min(lens_out[dec_col]), np.min(random_out[dec_col]))
    dec_max = max(np.max(lens_out[dec_col]), np.max(random_out[dec_col]))

    # Add small padding
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
    else:
        fig.suptitle(f"{bin_name} {bin_desc}")
    fig.tight_layout()


def run_prepare_pipeline(
    source_name,
    catalog_sources,
    binning_mode,
    top_counts,
    top_n,
    col_rank_edges_mass,
    col_rank_edges_richness,
    top_selection_order="desc",
    random_multiplier=20,
    rng_seed=None,
    make_plots=True,
    root_path=None,
):
    """Run the lens and random preparation pipeline and write catalogs to disk."""
    if root_path is None:
        current_dir = Path.cwd().resolve()
        marker = "pyproject.toml"
        while True:
            if not current_dir or current_dir == current_dir.parent:
                break
            if (current_dir / marker).exists():
                root_path = current_dir
                break
            current_dir = current_dir.parent
        if root_path is None:
            raise FileNotFoundError("Could not find the project root.")

    if source_name not in catalog_sources:
        allowed = ", ".join(sorted(catalog_sources))
        raise ValueError(f"Unknown SOURCE '{source_name}'. Allowed: {allowed}")

    cfg = catalog_sources[source_name]

    lens_path = resolve_path(cfg["lens_path"], root_path)
    random_path = resolve_path(cfg["random_path"], root_path)

    lens_format = cfg.get("lens_format", None)
    if lens_format == "pandas_dat":
        df_lens = read_dat_to_pandas(str(lens_path))
        lens = Table.from_pandas(df_lens)
    elif lens_format:
        lens = Table.read(lens_path, format=lens_format)
    else:
        lens = Table.read(lens_path)

    random_format = cfg.get("random_format", None)
    if random_format:
        random = Table.read(random_path, format=random_format)
    else:
        random = Table.read(random_path)

    # filter by columns if exist
    if "bsm_s18a" in lens.colnames:
        mask_bsm = lens["bsm_s18a"] > 0
        lens = lens[mask_bsm]
        logger.info("Applied bsm_s18a > 0 mask: %d objects remain.", len(lens))
    else:
        logger.warning(
            "'bsm_s18a' column not found in lens catalog; expected if not s16a_redm_hsc."
        )

    if "logm_cmod" in lens.colnames:
        mask_logm = lens["logm_cmod"] >= 11.2
        lens = lens[mask_logm]
        logger.info("Applied logm_cmod >= 11.2 mask: %d objects remain.", len(lens))
    else:
        logger.warning(
            "'logm_cmod' column not found in lens catalog; expected if not s16a_redm_hsc."
        )

    col_rank = cfg["columns"]["col_rank"]
    col_ra = cfg["columns"]["ra"]
    col_dec = cfg["columns"]["dec"]
    col_z = cfg["columns"]["z"]
    lens_label = cfg["label"]

    if col_rank in lens.colnames:
        mask_rank_finite = np.isfinite(lens[col_rank])
        lens = lens[mask_rank_finite]
        logger.info(
            "Applied finite mask on col_rank '%s': %d objects remain.",
            col_rank,
            len(lens),
        )
    else:
        logger.warning(
            "col_rank '%s' not found in lens catalog; this may cause issues with binning.",
            col_rank,
        )

    redshift_range = cfg.get("redshift_range")
    if redshift_range is not None:
        z_min, z_max = redshift_range
        mask_z = (lens[col_z] >= z_min) & (lens[col_z] <= z_max)
        lens = lens[mask_z]
        logger.info(
            "Applied redshift mask %s on '%s': %d objects remain.",
            redshift_range,
            col_z,
            len(lens),
        )

    ra_range = cfg.get("ra_range")
    if ra_range is not None:
        ra_min, ra_max = ra_range
        mask_ra = (lens[col_ra] >= ra_min) & (lens[col_ra] <= ra_max)
        lens = lens[mask_ra]
        logger.info(
            "Applied RA mask %s on '%s': %d objects remain.",
            ra_range,
            col_ra,
            len(lens),
        )

    dec_range = cfg.get("dec_range")
    if dec_range is not None:
        dec_min, dec_max = dec_range
        mask_dec = (lens[col_dec] >= dec_min) & (lens[col_dec] <= dec_max)
        lens = lens[mask_dec]
        logger.info(
            "Applied Dec mask %s on '%s': %d objects remain.",
            dec_range,
            col_dec,
            len(lens),
        )

    print("-" * 80)
    print(f"Using source: {source_name}")
    print(f"Column used for ranking: {col_rank}")
    print("-" * 80)
    print(f"Redshift range: {redshift_range}")
    print(f"Lens file: {lens_path}")
    print(f"Random file: {random_path}")
    print(f"Lens columns: {lens.colnames}")
    print(f"Random columns: {random.colnames}")

    binning_settings = get_binning_settings(
        source_name,
        cfg.get("top_counts_factor", 1.0),
        binning_mode=binning_mode,
        col_rank_edges_mass=col_rank_edges_mass,
        col_rank_edges_richness=col_rank_edges_richness,
        top_counts=top_counts,
        top_selection_order=top_selection_order,
        top_n=top_n,
    )

    print("-" * 80)
    if binning_settings["mode"] == "edges":
        print(
            f"Binning mode=edges, COL_RANK_EDGES={binning_settings['col_rank_edges']}"
        )
    elif binning_settings["mode"] == "top_n":
        print(
            f"Binning mode=top_n, TOP_N={top_n}, "
            f"top_counts_factor={cfg.get('top_counts_factor', 1.0)}, "
            f"top_n(scaled)={binning_settings['top_n']}, "
            f"TOP_SELECTION_ORDER={binning_settings['top_selection_order']}"
        )
    else:
        print(
            "Binning mode=top_counts, "
            f"TOP_COUNTS(raw)={top_counts}, "
            f"top_counts_factor={cfg.get('top_counts_factor', 1.0)}, "
            f"TOP_COUNTS(scaled)={binning_settings.get('top_counts')}, "
            f"TOP_SELECTION_ORDER={binning_settings['top_selection_order']}"
        )

    output_dir = root_path / f"output/{cfg['label']}/prepare"
    output_dir.mkdir(parents=True, exist_ok=True)

    prep_result = prepare_lens_random_tables(
        lens_catalog=lens,
        random_catalog=random,
        catalog_config=cfg,
        binning_settings=binning_settings,
        random_multiplier=random_multiplier,
        rng_seed=rng_seed,
    )

    if not prep_result:
        print("No valid objects left to prepare.")
        return

    global_lens_table = prep_result["global_lens_table"]
    global_random_table = prep_result["global_random_table"]
    bin_metadata = prep_result["bin_metadata"]

    import json

    metadata_json = json.dumps(bin_metadata)
    global_lens_table.meta["BIN_META"] = metadata_json
    global_random_table.meta["BIN_META"] = metadata_json

    lens_file = output_dir / f"{lens_label}_lenses.fits"
    random_file = output_dir / f"{lens_label}_randoms.fits"

    global_lens_table.write(lens_file, overwrite=True)
    global_random_table.write(random_file, overwrite=True)

    if make_plots:
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
                    lens_label=lens_label,
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

