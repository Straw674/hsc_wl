# %%
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        print("Error: pyproject.toml not found in parent directories.")
        # Handle the error appropriately, maybe raise an exception or exit
        # For now, just break to avoid infinite loop if marker is truly missing
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        print(f"Project root found: {root_path}")  # Confirm the path found
        break
    else:
        current_dir = current_dir.parent

if root_path:
    root_path_str = str(root_path)

    if root_path_str not in sys.path:
        sys.path.append(root_path_str)

    try:
        from initial import *
    except ModuleNotFoundError as e:
        print(f"Error importing 'initial': {e}")

else:
    print("Could not proceed without finding the project root.")

# %%
# =========================
# User-editable parameters
# =========================

SOURCE = "s16a"  # Choose from CATALOG_SOURCES keys

# Path can be absolute or relative to the project root.
CATALOG_SOURCES = {
    "pdr3": {
        "label": "pdr3",
        "lens_path": "/Users/xinq/redmapper_HSC/output/redmapper_run/add_geo_mask/run/hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260421_lgt05_catalog.fit",
        "random_path": "data/random_hectomap.fits",
        "random_multiplier": 20,
        "columns": {
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
    },
    "s16a": {
        "label": "s16a",
        "lens_path": "/Users/xinq/redmapper_HSC/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits",
        "random_path": "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        "random_multiplier": 20,
        "columns": {
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
    },
    "s16a_mass": {
        "label": "s16a_mass",
        "lens_path": "/Users/xinq/redmapper_HSC/data/reference/s16a_massive_logm_11.2.fits",
        "random_path": "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        "random_multiplier": 20,
        "columns": {
            "col_rank": "logm_50_100",
            "ra": "ra",
            "dec": "dec",
            "z": "z_best",
        },
    },
    "s16a_forced": {
        "label": "s16a_forced",
        "lens_path": "/Users/xinq/redmapper_HSC/output/s16a_massive_logm_11.2_forced_results.fits",
        "random_path": "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        "random_multiplier": 20,
        "columns": {
            "col_rank": "lam",
            "ra": "ra",
            "dec": "dec",
            "z": "z_best",
        },
    },
}

# Binning mode:
# - "edges": use COL_RANK_EDGES as left-closed right-open intervals.
# - "top_counts": sort by col_rank and split sequentially by TOP_COUNTS.
BINNING_MODE = "top_counts"  # "edges" or "top_counts"

# -----------------------------------------------------------------------
# EDGES

# Used only when BINNING_MODE == "edges".
COL_RANK_EDGES_RICHNESS = [6.0, 10.0, 20.0, 35.0, 120.0]
COL_RANK_EDGES_MASS = [10.63, 10.8, 11.0, 11.2, 11.6]

# -----------------------------------------------------------------------
# TOP_COUNTS

# Used only when BINNING_MODE == "top_counts".
# Example: [x1, x2, x3, x4] means pick top x1 first, then top x2 from remaining, etc.
# TOP_COUNTS = [50, 197, 662, 1165]
# exactly same as the number in topn paper
TOP_COUNTS = [53, 196, 660, 1159]

# Used only when BINNING_MODE == "top_counts".
# A multiplier for TOP_COUNTS. Example: 2 means [x1, x2, ...] -> [2*x1, 2*x2, ...].
TOP_COUNTS_FACTOR = 1
if SOURCE == "pdr3":
    TOP_COUNTS_FACTOR = 0.810458

# Used only when BINNING_MODE == "top_counts".
# "desc": larger col_rank is better (top first); "asc": smaller col_rank is better.
TOP_SELECTION_ORDER = "desc"

# -----------------------------------------------------------------------

# Set to an integer for reproducibility (e.g., 42), or None for random seed.
RNG_SEED = None

# Whether to draw alignment plots for each bin.
MAKE_PLOTS = True


# %%
def resolve_path(path_value, root_path):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return root_path / path_obj


def gaussian_kde_1d(np, values, grid=None, num_points=256, bandwidth=None):
    """Simple Gaussian KDE implementation without scipy."""
    samples = np.asarray(values, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.array([]), np.array([])

    if grid is None:
        data_min = float(np.min(samples))
        data_max = float(np.max(samples))
        if data_min == data_max:
            pad = 1.0 if data_min == 0.0 else abs(data_min) * 0.1
            grid = np.linspace(data_min - pad, data_max + pad, num_points)
        else:
            pad = max((data_max - data_min) * 0.15, 1e-3)
            grid = np.linspace(data_min - pad, data_max + pad, num_points)
    else:
        grid = np.asarray(grid, dtype=float)

    if samples.size == 1:
        width = max(abs(samples[0]) * 0.1, 1e-3)
        density = np.exp(-0.5 * ((grid - samples[0]) / width) ** 2)
        density /= width * np.sqrt(2.0 * np.pi)
        return grid, density

    if bandwidth is None:
        std = float(np.std(samples, ddof=1))
        if not np.isfinite(std) or std <= 0.0:
            std = float(np.std(samples))
        bandwidth = 1.06 * std * (samples.size ** (-1.0 / 5.0))
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            span = float(np.max(samples) - np.min(samples))
            bandwidth = max(span / 25.0, 1e-3)

    diff = (grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * diff**2).sum(axis=1)
    density /= samples.size * bandwidth * np.sqrt(2.0 * np.pi)
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


def get_binning_settings(np, source_name):
    if BINNING_MODE not in {"edges", "top_counts"}:
        raise ValueError("BINNING_MODE must be 'edges' or 'top_counts'.")

    if BINNING_MODE == "edges":
        if source_name.endswith("mass"):
            col_rank_edges = COL_RANK_EDGES_MASS
        else:
            col_rank_edges = COL_RANK_EDGES_RICHNESS

        if len(col_rank_edges) < 2:
            raise ValueError("COL_RANK_EDGES must contain at least two values.")

        return {
            "mode": "edges",
            "col_rank_edges": col_rank_edges,
            "top_counts": None,
            "top_selection_order": None,
        }

    if not TOP_COUNTS:
        raise ValueError("TOP_COUNTS must be non-empty when BINNING_MODE='top_counts'.")

    if any((not isinstance(c, int)) or c <= 0 for c in TOP_COUNTS):
        raise ValueError("TOP_COUNTS must contain only positive integers.")

    if not np.isfinite(TOP_COUNTS_FACTOR) or TOP_COUNTS_FACTOR <= 0:
        raise ValueError("TOP_COUNTS_FACTOR must be a positive finite number.")

    scaled_top_counts = [int(round(c * TOP_COUNTS_FACTOR)) for c in TOP_COUNTS]
    if any(c <= 0 for c in scaled_top_counts):
        raise ValueError(
            "Scaled TOP_COUNTS must be positive. Increase TOP_COUNTS_FACTOR."
        )

    if TOP_SELECTION_ORDER not in {"asc", "desc"}:
        raise ValueError("TOP_SELECTION_ORDER must be 'asc' or 'desc'.")

    return {
        "mode": "top_counts",
        "col_rank_edges": None,
        "top_counts": scaled_top_counts,
        "top_selection_order": TOP_SELECTION_ORDER,
    }


def build_bin_slices(
    np,
    lens,
    col_rank,
    *,
    binning_mode,
    col_rank_edges=None,
    top_counts=None,
    top_selection_order=None,
):
    if binning_mode == "edges":
        if col_rank_edges is None or len(col_rank_edges) < 2:
            raise ValueError("COL_RANK_EDGES must contain at least two values.")

        bin_slices = []
        for i in range(len(col_rank_edges) - 1):
            low_edge = col_rank_edges[i]
            high_edge = col_rank_edges[i + 1]
            lens_mask = (lens[col_rank] >= low_edge) & (lens[col_rank] < high_edge)
            lens_bin = lens[lens_mask]
            bin_name = f"bin{i + 1}"
            bin_desc = f"{col_rank} in [{low_edge}, {high_edge})"
            bin_slices.append((bin_name, lens_bin, low_edge, high_edge, bin_desc))
        return bin_slices

    if binning_mode == "top_counts":
        if not top_counts:
            raise ValueError(
                "TOP_COUNTS must be non-empty when BINNING_MODE='top_counts'."
            )

        if any((not isinstance(c, int)) or c <= 0 for c in top_counts):
            raise ValueError("TOP_COUNTS must contain only positive integers.")

        if top_selection_order not in {"asc", "desc"}:
            raise ValueError("TOP_SELECTION_ORDER must be 'asc' or 'desc'.")

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

        # Keep selection by TOP_SELECTION_ORDER, but present bins low->high in col_rank
        # so ordering is consistent with edge-based bins.
        if top_selection_order == "desc":
            ordered_bins = raw_bins[::-1]
        else:
            ordered_bins = raw_bins

        bin_slices = []
        for i, (lens_bin, low_edge, high_edge, bin_desc) in enumerate(
            ordered_bins, start=1
        ):
            bin_name = f"bin{i}"
            bin_slices.append((bin_name, lens_bin, low_edge, high_edge, bin_desc))

        return bin_slices

    raise ValueError("BINNING_MODE must be 'edges' or 'top_counts'.")


def summarize_bin_boundaries(np, lens_bin, col_rank, low_edge, high_edge, binning_mode):
    if len(lens_bin) == 0:
        return "lower=NA, upper=NA"

    if binning_mode == "edges":
        return f"lower={low_edge}, upper={high_edge}"

    rank_vals = np.asarray(lens_bin[col_rank])
    rank_min = float(np.min(rank_vals))
    rank_max = float(np.max(rank_vals))
    return f"lower={rank_min:.6g}, upper={rank_max:.6g}"


def run_pipeline(source_name):
    if source_name not in CATALOG_SOURCES:
        allowed = ", ".join(sorted(CATALOG_SOURCES))
        raise ValueError(f"Unknown SOURCE '{source_name}'. Allowed: {allowed}")

    cfg = CATALOG_SOURCES[source_name]

    lens_path = resolve_path(cfg["lens_path"], root_path)
    random_path = resolve_path(cfg["random_path"], root_path)

    lens = Table.read(lens_path)
    random = Table.read(random_path)

    # select by bsm_s18a and logm_cmod if those columns exist, otherwise proceed with a warning
    if "bsm_s18a" in lens.colnames:
        mask_bsm = lens["bsm_s18a"] > 0
        lens = lens[mask_bsm]
        print(f"Applied bsm_s18a > 0 mask: {np.sum(mask_bsm)} objects remain.")
    else:
        print(
            "Warning: 'bsm_s18a' column not found in lens catalog; This is expected if the input catalog is not from the s16a source."
        )
    if "logm_cmod" in lens.colnames:
        mask_logm = lens["logm_cmod"] >= 11.2
        lens = lens[mask_logm]
        print(f"Applied logm_cmod >= 11.2 mask: {np.sum(mask_logm)} objects remain.")
    else:
        print(
            "Warning: 'logm_cmod' column not found in lens catalog; This is expected if the input catalog is not from the s16a source."
        )
    # avoid nan values in col_rank if it exists, since they can cause issues with binning and random selection
    col_rank = cfg["columns"]["col_rank"]
    col_ra = cfg["columns"]["ra"]
    col_dec = cfg["columns"]["dec"]
    col_z = cfg["columns"]["z"]
    lens_label = cfg["label"]

    if col_rank in lens.colnames:
        mask_rank_finite = np.isfinite(lens[col_rank])
        lens = lens[mask_rank_finite]
        print(
            f"Applied finite mask on col_rank '{col_rank}': {np.sum(mask_rank_finite)} objects remain."
        )
    else:
        print(
            f"Warning: col_rank '{col_rank}' not found in lens catalog; This may cause issues with binning and random selection."
        )
    print("-" * 80)
    print(f"Using source: {source_name}")
    print(f"Column used for ranking: {col_rank}")
    print("-" * 80)
    print(f"Lens file: {lens_path}")
    print(f"Random file: {random_path}")
    print(f"Lens columns: {lens.colnames}")
    print(f"Random columns: {random.colnames}")

    binning_settings = get_binning_settings(np, source_name)

    print("-" * 80)
    if binning_settings["mode"] == "edges":
        print(
            f"Binning mode=edges, COL_RANK_EDGES={binning_settings['col_rank_edges']}"
        )
    else:
        print(
            "Binning mode=top_counts, "
            f"TOP_COUNTS(raw)={TOP_COUNTS}, "
            f"TOP_COUNTS_FACTOR={TOP_COUNTS_FACTOR}, "
            f"TOP_COUNTS(scaled)={binning_settings['top_counts']}, "
            f"TOP_SELECTION_ORDER={binning_settings['top_selection_order']}"
        )

    rng = np.random.default_rng(RNG_SEED)

    output_dir = root_path / f"output/{cfg['label']}/prepare"
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(random) == 0:
        raise ValueError("No random points found in the input catalog.")

    bin_slices = build_bin_slices(
        np=np,
        lens=lens,
        col_rank=col_rank,
        binning_mode=binning_settings["mode"],
        col_rank_edges=binning_settings["col_rank_edges"],
        top_counts=binning_settings["top_counts"],
        top_selection_order=binning_settings["top_selection_order"],
    )

    saved_lens_files = []
    saved_random_files = []

    for bin_name, lens_bin, low_edge, high_edge, bin_desc in bin_slices:
        n_bin = len(lens_bin)
        if n_bin == 0:
            print(f"{bin_name} has 0 objects; skip writing. ({bin_desc})")
            continue

        boundary_text = summarize_bin_boundaries(
            np=np,
            lens_bin=lens_bin,
            col_rank=col_rank,
            low_edge=low_edge,
            high_edge=high_edge,
            binning_mode=binning_settings["mode"],
        )

        print("-" * 80)
        print(f"{bin_name} ({bin_desc}) -> N_total={n_bin}")
        print(f"{bin_name} rank boundary -> {boundary_text}")

        lens_out = Table()
        lens_out["ra"] = lens_bin[col_ra]
        lens_out["dec"] = lens_bin[col_dec]
        lens_out["z"] = lens_bin[col_z]
        lens_out["wsys"] = np.ones(n_bin, dtype=float)

        n_random = n_bin * int(cfg["random_multiplier"])
        replace_ra_dec = n_random > len(random)
        rand_idx = rng.choice(len(random), size=n_random, replace=replace_ra_dec)
        z_idx = rng.choice(n_bin, size=n_random, replace=True)

        random_out = Table()
        random_out["ra"] = random[col_ra][rand_idx]
        random_out["dec"] = random[col_dec][rand_idx]
        random_out["z"] = lens_bin[col_z][z_idx]
        random_out["wsys"] = np.ones(n_random, dtype=float)

        lens_file = output_dir / f"{lens_label}_{bin_name}.fits"
        random_file = output_dir / f"{lens_label}_random_{bin_name}.fits"

        lens_out.write(lens_file, overwrite=True)
        random_out.write(random_file, overwrite=True)

        saved_lens_files.append(str(lens_file))
        saved_random_files.append(str(random_file))

        if MAKE_PLOTS:
            show_alignment_plot(
                plt=plt,
                lens_out=lens_out,
                random_out=random_out,
                lens_label=lens_label,
                ra_col="ra",
                dec_col="dec",
                z_col="z",
                bin_name=bin_name,
                low_edge=low_edge,
                high_edge=high_edge,
                bin_desc=bin_desc,
            )

    print("\n" + "=" * 30)

    print(f"Lenses saved to: {', '.join(saved_lens_files)}")
    print(f"Randoms saved to: {', '.join(saved_random_files)}")


if __name__ == "__main__":
    run_pipeline(SOURCE)
