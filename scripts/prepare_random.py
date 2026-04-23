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

SOURCE = "s16a"  # Supported: "pdr3", "s16a"

# Path can be absolute or relative to the project root.
CATALOG_SOURCES = {
    "pdr3": {
        "label": "pdr3",
        "lens_path": "/Users/xinq/redmapper_HSC/output/redmapper_run/add_geo_mask/run/hsc_run_redmapper_v0.9.1.dev2+g030802198_lgt05_catalog.fit",
        "random_path": "data/random_hectomap.fits",
        "random_multiplier": 1000,
        "columns": {
            "lambda": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z",
        },
    },
    "s16a": {
        "label": "s16a",
        "lens_path": "/Users/xinq/cluster_finder/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits",
        "random_path": "data/random_hectomap.fits",
        "random_multiplier": 10,
        "columns": {
            "lambda": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z",
        },
    },
}

# Shared bin definition: left-closed right-open.
LAMBDA_EDGES = [6.0, 10.0, 20.0, 35.0]

# Sky region limits.
RA_MIN, RA_MAX = 235, 250.0
DEC_MIN, DEC_MAX = 42.0, 44.5

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


def calculate_celestial_area(np, ra1, ra2, dec1, dec2):
    """Area (deg^2) of a spherical lon/lat rectangle."""
    if ra1 > ra2:
        ra1, ra2 = ra2, ra1
    if dec1 > dec2:
        dec1, dec2 = dec2, dec1
    delta_ra_rad = np.deg2rad(ra2 - ra1)
    dec1_rad = np.deg2rad(dec1)
    dec2_rad = np.deg2rad(dec2)
    area_sr = delta_ra_rad * (np.sin(dec2_rad) - np.sin(dec1_rad))
    area = area_sr * (180.0 / np.pi) ** 2
    return abs(area)


def max_cdf_diff(np, a, b):
    """Kolmogorov-like max CDF distance without scipy."""
    a_sorted = np.sort(np.asarray(a))
    b_sorted = np.sort(np.asarray(b))
    grid = np.unique(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, grid, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, grid, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


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
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    lens_count = len(lens_out)
    random_count = len(random_out)

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
    axes[0].set_xlim(RA_MIN, RA_MAX)
    axes[0].set_ylim(DEC_MIN, DEC_MAX)

    h2 = axes[1].hexbin(
        random_out[ra_col],
        random_out[dec_col],
        gridsize=45,
        extent=(RA_MIN, RA_MAX, DEC_MIN, DEC_MAX),
        mincnt=1,
        cmap="Blues",
    )
    axes[1].set_title(f"Random footprint (N={random_count})")
    axes[1].set_xlabel("RA")
    axes[1].set_ylabel("Dec")
    axes[1].set_xlim(RA_MIN, RA_MAX)
    axes[1].set_ylim(DEC_MIN, DEC_MAX)
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

    title_high = "+inf" if high_edge == float("inf") else f"{high_edge}"
    fig.suptitle(f"{bin_name} lambda in [{low_edge}, {title_high})")
    fig.tight_layout()


def run_pipeline(source_name):
    if source_name not in CATALOG_SOURCES:
        allowed = ", ".join(sorted(CATALOG_SOURCES))
        raise ValueError(f"Unknown SOURCE '{source_name}'. Allowed: {allowed}")

    cfg = CATALOG_SOURCES[source_name]

    lens_path = resolve_path(cfg["lens_path"], root_path)
    random_path = resolve_path(cfg["random_path"], root_path)

    lens = Table.read(lens_path)
    random = Table.read(random_path)

    col_lambda = cfg["columns"]["lambda"]
    col_ra = cfg["columns"]["ra"]
    col_dec = cfg["columns"]["dec"]
    col_z = cfg["columns"]["z"]
    lens_label = cfg["label"]

    print(f"Using source: {source_name}")
    print(f"Lens file: {lens_path}")
    print(f"Random file: {random_path}")
    print(f"Lens columns: {lens.colnames}")
    print(f"Random columns: {random.colnames}")

    sky_area_deg2 = calculate_celestial_area(np, RA_MIN, RA_MAX, DEC_MIN, DEC_MAX)
    print(f"Sky area of the target region: {sky_area_deg2:.4f} deg^2")

    rng = np.random.default_rng(RNG_SEED)

    output_dir = root_path / f"output/{cfg['label']}/prepare"
    output_dir.mkdir(parents=True, exist_ok=True)

    random_region_mask = (
        (random[col_ra] >= RA_MIN)
        & (random[col_ra] <= RA_MAX)
        & (random[col_dec] >= DEC_MIN)
        & (random[col_dec] <= DEC_MAX)
    )
    random_region = random[random_region_mask]

    if len(random_region) == 0:
        raise ValueError("No random points found in the target RA/Dec region.")

    bin_definitions = [
        ("bin1", LAMBDA_EDGES[0], LAMBDA_EDGES[1]),
        ("bin2", LAMBDA_EDGES[1], LAMBDA_EDGES[2]),
        ("bin3", LAMBDA_EDGES[2], LAMBDA_EDGES[3]),
        ("bin4", LAMBDA_EDGES[3], np.inf),
    ]

    for bin_name, low_edge, high_edge in bin_definitions:
        lens_mask = (lens[col_lambda] >= low_edge) & (lens[col_lambda] < high_edge)
        lens_bin = lens[lens_mask]

        n_bin = len(lens_bin)
        if n_bin == 0:
            print(f"{bin_name} [{low_edge}, {high_edge}) has 0 objects; skip writing.")
            continue

        in_region = (
            (lens_bin[col_ra] >= RA_MIN)
            & (lens_bin[col_ra] <= RA_MAX)
            & (lens_bin[col_dec] >= DEC_MIN)
            & (lens_bin[col_dec] <= DEC_MAX)
        )
        n_in_region = int(np.sum(in_region))
        density = n_in_region / sky_area_deg2

        print(
            f"{bin_name} [{low_edge}, {high_edge}) -> "
            f"N_total={n_bin}, N_region={n_in_region}, density={density:.6f} deg^-2"
        )

        lens_out = Table()
        lens_out["ra"] = lens_bin[col_ra]
        lens_out["dec"] = lens_bin[col_dec]
        lens_out["z"] = lens_bin[col_z]
        lens_out["wsys"] = np.ones(n_bin, dtype=float)

        n_random = n_bin * int(cfg["random_multiplier"])
        replace_ra_dec = n_random > len(random_region)
        rand_idx = rng.choice(len(random_region), size=n_random, replace=replace_ra_dec)
        z_idx = rng.choice(n_bin, size=n_random, replace=True)

        random_out = Table()
        random_out["ra"] = random_region[col_ra][rand_idx]
        random_out["dec"] = random_region[col_dec][rand_idx]
        random_out["z"] = lens_bin[col_z][z_idx]
        random_out["wsys"] = np.ones(n_random, dtype=float)

        z_mean_diff = abs(np.mean(lens_out["z"]) - np.mean(random_out["z"]))
        z_cdf_diff = max_cdf_diff(np, lens_out["z"], random_out["z"])

        lens_file = output_dir / f"{lens_label}_{bin_name}.fits"
        random_file = output_dir / f"{lens_label}_random_{bin_name}.fits"

        lens_out.write(lens_file, overwrite=True)
        random_out.write(random_file, overwrite=True)

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
            )

        print(f"{lens_label} Saved: {lens_file}")
        print(f"Random Saved: {random_file}")
        print(
            f"{bin_name} alignment check -> "
            f"N_{lens_label}={n_bin}, N_random={n_random}, "
            f"|mean(z)_diff|={z_mean_diff:.6e}, max_cdf_diff={z_cdf_diff:.6e}"
        )


if __name__ == "__main__":
    run_pipeline(SOURCE)
