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

s16a = Table.read(
    "/Users/xinq/cluster_finder/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits"
)

print(s16a.colnames)
# ['mem_match_id', 'ra', 'dec', 'z', 'refmag', 'refmag_err', 'lambda', 'lambda_e', 'z_lambda', 'z_lambda_e', 'cg_spec_z', 'z_spec_init', 'z_init', 'r_lambda', 'r_mask', 'zred', 'zred_e', 'ra_orig', 'dec_orig', 'ra_cent_1', 'ra_cent_2', 'ra_cent_3', 'ra_cent_4', 'ra_cent_5', 'dec_cent_1', 'dec_cent_2', 'dec_cent_3', 'dec_cent_4', 'dec_cent_5', 'id_cent_1', 'id_cent_2', 'id_cent_3', 'id_cent_4', 'id_cent_5', 'p_cen_1', 'p_cen_2', 'p_cen_3', 'p_cen_4', 'p_cen_5', 'flag']


random = Table.read(root_path / "data/random_hectomap.fits")
print(random.colnames)
# ['object_id', 'ra', 'dec']

# %%
# Lambda bins are left-closed right-open: bin1: [6, 10), bin2: [10, 20), bin3: [20, 35), bin4: [35, inf)
lambda_edges = [6.0, 10.0, 20.0, 35.0]
ra_min, ra_max = 210.0, 250.0
dec_min, dec_max = 42.0, 44.5


def calculate_celestial_area(ra1, ra2, dec1, dec2):
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


def max_cdf_diff(a, b):
    """Kolmogorov-like max CDF distance without scipy."""
    a_sorted = np.sort(np.asarray(a))
    b_sorted = np.sort(np.asarray(b))
    grid = np.unique(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, grid, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, grid, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def show_alignment_plot(s16a_out, random_out, bin_name, low_edge, high_edge):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    h1 = axes[0, 0].hexbin(
        s16a_out["ra"],
        s16a_out["dec"],
        gridsize=45,
        extent=(ra_min, ra_max, dec_min, dec_max),
        mincnt=1,
        cmap="Blues",
    )
    axes[0, 0].set_title("s16a footprint")
    axes[0, 0].set_xlabel("RA")
    axes[0, 0].set_ylabel("Dec")
    fig.colorbar(h1, ax=axes[0, 0], label="Counts")

    h2 = axes[0, 1].hexbin(
        random_out["ra"],
        random_out["dec"],
        gridsize=45,
        extent=(ra_min, ra_max, dec_min, dec_max),
        mincnt=1,
        cmap="Blues",
    )
    axes[0, 1].set_title("Random footprint")
    axes[0, 1].set_xlabel("RA")
    axes[0, 1].set_ylabel("Dec")
    fig.colorbar(h2, ax=axes[0, 1], label="Counts")

    axes[1, 0].hist(
        s16a_out["z"],
        bins="auto",
        density=True,
        alpha=0.6,
        label="s16a z",
        color="tab:blue",
    )
    axes[1, 0].hist(
        random_out["z"],
        bins="auto",
        density=True,
        alpha=0.5,
        label="Random z",
        color="tab:orange",
    )
    axes[1, 0].set_title("z distribution")
    axes[1, 0].set_xlabel("z")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()

    s16a_ra_region = s16a_out["ra"][
        (s16a_out["ra"] >= ra_min) & (s16a_out["ra"] <= ra_max)
    ]
    axes[1, 1].hist(
        s16a_ra_region,
        bins="auto",
        density=True,
        alpha=0.6,
        label="s16a RA (region)",
        color="tab:green",
    )
    axes[1, 1].hist(
        random_out["ra"],
        bins="auto",
        density=True,
        alpha=0.5,
        label="Random RA",
        color="tab:red",
    )
    axes[1, 1].set_title("RA distribution in target region")
    axes[1, 1].set_xlabel("RA")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    fig.suptitle(f"{bin_name}  lambda in [{low_edge}, {high_edge})")
    fig.tight_layout()


# %%
sky_area_deg2 = calculate_celestial_area(ra_min, ra_max, dec_min, dec_max)
print(f"Sky area of the target region: {sky_area_deg2:.4f} deg^2")

# Reproducible random sampling.
rng = np.random.default_rng()

output_dir = root_path / "output" / "prepare_random_s16a"
output_dir.mkdir(parents=True, exist_ok=True)

# Precompute sky-region mask on the random catalog to keep random points in target area.
random_region_mask = (
    (random["ra"] >= ra_min)
    & (random["ra"] <= ra_max)
    & (random["dec"] >= dec_min)
    & (random["dec"] <= dec_max)
)
random_region = random[random_region_mask]

if len(random_region) == 0:
    raise ValueError("No random points found in the target RA/Dec region.")

bin_definitions = [
    ("bin1", lambda_edges[0], lambda_edges[1]),
    ("bin2", lambda_edges[1], lambda_edges[2]),
    ("bin3", lambda_edges[2], lambda_edges[3]),
    ("bin4", lambda_edges[3], np.inf),
]

for bin_name, low_edge, high_edge in bin_definitions:
    # Left-closed right-open bin selection.
    s16a_mask = (s16a["lambda"] >= low_edge) & (s16a["lambda"] < high_edge)
    s16a_bin = s16a[s16a_mask]

    n_bin = len(s16a_bin)
    if n_bin == 0:
        print(f"{bin_name} ({low_edge}, {high_edge}] has 0 objects; skip writing.")
        continue

    # Density in the requested sky window.
    in_region = (
        (s16a_bin["ra"] >= ra_min)
        & (s16a_bin["ra"] <= ra_max)
        & (s16a_bin["dec"] >= dec_min)
        & (s16a_bin["dec"] <= dec_max)
    )
    n_in_region = int(np.sum(in_region))
    density = n_in_region / sky_area_deg2

    print(
        f"{bin_name} ({low_edge}, {high_edge}] -> "
        f"N_total={n_bin}, N_region={n_in_region}, density={density:.6f} deg^-2"
    )

    # Build the s16a output table with required columns.
    s16a_out = Table()
    s16a_out["ra"] = s16a_bin["ra"]
    s16a_out["dec"] = s16a_bin["dec"]
    s16a_out["z"] = s16a_bin["z"]
    s16a_out["wsys"] = np.ones(n_bin, dtype=float)

    # Build the random catalog with 10x size and z sampled from s16a z distribution.
    n_random = n_bin * 10
    replace_ra_dec = n_random > len(random_region)
    rand_idx = rng.choice(len(random_region), size=n_random, replace=replace_ra_dec)
    z_idx = rng.choice(n_bin, size=n_random, replace=True)

    random_out = Table()
    random_out["ra"] = random_region["ra"][rand_idx]
    random_out["dec"] = random_region["dec"][rand_idx]
    random_out["z"] = s16a_bin["z"][z_idx]
    random_out["wsys"] = np.ones(n_random, dtype=float)

    # Diagnostics for redshift-alignment between s16a and random.
    z_mean_diff = abs(np.mean(s16a_out["z"]) - np.mean(random_out["z"]))
    z_cdf_diff = max_cdf_diff(s16a_out["z"], random_out["z"])

    s16a_file = output_dir / f"s16a_{bin_name}.fits"
    random_file = output_dir / f"random_{bin_name}.fits"

    s16a_out.write(s16a_file, overwrite=True)
    random_out.write(random_file, overwrite=True)
    show_alignment_plot(s16a_out, random_out, bin_name, low_edge, high_edge)

    print(f"s16a Saved: {s16a_file}")
    print(f"Random Saved: {random_file}")
    print(
        f"{bin_name} alignment check -> "
        f"N_s16a={n_bin}, N_random={n_random}, "
        f"|mean(z)_diff|={z_mean_diff:.6e}, max_cdf_diff={z_cdf_diff:.6e}"
    )

# %%

s16a_bin4 = Table.read(output_dir / "s16a_bin4.fits")
print(len(s16a_bin4))
