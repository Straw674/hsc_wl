# %%
import sys
import logging
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        logger.error("Error: pyproject.toml not found in parent directories.")
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        logger.info(f"Project root found: {root_path}")  # Confirm the path found
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
        logger.error(f"Error importing 'initial': {e}")

else:
    print("Could not proceed without finding the project root.")

# %%
REGION_RA_MIN = 200.0
REGION_RA_MAX = 255.0
REGION_DEC_MIN = 42.0
REGION_DEC_MAX = 44.5
MATCH_TOLERANCE_ARCSEC = 1.0


# %%
def _read_shape_catalogs(root):
    shape1 = Table.read(root / "data/hsc_y3.fits")
    print(len(shape1), shape1.colnames)
    # 25260877 ['RA', 'Dec', 'e_1', 'e_2', 'e_rms', 'weight', 'm_corr', 'c_1', 'c_2', 'resolution', 'aperture_mag', 'z_bin', 'e1_psf', 'e2_psf']

    shape2 = Table.read(
        root / "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_source.fits"
    )
    print(len(shape2), shape2.colnames)
    # ['z', 'ra', 'dec', 'w', 'e_1', 'e_2', 'z_low', 'm', 'e_rms', 'R_2', 'field', 'z_upp', 'z_l_max']
    return shape1, shape2


def _filter_shape1(shape1):
    mask = (
        (shape1["i_ra"] >= REGION_RA_MIN)
        & (shape1["i_ra"] <= REGION_RA_MAX)
        & (shape1["i_dec"] >= REGION_DEC_MIN)
        & (shape1["i_dec"] <= REGION_DEC_MAX)
    )
    return shape1[mask]


def _match_by_skycoord(shape1_region, shape2_region):
    coord1 = SkyCoord(
        ra=shape1_region["i_ra"] * u.deg, dec=shape1_region["i_dec"] * u.deg
    )
    coord2 = SkyCoord(ra=shape2_region["ra"] * u.deg, dec=shape2_region["dec"] * u.deg)
    match_idx, sep2d, _ = coord1.match_to_catalog_sky(coord2)
    match_mask = sep2d.arcsec <= MATCH_TOLERANCE_ARCSEC

    matched_shape1 = shape1_region[match_mask]
    matched_shape2 = shape2_region[match_idx[match_mask]]
    matched_sep = sep2d[match_mask]
    return matched_shape1, matched_shape2, matched_sep, match_mask


def _column_summary(left, right):
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)

    valid = np.isfinite(left_arr) & np.isfinite(right_arr)
    n_total = int(len(left_arr))
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return {
            "n_total": n_total,
            "n_valid": 0,
            "exact_frac": np.nan,
            "close_frac": np.nan,
            "mean_abs_diff": np.nan,
            "median_abs_diff": np.nan,
            "max_abs_diff": np.nan,
        }

    left_valid = left_arr[valid]
    right_valid = right_arr[valid]
    diffs = np.abs(left_valid - right_valid)

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "exact_frac": float(np.mean(left_valid == right_valid)),
        "close_frac": float(
            np.mean(np.isclose(left_valid, right_valid, rtol=0.0, atol=1e-12))
        ),
        "mean_abs_diff": float(np.mean(diffs)),
        "median_abs_diff": float(np.median(diffs)),
        "max_abs_diff": float(np.max(diffs)),
    }


def _print_summary_table(records):
    summary = Table(
        rows=records,
        names=(
            "shape1_col",
            "shape2_col",
            "n_valid",
            "exact_frac",
            "close_frac",
            "mean_abs_diff",
            "median_abs_diff",
            "max_abs_diff",
        ),
    )
    print(summary)
    return summary


def _plot_e2_hist2d(matched_shape1, matched_shape2):
    left = np.asarray(matched_shape1["i_hsmshaperegauss_e2"], dtype=float)
    right = np.asarray(matched_shape2["e_2"], dtype=float)

    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]

    if len(left) == 0:
        logger.warning("No valid e2 pairs available for hist2d plot.")
        return None

    diff = right - left
    abs_max = float(np.nanmax(np.abs(np.concatenate([left, right]))))
    if not np.isfinite(abs_max) or abs_max == 0.0:
        abs_max = 1.0

    diff_low, diff_high = np.nanpercentile(diff, [1, 99])
    if not np.isfinite(diff_low) or not np.isfinite(diff_high) or diff_low == diff_high:
        diff_low, diff_high = float(np.nanmin(diff)), float(np.nanmax(diff))
        if diff_low == diff_high:
            diff_low -= 1.0
            diff_high += 1.0

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)

    hist = ax.hist2d(
        left,
        right,
        bins=200,
        range=[[-abs_max, abs_max], [-abs_max, abs_max]],
        cmap="viridis",
        norm=LogNorm(),
    )
    ax.plot([-abs_max, abs_max], [-abs_max, abs_max], color="white", lw=1.2)
    ax.set_xlabel("shape1 e_2")
    ax.set_ylabel("shape2 e_2")
    ax.set_title("e2 comparison (hist2d)")
    fig.colorbar(hist[3], ax=ax, label="count")
    plt.show()

    plt.close(fig)


# %%
shape1, shape2 = _read_shape_catalogs(root_path)

shape1_region = _filter_shape1(shape1)
shape2_region = shape2

print(
    f"shape1 region filter: {len(shape1_region)} / {len(shape1)} rows remain "
    f"for RA in [{REGION_RA_MIN}, {REGION_RA_MAX}] and Dec in [{REGION_DEC_MIN}, {REGION_DEC_MAX}]"
)

matched_shape1, matched_shape2, matched_sep, match_mask = _match_by_skycoord(
    shape1_region, shape2_region
)

print(
    f"matched pairs within {MATCH_TOLERANCE_ARCSEC:.1f} arcsec: "
    f"{len(matched_shape1)} / {len(shape1_region)} shape1 rows"
)
print(f"unmatched shape1 rows in region: {len(shape1_region) - len(matched_shape1)}")

sep_arcsec = np.asarray(matched_sep.arcsec, dtype=float)
print(
    "separation stats [arcsec]: "
    f"min={np.min(sep_arcsec):.6f}, "
    f"median={np.median(sep_arcsec):.6f}, "
    f"p95={np.percentile(sep_arcsec, 95):.6f}, "
    f"max={np.max(sep_arcsec):.6f}"
)

# %%
column_pairs = [
    ("i_ra", "ra"),
    ("i_dec", "dec"),
    ("i_hsmshaperegauss_e1", "e_1"),
    ("i_hsmshaperegauss_e2", "e_2"),
    ("i_hsmshaperegauss_derived_rms_e", "e_rms"),
    ("i_hsmshaperegauss_derived_weight", "w"),
    ("i_hsmshaperegauss_derived_shear_bias_m", "m"),
]

summary_records = []
for left_col, right_col in column_pairs:
    stats = _column_summary(matched_shape1[left_col], matched_shape2[right_col])
    summary_records.append(
        (
            left_col,
            right_col,
            stats["n_valid"],
            stats["exact_frac"],
            stats["close_frac"],
            stats["mean_abs_diff"],
            stats["median_abs_diff"],
            stats["max_abs_diff"],
        )
    )

print("\nColumn comparison summary:")
summary_table = _print_summary_table(summary_records)

print("\nOverall match summary:")
overall_table = Table(
    rows=[
        ("shape1_region_rows", len(shape1_region)),
        ("matched_rows", len(matched_shape1)),
        (
            "match_fraction",
            float(len(matched_shape1) / len(shape1_region))
            if len(shape1_region)
            else np.nan,
        ),
        ("unmatched_rows", len(shape1_region) - len(matched_shape1)),
        (
            "median_sep_arcsec",
            float(np.median(sep_arcsec)) if len(sep_arcsec) else np.nan,
        ),
        ("max_sep_arcsec", float(np.max(sep_arcsec)) if len(sep_arcsec) else np.nan),
    ],
    names=("metric", "value"),
)
print(overall_table)


_plot_e2_hist2d(matched_shape1, matched_shape2)
