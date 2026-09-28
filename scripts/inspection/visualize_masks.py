# %%
import sys
from pathlib import Path

# Dynamically locate the project root using pyproject.toml as a marker
project_root = Path(__file__).resolve().parent
while (
    project_root != project_root.parent
    and not (project_root / "pyproject.toml").exists()
):
    project_root = project_root.parent

if not (project_root / "pyproject.toml").exists():
    raise RuntimeError(
        "Could not find project root (containing pyproject.toml) in any parent directory."
    )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hsc_wl.coverage import (
    REFERENCE_Z_RANGE,
    build_config_mask,
    group_configs_by_mask,
    in_field,
    load_s16a_pixset_1024,
    load_s23b_cut_mask,
    load_y3_mask,
    s23b_cut_field_areas_deg2,
    volume_factor,
)
from initial import *

# %%
# Local Functions


def build_fullsky_mask_images(
    mask_groups: list[dict],
    masks: dict[str, hsp.HealSparseMap],
    w_full: WCS,
    shape_full: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Precompute full-sky 2D HPX-projected image arrays for each mask group."""
    images_full = {}
    for group in mask_groups:
        label = group["label"]
        images_full[label] = project_healsparse_to_wcs(masks[label], w_full, shape_full)
    return images_full


def build_regional_layer_image(
    mask_y3: hsp.HealSparseMap,
    mask_s23b: hsp.HealSparseMap,
    s16a_pix: set[int],
    wcs: WCS,
    shape: tuple[int, int],
    include_sliver: bool = False,
) -> np.ndarray:
    """Build a 2D composite image for a region distinguishing Y3 (S19A), S23B, S16A, and optional sliver."""
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    finite = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)

    v_y3 = mask_y3.get_values_pos(ra[finite], dec[finite], lonlat=True)
    v_s23b = mask_s23b.get_values_pos(ra[finite], dec[finite], lonlat=True)
    pix_1024 = hp.ang2pix(1024, ra[finite], dec[finite], nest=True, lonlat=True)
    v_s16a = np.isin(pix_1024, list(s16a_pix)) & v_y3

    img = np.full((ny, nx), np.nan, dtype=float)
    vals = np.zeros(np.sum(finite), dtype=float)
    # Layer 1: S19A Shape Catalog (HSC Y3)
    vals[v_y3] = 1.0
    # Layer 2: S23B Cut Photometric Catalog
    vals[v_s23b] = 2.0
    # Layer 3: S16A Baseline (Random Catalog overlap)
    vals[v_s16a] = 3.0
    if include_sliver:
        is_sliver = v_y3 & (ra[finite] > 250.0)
        vals[is_sliver] = 4.0

    img[finite] = vals
    return img


def build_version_single_images(
    mask_y3: hsp.HealSparseMap,
    mask_s23b: hsp.HealSparseMap,
    s16a_pix: set[int],
    wcs: WCS,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 2D binary projection arrays for S16A, S19A, and S23B individually."""
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    finite = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)

    v_y3 = mask_y3.get_values_pos(ra[finite], dec[finite], lonlat=True)
    v_s23b = mask_s23b.get_values_pos(ra[finite], dec[finite], lonlat=True)
    pix_1024 = hp.ang2pix(1024, ra[finite], dec[finite], nest=True, lonlat=True)
    v_s16a = np.isin(pix_1024, list(s16a_pix)) & v_y3

    img_s16a = np.full((ny, nx), np.nan, dtype=float)
    img_s19a = np.full((ny, nx), np.nan, dtype=float)
    img_s23b = np.full((ny, nx), np.nan, dtype=float)

    vals_s16a = np.zeros(np.sum(finite), dtype=float)
    vals_s19a = np.zeros(np.sum(finite), dtype=float)
    vals_s23b = np.zeros(np.sum(finite), dtype=float)

    vals_s16a[v_s16a] = 1.0
    vals_s19a[v_y3] = 1.0
    vals_s23b[v_s23b] = 1.0

    img_s16a[finite] = vals_s16a
    img_s19a[finite] = vals_s19a
    img_s23b[finite] = vals_s23b

    return img_s16a, img_s19a, img_s23b


def plot_combined_overview(
    mask_groups: list[dict],
    images_full: dict[str, np.ndarray],
    regional_data: list[dict],
    w_full: WCS,
    output_path: Path,
    palette: list[str],
) -> plt.Figure:
    """Plot the integrated overview: full-sky masks on the left and 3 regional fields on the right."""
    fig = plt.figure(figsize=(20, 12))
    subfigs = fig.subfigures(1, 2, width_ratios=[1.0, 1.45], wspace=0.06)

    # ------------------------------------------------------------------
    # Left Subfigure: Full-Sky HPX Panels
    # ------------------------------------------------------------------
    n_groups = len(mask_groups)
    subfigs[0].subplots_adjust(top=0.97, bottom=0.03, hspace=0.38)
    for i, group in enumerate(mask_groups):
        label = group["label"]
        color = palette[i % len(palette)]
        cmap = mpl.colors.ListedColormap(["#f2f4f7", color])
        cmap.set_bad("white")

        ax = subfigs[0].add_subplot(n_groups, 1, i + 1, projection=w_full)
        ax.imshow(images_full[label], origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)
        ax.coords["ra"].set_axislabel("RA", fontsize=8)
        ax.coords["dec"].set_axislabel("Dec", fontsize=8)
        ax.set_title(
            f"{group['title']} (Full Sky HPX)\n"
            f"Area: {group['area_deg2']:.2f} deg², Factor: {group['volume_factor']:.4f} "
            f"({len(group['all_config_names'])} configs)",
            fontsize=8.0,
            fontweight="normal",
            pad=2,
        )

    # ------------------------------------------------------------------
    # Right Subfigure: 3 Regional HPX Panels (HECTOMAP, SPRING, FALL)
    # ------------------------------------------------------------------
    height_ratios = [reg["shape"][0] for reg in regional_data]
    gs_right = subfigs[1].add_gridspec(
        3,
        1,
        height_ratios=height_ratios,
        hspace=0.48,
        top=0.96,
        bottom=0.04,
        left=0.05,
        right=0.98,
    )

    cmap_reg = mpl.colors.ListedColormap(["#f2f4f7", "#9ecae1", "#1f78b4", "#2ca02c"])
    cmap_reg.set_bad("white")

    cmap_hecto = mpl.colors.ListedColormap(
        ["#f2f4f7", "#9ecae1", "#1f78b4", "#2ca02c", "#d62728"]
    )
    cmap_hecto.set_bad("white")

    for j, reg in enumerate(regional_data):
        ax = subfigs[1].add_subplot(gs_right[j], projection=reg["wcs"])
        cmap_use = cmap_hecto if reg.get("has_sliver", False) else cmap_reg
        vmax_use = 4.0 if reg.get("has_sliver", False) else 3.0

        ax.imshow(reg["img"], origin="lower", cmap=cmap_use, vmin=0.0, vmax=vmax_use)
        ax.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)

        if reg["ra_unit"] == "deg":
            ax.coords["ra"].set_major_formatter("d")
            ax.coords["ra"].set_ticks(spacing=reg["ra_spacing"] * u.deg)
            ax.coords["ra"].set_ticks_position("b")
            ax.coords["ra"].set_ticklabel_position("b")
            ax.coords["ra"].set_axislabel("RA [deg]", fontsize=8.5)
        else:
            ax.coords["ra"].set_ticks(spacing=reg["ra_spacing"] * 15 * u.deg)
            ax.coords["ra"].set_ticks_position("b")
            ax.coords["ra"].set_ticklabel_position("b")
            ax.coords["ra"].set_axislabel("RA", fontsize=8.5)

        ax.coords["dec"].set_ticks(spacing=reg["dec_spacing"] * u.deg)
        ax.coords["dec"].set_ticks_position("l")
        ax.coords["dec"].set_ticklabel_position("l")
        ax.coords["dec"].set_axislabel("Dec", fontsize=8.5)
        ax.set_title(reg["title"], fontsize=8.0, fontweight="normal", pad=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig


def plot_standalone_regional_fields(
    regional_data: list[dict],
    output_path: Path,
) -> plt.Figure:
    """Plot a dedicated, high-resolution standalone figure of the 3 survey regions with 3-version composite layers."""
    fig = plt.figure(figsize=(15, 8.8))
    height_ratios = [reg["shape"][0] for reg in regional_data]
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=height_ratios,
        hspace=0.45,
        top=0.93,
        bottom=0.10,
        left=0.07,
        right=0.97,
    )

    cmap_reg = mpl.colors.ListedColormap(["#f2f4f7", "#9ecae1", "#1f78b4", "#2ca02c"])
    cmap_reg.set_bad("white")

    cmap_hecto = mpl.colors.ListedColormap(
        ["#f2f4f7", "#9ecae1", "#1f78b4", "#2ca02c", "#d62728"]
    )
    cmap_hecto.set_bad("white")

    for j, reg in enumerate(regional_data):
        ax = fig.add_subplot(gs[j], projection=reg["wcs"])
        cmap_use = cmap_hecto if reg.get("has_sliver", False) else cmap_reg
        vmax_use = 4.0 if reg.get("has_sliver", False) else 3.0

        ax.imshow(reg["img"], origin="lower", cmap=cmap_use, vmin=0.0, vmax=vmax_use)
        ax.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)

        if reg["ra_unit"] == "deg":
            ax.coords["ra"].set_major_formatter("d")
            ax.coords["ra"].set_ticks(spacing=reg["ra_spacing"] * u.deg)
            ax.coords["ra"].set_ticks_position("b")
            ax.coords["ra"].set_ticklabel_position("b")
            ax.coords["ra"].set_axislabel("RA [deg]", fontsize=9)
        else:
            ax.coords["ra"].set_ticks(spacing=reg["ra_spacing"] * 15 * u.deg)
            ax.coords["ra"].set_ticks_position("b")
            ax.coords["ra"].set_ticklabel_position("b")
            ax.coords["ra"].set_axislabel("RA", fontsize=9)

        ax.coords["dec"].set_ticks(spacing=reg["dec_spacing"] * u.deg)
        ax.coords["dec"].set_ticks_position("l")
        ax.coords["dec"].set_ticklabel_position("l")
        ax.coords["dec"].set_axislabel("Dec", fontsize=9)
        ax.set_title(reg["title"], fontsize=8.5, fontweight="normal", pad=4)

    legend_elements = [
        mpl.patches.Patch(
            facecolor="#2ca02c",
            edgecolor="none",
            label="S16A Baseline (Random Catalog: s16a_weak_lensing_medium_random.fits)",
        ),
        mpl.patches.Patch(
            facecolor="#1f78b4",
            edgecolor="none",
            label="S23B Cut Photometry (Cluster Finder: s23b_{field}_y3_cut_scalar.parquet)",
        ),
        mpl.patches.Patch(
            facecolor="#9ecae1",
            edgecolor="none",
            label="S19A Shape Catalog Excluded Area (hsc_y3_mask_nside8192.hs: star masks/tracts)",
        ),
        mpl.patches.Patch(
            facecolor="#d62728",
            edgecolor="none",
            label="HECTOMAP RA > 250 Sliver (S19A only, RA > 250)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig


def plot_version_comparison_fields(
    comparison_data: list[dict],
    output_path: Path,
) -> plt.Figure:
    """Plot a 3x3 side-by-side comparison matrix: 3 survey fields x 3 release versions (S16A, S19A, S23B)."""
    fig = plt.figure(figsize=(22, 11))
    height_ratios = [reg["shape"][0] for reg in comparison_data]

    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=height_ratios,
        hspace=0.55,
        wspace=0.10,
        top=0.88,
        bottom=0.08,
        left=0.05,
        right=0.98,
    )

    cmaps = [
        mpl.colors.ListedColormap(["#f2f4f7", "#2ca02c"]),  # S16A: Green
        mpl.colors.ListedColormap(["#f2f4f7", "#1f77b4"]),  # S19A: Blue
        mpl.colors.ListedColormap(["#f2f4f7", "#e6550d"]),  # S23B: Orange
    ]
    for c in cmaps:
        c.set_bad("white")

    col_headers = [
        "Version 1: S16A Baseline (Random Catalog)",
        "Version 2: S19A (HSC Y3 Cosmic Shear Shape Catalog)",
        "Version 3: S23B (Cut Photometry for Cluster Finder)",
    ]
    col_x_centers = [0.205, 0.515, 0.825]
    for col_idx, (x_pos, header_text) in enumerate(
        zip(col_x_centers, col_headers, strict=False)
    ):
        fig.text(
            x_pos,
            0.925,
            header_text,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="normal",
        )

    for row_idx, reg in enumerate(comparison_data):
        images = [reg["img_s16a"], reg["img_s19a"], reg["img_s23b"]]
        titles = [
            (
                f"{reg['name']} — S16A: {reg['area_s16a']:.2f} deg² ({reg['pct_s16a']:.1f}%)\n"
                f"Source: s16a_weak_lensing_medium_random.fits (NSIDE=1024)"
            ),
            (
                f"{reg['name']} — S19A: {reg['area_s19a']:.2f} deg² (100.0%)\n"
                f"Source: hsc_y3_mask_nside8192.hs (NSIDE=8192)"
            ),
            (
                f"{reg['name']} — S23B: {reg['area_s23b']:.2f} deg² ({reg['pct_s23b']:.1f}%)\n"
                f"Source: s23b_{reg['file_field']}_y3_cut_scalar.parquet (NSIDE=4096)"
            ),
        ]

        is_bottom_row = row_idx == len(comparison_data) - 1

        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection=reg["wcs"])
            ax.imshow(
                images[col_idx],
                origin="lower",
                cmap=cmaps[col_idx],
                vmin=0.0,
                vmax=1.0,
            )
            ax.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)

            if reg["ra_unit"] == "deg":
                ax.coords["ra"].set_major_formatter("d")
                ax.coords["ra"].set_ticks(spacing=reg["ra_spacing"] * u.deg)
            else:
                ax.coords["ra"].set_ticks(spacing=reg["ra_spacing"] * 15 * u.deg)

            ax.coords["ra"].set_ticks_position("b")
            ax.coords["ra"].set_ticklabel_position("b")

            if is_bottom_row:
                ra_label = "RA [deg]" if reg["ra_unit"] == "deg" else "RA"
                ax.coords["ra"].set_axislabel(ra_label, fontsize=8.5)
                ax.coords["ra"].set_ticklabel_visible(True)
            else:
                ax.coords["ra"].set_axislabel("")
                ax.coords["ra"].set_ticklabel_visible(False)

            ax.coords["dec"].set_ticks(spacing=reg["dec_spacing"] * u.deg)
            ax.coords["dec"].set_ticks_position("l")
            ax.coords["dec"].set_ticklabel_position("l")

            if col_idx == 0:
                ax.coords["dec"].set_axislabel("Dec", fontsize=8.5)
                ax.coords["dec"].set_ticklabel_visible(True)
            else:
                ax.coords["dec"].set_axislabel("")
                ax.coords["dec"].set_ticklabel_visible(False)

            ax.set_title(titles[col_idx], fontsize=8.0, fontweight="normal", pad=4)

    fig.suptitle(
        "HSC Survey Footprint Comparison: S16A (Y1 Random Baseline) vs S19A (Y3 Shape Catalog) vs S23B (PDR3 Cut Photometry)",
        fontsize=12,
        fontweight="normal",
        y=0.965,
    )

    fig.text(
        0.5,
        0.02,
        "Data Sources & Sample Definitions: "
        "• S16A: HSC Y1 medium random catalog tracing the baseline weak lensing footprint (~137.83 deg²). "
        "• S19A: Master HSC Y3 cosmic shear shape catalog HealSparse mask (Li et al. 2022, ~439.49 deg²). "
        "• S23B: HSC S23B Wide initial-galaxy sample cut by Y3 footprint mask, bright star masks, and i_cmodel ≤ m*(0.6)+2 (~453.90 deg²).",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#333333",
        fontweight="normal",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig


# %%
# Global Configuration

OUTPUT_DIR = project_root / "output" / "plots_for_agents"
SHAPE_FULL = (360, 720)
COLOR_PALETTE = ["#1f77b4", "#2b8cbe", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

# 1. Full-sky HPX WCS (unrotated native celestial orientation)
WCS_FULL = make_hpx_wcs(
    nx=SHAPE_FULL[1],
    ny=SHAPE_FULL[0],
    center_ra=180.0,
    center_dec=0.0,
)

# 2. Regional HPX WCS projections with isotropic pixel scale (|cdelt_ra| = cdelt_dec)
NX_REG = 1000
SHAPE_HECTO = (95, NX_REG)
SHAPE_SPRING = (130, NX_REG)
SHAPE_FALL = (220, NX_REG)

WCS_HECTO = make_shifted_hpx_wcs(
    nx=SHAPE_HECTO[1],
    ny=SHAPE_HECTO[0],
    center_ra=231.4,
    center_dec=43.3,
    cdelt_ra=-0.042,
    cdelt_dec=0.042,
    ref_ra=180.0,
    ref_dec=0.0,
)

WCS_SPRING = make_shifted_hpx_wcs(
    nx=SHAPE_SPRING[1],
    ny=SHAPE_SPRING[0],
    center_ra=177.0,
    center_dec=1.4,
    cdelt_ra=-0.104,
    cdelt_dec=0.104,
    ref_ra=180.0,
    ref_dec=0.0,
)

WCS_FALL = make_shifted_hpx_wcs(
    nx=SHAPE_FALL[1],
    ny=SHAPE_FALL[0],
    center_ra=5.0,
    center_dec=-0.25,
    cdelt_ra=-0.082,
    cdelt_dec=0.082,
    ref_ra=0.0,
    ref_dec=0.0,
)


# %% [Stage 1: Extract and Summarize Distinct Masks]

mask_groups = group_configs_by_mask(root=project_root)

logging.info("=" * 80)
logging.info("Extracted %d distinct HealSparse mask configurations:", len(mask_groups))
for idx, group in enumerate(mask_groups, start=1):
    logging.info(
        "Mask %d: [%s] Area=%.2f deg2, Factor=%.4f, Configs=%d (e.g. %s)",
        idx,
        group["label"],
        group["area_deg2"],
        group["volume_factor"],
        len(group["all_config_names"]),
        group["sample_config_name"],
    )
logging.info("=" * 80)


# %% [Stage 2: Build HealSparse Masks for Each Group, Load S23B and Precompute Regional Layers]

masks = {}
for group in mask_groups:
    label = group["label"]
    logging.info("Building HealSparse mask for [%s]...", label)
    masks[label] = build_config_mask(group["lens_config"], root=project_root)

# Load master S19A (HSC Y3) shape catalog mask and S16A random catalog baseline
mask_y3 = load_y3_mask(project_root)
s16a_pix = load_s16a_pixset_1024(project_root)

# Load S23B cut photometric catalog mask and field areas
mask_s23b = load_s23b_cut_mask(root=project_root, nside_sparse=4096)
s23b_areas = s23b_cut_field_areas_deg2(root=project_root, nside_sparse=4096)
s23b_factor = volume_factor(s23b_areas["Total"], REFERENCE_Z_RANGE, root=project_root)

# Append S23B Full Mask to overview mask groups display
mask_groups_display = list(mask_groups)
s23b_group_entry = {
    "label": "s23b_cut_full",
    "title": "Full HSC S23B Cut Footprint (Cluster Finder)",
    "description": "S23B Wide initial-galaxy sample cut by Y3 mask, bright star masks, and i_cmodel <= m*(0.6)+2.",
    "sample_config_name": "s23b_photometry_y3_cut",
    "all_config_names": ("s23b_photometry_y3_cut",),
    "area_deg2": s23b_areas["Total"],
    "volume_factor": s23b_factor,
}
mask_groups_display.insert(1, s23b_group_entry)
masks["s23b_cut_full"] = mask_s23b

# Precompute per-field statistics for S16A, S19A, and S23B
valid_pix_y3 = mask_y3.valid_pixels
ra_pix_y3, dec_pix_y3 = hp.pix2ang(
    mask_y3.nside_sparse, valid_pix_y3, nest=True, lonlat=True
)
pix_area_y3 = hp.nside2pixarea(mask_y3.nside_sparse, degrees=True)
parent_1024 = valid_pix_y3 // 64
in_s16a = np.isin(parent_1024, list(s16a_pix))

field_names = ("HECTOMAP", "SPRING", "FALL")
field_stats: dict[str, dict[str, float]] = {}

logging.info("=" * 80)
logging.info("Three-Version Survey Coverage Comparison (S16A vs S19A vs S23B):")
logging.info(
    "%-10s | %-16s | %-16s | %-16s",
    "Field",
    "S16A (Random)",
    "S19A (Y3 Shape)",
    "S23B (Cut Photo)",
)
logging.info("-" * 80)

total_area_s16a = 0.0
total_area_y3 = 0.0
total_area_s23b = s23b_areas["Total"]

for f in field_names:
    f_mask = in_field(ra_pix_y3, dec_pix_y3, f)
    area_y3_f = float(np.sum(f_mask) * pix_area_y3)
    area_s16a_f = float(np.sum(f_mask & in_s16a) * pix_area_y3)
    area_s23b_f = s23b_areas[f]
    pct_s16a_f = area_s16a_f / area_y3_f * 100.0 if area_y3_f > 0 else 0.0
    pct_s23b_f = area_s23b_f / area_y3_f * 100.0 if area_y3_f > 0 else 0.0

    field_stats[f] = {
        "area_s16a": area_s16a_f,
        "area_y3": area_y3_f,
        "area_s23b": area_s23b_f,
        "pct_s16a": pct_s16a_f,
        "pct_s23b": pct_s23b_f,
    }

    total_area_s16a += area_s16a_f
    total_area_y3 += area_y3_f

    logging.info(
        "%-10s | %6.2f deg2 (%4.1f%%) | %6.2f deg2 (100.0%%) | %6.2f deg2 (%5.1f%%)",
        f,
        area_s16a_f,
        pct_s16a_f,
        area_y3_f,
        area_s23b_f,
        pct_s23b_f,
    )

logging.info("-" * 80)
logging.info(
    "%-10s | %6.2f deg2 (%4.1f%%) | %6.2f deg2 (100.0%%) | %6.2f deg2 (%5.1f%%)",
    "Total",
    total_area_s16a,
    total_area_s16a / total_area_y3 * 100.0,
    total_area_y3,
    total_area_s23b,
    total_area_s23b / total_area_y3 * 100.0,
)
logging.info("=" * 80)

# Precompute full-sky 2D images
images_full = build_fullsky_mask_images(
    mask_groups=mask_groups_display,
    masks=masks,
    w_full=WCS_FULL,
    shape_full=SHAPE_FULL,
)

# Precompute regional images with 3-version composite layers
regional_data = [
    {
        "name": "HECTOMAP",
        "file_field": "hectomap",
        "shape": SHAPE_HECTO,
        "title": (
            "HECTOMAP Field (North, HPX Projection) — Green: S16A | Dark Blue: S23B | Light Blue: S19A | Red: Sliver\n"
            f"S16A: {field_stats['HECTOMAP']['area_s16a']:.2f} deg² ({field_stats['HECTOMAP']['pct_s16a']:.1f}%) | "
            f"S19A: {field_stats['HECTOMAP']['area_y3']:.2f} deg² | "
            f"S23B: {field_stats['HECTOMAP']['area_s23b']:.2f} deg² ({field_stats['HECTOMAP']['pct_s23b']:.1f}%)"
        ),
        "wcs": WCS_HECTO,
        "img": build_regional_layer_image(
            mask_y3, mask_s23b, s16a_pix, WCS_HECTO, SHAPE_HECTO, include_sliver=True
        ),
        "ra_unit": "deg",
        "ra_spacing": 5,
        "dec_spacing": 1,
        "has_sliver": True,
        "area_s16a": field_stats["HECTOMAP"]["area_s16a"],
        "area_s19a": field_stats["HECTOMAP"]["area_y3"],
        "area_s23b": field_stats["HECTOMAP"]["area_s23b"],
        "pct_s16a": field_stats["HECTOMAP"]["pct_s16a"],
        "pct_s23b": field_stats["HECTOMAP"]["pct_s23b"],
    },
    {
        "name": "SPRING",
        "file_field": "spring",
        "shape": SHAPE_SPRING,
        "title": (
            "SPRING Field (GAMA09H + WIDE12H + GAMA15H, HPX Projection) — Green: S16A | Dark Blue: S23B | Light Blue: S19A\n"
            f"S16A: {field_stats['SPRING']['area_s16a']:.2f} deg² ({field_stats['SPRING']['pct_s16a']:.1f}%) | "
            f"S19A: {field_stats['SPRING']['area_y3']:.2f} deg² | "
            f"S23B: {field_stats['SPRING']['area_s23b']:.2f} deg² ({field_stats['SPRING']['pct_s23b']:.1f}%)"
        ),
        "wcs": WCS_SPRING,
        "img": build_regional_layer_image(
            mask_y3, mask_s23b, s16a_pix, WCS_SPRING, SHAPE_SPRING, include_sliver=False
        ),
        "ra_unit": "hour",
        "ra_spacing": 2,
        "dec_spacing": 2,
        "has_sliver": False,
        "area_s16a": field_stats["SPRING"]["area_s16a"],
        "area_s19a": field_stats["SPRING"]["area_y3"],
        "area_s23b": field_stats["SPRING"]["area_s23b"],
        "pct_s16a": field_stats["SPRING"]["pct_s16a"],
        "pct_s23b": field_stats["SPRING"]["pct_s23b"],
    },
    {
        "name": "FALL",
        "file_field": "fall",
        "shape": SHAPE_FALL,
        "title": (
            "FALL Field (VVDS + XMM, HPX Projection) — Green: S16A | Dark Blue: S23B | Light Blue: S19A\n"
            f"S16A: {field_stats['FALL']['area_s16a']:.2f} deg² ({field_stats['FALL']['pct_s16a']:.1f}%) | "
            f"S19A: {field_stats['FALL']['area_y3']:.2f} deg² | "
            f"S23B: {field_stats['FALL']['area_s23b']:.2f} deg² ({field_stats['FALL']['pct_s23b']:.1f}%)"
        ),
        "wcs": WCS_FALL,
        "img": build_regional_layer_image(
            mask_y3, mask_s23b, s16a_pix, WCS_FALL, SHAPE_FALL, include_sliver=False
        ),
        "ra_unit": "hour",
        "ra_spacing": 2,
        "dec_spacing": 4,
        "has_sliver": False,
        "area_s16a": field_stats["FALL"]["area_s16a"],
        "area_s19a": field_stats["FALL"]["area_y3"],
        "area_s23b": field_stats["FALL"]["area_s23b"],
        "pct_s16a": field_stats["FALL"]["pct_s16a"],
        "pct_s23b": field_stats["FALL"]["pct_s23b"],
    },
]

# Precompute single version images for side-by-side comparative matrix
comparison_data = []
for reg in regional_data:
    img_16, img_19, img_23 = build_version_single_images(
        mask_y3=mask_y3,
        mask_s23b=mask_s23b,
        s16a_pix=s16a_pix,
        wcs=reg["wcs"],
        shape=reg["shape"],
    )
    reg_comp = dict(reg)
    reg_comp["img_s16a"] = img_16
    reg_comp["img_s19a"] = img_19
    reg_comp["img_s23b"] = img_23
    comparison_data.append(reg_comp)


# %% [Stage 3: Render and Save Combined Overview (Left: 6 Full-Sky Masks, Right: 3 Regions)]

overview_path = OUTPUT_DIR / "masks_overview_hpx.png"
logging.info("Generating combined overview figure -> %s", overview_path)
plot_combined_overview(
    mask_groups=mask_groups_display,
    images_full=images_full,
    regional_data=regional_data,
    w_full=WCS_FULL,
    output_path=overview_path,
    palette=COLOR_PALETTE,
)


# %% [Stage 4: Render and Save Standalone Regional Comparison (HECTOMAP + SPRING + FALL)]

regional_path = OUTPUT_DIR / "masks_regional_hpx.png"
logging.info("Generating standalone regional fields figure -> %s", regional_path)
plot_standalone_regional_fields(
    regional_data=regional_data,
    output_path=regional_path,
)


# %% [Stage 5: Render and Save 3-Version Side-by-Side Comparison Matrix (S16A vs S19A vs S23B)]

version_comparison_path = OUTPUT_DIR / "masks_version_comparison_hpx.png"
logging.info(
    "Generating 3-version comparison matrix figure -> %s",
    version_comparison_path,
)
plot_version_comparison_fields(
    comparison_data=comparison_data,
    output_path=version_comparison_path,
)
