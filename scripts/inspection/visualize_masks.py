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
    build_config_mask,
    group_configs_by_mask,
    load_s16a_pixset_1024,
    load_y3_mask,
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
    s16a_pix: set[int],
    wcs: WCS,
    shape: tuple[int, int],
    include_sliver: bool = False,
) -> np.ndarray:
    """Build a 2D composite image for a region distinguishing Y3, S16A, and optional boundary sliver."""
    ny, nx = shape
    yy, xx = np.indices((ny, nx), dtype=float)
    ra, dec = wcs.all_pix2world(xx, yy, 0)
    finite = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90.0) & (dec <= 90.0)

    v_y3 = mask_y3.get_values_pos(ra[finite], dec[finite], lonlat=True)
    pix_1024 = hp.ang2pix(1024, ra[finite], dec[finite], nest=True, lonlat=True)
    v_s16a = np.isin(pix_1024, list(s16a_pix)) & v_y3

    img = np.full((ny, nx), np.nan, dtype=float)
    vals = np.zeros(np.sum(finite), dtype=float)
    vals[v_y3] = 1.0
    vals[v_s16a] = 2.0
    if include_sliver:
        is_sliver = v_y3 & (ra[finite] > 250.0)
        vals[is_sliver] = 3.0

    img[finite] = vals
    return img


def plot_combined_overview(
    mask_groups: list[dict],
    images_full: dict[str, np.ndarray],
    regional_data: list[dict],
    w_full: WCS,
    output_path: Path,
    palette: list[str],
) -> plt.Figure:
    """Plot the integrated overview: 5 full-sky masks on the left and 3 regional fields on the right."""
    fig = plt.figure(figsize=(20, 11))
    subfigs = fig.subfigures(1, 2, width_ratios=[1.0, 1.45], wspace=0.06)

    # ------------------------------------------------------------------
    # Left Subfigure: 5 Full-Sky HPX Panels
    # ------------------------------------------------------------------
    subfigs[0].subplots_adjust(top=0.97, bottom=0.03, hspace=0.35)
    for i, group in enumerate(mask_groups):
        label = group["label"]
        color = palette[i % len(palette)]
        cmap = mpl.colors.ListedColormap(["#f2f4f7", color])
        cmap.set_bad("white")

        ax = subfigs[0].add_subplot(5, 1, i + 1, projection=w_full)
        ax.imshow(images_full[label], origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)
        ax.coords["ra"].set_axislabel("RA", fontsize=8)
        ax.coords["dec"].set_axislabel("Dec", fontsize=8)
        ax.set_title(
            f"{group['title']} (Full Sky HPX)\n"
            f"Area: {group['area_deg2']:.2f} deg², Factor: {group['volume_factor']:.4f} "
            f"({len(group['all_config_names'])} configs)",
            fontsize=8.5,
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

    cmap_reg = mpl.colors.ListedColormap(["#f2f4f7", "#9ecae1", "#2ca02c"])
    cmap_reg.set_bad("white")

    cmap_hecto = mpl.colors.ListedColormap(["#f2f4f7", "#9ecae1", "#2ca02c", "#d62728"])
    cmap_hecto.set_bad("white")

    for j, reg in enumerate(regional_data):
        ax = subfigs[1].add_subplot(gs_right[j], projection=reg["wcs"])
        cmap_use = cmap_hecto if reg.get("has_sliver", False) else cmap_reg
        vmax_use = 3.0 if reg.get("has_sliver", False) else 2.0

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
        ax.set_title(reg["title"], fontsize=8.5, fontweight="normal", pad=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig


def plot_standalone_regional_fields(
    regional_data: list[dict],
    output_path: Path,
) -> plt.Figure:
    """Plot a dedicated, high-resolution standalone figure of the 3 survey regions."""
    fig = plt.figure(figsize=(14, 8))
    height_ratios = [reg["shape"][0] for reg in regional_data]
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=height_ratios,
        hspace=0.38,
        top=0.94,
        bottom=0.07,
        left=0.07,
        right=0.97,
    )

    cmap_reg = mpl.colors.ListedColormap(["#f2f4f7", "#9ecae1", "#2ca02c"])
    cmap_reg.set_bad("white")

    cmap_hecto = mpl.colors.ListedColormap(["#f2f4f7", "#9ecae1", "#2ca02c", "#d62728"])
    cmap_hecto.set_bad("white")

    for j, reg in enumerate(regional_data):
        ax = fig.add_subplot(gs[j], projection=reg["wcs"])
        cmap_use = cmap_hecto if reg.get("has_sliver", False) else cmap_reg
        vmax_use = 3.0 if reg.get("has_sliver", False) else 2.0

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
        ax.set_title(reg["title"], fontsize=9.0, fontweight="normal", pad=4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig


# %%
# Global Configuration

OUTPUT_DIR = project_root / "output" / "plots_for_agents"
SHAPE_FULL = (360, 720)
COLOR_PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

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


# %% [Stage 2: Build HealSparse Masks for Each Group and Precompute Regional Layers]

masks = {}
for group in mask_groups:
    label = group["label"]
    logging.info("Building HealSparse mask for [%s]...", label)
    masks[label] = build_config_mask(group["lens_config"], root=project_root)

mask_y3 = load_y3_mask(project_root)
s16a_pix = load_s16a_pixset_1024(project_root)

# Precompute full-sky 2D images
images_full = build_fullsky_mask_images(
    mask_groups=mask_groups,
    masks=masks,
    w_full=WCS_FULL,
    shape_full=SHAPE_FULL,
)

# Precompute regional images with Y3 (S19A) vs S16A breakdown
regional_data = [
    {
        "name": "HECTOMAP",
        "shape": SHAPE_HECTO,
        "title": (
            "HECTOMAP Field (North, HPX Projection) — Light Blue: HSC Y3 (S19A) | Green: S16A Baseline | Red: RA > 250 Sliver\n"
            "Y3 Full: 43.44 deg² | Box (RA≤250): 43.37 deg² | S16A: 12.23 deg² (28.2%)"
        ),
        "wcs": WCS_HECTO,
        "img": build_regional_layer_image(
            mask_y3, s16a_pix, WCS_HECTO, SHAPE_HECTO, include_sliver=True
        ),
        "ra_unit": "deg",
        "ra_spacing": 5,
        "dec_spacing": 1,
        "has_sliver": True,
    },
    {
        "name": "SPRING",
        "shape": SHAPE_SPRING,
        "title": (
            "SPRING Field (GAMA09H + WIDE12H + GAMA15H, HPX Projection) — Light Blue: HSC Y3 (S19A) | Green: S16A Baseline\n"
            "Y3 (S19A): 265.33 deg² | S16A: 76.78 deg² (28.9%)"
        ),
        "wcs": WCS_SPRING,
        "img": build_regional_layer_image(
            mask_y3, s16a_pix, WCS_SPRING, SHAPE_SPRING, include_sliver=False
        ),
        "ra_unit": "hour",
        "ra_spacing": 2,
        "dec_spacing": 2,
        "has_sliver": False,
    },
    {
        "name": "FALL",
        "shape": SHAPE_FALL,
        "title": (
            "FALL Field (VVDS + XMM, HPX Projection) — Light Blue: HSC Y3 (S19A) | Green: S16A Baseline\n"
            "Y3 (S19A): 130.72 deg² | S16A: 48.82 deg² (37.3%)"
        ),
        "wcs": WCS_FALL,
        "img": build_regional_layer_image(
            mask_y3, s16a_pix, WCS_FALL, SHAPE_FALL, include_sliver=False
        ),
        "ra_unit": "hour",
        "ra_spacing": 2,
        "dec_spacing": 4,
        "has_sliver": False,
    },
]


# %% [Stage 3: Render and Save Combined Overview (Left: 5 Masks, Right: 3 Regions)]

overview_path = OUTPUT_DIR / "masks_overview_hpx.png"
logging.info("Generating combined overview figure -> %s", overview_path)
plot_combined_overview(
    mask_groups=mask_groups,
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
