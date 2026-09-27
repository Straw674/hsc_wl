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

from hsc_wl.coverage import build_config_mask, group_configs_by_mask
from initial import *

# %%
# Local Functions


def build_mask_images(
    mask_groups: list[dict],
    masks: dict[str, hsp.HealSparseMap],
    w_full: WCS,
    shape_full: tuple[int, int],
    w_zoom: WCS,
    shape_zoom: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Precompute 2D WCS HPX-projected image arrays for all mask groups."""
    images_full = {}
    images_zoom = {}
    for group in mask_groups:
        label = group["label"]
        mask = masks[label]
        images_full[label] = project_healsparse_to_wcs(mask, w_full, shape_full)
        images_zoom[label] = project_healsparse_to_wcs(mask, w_zoom, shape_zoom)
    return images_full, images_zoom


def plot_all_masks_overview(
    mask_groups: list[dict],
    images_full: dict[str, np.ndarray],
    images_zoom: dict[str, np.ndarray],
    w_full: WCS,
    w_zoom: WCS,
    output_path: Path,
    palette: list[str],
) -> plt.Figure:
    """Plot a comprehensive multi-panel figure displaying each mask in HPX projection."""
    n_masks = len(mask_groups)
    fig = plt.figure(figsize=(15, 3.6 * n_masks))

    for i, group in enumerate(mask_groups):
        label = group["label"]
        color = palette[i % len(palette)]
        cmap = mpl.colors.ListedColormap(["#f2f4f7", color])
        cmap.set_bad("white")

        # Column 1: Full-Sky HPX
        ax_full = fig.add_subplot(n_masks, 2, 2 * i + 1, projection=w_full)
        ax_full.imshow(
            images_full[label], origin="lower", cmap=cmap, vmin=0.0, vmax=1.0
        )
        ax_full.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)
        ax_full.coords["ra"].set_axislabel("RA", fontsize=8)
        ax_full.coords["dec"].set_axislabel("Dec", fontsize=8)
        ax_full.set_title(
            f"{group['title']} (Full Sky HPX)\n"
            f"Area: {group['area_deg2']:.2f} deg², Factor: {group['volume_factor']:.4f} "
            f"({len(group['all_config_names'])} configs)",
            fontsize=9,
            fontweight="normal",
        )

        # Column 2: HECTOMAP Regional Zoom HPX
        ax_zoom = fig.add_subplot(n_masks, 2, 2 * i + 2, projection=w_zoom)
        ax_zoom.imshow(
            images_zoom[label], origin="lower", cmap=cmap, vmin=0.0, vmax=1.0
        )
        ax_zoom.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)
        ax_zoom.coords["ra"].set_major_formatter("d")
        ax_zoom.coords["ra"].set_ticks(spacing=5 * u.deg)
        ax_zoom.coords["dec"].set_ticks(spacing=1 * u.deg)
        ax_zoom.coords["ra"].set_axislabel("RA [deg]", fontsize=8)
        ax_zoom.coords["dec"].set_axislabel("Dec [deg]", fontsize=8)
        ax_zoom.set_title(
            f"{group['title']} (HECTOMAP Zoom HPX)",
            fontsize=9,
            fontweight="normal",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig


def plot_masks_composite_overlay(
    masks: dict[str, hsp.HealSparseMap],
    w_full: WCS,
    shape_full: tuple[int, int],
    w_zoom: WCS,
    shape_zoom: tuple[int, int],
    output_path: Path,
) -> plt.Figure:
    """Plot comparative overlay views highlighting nested footprint relationships."""
    ny_f, nx_f = shape_full
    yy_f, xx_f = np.indices((ny_f, nx_f), dtype=float)
    ra_f, dec_f = w_full.all_pix2world(xx_f, yy_f, 0)
    fin_f = np.isfinite(ra_f) & np.isfinite(dec_f) & (dec_f >= -90.0) & (dec_f <= 90.0)

    # Full-sky composite:
    # 0 = unobserved background
    # 1 = Full Y3 wide (non-S16A)
    # 2 = S16A & Y3 overlap
    v_y3_f = masks["full_y3"].get_values_pos(ra_f[fin_f], dec_f[fin_f], lonlat=True)
    v_s16a_f = masks["s16a_full"].get_values_pos(ra_f[fin_f], dec_f[fin_f], lonlat=True)

    comp_f = np.full((ny_f, nx_f), np.nan, dtype=float)
    comp_vals_f = np.zeros(np.sum(fin_f), dtype=float)
    comp_vals_f[v_y3_f] = 1.0
    comp_vals_f[v_s16a_f] = 2.0
    comp_f[fin_f] = comp_vals_f

    # HECTOMAP zoom composite:
    # 0 = unobserved background
    # 1 = RA > 250 boundary sliver
    # 2 = HECTOMAP Boxed non-S16A
    # 3 = S16A overlap footprint
    ny_z, nx_z = shape_zoom
    yy_z, xx_z = np.indices((ny_z, nx_z), dtype=float)
    ra_z, dec_z = w_zoom.all_pix2world(xx_z, yy_z, 0)
    fin_z = np.isfinite(ra_z) & np.isfinite(dec_z) & (dec_z >= -90.0) & (dec_z <= 90.0)

    v_hf_z = masks["hectomap_full"].get_values_pos(
        ra_z[fin_z], dec_z[fin_z], lonlat=True
    )
    v_hb_z = masks["hectomap_box"].get_values_pos(
        ra_z[fin_z], dec_z[fin_z], lonlat=True
    )
    v_hs_z = masks["hectomap_box_s16a"].get_values_pos(
        ra_z[fin_z], dec_z[fin_z], lonlat=True
    )

    comp_z = np.full((ny_z, nx_z), np.nan, dtype=float)
    comp_vals_z = np.zeros(np.sum(fin_z), dtype=float)
    comp_vals_z[v_hf_z] = 1.0
    comp_vals_z[v_hb_z] = 2.0
    comp_vals_z[v_hs_z] = 3.0
    comp_z[fin_z] = comp_vals_z

    fig = plt.figure(figsize=(16, 6))

    # Left: Full-sky comparison
    cmap_f = mpl.colors.ListedColormap(["#f5f5f5", "#9ecae1", "#2ca02c"])
    cmap_f.set_bad("white")

    ax_left = fig.add_subplot(1, 2, 1, projection=w_full)
    ax_left.imshow(comp_f, origin="lower", cmap=cmap_f, vmin=0.0, vmax=2.0)
    ax_left.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)
    ax_left.coords["ra"].set_axislabel("RA")
    ax_left.coords["dec"].set_axislabel("Dec")
    ax_left.set_title(
        "Global Survey Footprints (HPX Projection)\n"
        "Light Blue: HSC Y3 Full (439.49 deg²) | Green: S16A Baseline (137.83 deg²)",
        fontsize=10,
        fontweight="normal",
    )

    # Right: HECTOMAP zoom comparison
    cmap_z = mpl.colors.ListedColormap(["#f5f5f5", "#fc9272", "#a1d99b", "#2171b5"])
    cmap_z.set_bad("white")

    ax_right = fig.add_subplot(1, 2, 2, projection=w_zoom)
    ax_right.imshow(comp_z, origin="lower", cmap=cmap_z, vmin=0.0, vmax=3.0)
    ax_right.coords.grid(color="#777777", ls=":", lw=0.5, alpha=0.6)
    ax_right.coords["ra"].set_major_formatter("d")
    ax_right.coords["ra"].set_ticks(spacing=5 * u.deg)
    ax_right.coords["dec"].set_ticks(spacing=1 * u.deg)
    ax_right.coords["ra"].set_axislabel("RA [deg]")
    ax_right.coords["dec"].set_axislabel("Dec [deg]")
    ax_right.set_title(
        "HECTOMAP Nested Footprints (HPX Projection)\n"
        "Red: RA>250 Sliver (0.07 deg²) | Green: Boxed non-S16A (31.14 deg²) | Blue: S16A Overlap (12.23 deg²)",
        fontsize=10,
        fontweight="normal",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig


# %%
# Global Configuration

OUTPUT_DIR = project_root / "output" / "plots_for_agents"
SHAPE_FULL = (500, 1000)
SHAPE_ZOOM = (300, 700)
COLOR_PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

WCS_FULL = make_hpx_wcs(
    nx=SHAPE_FULL[1],
    ny=SHAPE_FULL[0],
    center_ra=180.0,
    center_dec=0.0,
)

WCS_ZOOM = make_hpx_wcs(
    nx=SHAPE_ZOOM[1],
    ny=SHAPE_ZOOM[0],
    center_ra=231.0,
    center_dec=43.0,
    cdelt_ra=-0.075,
    cdelt_dec=0.045,
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


# %% [Stage 2: Build HealSparse Masks for Each Group]

masks = {}
for group in mask_groups:
    label = group["label"]
    logging.info("Building HealSparse mask for [%s]...", label)
    masks[label] = build_config_mask(group["lens_config"], root=project_root)


# %% [Stage 3: Render and Display All Masks Overview (HPX Projection)]

images_full, images_zoom = build_mask_images(
    mask_groups=mask_groups,
    masks=masks,
    w_full=WCS_FULL,
    shape_full=SHAPE_FULL,
    w_zoom=WCS_ZOOM,
    shape_zoom=SHAPE_ZOOM,
)

overview_path = OUTPUT_DIR / "masks_overview_hpx.png"
logging.info("Generating all masks overview figure -> %s", overview_path)
plot_all_masks_overview(
    mask_groups=mask_groups,
    images_full=images_full,
    images_zoom=images_zoom,
    w_full=WCS_FULL,
    w_zoom=WCS_ZOOM,
    output_path=overview_path,
    palette=COLOR_PALETTE,
)


# %% [Stage 4: Render and Display Comparative Composite (HPX Projection)]

composite_path = OUTPUT_DIR / "masks_composite_hpx.png"
logging.info("Generating comparative composite figure -> %s", composite_path)
plot_masks_composite_overlay(
    masks=masks,
    w_full=WCS_FULL,
    shape_full=SHAPE_FULL,
    w_zoom=WCS_ZOOM,
    shape_zoom=SHAPE_ZOOM,
    output_path=composite_path,
)
