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
    S23B_PHOTOMETRY_DIR,
    build_config_mask,
    group_configs_by_mask,
    load_s16a_pixset_1024,
    load_y3_mask,
)
from initial import *
from src.data import load_s23b_scalar_sample

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


def build_regional_data(mask_y3, s16a_pix, regions, samples):
    """Project masks and classify sampled positions without inferring coverage from gaps."""
    result = []
    for region in regions:
        wcs, shape = region["wcs"].deepcopy(), region["shape"]
        sample = samples[region["name"]]
        sample_x, sample_y = wcs.all_world2pix(sample["ra"], sample["dec"], 0)
        # Retain the original mask window and include the uncut catalog extent.
        finite_sample = np.isfinite(sample_x) & np.isfinite(sample_y)
        if finite_sample.any():
            lower = np.floor(
                np.minimum(
                    [0, 0],
                    [
                        sample_x[finite_sample].min() - 10,
                        sample_y[finite_sample].min() - 10,
                    ],
                )
            ).astype(int)
            upper = np.ceil(
                np.maximum(
                    [shape[1], shape[0]],
                    [
                        sample_x[finite_sample].max() + 10,
                        sample_y[finite_sample].max() + 10,
                    ],
                )
            ).astype(int)
            wcs.wcs.crpix -= lower
            shape = (int(upper[1] - lower[1]), int(upper[0] - lower[0]))
        yy, xx = np.indices(shape, dtype=float)
        ra, dec = wcs.all_pix2world(xx, yy, 0)
        finite = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90)
        y3 = mask_y3.get_values_pos(ra[finite], dec[finite], lonlat=True)
        s16a = np.isin(
            hp.ang2pix(1024, ra[finite], dec[finite], nest=True, lonlat=True),
            list(s16a_pix),
        )
        img = np.full(shape, np.nan)
        img[finite] = y3.astype(int) + 2 * s16a.astype(int)
        sample = samples[region["name"]]
        ra, dec = sample["ra"].to_numpy(), sample["dec"].to_numpy()
        y3 = mask_y3.get_values_pos(ra, dec, lonlat=True)
        s16a = np.isin(
            hp.ang2pix(1024, ra, dec, nest=True, lonlat=True), list(s16a_pix)
        )
        x, y = wcs.all_world2pix(ra, dec, 0)
        result.append(
            dict(
                region,
                wcs=wcs,
                shape=shape,
                img=img,
                x=x,
                y=y,
                membership=y3.astype(int) + 2 * s16a.astype(int),
            )
        )
    return result


def draw_regional_layer(ax, reg):
    """Overlay sampled S23B points on independently evaluated S16A and Y3 masks."""
    cmap = mpl.colors.ListedColormap(["#f2f4f7", "#bdd7e7", "#d9d9d9", "#9e9ac8"])
    cmap.set_bad("white")
    ax.imshow(reg["img"], origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5)
    for code, color in enumerate(("#e6550d", "#2171b5", "#636363", "#238b45")):
        selected = reg["membership"] == code
        ax.scatter(
            reg["x"][selected],
            reg["y"][selected],
            s=0.35,
            color=color,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlim(-0.5, reg["shape"][1] - 0.5)
    ax.set_ylim(-0.5, reg["shape"][0] - 0.5)
    # Isotropic HPX pixels provide sky projection geometry; negative CDELT1 reverses RA.
    ax.set_aspect("equal")


def add_coverage_legend(fig):
    """Distinguish mask membership from sampled S23B membership."""
    patches = [
        mpl.patches.Patch(facecolor=color, label=label)
        for color, label in (
            ("#bdd7e7", "Y3 only (mask)"),
            ("#d9d9d9", "S16A only (mask)"),
            ("#9e9ac8", "S16A + Y3 (masks)"),
        )
    ]
    points = [
        mpl.lines.Line2D(
            [], [], marker="o", linestyle="", markersize=4, color=color, label=label
        )
        for color, label in (
            ("#e6550d", "S23B only"),
            ("#2171b5", "S23B + Y3"),
            ("#636363", "S23B + S16A"),
            ("#238b45", "S23B + S16A + Y3"),
        )
    ]
    fig.legend(
        handles=patches + points,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        title="S23B: scalar sample; halo / ghost / blooming = False",
        title_fontsize=8,
    )


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
        bottom=0.12,
        left=0.05,
        right=0.98,
    )

    for j, reg in enumerate(regional_data):
        ax = subfigs[1].add_subplot(gs_right[j], projection=reg["wcs"])
        draw_regional_layer(ax, reg)
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

    add_coverage_legend(fig)

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

    for j, reg in enumerate(regional_data):
        ax = fig.add_subplot(gs[j], projection=reg["wcs"])
        draw_regional_layer(ax, reg)
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

    add_coverage_legend(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show()
    return fig


def prepare_masks(root, wcs, shape):
    """Load the configured analysis masks and render full-sky projections."""
    groups = group_configs_by_mask(root=root)
    logging.info("Building %d analysis masks", len(groups))
    masks = {
        group["label"]: build_config_mask(group["lens_config"], root=root)
        for group in groups
    }
    return groups, build_fullsky_mask_images(groups, masks, wcs, shape)


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


# %% [Stage 1: Project Configured Analysis Masks]
mask_groups, images_full = prepare_masks(project_root, WCS_FULL, SHAPE_FULL)

# %% [Stage 2: Sample Uncut S23B Photometry and Project Regional Masks]
S23B_SAMPLE_SIZE = 100_000
S23B_RANDOM_SEED = 42
S23B_BATCH_SIZE = 250_000
REGIONS = [
    dict(
        name="HECTOMAP",
        title="HECTOMAP",
        wcs=WCS_HECTO,
        shape=SHAPE_HECTO,
        ra_unit="deg",
        ra_spacing=5,
        dec_spacing=1,
    ),
    dict(
        name="SPRING",
        title="SPRING",
        wcs=WCS_SPRING,
        shape=SHAPE_SPRING,
        ra_unit="hour",
        ra_spacing=2,
        dec_spacing=2,
    ),
    dict(
        name="FALL",
        title="FALL",
        wcs=WCS_FALL,
        shape=SHAPE_FALL,
        ra_unit="hour",
        ra_spacing=2,
        dec_spacing=4,
    ),
]
s23b_samples = load_s23b_scalar_sample(
    S23B_PHOTOMETRY_DIR,
    tuple(reg["name"] for reg in REGIONS),
    S23B_SAMPLE_SIZE,
    S23B_RANDOM_SEED,
    S23B_BATCH_SIZE,
)
regional_data = build_regional_data(
    load_y3_mask(project_root),
    load_s16a_pixset_1024(project_root),
    REGIONS,
    s23b_samples,
)

# %% [Stage 3: Render Combined Overview]
plot_combined_overview(
    mask_groups,
    images_full,
    regional_data,
    WCS_FULL,
    OUTPUT_DIR / "masks_overview_hpx.png",
    COLOR_PALETTE,
)

# %% [Stage 4: Render Regional Overlays]
plot_standalone_regional_fields(
    regional_data,
    OUTPUT_DIR / "masks_regional_hpx.png",
)
