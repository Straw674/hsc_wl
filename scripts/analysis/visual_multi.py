# %% [Initialization]
import sys
from datetime import datetime
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

from initial import *  # noqa: F401,F403

# %% Local Functions


def _format_ls_time(path: Path) -> str:
    """Format a file timestamp in the same style as ``ls -lh``."""
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    now = datetime.now()
    if abs((now - mtime).days) >= 180:
        return mtime.strftime("%b %e  %Y")
    return mtime.strftime("%b %e %H:%M")


def _get_result_time_text(base_dir: Path) -> str:
    """Get a human-readable timestamp for the newest result file in ``base_dir``."""
    result_files = sorted(base_dir.glob("hsc_hsc_*.*"))
    if not result_files:
        return "unknown time"

    latest_file = max(result_files, key=lambda path: path.stat().st_mtime)
    return _format_ls_time(latest_file)


def load_comparison_data(configs_to_compare, root_path, display_names=None):
    """Load dsigma result tables for all (catalog_id, nbins, version) configurations.

    Returns
    -------
    present_labels : list[str]
        Display names (e.g. ``"catalog_id (nbins)"``) in load order.
    loaded_tables : list[list[Table]]
        Per-config list of bin tables.
    label_time_texts : list[str]
        Formatted file timestamps for each config (for informative prints).
    """
    display_names = display_names or {}
    present_labels = []
    loaded_tables = []
    label_time_texts = []

    for catalog_id, nbins, version_name in configs_to_compare:
        current_dir = root_path / f"output/{catalog_id}/{nbins}/{version_name}/dsigma"
        if not current_dir.exists():
            print(f"Warning: {current_dir} does not exist. Skipping.")
            continue
        time_text = _get_result_time_text(current_dir)
        key = f"{catalog_id}_{nbins}"
        display_name = display_names.get(
            key, display_names.get(catalog_id, f"{catalog_id} ({nbins})")
        )
        current_tables = load_result_tables(current_dir, only_complete=True)
        print(
            f"Loading data for {display_name} ({len(current_tables)} complete bins) | "
            f"file time: {time_text}"
        )
        present_labels.append(display_name)
        loaded_tables.append(current_tables)
        label_time_texts.append(time_text)

    if not loaded_tables:
        print("No data loaded. Exiting.")
        return present_labels, loaded_tables, label_time_texts

    min_bins = min(len(tables) for tables in loaded_tables)
    max_bins = max(len(tables) for tables in loaded_tables)
    if min_bins != max_bins:
        print(
            f"Warning: Configurations have differing complete bin counts ({min_bins} vs {max_bins}). "
            f"Comparing first {min_bins} common complete bins."
        )
        loaded_tables = [tables[:min_bins] for tables in loaded_tables]
    n_bins = min_bins

    return present_labels, loaded_tables, label_time_texts


def resolve_comparison_styles(
    configs_to_compare,
    present_labels,
    catalog_styles,
    default_palette,
    default_markers,
    display_names=None,
):
    """Resolve distinct colors and markers for each loaded configuration."""
    display_names = display_names or {}
    config_by_display = {}
    for catalog_id, nbins, _ in configs_to_compare:
        key = f"{catalog_id}_{nbins}"
        dname = display_names.get(
            key, display_names.get(catalog_id, f"{catalog_id} ({nbins})")
        )
        config_by_display[dname] = catalog_id

    resolved_palette = []
    resolved_markers = []
    for i, label in enumerate(present_labels):
        cat_id = config_by_display.get(label, label)
        base_cat = cat_id.removesuffix("_4bin").removesuffix("_1bin")
        matched = None
        for key in (label, cat_id, base_cat):
            if key in catalog_styles:
                matched = catalog_styles[key]
                break

        if matched:
            resolved_palette.append(matched["color"])
            resolved_markers.append(matched["marker"])
        else:
            resolved_palette.append(default_palette[i % len(default_palette)])
            resolved_markers.append(default_markers[i % len(default_markers)])

    return resolved_palette, resolved_markers


def plot_main_comparison(
    present_labels,
    loaded_tables,
    n_bins,
    multiply_by_radius,
    use_spline,
    use_log_y,
    reference_line_y,
    color_mode,
    palette=None,
    markers=None,
    marker_size=4.0,
    capsize=0.0,
    elinewidth=0.8,
):
    """Plot the main ΔΣ comparison profile across all labels."""
    palette = palette or LABEL_PALETTE
    markers = markers or MARKERS

    fig_height = max(4.0, 3.3 * n_bins)
    fig, axes = plt.subplots(
        n_bins, 1, figsize=(8.6, fig_height), sharex=True, sharey=False
    )
    axes = np.atleast_1d(axes)

    for i, (display_name, current_tables) in enumerate(
        zip(present_labels, loaded_tables)
    ):
        plot_radial_profile(
            current_tables,
            value_column="ds",
            title_label="Comparison",
            ax_list=axes,
            label_text=display_name,
            label_index=i,
            n_labels=len(present_labels),
            marker=markers[i % len(markers)],
            color_mode=color_mode,
            palette=palette,
            multiply_by_radius=multiply_by_radius,
            use_spline=use_spline,
            use_log_y=use_log_y,
            reference_line_y=reference_line_y,
            marker_size=marker_size,
            capsize=capsize,
            elinewidth=elinewidth,
        )

    fig.suptitle("Comparison of ΔΣ Profiles", y=0.996)
    fig.tight_layout()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            title="label",
            fontsize="small",
            frameon=False,
        )
    return fig, axes


def _build_ratio_table(current_table, reference_table):
    """Build a per-bin ratio table (current / reference) with propagated errors."""
    current_sorted = current_table.copy()
    reference_sorted = reference_table.copy()

    current_sorted.sort(keys="rp")
    reference_sorted.sort(keys="rp")

    current_rp = np.asarray(current_sorted["rp"], dtype=float)
    reference_rp = np.asarray(reference_sorted["rp"], dtype=float)
    if current_rp.shape != reference_rp.shape or not np.allclose(
        current_rp, reference_rp
    ):
        raise ValueError("Ratio tables must share the same rp bins.")

    current_ds = np.asarray(current_sorted["ds"], dtype=float)
    current_err = np.asarray(current_sorted["ds_err"], dtype=float)
    reference_ds = np.asarray(reference_sorted["ds"], dtype=float)
    reference_err = np.asarray(reference_sorted["ds_err"], dtype=float)

    ratio = np.divide(
        current_ds,
        reference_ds,
        out=np.full_like(current_ds, np.nan),
        where=reference_ds != 0,
    )
    ratio_err = np.sqrt(
        np.divide(
            current_err,
            reference_ds,
            out=np.full_like(current_err, np.nan),
            where=reference_ds != 0,
        )
        ** 2
        + np.divide(
            current_ds * reference_err,
            reference_ds**2,
            out=np.full_like(current_ds, np.nan),
            where=reference_ds != 0,
        )
        ** 2
    )

    ratio_table = current_sorted.copy()
    ratio_table["ds"] = ratio
    ratio_table["ds_err"] = ratio_err
    return ratio_table


def plot_ratio_comparison(
    present_labels,
    loaded_tables,
    n_bins,
    multiply_by_radius,
    use_spline,
    use_log_y,
    reference_line_y,
    color_mode,
    palette=None,
    markers=None,
    marker_size=4.0,
    capsize=0.0,
    elinewidth=0.8,
):
    """Plot ratio ΔΣ / ΔΣ_ref across all non-reference labels."""
    if not loaded_tables or len(loaded_tables) <= 1:
        return None

    palette = palette or LABEL_PALETTE
    markers = markers or MARKERS

    reference_tables = loaded_tables[0]
    fig_height = max(4.0, 3.3 * n_bins)
    ratio_fig, ratio_axes = plt.subplots(
        n_bins, 1, figsize=(8.6, fig_height), sharex=True, sharey=False
    )
    ratio_axes = np.atleast_1d(ratio_axes)

    for ratio_index, (label_name, current_tables) in enumerate(
        zip(present_labels[1:], loaded_tables[1:])
    ):
        if len(current_tables) != len(reference_tables):
            raise ValueError("All labels must have the same number of lens bins.")

        ratio_tables = [
            _build_ratio_table(current_table, reference_table)
            for current_table, reference_table in zip(current_tables, reference_tables)
        ]

        config_index = ratio_index + 1

        plot_radial_profile(
            ratio_tables,
            value_column="ds",
            title_label=f"Ratio to {present_labels[0]}",
            ax_list=ratio_axes,
            label_text=label_name,
            label_index=config_index,
            n_labels=len(present_labels),
            marker=markers[config_index % len(markers)],
            color_mode=color_mode,
            palette=palette,
            multiply_by_radius=multiply_by_radius,
            use_spline=use_spline,
            use_log_y=use_log_y,
            reference_line_y=reference_line_y,
            y_label=r"$\Delta\Sigma / \Delta\Sigma_{\mathrm{ref}}$",
            title_suffix="Ratio Profiles",
            marker_size=marker_size,
            capsize=capsize,
            elinewidth=elinewidth,
        )

    ratio_fig.suptitle(f"ΔΣ Ratio relative to {present_labels[0]}", y=0.996)
    ratio_fig.tight_layout()
    handles, legend_labels = ratio_axes[0].get_legend_handles_labels()
    if handles:
        ratio_fig.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            title="label",
            fontsize="small",
            frameon=False,
        )
    return ratio_fig, ratio_axes


def calculate_comparison_statistics(present_labels, loaded_tables):
    """Compute and print pairwise chi-square statistics across all configurations."""
    if len(loaded_tables) <= 1:
        return

    print("\n" + "=" * 70)
    print(f"{'Statistical Comparison (Chi-Square)':^70}")
    print("=" * 70)

    for i in range(len(present_labels)):
        for j in range(i + 1, len(present_labels)):
            label_1, tables_1 = present_labels[i], loaded_tables[i]
            label_2, tables_2 = present_labels[j], loaded_tables[j]

            print(f"\nComparing '{label_2}' vs '{label_1}':")
            chi2_total = 0.0
            ndof_total = 0

            for bin_idx, (tab_2, tab_1) in enumerate(zip(tables_2, tables_1)):
                t_2 = tab_2.copy()
                t_1 = tab_1.copy()
                t_2.sort("rp")
                t_1.sort("rp")

                ds_1 = np.asarray(t_1["ds"], dtype=float)
                err_1 = np.asarray(t_1["ds_err"], dtype=float)
                ds_2 = np.asarray(t_2["ds"], dtype=float)
                err_2 = np.asarray(t_2["ds_err"], dtype=float)

                err_comb2 = err_1**2 + err_2**2
                mask = (
                    np.isfinite(ds_1)
                    & np.isfinite(ds_2)
                    & np.isfinite(err_comb2)
                    & (err_comb2 > 0)
                )

                if np.any(mask):
                    chi2_bin = np.sum(
                        ((ds_2[mask] - ds_1[mask]) ** 2) / err_comb2[mask]
                    )
                    ndof_bin = np.sum(mask)
                    chi2_total += chi2_bin
                    ndof_total += ndof_bin
                    red_chi2 = chi2_bin / ndof_bin
                    print(
                        f"  Bin {bin_idx}: chi2 = {chi2_bin:7.2f} | "
                        f"ndof = {ndof_bin:2d} | red_chi2 = {red_chi2:6.2f}"
                    )
                else:
                    print(f"  Bin {bin_idx}: No valid data points.")

            if ndof_total > 0:
                print("-" * 70)
                print(
                    f"  OVERALL: chi2 = {chi2_total:7.2f} | ndof = {ndof_total:3d} | "
                    f"red_chi2 = {chi2_total / ndof_total:6.2f}"
                )
            print("-" * 70)


# %% Global Configuration

# List of (catalog_id, nbins, version_name) triples to compare.
# Available catalogs include observational sets ("camira", "cosine", "redm_pdr3_5band_free", ...)
# as well as ideal theoretical zero-scatter upper limits ("ideal_mdpl2", "ideal_colossus").
# If an ideal catalog is placed first, the ratio plot directly shows DeltaSigma_obs / DeltaSigma_ideal.

CONFIGS_TO_COMPARE = [
    ("ideal_mdpl2", "1bin", "Y3"),
    ("wh24", "1bin", "Y3"),
    ("wh20", "1bin", "Y3"),
]

# CONFIGS_TO_COMPARE = [
#     ("ideal_mdpl2", "4bin", "Y3"),
#     ("wh24", "4bin", "Y3"),
#     ("wh20", "4bin", "Y3"),
#     ("camira", "4bin", "Y3"),
#     ("rz_diff", "4bin", "Y3"),
# ]

CONFIGS_TO_COMPARE = [
    ("ideal_mdpl2", "1bin", "Y3"),
    ("redm_r16", "1bin", "Y3"),
    ("camira", "1bin", "Y3"),
    ("rz_diff", "1bin", "Y3"),
    ("rz_diff_mag_2_p_5", "1bin", "Y3"),
    ("rz_diff_mag_3_p_0", "1bin", "Y3"),
    ("rz_diff_two_box_red", "1bin", "Y3"),
]

# CONFIGS_TO_COMPARE = [
#     ("ideal_mdpl2", "1bin", "Y3"),
#     # ("redm_pdr3_5band_free", "1bin", "Y3"),
#     # ("logm_s16a", "1bin", "Y3"),
#     ("redm_r16", "1bin", "Y3"),
#     ("camira", "1bin", "Y3"),
#     # ("wh24", "1bin", "Y3"),
#     # ("wh20", "1bin", "Y3"),
#     # ("amico", "1bin", "Y3"),
#     # ("cosine", "1bin", "Y3"),
#     # ("pls", "1bin", "Y3"),
#     # ("regression", "1bin", "Y3"),
#     ("rz_diff", "1bin", "Y3"),
#     # ("rz_diff_lum", "1bin", "Y3"),
#     ("rz_diff_preset", "1bin", "Y3"),
#     # ("rz_diff_preset_lum", "1bin", "Y3"),
#     ("rz_diff_single_box", "1bin", "Y3"),
#     # ("rz_diff_single_box_lum", "1bin", "Y3"),
#     ("rz_diff_two_box_red", "1bin", "Y3"),
#     # ("cca1", "1bin", "Y3"),
#     # ("cca2", "1bin", "Y3"),
# ]

# CONFIGS_TO_COMPARE = [
#     ("ideal_mdpl2", "4bin", "Y3"),
#     # ("redm_pdr3_5band_free", "1bin", "Y3"),
#     # ("logm_s16a", "1bin", "Y3"),
#     # ("redm_r16", "4bin", "Y3"),  # Only has 3 bins (lambda >= 20 cut exhausts catalog before bin3)
#     ("camira", "4bin", "Y3"),
#     # ("wh24", "4bin", "Y3"),
#     # ("wh20", "4bin", "Y3"),
#     # ("amico", "1bin", "Y3"),
#     # ("cosine", "1bin", "Y3"),
#     # ("pls", "1bin", "Y3"),
#     # ("regression", "1bin", "Y3"),
#     ("rz_diff", "4bin", "Y3"),
#     ("rz_diff_lum", "4bin", "Y3"),
#     ("rz_diff_preset", "4bin", "Y3"),
#     ("rz_diff_preset_lum", "4bin", "Y3"),
#     # ("cca1", "1bin", "Y3"),
#     # ("cca2", "1bin", "Y3"),
# ]

# CONFIGS_TO_COMPARE = [
#     ("ideal_mdpl2", "1bin", "Y3"),
#     ("rz_diff", "1bin", "Y3"),
# ]


# CONFIGS_TO_COMPARE = [
#     ("ideal_mdpl2", "1bin", "Y3"),
#     ("cca1", "1bin", "Y3"),
#     ("cca1_specz", "1bin", "Y3"),
#     ("cca2", "1bin", "Y3"),
#     ("cca2_specz", "1bin", "Y3"),
# ]

# CONFIGS_TO_COMPARE = [
#     ("ideal_mdpl2", "1bin", "Y3"),
#     # ("redm_r16", "1bin", "Y3"),
#     ("redm_r16_hectomap", "1bin", "Y3"),
#     ("redm_pdr3_5band_free", "1bin", "Y3"),
# ]

# CONFIGS_TO_COMPARE = [
#     ("ideal_mdpl2", "4bin", "Y3"),
#     ("camira", "4bin", "Y3"),
#     # ("redm_r16", "4bin", "Y3"),
#     ("wh24", "4bin", "Y3"),
#     ("wh20", "4bin", "Y3"),
#     ("zou21", "4bin", "Y3"),
#     # ("yang21_mass", "4bin", "Y3"),
#     ("yang21_richness", "4bin", "Y3"),
#     # ("clumpr_mass", "4bin", "Y3"),
#     ("clumpr_richness", "4bin", "Y3"),
#     ("rz_diff", "4bin", "Y3"),
#     # ("rz_diff_preset", "4bin", "Y3"),
#     # ("rz_diff_single_box", "4bin", "Y3"),
#     # ("rz_diff_two_box_red", "4bin", "Y3"),
# ]

DISPLAY_NAMES = {
    "ideal_mdpl2": "Ideal MDPL2",
    "ideal_colossus": "Ideal Colossus",
    "camira": "CAMIRA",
    "redm_r16": "redMaPPer R16",
    "redm_r16_hectomap": "redMaPPer R16 (HectoMAP)",
    "redm_pdr3_5band_free": "redMaPPer PDR3",
    "wh20": "WH20",
    "wh24": "WH24",
    "zou21": "Zou21",
    "yang21_mass": "Yang21 (Halo Mass)",
    "yang21_richness": "Yang21 (Richness)",
    "clumpr_mass": "CluMPR (Mass)",
    "clumpr_richness": "CluMPR (Richness)",
    "amico": "AMICO",
    "cosine": "COSINE",
    "pls": "PLS",
    "regression": "Regression",
    "rz_diff": "r-z Diff",
    "rz_diff_lum": "r-z Diff (Lum)",
    "rz_diff_preset": "r-z Diff (Preset)",
    "rz_diff_preset_lum": "r-z Diff (Preset Lum)",
    "rz_diff_single_box": "r-z Diff (Single Box)",
    "rz_diff_single_box_lum": "r-z Diff (Single Box Lum)",
    "rz_diff_two_box_red": "r-z Diff (Two Box Red)",
    "rz_diff_two_box_red_lum": "r-z Diff (Two Box Red Lum)",
    "rz_diff_mag_2_p_5": "r-z Diff (m*+2.5)",
    "rz_diff_mag_3_p_0": "r-z Diff (m*+3.0)",
    "cca1": "CCA1",
    "cca1_specz": "CCA1 (spec-z)",
    "cca2": "CCA2",
    "cca2_specz": "CCA2 (spec-z)",
    "logm_s16a": "logM* (S16a)",
}

CATALOG_STYLES = {
    "ideal_mdpl2": {"color": "#000000", "marker": "o"},
    "ideal_colossus": {"color": "#555555", "marker": "s"},
    "camira": {"color": "#EE6677", "marker": "s"},
    "redm_r16": {"color": "#4477AA", "marker": "d"},
    "redm_r16_hectomap": {"color": "#4477AA", "marker": "d"},
    "redm_s16a": {"color": "#4477AA", "marker": "x"},
    "redm_pdr3_5band_free": {"color": "#CC79A7", "marker": "+"},
    "wh20": {"color": "#66BB55", "marker": "^"},
    "wh24": {"color": "#228833", "marker": "o"},
    "zou21": {"color": "#CCBB44", "marker": "v"},
    "yang21_mass": {"color": "#66CCEE", "marker": "^"},
    "yang21_richness": {"color": "#332288", "marker": "D"},
    "clumpr_mass": {"color": "#AA3377", "marker": "p"},
    "clumpr_richness": {"color": "#EE8866", "marker": "h"},
    "rz_diff": {"color": "#E69F00", "marker": "P"},
    "rz_diff_lum": {"color": "#E69F00", "marker": "P"},
    "rz_diff_preset": {"color": "#D55E00", "marker": "*"},
    "rz_diff_preset_lum": {"color": "#D55E00", "marker": "*"},
    "rz_diff_single_box": {"color": "#009E73", "marker": ">"},
    "rz_diff_single_box_lum": {"color": "#009E73", "marker": ">"},
    "rz_diff_two_box_red": {"color": "#B91C1C", "marker": "8"},
    "rz_diff_two_box_red_lum": {"color": "#B91C1C", "marker": "8"},
    "rz_diff_mag_2_p_5": {"color": "#0284C7", "marker": "^"},
    "rz_diff_mag_3_p_0": {"color": "#059669", "marker": "v"},
    "rz_diff_six_param": {"color": "#2563EB", "marker": "X"},
    "amico": {"color": "#10B981", "marker": "*"},
    "cosine": {"color": "#009988", "marker": "H"},
    "pls": {"color": "#7B2CBF", "marker": "d"},
    "regression": {"color": "#334155", "marker": "p"},
    "cca1": {"color": "#0F766E", "marker": "^"},
    "cca1_specz": {"color": "#0F766E", "marker": "^"},
    "cca2": {"color": "#56B4E9", "marker": "v"},
    "cca2_specz": {"color": "#56B4E9", "marker": "v"},
    "logm_s16a": {"color": "#DDAA33", "marker": "s"},
}

# Distinct marker symbols ensuring no repetition for up to 21 samples
MARKERS = [
    "o",  # Circle
    "s",  # Square
    "d",  # Diamond (thin)
    "^",  # Triangle Up
    "v",  # Triangle Down
    "<",  # Triangle Left
    "D",  # Diamond
    "p",  # Pentagon
    "h",  # Hexagon 1
    "P",  # Plus (filled)
    "*",  # Star
    ">",  # Triangle Right
    "8",  # Octagon
    "X",  # Cross (filled)
    "H",  # Hexagon 2
    "x",  # Cross (unfilled)
    "+",  # Plus (unfilled)
    "1",  # Tri Down
    "2",  # Tri Up
    "3",  # Tri Left
    "4",  # Tri Right
]

# Curated high-contrast 30-color palette aligned across project scripts
LABEL_PALETTE = [
    "#000000",  # Black (Ideal MDPL2)
    "#EE6677",  # Red (CAMIRA)
    "#4477AA",  # Blue (redMaPPer R16)
    "#228833",  # Dark Green (WH24)
    "#CCBB44",  # Yellow-Olive (Zou21)
    "#66CCEE",  # Cyan (Yang21 Halo Mass)
    "#332288",  # Indigo (Yang21 Richness)
    "#AA3377",  # Purple (CluMPR Mass)
    "#EE8866",  # Coral (CluMPR Richness)
    "#E69F00",  # Orange (r-z Diff)
    "#10B981",  # Emerald (AMICO)
    "#D55E00",  # Vermilion (r-z Diff Preset)
    "#009E73",  # Bluish Green (r-z Diff Single Box)
    "#B91C1C",  # Crimson (r-z Diff Two Box Red)
    "#2563EB",  # Royal Blue (r-z Diff Six Param)
    "#7B2CBF",  # Violet (PLS)
    "#0F766E",  # Deep Teal (CCA1)
    "#334155",  # Slate (Regression)
    "#CC79A7",  # Rose Pink (redMaPPer PDR3)
    "#56B4E9",  # Sky Blue (CCA2)
    "#009988",  # Teal (COSINE)
    "#555555",  # Dark Gray (Ideal Colossus)
    "#DDAA33",  # Sand / Ochre (logM* S16a)
    "#8B4513",  # Saddle Brown
    "#D81B60",  # Magenta / Cerise
    "#88BB44",  # Apple Green
    "#662244",  # Dark Plum
    "#00A389",  # Mint / Turquoise
    "#2E3440",  # Nordic Charcoal
    "#994F00",  # Burnt Orange
]

COLOR_MODE = "by_label"
MARKER_SIZE = 4.0
CAPSIZE = 0.0
ERRORBAR_LINEWIDTH = 0.8

OUTPUT_MAIN_FIG = project_root / "output/plots_for_agents/visual_multi_main.png"
OUTPUT_RATIO_FIG = project_root / "output/plots_for_agents/visual_multi_ratio.png"


# %% [Stage 1: Load comparison data]
present_labels, loaded_tables, label_time_texts = load_comparison_data(
    CONFIGS_TO_COMPARE, project_root, display_names=DISPLAY_NAMES
)
active_palette, active_markers = resolve_comparison_styles(
    CONFIGS_TO_COMPARE,
    present_labels,
    CATALOG_STYLES,
    LABEL_PALETTE,
    MARKERS,
    display_names=DISPLAY_NAMES,
)


# %% [Stage 2: Plot main comparison]
MAIN_MULTIPLY_BY_RADIUS = True
MAIN_USE_LOG_Y = not MAIN_MULTIPLY_BY_RADIUS
MAIN_USE_SPLINE = False
MAIN_REFERENCE_LINE_Y = 0.0

if loaded_tables:
    n_bins = len(loaded_tables[0])
    fig, axes = plot_main_comparison(
        present_labels=present_labels,
        loaded_tables=loaded_tables,
        n_bins=n_bins,
        multiply_by_radius=MAIN_MULTIPLY_BY_RADIUS,
        use_spline=MAIN_USE_SPLINE,
        use_log_y=MAIN_USE_LOG_Y,
        reference_line_y=MAIN_REFERENCE_LINE_Y,
        color_mode=COLOR_MODE,
        palette=active_palette,
        markers=active_markers,
        marker_size=MARKER_SIZE,
        capsize=CAPSIZE,
        elinewidth=ERRORBAR_LINEWIDTH,
    )
    fig.savefig(OUTPUT_MAIN_FIG, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %% [Stage 3: Plot ratio comparison]
RATIO_MULTIPLY_BY_RADIUS = False
RATIO_USE_LOG_Y = False
RATIO_USE_SPLINE = False
RATIO_REFERENCE_LINE_Y = 1.0

if loaded_tables:
    n_bins = len(loaded_tables[0])
    ratio_result = plot_ratio_comparison(
        present_labels=present_labels,
        loaded_tables=loaded_tables,
        n_bins=n_bins,
        multiply_by_radius=RATIO_MULTIPLY_BY_RADIUS,
        use_spline=RATIO_USE_SPLINE,
        use_log_y=RATIO_USE_LOG_Y,
        reference_line_y=RATIO_REFERENCE_LINE_Y,
        color_mode=COLOR_MODE,
        palette=active_palette,
        markers=active_markers,
        marker_size=MARKER_SIZE,
        capsize=CAPSIZE,
        elinewidth=ERRORBAR_LINEWIDTH,
    )
    if ratio_result is not None:
        ratio_fig, ratio_axes = ratio_result
        ratio_fig.savefig(OUTPUT_RATIO_FIG, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(ratio_fig)


# %% [Stage 4: Statistical comparison]
if loaded_tables:
    calculate_comparison_statistics(present_labels, loaded_tables)
