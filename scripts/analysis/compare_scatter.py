# %% [Initialization]
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

import pickle  # noqa: E402

from hsc_wl.config import resolve_run_dir  # noqa: E402
from initial import *  # noqa: F401,F403

# %% Local Functions


def load_scatter_summaries(labels, root_path, version: str = "Y3"):
    """Load scatter summary pickles for each run label.

    Returns
    -------
    dict
        Mapping from label -> "custom_sample" summary table.
    """
    data_dict = {}
    for label in labels:
        catalog_id, nbins = label.rsplit("_", 1)
        pkl_dir = resolve_run_dir(label, "pkl", root=root_path, version=version)
        pkl_path = pkl_dir / f"{catalog_id}_{nbins}_{version}_sum.pkl"
        if not pkl_path.exists():
            pkl_path = pkl_dir / f"{catalog_id}_{nbins}_sum.pkl"
        if not pkl_path.exists():
            print(f"Warning: {pkl_path} does not exist. Skipping.")
            continue

        with open(pkl_path, "rb") as f:
            res = pickle.load(f)
            data_dict[label] = res["custom_sample"]

    print(f"Loaded data for: {list(data_dict.keys())}")
    return data_dict


def get_sample_style(
    label,
    sample_styles,
    default_colors=None,
    default_markers=None,
    fallback_index=0,
):
    """Resolve catalog-family styles independently of sample order or presence."""
    lens_label = label[0] if isinstance(label, tuple) else str(label)
    catalog_id = lens_label.removesuffix("_4bin")
    catalog_family = lens_label.split("_", 1)[0]
    for key in (label, lens_label, catalog_id, catalog_family):
        if key in sample_styles:
            return sample_styles[key]
    colors = default_colors or COMPARISON_COLORS
    markers = default_markers or MARKERS
    return {
        "color": colors[fallback_index % len(colors)],
        "marker": markers[fallback_index % len(markers)],
        "filled": True,
    }


def plot_scatter_comparison(
    data_dict, labels, display_names, rho_bins, styles, output_path
):
    """Plot a scatter comparison figure in the style of fig8 in jianbing.

    Parameters
    ----------
    data_dict : dict
        Mapping from label tuple -> summary table (with sig_med_bt, sig_err_bt).
    labels : list[tuple[str, str]]
        Ordered list of (lens_label, source_label) pairs to style against.
    display_names : dict
        Optional display-name overrides for legend entries.
    rho_bins : np.ndarray
        Number-density bin edges/centers (Mpc^-3).
    styles : dict
        Mapping from label tuple to color, marker, and fill settings.
    output_path : Path
        Destination for saving the figure.
    """
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 14

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xscale("log", nonpositive="clip")
    ax.grid(False)

    available_labels = [label for label in labels if label in data_dict]
    for i, label in enumerate(available_labels):
        sum_tab = data_dict[label]
        y_val = np.asarray(sum_tab["sig_med_bt"])
        y_err = np.asarray(sum_tab["sig_err_bt"])
        x_val = rho_bins[: len(y_val)]

        style = styles[label]
        color = style["color"]
        marker = style["marker"]

        lens_label = label
        catalog_id = lens_label.removesuffix("_4bin").removesuffix("_1bin")
        display_name = (
            display_names.get(label)
            or display_names.get(lens_label)
            or display_names.get(catalog_id)
            or catalog_id
        )

        offset = 1.0 + 0.05 * (i - (len(data_dict) - 1) / 2.0)
        ax.errorbar(
            x_val * offset,
            y_val,
            yerr=y_err,
            fmt="none",
            color=color,
            elinewidth=1.2,
            capsize=3,
            alpha=0.9,
        )
        is_unfilled_marker = marker in {"x", "+", "1", "2", "3", "4", "|", "_"}
        if is_unfilled_marker:
            ax.scatter(
                x_val * offset,
                y_val,
                s=150,
                marker=marker,
                alpha=0.9,
                color=color,
                linewidth=1.5,
                label=display_name,
            )
        else:
            ax.scatter(
                x_val * offset,
                y_val,
                s=150,
                marker=marker,
                alpha=0.9,
                facecolor=color if style["filled"] else "none",
                edgecolor="k" if style["filled"] else color,
                linewidth=1.5,
                label=display_name,
            )

    ax.set_xlabel(r"$N(>M)\ [\rm Mpc^{-3}]$", fontsize=20)
    ax.set_ylabel(r"$\sigma_{\mathcal{M}|\mathcal{O}}\ [\rm dex]$", fontsize=20)
    ax.legend(loc="best", fontsize=11, ncol=2)

    ax.set_xlim(np.max(rho_bins) * 1.5, np.min(rho_bins) * 0.5)

    for i, rho in enumerate(rho_bins):
        ax.text(
            rho,
            ax.get_ylim()[1] * 0.95,
            f"Bin {i + 1}",
            horizontalalignment="center",
            fontsize=12,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved scatter comparison: {output_path}")
    plt.show()
    plt.close(fig)


def plot_grouped_scatter_comparison(
    data_dict, labels, display_names, rho_bins, colors, offset_width, output_path
):
    """Compare scatter within discrete bins using offset points and error bars."""
    available_labels = [label for label in labels if label in data_dict]
    bin_positions = np.arange(1, len(rho_bins) + 1)
    offsets = (
        np.linspace(-offset_width / 2, offset_width / 2, len(available_labels))
        if len(available_labels) > 1
        else np.array([0.0])
    )

    with plt.rc_context({"font.size": 12, "font.weight": "normal"}):
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for label, offset in zip(available_labels, offsets, strict=True):
            summary = data_dict[label]
            idx = labels.index(label)
            if isinstance(colors, dict):
                style_entry = colors.get(label, {})
                color = (
                    style_entry.get("color")
                    if isinstance(style_entry, dict)
                    else style_entry
                )
                marker = (
                    style_entry.get("marker", "o")
                    if isinstance(style_entry, dict)
                    else "o"
                )
            else:
                color = colors[idx % len(colors)]
                marker = MARKERS[idx % len(MARKERS)]

            run_label = label
            catalog_id = run_label.removesuffix("_4bin").removesuffix("_1bin")
            display_name = (
                display_names.get(label)
                or display_names.get(run_label)
                or display_names.get(catalog_id)
                or catalog_id
            )
            n_b = len(summary["sig_med_bt"])
            ax.errorbar(
                bin_positions[:n_b] + offset,
                np.asarray(summary["sig_med_bt"]),
                yerr=np.asarray(summary["sig_err_bt"]),
                fmt=marker,
                color=color,
                markersize=6,
                markeredgewidth=1,
                elinewidth=1.2,
                capsize=3,
                label=display_name,
            )

        ax.set_xticks(bin_positions, [f"Bin {i}" for i in bin_positions])
        ax.set_xlim(len(rho_bins) + 0.55, 0.45)
        ax.set_ylabel(r"$\sigma_{\mathcal{M}|\mathcal{O}}\ [\rm dex]$")
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.yaxis.grid(True, color="0.9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(2, len(available_labels)),
            frameon=False,
            fontsize=11,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved grouped scatter comparison: {output_path}")
        plt.show()
        plt.close(fig)


def plot_panel_scatter_comparison(
    data_dict, labels, display_names, rho_bins, colors, output_path
):
    """Compare samples in one panel per bin using shared scatter limits."""
    available_labels = [label for label in labels if label in data_dict]
    sample_names = [
        display_names.get(label)
        or display_names.get(label.removesuffix("_4bin").removesuffix("_1bin"))
        or label
        for label in available_labels
    ]
    with plt.rc_context({"font.size": 12, "font.weight": "normal"}):
        fig, axes = plt.subplots(
            1,
            len(rho_bins),
            figsize=(12, 4),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        for ax, bin_index in zip(
            axes.flat, reversed(range(len(rho_bins))), strict=True
        ):
            for sample_index, label in enumerate(available_labels):
                summary = data_dict[label]
                if bin_index >= len(summary["sig_med_bt"]):
                    continue
                idx = labels.index(label)
                if isinstance(colors, dict):
                    style_entry = colors.get(label, {})
                    color = (
                        style_entry.get("color")
                        if isinstance(style_entry, dict)
                        else style_entry
                    )
                    marker = (
                        style_entry.get("marker", "o")
                        if isinstance(style_entry, dict)
                        else "o"
                    )
                else:
                    color = colors[idx % len(colors)]
                    marker = MARKERS[idx % len(MARKERS)]

                ax.errorbar(
                    np.asarray(summary["sig_med_bt"])[bin_index],
                    sample_index,
                    xerr=np.asarray(summary["sig_err_bt"])[bin_index],
                    fmt=marker,
                    color=color,
                    markersize=6,
                    elinewidth=1.2,
                    capsize=3,
                )
            ax.set_title(f"Bin {bin_index + 1}", fontweight="normal")
            ax.set_axisbelow(True)
            ax.grid(False)
            ax.xaxis.grid(True, color="0.9", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_yticks(np.arange(len(available_labels)), sample_names)

        axes[0, 0].set_ylim(len(available_labels) - 0.5, -0.5)
        axes[0, 0].invert_xaxis()
        fig.supxlabel(
            r"$\sigma_{\mathcal{M}|\mathcal{O}}\ [\rm dex]$", fontweight="normal"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved panel scatter comparison: {output_path}")
        plt.show()
        plt.close(fig)


# %% Global Configuration

# Labels to compare (must be 4bin configurations, e.g. "amico_4bin").
# For all available labels, refer to `RUN_REGISTRY` in `src/hsc_wl/config.py`.
LABELS = [
    "rz_diff_4bin",
    # "rz_diff_preset_4bin",
    # "rz_diff_single_box_4bin",
    # "rz_diff_two_box_red_4bin",
    "camira_4bin",
    "wh24_4bin",
    "zou21_4bin",
    "yang21_mass_4bin",
    "yang21_richness_4bin",
    "clumpr_mass_4bin",
    "clumpr_richness_4bin",
]

# Display names for labels in the legend
DISPLAY_NAMES = {
    "camira_4bin": "CAMIRA",
    "wh24_4bin": "WH24",
    "zou21_4bin": "Zou21",
    "yang21_mass_4bin": "Yang21 (Halo Mass)",
    "yang21_richness_4bin": "Yang21 (Richness)",
    "clumpr_mass_4bin": "CluMPR (Mass)",
    "clumpr_richness_4bin": "CluMPR (Richness)",
    "rz_diff_4bin": "r-z Diff",
    "rz_diff_preset_4bin": "r-z Diff (Preset)",
    "rz_diff_single_box_4bin": "r-z Diff (Single Box)",
    "rz_diff_two_box_red_4bin": "r-z Diff (Two Box Red)",
    "rz_diff_six_param_4bin": "r-z Diff (Six Param)",
    "redm_r16_4bin": "redMaPPer R16",
    "amico_4bin": "AMICO",
    "cosine_4bin": "COSINE",
    "pls_4bin": "PLS",
    "regression_4bin": "Regression",
    "cca1_4bin": "CCA1",
    "cca2_4bin": "CCA2",
    "ideal_mdpl2_4bin": "Ideal MDPL2",
    "ideal_colossus_4bin": "Ideal Colossus",
}

# Bind first-figure styles to catalog families across run and source versions.
SAMPLE_STYLES = {
    "camira": {"color": "#EE6677", "marker": "s", "filled": False},
    "wh24": {"color": "#228833", "marker": "o", "filled": True},
    "zou21": {"color": "#CCBB44", "marker": "v", "filled": True},
    "yang21_mass": {"color": "#66CCEE", "marker": "^", "filled": True},
    "yang21_richness": {"color": "#332288", "marker": "D", "filled": True},
    "clumpr_mass": {"color": "#AA3377", "marker": "p", "filled": True},
    "clumpr_richness": {"color": "#EE8866", "marker": "h", "filled": True},
    "redm_r16": {"color": "#4477AA", "marker": "d", "filled": True},
    "redm_s16a": {"color": "#4477AA", "marker": "x", "filled": False},
    "redm_pdr3_5band_free": {"color": "#CC79A7", "marker": "+", "filled": False},
    "amico": {"color": "#10B981", "marker": "*", "filled": True},
    "rz_diff": {"color": "#E69F00", "marker": "P", "filled": True},
    "rz_diff_preset": {"color": "#D55E00", "marker": "*", "filled": True},
    "rz_diff_single_box": {"color": "#009E73", "marker": ">", "filled": True},
    "rz_diff_two_box_red": {"color": "#B91C1C", "marker": "8", "filled": True},
    "rz_diff_six_param": {"color": "#2563EB", "marker": "X", "filled": True},
    "cosine": {"color": "#009988", "marker": "H", "filled": True},
    "pls": {"color": "#7B2CBF", "marker": "d", "filled": True},
    "regression": {"color": "#334155", "marker": "p", "filled": True},
    "cca1": {"color": "#0F766E", "marker": "^", "filled": True},
    "cca2": {"color": "#56B4E9", "marker": "v", "filled": True},
    "ideal_mdpl2": {"color": "#000000", "marker": "o", "filled": False},
    "ideal_colossus": {"color": "#555555", "marker": "s", "filled": False},
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
COMPARISON_COLORS = [
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
    "#000000",  # Black (Ideal MDPL2)
]

# Hardcoded rho bins (Mpc^-3) as they might be missing from some pkl files
RHO_BINS = np.array(
    [
        5.315651368706627e-07,
        2.0035916697432675e-06,
        6.7467882756661044e-06,
        1.184776910832881e-05,
    ]
)

# Output destination for the saved figure
OUTPUT_FIG = project_root / "output/plots_for_agents/compare_scatter.png"


# %% [Stage 1: Load data]
data_dict = load_scatter_summaries(LABELS, project_root)
styles = {
    label: get_sample_style(
        label,
        SAMPLE_STYLES,
        default_colors=COMPARISON_COLORS,
        default_markers=MARKERS,
        fallback_index=i,
    )
    for i, label in enumerate(LABELS)
}


# %% [Stage 2: Plot comparison]
if data_dict:
    plot_scatter_comparison(
        data_dict=data_dict,
        labels=LABELS,
        display_names=DISPLAY_NAMES,
        rho_bins=RHO_BINS,
        styles=styles,
        output_path=OUTPUT_FIG,
    )
else:
    print("No data loaded. Skipping plot.")


# %% [Stage 3: Plot grouped comparison]
GROUPED_OFFSET_WIDTH = 0.15
GROUPED_OUTPUT_FIG = (
    project_root / "output/plots_for_agents/compare_scatter_grouped.png"
)

if data_dict:
    plot_grouped_scatter_comparison(
        data_dict=data_dict,
        labels=LABELS,
        display_names=DISPLAY_NAMES,
        rho_bins=RHO_BINS,
        colors=styles,
        offset_width=GROUPED_OFFSET_WIDTH,
        output_path=GROUPED_OUTPUT_FIG,
    )
else:
    print("No data loaded. Skipping grouped plot.")


# %% [Stage 4: Plot panel comparison]
PANEL_OUTPUT_FIG = project_root / "output/plots_for_agents/compare_scatter_panels.png"

if data_dict:
    plot_panel_scatter_comparison(
        data_dict=data_dict,
        labels=LABELS,
        display_names=DISPLAY_NAMES,
        rho_bins=RHO_BINS,
        colors=styles,
        output_path=PANEL_OUTPUT_FIG,
    )
else:
    print("No data loaded. Skipping panel plot.")
