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

from initial import *  # noqa: F401,F403

# %% Local Functions


def load_scatter_summaries(labels, root_path):
    """Load scatter summary pickles for each (run_label, version) pair.

    Returns
    -------
    dict
        Mapping from label tuple -> "custom_sample" summary table.
    """
    data_dict = {}
    for label in labels:
        run_label, version = label
        catalog_id, nbins = run_label.rsplit("_", 1)
        pkl_path = (
            root_path
            / f"output/{catalog_id}/{nbins}/{version}/pkl/{catalog_id}_{nbins}_{version}_sum.pkl"
        )
        if not pkl_path.exists():
            print(f"Warning: {pkl_path} does not exist. Skipping.")
            continue

        with open(pkl_path, "rb") as f:
            res = pickle.load(f)
            data_dict[label] = res["custom_sample"]

    print(f"Loaded data for: {list(data_dict.keys())}")
    return data_dict


def plot_scatter_comparison(data_dict, labels, display_names, rho_bins, output_path):
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
    output_path : Path
        Destination for saving the figure.
    """
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 14

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xscale("log", nonpositive="clip")
    ax.grid(False)

    for i, label in enumerate(data_dict.keys()):
        sum_tab = data_dict[label]
        x_val = rho_bins
        y_val = sum_tab["sig_med_bt"]
        y_err = sum_tab["sig_err_bt"]

        idx = labels.index(label)
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        lens_label, source_label = label
        display_name = display_names.get(label, f"{lens_label} ({source_label})")

        offset = 1.0 + 0.05 * (i - (len(data_dict) - 1) / 2.0)
        ax.errorbar(
            x_val * offset,
            y_val,
            yerr=y_err,
            fmt="-",
            color=color,
            linewidth=1.5,
            elinewidth=1.2,
            capsize=3,
            alpha=0.9,
        )
        ax.scatter(
            x_val * offset,
            y_val,
            s=150,
            marker=marker,
            alpha=0.9,
            facecolor=color,
            edgecolor="k",
            linewidth=1.5,
            label=display_name,
        )

    ax.set_xlabel(r"$N(>M)\ [\rm Mpc^{-3}]$", fontsize=20)
    ax.set_ylabel(r"$\sigma_{\mathcal{M}|\mathcal{O}}\ [\rm dex]$", fontsize=20)
    ax.legend(loc="best", fontsize=15)

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
            color = colors[labels.index(label) % len(colors)]
            run_label, version = label
            display_name = display_names.get(
                label, f"{run_label.removesuffix('_4bin')} ({version})"
            )
            ax.errorbar(
                bin_positions + offset,
                np.asarray(summary["sig_med_bt"]),
                yerr=np.asarray(summary["sig_err_bt"]),
                fmt="o",
                color=color,
                markersize=5,
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
        display_names.get(label, f"{label[0].removesuffix('_4bin')} ({label[1]})")
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
                ax.errorbar(
                    np.asarray(summary["sig_med_bt"])[bin_index],
                    sample_index,
                    xerr=np.asarray(summary["sig_err_bt"])[bin_index],
                    fmt="o",
                    color=colors[labels.index(label) % len(colors)],
                    markersize=5,
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

# Labels to compare (must be 4bin configurations, e.g. ("amico_4bin", "Y3")).
# For all available labels, refer to `RUN_REGISTRY` in `src/hsc_wl/config.py`.
LABELS = [
    ("redm_s16a_hectomap_4bin", "Y3"),
    ("logm_s16a_hectomap_4bin", "Y3"),
    ("redm_pdr3_3band_fixed_s16a_4bin", "Y3"),
    ("cosine_s16a_4bin", "Y3"),
    ("camira_hecto_s16a_4bin", "Y3"),
    ("amico_4bin", "Y3"),
]

# Optional: display names for labels in the legend
DISPLAY_NAMES = {}

# Style palette (consistent across stages if reused)
COLORS = ["#33a02c", "#984ea3", "#ff7f00", "#1f78b4", "#e41a1c", "#a65628", "#f781bf"]
MARKERS = ["H", "P", "s", "o", "D", "v", "^"]

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


# %% [Stage 2: Plot comparison]
if data_dict:
    plot_scatter_comparison(
        data_dict=data_dict,
        labels=LABELS,
        display_names=DISPLAY_NAMES,
        rho_bins=RHO_BINS,
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
        colors=COLORS,
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
        colors=COLORS,
        output_path=PANEL_OUTPUT_FIG,
    )
else:
    print("No data loaded. Skipping panel plot.")
