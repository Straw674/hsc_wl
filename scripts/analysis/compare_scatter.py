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

        ax.fill_between(x_val, y_val - y_err, y_val + y_err, color=color, alpha=0.2)

        offset = 1.0 + 0.05 * (i - (len(data_dict) - 1) / 2.0)
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
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
