# %% [Initialization]
import sys
from pathlib import Path
from typing import Any

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

from astropy.table import Table  # noqa: E402

from hsc_wl.config import resolve_run_dir  # noqa: E402
from initial import *  # noqa: F401, F403

# %% Local Functions


def load_profile_summaries(
    labels: list[str], version: str, root_path: Path
) -> dict[str, Table]:
    """Load profile scatter summary FITS tables for configured labels.

    Parameters
    ----------
    labels : list of str
        Run labels (e.g. "rz_diff_1bin").
    version : str
        Data version (e.g. "Y3").
    root_path : Path
        Project root path.

    Returns
    -------
    dict of {str: Table}
        Loaded summary table for each label.
    """
    data = {}
    for lbl in labels:
        catalog_id, nbins = lbl.rsplit("_", 1)
        fits_path = (
            resolve_run_dir(lbl, "fit", root=root_path, version=version)
            / f"{catalog_id}_{nbins}_profile_fit.fits"
        )
        if not fits_path.exists():
            print(f"Warning: Summary FITS not found for {lbl} at {fits_path}")
            continue
        data[lbl] = Table.read(fits_path)
    return data


def _clean_err(err_val: Any) -> float:
    """Return standard float or np.nan, safely handling MaskedConstant."""
    if err_val is None or np.ma.is_masked(err_val) or not np.isfinite(err_val):
        return np.nan
    return float(err_val)


def _fmt_val_err(val: Any, err: Any, val_fmt: str = ".2f", err_fmt: str = ".2f") -> str:
    """Format value +/- error for markdown table."""
    v_str = (
        f"{float(val):{val_fmt}}"
        if (val is not None and not np.ma.is_masked(val) and np.isfinite(val))
        else "N/A"
    )
    if err is None or np.ma.is_masked(err) or not np.isfinite(err):
        return f"{v_str} ± --"
    return f"{v_str} ± {float(err):{err_fmt}}"


def format_comparison_table(summaries: dict[str, Table]) -> str:
    """Format a clean markdown table summarizing parameters across all catalogs."""
    header = (
        "| Catalog Label | Scatter $\\sigma$ | $\\alpha_c$ | $\\langle c \\rangle$ "
        "| $\\alpha_b$ | $\\langle b \\rangle$ | $\\langle \\log M \\rangle$ | $\\chi^2/\\nu$ |"
    )
    separator = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    rows = [header, separator]

    for lbl, tbl in summaries.items():
        for row in tbl:
            b_suffix = f" (bin {row['bin_id']})" if len(tbl) > 1 else ""
            line = (
                f"| `{lbl}{b_suffix}` "
                f"| {_fmt_val_err(row['scatter'], row['scatter_err'], '.3f', '.3f')} "
                f"| {_fmt_val_err(row['alpha_c'], row['alpha_c_err'], '.2f', '.2f')} "
                f"| {float(row['mean_c']):.1f} "
                f"| {_fmt_val_err(row['alpha_b'], row['alpha_b_err'], '.2f', '.2f')} "
                f"| {float(row['mean_b']):.2f} "
                f"| {float(row['mean_logm']):.3f} "
                f"| {float(row['chi2_reduced']):.3f} |"
            )
            rows.append(line)
    return "\n".join(rows)


def plot_profile_parameter_comparison(
    summaries: dict[str, Table],
    display_names: dict[str, str],
    output_path: Path,
) -> None:
    """Create a multi-panel comparison figure showing 2D parameter plane distributions."""
    labels = list(summaries.keys())
    n_samples = len(labels)
    if n_samples == 0:
        print("No samples available to plot.")
        return

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_samples, 10)))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    ax_mass_scatter = axes[0, 0]
    ax_mass_conc = axes[0, 1]
    ax_mass_bias = axes[1, 0]
    ax_mass_chi2 = axes[1, 1]

    handles = []
    catalog_names = []

    for i, lbl in enumerate(labels):
        row = summaries[lbl][0]  # First bin for 1-bin mode
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        name = display_names.get(lbl, lbl)
        catalog_names.append(name)

        logm = float(row["mean_logm"])
        scatter = float(row["scatter"])
        scatter_err = _clean_err(row["scatter_err"])
        alpha_c = float(row["alpha_c"])
        alpha_c_err = _clean_err(row["alpha_c_err"])
        alpha_b = float(row["alpha_b"])
        alpha_b_err = _clean_err(row["alpha_b_err"])
        chi2_red = float(row["chi2_reduced"])

        # 1. Mass - Scatter
        h = ax_mass_scatter.errorbar(
            logm,
            scatter,
            yerr=scatter_err,
            fmt=m,
            color=c,
            ecolor=c,
            capsize=3,
            elinewidth=1.2,
            markersize=7,
            label=name,
        )
        handles.append(h)

        # 2. Mass - Concentration
        ax_mass_conc.errorbar(
            logm,
            alpha_c,
            yerr=alpha_c_err,
            fmt=m,
            color=c,
            ecolor=c,
            capsize=3,
            elinewidth=1.2,
            markersize=7,
        )

        # 3. Mass - Bias
        ax_mass_bias.errorbar(
            logm,
            alpha_b,
            yerr=alpha_b_err,
            fmt=m,
            color=c,
            ecolor=c,
            capsize=3,
            elinewidth=1.2,
            markersize=7,
        )

        # 4. Mass - Reduced Chi2
        ax_mass_chi2.errorbar(
            logm,
            chi2_red,
            fmt=m,
            color=c,
            ecolor=c,
            capsize=3,
            elinewidth=1.2,
            markersize=7,
        )

    # Reference lines and styling
    # Panel 1: Mass - Scatter
    ax_mass_scatter.set_xlabel(
        r"$\langle \log M_{200m} \rangle\ [M_\odot/h]$", fontweight="normal"
    )
    ax_mass_scatter.set_ylabel(r"Scatter $\sigma_{\ln M}$", fontweight="normal")
    ax_mass_scatter.set_title("Mass – Scatter", fontweight="normal")
    ax_mass_scatter.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Mass - Concentration
    ax_mass_conc.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_mass_conc.set_xlabel(
        r"$\langle \log M_{200m} \rangle\ [M_\odot/h]$", fontweight="normal"
    )
    ax_mass_conc.set_ylabel(r"$\alpha_c = c / c_{\rm Diemer19}$", fontweight="normal")
    ax_mass_conc.set_title("Mass – Concentration", fontweight="normal")
    ax_mass_conc.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: Mass - Bias
    ax_mass_bias.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_mass_bias.set_xlabel(
        r"$\langle \log M_{200m} \rangle\ [M_\odot/h]$", fontweight="normal"
    )
    ax_mass_bias.set_ylabel(r"$\alpha_b = b / b_{\rm Tinker10}$", fontweight="normal")
    ax_mass_bias.set_title("Mass – Bias", fontweight="normal")
    ax_mass_bias.grid(True, linestyle=":", alpha=0.5)

    # Panel 4: Mass - Reduced Chi2
    ax_mass_chi2.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_mass_chi2.set_xlabel(
        r"$\langle \log M_{200m} \rangle\ [M_\odot/h]$", fontweight="normal"
    )
    ax_mass_chi2.set_ylabel(r"Reduced $\chi^2 / \nu$", fontweight="normal")
    ax_mass_chi2.set_title(r"Mass – Reduced $\chi^2$", fontweight="normal")
    ax_mass_chi2.grid(True, linestyle=":", alpha=0.5)

    data_axes = [
        ax_mass_scatter,
        ax_mass_conc,
        ax_mass_bias,
        ax_mass_chi2,
    ]
    for ax in data_axes:
        ax.margins(0.08)

    fig.legend(
        handles,
        catalog_names,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=True,
        fontsize=9.5,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure to: {output_path}")
    plt.show()
    plt.close(fig)


# %% Global Configuration

LABELS = [
    "redm_r16_1bin",
    "camira_1bin",
    "rz_diff_1bin",
    "rz_diff_preset_1bin",
    "rz_diff_single_box_1bin",
    "rz_diff_two_box_red_match_recovery_1bin",
    "wh24_1bin",
]

VERSION = "Y3"

DISPLAY_NAMES = {
    "redm_r16_1bin": "redMaPPer (R16)",
    "camira_1bin": "CAMIRA",
    "rz_diff_1bin": "rz_diff (Baseline)",
    "rz_diff_preset_1bin": "rz_diff (Preset)",
    "rz_diff_single_box_1bin": "rz_diff (Single Box)",
    "rz_diff_two_box_red_match_recovery_1bin": "rz_diff (Two Box Recovery)",
    "wh24_1bin": "Wen & Han (2024)",
    "zou21_1bin": "Zou et al. (2021)",
}

OUTPUT_FIG = project_root / "output/plots_for_agents/compare_theory_scatter_1bin.png"

# %% [Stage 1: Load summaries and display table]
summaries = load_profile_summaries(LABELS, version=VERSION, root_path=project_root)
table_md = format_comparison_table(summaries)
print("\n" + table_md + "\n")

# %% [Stage 2: Plot multi-parameter comparison figure]
plot_profile_parameter_comparison(
    summaries=summaries,
    display_names=DISPLAY_NAMES,
    output_path=OUTPUT_FIG,
)
