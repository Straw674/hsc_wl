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

from hsc_wl.theoretical_limit import get_theoretical_upper_limit
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


def load_comparison_data(configs_to_compare, root_path):
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
    present_labels = []
    loaded_tables = []
    label_time_texts = []

    for catalog_id, nbins, version_name in configs_to_compare:
        current_dir = root_path / f"output/{catalog_id}/{nbins}/{version_name}/dsigma"
        if not current_dir.exists():
            print(f"Warning: {current_dir} does not exist. Skipping.")
            continue
        time_text = _get_result_time_text(current_dir)
        display_name = f"{catalog_id} ({nbins})"
        print(f"Loading data for {display_name} | file time: {time_text}")
        current_tables = load_result_tables(current_dir)
        present_labels.append(display_name)
        loaded_tables.append(current_tables)
        label_time_texts.append(time_text)

    if not loaded_tables:
        print("No data loaded. Exiting.")
        return present_labels, loaded_tables, label_time_texts

    n_bins = len(loaded_tables[0])
    for tables in loaded_tables:
        if len(tables) != n_bins:
            raise ValueError("All labels must have the same number of lens bins.")

    return present_labels, loaded_tables, label_time_texts


def load_theoretical_limit_data(nbins, root_path, source="simulation", top_n=500):
    """Load theoretical upper limit tables (sigma=0) for comparison.

    Returns
    -------
    limit_tables : list[Table] or None
    limit_label : str or None
    """
    try:
        limit_tables = get_theoretical_upper_limit(
            root_path=root_path,
            nbins=nbins,
            top_n=top_n,
            source=source,
        )
        src_label = "MDPL2" if source == "simulation" else "Colossus"
        limit_label = f"Ideal Upper Limit ({src_label}, $\\sigma=0$)"
        print(f"Loaded {limit_label} for {nbins} (source={source})")
        return limit_tables, limit_label
    except Exception as exc:
        print(f"Warning: Could not load theoretical upper limit: {exc}")
        return None, None


def plot_main_comparison(
    present_labels,
    loaded_tables,
    n_bins,
    multiply_by_radius,
    use_spline,
    use_log_y,
    reference_line_y,
    color_mode,
    limit_tables=None,
    limit_label=None,
):
    """Plot the main ΔΣ comparison profile across all labels."""
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
            marker=MARKERS[i % len(MARKERS)],
            color_mode=color_mode,
            palette=LABEL_PALETTE,
            multiply_by_radius=multiply_by_radius,
            use_spline=use_spline,
            use_log_y=use_log_y,
            reference_line_y=reference_line_y,
        )

    # Overlay theoretical upper limit (sigma=0) if available
    if limit_tables is not None:
        for b_idx in range(min(n_bins, len(limit_tables))):
            plot_theoretical_upper_limit(
                ax=axes[b_idx],
                limit_table=limit_tables[b_idx],
                multiply_by_radius=multiply_by_radius,
                use_log_y=use_log_y,
                label=limit_label if b_idx == 0 else None,
                color="#111111",
                ls="--",
                lw=2.2,
                alpha=0.95,
                show_band=True,
                fill_alpha=0.12,
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
    limit_tables=None,
    limit_label=None,
):
    """Plot ratio ΔΣ / ΔΣ_ref across all non-reference labels."""
    if not loaded_tables or len(loaded_tables) <= 1:
        return None

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
            marker=MARKERS[config_index % len(MARKERS)],
            color_mode=color_mode,
            palette=LABEL_PALETTE,
            multiply_by_radius=multiply_by_radius,
            use_spline=use_spline,
            use_log_y=use_log_y,
            reference_line_y=reference_line_y,
            y_label=r"$\Delta\Sigma / \Delta\Sigma_{\mathrm{ref}}$",
            title_suffix="Ratio Profiles",
        )

    # Overlay theoretical limit ratio (limit / ref) if available
    if limit_tables is not None:
        for b_idx in range(min(n_bins, len(limit_tables))):
            plot_theoretical_limit_ratio(
                ax=ratio_axes[b_idx],
                limit_table=limit_tables[b_idx],
                reference_table=reference_tables[b_idx],
                use_log_y=use_log_y,
                label=f"{limit_label} / Ref" if b_idx == 0 else None,
                color="#111111",
                ls="--",
                lw=2.2,
                alpha=0.95,
                show_band=True,
                fill_alpha=0.12,
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


def calculate_comparison_statistics(
    present_labels, loaded_tables, limit_tables=None, limit_label=None
):
    """Compute and print pairwise chi-square and theoretical limit recovery statistics."""
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

    # Theoretical limit recovery metrics
    if limit_tables is not None and len(loaded_tables) > 0:
        from scipy.interpolate import interp1d

        print("\n" + "=" * 70)
        print(f"{'Recovery of Theoretical Upper Limit (sigma=0)':^70}")
        print("=" * 70)

        for label_name, tables in zip(present_labels, loaded_tables):
            print(f"\nCatalog: '{label_name}' vs {limit_label}:")
            for bin_idx, (t_obs, t_lim) in enumerate(zip(tables, limit_tables)):
                rp_obs = np.asarray(t_obs["rp"], dtype=float)
                ds_obs = np.asarray(t_obs["ds"], dtype=float)
                rp_lim = np.asarray(t_lim["rp"], dtype=float)
                ds_lim = np.asarray(t_lim["ds"], dtype=float)

                f_lim = interp1d(
                    np.log10(rp_lim),
                    ds_lim,
                    kind="cubic",
                    fill_value="extrapolate",
                )
                ds_lim_at_obs = f_lim(np.log10(rp_obs))
                ratio = ds_obs / ds_lim_at_obs

                inner_mask = (rp_obs < 1.0) & np.isfinite(ratio)
                all_mask = np.isfinite(ratio)

                mean_ratio_inner = (
                    np.nanmean(ratio[inner_mask]) * 100
                    if np.any(inner_mask)
                    else np.nan
                )
                mean_ratio_all = (
                    np.nanmean(ratio[all_mask]) * 100 if np.any(all_mask) else np.nan
                )

                print(
                    f"  Bin {bin_idx}: Inner (<1 Mpc) = {mean_ratio_inner:5.1f}% | "
                    f"Overall (0.1-20 Mpc) = {mean_ratio_all:5.1f}% of theoretical max"
                )
        print("-" * 70)


# %% Global Configuration

# List of (catalog_id, nbins, version_name) triples to compare.
# To compare 4-bin configurations, change "1bin" to "4bin".
# For available catalog IDs and nbins, refer to `RUN_REGISTRY` in `src/hsc_wl/config.py`.
# CONFIGS_TO_COMPARE = [
#     # ("logm_s16a_hectomap", "1bin", "Y3"),
#     ("camira", "1bin", "Y3"),
#     ("redm_s16a_hectomap", "1bin", "Y3"),
#     ("redm_pdr3_3band_fixed_s16a", "1bin", "Y3"),
#     # ("redm_pdr3_5band_free_s16a", "1bin", "Y3"),
#     # ("redm_pdr3_3band_free_s16a", "1bin", "Y3"),
#     ("cosine_s16a", "1bin", "Y3"),
#     ("camira_hecto_s16a", "1bin", "Y3"),
#     ("redm_r16_hecto_s16a", "1bin", "Y3"),
# ]

CONFIGS_TO_COMPARE = [
    ("camira", "1bin", "Y3"),
    ("redm_pdr3_5band_free", "1bin", "Y3"),
    ("camira_hectomap", "1bin", "Y3"),
    ("redm_r16_hectomap", "1bin", "Y3"),
    # ("amico", "1bin", "Y3"),
    ("cosine", "1bin", "Y3"),
]

# ---------------------------------------------------------------------------
# Theoretical Upper Limit Configuration (sigma = 0 ideal benchmark)
# ---------------------------------------------------------------------------
SHOW_THEORETICAL_LIMIT = True
THEORETICAL_LIMIT_SOURCE = (
    "simulation"  # "simulation" (MDPL2) or "colossus" (Analytic HMF+NFW)
)
THEORETICAL_LIMIT_TOP_N = 500  # For 1bin mode

MARKERS = ["o", "x", "s", "^", "D", "v", "P", "*", "H", "<", ">"]

# Paul Tol "bright"-based palette (grey/yellow dropped for white-bg visibility,
# orange & teal added for extra distinguishability). 8 distinct hues.
LABEL_PALETTE = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#EE7733",  # orange
    "#009988",  # teal
    "#332288",  # indigo
]

COLOR_MODE = "by_label"

OUTPUT_MAIN_FIG = project_root / "output/plots_for_agents/visual_multi_main.png"
OUTPUT_RATIO_FIG = project_root / "output/plots_for_agents/visual_multi_ratio.png"


# [Stage 1: Load comparison data and theoretical limit]
present_labels, loaded_tables, label_time_texts = load_comparison_data(
    CONFIGS_TO_COMPARE, project_root
)

limit_tables, limit_label = None, None
if SHOW_THEORETICAL_LIMIT and loaded_tables:
    nbins_mode = CONFIGS_TO_COMPARE[0][1]
    limit_tables, limit_label = load_theoretical_limit_data(
        nbins=nbins_mode,
        root_path=project_root,
        source=THEORETICAL_LIMIT_SOURCE,
        top_n=THEORETICAL_LIMIT_TOP_N,
    )


# [Stage 2: Plot main comparison]
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
        limit_tables=limit_tables,
        limit_label=limit_label,
    )
    fig.savefig(OUTPUT_MAIN_FIG, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %%[Stage 3: Plot ratio comparison]
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
        limit_tables=limit_tables,
        limit_label=limit_label,
    )
    if ratio_result is not None:
        ratio_fig, ratio_axes = ratio_result
        ratio_fig.savefig(OUTPUT_RATIO_FIG, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(ratio_fig)


# %% [Stage 4: Statistical comparison]
if loaded_tables:
    calculate_comparison_statistics(
        present_labels,
        loaded_tables,
        limit_tables=limit_tables,
        limit_label=limit_label,
    )
