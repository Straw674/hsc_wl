# %%
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        print("Error: pyproject.toml not found in parent directories.")
        # Handle the error appropriately, maybe raise an exception or exit
        # For now, just break to avoid infinite loop if marker is truly missing
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        print(f"Project root found: {root_path}")  # Confirm the path found
        break
    else:
        current_dir = current_dir.parent

if root_path:
    root_path_str = str(root_path)

    if root_path_str not in sys.path:
        sys.path.append(root_path_str)

    try:
        from initial import *
    except ModuleNotFoundError as e:
        print(f"Error importing 'initial': {e}")

else:
    print("Could not proceed without finding the project root.")

# %%

# List of (label, version) pairs to compare
# e.g., [("s16a_redm_hsc", "Y1"), ("s16a_redm_hsc", "Y3"), ("pdr3_redm_hsc", "Y3")]
# configs_to_compare = [("huang2022_redm_hsc", "Y1"), ("s16a_redm_hsc", "Y1")]
# configs_to_compare = [("huang2022_redm_hsc", "Y1"), ("s16a_redm_hsc", "Y1"), ("s16a_redm_hsc", "Y3")]
configs_to_compare = [("huang2022_logm_50_100", "Y1"), ("s16a_logm_50_100", "Y1")]
# configs_to_compare = [ ("huang2022_logm_50_100", "Y1"), ("s16a_logm_50_100", "Y1"), ("s16a_logm_50_100", "Y3") ]


# Main comparison style.
MAIN_MULTIPLY_BY_RADIUS = True
MAIN_USE_LOG_Y = not MAIN_MULTIPLY_BY_RADIUS
MAIN_USE_SPLINE = False
MAIN_REFERENCE_LINE_Y = 0.0


MARKERS = ["o", "x", "s", "^", "D"]


fig, axes = plt.subplots(4, 1, figsize=(8.6, 13.2), sharex=True, sharey=False)
axes = np.atleast_1d(axes)
present_labels = []
loaded_tables = []

for i, (label_name, version_name) in enumerate(configs_to_compare):
    display_name = f"{label_name} ({version_name})"
    print(f"Loading data for {display_name}...")
    current_dir = root_path / f"output/{label_name}/{version_name}/dsigma"
    if not current_dir.exists():
        print(f"Warning: {current_dir} does not exist. Skipping.")
        continue
    current_tables = load_result_tables(current_dir)
    present_labels.append(display_name)
    loaded_tables.append(current_tables)

    plot_radial_profile(
        current_tables,
        value_column="ds",
        title_label="Comparison",
        ax_list=axes,
        label_text=display_name,
        label_index=i,
        n_labels=len(configs_to_compare),
        marker=MARKERS[i % len(MARKERS)],
        multiply_by_radius=MAIN_MULTIPLY_BY_RADIUS,
        use_spline=MAIN_USE_SPLINE,
        use_log_y=MAIN_USE_LOG_Y,
        reference_line_y=MAIN_REFERENCE_LINE_Y,
    )

handles, legend_labels = axes[0].get_legend_handles_labels()
if handles:
    axes[0].legend(handles, legend_labels, loc="best", title="label")

fig.suptitle(f"Comparison of ΔΣ Profiles: {', '.join(present_labels)}", y=0.996)
fig.tight_layout()
plt.show()


# %%

# Ratio comparison style.
RATIO_MULTIPLY_BY_RADIUS = False
RATIO_USE_LOG_Y = False
RATIO_USE_SPLINE = False
RATIO_REFERENCE_LINE_Y = 1.0


def _build_ratio_table(current_table, reference_table):
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


if loaded_tables:
    reference_tables = loaded_tables[0]
    ratio_fig, ratio_axes = plt.subplots(
        4, 1, figsize=(8.6, 13.2), sharex=True, sharey=False
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

        plot_radial_profile(
            ratio_tables,
            value_column="ds",
            title_label=f"Ratio to {present_labels[0]}",
            ax_list=ratio_axes,
            label_text=label_name,
            label_index=ratio_index,
            n_labels=len(present_labels) - 1,
            marker=MARKERS[ratio_index % len(MARKERS)],
            multiply_by_radius=RATIO_MULTIPLY_BY_RADIUS,
            use_spline=RATIO_USE_SPLINE,
            use_log_y=RATIO_USE_LOG_Y,
            reference_line_y=RATIO_REFERENCE_LINE_Y,
            y_label=r"$\Delta\Sigma / \Delta\Sigma_{\mathrm{ref}}$",
            title_suffix="Ratio Profiles",
        )

    handles, legend_labels = ratio_axes[0].get_legend_handles_labels()
    if handles:
        ratio_axes[0].legend(handles, legend_labels, loc="best", title="label")

    ratio_fig.suptitle(
        f"ΔΣ Ratio relative to {present_labels[0]}: {', '.join(present_labels[1:])}",
        y=0.996,
    )
    ratio_fig.tight_layout()
    plt.show()


# %%
def calculate_comparison_statistics(present_labels, loaded_tables):
    if len(loaded_tables) <= 1:
        return

    print("\n" + "=" * 70)
    print(f"{'Statistical Comparison (Chi-Square)':^70}")
    print("=" * 70)

    # Calculate chi2 for all unique pairs (n*(n-1)/2 combinations)
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
                        f"  Bin {bin_idx}: chi2 = {chi2_bin:7.2f} | ndof = {ndof_bin:2d} | red_chi2 = {red_chi2:6.2f}"
                    )
                else:
                    print(f"  Bin {bin_idx}: No valid data points.")

            if ndof_total > 0:
                print("-" * 70)
                print(
                    f"  OVERALL: chi2 = {chi2_total:7.2f} | ndof = {ndof_total:3d} | red_chi2 = {chi2_total / ndof_total:6.2f}"
                )
            print("-" * 70)


if loaded_tables:
    calculate_comparison_statistics(present_labels, loaded_tables)
