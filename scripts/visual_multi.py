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

labels_to_compare = ["s16a", "pdr3", "s16a_forced"]
# Main comparison style.
MAIN_MULTIPLY_BY_RADIUS = True
MAIN_USE_LOG_Y = False
MAIN_USE_SPLINE = True
MAIN_REFERENCE_LINE_Y = 0.0


MARKERS = ["o", "x", "s", "^", "D"]


fig, axes = plt.subplots(4, 1, figsize=(8.6, 13.2), sharex=True, sharey=False)
axes = np.atleast_1d(axes)
present_labels = []
loaded_tables = []

for i, label_name in enumerate(labels_to_compare):
    print(f"Loading data for {label_name}...")
    current_dir = root_path / f"output/{label_name}/dsigma"
    if not current_dir.exists():
        print(f"Warning: {current_dir} does not exist. Skipping.")
        continue
    current_tables = load_result_tables(current_dir)
    present_labels.append(label_name)
    loaded_tables.append(current_tables)

    plot_radial_profile(
        current_tables,
        value_column="ds",
        title_label="Comparison",
        ax_list=axes,
        label_text=label_name,
        label_index=i,
        n_labels=len(labels_to_compare),
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
