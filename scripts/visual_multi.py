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

labels_to_compare = ["s16a", "s16a_forced", "pdr3"]
MULTIPLY_BY_RADIUS = True
USE_POINT_ERRORBAR_MODE = False
MARKERS = ["o", "s", "^", "D"]


fig, axes = plt.subplots(4, 1, figsize=(8.6, 13.2), sharex=True, sharey=False)
axes = np.atleast_1d(axes)
present_labels = []

for i, label_name in enumerate(labels_to_compare):
    print(f"Loading data for {label_name}...")
    current_dir = root_path / f"output/{label_name}/dsigma"
    if not current_dir.exists():
        print(f"Warning: {current_dir} does not exist. Skipping.")
        continue
    current_tables = load_result_tables(current_dir)
    present_labels.append(label_name)

    plot_radial_profile(
        current_tables,
        value_column="ds",
        title_label="Comparison",
        ax_list=axes,
        label_text=label_name,
        label_index=i,
        n_labels=len(labels_to_compare),
        marker=MARKERS[i % len(MARKERS)],
        multiply_by_radius=MULTIPLY_BY_RADIUS,
        point_errorbar_mode=USE_POINT_ERRORBAR_MODE,
    )

handles, legend_labels = axes[0].get_legend_handles_labels()
if handles:
    axes[0].legend(handles, legend_labels, loc="best", title="label")

fig.suptitle(f"Comparison of ΔΣ Profiles: {', '.join(present_labels)}", y=0.996)
fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.985])
plt.show()
# %%
