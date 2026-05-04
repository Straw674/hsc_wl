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

label = "s16a"
version = "Y1"  # "Y1" or "Y3"

# Whether to plot random and raw\
PLOT_RANDOM = True
PLOT_RAW = False

if label.startswith("huang2022"):
    PLOT_RANDOM = False

# Main profile style.
MAIN_MULTIPLY_BY_RADIUS = True
MAIN_USE_LOG_Y = not MAIN_MULTIPLY_BY_RADIUS
MAIN_USE_SPLINE = False
MAIN_REFERENCE_LINE_Y = 0.0

# Random-subtraction profile style.
if PLOT_RANDOM:
    RANDOM_MULTIPLY_BY_RADIUS = False
    RANDOM_USE_LOG_Y = False
    RANDOM_USE_SPLINE = False
    RANDOM_REFERENCE_LINE_Y = 0.0


result_dir = root_path / f"output/{label}/{version}/dsigma"
tables = load_result_tables(result_dir)

expected_cols = ["rp", "ds", "ds_err"]
if PLOT_RANDOM:
    expected_cols.append("ds_r")
if PLOT_RAW:
    expected_cols.append("ds_raw")

missing_cols = [c for c in expected_cols if c not in tables[0].colnames]
if missing_cols:
    raise KeyError(f"Missing expected columns: {missing_cols}")

basic_fig = plot_radial_profile(
    tables,
    value_column="ds",
    title_label=f"{label} - corrected (main)",
    multiply_by_radius=MAIN_MULTIPLY_BY_RADIUS,
    use_spline=MAIN_USE_SPLINE,
    use_log_y=MAIN_USE_LOG_Y,
    reference_line_y=MAIN_REFERENCE_LINE_Y,
)

if PLOT_RAW:
    raw_fig = plot_radial_profile(
        tables,
        value_column="ds_raw",
        title_label=f"{label} - raw",
        multiply_by_radius=MAIN_MULTIPLY_BY_RADIUS,
        use_spline=MAIN_USE_SPLINE,
        use_log_y=MAIN_USE_LOG_Y,
        reference_line_y=MAIN_REFERENCE_LINE_Y,
    )
if PLOT_RANDOM:
    rds_fig = plot_radial_profile(
        tables,
        value_column="ds_r",
        title_label=f"{label} - random",
        multiply_by_radius=RANDOM_MULTIPLY_BY_RADIUS,
        use_spline=RANDOM_USE_SPLINE,
        use_log_y=RANDOM_USE_LOG_Y,
        reference_line_y=RANDOM_REFERENCE_LINE_Y,
    )
