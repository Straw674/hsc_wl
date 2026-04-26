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

label = "pdr3"
RANDOM_SUBTRACTION = False

# Select which plot function to use for main visualization
MULTIPLY_BY_RADIUS = True

# False: keep current spline + shaded error band.
# True: disable spline/shaded band and draw only points with vertical error bars.
USE_POINT_ERRORBAR_MODE = False


result_dir = root_path / f"output/{label}/dsigma"
tables = load_result_tables(result_dir)

expected_cols = [
    "rp_min",
    "rp_max",
    "n_pairs",
    "rp",
    "ds_raw",
    "ds",
    "z_l",
    "z_s",
    "1+m",
    "2R",
    "1+m_sel",
    "ds_err",
]
if RANDOM_SUBTRACTION:
    expected_cols.append("ds_r")
print("Columns in result table:")
print(tables[0].colnames)
missing_cols = [c for c in expected_cols if c not in tables[0].colnames]
if missing_cols:
    raise KeyError(f"Missing expected columns: {missing_cols}")

column_meaning = {
    "rp_min": "radial bin lower edge",
    "rp_max": "radial bin upper edge",
    "n_pairs": "number of lens-source pairs",
    "rp": "radial bin center",
    "ds_raw": "raw DeltaSigma before all enabled corrections",
    "ds": "corrected DeltaSigma (main WL signal)",
    "z_l": "weighted mean lens redshift",
    "z_s": "weighted mean source redshift",
    "1+m": "multiplicative shear-bias factor",
    "2R": "shear responsivity factor",
    "1+m_sel": "selection-bias correction factor",
    "ds_r": "random-point DeltaSigma term",
    "ds_err": "jackknife error on ds",
}

print("\nColumn meanings:")
for k in expected_cols:
    print(f"- {k:8s}: {column_meaning[k]}")

basic_fig = plot_radial_profile(
    tables,
    value_column="ds",
    title_label=f"{label} - corrected (main)",
    show_pair_sizes=False,
    multiply_by_radius=MULTIPLY_BY_RADIUS,
    point_errorbar_mode=USE_POINT_ERRORBAR_MODE,
)
raw_fig = plot_radial_profile(
    tables,
    value_column="ds_raw",
    title_label=f"{label} - raw",
    multiply_by_radius=MULTIPLY_BY_RADIUS,
    point_errorbar_mode=USE_POINT_ERRORBAR_MODE,
)
if RANDOM_SUBTRACTION:
    rds_fig = plot_radial_profile(
        tables,
        value_column="ds_r",
        title_label=f"{label} - random",
        multiply_by_radius=MULTIPLY_BY_RADIUS,
        point_errorbar_mode=USE_POINT_ERRORBAR_MODE,
    )
# pair_fig = plot_pair_counts_vs_rp(tables)
# factor_fig = plot_correction_factors_radial(tables)
