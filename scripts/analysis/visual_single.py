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

from initial import *  # noqa: F401,F403

# %% Local Functions


def load_profile_tables(root_path, label, version):
    """Load dsigma result tables for a given (label, version) configuration."""
    catalog_id, nbins = label.rsplit("_", 1)
    result_dir = root_path / f"output/{catalog_id}/{nbins}/{version}/dsigma"
    return load_result_tables(result_dir), result_dir


def validate_columns(tables, plot_random, plot_raw):
    """Ensure expected columns are present in the loaded tables."""
    expected_cols = ["rp", "ds", "ds_err"]
    if plot_random:
        expected_cols.append("ds_r")
    if plot_raw:
        expected_cols.append("ds_raw")

    missing_cols = [c for c in expected_cols if c not in tables[0].colnames]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}")


def save_figure(fig, output_path):
    """Save a matplotlib figure to the given path, then close it."""
    if fig is None:
        return
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


# %% Global Configuration

# For all available labels, refer to `RUN_REGISTRY` in `src/hsc_wl/config.py`.
LABEL = "amico_4bin"
VERSION = "Y3"  # "Y1" or "Y3"

# Whether to plot random and raw profiles
PLOT_RANDOM = True
PLOT_RAW = False

if LABEL.startswith("huang2022"):
    PLOT_RANDOM = False

# Main profile style
MAIN_MULTIPLY_BY_RADIUS = True
MAIN_USE_LOG_Y = not MAIN_MULTIPLY_BY_RADIUS
MAIN_USE_SPLINE = False
MAIN_REFERENCE_LINE_Y = 0.0

# Random-subtraction profile style
RANDOM_MULTIPLY_BY_RADIUS = False
RANDOM_USE_LOG_Y = False
RANDOM_USE_SPLINE = False
RANDOM_REFERENCE_LINE_Y = 0.0

OUTPUT_MAIN_FIG = project_root / "output/plots_for_agents/visual_single_main.png"
OUTPUT_RAW_FIG = project_root / "output/plots_for_agents/visual_single_raw.png"
OUTPUT_RANDOM_FIG = project_root / "output/plots_for_agents/visual_single_random.png"


# [Stage 1: Load data and validate columns]
tables, result_dir = load_profile_tables(project_root, LABEL, VERSION)
validate_columns(tables, PLOT_RANDOM, PLOT_RAW)


# [Stage 2: Plot main profile]
basic_fig = plot_radial_profile(
    tables,
    value_column="ds",
    title_label=f"{LABEL} - corrected (main)",
    multiply_by_radius=MAIN_MULTIPLY_BY_RADIUS,
    use_spline=MAIN_USE_SPLINE,
    use_log_y=MAIN_USE_LOG_Y,
    reference_line_y=MAIN_REFERENCE_LINE_Y,
)
save_figure(basic_fig, OUTPUT_MAIN_FIG)
plt.show()


# [Stage 3: Plot raw profile (optional)]
if PLOT_RAW:
    raw_fig = plot_radial_profile(
        tables,
        value_column="ds_raw",
        title_label=f"{LABEL} - raw",
        multiply_by_radius=MAIN_MULTIPLY_BY_RADIUS,
        use_spline=MAIN_USE_SPLINE,
        use_log_y=MAIN_USE_LOG_Y,
        reference_line_y=MAIN_REFERENCE_LINE_Y,
    )
    save_figure(raw_fig, OUTPUT_RAW_FIG)
    plt.show()


#  [Stage 4: Plot random profile (optional)]
if PLOT_RANDOM:
    rds_fig = plot_radial_profile(
        tables,
        value_column="ds_r",
        title_label=f"{LABEL} - random",
        multiply_by_radius=RANDOM_MULTIPLY_BY_RADIUS,
        use_spline=RANDOM_USE_SPLINE,
        use_log_y=RANDOM_USE_LOG_Y,
        reference_line_y=RANDOM_REFERENCE_LINE_Y,
    )
    save_figure(rds_fig, OUTPUT_RANDOM_FIG)
    plt.show()
