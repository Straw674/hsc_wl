# %% [Initialization]
import sys
from pathlib import Path

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

# %%
from dataclasses import replace

from hsc_wl.config import RUN_REGISTRY
from hsc_wl.prepare import run_prepare_pipeline
from hsc_wl.wl_compute import run_pipeline
from initial import *  # noqa: F401,F403

# %% [Global Configuration]
# Either a single label or a list of labels to run sequentially.
# For all available labels, refer to `RUN_REGISTRY` in `src/hsc_wl/config.py`.
# Examples:
#   RUN_LABEL = "cosine_4bin"
#   RUN_LABEL = ["redm_s16a_hectomap_4bin", "camira_hectomap_4bin"]
#   RUN_LABEL = list(RUN_REGISTRY.keys())  # run all configurations
RUN_LABEL = ["cca1_1bin", "cca2_1bin"]
REDSHIFT_TYPE = "specz"  # "photoz" or "specz"

# %% Local Functions


def _as_list(label):
    """Normalize *label* to a list of labels."""
    return [label] if isinstance(label, str) else list(label)


# %% [Stage 1: Prepare lens and random catalogs]
labels = _as_list(RUN_LABEL)
configs = [
    replace(RUN_REGISTRY[label], redshift_type=REDSHIFT_TYPE) for label in labels
]

for cfg in configs:
    run_prepare_pipeline(cfg, root=project_root)

# %% [Stage 2: Run weak-lensing pipeline]
for cfg in configs:
    run_pipeline(cfg, root=project_root)
