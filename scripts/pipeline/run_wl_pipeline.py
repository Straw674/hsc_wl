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
from hsc_wl.config import RUN_REGISTRY
from hsc_wl.prepare import run_prepare_pipeline
from hsc_wl.wl_compute import run_pipeline
from initial import *  # noqa: F401,F403

# %% [Global Configuration]
# Either a single label or a list of labels to run sequentially.
# Examples:
#   RUN_LABEL = "pdr3_redm_hsc_no_mask"
#   RUN_LABEL = ["cosine", "cosine_4bin"]
RUN_LABEL = [
    "pdr3_redm_hsc_no_mask",
    "pdr3_redm_hsc_5bands_offdiag",
    "pdr3_redm_hsc_free_offdiag",
    "s16a_redm_hsc",
    "s16a_logm_50_100",
    "s16a_redm_hsc_topn",
    "s16a_logm_50_100_topn",
    "forced",
    "camira",
    "camira_4bin",
    "cosine",
    "cosine_4bin",
]

# %% Local Functions


def _as_list(label):
    """Normalize *label* to a list of labels."""
    return [label] if isinstance(label, str) else list(label)


# %% [Stage 1: Prepare lens and random catalogs]
labels = _as_list(RUN_LABEL)
for label in labels:
    run_prepare_pipeline(RUN_REGISTRY[label], root=project_root)

# %% [Stage 2: Run weak-lensing pipeline]
for label in labels:
    run_pipeline(RUN_REGISTRY[label], root=project_root)
