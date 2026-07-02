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
from initial import *

# %% [Global Configuration]
SOURCE = "pdr3_redm_hsc_no_mask"

# %% [Stage 1: Prepare lens and random catalogs]
cfg = RUN_REGISTRY[SOURCE]
run_prepare_pipeline(cfg, root=project_root)
