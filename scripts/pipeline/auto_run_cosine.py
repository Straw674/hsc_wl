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
from initial import *

# %% [Stage 1: Prepare lens & random for 'cosine' (top_n -> 1 bin)]
run_prepare_pipeline(RUN_REGISTRY["cosine"], root=project_root)

# %% [Stage 2: Prepare lens & random for 'cosine_4bin' (top_counts -> 4 bins)]
run_prepare_pipeline(RUN_REGISTRY["cosine_4bin"], root=project_root)

# %% [Stage 3: Compute lensing profile for 'cosine']
run_pipeline(RUN_REGISTRY["cosine"], root=project_root)

# %% [Stage 4: Compute lensing profile for 'cosine_4bin']
run_pipeline(RUN_REGISTRY["cosine_4bin"], root=project_root)
