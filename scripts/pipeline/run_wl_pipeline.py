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
#   RUN_LABEL = "cosine_4bin"
#   RUN_LABEL = ["redm_s16a_hectomap_4bin", "camira_hectomap_4bin"]
#
# Label convention: {catalog_id}_{nbins}
#   catalog_id encodes the lens catalog and its sky footprint.
#   nbins is either "1bin" (single top-N sample) or "4bin" (four richness bins).
#
# All available labels:
#
#   redMapper PDR3 (natural HectoMap footprint / s16a-constrained footprint):
#     "redm_pdr3_3band_fixed_4bin", "redm_pdr3_3band_fixed_1bin"
#     "redm_pdr3_3band_fixed_s16a_4bin", "redm_pdr3_3band_fixed_s16a_1bin"
#     "redm_pdr3_5band_free_4bin",  "redm_pdr3_5band_free_1bin"
#     "redm_pdr3_5band_free_s16a_4bin",  "redm_pdr3_5band_free_s16a_1bin"
#     "redm_pdr3_3band_free_4bin",  "redm_pdr3_3band_free_1bin"
#     "redm_pdr3_3band_free_s16a_4bin",  "redm_pdr3_3band_free_s16a_1bin"
#
#   redMapper S16a:
#     "redm_s16a_4bin",             "redm_s16a_1bin"
#     "redm_s16a_hectomap_4bin",    "redm_s16a_hectomap_1bin"
#
#   logM S16a massive galaxies:
#     "logm_s16a_4bin",             "logm_s16a_1bin"
#     "logm_s16a_hectomap_4bin",    "logm_s16a_hectomap_1bin"
#
#   Forced-richness S16a:
#     "forced_4bin",                "forced_1bin"
#     "forced_hectomap_4bin",       "forced_hectomap_1bin"
#
#   CAMIRA S23b wide (full / hectomap / hectomap + s16a footprint):
#     "camira_4bin",                "camira_1bin"
#     "camira_hectomap_4bin",       "camira_hectomap_1bin"
#     "camira_hecto_s16a_4bin",     "camira_hecto_s16a_1bin"
#
#   redMapper SDSS R16 (full / hectomap / s16a-hectomap footprint):
#     "redm_r16_4bin",              "redm_r16_1bin"
#     "redm_r16_hectomap_4bin",     "redm_r16_hectomap_1bin"
#     "redm_r16_hecto_s16a_4bin",   "redm_r16_hecto_s16a_1bin"
#
#   COSINE cluster finder (natural HectoMap / s16a-constrained footprint):
#     "cosine_4bin",                "cosine_1bin"
#     "cosine_s16a_4bin",           "cosine_s16a_1bin"
#
RUN_LABEL = list(RUN_REGISTRY.keys())  # run all 34 configurations

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
