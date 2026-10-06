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

from hsc_wl.reference_catalogs import prepare_reference_catalogs
from initial import *

# %% [Stage 1: Prepare Eight External References]

REDSHIFT_RANGE = (0.19, 0.52)
reference_summary = prepare_reference_catalogs(project_root, REDSHIFT_RANGE)
print(reference_summary.to_string(index=False))
