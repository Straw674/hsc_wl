## Coding Style & Conventions

- **Data Processing**: Use `pandas.DataFrame` standardly for organizing tabular data. Save formats should prioritize Parquet (`.parquet`) for tabular data, with FITS (`.fits`) as a secondary option.
- **Data Import**: Always use the functions provided in the `src/data` module when importing data.
- **Caveat for Pandas (Precision Loss)**: When dealing with massive integer IDs (e.g., > $10^{15}$), never extract a row containing mixed types into a `pd.Series` (e.g., `row = df.iloc[i]`) before fetching the ID. Pandas will implicitly cast the entire row to `float64`, permanently truncating the precision of the large integer. Always extract the ID directly from the column first: `id = df["object_id"].iloc[i]`.
- **Raw Strings for LaTeX / Escape Sequences**: Python 3.12+ raises `SyntaxWarning` for unrecognized escape sequences (e.g., `\m`, `\d`, `\s`) inside standard strings. Always use raw strings (`r"..."`) when writing LaTeX or strings containing backslashes.
## Project Structure

The project should minimally maintain the following standard directories:

- `data/`: Dataset files and resources.
- `scripts/`: Main execution and analysis scripts.
- `output/`: Generated outputs, mask files, plots, etc.
- `src/`: Core Python modules and functions intended for reuse across multiple scripts.
- `scratch/`: Unmanaged directory for temporary scripts and exploratory testing files.

## Interactive Scripts

Python scripts in `scripts/` are generally used as interactive scripts. Follow these specific rules:

- **Cell Division**: Divide the script into logical execution cells using `# %%` markers. **Do not** use `if __name__ == "__main__":` blocks, as they disrupt step-by-step interactive execution.
- **Script Structure**: Scripts should be structured in the following top-to-bottom order:
  1. **Initialization**: Start with a code block that dynamically locates the project root using `pyproject.toml` as a marker, adds the root to `sys.path`, and imports everything from `initial.py` (e.g., `from initial import *`). The `initial.py` file provides shared conveniences for scripts. Do not alter this standard initialization pattern, and **never** import `initial.py` within the reusable modules in `src/`.
     - **Standard Code Style**: All scripts under `scripts/` must strictly use the unified initialization pattern defined in [initialization_template.py](file:////Users/xinq/dev/repos/hsc_wl/initialization_template.py):

       ```python
       import sys
       from pathlib import Path

       # Dynamically locate the project root using pyproject.toml as a marker
       project_root = Path(__file__).resolve().parent
       while project_root != project_root.parent and not (project_root / "pyproject.toml").exists():
           project_root = project_root.parent

       if not (project_root / "pyproject.toml").exists():
           raise RuntimeError("Could not find project root (containing pyproject.toml) in any parent directory.")

       if str(project_root) not in sys.path:
           sys.path.insert(0, str(project_root))

       from initial import *
       ```

     - `initial.py` is used to consolidate common imports and boilerplate (e.g., logging, matplotlib settings).
     - **Convention**: In Python scripts, use `from initial import *` to avoid repetitive import blocks. While this is not considered a general best practice, it is the established standard for this workspace. **Do not refactor this.**
     - If `initial.py` is imported, do not repeat the imports it already provides later in the file.

  2. **Local Functions (High Modularity)**: Scripts must be highly modular and functional. Define functions that are strictly used within this single script. If a function might be needed by other files, it must be moved to `src/`. Concrete operational logic and complex data processing steps MUST be encapsulated within these local functions rather than written directly in the stage cells.
  3. **Global Configuration**: Define adjustable parameters used globally across multiple cells as UPPERCASE variables. Place these in a dedicated, standalone cell (starting with `# %%`) immediately following the local functions and before any stage cells. Crucially, distinguish between parameters used by a single stage and global parameters: prioritize placing parameters at the top of their respective stage cell. Only define parameters in the Global Configuration cell if they are shared across multiple stages and allocating them to a single stage is inappropriate. Do not use command-line arguments or separate configuration files (like YAML).
  4. **Stages (Stage 1/2/3...)**: The step-by-step execution logic, split into sequential cells named in a `# %% [Stage 1: <Description>]`, `# %% [Stage 2: <Description>]`, etc., format. If an adjustable parameter is only used within a specific stage, define it as an UPPERCASE variable at the very top of that specific stage cell. The execution code within these stage cells should be minimal, ideally consisting _only_ of function calls (defined in local functions or `src/`) and cell-level configuration, rather than raw implementation logic.

- **Interactive Visualization**: For all data visualizations, scripts must **both** display the image (`plt.show()`) for interactive inspection in VS Code AND save the figure (`plt.savefig()`).
  - **Save Path**: The saved plots should be placed in `output/plots_for_agents/`. This serves as an unmanaged scratchpad for the AI to view the output.
  - **Headless AI Testing**: When the AI executes scripts for testing or verification, it MUST run the command in `Agg` mode (e.g., `MPLBACKEND=Agg .venv/bin/python ...`) to prevent UI windows from popping up on the user's machine. The AI can then use its tools to inspect the saved plot in `output/plots_for_agents/`.
