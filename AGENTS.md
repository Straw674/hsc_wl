# HSC Weak Lensing Analysis (hsc-wl)

This project focuses on Hyper Suprime-Cam (HSC) weak lensing analysis, specifically investigating the outer stellar mass of massive galaxies as a tracer of halo mass. It builds upon and integrates the `jianbing` project for reference.

## Key Technologies

- **Python (>=3.10)**: Managed by `uv`.
- **dsigma**: The core library used for all weak lensing calculations.
- **Astropy**: Used for FITS I/O and table management.
- **Data Stack**: Prefer `pandas.DataFrame` for tabular data operations.

## Environment & Dependencies

- **Dependency Management**: All dependencies should be managed via `uv`, with configurations in `pyproject.toml`.
- **Python Execution**: **Always** use the virtual environment's python (e.g., `.venv/bin/python`) for executing scripts. Do not use the system `python` or `python3` command.

## Code Quality & Linting

- **Linting & Formatting**: Rules are defined in `pyproject.toml` and enforced by `ruff`. Ensure code is clean and adheres to the project's formatting standards.

## Core Philosophy

- **Functional Programming**: Prefer functional programming (FP) style. Prefer small, pure, composable functions. Avoid side effects and shared mutable state. Use Object-Oriented Programming (OOP) only when strictly necessary (e.g., API clients or complex state machines).
- **Import Management (`initial.py`)**:
  - `initial.py` is used to consolidate common imports and boilerplate (e.g., logging, matplotlib settings).
  - **Convention**: In notebooks or scripts, use `from initial import *` to avoid repetitive import blocks. While this is not considered a general best practice, it is the established standard for this workspace. **Do not refactor this.**
  - If `initial.py` is imported, do not repeat the imports it already provides later in the file.

## Coding Style & Conventions

- **Language**: English must be used for all code, comments, variables, and documentation.
- **Data Processing**: Use `pandas.DataFrame` standardly for organizing tabular data, and save forms in the FITS (`.fits`) format.

## Project Structure

The project should minimally maintain the following standard directories:

- `data/`: Dataset files and resources. Local data directory for FITS files and catalogs.
- `sandbox/`: Playground for testing, experimenting, and temporary scripts. Experimental scripts and custom fitting logic.
- `scripts/`: Main execution and analysis scripts. Project-specific scripts for data preparation and execution.
- `output/`: Generated outputs, mask files, plots, etc. Directory where results (e.g., stacked lensing profiles) are saved.
- `src/`: Core Python modules and functions intended for reuse across multiple scripts. Shared source code for the project.
- `libs/`: Contains reference projects (e.g., `jianbing`, `merian`). **These are for reference only and are completely independent from the main project logic.** Do not modify or depend on them for core functionality.

## Execution Workflow

To reproduce the analysis or run the pipeline, follow this specific order:

1.  **Lens and Random Preparation**:
    ```bash
    .venv/bin/python scripts/prepare_lens_and_random.py
    ```
2.  **Lensing Profile Computation**:
    ```bash
    .venv/bin/python scripts/hsc_wl.py
    ```
3.  **Visualization**:
    Use `scripts/visual_single.py` or `scripts/visual_multi.py` to inspect results.
4.  **Custom Analysis**:
    Perform scatter fitting and other advanced analysis using `sandbox/fit_custom_scatter.py`.

## Interactive Scripts

Python scripts in `sandbox/` and `scripts/` are generally used as interactive scripts. Follow these specific rules:

- **Cell Division**: Divide the script into logical execution cells using `# %%` markers. **Do not** use `if __name__ == "__main__":` blocks, as they disrupt step-by-step interactive execution. This facilitates interactive exploration and debugging in environments like VS Code or Jupyter without needing `.ipynb` files.
- **Script Structure**: Scripts should be structured in the following top-to-bottom order:
  1. **Initialization**: Start with a code block that dynamically locates the project root using `pyproject.toml` as a marker, adds the root to `sys.path`, and imports everything from `initial.py` (e.g., `from initial import *`). The `initial.py` file provides shared conveniences for scripts. Do not alter this standard initialization pattern, and **never** import `initial.py` within the reusable modules in `src/`.
  2. **Configuration**: Define all adjustable parameters as UPPERCASE variables immediately following the imports. Modify these variables directly in the script for each run. Do not use command-line arguments or separate configuration files (like YAML) for these parameters.
  3. **Local Functions**: Define functions that are strictly used within this single script. If a function might be needed by other files, it must be moved to `src/`.
  4. **Execution**: The step-by-step logic, separated by `# %%`.
- **Interactive Visualization**: Calling `plt.savefig()` (or similar static save functions) is unnecessary. Just ensure the image is displayed (e.g., via `plt.show()`); figures can be inspected and manually saved directly within the interactive environment.
