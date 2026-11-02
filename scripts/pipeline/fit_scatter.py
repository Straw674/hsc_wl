# %% [Initialization]
import sys
from pathlib import Path
from typing import Any

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

from astropy.table import Table  # noqa: E402
from colossus.cosmology import cosmology  # noqa: E402

from hsc_wl.config import RUN_REGISTRY, resolve_config, resolve_run_dir  # noqa: E402
from hsc_wl.profile_scatter import (  # noqa: E402
    build_profile_scatter_model,
    compute_stacked_profile_dsigma,
    fit_profile_scatter_map,
    load_dsigma_profile,
)
from initial import *  # noqa: F401, F403

# %% Local Functions


def load_label_metadata(label: str, root_path: Path):
    """Resolve sky area, redshift range, and per-bin lens counts for a run label.

    Parameters
    ----------
    label : str
        Run label from RUN_REGISTRY.
    root_path : Path
        Project root directory.

    Returns
    -------
    area_deg2 : float
    z_min, z_max : float
    bin_counts : list of int
    """
    cfg = RUN_REGISTRY[label]
    resolved_cfg = resolve_config(cfg, root_path)

    area_deg2 = (
        resolved_cfg.lens.area_deg2
        if resolved_cfg.lens.area_deg2 is not None
        else 439.4937
    )
    z_min, z_max = resolved_cfg.lens.redshift_range

    catalog_id, nbins = label.rsplit("_", 1)
    lenses_path = (
        resolve_run_dir(label, "prepare", root=root_path)
        / f"{catalog_id}_{nbins}_lenses.fits"
    )

    if lenses_path.exists():
        lenses_tbl = Table.read(lenses_path)
        if "bin_id" in lenses_tbl.colnames and nbins != "1bin":
            unique_bins = sorted(np.unique(lenses_tbl["bin_id"]))
            bin_counts = [int(np.sum(lenses_tbl["bin_id"] == b)) for b in unique_bins]
        else:
            bin_counts = [len(lenses_tbl)]
    else:
        # Fallback default lens counts if prepared catalog does not exist
        bin_counts = [1000] if nbins == "1bin" else [170, 625, 2100, 3700]

    return area_deg2, z_min, z_max, bin_counts


def fit_label_profile_scatter(
    label: str,
    version: str,
    root_path: Path,
    cosmology_name: str = "planck18",
    mass_def: str = "200m",
    f_mis_prior: tuple[float, float] | None = (0.25, 0.10),
    sigma_r_prior: tuple[float, float] | None = (100.0, 40.0),
) -> tuple[Table, dict]:
    """Fit all bins of a given run label with flexible profile scatter model.

    Parameters
    ----------
    label : str
        Unified run label.
    version : str
        HSC version (e.g. "Y3").
    root_path : Path
        Project root path.
    cosmology_name : str
        Colossus cosmology.
    mass_def : str
        Halo mass definition.
    f_mis_prior : tuple of float or None
        Gaussian prior (mean, std) on f_mis.
    sigma_r_prior : tuple of float or None
        Gaussian prior (mean, std) on sigma_R in kpc/h.

    Returns
    -------
    summary_table : Table
        Astropy table of per-bin fit results and components.
    model_state : dict
        Precomputed background model state.
    """
    catalog_id, nbins = label.rsplit("_", 1)
    dsigma_dir = resolve_run_dir(label, "dsigma", root=root_path, version=version)

    fits_files = sorted(
        list(dsigma_dir.glob("hsc_hsc_bin*.fits")),
        key=lambda p: int(p.stem.replace("hsc_hsc_bin", "")),
    )
    if not fits_files:
        raise FileNotFoundError(f"No dsigma FITS found in {dsigma_dir}")

    area_deg2, z_min, z_max, bin_counts = load_label_metadata(label, root_path)

    # Adjust bin_counts length to match available dsigma fits files if needed
    if len(bin_counts) > len(fits_files):
        bin_counts = bin_counts[: len(fits_files)]
    elif len(bin_counts) < len(fits_files):
        bin_counts = bin_counts + [bin_counts[-1]] * (len(fits_files) - len(bin_counts))

    cosmo = cosmology.setCosmology(cosmology_name)
    h = cosmo.h
    factor = 1e6 / h

    # Pre-read first bin to initialize radii and redshift
    rp_mpc_ref, _, _, _, z_lens_ref = load_dsigma_profile(str(fits_files[0]))
    rp_kpc_h_ref = rp_mpc_ref * 1000.0 * h

    logging.info(
        f"Building profile scatter model state for {label} (N_bins={len(fits_files)})..."
    )
    model_state = build_profile_scatter_model(
        rp_kpc_h=rp_kpc_h_ref,
        z_lens=z_lens_ref,
        bin_counts=bin_counts,
        area_sq_deg=area_deg2,
        z_min=z_min,
        z_max=z_max,
        cosmology_name=cosmology_name,
        mass_def=mass_def,
    )

    summary_rows = []
    print(f"\n{'=' * 75}")
    print(f"Fitting Profile Scatter for: {label} ({version})")
    print(f"{'=' * 75}")

    for b_idx, f_path in enumerate(fits_files):
        rp_mpc, ds_data, ds_err, cov_data, _ = load_dsigma_profile(str(f_path))
        ds_obs_colossus = ds_data * factor
        cov_colossus = cov_data * factor**2
        cov_inv_colossus = np.linalg.pinv(cov_colossus)

        fit_res = fit_profile_scatter_map(
            model_state=model_state,
            ds_obs_colossus=ds_obs_colossus,
            cov_inv_colossus=cov_inv_colossus,
            bin_index=b_idx,
            f_mis_prior=f_mis_prior,
            sigma_r_prior=sigma_r_prior,
        )

        # Compute best fit components for visualization and storage
        p_scatter = fit_res["scatter"]
        p_ac = fit_res["alpha_c"]
        p_ab = fit_res["alpha_b"]
        p_fmis = fit_res["f_mis"]
        p_sigmar = fit_res["sigma_R"]

        ds_tot_col, ds_1h_c_col, ds_1h_m_col, ds_2h_col = (
            compute_stacked_profile_dsigma(
                scatter=p_scatter,
                alpha_c=p_ac,
                alpha_b=p_ab,
                f_mis=p_fmis,
                sigma_R=p_sigmar,
                bin_index=b_idx,
                model_state=model_state,
                return_components=True,
            )
        )

        ds_tot_phys = ds_tot_col / factor
        ds_1h_c_phys = ds_1h_c_col / factor
        ds_1h_m_phys = ds_1h_m_col / factor
        ds_2h_phys = ds_2h_col / factor

        print(
            f"Bin {b_idx}: "
            f"Scatter={p_scatter:.3f} +/- {fit_res['scatter_err']:.3f}, "
            f"alpha_c={p_ac:.3f} +/- {fit_res['alpha_c_err']:.3f}, "
            f"alpha_b={p_ab:.3f} +/- {fit_res['alpha_b_err']:.3f}, "
            f"f_mis={p_fmis:.3f} +/- {fit_res['f_mis_err']:.3f}, "
            f"sigma_R={p_sigmar:.1f} +/- {fit_res['sigma_R_err']:.1f} kpc/h | "
            f"<logM>={fit_res['mean_logm']:.3f}, <c>={fit_res['mean_c']:.2f}, <b>={fit_res['mean_b']:.2f} | "
            f"Chi2/dof={fit_res['chi2_reduced']:.3f} (Chi2={fit_res['chi2']:.2f}, dof={fit_res['n_dof']})"
        )

        row = {
            "bin_id": b_idx + 1,
            "dsigma": ds_data,
            "dsig_err_jk": ds_err,
            "dsig_cov_jk": cov_data,
            "dsigma_mod": ds_tot_phys,
            "dsigma_1h_cen": ds_1h_c_phys,
            "dsigma_1h_mis": ds_1h_m_phys,
            "dsigma_2h": ds_2h_phys,
            **fit_res,
        }
        summary_rows.append(row)

    summary_tbl = Table(summary_rows)
    summary_tbl["r_mpc"] = [rp_mpc_ref] * len(summary_tbl)
    summary_tbl.meta["label"] = label
    summary_tbl.meta["version"] = version

    # Save summary FITS table
    fit_dir = resolve_run_dir(label, "fit", root=root_path, version=version)
    fit_dir.mkdir(parents=True, exist_ok=True)
    out_fits = fit_dir / f"{catalog_id}_{nbins}_profile_fit.fits"
    summary_tbl.write(out_fits, format="fits", overwrite=True)
    logging.info(f"Saved summary results to: {out_fits}")

    return summary_tbl, model_state


def _format_fit_val_err(
    val: Any, err: Any, val_fmt: str = ".2f", err_fmt: str = ".2f"
) -> str:
    """Safely format value +/- error, handling masked constants or NaNs gracefully."""
    if val is None or np.ma.is_masked(val) or not np.isfinite(val):
        v_str = r"\mathrm{N/A}"
    else:
        v_str = f"{float(val):{val_fmt}}"

    if err is None or np.ma.is_masked(err) or not np.isfinite(err):
        return v_str
    return rf"{v_str} \pm {float(err):{err_fmt}}"


def plot_profile_fit_panel(
    ax: plt.Axes,
    rp_mpc: np.ndarray,
    row: dict,
    panel_title: str,
    is_multi_bin: bool = False,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> None:
    """Plot profile fit decomposition for a single bin onto the given Axes."""
    ds_data = row["dsigma"]
    ds_err = row["dsig_err_jk"]
    ds_mod = row["dsigma_mod"]
    ds_1h_cen = row["dsigma_1h_cen"]
    ds_1h_mis = row["dsigma_1h_mis"]
    ds_2h = row["dsigma_2h"]

    ax.errorbar(
        rp_mpc,
        ds_data,
        yerr=ds_err,
        fmt="o",
        color="black",
        label="Data",
        capsize=2.5,
        markersize=4.5,
        elinewidth=1.0,
        zorder=3,
    )
    ax.plot(
        rp_mpc,
        ds_mod,
        "-",
        color="red",
        linewidth=2.0,
        label="Total Model",
        zorder=4,
    )
    ax.plot(
        rp_mpc,
        ds_1h_cen,
        "--",
        color="blue",
        linewidth=1.4,
        label="1-halo (Centered)",
        zorder=2,
    )
    if row["f_mis"] > 0.0:
        ax.plot(
            rp_mpc,
            ds_1h_mis,
            "--",
            color="orange",
            linewidth=1.4,
            label="1-halo (Miscentered)",
            zorder=2,
        )
    ax.plot(
        rp_mpc,
        ds_2h,
        ":",
        color="green",
        linewidth=1.5,
        label="2-halo Term",
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.maximum(1e-1, np.nanmin(ds_data) * 0.4))
    if show_xlabel:
        ax.set_xlabel(r"$R_p\ [\mathrm{Mpc}]$")
    if show_ylabel:
        ax.set_ylabel(r"$\Delta\Sigma\ [\mathrm{M_\odot / pc^2}]$")

    ax.grid(True, which="both", linestyle=":", alpha=0.35)

    fontsize = 8.5 if is_multi_bin else 9.5
    summary_text = "\n".join(
        [
            rf"$\sigma = {_format_fit_val_err(row['scatter'], row['scatter_err'], '.3f', '.3f')}$",
            rf"$\alpha_c = {_format_fit_val_err(row['alpha_c'], row['alpha_c_err'], '.2f', '.2f')}$ ($\langle c \rangle = {row['mean_c']:.1f}$)",
            rf"$\alpha_b = {_format_fit_val_err(row['alpha_b'], row['alpha_b_err'], '.2f', '.2f')}$ ($\langle b \rangle = {row['mean_b']:.2f}$)",
            rf"$f_{{\rm mis}} = {_format_fit_val_err(row['f_mis'], row['f_mis_err'], '.2f', '.2f')}$",
            rf"$\sigma_R = {_format_fit_val_err(row['sigma_R'], row['sigma_R_err'], '.1f', '.1f')}\ \mathrm{{kpc}}/h$",
            rf"$\chi^2/\nu = {row['chi2_reduced']:.3f}\ (\chi^2={row['chi2']:.1f}, \nu={row['n_dof']})$",
            rf"$\langle \log M_{{200m}} \rangle = {row['mean_logm']:.3f}$",
        ]
    )
    ax.text(
        0.05,
        0.05,
        summary_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.88,
            edgecolor="0.75",
        ),
    )

    ax.set_title(
        panel_title,
        fontweight="normal",
        fontsize=10.5 if is_multi_bin else 11.5,
    )
    ax.legend(loc="upper right", fontsize=fontsize)


def plot_profile_fit_decomposition(
    summary: Table,
    label: str,
    out_plot_path: Path,
) -> None:
    """Plot profile fit decomposition for all bins in a single figure.

    For 1-bin data, produces a single-panel plot. For 4-bin (or multi-bin) data,
    arranges bins into a multi-panel grid (e.g. 2x2 for 4 bins).
    """
    n_bins = len(summary)
    rp_mpc = (
        summary["r_mpc"][0] if "r_mpc" in summary.colnames else summary.meta["r_mpc"]
    )

    if n_bins == 1:
        fig, ax = plt.subplots(figsize=(7, 6))
        row = summary[0]
        bin_num = row["bin_id"]
        panel_title = f"{label} - Bin {bin_num} Profile Fit Decomposition"
        plot_profile_fit_panel(
            ax=ax,
            rp_mpc=rp_mpc,
            row=row,
            panel_title=panel_title,
            is_multi_bin=False,
            show_ylabel=True,
            show_xlabel=True,
        )
    else:
        ncols = 2
        nrows = int(np.ceil(n_bins / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(12, 5.0 * nrows),
            sharex=True,
            sharey=False,
        )
        axes_flat = axes.flatten()

        for b_idx in range(n_bins):
            ax = axes_flat[b_idx]
            row = summary[b_idx]
            bin_num = row["bin_id"]
            row_idx = b_idx // ncols
            col_idx = b_idx % ncols

            show_ylabel = col_idx == 0
            show_xlabel = (row_idx == nrows - 1) or (b_idx + ncols >= n_bins)

            panel_title = f"{label} - Bin {bin_num}"
            plot_profile_fit_panel(
                ax=ax,
                rp_mpc=rp_mpc,
                row=row,
                panel_title=panel_title,
                is_multi_bin=True,
                show_ylabel=show_ylabel,
                show_xlabel=show_xlabel,
            )

        for b_idx in range(n_bins, len(axes_flat)):
            axes_flat[b_idx].set_visible(False)

    plt.tight_layout()
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved plot to: {out_plot_path}")
    plt.show()
    plt.close(fig)


# %% Global Configuration

# List of labels to fit. Supports both 1-bin and 4-bin configurations.
# Available labels in RUN_REGISTRY include:
# "rz_diff_lum_1bin", "rz_diff_1bin", "rz_diff_4bin", "rz_diff_lum_4bin",
# "camira_1bin", "camira_4bin", "redm_s16a_1bin", etc.
LABELS = [
    "redm_r16_1bin",
    "camira_1bin",
    "rz_diff_1bin",
    "rz_diff_preset_1bin",
    "rz_diff_single_box_1bin",
    "rz_diff_two_box_red_match_recovery_1bin",
    "wh24_1bin",
]

VERSION = "Y3"  # "Y1" or "Y3"
COSMOLOGY_NAME = "planck18"
MASS_DEF = "200m"

# Miscentering priors: (mean, std) based on optical cluster centering literature
# Keeps miscentering parameters bounded to physical ranges while allowing finder-specific variations
F_MIS_PRIOR = (0.25, 0.10)
SIGMA_R_PRIOR = (100.0, 40.0)  # kpc/h

OUTPUT_PLOT_DIR = project_root / "output/plots_for_agents"

# %% [Stage 1: Process and fit all configured labels]
all_summaries = {}
for lbl in LABELS:
    try:
        tbl_summary, _ = fit_label_profile_scatter(
            label=lbl,
            version=VERSION,
            root_path=project_root,
            cosmology_name=COSMOLOGY_NAME,
            mass_def=MASS_DEF,
            f_mis_prior=F_MIS_PRIOR,
            sigma_r_prior=SIGMA_R_PRIOR,
        )
        all_summaries[lbl] = tbl_summary
    except Exception as e:
        logging.error(f"Failed fitting for label {lbl}: {e}")
        import traceback

        traceback.print_exc()

# %% [Stage 2: Visualize fit decompositions]
for lbl, summary in all_summaries.items():
    out_plot = OUTPUT_PLOT_DIR / f"fit_profile_{lbl}_{VERSION}.png"
    plot_profile_fit_decomposition(
        summary=summary,
        label=lbl,
        out_plot_path=out_plot,
    )
