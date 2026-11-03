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
from colossus.halo import mass_so  # noqa: E402

from hsc_wl.config import RUN_REGISTRY, resolve_config, resolve_run_dir  # noqa: E402
from hsc_wl.profile_scatter import (  # noqa: E402
    build_profile_scatter_model,
    compute_stacked_profile_dsigma,
    fit_profile_scatter_map,
    load_dsigma_profile,
)
from hsc_wl.visual import get_complete_bin_indices  # noqa: E402
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
            complete_indices = get_complete_bin_indices(lenses_path.parent)
            if complete_indices is not None:
                unique_bins = [b for b in unique_bins if b in complete_indices]
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
    f_mis: float = 0.25,
    sigma_r: float = 100.0,
    max_valid_bins: int | None = None,
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
    f_mis : float, default=0.25
        Fixed miscentered cluster fraction.
    sigma_r : float, default=100.0
        Fixed miscentering scale sigma_R in kpc/h.
    max_valid_bins : int or None, optional
        Maximum number of valid bins to process. If specified, truncates bins
        beyond this limit (e.g. for catalogs with completeness cutoffs).

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

    complete_indices = get_complete_bin_indices(dsigma_dir)
    if complete_indices is not None:
        fits_files = [
            f
            for f in fits_files
            if int(f.stem.replace("hsc_hsc_bin", "")) in complete_indices
        ]

    area_deg2, z_min, z_max, bin_counts = load_label_metadata(label, root_path)

    # Restrict to valid complete bins if configured
    if max_valid_bins is not None:
        fits_files = fits_files[:max_valid_bins]
        bin_counts = bin_counts[:max_valid_bins]

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
            f_mis=f_mis,
            sigma_r=sigma_r,
        )

        # Compute best fit components for visualization and storage
        p_scatter = fit_res["scatter"]
        p_ac = fit_res["alpha_c"]
        p_ab = fit_res["alpha_b"]
        p_fmis = fit_res["f_mis"]
        p_sigmar = fit_res["sigma_R"]

        ds_tot_col, ds_1h_col, ds_1h_cen_col, ds_2h_col = (
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
        ds_1h_phys = ds_1h_col / factor
        ds_1h_cen_phys = ds_1h_cen_col / factor
        ds_2h_phys = ds_2h_col / factor

        print(
            f"Bin {b_idx}: "
            f"Scatter={p_scatter:.3f} +/- {fit_res['scatter_err']:.3f}, "
            f"alpha_c={p_ac:.3f} +/- {fit_res['alpha_c_err']:.3f}, "
            f"alpha_b={p_ab:.3f} +/- {fit_res['alpha_b_err']:.3f} | "
            f"<logM>={fit_res['mean_logm']:.3f} +/- {fit_res['mean_logm_err']:.3f}, "
            f"<c>={fit_res['mean_c']:.2f} +/- {fit_res['mean_c_err']:.2f}, "
            f"<b>={fit_res['mean_b']:.2f} +/- {fit_res['mean_b_err']:.2f} | "
            f"Chi2/dof={fit_res['chi2_reduced']:.3f} (Chi2={fit_res['chi2']:.2f}, dof={fit_res['n_dof']})"
        )

        r200m_val = compute_r200m_physical(
            mean_logm=fit_res["mean_logm"],
            z_lens=z_lens_ref,
            cosmology_name=cosmology_name,
            mass_def=mass_def,
        )
        row = {
            "bin_id": b_idx + 1,
            "dsigma": ds_data,
            "dsig_err_jk": ds_err,
            "dsig_cov_jk": cov_data,
            "dsigma_mod": ds_tot_phys,
            "dsigma_1h": ds_1h_phys,
            "dsigma_1h_cen": ds_1h_cen_phys,
            "dsigma_2h": ds_2h_phys,
            "r200m": r200m_val,
            "z_lens": z_lens_ref,
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


def compute_r200m_physical(
    mean_logm: float,
    z_lens: float = 0.35,
    cosmology_name: str = "planck18",
    mass_def: str = "200m",
) -> float:
    """Compute physical R200m in Mpc from halo mass and lens redshift."""
    try:
        cosmo = cosmology.setCosmology(cosmology_name)
        h = cosmo.h
        r_kpc_h = mass_so.M_to_R(10.0**mean_logm, z_lens, mass_def)
        return float((r_kpc_h / (1.0 + z_lens)) / 1000.0 / h)
    except Exception as e:
        logging.warning(f"Failed to compute R200m: {e}")
        return float("nan")


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


def plot_bin_column(
    ax_log: plt.Axes,
    ax_lin: plt.Axes,
    rp_mpc: np.ndarray,
    row: dict,
    bin_title: str,
    is_multi_bin: bool = False,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> None:
    """Plot profile fit decomposition for a single bin in log (top) and linear (bottom) subplots."""
    ds_data = row["dsigma"]
    ds_err = row["dsig_err_jk"]
    ds_mod = row["dsigma_mod"]
    ds_1h = row["dsigma_1h"]
    ds_1h_cen = row["dsigma_1h_cen"]
    ds_2h = row["dsigma_2h"]

    has_r200m = (
        ("r200m" in row.colnames) if hasattr(row, "colnames") else ("r200m" in row)
    )
    if has_r200m:
        r200m = float(row["r200m"])
    else:
        has_z = (
            ("z_lens" in row.colnames)
            if hasattr(row, "colnames")
            else ("z_lens" in row)
        )
        z_val = float(row["z_lens"]) if has_z else 0.35
        r200m = compute_r200m_physical(float(row["mean_logm"]), z_lens=z_val)

    # ---------------- Top panel: Log scale ----------------
    ax_log.errorbar(
        rp_mpc,
        ds_data,
        yerr=ds_err,
        fmt="o",
        color="black",
        label="Data",
        capsize=2.0,
        markersize=3.8,
        elinewidth=0.9,
        zorder=3,
    )
    ax_log.plot(
        rp_mpc,
        ds_mod,
        "-",
        color="red",
        linewidth=1.8,
        label="Total Model",
        zorder=4,
    )
    ax_log.plot(
        rp_mpc,
        ds_1h,
        "--",
        color="blue",
        linewidth=1.4,
        label="1-halo",
        zorder=2,
    )
    if row["f_mis"] > 0.0:
        ax_log.plot(
            rp_mpc,
            ds_1h_cen,
            "-.",
            color="deepskyblue",
            linewidth=1.3,
            label="1-halo (Centered)",
            zorder=2,
        )
    ax_log.plot(
        rp_mpc,
        ds_2h,
        ":",
        color="green",
        linewidth=1.4,
        label="2-halo Term",
        zorder=2,
    )

    if np.isfinite(r200m):
        ax_log.axvline(
            r200m,
            color="slategray",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            zorder=1,
        )
        ax_log.text(
            r200m * 1.05,
            0.90,
            r"$R_{\rm 200m}$",
            transform=ax_log.get_xaxis_transform(),
            color="slategray",
            fontsize=6.8 if is_multi_bin else 7.8,
            va="top",
        )

    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_ylim(bottom=np.maximum(1e-1, np.nanmin(ds_data) * 0.4))
    ax_log.grid(True, which="both", linestyle=":", alpha=0.35)
    if show_ylabel:
        ax_log.set_ylabel(r"$\Delta\Sigma\ [\mathrm{M_\odot / pc^2}]$")
    ax_log.set_title(
        bin_title,
        fontweight="normal",
        fontsize=10.0 if is_multi_bin else 11.0,
    )

    # Parameter summary card (compact font size)
    fontsize_card = 6.8 if is_multi_bin else 8.0
    summary_items = [
        rf"$\sigma = {_format_fit_val_err(row['scatter'], row['scatter_err'], '.3f', '.3f')}$",
        rf"$\alpha_c = {_format_fit_val_err(row['alpha_c'], row['alpha_c_err'], '.2f', '.2f')}$ ($\langle c \rangle = {row['mean_c']:.1f}$)",
        rf"$\alpha_b = {_format_fit_val_err(row['alpha_b'], row['alpha_b_err'], '.2f', '.2f')}$ ($\langle b \rangle = {row['mean_b']:.2f}$)",
        rf"$\chi^2/\nu = {row['chi2_reduced']:.3f}\ (\chi^2={row['chi2']:.1f})$",
        rf"$\langle \log M_{{200m}} \rangle = {row['mean_logm']:.3f}$",
    ]
    if np.isfinite(r200m):
        summary_items.append(rf"$R_{{\rm 200m}} = {r200m:.2f}\ \mathrm{{Mpc}}$")
    summary_text = "\n".join(summary_items)
    ax_log.text(
        0.04,
        0.04,
        summary_text,
        transform=ax_log.transAxes,
        fontsize=fontsize_card,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            alpha=0.88,
            edgecolor="0.75",
        ),
    )

    if show_legend:
        ax_log.legend(
            loc="upper right",
            fontsize=6.8 if is_multi_bin else 8.0,
            framealpha=0.85,
        )

    # ---------------- Bottom panel: Linear scale of R * DeltaSigma ----------------
    rds_data = rp_mpc * ds_data
    rds_err = rp_mpc * ds_err
    rds_mod = rp_mpc * ds_mod
    rds_1h = rp_mpc * ds_1h
    rds_1h_cen = rp_mpc * ds_1h_cen
    rds_2h = rp_mpc * ds_2h

    ax_lin.errorbar(
        rp_mpc,
        rds_data,
        yerr=rds_err,
        fmt="o",
        color="black",
        capsize=2.0,
        markersize=3.8,
        elinewidth=0.9,
        zorder=3,
    )
    ax_lin.plot(rp_mpc, rds_mod, "-", color="red", linewidth=1.8, zorder=4)
    ax_lin.plot(
        rp_mpc,
        rds_1h,
        "--",
        color="blue",
        linewidth=1.4,
        zorder=2,
    )
    if row["f_mis"] > 0.0:
        ax_lin.plot(
            rp_mpc,
            rds_1h_cen,
            "-.",
            color="deepskyblue",
            linewidth=1.3,
            zorder=2,
        )
    ax_lin.plot(
        rp_mpc,
        rds_2h,
        ":",
        color="green",
        linewidth=1.4,
        zorder=2,
    )

    ax_lin.axhline(
        0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1
    )
    if np.isfinite(r200m):
        ax_lin.axvline(
            r200m,
            color="slategray",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            zorder=1,
        )
        ax_lin.text(
            r200m * 1.05,
            0.90,
            r"$R_{\rm 200m}$",
            transform=ax_lin.get_xaxis_transform(),
            color="slategray",
            fontsize=6.8 if is_multi_bin else 7.8,
            va="top",
        )

    ax_lin.set_xscale("log")
    ax_lin.set_yscale("linear")
    y_max = max(float(np.nanmax(rds_data + rds_err)), float(np.nanmax(rds_mod))) * 1.10
    y_min = min(-0.04 * y_max, float(np.nanmin(rds_data - rds_err)) * 1.1)
    ax_lin.set_ylim(bottom=y_min, top=y_max)
    ax_lin.grid(True, which="both", linestyle=":", alpha=0.35)
    if show_ylabel:
        ax_lin.set_ylabel(r"$R_p\,\Delta\Sigma\ [10^6\,\mathrm{M_\odot / pc}]$")
    ax_lin.set_xlabel(r"$R_p\ [\mathrm{Mpc}]$")


def plot_profile_fit_decomposition(
    summary: Table,
    label: str,
    out_plot_path: Path,
) -> None:
    """Plot profile fit decomposition in a 2xN grid (2x1 for 1bin, 2x4 for 4bin).

    Top row displays log-scale DeltaSigma; bottom row displays linear-scale Rp * DeltaSigma.
    """
    n_bins = len(summary)
    rp_mpc = (
        summary["r_mpc"][0] if "r_mpc" in summary.colnames else summary.meta["r_mpc"]
    )
    is_multi_bin = n_bins > 1

    fig, axes = plt.subplots(
        2,
        n_bins,
        figsize=(3.8 * n_bins + 1.0, 7.2) if is_multi_bin else (6.5, 7.5),
        sharex="col",
        squeeze=False,
    )

    for b_idx in range(n_bins):
        ax_log = axes[0, b_idx]
        ax_lin = axes[1, b_idx]
        row = summary[b_idx]
        bin_num = row["bin_id"]

        bin_title = f"{label} - Bin {bin_num}" if not is_multi_bin else f"Bin {bin_num}"
        show_ylabel = b_idx == 0
        show_legend = b_idx == 0

        plot_bin_column(
            ax_log=ax_log,
            ax_lin=ax_lin,
            rp_mpc=rp_mpc,
            row=row,
            bin_title=bin_title,
            is_multi_bin=is_multi_bin,
            show_ylabel=show_ylabel,
            show_legend=show_legend,
        )

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

# Fixed miscentering parameters based on optical cluster centering literature
FIXED_F_MIS = 0.25
FIXED_SIGMA_R = 100.0  # kpc/h

# Upstream bin filtering: maximum valid complete bins to process per label.
# - redm_r16_4bin has a completeness cutoff (lambda >= 20) and only populates the top 2 bins.
# - camira_4bin has a richness cut (N_mem >= 10) and only populates the top 3 bins.
# Incomplete tail bins are discarded upstream so downstream only displays valid bins.
LABEL_MAX_VALID_BINS: dict[str, int] = {
    "redm_r16_4bin": 2,
    "camira_4bin": 3,
}

OUTPUT_PLOT_DIR = project_root / "output/plots_for_agents"

# %% [Stage 1: Process and fit all configured labels]
all_summaries = {}
for lbl in LABELS:
    try:
        max_b = LABEL_MAX_VALID_BINS.get(lbl, None)
        tbl_summary, _ = fit_label_profile_scatter(
            label=lbl,
            version=VERSION,
            root_path=project_root,
            cosmology_name=COSMOLOGY_NAME,
            mass_def=MASS_DEF,
            f_mis=FIXED_F_MIS,
            sigma_r=FIXED_SIGMA_R,
            max_valid_bins=max_b,
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
