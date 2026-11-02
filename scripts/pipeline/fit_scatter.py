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
        rp_kpc_h = rp_mpc * 1000.0 * h
        ds_obs_colossus = ds_data * factor
        cov_colossus = cov_data * factor**2
        cov_inv_colossus = np.linalg.pinv(cov_colossus)

        fit_res = fit_profile_scatter_map(
            model_state=model_state,
            ds_obs_colossus=ds_obs_colossus,
            cov_inv_colossus=cov_inv_colossus,
            bin_index=b_idx,
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


def plot_profile_fit_decomposition(
    rp_mpc: np.ndarray,
    row: dict,
    label: str,
    out_plot_path: Path,
) -> None:
    """Plot data, total model, 1-halo centered, 1-halo miscentered, and 2-halo."""
    fig, ax = plt.subplots(figsize=(7, 6))

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
        capsize=3,
    )
    ax.plot(rp_mpc, ds_mod, "-", color="red", linewidth=2.2, label="Total Model")
    ax.plot(
        rp_mpc,
        ds_1h_cen,
        "--",
        color="blue",
        linewidth=1.5,
        label="1-halo (Centered)",
    )
    if row["f_mis"] > 0.0:
        ax.plot(
            rp_mpc,
            ds_1h_mis,
            "--",
            color="orange",
            linewidth=1.5,
            label="1-halo (Miscentered)",
        )
    ax.plot(rp_mpc, ds_2h, ":", color="green", linewidth=1.5, label="2-halo Term")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=np.maximum(1e-1, np.nanmin(ds_data) * 0.4))
    ax.set_xlabel(r"$R_p\ [\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\Delta\Sigma\ [\mathrm{M_\odot / pc^2}]$")

    summary_text = "\n".join(
        [
            rf"$\sigma = {row['scatter']:.3f} \pm {row['scatter_err']:.3f}$",
            rf"$\alpha_c = {row['alpha_c']:.2f} \pm {row['alpha_c_err']:.2f}$ ($\langle c \rangle = {row['mean_c']:.1f}$)",
            rf"$\alpha_b = {row['alpha_b']:.2f} \pm {row['alpha_b_err']:.2f}$ ($\langle b \rangle = {row['mean_b']:.2f}$)",
            rf"$f_{{\rm mis}} = {row['f_mis']:.2f} \pm {row['f_mis_err']:.2f}$",
            rf"$\sigma_R = {row['sigma_R']:.1f} \pm {row['sigma_R_err']:.1f}\ \mathrm{{kpc}}/h$",
            rf"$\chi^2/\nu = {row['chi2_reduced']:.3f}\ (\chi^2={row['chi2']:.1f}, \nu={row['n_dof']})$",
            rf"$\langle \log M_{{200m}} \rangle = {row['mean_logm']:.3f}$",
        ]
    )
    ax.text(
        0.05,
        0.05,
        summary_text,
        transform=ax.transAxes,
        fontsize=9.5,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="0.7"
        ),
    )

    bin_num = row["bin_id"]
    ax.set_title(
        f"{label} - Bin {bin_num} Profile Fit Decomposition", fontweight="normal"
    )
    ax.legend(loc="upper right", fontsize=9.5)
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
    "zou21_1bin",
]

VERSION = "Y3"  # "Y1" or "Y3"
COSMOLOGY_NAME = "planck18"
MASS_DEF = "200m"

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
        )
        all_summaries[lbl] = tbl_summary
    except Exception as e:
        logging.error(f"Failed fitting for label {lbl}: {e}")
        import traceback

        traceback.print_exc()

# %% [Stage 2: Visualize fit decompositions]
for lbl, summary in all_summaries.items():
    rp_mpc = (
        summary["r_mpc"][0] if "r_mpc" in summary.colnames else summary.meta["r_mpc"]
    )
    for row in summary:
        b_id = row["bin_id"]
        out_plot = OUTPUT_PLOT_DIR / f"fit_profile_{lbl}_{VERSION}_bin{b_id}.png"
        plot_profile_fit_decomposition(
            rp_mpc=rp_mpc,
            row=row,
            label=lbl,
            out_plot_path=out_plot,
        )
