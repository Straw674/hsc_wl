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

from initial import *  # noqa: F401,F403

# %% [Local Functions]


def load_wl_signals_stacker(wl_signals_path: Path):
    """Build a fast stacker function from precomputed weak-lensing signals table."""
    from astropy.table import Table

    wl_table = Table.read(wl_signals_path, path="data")
    wl_obj_id = np.asarray(wl_table["object_id"], dtype=np.int64)
    wl_idx_map = {oid: idx for idx, oid in enumerate(wl_obj_id)}

    bins = wl_table.meta.get("bins", np.logspace(np.log10(0.1), np.log10(20.0), 12))
    rp_centers = np.sqrt(bins[:-1] * bins[1:])
    h = 0.6766
    rp_h = rp_centers * h

    def stacker(obj_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        indices = [wl_idx_map[oid] for oid in obj_ids if oid in wl_idx_map]
        n_matched = len(indices)
        if n_matched == 0:
            raise ValueError("No objects matched in weak lensing table.")
        sub = wl_table[indices]
        sum_w = np.sum(np.asarray(sub["sum w_ls"]), axis=0)
        sum_num = np.sum(np.asarray(sub["sum w_ls e_t sigma_crit"]), axis=0)
        sum_m = np.sum(np.asarray(sub["sum w_ls m"]), axis=0)
        sum_r = np.sum(np.asarray(sub["sum w_ls (1 - e_rms^2)"]), axis=0)
        sum_msel = np.sum(np.asarray(sub["sum w_ls m_sel"]), axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            resp = (
                (1.0 + sum_m / sum_w) * (2.0 * sum_r / sum_w) * (1.0 + sum_msel / sum_w)
            )
            ds = (sum_num / sum_w) / resp
            err = (1.0 / np.sqrt(sum_w)) / np.abs(resp)
        return ds, err, n_matched

    return stacker, rp_h


def load_variant_catalogs_and_scores(
    variant_specs: dict[str, dict],
    wl_proxy_path: Path,
    top_n: int = 1020,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load top_n cluster IDs, photo-z distributions, and calculate proxy objective metrics."""
    df_proxy = pd.read_parquet(wl_proxy_path)
    clean_proxy = df_proxy.dropna(
        subset=["wl_m500c_proxy_1e14_msun_h", "wl_m500c_proxy_err_shape_1e14_msun_h"]
    )
    proxy_map = dict(
        zip(clean_proxy["object_id"], clean_proxy["wl_m500c_proxy_1e14_msun_h"])
    )
    err_map = dict(
        zip(
            clean_proxy["object_id"],
            clean_proxy["wl_m500c_proxy_err_shape_1e14_msun_h"],
        )
    )

    top_oids_dict = {}
    summary_rows = []

    for key, spec in variant_specs.items():
        cat_df = pd.read_parquet(spec["path"])
        top_slice = cat_df.iloc[:top_n]
        oids = top_slice["object_id"].to_numpy(dtype=np.int64)
        top_oids_dict[key] = oids

        m_vals = []
        w_vals = []
        for oid in oids:
            if oid in proxy_map:
                m = proxy_map[oid]
                e = err_map[oid]
                if e > 0:
                    m_vals.append(m)
                    w_vals.append(1.0 / (e**2))
        m_arr = np.array(m_vals, dtype=float)
        w_arr = np.array(w_vals, dtype=float)
        sum_w = float(np.sum(w_arr))
        weighted_m = float(np.sum(m_arr * w_arr) / sum_w) if sum_w > 0 else np.nan
        snr = float(weighted_m * np.sqrt(sum_w)) if sum_w > 0 else np.nan
        unweighted_m = float(np.mean(m_arr)) if len(m_arr) > 0 else np.nan
        z_col = "z_cl" if "z_cl" in top_slice.columns else "z"
        mean_z = float(top_slice[z_col].mean())

        summary_rows.append(
            {
                "key": key,
                "label": spec["label"],
                "group": spec["group"],
                "color": spec["color"],
                "marker": spec["marker"],
                "linestyle": spec["linestyle"],
                "n_matched_proxy": len(m_vals),
                "weighted_mass_proxy_1e14": weighted_m,
                "stacked_snr": snr,
                "unweighted_mass_1e14": unweighted_m,
                "mean_z": mean_z,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    return top_oids_dict, df_summary


def compute_pairwise_matrices(
    top_oids_dict: dict[str, np.ndarray],
    profiles_dict: dict[str, np.ndarray],
    keys: list[str],
    inner_bins_count: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute top-N sample overlap matrix and profile relative difference matrix."""
    n = len(keys)
    overlap_mat = np.zeros((n, n), dtype=int)
    diff_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        s1 = set(top_oids_dict[keys[i]])
        ds1 = profiles_dict[keys[i]][:inner_bins_count]
        for j in range(n):
            s2 = set(top_oids_dict[keys[j]])
            overlap_mat[i, j] = len(s1.intersection(s2))

            ds2 = profiles_dict[keys[j]][:inner_bins_count]
            rel_diff = np.abs(ds1 - ds2) / ds1
            diff_mat[i, j] = float(np.mean(rel_diff) * 100.0)

    df_overlap = pd.DataFrame(overlap_mat, index=keys, columns=keys)
    df_diff = pd.DataFrame(diff_mat, index=keys, columns=keys)
    return df_overlap, df_diff


def plot_profile_comparison(
    rp_h: np.ndarray,
    profiles_dict: dict[str, np.ndarray],
    errors_dict: dict[str, np.ndarray],
    df_summary: pd.DataFrame,
    df_diff: pd.DataFrame,
    ref_key: str,
    output_png: Path,
) -> None:
    """Create comprehensive 4-panel diagnostic comparison figure."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], hspace=0.28, wspace=0.24)

    ax_prof = fig.add_subplot(gs[0, 0])
    ax_ratio = fig.add_subplot(gs[1, 0], sharex=ax_prof)
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_score = fig.add_subplot(gs[1, 1])

    ref_ds = profiles_dict[ref_key]
    indexed_summary = df_summary.set_index("key")

    # Panel 1: Stacked Profile R * DeltaSigma
    for key, ds in profiles_dict.items():
        err = errors_dict[key]
        row = indexed_summary.loc[key]
        y_val = rp_h * ds
        y_err = rp_h * err
        ax_prof.errorbar(
            rp_h,
            y_val,
            yerr=y_err,
            label=row["label"],
            color=row["color"],
            marker=row["marker"],
            linestyle=row["linestyle"],
            linewidth=1.3 if key == ref_key else 1.0,
            markersize=4.5,
            capsize=0.0,
            alpha=0.9 if row["group"] == "WL-Optimized" else 0.75,
        )

    ax_prof.set_xscale("log")
    ax_prof.set_ylabel(
        r"$R_p\,\Delta\Sigma(R_p)\;[10^6\,M_\odot / \mathrm{pc}]$",
        fontsize=11,
        fontweight="normal",
    )
    ax_prof.set_title(
        r"Top 1020 Stacked Weak Lensing Profiles $R_p\,\Delta\Sigma(R_p)$",
        fontsize=11.5,
        fontweight="normal",
    )
    ax_prof.legend(loc="upper left", fontsize=7.2, framealpha=0.9, ncol=2)
    ax_prof.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: Ratio to Reference (Two Box · WL)
    ax_ratio.axhline(1.0, color="#111827", linestyle="--", linewidth=1.1)
    ax_ratio.axhspan(0.95, 1.05, color="#9CA3AF", alpha=0.15, label=r"$\pm 5\%$ band")

    for key, ds in profiles_dict.items():
        row = indexed_summary.loc[key]
        ratio = ds / ref_ds
        ax_ratio.plot(
            rp_h,
            ratio,
            label=row["label"],
            color=row["color"],
            marker=row["marker"],
            linestyle=row["linestyle"],
            linewidth=1.3 if key == ref_key else 1.0,
            markersize=4.0,
            alpha=0.9 if row["group"] == "WL-Optimized" else 0.75,
        )

    ax_ratio.set_xscale("log")
    ax_ratio.set_ylim(0.70, 1.25)
    ax_ratio.set_xlabel(
        r"$R_p\;[h^{-1}\,\mathrm{Mpc}]\quad\mathrm{(physical)}$",
        fontsize=11,
        fontweight="normal",
    )
    ax_ratio.set_ylabel(
        r"$\Delta\Sigma\,/\,\Delta\Sigma_{\mathrm{Two\,Box\cdot WL}}$",
        fontsize=11,
        fontweight="normal",
    )
    ax_ratio.set_title(
        r"Profile Ratio Relative to Baseline (Two Box · WL)",
        fontsize=11.5,
        fontweight="normal",
    )
    ax_ratio.grid(True, linestyle="--", alpha=0.3)

    # Panel 3: Pairwise Difference Matrix Heatmap (< 2 Mpc/h)
    diff_arr = df_diff.to_numpy()
    im = ax_heat.imshow(diff_arr, cmap="YlGnBu", vmin=0.0, vmax=20.0)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label(
        r"Mean Relative Difference [\%] ($R_p < 2\,h^{-1}\mathrm{Mpc}$)",
        fontsize=9.5,
        fontweight="normal",
    )

    label_short_map = {
        "rz_diff_preset_1bin": "Preset",
        "rz_diff_single_box_1bin": "1-Box (WL)",
        "rz_diff_single_box_match_recovery_1bin": "1-Box (Match)",
        "rz_diff_1bin": "2-Box (WL)",
        "rz_diff_two_box_red_1bin": "2-Box Red (WL)",
        "rz_diff_six_param_1bin": "6-Param (WL)",
        "rz_diff_two_box_match_recovery_1bin": "2-Box (Match)",
        "rz_diff_two_box_red_match_recovery_1bin": "2-Box Red (Match)",
        "rz_diff_six_param_match_recovery_1bin": "6-Param (Match)",
        "rz_diff_two_box_red_redmapper_matching_1bin": "Red (redM)",
        "rz_diff_two_box_red_camira_matching_1bin": "Red (CAMIRA)",
    }
    short_labels = [label_short_map.get(k, k) for k in df_diff.columns]
    ax_heat.set_xticks(np.arange(len(short_labels)))
    ax_heat.set_yticks(np.arange(len(short_labels)))
    ax_heat.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8.2)
    ax_heat.set_yticklabels(short_labels, fontsize=8.2)
    ax_heat.set_title(
        r"Profile Difference Matrix Across Inner/Virial Bins ($R_p < 2\,h^{-1}\mathrm{Mpc}$)",
        fontsize=10.5,
        fontweight="normal",
    )

    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            val = diff_arr[i, j]
            color = "white" if val > 12 else "black"
            ax_heat.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color=color,
            )

    # Panel 4: Objective Benchmark Score (Weighted Mass Proxy)
    sorted_df = df_summary.sort_values(by="weighted_mass_proxy_1e14", ascending=True)
    y_pos = np.arange(len(sorted_df))
    bar_colors = sorted_df["color"].tolist()
    bars = ax_score.barh(
        y_pos,
        sorted_df["weighted_mass_proxy_1e14"],
        color=bar_colors,
        alpha=0.85,
        edgecolor="#374151",
        height=0.65,
    )

    ax_score.set_yticks(y_pos)
    ax_score.set_yticklabels(sorted_df["label"], fontsize=8.5)
    ax_score.set_xlabel(
        r"$\langle M_{500c}^{\mathrm{WL}} \rangle\;[10^{14}\,h^{-1}M_\odot]$",
        fontsize=10.5,
        fontweight="normal",
    )
    ax_score.set_title(
        r"Top 1020 Weak Lensing Mass Proxy Benchmark $\langle M_{500c}^{\mathrm{WL}} \rangle$",
        fontsize=11.0,
        fontweight="normal",
    )
    ax_score.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax_score.set_xlim(0.95, 1.42)

    for bar, m_val, snr in zip(
        bars,
        sorted_df["weighted_mass_proxy_1e14"],
        sorted_df["stacked_snr"],
        strict=True,
    ):
        ax_score.text(
            m_val + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{m_val:.3f} (SNR={snr:.1f})",
            va="center",
            fontsize=7.8,
            color="#1F2937",
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    logger.info("Saved profile comparison figure to %s", output_png)
    plt.show()
    plt.close(fig)


# %% [Global Configuration]

logger = logging.getLogger(__name__)

CLUSTER_FINDER_ROOT = Path("/Users/xinq/cluster_finder")
CF_OUTPUT_ROOT = CLUSTER_FINDER_ROOT / "output/fall_hectomap_spring"
WL_SIGNALS_PATH = CF_OUTPUT_ROOT / "wl_signals.hdf5"
WL_PROXY_PATH = CF_OUTPUT_ROOT / "wl_mass_proxy.parquet"

TOP_N_CLUSTERS = 1020
REFERENCE_KEY = "rz_diff_1bin"
OUTPUT_COMPARISON_PNG = (
    project_root / "output/plots_for_agents/compare_rz_diff_wl_profiles.png"
)

VARIANT_SPECS = {
    "rz_diff_preset_1bin": {
        "path": CF_OUTPUT_ROOT / "rz_diff/preset/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Preset)",
        "group": "Preset",
        "color": "#C7682E",
        "marker": "o",
        "linestyle": "--",
    },
    "rz_diff_single_box_1bin": {
        "path": CF_OUTPUT_ROOT
        / "rz_diff/single_box_wl/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Single Box · WL)",
        "group": "WL-Optimized",
        "color": "#EA580C",
        "marker": "s",
        "linestyle": "-",
    },
    "rz_diff_single_box_match_recovery_1bin": {
        "path": CF_OUTPUT_ROOT
        / "rz_diff/single_box_match_recovery/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Single Box · Match)",
        "group": "Intermediate",
        "color": "#A16207",
        "marker": "d",
        "linestyle": "-.",
    },
    "rz_diff_1bin": {
        "path": CF_OUTPUT_ROOT / "rz_diff/baseline/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Two Box · WL)",
        "group": "WL-Optimized",
        "color": "#D97706",
        "marker": "^",
        "linestyle": "-",
    },
    "rz_diff_two_box_red_1bin": {
        "path": CF_OUTPUT_ROOT
        / "rz_diff/two_box_red_wl/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Two Box Red · WL)",
        "group": "WL-Optimized",
        "color": "#B91C1C",
        "marker": "v",
        "linestyle": "-",
    },
    "rz_diff_six_param_1bin": {
        "path": CF_OUTPUT_ROOT / "rz_diff/six_param_wl/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Six Param · WL)",
        "group": "WL-Optimized",
        "color": "#2563EB",
        "marker": "P",
        "linestyle": "-",
    },
    "rz_diff_two_box_match_recovery_1bin": {
        "path": CF_OUTPUT_ROOT
        / "rz_diff/two_box_match_recovery/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Two Box · Match)",
        "group": "Match-Optimized",
        "color": "#7B2CBF",
        "marker": "X",
        "linestyle": ":",
    },
    "rz_diff_two_box_red_match_recovery_1bin": {
        "path": CF_OUTPUT_ROOT
        / "rz_diff/two_box_red_match_recovery/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Two Box Red · Match)",
        "group": "Match-Optimized",
        "color": "#0F766E",
        "marker": "*",
        "linestyle": ":",
    },
    "rz_diff_six_param_match_recovery_1bin": {
        "path": CF_OUTPUT_ROOT
        / "rz_diff/six_param_match_recovery/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Six Param · Match)",
        "group": "Match-Optimized",
        "color": "#334155",
        "marker": "h",
        "linestyle": ":",
    },
    "rz_diff_two_box_red_redmapper_matching_1bin": {
        "path": CF_OUTPUT_ROOT
        / "rz_diff/two_box_red_redmapper_matching/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Two Box Red · redMaPPer Match)",
        "group": "Match-Optimized",
        "color": "#CC79A7",
        "marker": "<",
        "linestyle": ":",
    },
    "rz_diff_two_box_red_camira_matching_1bin": {
        "path": CF_OUTPUT_ROOT
        / "rz_diff/two_box_red_camira_matching/rz_diff_cluster_catalog.parquet",
        "label": "r-z Diff (Two Box Red · CAMIRA Match)",
        "group": "Match-Optimized",
        "color": "#56B4E9",
        "marker": ">",
        "linestyle": ":",
    },
}

# %% [Stage 1: Load Catalogs and Compute Objective Scores]

logger.info(
    "Loading 11 rz_diff cluster catalogs and computing WL objective metrics for top %d...",
    TOP_N_CLUSTERS,
)
top_oids_dict, df_summary = load_variant_catalogs_and_scores(
    variant_specs=VARIANT_SPECS,
    wl_proxy_path=WL_PROXY_PATH,
    top_n=TOP_N_CLUSTERS,
)

print("\n=== Top 1020 Weak Lensing Mass Proxy Benchmark Summary ===")
print(
    df_summary[
        [
            "label",
            "group",
            "weighted_mass_proxy_1e14",
            "stacked_snr",
            "unweighted_mass_1e14",
            "mean_z",
        ]
    ].to_string(index=False)
)

# %% [Stage 2: Stack Weak Lensing Shear Profiles]

logger.info("Loading weak lensing signals and stacking shear profiles...")
stacker, rp_h = load_wl_signals_stacker(WL_SIGNALS_PATH)

profiles_dict = {}
errors_dict = {}
for key, oids in top_oids_dict.items():
    ds, err, _ = stacker(oids)
    profiles_dict[key] = ds
    errors_dict[key] = err

# %% [Stage 3: Compute Overlap and Relative Difference Matrices]

variant_keys = list(VARIANT_SPECS.keys())
df_overlap, df_diff = compute_pairwise_matrices(
    top_oids_dict=top_oids_dict,
    profiles_dict=profiles_dict,
    keys=variant_keys,
    inner_bins_count=7,
)

print("\n=== Top 1020 Member Overlap Matrix ===")
print(df_overlap.to_string())

print("\n=== Inner/Virial (<2 Mpc/h) Profile Relative Difference Matrix (%) ===")
print(df_diff.round(1).to_string())

# %% [Stage 4: Generate Diagnostic Comparison Figure]

logger.info("Rendering diagnostic comparison plot to %s...", OUTPUT_COMPARISON_PNG)
plot_profile_comparison(
    rp_h=rp_h,
    profiles_dict=profiles_dict,
    errors_dict=errors_dict,
    df_summary=df_summary,
    df_diff=df_diff,
    ref_key=REFERENCE_KEY,
    output_png=OUTPUT_COMPARISON_PNG,
)
