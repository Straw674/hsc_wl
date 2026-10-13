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

from hsc_wl.coverage import (
    CANONICAL_FIELDS,
    comoving_volume_shell_gpc3,
    in_field,
    load_y3_mask,
)
from initial import *  # noqa: F401,F403

# %% Local Functions


def compute_field_areas(mask: hsp.HealSparseMap) -> dict[str, float]:
    """Compute effective areas (deg^2) for canonical regions inside the Y3 mask."""
    pix_area = hp.nside2pixarea(mask.nside_sparse, degrees=True)
    ra_pix, dec_pix = hp.pix2ang(
        mask.nside_sparse, mask.valid_pixels, nest=True, lonlat=True
    )

    areas = {
        "HECTOMAP": float(np.sum(dec_pix > 30.0) * pix_area),
        "GAMA09H": float(np.sum(in_field(ra_pix, dec_pix, "GAMA09H")) * pix_area),
        "WIDE12H": float(np.sum(in_field(ra_pix, dec_pix, "WIDE12H")) * pix_area),
        "GAMA15H": float(np.sum(in_field(ra_pix, dec_pix, "GAMA15H")) * pix_area),
        "XMM": float(np.sum(in_field(ra_pix, dec_pix, "XMM")) * pix_area),
        "VVDS": float(np.sum(in_field(ra_pix, dec_pix, "VVDS")) * pix_area),
        "TOTAL": float(len(mask.valid_pixels) * pix_area),
    }
    areas["SPRING"] = areas["GAMA09H"] + areas["WIDE12H"] + areas["GAMA15H"]
    areas["FALL"] = areas["XMM"] + areas["VVDS"]
    areas["SPRING+FALL"] = areas["SPRING"] + areas["FALL"]
    return areas


def collect_catalog_statistics(
    root: Path,
    mask: hsp.HealSparseMap,
    field_areas: dict[str, float],
    z_range: tuple[float, float],
) -> pd.DataFrame:
    """Collect object counts and calculate volume-compensated equivalent Top-N."""
    records = []
    a_hm = field_areas["HECTOMAP"]
    a_total = field_areas["TOTAL"]
    v_shell = comoving_volume_shell_gpc3(*z_range)  # Gpc^3
    full_sky_deg2 = 4.0 * np.pi * (180.0 / np.pi) ** 2
    v_hm_mpc3 = (a_hm / full_sky_deg2) * v_shell * 1e9

    # 1. Optical Catalogs (cut to Y3 mask & z_range)
    # CAMIRA S23B Wide
    with open(root / "data/camira_s23b_wide_sm_v3.dat") as f:
        hdr = f.readline().replace("#", "").strip().split()
    camira = pd.read_csv(
        root / "data/camira_s23b_wide_sm_v3.dat",
        sep=r"\s+",
        comment="#",
        names=hdr,
    )
    c_z = camira[camira["z_cl"].between(*z_range)]
    c_in = mask.get_values_pos(c_z["RA"].values, c_z["Dec"].values, lonlat=True)
    n_camira = int(np.sum(c_in))
    records.append(
        {
            "name": "CAMIRA (S23B)",
            "category": "Optical",
            "probe": "HSC Optical",
            "cutoff_desc": r"$N_{\mathrm{mem}} \geq 10$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": n_camira,
            "topn_hm": n_camira / (a_total / a_hm),
        }
    )

    # redMaPPer SDSS R16
    r16 = Table.read(str(root / "data/R16/R16_cluster_catalog_bin.fit")).to_pandas()
    r16_z = r16[r16["zlambda"].between(*z_range)]
    r16_in = mask.get_values_pos(
        r16_z["RAJ2000"].values, r16_z["DEJ2000"].values, lonlat=True
    )
    n_r16 = int(np.sum(r16_in))
    records.append(
        {
            "name": "redMaPPer SDSS (R16)",
            "category": "Optical",
            "probe": "SDSS Optical",
            "cutoff_desc": r"$\lambda \geq 20$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": n_r16,
            "topn_hm": n_r16 / (a_total / a_hm),
        }
    )

    # WH24 (DESI Legacy / unWISE)
    wh24 = pd.read_parquet(root / "data/external_clusters/wh24.parquet")
    wh24_z = wh24[wh24["z"].between(*z_range)]
    wh24_in = mask.get_values_pos(
        wh24_z["ra"].values, wh24_z["dec"].values, lonlat=True
    )
    n_wh24 = int(np.sum(wh24_in))
    records.append(
        {
            "name": "WH24",
            "category": "Optical",
            "probe": "DESI / unWISE",
            "cutoff_desc": r"$\lambda_{500} \geq 10$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": n_wh24,
            "topn_hm": n_wh24 / (a_total / a_hm),
        }
    )

    # WH20 (DESI Legacy)
    wh20 = pd.read_parquet(root / "data/external_clusters/wh20.parquet")
    wh20_z = wh20[wh20["z"].between(*z_range)]
    wh20_in = mask.get_values_pos(
        wh20_z["ra"].values, wh20_z["dec"].values, lonlat=True
    )
    n_wh20 = int(np.sum(wh20_in))
    records.append(
        {
            "name": "WH20",
            "category": "Optical",
            "probe": "DESI Optical",
            "cutoff_desc": r"$N_{\mathrm{gals}} \geq 15$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": n_wh20,
            "topn_hm": n_wh20 / (a_total / a_hm),
        }
    )

    # Zou21 (DESI Legacy / unWISE)
    zou21 = pd.read_parquet(root / "data/external_clusters/zou21.parquet")
    zou21_z = zou21[zou21["z"].between(*z_range)]
    zou21_in = mask.get_values_pos(
        zou21_z["ra"].values, zou21_z["dec"].values, lonlat=True
    )
    n_zou21 = int(np.sum(zou21_in))
    records.append(
        {
            "name": "Zou21",
            "category": "Optical",
            "probe": "DESI / unWISE",
            "cutoff_desc": r"$\mathrm{richness} \geq 10$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": n_zou21,
            "topn_hm": n_zou21 / (a_total / a_hm),
        }
    )

    # CluMPR
    clumpr = pd.read_parquet(root / "data/external_clusters/clumpr.parquet")
    cl_z = clumpr[clumpr["z"].between(*z_range)]
    cl_in = mask.get_values_pos(cl_z["ra"].values, cl_z["dec"].values, lonlat=True)
    n_clumpr = int(np.sum(cl_in))
    records.append(
        {
            "name": "CluMPR",
            "category": "Optical",
            "probe": "DESI Optical",
            "cutoff_desc": r"$\mathrm{richness} \geq 5.3$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": n_clumpr,
            "topn_hm": n_clumpr / (a_total / a_hm),
        }
    )

    # DES Y3 redMaPPer (Fall footprint)
    des_rm = pd.read_parquet(
        root / "data/reference_catalogs/des_y3_redmapper_y3.parquet"
    )
    n_des_rm = len(des_rm)
    a_fall = field_areas["FALL"]
    records.append(
        {
            "name": "DES Y3 redMaPPer",
            "category": "Optical",
            "probe": "DES Optical",
            "cutoff_desc": r"$\lambda \geq 20$",
            "area_deg2": a_fall,
            "region": "Fall",
            "n_y3": n_des_rm,
            "topn_hm": n_des_rm / (a_fall / a_hm),
        }
    )

    # DES Y6 WaZP (Fall footprint)
    des_wazp = pd.read_parquet(root / "data/reference_catalogs/des_y6_wazp_y3.parquet")
    n_wazp = len(des_wazp)
    records.append(
        {
            "name": "DES Y6 WaZP",
            "category": "Optical",
            "probe": "DES Optical",
            "cutoff_desc": r"$N_{\mathrm{gals}} \geq 25$",
            "area_deg2": a_fall,
            "region": "Fall",
            "n_y3": n_wazp,
            "topn_hm": n_wazp / (a_fall / a_hm),
        }
    )

    # KiDS DR3 AMICO (Spring footprint)
    kids = pd.read_parquet(root / "data/reference_catalogs/kids_dr3_amico_y3.parquet")
    n_kids = len(kids)
    a_spring = field_areas["SPRING"]
    records.append(
        {
            "name": "KiDS DR3 AMICO",
            "category": "Optical",
            "probe": "KiDS Optical",
            "cutoff_desc": r"$\lambda^* \geq 15, S/N \geq 3.5$",
            "area_deg2": a_spring,
            "region": "Spring",
            "n_y3": n_kids,
            "topn_hm": n_kids / (a_spring / a_hm),
        }
    )

    # Yang21 Group Catalog thresholds
    yang = pd.read_parquet(root / "data/external_clusters/yang21.parquet")
    y_z = yang[yang["z"].between(*z_range)]
    y_in = mask.get_values_pos(y_z["ra"].values, y_z["dec"].values, lonlat=True)
    y_y3 = y_z[y_in]
    n_yang_m14 = int(np.sum(y_y3["log_m_h"] >= 14.0))
    records.append(
        {
            "name": r"Yang21 ($\log M_h \geq 14.0$)",
            "category": "Optical",
            "probe": "DESI Groups",
            "cutoff_desc": r"$\log M_h \geq 14.0$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": n_yang_m14,
            "topn_hm": n_yang_m14 / (a_total / a_hm),
        }
    )
    n_yang_m135 = int(np.sum(y_y3["log_m_h"] >= 13.5))
    records.append(
        {
            "name": r"Yang21 ($\log M_h \geq 13.5$)",
            "category": "Optical",
            "probe": "DESI Groups",
            "cutoff_desc": r"$\log M_h \geq 13.5$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": n_yang_m135,
            "topn_hm": n_yang_m135 / (a_total / a_hm),
        }
    )

    # 2. X-ray / SZ / WL Shear References
    # ACT DR6 SZ
    act = pd.read_parquet(root / "data/reference_catalogs/act_dr6_y3.parquet")
    a_act = field_areas["SPRING+FALL"]
    records.append(
        {
            "name": "ACT DR6 (SZ)",
            "category": "SZ",
            "probe": "ACT SZ",
            "cutoff_desc": r"$S/N > 5.5$",
            "area_deg2": a_act,
            "region": "Spring+Fall",
            "n_y3": len(act),
            "topn_hm": len(act) / (a_act / a_hm),
        }
    )

    # Chen 2024 (WL Shear Peaks)
    chen = pd.read_parquet(root / "data/reference_catalogs/chen2024_y3.parquet")
    records.append(
        {
            "name": "Chen+2024 (WL Shear)",
            "category": "WL Shear",
            "probe": "HSC WL Shear",
            "cutoff_desc": r"$S/N \geq 4.7$",
            "area_deg2": a_total,
            "region": "Full Y3",
            "n_y3": len(chen),
            "topn_hm": len(chen) / (a_total / a_hm),
        }
    )

    # eRASS:3 (X-ray Main)
    erass3 = pd.read_parquet(root / "data/reference_catalogs/erass3_y3.parquet")
    records.append(
        {
            "name": "eRASS:3 (X-ray)",
            "category": "X-ray",
            "probe": "eROSITA X-ray",
            "cutoff_desc": r"$\mathrm{EXT\_LIKE} \geq 6$",
            "area_deg2": a_spring,
            "region": "Spring",
            "n_y3": len(erass3),
            "topn_hm": len(erass3) / (a_spring / a_hm),
        }
    )

    # eFEDS (X-ray)
    efeds = pd.read_parquet(root / "data/reference_catalogs/efeds_y3.parquet")
    a_gama09 = field_areas["GAMA09H"]
    records.append(
        {
            "name": "eFEDS (X-ray)",
            "category": "X-ray",
            "probe": "eROSITA X-ray",
            "cutoff_desc": r"$\mathrm{EXT\_LIKE} \geq 6$",
            "area_deg2": a_gama09,
            "region": "GAMA09H",
            "n_y3": len(efeds),
            "topn_hm": len(efeds) / (a_gama09 / a_hm),
        }
    )

    # XXL DR2 (X-ray)
    xxl = pd.read_parquet(root / "data/reference_catalogs/xxl_dr2_y3.parquet")
    a_xmm = field_areas["XMM"]
    records.append(
        {
            "name": "XXL DR2 (X-ray)",
            "category": "X-ray",
            "probe": "XMM X-ray",
            "cutoff_desc": r"$\mathrm{Bextlike} \geq 33$",
            "area_deg2": a_xmm,
            "region": "XMM",
            "n_y3": len(xxl),
            "topn_hm": len(xxl) / (a_xmm / a_hm),
        }
    )

    # 3. Huang et al. 2022 (redMaPPer S16A bins)
    a_s16a = 137.83
    huang_bins = [
        ("Huang22 Bin 0", r"$\lambda \geq 35$", 53),
        ("Huang22 Bin 0+1", r"$\lambda \geq 20$", 249),
        ("Huang22 Bin 0..2", r"$\lambda \geq 10$", 909),
        ("Huang22 4bin Total", r"$\lambda \geq 6$", 2068),
        ("Huang22 Catalog Limit", r"$\lambda \geq 5$", 2784),
    ]
    for bname, desc, n_s16a in huang_bins:
        records.append(
            {
                "name": bname,
                "category": "Huang 2022",
                "probe": "redMaPPer S16A",
                "cutoff_desc": desc,
                "area_deg2": a_s16a,
                "region": "S16A Ref",
                "n_y3": n_s16a,
                "topn_hm": n_s16a / (a_s16a / a_hm),
            }
        )

    df_out = pd.DataFrame(records)
    df_out["vol_density_mpc3"] = df_out["topn_hm"] / v_hm_mpc3
    return df_out.sort_values("topn_hm", ascending=True).reset_index(drop=True)


def plot_catalog_cutoffs_visualization(df: pd.DataFrame, output_png: Path) -> None:
    """Generate and save publication-grade visualization of catalog truncation points."""
    fig, ax = plt.subplots(figsize=(11, 9))

    category_colors = {
        "Optical": "#228833",  # Tol green
        "Huang 2022": "#4477AA",  # Tol blue
        "X-ray": "#EE6677",  # Tol red
        "SZ": "#EE7733",  # Tol orange
        "WL Shear": "#AA3377",  # Tol purple
    }

    category_markers = {
        "Optical": "o",
        "Huang 2022": "s",
        "X-ray": "^",
        "SZ": "D",
        "WL Shear": "P",
    }

    y_positions = np.arange(len(df))

    # Reference benchmark vertical lines
    ax.axvline(
        x=100.0,
        color="#000000",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        zorder=1,
        label=r"Current 1bin Standard (Top-100, $\lambda \approx 20$)",
    )
    ax.axvline(
        x=1000.0,
        color="#882255",
        linestyle=":",
        linewidth=2.0,
        alpha=0.85,
        zorder=1,
        label=r"10$\times$ Expanded Standard (Top-1k, $M_{\mathrm{vir}} \sim 4.5\times 10^{13} M_\odot$)",
    )

    # Reference region bands
    ax.axvspan(100.0, 1000.0, color="#CCCCCC", alpha=0.18, zorder=0)

    for i, row in df.iterrows():
        cat = row["category"]
        c = category_colors.get(cat, "#333333")
        m = category_markers.get(cat, "o")
        val = row["topn_hm"]

        # Horizontal connecting guide line
        ax.hlines(
            y=i,
            xmin=1.0,
            xmax=val,
            color=c,
            alpha=0.35,
            linewidth=1.4,
            zorder=2,
        )

        # Main marker
        ax.scatter(
            val,
            i,
            color=c,
            edgecolor="#222222",
            marker=m,
            s=80,
            linewidth=1.1,
            zorder=3,
        )

        # Annotate text
        text_str = f" {val:.1f} ({row['cutoff_desc']})"
        ax.text(
            val * 1.06,
            i,
            text_str,
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="normal",
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
            zorder=4,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["name"], fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(3.0, 3000.0)
    ax.set_ylim(-0.8, len(df) - 0.2)

    ax.set_xlabel(
        r"Volume-Compensated Equivalent Top-N (HectoMAP Volume Scale, $V \approx 3.37\times 10^7\ \mathrm{Mpc}^3$)",
        fontsize=10.5,
        fontweight="normal",
    )
    ax.set_ylabel("Cluster / Group Catalog", fontsize=10.5, fontweight="normal")
    ax.set_title(
        r"Cluster Catalog Truncation Thresholds Normalized to Volume Density",
        fontsize=11.5,
        fontweight="normal",
        pad=12,
    )

    # Secondary upper x-axis showing volume density in Mpc^-3
    v_hm = 3.370e7  # Mpc^3
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda topn: topn / v_hm,
            lambda n: n * v_hm,
        ),
    )
    secax.set_xlabel(
        r"Comoving Volume Density $n$ $[\mathrm{Mpc}^{-3}]$",
        fontsize=10.0,
        fontweight="normal",
        labelpad=8,
    )

    # Custom category legend handles
    handles, labels = ax.get_legend_handles_labels()
    for cat, col in category_colors.items():
        m = category_markers[cat]
        line = plt.Line2D(
            [0],
            [0],
            marker=m,
            color="w",
            markerfacecolor=col,
            markeredgecolor="#222222",
            markersize=8,
            label=cat,
        )
        handles.append(line)
        labels.append(cat)

    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        framealpha=0.92,
        fontsize=8.5,
    )

    ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.35, zorder=0)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %% Global Configuration

REDSHIFT_RANGE = (0.19, 0.52)
OUTPUT_PLOT_PATH = project_root / "output/plots_for_agents/catalog_cutoffs_topn.png"

# %% [Stage 1: Load Mask and Compute Field Areas]

mask = load_y3_mask(project_root)
field_areas = compute_field_areas(mask)
print("Effective Field Areas inside Y3 Mask:")
for fname, farea in field_areas.items():
    print(f"  {fname:<12}: {farea:.2f} deg^2")

# %% [Stage 2: Collect Catalog Statistics]

catalog_summary_df = collect_catalog_statistics(
    project_root, mask, field_areas, REDSHIFT_RANGE
)
print("\nCatalog Truncation Summary:")
print(
    catalog_summary_df[
        [
            "name",
            "category",
            "cutoff_desc",
            "n_y3",
            "area_deg2",
            "topn_hm",
            "vol_density_mpc3",
        ]
    ].to_string()
)

# %% [Stage 3: Generate and Save Visualization]

plot_catalog_cutoffs_visualization(catalog_summary_df, OUTPUT_PLOT_PATH)
print(f"\nVisualization saved to {OUTPUT_PLOT_PATH}")
