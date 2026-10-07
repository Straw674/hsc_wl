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

from hsc_wl.prepare import gaussian_kde_1d, load_stratified_candidates
from initial import *  # noqa: F401,F403
from src.data import read_catalog_frame

# %% Local Functions


def load_lens_data(labels: list[str | tuple], root: Path) -> dict[str, Table]:
    """Load prepared lens tables for the given configurations.

    Parameters
    ----------
    labels : list of str or tuple
        Run labels to load, e.g. ["camira_1bin", ...].
    root : Path
        Project root path.

    Returns
    -------
    dict
        Dictionary mapping run label to the loaded astropy Table.
    """
    from hsc_wl.config import RUN_REGISTRY

    dfs = {}
    for item in labels:
        if isinstance(item, tuple):
            label = f"{item[0]}_{item[1]}"
        else:
            label = str(item)

        if label in RUN_REGISTRY:
            cfg = RUN_REGISTRY[label]
            save_root = cfg.resolved_save_root(root)
            file_path = save_root / f"prepare/{label}_lenses.fits"
        else:
            catalog_id, nbins = label.rsplit("_", 1)
            file_path = (
                root / f"output/{catalog_id}/{nbins}/prepare/{label}_lenses.fits"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"Prepared lens catalog not found at {file_path}")

        print(f"Loading prepared lenses from {file_path.name}...")
        tbl = Table.from_pandas(read_catalog_frame(file_path))
        tbl["rank"] = np.arange(1, len(tbl) + 1)
        dfs[label] = tbl

    return dfs


def match_clusters_with_redshift(
    source: Table,
    target: Table,
    *,
    r_phys_mpc_h: float,
    max_delta_z: float,
) -> np.ndarray:
    """Find any target within the source-centered aperture and absolute redshift cut."""
    from astropy.coordinates import search_around_sky

    matched = np.zeros(len(source), dtype=bool)
    if len(source) == 0 or len(target) == 0:
        return matched
    source_z = np.asarray(source["z"], dtype=float)
    target_z = np.asarray(target["z"], dtype=float)
    source_valid = np.isfinite(source_z) & (source_z > 0)
    target_valid = np.isfinite(target_z) & (target_z > 0)
    source_indices = np.flatnonzero(source_valid)
    if not source_valid.any() or not target_valid.any():
        return matched
    source_coords = SkyCoord(
        ra=np.asarray(source["ra"], dtype=float)[source_valid] * u.deg,
        dec=np.asarray(source["dec"], dtype=float)[source_valid] * u.deg,
    )
    target_coords = SkyCoord(
        ra=np.asarray(target["ra"], dtype=float)[target_valid] * u.deg,
        dec=np.asarray(target["dec"], dtype=float)[target_valid] * u.deg,
    )
    distances = Planck18.angular_diameter_distance(source_z[source_valid]).value
    radii_rad = (r_phys_mpc_h / Planck18.h) / distances
    source_idx, target_idx, separations, _ = search_around_sky(
        source_coords, target_coords, np.max(radii_rad) * u.rad
    )
    accepted = (separations.rad < radii_rad[source_idx]) & (
        np.abs(source_z[source_valid][source_idx] - target_z[target_valid][target_idx])
        < max_delta_z
    )
    matched[source_indices[source_idx[accepted]]] = True
    return matched


def compute_pairwise_matches(
    dfs: dict[str, Table],
    *,
    r_phys_mpc_h: float,
    max_delta_z: float,
) -> pd.DataFrame:
    """Count clusters with any spatially and redshift-consistent counterpart."""
    matrix = [
        [
            int(
                match_clusters_with_redshift(
                    source,
                    target,
                    r_phys_mpc_h=r_phys_mpc_h,
                    max_delta_z=max_delta_z,
                ).sum()
            )
            for target in dfs.values()
        ]
        for source in dfs.values()
    ]
    return pd.DataFrame(matrix, index=list(dfs), columns=list(dfs))


def is_homologous_pair(name_i: str, name_j: str) -> bool:
    """Determine if two catalogs belong to the same method/family or are diagonal."""
    if name_i == name_j:
        return True
    families = ("rz_diff", "yang21", "clumpr")
    for fam in families:
        if name_i.startswith(fam) and name_j.startswith(fam):
            return True
    return False


def plot_matching_heatmap(
    df_match: pd.DataFrame,
    save_path: Path,
    display_names: dict[str, str] | None = None,
    r_phys_mpc_h: float = 0.5,
    *,
    max_delta_z: float,
):
    """Plot pairwise matching statistics as a heatmap grid using matplotlib.

    Parameters
    ----------
    df_match : pd.DataFrame
        Matrix of match counts.
    save_path : Path
        Output image path.
    display_names : dict of str -> str, optional
        Human-readable labels for catalogs.
    r_phys_mpc_h : float, default 0.5
        Physical transverse matching radius in Mpc/h.
    """
    from matplotlib.colors import Normalize

    names_map = display_names or {}
    n = len(df_match.index)
    fig_w = max(8.5, 0.75 * n + 2.0)
    fig_h = max(7.2, 0.65 * n + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = df_match.values
    cat_keys = list(df_match.index)
    labels = [names_map.get(k, k) for k in cat_keys]

    # Row i is the source catalog, cell (i, j) is the fraction of row i matched to column j
    row_totals = np.diag(data)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    data_pct = (data / row_totals_safe[:, None]) * 100.0

    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            mask[i, j] = is_homologous_pair(cat_keys[i], cat_keys[j])

    cross_method_vals = data_pct[~mask]
    if len(cross_method_vals) > 0:
        vmin = max(0.0, float(np.floor(cross_method_vals.min() / 5.0) * 5.0))
        vmax = min(100.0, float(np.ceil(cross_method_vals.max() / 5.0) * 5.0))
        if vmin >= vmax:
            vmax = min(100.0, vmin + 5.0)
    else:
        vmin, vmax = 0.0, 100.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="#E5E7EB")
    data_pct_masked = np.ma.masked_array(data_pct, mask=mask)

    im = ax.imshow(data_pct_masked, cmap=cmap, aspect="equal", norm=norm)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Cross-Method Match Fraction (%)", fontsize=10.5)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(
        labels, rotation=25, ha="right", rotation_mode="anchor", fontsize=10
    )
    ax.set_yticklabels(labels, fontsize=10)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(False, which="both")
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            pct = data_pct[i, j]
            total = row_totals[i]

            if mask[i, j]:
                if i == j:
                    text = f"{val}\n(100.0%)"
                else:
                    text = f"{val}/{total}\n({pct:.1f}%)"
                text_color = "#4B5563"
            else:
                text = f"{val}/{total}\n({pct:.1f}%)"
                norm_val = norm(pct)
                text_color = "white" if norm_val > 0.60 else "black"

            cell_fs = 9.5 if n <= 8 else (8.0 if n <= 12 else 7.0)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontweight="normal",
                fontsize=cell_fs,
            )

    ax.set_title(
        f"Pairwise Lens Match Fractions (R < {r_phys_mpc_h:g} Mpc/h, |Δz| < {max_delta_z:g})\n"
        f"Stratified Redshift-Controlled (Matched redMaPPer N(z), N={row_totals[0]} per catalog)",
        fontsize=11.5,
        pad=14,
        fontweight="normal",
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Matching heatmap saved to {save_path}")
    plt.show()
    plt.close(fig)


def compute_consensus_breakdown(
    dfs: dict[str, Table],
    r_phys_mpc_h: float = 0.5,
    *,
    max_delta_z: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute consensus counts with projected-radius and absolute-redshift cuts.

    Parameters
    ----------
    dfs : dict of str -> Table
        Loaded lens catalogs.
    r_phys_mpc_h : float, default 0.5
        Physical transverse matching radius in Mpc/h.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        df_counts: Table of raw counts.
        df_pct: Table of percentages relative to each catalog's total.
    """
    catalog_names = list(dfs.keys())
    n_cats = len(catalog_names)

    counts_matrix = np.zeros((n_cats, n_cats), dtype=int)

    for i, name in enumerate(catalog_names):
        n_obj = len(dfs[name])

        matched_counts = np.zeros(n_obj, dtype=int)
        for j, other_name in enumerate(catalog_names):
            if i == j:
                continue
            matched_counts += match_clusters_with_redshift(
                dfs[name],
                dfs[other_name],
                r_phys_mpc_h=r_phys_mpc_h,
                max_delta_z=max_delta_z,
            ).astype(int)

        for k in range(n_cats):
            counts_matrix[i, k] = np.sum(matched_counts == k)

    col_labels = [f"{k} Other Catalogs" for k in range(n_cats)]
    df_counts = pd.DataFrame(counts_matrix, index=catalog_names, columns=col_labels)

    row_totals = counts_matrix.sum(axis=1)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    pct_matrix = (counts_matrix / row_totals_safe[:, None]) * 100.0
    df_pct = pd.DataFrame(pct_matrix, index=catalog_names, columns=col_labels)

    return df_counts, df_pct


def plot_consensus_breakdown(
    df_counts: pd.DataFrame,
    df_pct: pd.DataFrame,
    colors: list[str],
    markers: list[str],
    save_path: Path,
    display_names: dict[str, str] | None = None,
    r_phys_mpc_h: float = 0.5,
    *,
    max_delta_z: float,
):
    """Plot consensus level profiles across catalogs as a multi-line plot."""
    names_map = display_names or {}
    pct_matrix = df_pct.values
    catalog_names = list(df_pct.index)
    n_cats = len(catalog_names)

    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    x_vals = np.arange(n_cats)

    for idx, name in enumerate(catalog_names):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        disp_name = names_map.get(name, name)
        ax.plot(
            x_vals,
            pct_matrix[idx],
            label=disp_name,
            color=c,
            marker=m,
            markersize=7,
            linewidth=1.8,
            alpha=0.9,
        )

    ax.set_xticks(x_vals)
    ax.set_xticklabels(
        [
            f"{k}\n(Unique)"
            if k == 0
            else f"{k}\n(All {n_cats})"
            if k == n_cats - 1
            else str(k)
            for k in x_vals
        ],
        fontsize=10.5,
    )
    ax.set_xlabel(
        "Number of Other Matched Catalogs (Consensus Level)",
        fontsize=11.0,
        labelpad=8,
    )
    ax.set_ylabel("Cluster Fraction (%)", fontsize=11.0)
    ax.set_title(
        f"Consensus Profiles across Stratified Catalogs (R < {r_phys_mpc_h:g} Mpc/h, |Δz| < {max_delta_z:g}, Matched redMaPPer N(z))",
        fontsize=11.5,
        pad=10,
        fontweight="normal",
    )
    ax.grid(False, which="both")
    ax.set_ylim(0, max(50.0, float(np.max(pct_matrix)) + 8.0))
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9, ncol=2)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Consensus breakdown plot saved to {save_path}")
    plt.show()
    plt.close(fig)


def compute_tier_consensus_breakdown(
    dfs: dict[str, Table],
    n_bins: int = 4,
    display_names: dict[str, str] | None = None,
    r_phys_mpc_h: float = 0.5,
    *,
    max_delta_z: float,
) -> pd.DataFrame:
    """Compute redshift-consistent consensus statistics within proxy rank tiers."""
    catalog_names = list(dfs.keys())
    n_cats = len(catalog_names)
    names_map = display_names or {}

    records = []

    for i, name_i in enumerate(catalog_names):
        n_i = len(dfs[name_i])

        matched_counts = np.zeros(n_i, dtype=int)
        for j, name_j in enumerate(catalog_names):
            if i == j:
                continue
            matched_counts += match_clusters_with_redshift(
                dfs[name_i],
                dfs[name_j],
                r_phys_mpc_h=r_phys_mpc_h,
                max_delta_z=max_delta_z,
            ).astype(int)

        bin_splits = np.array_split(np.arange(n_i), n_bins)

        for b_idx, idx_slice in enumerate(bin_splits):
            sub_k = matched_counts[idx_slice]
            n_sub = len(idx_slice)
            r_start = idx_slice[0] + 1
            r_end = idx_slice[-1] + 1

            mean_val = float(np.mean(sub_k))
            sem_val = (
                float(np.std(sub_k, ddof=1) / np.sqrt(n_sub)) if n_sub > 1 else 0.0
            )
            median_val = float(np.median(sub_k))

            pct_solo = float(np.mean(sub_k == 0) * 100.0)
            pct_low = float(np.mean(sub_k == 1) * 100.0)
            pct_med = float(np.mean(sub_k == 2) * 100.0)
            pct_high = float(np.mean(sub_k == (n_cats - 1)) * 100.0)

            rec = {
                "catalog": name_i,
                "display_name": names_map.get(name_i, name_i),
                "bin_idx": b_idx,
                "bin_label": f"Tier {b_idx + 1}\n(Ranks {r_start}–{r_end})",
                "tier_name": f"Tier {b_idx + 1}",
                "n_clusters": n_sub,
                "rank_range": (r_start, r_end),
                "mean_matches": mean_val,
                "sem_matches": sem_val,
                "median_matches": median_val,
                "pct_solo": pct_solo,
                "pct_low": pct_low,
                "pct_med": pct_med,
                "pct_high": pct_high,
            }
            for k in range(n_cats):
                rec[f"count_k_{k}"] = int(np.sum(sub_k == k))
                rec[f"pct_k_{k}"] = float(np.sum(sub_k == k) / n_sub * 100.0)

            records.append(rec)

    return pd.DataFrame(records)


def compute_tier_pairwise_matches(
    dfs: dict[str, Table],
    n_bins: int = 4,
    display_names: dict[str, str] | None = None,
    r_phys_mpc_h: float = 0.5,
    *,
    max_delta_z: float,
) -> dict[int, pd.DataFrame]:
    """Match each proxy tier to full catalogs using spatial and redshift cuts."""
    catalog_names = list(dfs.keys())
    n_cats = len(catalog_names)
    names_map = display_names or {}

    tier_matrices = {}

    for b_idx in range(n_bins):
        mat = np.zeros((n_cats, n_cats), dtype=float)
        for i, name_i in enumerate(catalog_names):
            n_i = len(dfs[name_i])
            bin_splits = np.array_split(np.arange(n_i), n_bins)
            idx_slice = bin_splits[b_idx]
            n_sub = len(idx_slice)

            source = dfs[name_i][idx_slice]

            for j, name_j in enumerate(catalog_names):
                if i == j:
                    mat[i, j] = 100.0
                    continue
                matched = match_clusters_with_redshift(
                    source,
                    dfs[name_j],
                    r_phys_mpc_h=r_phys_mpc_h,
                    max_delta_z=max_delta_z,
                ).sum()
                mat[i, j] = (matched / n_sub) * 100.0

        disp_labels = [names_map.get(name, name) for name in catalog_names]
        tier_matrices[b_idx] = pd.DataFrame(mat, index=disp_labels, columns=disp_labels)

    return tier_matrices


def plot_tier_consensus_profiles(
    tier_df: pd.DataFrame,
    catalog_order: list[str],
    colors: list[str],
    markers: list[str],
    save_path: Path,
    display_names: dict[str, str] | None = None,
    *,
    r_phys_mpc_h: float,
    max_delta_z: float,
):
    """Plot multi-panel tiered consensus breakdown figure."""
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    names_map = display_names or {}
    n_cats = len(catalog_order)

    ncols = 3 if n_cats > 4 else 2
    nrows = (n_cats + ncols - 1) // ncols

    fig = plt.figure(figsize=(max(14.0, 1.05 * n_cats), max(10.0, 2.0 * nrows + 4.0)))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.8, 1.4], hspace=0.35)

    # Top Subplot: Transposed Heatmap Matrix (4 bins x n_cats catalogs)
    ax_heat = fig.add_subplot(gs[0])
    mat_mean = np.zeros((4, n_cats))
    mat_high = np.zeros((4, n_cats))
    mat_solo = np.zeros((4, n_cats))

    for j, name in enumerate(catalog_order):
        c_df = tier_df[tier_df["catalog"] == name].sort_values("bin_idx")
        mat_mean[:, j] = c_df["mean_matches"].values
        mat_high[:, j] = c_df["pct_high"].values
        mat_solo[:, j] = c_df["pct_solo"].values

    norm = Normalize(vmin=float(mat_mean.min()), vmax=float(mat_mean.max()))

    im = ax_heat.imshow(mat_mean, cmap="YlGnBu", aspect="auto", norm=norm)

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.02)
    cbar.set_label(
        f"Mean Matched Catalogs (out of {n_cats - 1})",
        fontsize=10.0,
    )

    cat_labels = [names_map.get(name, name) for name in catalog_order]
    bin_row_labels = [
        "Tier 1 (Q1: Ranks 1–255)",
        "Tier 2 (Q2: Ranks 256–510)",
        "Tier 3 (Q3: Ranks 511–765)",
        "Tier 4 (Q4: Ranks 766–1020)",
    ]

    ax_heat.set_xticks(np.arange(n_cats))
    ax_heat.set_xticklabels(cat_labels, fontsize=9, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(4))
    ax_heat.set_yticklabels(bin_row_labels, fontsize=10)
    ax_heat.set_title(
        f"(a) Mean Consensus Score by Proxy Tier (R < {r_phys_mpc_h:g} Mpc/h, |Δz| < {max_delta_z:g})",
        fontsize=11.5,
        pad=10,
        fontweight="normal",
    )

    for i in range(4):
        for j in range(n_cats):
            val = mat_mean[i, j]
            h_val = mat_high[i, j]
            solo_val = mat_solo[i, j]
            norm_val = norm(val)
            text_color = "white" if norm_val > 0.55 else "black"
            txt = f"{val:.2f}\n({h_val:.0f}% All {n_cats})"
            if solo_val > 0:
                txt += f"\n[{solo_val:.0f}% solo]"
            ax_heat.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="normal",
            )

    ax_heat.set_xticks(np.arange(n_cats + 1) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(5) - 0.5, minor=True)
    ax_heat.grid(False, which="both")
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    # Bottom Subplot: Stacked Consensus Spectrum (Grid)
    gs_spec = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=gs[1], hspace=0.8, wspace=0.22
    )
    cmap = plt.colormaps["RdYlBu"]
    spec_colors = [cmap(i / max(1, n_cats - 1)) for i in range(n_cats)]
    spec_labels = [
        "Solo (k=0)"
        if k == 0
        else f"Full (k={k}, All {n_cats})"
        if k == n_cats - 1
        else f"k={k}"
        for k in range(n_cats)
    ]

    bars = []
    for idx, name in enumerate(catalog_order):
        row_i, col_i = divmod(idx, ncols)
        ax_b = fig.add_subplot(gs_spec[row_i, col_i])
        c_df = tier_df[tier_df["catalog"] == name].sort_values(
            "bin_idx", ascending=False
        )

        y_pos = np.arange(4)
        means = c_df["mean_matches"].values

        left_cum = np.zeros(4)
        for k in range(n_cats):
            p_k = c_df[f"pct_k_{k}"].values
            b = ax_b.barh(
                y_pos,
                p_k,
                left=left_cum,
                color=spec_colors[k],
                edgecolor="white",
                height=0.65,
                label=spec_labels[k] if idx == 0 else "",
            )
            left_cum += p_k
            if idx == 0:
                bars.append(b)

        for y_i, m_val in enumerate(means):
            ax_b.text(
                102,
                y_i,
                f"μ={m_val:.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#333333",
            )

        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(["Tier 4", "Tier 3", "Tier 2", "Tier 1"], fontsize=9.0)
        ax_b.set_title(
            names_map.get(name, name), fontsize=10.5, fontweight="normal", pad=4
        )
        ax_b.set_xlim(0, 122)
        ax_b.set_xticks([0, 25, 50, 75, 100])
        ax_b.grid(False, which="both")
        if row_i == nrows - 1:
            ax_b.set_xlabel("Composition (%)", fontsize=9.5)

    fig.legend(
        bars,
        spec_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=min(6, n_cats),
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Tiered consensus profiles plot saved to {save_path}")
    plt.show()
    plt.close(fig)


def plot_tier_pairwise_heatmaps(
    tier_pairwise_dict: dict[int, pd.DataFrame],
    save_path: Path,
    r_phys_mpc_h: float = 0.5,
    *,
    max_delta_z: float,
    catalog_order: list[str] | None = None,
):
    """Plot 2x2 grid of pairwise match fractions across proxy tiers."""
    from matplotlib.colors import Normalize

    bin_titles = [
        "Tier 1: Ranks 1–255 (Highest Proxy / Richness)",
        "Tier 2: Ranks 256–510 (Upper-Mid Proxy)",
        "Tier 3: Ranks 511–765 (Lower-Mid Proxy)",
        "Tier 4: Ranks 766–1020 (Lowest Proxy in Top 1020)",
    ]

    labels = list(tier_pairwise_dict[0].index)
    n = len(labels)
    fig_w = max(13.0, 0.85 * n + 2.5)
    fig_h = max(11.0, 0.75 * n + 2.0)
    label_fs = 9.0 if n <= 8 else (8.0 if n <= 12 else 7.0)
    cell_fs = 9.0 if n <= 8 else (7.5 if n <= 12 else 6.5)

    mask = np.zeros((n, n), dtype=bool)
    if catalog_order and len(catalog_order) == n:
        for i in range(n):
            for j in range(n):
                mask[i, j] = is_homologous_pair(catalog_order[i], catalog_order[j])
    else:
        mask = np.eye(n, dtype=bool)

    cross_vals = []
    for b_idx in range(4):
        d = tier_pairwise_dict[b_idx].values
        cross_vals.extend(d[~mask])
    cross_vals = np.array(cross_vals)
    if len(cross_vals) > 0:
        vmin = max(0.0, float(np.floor(cross_vals.min() / 5.0) * 5.0))
        vmax = min(100.0, float(np.ceil(cross_vals.max() / 5.0) * 5.0))
    else:
        vmin, vmax = 20.0, 75.0

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_bad(color="#E5E7EB")

    for b_idx, ax in enumerate(axes.flat):
        df_mat = tier_pairwise_dict[b_idx]
        data = df_mat.values

        data_masked = np.ma.masked_array(data, mask=mask)
        im = ax.imshow(data_masked, cmap=cmap, norm=norm, aspect="equal")
        ax.set_title(
            f"({chr(97 + b_idx)}) {bin_titles[b_idx]}",
            fontsize=10.5,
            fontweight="normal",
            pad=8,
        )
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=label_fs)
        ax.set_yticklabels(labels, fontsize=label_fs)

        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(False, which="both")
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.spines[:].set_visible(False)

        for i in range(n):
            for j in range(n):
                val = data[i, j]
                if mask[i, j]:
                    txt = f"{val:.0f}%" if i != j else "100%"
                    text_col = "#4B5563"
                else:
                    txt = f"{val:.0f}%"
                    norm_val = norm(val)
                    text_col = "white" if norm_val > 0.60 else "black"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color=text_col,
                    fontsize=cell_fs,
                    fontweight="normal",
                )

    fig.subplots_adjust(right=0.88, hspace=0.22, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(
        "Fraction of Row Tier Matched in Column Full Top 1020 (%)",
        fontsize=10.5,
        fontweight="normal",
    )

    fig.suptitle(
        f"Tier-Resolved Pairwise Lens Matching Fractions (R < {r_phys_mpc_h:g} Mpc/h, |Δz| < {max_delta_z:g})\n"
        "Row: Clusters in Given Proxy Tier  |  Column: Matched in Full Stratified Top 1020 of Target Catalog",
        fontsize=12.0,
        fontweight="normal",
        y=0.97,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Tier-resolved pairwise heatmaps saved to {save_path}")
    plt.show()
    plt.close(fig)


def plot_redshift_distributions(
    raw_dfs: dict[str, Table],
    strat_dfs: dict[str, Table],
    colors: list[str],
    save_path: Path,
    display_names: dict[str, str] | None = None,
    redshift_range: tuple[float, float] = (0.19, 0.52),
    n_bins: int = 10,
):
    """Plot redshift distribution comparison between raw top candidates and stratified samples.

    Uses Gaussian KDE for smooth density curves, with underlying histogram shown as
    semi-transparent stepped bars for reference.
    """
    names_map = display_names or {}
    catalog_names = list(strat_dfs.keys())
    z_edges = np.linspace(redshift_range[0], redshift_range[1], n_bins + 1)
    delta_z = z_edges[1] - z_edges[0]
    z_grid = np.linspace(redshift_range[0] - 0.02, redshift_range[1] + 0.02, 256)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.0), sharey=True)

    # Panel 1: Raw Top 1020
    for idx, name in enumerate(catalog_names):
        z_vals = np.asarray(raw_dfs[name]["z"], dtype=float)
        n_total = len(z_vals)
        z_mean = float(np.mean(z_vals))
        c = colors[idx % len(colors)]
        disp_name = names_map.get(name, name)

        # Histogram as faint stepped background
        counts, _ = np.histogram(z_vals, bins=z_edges)
        ax1.stairs(counts, z_edges, color=c, alpha=0.15, fill=True, linewidth=0)

        # KDE curve scaled to cluster counts
        _, density = gaussian_kde_1d(np, z_vals, grid=z_grid)
        kde_counts = density * n_total * delta_z
        ax1.plot(
            z_grid,
            kde_counts,
            linewidth=2.0,
            color=c,
            label=f"{disp_name} (mean z={z_mean:.3f})",
            alpha=0.9,
        )

    ax1.set_title(
        "(a) Raw Top 1020 (Unstratified)",
        fontsize=11.5,
        pad=10,
        fontweight="normal",
    )
    ax1.set_xlabel("Redshift z", fontsize=11.0)
    ax1.set_ylabel(f"Cluster Count (per $\\Delta z = {delta_z:.3f}$)", fontsize=11.0)
    ax1.set_xlim(redshift_range[0] - 0.01, redshift_range[1] + 0.01)
    ax1.grid(False, which="both")
    ax1.legend(fontsize=8.0, loc="upper right", framealpha=0.9)

    # Panel 2: Stratified Top 1020 (Matched redMaPPer N(z))
    for idx, name in enumerate(catalog_names):
        z_vals = np.asarray(strat_dfs[name]["z"], dtype=float)
        n_total = len(z_vals)
        z_mean = float(np.mean(z_vals))
        c = colors[idx % len(colors)]
        disp_name = names_map.get(name, name)
        is_ref = "redm" in name

        # Histogram as faint stepped background
        counts, _ = np.histogram(z_vals, bins=z_edges)
        ax2.stairs(counts, z_edges, color=c, alpha=0.15, fill=True, linewidth=0)

        # KDE curve scaled to cluster counts
        _, density = gaussian_kde_1d(np, z_vals, grid=z_grid)
        kde_counts = density * n_total * delta_z
        ax2.plot(
            z_grid,
            kde_counts,
            linewidth=2.5 if is_ref else 1.5,
            linestyle="--" if is_ref else "-",
            color=c,
            label=f"{disp_name} (mean z={z_mean:.3f})",
            alpha=0.95 if is_ref else 0.8,
        )

    ax2.set_title(
        "(b) Stratified (Matched redMaPPer N(z))",
        fontsize=11.5,
        pad=10,
        fontweight="normal",
    )
    ax2.set_xlabel("Redshift z", fontsize=11.0)
    ax2.set_xlim(redshift_range[0] - 0.01, redshift_range[1] + 0.01)
    ax2.grid(False, which="both")
    ax2.legend(fontsize=8.0, loc="upper right", framealpha=0.9)

    fig.suptitle(
        f"Lens Catalog Redshift Distribution Comparison (N=1020 per catalog, {redshift_range[0]:.2f} ≤ z ≤ {redshift_range[1]:.2f})\n"
        "Raw proxy-selected catalogs exhibit large redshift skew (AMICO low-z, r-z diff high-z); stratification equalizes N(z)",
        fontsize=12.0,
        fontweight="normal",
        y=1.02,
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Redshift distributions comparison plot saved to {save_path}")
    plt.show()
    plt.close(fig)


# %% Global Configuration

LABELS_TO_COMPARE = [
    # External benchmarks (SDSS & DESI Legacy Surveys)
    "camira_1bin",
    "redm_r16_1bin",
    "wh24_1bin",
    "zou21_1bin",
    "yang21_mass_1bin",
    "yang21_richness_1bin",
    "clumpr_mass_1bin",
    "clumpr_richness_1bin",
    # HSC reproduced / pipeline candidates
    "amico_1bin",
    "rz_diff_preset_1bin",
    "rz_diff_1bin",
    "rz_diff_two_box_match_recovery_1bin",
    "rz_diff_single_box_1bin",
    "rz_diff_single_box_match_recovery_1bin",
    "rz_diff_two_box_red_1bin",
    "rz_diff_two_box_red_match_recovery_1bin",
    "rz_diff_two_box_red_redmapper_matching_1bin",
    "rz_diff_two_box_red_camira_matching_1bin",
    "rz_diff_six_param_1bin",
    "rz_diff_six_param_match_recovery_1bin",
]

DISPLAY_NAMES = {
    "camira_1bin": "CAMIRA",
    "redm_r16_1bin": "redMaPPer R16",
    "wh24_1bin": "WH24",
    "zou21_1bin": "Zou21",
    "yang21_mass_1bin": "Yang21 (Halo Mass)",
    "yang21_richness_1bin": "Yang21 (Richness)",
    "clumpr_mass_1bin": "CluMPR (Mass)",
    "clumpr_richness_1bin": "CluMPR (Richness)",
    "amico_1bin": "AMICO",
    "rz_diff_1bin": "r-z Diff (Two Box · WL)",
    "rz_diff_preset_1bin": "r-z Diff (Preset)",
    "rz_diff_single_box_1bin": "r-z Diff (Single Box · WL)",
    "rz_diff_two_box_red_1bin": "r-z Diff (Two Box Red · WL)",
    "rz_diff_two_box_match_recovery_1bin": "r-z Diff (Two Box · Match)",
    "rz_diff_single_box_match_recovery_1bin": "r-z Diff (Single Box · Match)",
    "rz_diff_two_box_red_match_recovery_1bin": "r-z Diff (Two Box Red · Match)",
    "rz_diff_two_box_red_redmapper_matching_1bin": "r-z Diff (Two Box Red · redMaPPer Match)",
    "rz_diff_two_box_red_camira_matching_1bin": "r-z Diff (Two Box Red · CAMIRA Match)",
    "rz_diff_six_param_1bin": "r-z Diff (Six Param · WL)",
    "rz_diff_six_param_match_recovery_1bin": "r-z Diff (Six Param · Match)",
}

PALETTE = [
    "#EE6677",  # Red (CAMIRA)
    "#4477AA",  # Blue (redMaPPer R16)
    "#228833",  # Dark Green (WH24)
    "#CCBB44",  # Yellow-Olive (Zou21)
    "#66CCEE",  # Cyan (Yang21 Halo Mass)
    "#332288",  # Indigo (Yang21 Richness)
    "#AA3377",  # Purple (CluMPR Mass)
    "#EE8866",  # Coral (CluMPR Richness)
    "#10B981",  # Emerald (AMICO)
    "#D55E00",  # Preset
    "#E69F00",  # Two Box WL
    "#7B2CBF",  # Two Box Match
    "#009E73",  # Single Box WL
    "#A16207",  # Single Box Match
    "#B91C1C",  # Two Box Red WL
    "#0F766E",  # Two Box Red Match
    "#CC79A7",  # Two Box Red redMaPPer Match
    "#56B4E9",  # Two Box Red CAMIRA Match
    "#2563EB",  # Six Param WL
    "#334155",  # Six Param Match
]

MARKERS = [
    "s",
    "x",
    "o",
    "v",
    "^",
    "D",
    "p",
    "h",
    "*",
    "<",
    "8",
    "X",
    ">",
    "d",
    "P",
    "+",
    "s",
    "^",
    "o",
    "H",
]

REDSHIFT_RANGE = (0.19, 0.52)
N_STRATIFIED_BINS = 10
TOTAL_STRATIFIED_TOP_N = 1020
MATCH_RADIUS_MPC_H = 0.5
MATCH_MAX_DELTA_Z = 0.05

OUTPUT_REDSHIFT_DISTRIBUTIONS = (
    project_root / "output/plots_for_agents/matching_redshift_distributions.png"
)
OUTPUT_MATCH_HEATMAP = project_root / "output/plots_for_agents/matching_statistics.png"
OUTPUT_CONSENSUS_BREAKDOWN = (
    project_root / "output/plots_for_agents/consensus_breakdown.png"
)
OUTPUT_TIER_CONSENSUS_PROFILES = (
    project_root / "output/plots_for_agents/tier_consensus_profiles.png"
)
OUTPUT_TIER_PAIRWISE_HEATMAPS = (
    project_root / "output/plots_for_agents/tier_pairwise_heatmaps.png"
)


# %% [Stage 1: Load Raw & Stratified Lens Catalogs]

raw_dfs_dict = load_lens_data(LABELS_TO_COMPARE, project_root)

strat_dfs_dict = load_stratified_candidates(
    candidate_keys=LABELS_TO_COMPARE,
    root=project_root,
    redshift_range=REDSHIFT_RANGE,
    n_bins=N_STRATIFIED_BINS,
    total_top_n=TOTAL_STRATIFIED_TOP_N,
    ref_lens_table=raw_dfs_dict.get("redm_r16_1bin"),
    include_all_candidates=False,
)

print(
    f"\nLoaded {len(LABELS_TO_COMPARE)} catalogs (Raw and Stratified N={TOTAL_STRATIFIED_TOP_N})."
)
for k in LABELS_TO_COMPARE:
    z_raw = np.asarray(raw_dfs_dict[k]["z"], float)
    z_strat = np.asarray(strat_dfs_dict[k]["z"], float)
    disp = DISPLAY_NAMES.get(k, k)
    print(
        f"  {disp:24s}: Raw mean z = {np.mean(z_raw):.4f} -> Stratified mean z = {np.mean(z_strat):.4f}"
    )


# %% [Stage 2: Redshift Distribution Validation (Raw vs Stratified)]

plot_redshift_distributions(
    raw_dfs=raw_dfs_dict,
    strat_dfs=strat_dfs_dict,
    colors=PALETTE,
    save_path=OUTPUT_REDSHIFT_DISTRIBUTIONS,
    display_names=DISPLAY_NAMES,
    redshift_range=REDSHIFT_RANGE,
    n_bins=N_STRATIFIED_BINS,
)


# %% [Stage 3: Pairwise Matching Heatmap (Stratified Redshift-Controlled)]

match_df = compute_pairwise_matches(
    strat_dfs_dict, r_phys_mpc_h=MATCH_RADIUS_MPC_H, max_delta_z=MATCH_MAX_DELTA_Z
)

plot_matching_heatmap(
    match_df,
    save_path=OUTPUT_MATCH_HEATMAP,
    display_names=DISPLAY_NAMES,
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
    max_delta_z=MATCH_MAX_DELTA_Z,
)


# %% [Stage 4: Overall Consensus Breakdown Analysis]

consensus_counts_df, consensus_pct_df = compute_consensus_breakdown(
    strat_dfs_dict, r_phys_mpc_h=MATCH_RADIUS_MPC_H, max_delta_z=MATCH_MAX_DELTA_Z
)

plot_consensus_breakdown(
    consensus_counts_df,
    consensus_pct_df,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_CONSENSUS_BREAKDOWN,
    display_names=DISPLAY_NAMES,
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
    max_delta_z=MATCH_MAX_DELTA_Z,
)


# %% [Stage 5: Tiered Proxy Consensus Analysis (4 Quartiles)]

tier_consensus_df = compute_tier_consensus_breakdown(
    strat_dfs_dict,
    n_bins=4,
    display_names=DISPLAY_NAMES,
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
    max_delta_z=MATCH_MAX_DELTA_Z,
)

plot_tier_consensus_profiles(
    tier_consensus_df,
    catalog_order=LABELS_TO_COMPARE,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_TIER_CONSENSUS_PROFILES,
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
    max_delta_z=MATCH_MAX_DELTA_Z,
    display_names=DISPLAY_NAMES,
)


# %% [Stage 6: Tier-Resolved Pairwise Matching Heatmaps]

tier_pairwise_dict = compute_tier_pairwise_matches(
    strat_dfs_dict,
    n_bins=4,
    display_names=DISPLAY_NAMES,
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
    max_delta_z=MATCH_MAX_DELTA_Z,
)
plot_tier_pairwise_heatmaps(
    tier_pairwise_dict,
    save_path=OUTPUT_TIER_PAIRWISE_HEATMAPS,
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
    max_delta_z=MATCH_MAX_DELTA_Z,
    catalog_order=LABELS_TO_COMPARE,
)
