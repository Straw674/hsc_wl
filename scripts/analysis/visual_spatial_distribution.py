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

# %% Local Functions


def load_chen2024_clusters(
    root: Path,
    ra_range: tuple[float, float] = (200.0, 250.0),
    dec_range: tuple[float, float] = (42.0, 44.5),
    redshift_range: tuple[float, float] = (0.19, 0.52),
) -> Table:
    """Load and filter the Chen et al. (2024) WL shear-selected cluster catalog.

    Applies HectoMAP sky footprint, redshift range, and HSC Y3 shape catalog mask.

    Parameters
    ----------
    root : Path
        Project root path.
    ra_range : tuple of (float, float), default (200.0, 250.0)
        RA bounds in degrees.
    dec_range : tuple of (float, float), default (42.0, 44.5)
        Dec bounds in degrees.
    redshift_range : tuple of (float, float), default (0.19, 0.52)
        Redshift bounds.

    Returns
    -------
    astropy.table.Table
        Filtered Chen+2024 cluster table with columns [peak_id, ra, dec, z, snr, ...].
    """
    from hsc_wl.coverage import filter_lens_by_mask

    parquet_path = root / "data/chen2024_shear_selected_clusters.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Chen 2024 catalog not found at {parquet_path}. "
            "Ensure data/chen2024_shear_selected_clusters.parquet exists."
        )

    df = pd.read_parquet(parquet_path)

    # Filter by spatial box and redshift
    mask = (
        (df["ra"] >= ra_range[0])
        & (df["ra"] <= ra_range[1])
        & (df["dec"] >= dec_range[0])
        & (df["dec"] <= dec_range[1])
        & (df["z_cl"] >= redshift_range[0])
        & (df["z_cl"] <= redshift_range[1])
    )
    filtered_df = df[mask].sort_values(by="snr", ascending=False).reset_index(drop=True)
    tbl = Table.from_pandas(filtered_df)

    # Filter by Y3 shape mask
    tbl = filter_lens_by_mask(tbl, root=root, ra_col="ra", dec_col="dec")
    tbl["z"] = tbl["z_cl"]
    tbl["rank"] = np.arange(1, len(tbl) + 1)
    return tbl


def load_lens_data(labels: list[str | tuple], root: Path) -> dict[str, Table]:
    """Load prepared lens tables for the given configurations.

    Parameters
    ----------
    labels : list of str or tuple
        Run labels to load, e.g. ["redm_s16a_hectomap_1bin", ...] or
        [("redm_s16a_hectomap", "1bin", ...), ...].
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
            # Fallback to standard convention if not found in registry
            catalog_id, nbins = label.rsplit("_", 1)
            file_path = (
                root / f"output/{catalog_id}/{nbins}/prepare/{label}_lenses.fits"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"Prepared lens catalog not found at {file_path}")

        print(f"Loading prepared lenses from {file_path.name}...")
        tbl = Table.read(file_path)
        # Add rank column based on order (the preparation pipeline sorts them by richness/mass)
        tbl["rank"] = np.arange(1, len(tbl) + 1)
        dfs[label] = tbl

    return dfs


def compute_pairwise_matches(dfs: dict[str, Table]) -> pd.DataFrame:
    """Compute pairwise matching statistics within 0.5 Mpc/h physical radius.

    Parameters
    ----------
    dfs : dict of label -> Table
        Loaded lens catalogs.

    Returns
    -------
    pd.DataFrame
        Table of pairwise match counts.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18

    catalog_names = list(dfs.keys())
    n_cats = len(catalog_names)
    matrix = np.zeros((n_cats, n_cats), dtype=int)

    coords = {
        name: SkyCoord(
            ra=np.asarray(dfs[name]["ra"]) * u.deg,
            dec=np.asarray(dfs[name]["dec"]) * u.deg,
        )
        for name in catalog_names
    }

    h = Planck18.h

    for i in range(n_cats):
        for j in range(n_cats):
            if i == j:
                matrix[i, j] = len(dfs[catalog_names[i]])
                continue

            c1 = coords[catalog_names[i]]
            c2 = coords[catalog_names[j]]

            # Match each object in c1 to the nearest neighbor in c2
            idx, d2d, _ = c1.match_to_catalog_sky(c2)

            # Compute matching radius for each object in c1 based on its redshift
            z1 = np.clip(
                np.asarray(dfs[catalog_names[i]]["z"], dtype=float), 1e-4, None
            )
            da1 = Planck18.angular_diameter_distance(
                z1
            ).value  # angular diameter distance in Mpc

            # 0.5 Mpc/h in degrees: (0.5 / h) / da1 * (180 / pi)
            match_radius_deg = (0.5 / h) / da1 * (180.0 / np.pi)

            # Compare distance in degrees
            matched = d2d.deg < match_radius_deg
            matrix[i, j] = np.sum(matched)

    df_match = pd.DataFrame(matrix, index=catalog_names, columns=catalog_names)
    return df_match


def plot_matching_heatmap(df_match: pd.DataFrame, save_path: Path):
    """Plot pairwise matching statistics as a heatmap grid using matplotlib.

    Applies histogram equalization stretch (HistEqStretch) to emphasize contrast
    among matching fractions across catalogs.

    Saves results as a PNG file and displays the image.
    """
    import matplotlib.pyplot as plt
    from astropy.visualization import HistEqStretch, ImageNormalize

    fig, ax = plt.subplots(figsize=(11, 9))

    data = df_match.values
    labels = list(df_match.index)
    n = len(labels)

    # Calculate row-normalized match fraction (percentage)
    # Row i is the source catalog, cell (i, j) is the fraction of row i matched to column j
    row_totals = np.diag(data)
    # Avoid division by zero just in case
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    data_pct = (data / row_totals_safe[:, None]) * 100.0

    # Equalized histogram stretch to make matching rate differences pronounced
    stretch = HistEqStretch(data_pct)
    norm = ImageNormalize(vmin=float(data_pct.min()), vmax=100.0, stretch=stretch)

    # Use a sequential colormap ('YlGnBu') representing matching percentage
    im = ax.imshow(data_pct, cmap="YlGnBu", aspect="equal", norm=norm)

    # Dynamic non-overlapping colorbar ticks mapped from equalized space
    norm_positions = np.linspace(0.05, 0.95, 6)
    tick_vals = norm.inverse(norm_positions)
    ticks = sorted(
        list(set(int(round(t)) for t in tick_vals)) + [int(round(data_pct.min())), 100]
    )
    cbar = fig.colorbar(im, ax=ax, ticks=ticks, shrink=0.8)
    cbar.set_label("Match Fraction (%) [Equalized Hist Stretch]", fontsize=11)

    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    # Adjust ticks parameter and remove spines for a clean grid look
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.spines[:].set_visible(False)

    # Create white grid boundaries between cells
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate matching count and percentage inside cells
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            pct = data_pct[i, j]
            total = row_totals[i]

            # Text layout: Count / Total \n (Pct%)
            if i == j:
                text = f"{val}\n(100.0%)"
            else:
                text = f"{val}/{total}\n({pct:.1f}%)"

            # Contrasting text color based on normalized cell brightness
            norm_val = float(norm(np.array([pct]))[0])
            text_color = "white" if norm_val > 0.60 else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=8.5,
            )

    ax.set_title(
        "Pairwise Lens Match Fractions (0.5 Mpc/h Physical Radius)\nRow Normalized: Fraction of Row Catalog Matched in Column Catalog [HistEq Stretch]",
        fontsize=12,
        pad=18,
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Matching heatmap saved to {save_path}")
    plt.show()
    plt.close(fig)


def compute_consensus_breakdown(
    dfs: dict[str, Table],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute multi-catalog consensus counts and percentages.

    For each catalog, counts how many objects are matched in exactly k other
    catalogs (k = 0, 1, ..., n_cats - 1) within 0.5 Mpc/h physical radius.

    Parameters
    ----------
    dfs : dict of str -> Table
        Loaded lens catalogs.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        df_counts: Table of raw counts.
        df_pct: Table of percentages relative to each catalog's total.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18

    catalog_names = list(dfs.keys())
    n_cats = len(catalog_names)

    coords = {
        name: SkyCoord(
            ra=np.asarray(dfs[name]["ra"]) * u.deg,
            dec=np.asarray(dfs[name]["dec"]) * u.deg,
        )
        for name in catalog_names
    }

    radii_deg = {}
    h = Planck18.h
    for name in catalog_names:
        z = np.clip(np.asarray(dfs[name]["z"], float), 1e-4, None)
        da = Planck18.angular_diameter_distance(z).value
        radii_deg[name] = (0.5 / h) / da * (180.0 / np.pi)

    counts_matrix = np.zeros((n_cats, n_cats), dtype=int)

    for i, name in enumerate(catalog_names):
        c_self = coords[name]
        r_self = radii_deg[name]
        n_obj = len(dfs[name])

        matched_counts = np.zeros(n_obj, dtype=int)
        for j, other_name in enumerate(catalog_names):
            if i == j:
                continue
            c_other = coords[other_name]
            idx, d2d, _ = c_self.match_to_catalog_sky(c_other)
            matched_counts += (d2d.deg < r_self).astype(int)

        for k in range(n_cats):
            counts_matrix[i, k] = np.sum(matched_counts == k)

    col_labels = [
        f"{k} Other Catalogs" if k == 1 else f"{k} Other Catalogs"
        for k in range(n_cats)
    ]
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
):
    """Plot consensus level profiles across catalogs as a clean multi-line plot.

    Shows the fraction of clusters in each catalog that match k other catalogs
    (k = 0, 1, ..., n_cats - 1) within 0.5 Mpc/h physical radius.
    """
    import matplotlib.pyplot as plt

    pct_matrix = df_pct.values
    catalog_names = list(df_pct.index)
    n_cats = len(catalog_names)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x_vals = np.arange(n_cats)

    for idx, name in enumerate(catalog_names):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        lw = 2.4 if "amico" in name else 1.8
        alpha = 1.0 if "amico" in name else 0.85
        zorder = 5 if "amico" in name else 3
        ax.plot(
            x_vals,
            pct_matrix[idx],
            label=name,
            color=c,
            marker=m,
            markersize=8,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
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
        fontsize=11.5,
        labelpad=10,
    )
    ax.set_ylabel("Cluster Fraction (%)", fontsize=11.5)
    ax.set_title(
        "Consensus Profiles across Catalogs (0.5 Mpc/h Matching)",
        fontsize=12.5,
        pad=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(45.0, float(np.max(pct_matrix)) + 6.0))
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.9)

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
) -> pd.DataFrame:
    """Compute consensus statistics for each catalog partitioned into proxy rank tiers.

    Partitions each catalog's top objects (already sorted descending by proxy/richness)
    into ``n_bins`` (default 4, quartiles) rank bins. For every cluster in a tier,
    matches against the full top 100 of all other catalogs within 0.5 Mpc/h physical radius.

    Parameters
    ----------
    dfs : dict of str -> Table
        Loaded lens catalogs.
    n_bins : int, default 4
        Number of equal-frequency rank tiers per catalog.
    display_names : dict of str -> str, optional
        Human-readable catalog labels for display.

    Returns
    -------
    pd.DataFrame
        Table containing tier summary metrics (mean matches, SEM, solo rate, etc.).
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18

    catalog_names = list(dfs.keys())
    n_cats = len(catalog_names)
    names_map = display_names or {}

    coords = {
        name: SkyCoord(
            ra=np.asarray(dfs[name]["ra"], dtype=float) * u.deg,
            dec=np.asarray(dfs[name]["dec"], dtype=float) * u.deg,
        )
        for name in catalog_names
    }

    radii_deg = {}
    h = Planck18.h
    for name in catalog_names:
        z = np.clip(np.asarray(dfs[name]["z"], float), 1e-4, None)
        da = Planck18.angular_diameter_distance(z).value
        radii_deg[name] = (0.5 / h) / da * (180.0 / np.pi)

    records = []

    for i, name_i in enumerate(catalog_names):
        c_i = coords[name_i]
        r_i = radii_deg[name_i]
        n_i = len(dfs[name_i])

        # Match each cluster in catalog i against the full top 100 of all other catalogs
        matched_counts = np.zeros(n_i, dtype=int)
        for j, name_j in enumerate(catalog_names):
            if i == j:
                continue
            c_j = coords[name_j]
            idx, d2d, _ = c_i.match_to_catalog_sky(c_j)
            matched_counts += (d2d.deg < r_i).astype(int)

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
            pct_low = float(np.mean((sub_k >= 1) & (sub_k <= 2)) * 100.0)
            pct_med = float(np.mean((sub_k >= 3) & (sub_k <= 4)) * 100.0)
            pct_high = float(np.mean(sub_k >= 5) * 100.0)
            pct_ge4 = float(np.mean(sub_k >= 4) * 100.0)

            rec = {
                "catalog": name_i,
                "display_name": names_map.get(name_i, name_i),
                "bin_idx": b_idx,
                "bin_label": f"Bin {b_idx + 1}\n(Ranks {r_start}–{r_end})",
                "tier_name": f"Bin {b_idx + 1}",
                "n_clusters": n_sub,
                "rank_range": (r_start, r_end),
                "mean_matches": mean_val,
                "sem_matches": sem_val,
                "median_matches": median_val,
                "pct_solo": pct_solo,
                "pct_low": pct_low,
                "pct_med": pct_med,
                "pct_high": pct_high,
                "pct_ge4": pct_ge4,
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
) -> dict[int, pd.DataFrame]:
    """Compute pairwise match matrices for each proxy tier against full top 100.

    Parameters
    ----------
    dfs : dict of str -> Table
        Loaded lens catalogs.
    n_bins : int, default 4
        Number of proxy rank bins.
    display_names : dict of str -> str, optional
        Human-readable catalog labels.

    Returns
    -------
    dict of int -> pd.DataFrame
        Mapping bin_index (0..n_bins-1) to an N_cats x N_cats match percentage DataFrame.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18

    catalog_names = list(dfs.keys())
    n_cats = len(catalog_names)
    names_map = display_names or {}

    coords = {
        name: SkyCoord(
            ra=np.asarray(dfs[name]["ra"], dtype=float) * u.deg,
            dec=np.asarray(dfs[name]["dec"], dtype=float) * u.deg,
        )
        for name in catalog_names
    }

    radii_deg = {}
    h = Planck18.h
    for name in catalog_names:
        z = np.clip(np.asarray(dfs[name]["z"], float), 1e-4, None)
        da = Planck18.angular_diameter_distance(z).value
        radii_deg[name] = (0.5 / h) / da * (180.0 / np.pi)

    tier_matrices = {}

    for b_idx in range(n_bins):
        mat = np.zeros((n_cats, n_cats), dtype=float)
        for i, name_i in enumerate(catalog_names):
            n_i = len(dfs[name_i])
            bin_splits = np.array_split(np.arange(n_i), n_bins)
            idx_slice = bin_splits[b_idx]
            n_sub = len(idx_slice)

            c_sub = coords[name_i][idx_slice]
            r_sub = radii_deg[name_i][idx_slice]

            for j, name_j in enumerate(catalog_names):
                if i == j:
                    mat[i, j] = 100.0
                    continue
                c_j = coords[name_j]
                idx, d2d, _ = c_sub.match_to_catalog_sky(c_j)
                matched = np.sum(d2d.deg < r_sub)
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
):
    """Plot multi-panel tiered consensus breakdown figure.

    Panel (a): Consensus retention curve across proxy tiers (mean external matches & solo rate).
    Panel (b): Heatmap matrix of mean consensus scores and high-consensus rates (Catalog x Tier).
    Panel (c): Small-multiples stacked horizontal bars showing full consensus composition per tier.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    names_map = display_names or {}
    n_cats = len(catalog_order)

    fig = plt.figure(figsize=(19, 13))
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1.15, 1.0],
        width_ratios=[1.15, 0.95],
        hspace=0.32,
        wspace=0.25,
    )

    # --- Top-Left: Retention Curves (Subplots for Mean Matches & Solo Rate) ---
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 0], height_ratios=[1.4, 1.0], hspace=0.22
    )
    ax_mean = fig.add_subplot(gs_left[0, 0])
    ax_solo = fig.add_subplot(gs_left[1, 0], sharex=ax_mean)

    x_bins = np.arange(4)
    bin_labels = [
        "Bin 1 (Q1)\nRanks 1–25",
        "Bin 2 (Q2)\nRanks 26–50",
        "Bin 3 (Q3)\nRanks 51–75",
        "Bin 4 (Q4)\nRanks 76–100",
    ]

    for idx, name in enumerate(catalog_order):
        c_df = tier_df[tier_df["catalog"] == name].sort_values("bin_idx")
        d_name = names_map.get(name, name)
        col = colors[idx % len(colors)]
        mrk = markers[idx % len(markers)]
        lw = 2.4 if "redm_pdr3" in name or "amico" in name else 1.8

        ax_mean.errorbar(
            x_bins,
            c_df["mean_matches"],
            yerr=c_df["sem_matches"],
            label=d_name,
            color=col,
            marker=mrk,
            markersize=8,
            linewidth=lw,
            capsize=4,
            capthick=1.2,
            alpha=0.9,
        )
        ax_solo.plot(
            x_bins,
            c_df["pct_solo"],
            label=d_name,
            color=col,
            marker=mrk,
            markersize=7,
            linewidth=lw,
            alpha=0.9,
        )

    ax_mean.set_ylabel(
        "Mean External Matches (out of 7)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax_mean.set_title(
        "(a) Consensus Retention Curve across Proxy Tiers (0.5 Mpc/h)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_mean.grid(True, linestyle="--", alpha=0.5)
    ax_mean.set_ylim(0.5, 6.2)
    ax_mean.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.95)
    plt.setp(ax_mean.get_xticklabels(), visible=False)

    ax_solo.set_ylabel(
        "Solo / Unique Rate (% k=0)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax_solo.set_xlabel(
        "Proxy / Score Tier (Rank Ordered)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax_solo.set_xticks(x_bins)
    ax_solo.set_xticklabels(bin_labels, fontsize=10)
    ax_solo.grid(True, linestyle="--", alpha=0.5)
    ax_solo.set_ylim(-2, 70)

    # --- Top-Right: Heatmap Matrix (Catalogs x Tiers) ---
    ax_heat = fig.add_subplot(gs[0, 1])
    mat_mean = np.zeros((n_cats, 4))
    mat_ge4 = np.zeros((n_cats, 4))
    mat_solo = np.zeros((n_cats, 4))

    for idx, name in enumerate(catalog_order):
        c_df = tier_df[tier_df["catalog"] == name].sort_values("bin_idx")
        mat_mean[idx] = c_df["mean_matches"].values
        mat_ge4[idx] = c_df["pct_ge4"].values
        mat_solo[idx] = c_df["pct_solo"].values

    norm_heat = Normalize(vmin=1.0, vmax=6.0)
    im = ax_heat.imshow(mat_mean, cmap="YlGnBu", aspect="auto", norm=norm_heat)

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.03)
    cbar.set_label("Mean Matched Catalogs (out of 7)", fontsize=10.5, fontweight="bold")

    ax_heat.set_xticks(np.arange(4))
    ax_heat.set_xticklabels(
        ["Bin 1 (Q1)", "Bin 2 (Q2)", "Bin 3 (Q3)", "Bin 4 (Q4)"],
        fontsize=10.5,
        fontweight="bold",
    )
    ax_heat.set_yticks(np.arange(n_cats))
    ax_heat.set_yticklabels(
        [names_map.get(name, name) for name in catalog_order],
        fontsize=10.5,
        fontweight="bold",
    )
    ax_heat.set_title(
        "(b) Mean Consensus Score by Tier (Catalog × Proxy Bin)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Annotate cells
    for i in range(n_cats):
        for j in range(4):
            val = mat_mean[i, j]
            ge4_val = mat_ge4[i, j]
            solo_val = mat_solo[i, j]
            text_color = "white" if val > 3.8 else "black"
            txt = f"{val:.2f}\n({ge4_val:.0f}% ≥4)"
            if solo_val > 0:
                txt += f"\n[{solo_val:.0f}% solo]"
            ax_heat.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold",
            )

    ax_heat.set_xticks(np.arange(5) - 0.5, minor=True)
    ax_heat.set_yticks(np.arange(n_cats + 1) - 0.5, minor=True)
    ax_heat.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    # --- Bottom: Small Multiples Stacked Consensus Spectrum ---
    gs_spec = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=gs[1, :], hspace=0.38, wspace=0.22
    )
    spec_colors = ["#d73027", "#fdae61", "#a6d96a", "#313695"]
    spec_labels = ["Solo (k=0)", "Low (k=1–2)", "Moderate (k=3–4)", "High (k=5–7)"]

    for idx, name in enumerate(catalog_order):
        ax_b = fig.add_subplot(gs_spec[idx // 4, idx % 4])
        c_df = tier_df[tier_df["catalog"] == name].sort_values(
            "bin_idx", ascending=False
        )

        y_pos = np.arange(4)
        p_solo = c_df["pct_solo"].values
        p_low = c_df["pct_low"].values
        p_med = c_df["pct_med"].values
        p_high = c_df["pct_high"].values
        means = c_df["mean_matches"].values

        b1 = ax_b.barh(
            y_pos,
            p_solo,
            color=spec_colors[0],
            edgecolor="white",
            height=0.65,
            label=spec_labels[0] if idx == 0 else "",
        )
        b2 = ax_b.barh(
            y_pos,
            p_low,
            left=p_solo,
            color=spec_colors[1],
            edgecolor="white",
            height=0.65,
            label=spec_labels[1] if idx == 0 else "",
        )
        b3 = ax_b.barh(
            y_pos,
            p_med,
            left=p_solo + p_low,
            color=spec_colors[2],
            edgecolor="white",
            height=0.65,
            label=spec_labels[2] if idx == 0 else "",
        )
        b4 = ax_b.barh(
            y_pos,
            p_high,
            left=p_solo + p_low + p_med,
            color=spec_colors[3],
            edgecolor="white",
            height=0.65,
            label=spec_labels[3] if idx == 0 else "",
        )

        for y_i, m_val in enumerate(means):
            ax_b.text(
                102,
                y_i,
                f"μ={m_val:.1f}",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
                color="#333333",
            )

        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(["Bin 4", "Bin 3", "Bin 2", "Bin 1"], fontsize=9)
        ax_b.set_title(
            names_map.get(name, name), fontsize=10.5, fontweight="bold", pad=4
        )
        ax_b.set_xlim(0, 125)
        ax_b.set_xticks([0, 25, 50, 75, 100])
        ax_b.grid(axis="x", linestyle=":", alpha=0.6)
        if idx // 4 == 1:
            ax_b.set_xlabel("Composition (%)", fontsize=9.5)

    fig.legend(
        [b1, b2, b3, b4],
        spec_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=10.5,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Tiered consensus profiles plot saved to {save_path}")
    plt.close(fig)


def plot_tier_pairwise_heatmaps(
    tier_pairwise_dict: dict[int, pd.DataFrame],
    save_path: Path,
):
    """Plot 2x2 grid of pairwise match fractions across proxy tiers."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    bin_titles = [
        "Tier 1: Ranks 1–25 (Highest Proxy / Richness)",
        "Tier 2: Ranks 26–50 (Upper-Mid Proxy)",
        "Tier 3: Ranks 51–75 (Lower-Mid Proxy)",
        "Tier 4: Ranks 76–100 (Lowest Proxy in Top 100)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 15), sharex=True, sharey=True)
    norm = Normalize(vmin=10.0, vmax=100.0)

    for b_idx, ax in enumerate(axes.flat):
        df_mat = tier_pairwise_dict[b_idx]
        data = df_mat.values
        labels = list(df_mat.index)
        n = len(labels)

        im = ax.imshow(data, cmap="YlGnBu", norm=norm, aspect="equal")
        ax.set_title(
            f"({chr(97 + b_idx)}) {bin_titles[b_idx]}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9.5)
        ax.set_yticklabels(labels, fontsize=9.5)

        # White minor grid
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.spines[:].set_visible(False)

        for i in range(n):
            for j in range(n):
                val = data[i, j]
                text_col = "white" if val > 65.0 else "black"
                txt = f"{val:.0f}%" if i != j else "100%"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color=text_col,
                    fontsize=8.5,
                    fontweight="bold",
                )

    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(
        "Fraction of Row Tier Matched in Column Full Top 100 (%)",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        "Tier-Resolved Pairwise Lens Matching Fractions (0.5 Mpc/h Matching Radius)\n"
        "Row: Clusters in Given Proxy Tier of Catalog A  |  Column: Matched in Full Top 100 of Catalog B",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Tier-resolved pairwise heatmaps saved to {save_path}")
    plt.close(fig)


def plot_bokeh_spatial(
    dfs: dict[str, Table],
    colors: list[str],
    markers: list[str],
    save_path: Path,
    chen_table: Table | None = None,
):
    """Generate an interactive Bokeh plot allowing zoom and hover inspection.

    Saves results as a self-contained, screen-filling HTML file.
    """
    from bokeh.models import ColumnDataSource, HoverTool, Range1d
    from bokeh.plotting import figure, output_file, save

    # Prepare output file
    output_file(filename=str(save_path), title="Lens Spatial Distribution Explorer")

    # Determine adaptive coordinates and spans
    all_dfs = list(dfs.values())
    if chen_table is not None and len(chen_table) > 0:
        all_dfs.append(chen_table)

    all_dec = np.concatenate([np.asarray(t["dec"]) for t in all_dfs])
    all_ra = np.concatenate([np.asarray(t["ra"]) for t in all_dfs])

    if len(all_dec) == 0:
        dec_mean = 0.0
        ra_min, ra_max = 0.0, 360.0
        dec_min, dec_max = -90.0, 90.0
    else:
        dec_mean = np.mean(all_dec)
        ra_min, ra_max = np.min(all_ra), np.max(all_ra)
        dec_min, dec_max = np.min(all_dec), np.max(all_dec)

    cos_dec = np.cos(np.radians(dec_mean))

    ra_span = ra_max - ra_min
    dec_span = dec_max - dec_min

    if ra_span == 0:
        ra_span = 1.0
    if dec_span == 0:
        dec_span = 1.0

    ra_padding = ra_span * 0.05
    dec_padding = dec_span * 0.05

    # Coordinates range for Bokeh
    # RA increases to the left (inverted) per astronomical convention
    x_start = ra_max + ra_padding
    x_end = ra_min - ra_padding
    y_start = dec_min - dec_padding
    y_end = dec_max + dec_padding

    # Calculate screen aspect ratio so scales match visually
    ra_span_padded = x_start - x_end
    dec_span_padded = y_end - y_start
    aspect = (ra_span_padded * cos_dec) / dec_span_padded

    # Constrain aspect ratio to reasonable limits
    plot_width = 1200
    if aspect > 3.0:
        plot_height = 400
    elif aspect < 0.3:
        plot_height = 1000
    else:
        plot_height = int(plot_width / aspect)

    p = figure(
        title="Interactive Lens Spatial Distribution Explorer",
        width=plot_width,
        height=plot_height + 80,  # add padding for title/toolbar
        sizing_mode="scale_both",
        match_aspect=True,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_axis_label="Right Ascension (deg)",
        y_axis_label="Declination (deg)",
        toolbar_location="above",
    )

    # Invert x-axis (RA increases to the left)
    p.x_range = Range1d(start=x_start, end=x_end)
    p.y_range = Range1d(start=y_start, end=y_end)

    # Apply premium dark theme styling
    p.background_fill_color = "#1e1e1e"
    p.border_fill_color = "#181818"
    p.grid.grid_line_color = "#3a3a3a"
    p.grid.grid_line_alpha = 0.5
    p.title.text_color = "#ffffff"
    p.title.text_font_size = "14pt"
    p.xaxis.axis_label_text_color = "#cccccc"
    p.yaxis.axis_label_text_color = "#cccccc"
    p.xaxis.major_label_text_color = "#aaaaaa"
    p.yaxis.major_label_text_color = "#aaaaaa"

    # Draw each optical catalog's markers
    for idx, name in enumerate(dfs.keys()):
        tbl = dfs[name]
        color = colors[idx % len(colors)]
        marker_name = markers[idx % len(markers)]

        # Prepare source data
        source = ColumnDataSource(
            data={
                "ra": np.asarray(tbl["ra"], dtype=float),
                "dec": np.asarray(tbl["dec"], dtype=float),
                "z": np.asarray(tbl["z"], dtype=float),
                "rank": np.arange(1, len(tbl) + 1, dtype=int),
                "catalog": [name] * len(tbl),
                "total": [len(tbl)] * len(tbl),
            }
        )

        # Plot based on marker style (use larger hollow markers so overlapping can be seen)
        # Map marker names to Bokeh marker types
        bokeh_marker_map = {
            "o": "circle",
            "s": "square",
            "^": "triangle",
            "v": "inverted_triangle",
            "D": "diamond",
            "d": "diamond",
            "h": "hex",
            "*": "star",
            "x": "x",
            "+": "cross",
        }
        bokeh_marker = bokeh_marker_map.get(marker_name, "circle")
        size = 12 + idx * 3
        line_only = bokeh_marker in {"cross", "x", "plus"}
        renderer = p.scatter(
            x="ra",
            y="dec",
            source=source,
            size=size if not line_only else max(8, size - 3),
            marker=bokeh_marker,
            color=color,
            fill_color=None,
            line_width=2.0 if line_only else 2.5,
            legend_label=f"{name} (N={len(tbl)})",
        )

        # Configure individual hover tool for this renderer
        hover = HoverTool(
            renderers=[renderer],
            tooltips=[
                ("Catalog", "@catalog"),
                ("Rank", "#@rank of @total"),
                ("RA", "@ra{0.0000} deg"),
                ("Dec", "@dec{0.0000} deg"),
                ("Redshift z", "@z{0.0000}"),
            ],
        )
        p.add_tools(hover)

    # Draw Chen 2024 WL shear-selected clusters with an emphasized larger white hollow marker
    if chen_table is not None and len(chen_table) > 0:
        chen_source = ColumnDataSource(
            data={
                "ra": np.asarray(chen_table["ra"], dtype=float),
                "dec": np.asarray(chen_table["dec"], dtype=float),
                "z": np.asarray(chen_table["z"], dtype=float),
                "snr": np.asarray(chen_table["snr"], dtype=float),
                "peak_id": np.asarray(chen_table["peak_id"], dtype=int),
                "richness": np.asarray(chen_table["richness"], dtype=float),
                "opt_name": [str(x) for x in chen_table["opt_name"]],
                "sep_mpc_h": np.asarray(chen_table["sep_mpc_h"], dtype=float),
                "catalog": ["Chen+2024 WL Shear-Selected"] * len(chen_table),
            }
        )

        # Prominent larger white hollow circle (size=26, line_width=3.0) to highlight WL shear-selected clusters
        chen_renderer = p.scatter(
            x="ra",
            y="dec",
            source=chen_source,
            size=26,
            marker="circle",
            color="#FFFFFF",
            fill_color=None,
            line_width=3.0,
            line_alpha=0.95,
            legend_label=f"Chen+2024 WL Selected (N={len(chen_table)})",
        )

        chen_hover = HoverTool(
            renderers=[chen_renderer],
            tooltips=[
                ("Catalog", "@catalog"),
                ("Peak ID", "#@peak_id"),
                ("WL Peak S/N", "@snr{0.00}"),
                ("RA", "@ra{0.0000} deg"),
                ("Dec", "@dec{0.0000} deg"),
                ("Redshift z", "@z{0.0000}"),
                ("Optical Match", "@opt_name (Richness: @richness{0.0})"),
                ("Separation", "@sep_mpc_h{0.00} Mpc/h"),
            ],
        )
        p.add_tools(chen_hover)

    # Customize layout and legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"  # Hide/show catalog by clicking legend
    p.legend.title = "Catalogs (Click to Toggle)"
    p.legend.background_fill_color = "#1e1e1e"
    p.legend.background_fill_alpha = 0.85
    p.legend.label_text_color = "#ffffff"
    p.legend.title_text_color = "#cccccc"
    p.legend.border_line_color = "#3a3a3a"

    # Save to file
    save(p)

    # Post-process HTML to make the layout center and fill the viewport
    html_content = save_path.read_text(encoding="utf-8")
    style_injection = """
    <style>
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background-color: #181818 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            overflow: hidden !important;
        }
        .bk-root {
            width: 98vw !important;
            height: 95vh !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
    </style>
    """
    html_content = html_content.replace("</head>", f"{style_injection}</head>")
    save_path.write_text(html_content, encoding="utf-8")
    print(f"Interactive Bokeh HTML saved and post-processed at {save_path}")


# %% Global Configuration

LABELS_TO_COMPARE = [
    "redm_pdr3_5band_free_1bin",
    "camira_hectomap_1bin",
    "redm_r16_hectomap_1bin",
    "amico_1bin",
    "cosine_1bin",
    "pls_1bin",
    "regression_1bin",
    "rz_diff_1bin",
]

DISPLAY_NAMES = {
    "redm_pdr3_5band_free_1bin": "redMaPPer PDR3",
    "camira_hectomap_1bin": "CAMIRA",
    "redm_r16_hectomap_1bin": "redMaPPer R16",
    "amico_1bin": "AMICO",
    "cosine_1bin": "Cosine Finder",
    "pls_1bin": "PLS Finder",
    "regression_1bin": "Regression Finder",
    "rz_diff_1bin": "r-z Diff Finder",
}

# Color palette: Paul Tol Bright/Muted adapted (10 distinct hues for high contrast)
PALETTE = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Purple
    "#EE7733",  # Orange
    "#009988",  # Teal
    "#332288",  # Indigo
    "#BBBBBB",  # Gray
]

# Plotting marker config (distinct shapes supported across matplotlib and Bokeh)
MARKERS = ["o", "s", "^", "v", "D", "h", "*", "d", "x", "+"]

OUTPUT_MATCH_HEATMAP = project_root / "output/plots_for_agents/matching_statistics.png"
OUTPUT_CONSENSUS_BREAKDOWN = (
    project_root / "output/plots_for_agents/consensus_breakdown.png"
)
OUTPUT_BOKEH_HTML = project_root / "output/plots_for_agents/spatial_distribution.html"
OUTPUT_TIER_CONSENSUS_PROFILES = (
    project_root / "output/plots_for_agents/tier_consensus_profiles.png"
)
OUTPUT_TIER_PAIRWISE_HEATMAPS = (
    project_root / "output/plots_for_agents/tier_pairwise_heatmaps.png"
)


# %% [Stage 1: Load and Match Catalogs]

dfs_dict = load_lens_data(LABELS_TO_COMPARE, project_root)
chen_tbl = load_chen2024_clusters(project_root)

print(
    f"\nLoaded Chen+2024 WL shear-selected clusters in HectoMAP: N={len(chen_tbl)} (z in [0.19, 0.52], Y3 mask)"
)

print("\n--- Pairwise Cluster Matching Statistics (0.5 Mpc/h) ---")
match_df = compute_pairwise_matches(dfs_dict)
with pd.option_context("display.max_columns", None, "display.width", 1000):
    print(match_df)

# %% [Stage 2: Plot Matching Heatmap]

plot_matching_heatmap(match_df, save_path=OUTPUT_MATCH_HEATMAP)

# %% [Stage 3: Overall Consensus Breakdown Analysis]

print("\n--- Multi-Catalog Consensus Breakdown ---")
consensus_counts_df, consensus_pct_df = compute_consensus_breakdown(dfs_dict)
print("Raw Counts (Count of Clusters matching k other catalogs):")
with pd.option_context("display.max_columns", None, "display.width", 1000):
    print(consensus_counts_df)

print("\nPercentages (% of Catalog):")
with pd.option_context("display.max_columns", None, "display.width", 1000):
    print(consensus_pct_df.round(1))

plot_consensus_breakdown(
    consensus_counts_df,
    consensus_pct_df,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_CONSENSUS_BREAKDOWN,
)

# %% [Stage 4: Plot Bokeh Interactive Visualization]

plot_bokeh_spatial(
    dfs_dict,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_BOKEH_HTML,
    chen_table=chen_tbl,
)

# %% [Stage 5: Tiered Proxy Consensus Analysis (4 Bins per Catalog)]

print("\n--- Tiered Proxy Rank Consensus Analysis (4 Bins per Catalog) ---")
tier_consensus_df = compute_tier_consensus_breakdown(
    dfs_dict, n_bins=4, display_names=DISPLAY_NAMES
)
print("Summary of Tiered Consensus Breakdown:")
summary_cols = [
    "display_name",
    "tier_name",
    "n_clusters",
    "mean_matches",
    "sem_matches",
    "pct_solo",
    "pct_ge4",
]
with pd.option_context("display.max_columns", None, "display.width", 1000):
    print(tier_consensus_df[summary_cols].round(2))

plot_tier_consensus_profiles(
    tier_consensus_df,
    catalog_order=LABELS_TO_COMPARE,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_TIER_CONSENSUS_PROFILES,
    display_names=DISPLAY_NAMES,
)

# %% [Stage 6: Tier-Resolved Pairwise Matching Heatmaps]

print("\n--- Tier-Resolved Pairwise Matching Matrices ---")
tier_pairwise_dict = compute_tier_pairwise_matches(
    dfs_dict, n_bins=4, display_names=DISPLAY_NAMES
)
plot_tier_pairwise_heatmaps(
    tier_pairwise_dict,
    save_path=OUTPUT_TIER_PAIRWISE_HEATMAPS,
)
