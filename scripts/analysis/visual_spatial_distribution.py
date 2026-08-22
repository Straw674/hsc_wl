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


def load_lens_data(labels: list[str], root: Path) -> dict[str, Table]:
    """Load prepared lens tables for the given configurations.

    Parameters
    ----------
    labels : list of str
        Run labels to load, e.g. ["redm_s16a_hectomap_1bin", ...].
    root : Path
        Project root path.

    Returns
    -------
    dict
        Dictionary mapping run label to the loaded astropy Table.
    """
    from hsc_wl.config import RUN_REGISTRY

    dfs = {}
    for label in labels:
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

    Saves results as a PNG file and displays the image.
    """
    import matplotlib.pyplot as plt

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

    # Use a sequential colormap ('YlGnBu') representing matching percentage
    im = ax.imshow(data_pct, cmap="YlGnBu", aspect="equal", vmin=0, vmax=100)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Match Fraction (%)", fontsize=11)

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

            # Contrasting text color based on cell brightness
            text_color = "white" if pct > 60.0 else "black"
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
        "Pairwise Lens Match Fractions (0.5 Mpc/h Physical Radius)\nRow Normalized: Fraction of Row Catalog Matched in Column Catalog",
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
        size = 12 + idx * 3  # Increase size per catalog to nest them
        # Map simple marker names to bokeh methods
        if marker_name == "o":
            renderer = p.scatter(
                x="ra",
                y="dec",
                source=source,
                size=size,
                color=color,
                fill_color=None,
                line_width=2.5,
                legend_label=f"{name} (N={len(tbl)})",
            )
        elif marker_name == "s":
            renderer = p.scatter(
                x="ra",
                y="dec",
                source=source,
                size=size,
                marker="square",
                color=color,
                fill_color=None,
                line_width=2.5,
                legend_label=f"{name} (N={len(tbl)})",
            )
        elif marker_name == "v":
            renderer = p.scatter(
                x="ra",
                y="dec",
                source=source,
                size=size,
                marker="triangle",
                color=color,
                fill_color=None,
                line_width=2.5,
                legend_label=f"{name} (N={len(tbl)})",
            )
        elif marker_name == "D":
            renderer = p.scatter(
                x="ra",
                y="dec",
                source=source,
                size=size,
                marker="diamond",
                color=color,
                fill_color=None,
                line_width=2.5,
                legend_label=f"{name} (N={len(tbl)})",
            )
        else:
            renderer = p.scatter(
                x="ra",
                y="dec",
                source=source,
                size=size - 3,
                marker="cross",
                color=color,
                line_width=2,
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
                "catalog": ["Chen+2024 WL Selected (Ground Truth)"] * len(chen_table),
            }
        )

        # Prominent larger white hollow circle (size=26, line_width=3.0) to highlight WL ground truth
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

# LABELS_TO_COMPARE = [
#     "redm_s16a_hectomap_1bin",
#     "logm_s16a_hectomap_1bin",
#     "redm_pdr3_3band_fixed_s16a_1bin",
#     "cosine_s16a_1bin",
#     "camira_hecto_s16a_1bin",
#     "redm_r16_hecto_s16a_1bin",
# ]

LABELS_TO_COMPARE = [
    ("redm_pdr3_5band_free_1bin"),
    ("camira_hectomap_1bin"),
    ("redm_r16_hectomap_1bin"),
    ("amico_1bin"),
    ("cosine_1bin"),
]


# Color palette: Paul Tol Bright/Muted adapted
PALETTE = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#AA3377",  # Purple
    "#EE7733",  # Orange
    "#BBBBBB",  # Gray
]

# Plotting marker config (increasing size order so they layer neatly)
MARKERS = ["o", "s", "v", "D", "x"]

OUTPUT_MATCH_HEATMAP = project_root / "output/plots_for_agents/matching_statistics.png"
OUTPUT_CONSENSUS_BREAKDOWN = (
    project_root / "output/plots_for_agents/consensus_breakdown.png"
)
OUTPUT_BOKEH_HTML = project_root / "output/plots_for_agents/spatial_distribution.html"

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

# %% [Stage 3: Consensus Breakdown Analysis]

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
