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
    redshift_range: tuple[float, float] = (0.19, 0.52),
) -> Table:
    """Load and filter the Chen et al. (2024) WL shear-selected cluster catalog.

    Applies survey redshift range and the full HSC Y3 shape catalog mask.

    Parameters
    ----------
    root : Path
        Project root path.
    redshift_range : tuple of (float, float), default (0.19, 0.52)
        Redshift bounds.

    Returns
    -------
    astropy.table.Table
        Filtered Chen+2024 cluster table with columns [peak_id, ra, dec, z, snr, ...].
    """
    from hsc_wl.coverage import load_y3_mask

    parquet_path = root / "data/chen2024_shear_selected_clusters.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Chen 2024 catalog not found at {parquet_path}. "
            "Ensure data/chen2024_shear_selected_clusters.parquet exists."
        )

    df = pd.read_parquet(parquet_path)

    # Filter by redshift range
    mask = (df["z_cl"] >= redshift_range[0]) & (df["z_cl"] <= redshift_range[1])
    filtered_df = df[mask].sort_values(by="snr", ascending=False).reset_index(drop=True)
    tbl = Table.from_pandas(filtered_df)

    # Filter by master Y3 shape mask
    y3_mask = load_y3_mask(root)
    ra = np.asarray(tbl["ra"], dtype=float)
    dec = np.asarray(tbl["dec"], dtype=float)
    inside = y3_mask.get_values_pos(ra, dec, lonlat=True)
    tbl = tbl[inside]

    tbl["z"] = tbl["z_cl"]
    tbl["rank"] = np.arange(1, len(tbl) + 1)
    return tbl


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
        tbl = Table.read(file_path)
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
            da1 = Planck18.angular_diameter_distance(z1).value  # Mpc

            # 0.5 Mpc/h in degrees: (0.5 / h) / da1 * (180 / pi)
            match_radius_deg = (0.5 / h) / da1 * (180.0 / np.pi)

            matched = d2d.deg < match_radius_deg
            matrix[i, j] = np.sum(matched)

    df_match = pd.DataFrame(matrix, index=catalog_names, columns=catalog_names)
    return df_match


def plot_matching_heatmap(
    df_match: pd.DataFrame,
    save_path: Path,
    display_names: dict[str, str] | None = None,
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
    """
    from astropy.visualization import HistEqStretch, ImageNormalize

    names_map = display_names or {}
    fig, ax = plt.subplots(figsize=(8.5, 7.2))

    data = df_match.values
    labels = [names_map.get(k, k) for k in df_match.index]
    n = len(labels)

    # Row i is the source catalog, cell (i, j) is the fraction of row i matched to column j
    row_totals = np.diag(data)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    data_pct = (data / row_totals_safe[:, None]) * 100.0

    stretch = HistEqStretch(data_pct)
    norm = ImageNormalize(vmin=float(data_pct.min()), vmax=100.0, stretch=stretch)

    im = ax.imshow(data_pct, cmap="YlGnBu", aspect="equal", norm=norm)

    norm_positions = np.linspace(0.05, 0.95, 6)
    tick_vals = norm.inverse(norm_positions)
    ticks = sorted(
        list(set(int(round(t)) for t in tick_vals)) + [int(round(data_pct.min())), 100]
    )
    cbar = fig.colorbar(im, ax=ax, ticks=ticks, shrink=0.82)
    cbar.set_label("Match Fraction (%) [Equalized Hist Stretch]", fontsize=10.5)

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
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            pct = data_pct[i, j]
            total = row_totals[i]

            text = f"{val}\n(100.0%)" if i == j else f"{val}/{total}\n({pct:.1f}%)"
            norm_val = float(norm(np.array([pct]))[0])
            text_color = "white" if norm_val > 0.60 else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontweight="normal",
                fontsize=9.5,
            )

    ax.set_title(
        "Pairwise Lens Match Fractions (0.5 Mpc/h Physical Radius)\n"
        "Full HSC Survey Footprint (439 deg², N=1020 per catalog)",
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute multi-catalog consensus counts and percentages.

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
):
    """Plot consensus level profiles across catalogs as a multi-line plot."""
    names_map = display_names or {}
    pct_matrix = df_pct.values
    catalog_names = list(df_pct.index)
    n_cats = len(catalog_names)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
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
        "Consensus Profiles across Full Survey Catalogs (0.5 Mpc/h Matching)",
        fontsize=11.5,
        pad=10,
        fontweight="normal",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(0, max(50.0, float(np.max(pct_matrix)) + 8.0))
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Consensus breakdown plot saved to {save_path}")
    plt.show()
    plt.close(fig)


def plot_spatial_distribution_regions(
    dfs: dict[str, Table],
    colors: list[str],
    markers: list[str],
    save_path: Path,
    chen_table: Table | None = None,
    display_names: dict[str, str] | None = None,
):
    """Plot static 3-panel sky distribution partitioned into canonical HSC fields.

    Partitions coordinates into SPRING, FALL, and HECTOMAP to avoid massive blank spaces.
    Accounts for sky projection aspect ratio and inverts RA per astronomical convention.
    """
    from hsc_wl.coverage import in_field

    names_map = display_names or {}
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10.5), gridspec_kw={"height_ratios": [1.2, 1.2, 1.0]}
    )

    regions_info = [
        {
            "name": "SPRING",
            "title": "SPRING (GAMA09H + WIDE12H + GAMA15H)",
            "ax": axes[0],
            "xlim": (230.0, 125.0),
            "ylim": (-3.0, 6.0),
            "dec_ref": 1.5,
            "shift_fall": False,
        },
        {
            "name": "FALL",
            "title": "FALL (XMM + VVDS)",
            "ax": axes[1],
            "xlim": (48.0, -43.0),
            "ylim": (-8.0, 8.0),
            "dec_ref": 0.0,
            "shift_fall": True,
        },
        {
            "name": "HECTOMAP",
            "title": "HECTOMAP (North)",
            "ax": axes[2],
            "xlim": (252.0, 210.0),
            "ylim": (41.5, 45.0),
            "dec_ref": 43.3,
            "shift_fall": False,
        },
    ]

    for reg in regions_info:
        ax = reg["ax"]
        f_name = reg["name"]
        shift = reg["shift_fall"]

        for idx, (k, tbl) in enumerate(dfs.items()):
            ra = np.asarray(tbl["ra"], dtype=float)
            dec = np.asarray(tbl["dec"], dtype=float)
            m = in_field(ra, dec, f_name)
            if not np.any(m):
                continue

            ra_sel = ra[m]
            dec_sel = dec[m]
            if shift:
                ra_sel = np.where(ra_sel > 180.0, ra_sel - 360.0, ra_sel)

            disp_name = names_map.get(k, k)
            ax.scatter(
                ra_sel,
                dec_sel,
                s=16,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                alpha=0.75,
                label=f"{disp_name} (N={np.sum(m)})" if reg["name"] == "SPRING" else "",
            )

        # Plot Chen+2024 WL shear-selected clusters if available
        if chen_table is not None and len(chen_table) > 0:
            c_ra = np.asarray(chen_table["ra"], dtype=float)
            c_dec = np.asarray(chen_table["dec"], dtype=float)
            c_m = in_field(c_ra, c_dec, f_name)
            if np.any(c_m):
                c_ra_sel = c_ra[c_m]
                c_dec_sel = c_dec[c_m]
                if shift:
                    c_ra_sel = np.where(c_ra_sel > 180.0, c_ra_sel - 360.0, c_ra_sel)
                ax.scatter(
                    c_ra_sel,
                    c_dec_sel,
                    s=64,
                    facecolors="none",
                    edgecolors="#000000",
                    linewidths=1.6,
                    marker="o",
                    label=f"Chen+2024 WL Selected (N={np.sum(c_m)})"
                    if reg["name"] == "SPRING"
                    else "",
                )

        ax.set_xlim(reg["xlim"])
        ax.set_ylim(reg["ylim"])
        ax.set_aspect(1.0 / np.cos(np.radians(reg["dec_ref"])))
        ax.set_title(reg["title"], fontsize=11, fontweight="normal", pad=4)
        ax.set_ylabel("Dec [deg]", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        if shift:
            ticks = np.arange(-40, 50, 15)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(t % 360)}°" for t in ticks])

    axes[2].set_xlabel(
        "RA [deg] (Astronomical Convention: Increasing to Left)", fontsize=10.5
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.998),
            ncol=len(handles),
            fontsize=9.5,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Regional spatial distribution plot saved to {save_path}")
    plt.show()
    plt.close(fig)


def build_interactive_regions(
    dfs: dict[str, Table], chen_table: Table | None
) -> list[dict]:
    """Use canonical field membership and padded catalog bounds for six panels."""
    from hsc_wl.coverage import in_field

    tables = list(dfs.values()) + ([] if chen_table is None else [chen_table])
    ra = np.concatenate([np.asarray(tbl["ra"], float) for tbl in tables])
    dec = np.concatenate([np.asarray(tbl["dec"], float) for tbl in tables])
    regions = []
    # Keep each row in decreasing RA order, matching the coordinate axes.
    for name, bounds in (
        ("GAMA15H", (203, 235, -3, 6)),
        ("WIDE12H", (153.5, 203, -3, 6)),
        ("GAMA09H", (125, 153.5, -3, 6)),
        ("XMM", (25, 45, -8, 8)),
        ("VVDS", (-40, 15, -8, 8)),
        ("HECTOMAP", (210, 252, 41.5, 45)),
    ):
        mask = in_field(ra, dec, name) & np.isfinite(ra) & np.isfinite(dec)
        x = ra[mask]
        y = dec[mask]
        if name == "VVDS":
            x = np.where(x > 180, x - 360, x)
        if len(x):
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        else:
            xmin, xmax, ymin, ymax = bounds
        xpad = max(0.4, (xmax - xmin) * 0.04)
        ypad = max(0.4, (ymax - ymin) * 0.06)
        regions.append(
            dict(
                name=name,
                title=name,
                x_start=float(xmax + xpad),
                x_end=float(xmin - xpad),
                y_start=float(ymin - ypad),
                y_end=float(ymax + ypad),
                dec_ref=float((ymin + ymax) / 2),
                shift_fall=name == "VVDS",
            )
        )
    return regions


def configure_sky_aspect(plot, region: dict):
    """Preserve the current view when layout changes while maintaining sky aspect."""
    from bokeh.models import ColumnDataSource, CustomJS

    state = ColumnDataSource(data={"initialized": [False]})

    callback = CustomJS(
        args=dict(plot=plot, region=region, state=state),
        code="""
        const w = plot.inner_width, h = plot.inner_height;
        if (!(w > 0 && h > 0)) return;
        const c = Math.cos(region.dec_ref * Math.PI / 180);
        const dx = region.x_start - region.x_end;
        const dy = region.y_end - region.y_start;
        const scale = Math.max(dx * c / w, dy / h);
        const cx = (region.x_start + region.x_end) / 2;
        const cy = (region.y_start + region.y_end) / 2;
        const sx = scale * w / c / 2, sy = scale * h / 2;
        plot.x_range.setv({reset_start: cx + sx, reset_end: cx - sx});
        plot.y_range.setv({reset_start: cy - sy, reset_end: cy + sy});
        if (!state.data.initialized[0]) {
            state.data.initialized[0] = true;
            plot.x_range.setv({start: cx + sx, end: cx - sx});
            plot.y_range.setv({start: cy - sy, end: cy + sy});
        } else {
            // Tick-label layout changes must not reset the user's zoom or pan.
            const center = (plot.x_range.start + plot.x_range.end) / 2;
            const half = (plot.y_range.end - plot.y_range.start) * w / h / c / 2;
            plot.x_range.setv({start: center + half, end: center - half});
        }
    """,
    )
    plot.js_on_change("inner_width", callback)
    plot.js_on_change("inner_height", callback)


def plot_bokeh_spatial_regional(
    dfs: dict[str, Table],
    colors: list[str],
    save_path: Path,
    chen_table: Table | None = None,
    display_names: dict[str, str] | None = None,
):
    """Save six responsive sky panels grouped by survey season as offline HTML."""
    from bokeh.embed import components
    from bokeh.models import (
        BoxZoomTool,
        ColumnDataSource,
        CustomJS,
        CustomJSTickFormatter,
        HoverTool,
        Range1d,
    )
    from bokeh.plotting import figure
    from bokeh.resources import Resources

    from hsc_wl.coverage import in_field

    names_map = display_names or {}
    region_defs = build_interactive_regions(dfs, chen_table)

    p_list = []

    for reg in region_defs:
        f_name = reg["name"]
        shift = reg["shift_fall"]

        p = figure(
            name=f_name,
            title=reg["title"],
            sizing_mode="stretch_both",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            x_axis_label="RA [deg]",
            y_axis_label="Dec [deg]",
            toolbar_location="above",
        )
        p.x_range = Range1d(start=reg["x_start"], end=reg["x_end"])
        p.y_range = Range1d(start=reg["y_start"], end=reg["y_end"])
        configure_sky_aspect(p, reg)
        p.select_one(BoxZoomTool).match_aspect = True
        if shift:
            p.xaxis.formatter = CustomJSTickFormatter(
                code="return ((tick % 360) + 360) % 360;"
            )

        p.background_fill_color = "#1e1e1e"
        p.border_fill_color = "#181818"
        p.grid.grid_line_color = "#3a3a3a"
        p.grid.grid_line_alpha = 0.5
        p.title.text_color = "#ffffff"
        p.title.text_font_size = "11pt"
        p.title.text_font_style = "normal"
        p.axis.axis_label_text_font_style = "normal"
        p.xaxis.axis_label_text_color = "#cccccc"
        p.yaxis.axis_label_text_color = "#cccccc"
        p.xaxis.major_label_text_color = "#aaaaaa"
        p.yaxis.major_label_text_color = "#aaaaaa"

        for idx, (k, tbl) in enumerate(dfs.items()):
            ra = np.asarray(tbl["ra"], dtype=float)
            dec = np.asarray(tbl["dec"], dtype=float)
            m = in_field(ra, dec, f_name)
            if not np.any(m):
                continue

            ra_sel = ra[m]
            dec_sel = dec[m]
            z_sel = np.asarray(tbl["z"], dtype=float)[m]
            rank_sel = np.asarray(tbl["rank"], dtype=int)[m]

            if shift:
                ra_sel = np.where(ra_sel > 180.0, ra_sel - 360.0, ra_sel)

            disp_name = names_map.get(k, k)
            source = ColumnDataSource(
                data={
                    "ra": ra_sel,
                    "ra_orig": ra[m],
                    "dec": dec_sel,
                    "z": z_sel,
                    "rank": rank_sel,
                    "catalog": [disp_name] * len(ra_sel),
                    "total": [len(tbl)] * len(ra_sel),
                }
            )

            diameter = 0.14 + idx * 0.035
            renderer = p.ellipse(
                x="ra",
                y="dec",
                source=source,
                width=diameter / np.cos(np.radians(reg["dec_ref"])),
                height=diameter,
                color=colors[idx % len(colors)],
                fill_color=None,
                line_width=2.0,
                legend_label=f"{disp_name} (N={np.sum(m)})",
            )

            hover = HoverTool(
                renderers=[renderer],
                tooltips=[
                    ("Catalog", "@catalog"),
                    ("Rank", "#@rank of @total"),
                    ("RA", "@ra_orig{0.0000} deg"),
                    ("Dec", "@dec{0.0000} deg"),
                    ("Redshift z", "@z{0.0000}"),
                ],
            )
            p.add_tools(hover)

        # Overlay Chen+2024 WL clusters
        if chen_table is not None and len(chen_table) > 0:
            c_ra = np.asarray(chen_table["ra"], dtype=float)
            c_dec = np.asarray(chen_table["dec"], dtype=float)
            c_m = in_field(c_ra, c_dec, f_name)
            if np.any(c_m):
                c_ra_sel = c_ra[c_m]
                c_dec_sel = c_dec[c_m]
                c_z_sel = np.asarray(chen_table["z"], dtype=float)[c_m]
                c_snr_sel = np.asarray(chen_table["snr"], dtype=float)[c_m]
                c_pk_sel = np.asarray(chen_table["peak_id"], dtype=int)[c_m]
                c_rich_sel = np.asarray(chen_table["richness"], dtype=float)[c_m]
                c_opt_sel = [str(x) for x in chen_table["opt_name"][c_m]]
                c_sep_sel = np.asarray(chen_table["sep_mpc_h"], dtype=float)[c_m]

                if shift:
                    c_ra_sel = np.where(c_ra_sel > 180.0, c_ra_sel - 360.0, c_ra_sel)

                chen_source = ColumnDataSource(
                    data={
                        "ra": c_ra_sel,
                        "ra_orig": c_ra[c_m],
                        "dec": c_dec_sel,
                        "z": c_z_sel,
                        "snr": c_snr_sel,
                        "peak_id": c_pk_sel,
                        "richness": c_rich_sel,
                        "opt_name": c_opt_sel,
                        "sep_mpc_h": c_sep_sel,
                        "catalog": ["Chen+2024 WL Selected"] * len(c_ra_sel),
                    }
                )

                chen_renderer = p.ellipse(
                    x="ra",
                    y="dec",
                    source=chen_source,
                    width=0.36 / np.cos(np.radians(reg["dec_ref"])),
                    height=0.36,
                    color="#FFFFFF",
                    fill_color=None,
                    line_width=2.5,
                    line_alpha=0.95,
                    legend_label=f"Chen+2024 WL Selected (N={np.sum(c_m)})",
                )

                chen_hover = HoverTool(
                    renderers=[chen_renderer],
                    tooltips=[
                        ("Catalog", "@catalog"),
                        ("Peak ID", "#@peak_id"),
                        ("WL Peak S/N", "@snr{0.00}"),
                        ("RA", "@ra_orig{0.0000} deg"),
                        ("Dec", "@dec{0.0000} deg"),
                        ("Redshift z", "@z{0.0000}"),
                        ("Optical Match", "@opt_name (Richness: @richness{0.0})"),
                        ("Separation", "@sep_mpc_h{0.00} Mpc/h"),
                    ],
                )
                p.add_tools(chen_hover)

        if p.legend:
            legend = p.legend[0]
            p.add_layout(legend, "below")
            legend.orientation = "horizontal"
            legend.ncols = 2
            p.js_on_change(
                "outer_width",
                CustomJS(
                    args=dict(plot=p, legend=legend),
                    code="""
                    const columns = Math.max(1, Math.min(legend.items.length,
                        Math.floor(plot.outer_width / 180)));
                    if (legend.ncols !== columns) legend.ncols = columns;
                """,
                ),
            )
        p.legend.location = "center"
        p.legend.label_text_font_size = "8pt"
        p.legend.padding = 4
        p.legend.spacing = 6
        p.legend.glyph_width = 14
        p.legend.label_standoff = 4
        p.legend.click_policy = "hide"
        p.legend.background_fill_color = "#1e1e1e"
        p.legend.background_fill_alpha = 0.85
        p.legend.label_text_color = "#ffffff"
        p.legend.title_text_color = "#cccccc"
        p.legend.border_line_color = "#3a3a3a"

        p_list.append(p)

    script, divs = components(p_list)
    # Include only the core runtime used by these plots, not imported Panel extensions.
    resources = "\n".join(
        f"<script>{code}</script>"
        for code in Resources(mode="inline", components=["bokeh"]).js_raw
    )
    sections = []
    for title, indices in (("SPRING", (0, 1, 2)), ("FALL", (3, 4)), ("HECTOMAP", (5,))):
        panels = "".join(f'<div class="sky-panel">{divs[i]}</div>' for i in indices)
        sections.append(
            f'<section><h2>{title}</h2><div class="region-row" '
            f'style="--columns:{len(indices)}">{panels}</div></section>'
        )
    html_content = (
        """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Catalog comparison</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; background: #181818; color: #eee; }
body { font: 15px system-ui, sans-serif; }
main { width: 100%; margin: 0 auto; padding: 20px clamp(12px, 2vw, 40px); }
h2 { font-size: 17px; font-weight: normal; margin: 24px 0 8px; }
.region-row { display: grid; grid-template-columns: repeat(var(--columns), minmax(0, 1fr)); gap: 16px; }
.sky-panel { min-width: 0; height: clamp(460px, 62vh, 820px); }
.sky-panel > div { width: 100%; height: 100%; }
@media (max-width: 900px) { .region-row { grid-template-columns: 1fr; } }
</style>
"""
        + resources
        + """</head><body><main>
"""
        + "".join(sections)
        + "</main>"
        + script
        + "</body></html>"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_content, encoding="utf-8")
    print(f"Interactive Bokeh HTML saved at {save_path}")


def compute_tier_consensus_breakdown(
    dfs: dict[str, Table],
    n_bins: int = 4,
    display_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute consensus statistics for each catalog partitioned into proxy rank tiers."""
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
) -> dict[int, pd.DataFrame]:
    """Compute pairwise match matrices for each proxy tier against full catalog."""
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
    """Plot multi-panel tiered consensus breakdown figure."""
    import matplotlib.gridspec as gridspec
    from astropy.visualization import HistEqStretch, ImageNormalize

    names_map = display_names or {}
    n_cats = len(catalog_order)

    ncols = 3 if n_cats > 4 else 2
    nrows = (n_cats + ncols - 1) // ncols

    fig = plt.figure(figsize=(14 if ncols == 3 else 13, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.2], hspace=0.35)

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

    stretch = HistEqStretch(mat_mean)
    norm = ImageNormalize(
        vmin=float(mat_mean.min()), vmax=float(mat_mean.max()), stretch=stretch
    )

    im = ax_heat.imshow(mat_mean, cmap="YlGnBu", aspect="auto", norm=norm)

    norm_positions = np.linspace(0.05, 0.95, 6)
    tick_vals = norm.inverse(norm_positions)
    ticks = sorted(
        list(set(round(float(t), 1) for t in tick_vals))
        + [round(float(mat_mean.min()), 1), round(float(mat_mean.max()), 1)]
    )
    cbar = fig.colorbar(im, ax=ax_heat, ticks=ticks, shrink=0.85, pad=0.02)
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
    ax_heat.set_xticklabels(cat_labels, fontsize=10, rotation=15, ha="right")
    ax_heat.set_yticks(np.arange(4))
    ax_heat.set_yticklabels(bin_row_labels, fontsize=10)
    ax_heat.set_title(
        "(a) Mean Consensus Score by Proxy Tier [Equalized Hist Stretch]",
        fontsize=11.5,
        pad=10,
        fontweight="normal",
    )

    for i in range(4):
        for j in range(n_cats):
            val = mat_mean[i, j]
            h_val = mat_high[i, j]
            solo_val = mat_solo[i, j]
            norm_val = float(norm(np.array([val]))[0])
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
    ax_heat.grid(which="minor", color="white", linestyle="-", linewidth=2.5)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    # Bottom Subplot: Stacked Consensus Spectrum (Grid)
    gs_spec = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=gs[1], hspace=0.35, wspace=0.22
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
        ax_b.grid(axis="x", linestyle=":", alpha=0.5)
        if row_i == nrows - 1:
            ax_b.set_xlabel("Composition (%)", fontsize=9.5)

    fig.legend(
        bars,
        spec_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=n_cats,
        fontsize=10,
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
):
    """Plot 2x2 grid of pairwise match fractions across proxy tiers."""
    from matplotlib.colors import Normalize

    bin_titles = [
        "Tier 1: Ranks 1–255 (Highest Proxy / Richness)",
        "Tier 2: Ranks 256–510 (Upper-Mid Proxy)",
        "Tier 3: Ranks 511–765 (Lower-Mid Proxy)",
        "Tier 4: Ranks 766–1020 (Lowest Proxy in Top 1020)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    norm = Normalize(vmin=20.0, vmax=100.0)

    for b_idx, ax in enumerate(axes.flat):
        df_mat = tier_pairwise_dict[b_idx]
        data = df_mat.values
        labels = list(df_mat.index)
        n = len(labels)

        im = ax.imshow(data, cmap="YlGnBu", norm=norm, aspect="equal")
        ax.set_title(
            f"({chr(97 + b_idx)}) {bin_titles[b_idx]}",
            fontsize=10.5,
            fontweight="normal",
            pad=8,
        )
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9.0)
        ax.set_yticklabels(labels, fontsize=9.0)

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
                    fontsize=9.0,
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
        "Tier-Resolved Pairwise Lens Matching Fractions (0.5 Mpc/h Matching Radius)\n"
        "Row: Clusters in Given Proxy Tier  |  Column: Matched in Full Top 1020 of Target Catalog",
        fontsize=12.0,
        fontweight="normal",
        y=0.97,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Tier-resolved pairwise heatmaps saved to {save_path}")
    plt.show()
    plt.close(fig)


# %% Global Configuration

LABELS_TO_COMPARE = [
    "camira_1bin",
    "redm_r16_1bin",
    "rz_diff_1bin",
    "rz_diff_lum_1bin",
    "rz_diff_fixed_1bin",
]

DISPLAY_NAMES = {
    "camira_1bin": "CAMIRA",
    "redm_r16_1bin": "redMaPPer R16",
    "rz_diff_1bin": "r-z Diff (Richness)",
    "rz_diff_lum_1bin": "r-z Diff (Luminosity)",
    "rz_diff_fixed_1bin": "r-z Diff (Fixed)",
}

PALETTE = [
    "#EE6677",  # Red (CAMIRA)
    "#4477AA",  # Blue (redMaPPer R16)
    "#228833",  # Green (r-z Richness)
    "#66CCEE",  # Cyan (r-z Luminosity)
    "#AA3377",  # Purple (r-z Fixed)
]

MARKERS = ["s", "x", "^", "D", "v"]

OUTPUT_MATCH_HEATMAP = project_root / "output/plots_for_agents/matching_statistics.png"
OUTPUT_CONSENSUS_BREAKDOWN = (
    project_root / "output/plots_for_agents/consensus_breakdown.png"
)
OUTPUT_SPATIAL_PNG = project_root / "output/plots_for_agents/spatial_distribution.png"
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
    f"\nLoaded Chen+2024 WL shear-selected clusters in full survey: N={len(chen_tbl)} (z in [0.19, 0.52], Y3 mask)"
)

match_df = compute_pairwise_matches(dfs_dict)


# %% [Stage 2: Plot Matching Heatmap]

plot_matching_heatmap(
    match_df, save_path=OUTPUT_MATCH_HEATMAP, display_names=DISPLAY_NAMES
)


# %% [Stage 3: Overall Consensus Breakdown Analysis]

consensus_counts_df, consensus_pct_df = compute_consensus_breakdown(dfs_dict)

plot_consensus_breakdown(
    consensus_counts_df,
    consensus_pct_df,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_CONSENSUS_BREAKDOWN,
    display_names=DISPLAY_NAMES,
)


# %% [Stage 4: Regional Spatial Distribution (Static PNG & Interactive Bokeh)]

plot_spatial_distribution_regions(
    dfs_dict,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_SPATIAL_PNG,
    chen_table=chen_tbl,
    display_names=DISPLAY_NAMES,
)

plot_bokeh_spatial_regional(
    dfs_dict,
    colors=PALETTE,
    save_path=OUTPUT_BOKEH_HTML,
    chen_table=chen_tbl,
    display_names=DISPLAY_NAMES,
)


# %% [Stage 5: Tiered Proxy Consensus Analysis (4 Quartiles per Catalog)]

tier_consensus_df = compute_tier_consensus_breakdown(
    dfs_dict, n_bins=4, display_names=DISPLAY_NAMES
)

plot_tier_consensus_profiles(
    tier_consensus_df,
    catalog_order=LABELS_TO_COMPARE,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_TIER_CONSENSUS_PROFILES,
    display_names=DISPLAY_NAMES,
)


# %% [Stage 6: Tier-Resolved Pairwise Matching Heatmaps]

tier_pairwise_dict = compute_tier_pairwise_matches(
    dfs_dict, n_bins=4, display_names=DISPLAY_NAMES
)
plot_tier_pairwise_heatmaps(
    tier_pairwise_dict,
    save_path=OUTPUT_TIER_PAIRWISE_HEATMAPS,
)
