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

from hsc_wl.reference_catalogs import REFERENCE_CATALOGS, load_reference_catalogs
from initial import *  # noqa: F401,F403
from src.data import read_catalog_frame

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

    df = read_catalog_frame(parquet_path)

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
        tbl = Table.from_pandas(read_catalog_frame(file_path))
        tbl["rank"] = np.arange(1, len(tbl) + 1)
        dfs[label] = tbl

    return dfs


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
        ax.grid(False, which="both")

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
    dfs: dict[str, Table],
    chen_table: Table | None,
    reference_catalogs: dict[str, pd.DataFrame],
) -> list[dict]:
    """Use canonical field membership and padded catalog bounds for six panels."""
    from hsc_wl.coverage import in_field

    tables = (
        list(dfs.values())
        + ([] if chen_table is None else [chen_table])
        + list(reference_catalogs.values())
    )
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
    """Size each frame to its field aspect without resetting zoom or pan."""
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
        // Use the full-field aspect for layout, independent of interactive zoom.
        const frameHeight = Math.max(1, Math.round(w * dy / (dx * c)));
        const chromeHeight = plot.outer_height - h;
        const height = frameHeight + chromeHeight;
        if (Math.abs(h - frameHeight) > 1) {
            if (plot.height !== height) plot.height = height;
            return;
        }
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


def draw_sky_marker(plot, region: dict, source, style: dict, key: str):
    """Draw consistent outline symbols in sky coordinates, including polygon glyphs."""
    radius = style["diameter"] / 2
    scale = 1 / np.cos(np.radians(region["dec_ref"]))
    common = dict(
        source=source,
        name=f"catalog_{key}_{region['name']}",
        line_color=style["color"],
        line_width=style["line_width"],
        line_alpha=style["alpha"],
        visible=style.get("visible", True),
    )
    shape = style["shape"]
    if shape in ("circle", "square"):
        glyph = plot.ellipse if shape == "circle" else plot.rect
        return glyph(
            x="ra",
            y="dec",
            width=radius * 2 * scale,
            height=radius * 2,
            fill_color=None,
            **common,
        )
    vertices = {
        "diamond": [(0, 1), (1, 0), (0, -1), (-1, 0)],
        "triangle": [(0, 1), (0.87, -0.5), (-0.87, -0.5)],
        "inverted_triangle": [(0, -1), (0.87, 0.5), (-0.87, 0.5)],
        "plus": [(-1, 0), (1, 0), (np.nan, np.nan), (0, -1), (0, 1)],
        "hexagon": [
            (np.cos(a), np.sin(a))
            for a in np.linspace(np.pi / 2, 5 * np.pi / 2, 6, endpoint=False)
        ],
    }[shape]
    offsets = np.asarray(vertices) * radius
    source.data["marker_x"] = [x + offsets[:, 0] * scale for x in source.data["ra"]]
    source.data["marker_y"] = [y + offsets[:, 1] for y in source.data["dec"]]
    if shape == "plus":
        return plot.multi_line(xs="marker_x", ys="marker_y", **common)
    return plot.patches(xs="marker_x", ys="marker_y", fill_color=None, **common)


def add_reference_layer(plot, region: dict, key: str, frame: pd.DataFrame, style: dict):
    """Overlay one native catalog center using glyph dimensions in sky degrees."""
    from bokeh.models import ColumnDataSource, HoverTool

    from hsc_wl.coverage import in_field

    selected = frame.loc[
        in_field(frame["ra"].to_numpy(), frame["dec"].to_numpy(), region["name"])
    ].copy()
    if selected.empty:
        return
    spec = REFERENCE_CATALOGS[key]
    selected["ra_orig"] = selected["ra"]
    if region["shift_fall"]:
        selected["ra"] = np.where(
            selected["ra"] > 180, selected["ra"] - 360, selected["ra"]
        )
    source = ColumnDataSource(selected)
    renderer = draw_sky_marker(
        plot,
        region,
        source,
        style,
        key,
    )
    tooltips = [
        ("Catalog", spec["label"]),
        ("Name / ID", "@name"),
        ("RA", "@ra_orig{0.0000} deg"),
        ("Dec", "@dec{0.0000} deg"),
        ("Redshift z", "@z{0.0000}"),
    ]
    for column, label in (
        ("snr", "S/N"),
        ("richness", "Richness"),
        ("contamination", "Pcont / Fcont"),
    ):
        if selected[column].notna().any():
            tooltips.append((label, f"@{column}{{0.000}}"))
    if selected["mass"].notna().any():
        tooltips.append((str(selected["mass_definition"].iloc[0]), "@mass{0.000}"))
    if selected["ra_opt"].notna().any():
        tooltips.extend(
            [
                ("Optical RA", "@ra_opt{0.0000} deg"),
                ("Optical Dec", "@dec_opt{0.0000} deg"),
            ]
        )
    quality_labels = {
        "des_y6_wazp": "COVER_FRAC_1MPC",
        "act_dr6": "flags",
        "erass1": "EXT_LIKE",
        "des_y3_redmapper": "maskfrac",
        "xxl_dr2": "Class",
    }
    if key in quality_labels:
        tooltips.append((quality_labels[key], "@quality"))
    plot.add_tools(HoverTool(renderers=[renderer], tooltips=tooltips))


def marker_swatch_html(style: dict) -> str:
    """Match the control swatch to the plotted outline symbol."""
    shapes = {
        "circle": '<circle cx="10" cy="10" r="6"/>',
        "square": '<rect x="4" y="4" width="12" height="12"/>',
        "diamond": '<path d="M10 3 L17 10 L10 17 L3 10 Z"/>',
        "triangle": '<path d="M10 3 L16 14 L4 14 Z"/>',
        "inverted_triangle": '<path d="M4 6 L16 6 L10 17 Z"/>',
        "plus": '<path d="M3 10 H17 M10 3 V17"/>',
        "hexagon": '<path d="M10 3 L16 6.5 L16 13.5 L10 17 L4 13.5 L4 6.5 Z"/>',
    }
    return (
        f'<svg class="marker-swatch" viewBox="0 0 20 20" aria-hidden="true" '
        f'fill="none" stroke="{style["color"]}" stroke-width="1.8">'
        + shapes[style["shape"]]
        + "</svg>"
    )


def build_catalog_controls_html(
    layers: dict[str, dict],
    groups: dict[str, tuple[str, ...]],
    redshift_range: tuple[float, float],
) -> str:
    """Group all layer switches and counts using their plotted color and shape."""
    from html import escape

    rows = []
    for group, keys in groups.items():
        switches = []
        for key in keys:
            layer = layers[key]
            checked_attr = " checked" if layer.get("visible", True) else ""
            switches.append(
                f'<label><input type="checkbox" class="catalog-toggle" data-catalog="{key}"{checked_attr}>'
                + marker_swatch_html(layer)
                + f'<span>{escape(layer["label"])}</span><span class="count">{layer["count"]}</span></label>'
            )
        group_all_checked = all(layers[k].get("visible", True) for k in keys)
        group_checked_attr = " checked" if group_all_checked else ""
        rows.append(
            '<div class="catalog-group">'
            f'<label class="group-label"><input type="checkbox" class="group-toggle"{group_checked_attr}>'
            f'{escape(group)}</label><div class="catalog-switches">'
            + "".join(switches)
            + "</div></div>"
        )
    return (
        "<header><h1>Catalog comparison</h1>"
        f"<p>{redshift_range[0]} ≤ z ≤ {redshift_range[1]}</p>"
        + "".join(rows)
        + "</header>"
    )


def build_panel_legend_html(plot, layers: dict[str, dict], order: list[str]) -> str:
    """Use actual marker swatches and field counts for clickable panel legends."""
    from html import escape

    renderers = {renderer.name: renderer for renderer in plot.renderers}
    buttons = []
    for key in order:
        renderer = renderers.get(f"catalog_{key}_{plot.name}")
        if renderer is None:
            continue
        layer = layers[key]
        count = len(renderer.data_source.data["ra"])
        is_visible = getattr(renderer, "visible", True)
        pressed_attr = "true" if is_visible else "false"
        buttons.append(
            f'<button type="button" class="panel-layer" data-catalog="{key}" '
            f'data-field="{plot.name}" aria-pressed="{pressed_attr}">'
            + marker_swatch_html(layer)
            + f'{escape(layer["label"])} <span class="count">{count}</span></button>'
        )
    return '<div class="panel-legend">' + "".join(buttons) + "</div>"


def plot_bokeh_spatial_regional(
    dfs: dict[str, Table],
    save_path: Path,
    reference_catalogs: dict[str, pd.DataFrame],
    redshift_range: tuple[float, float],
    styles: dict[str, dict],
    groups: dict[str, tuple[str, ...]],
    chen_table: Table | None = None,
    display_names: dict[str, str] | None = None,
):
    """Save six responsive sky panels grouped by survey season as offline HTML."""
    from bokeh.embed import components
    from bokeh.models import (
        BoxZoomTool,
        ColumnDataSource,
        CustomJSTickFormatter,
        HoverTool,
        Range1d,
    )
    from bokeh.plotting import figure
    from bokeh.resources import Resources

    from hsc_wl.coverage import in_field

    names_map = display_names or {}
    region_defs = build_interactive_regions(dfs, chen_table, reference_catalogs)
    layers = {
        key: dict(**styles[key], label=names_map.get(key, key), count=len(tbl))
        for key, tbl in dfs.items()
    }
    layers.update(
        {
            key: dict(
                **styles[key], label=REFERENCE_CATALOGS[key]["label"], count=len(frame)
            )
            for key, frame in reference_catalogs.items()
        }
    )
    if chen_table is not None:
        layers["chen2024"] = dict(
            **styles["chen2024"], label="Chen+2024 WL", count=len(chen_table)
        )
    layer_order = [key for keys in groups.values() for key in keys]

    p_list = []

    for reg in region_defs:
        f_name = reg["name"]
        shift = reg["shift_fall"]

        p = figure(
            name=f_name,
            title=reg["title"],
            sizing_mode="stretch_width",
            height=360,
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

        p.background_fill_color = "#ffffff"
        p.border_fill_color = "#ffffff"
        p.grid.grid_line_color = None
        p.outline_line_color = "#dce1e6"
        p.axis.axis_line_color = "#c4cbd3"
        p.axis.major_tick_line_color = "#c4cbd3"
        p.axis.minor_tick_line_color = None
        p.title.text_color = "#374151"
        p.title.text_font_size = "11pt"
        p.title.text_font_style = "normal"
        p.axis.axis_label_text_font_style = "normal"
        p.xaxis.axis_label_text_color = "#4b5563"
        p.yaxis.axis_label_text_color = "#4b5563"
        p.xaxis.major_label_text_color = "#64748b"
        p.yaxis.major_label_text_color = "#64748b"

        # Draw subdued references first so the main catalogs remain legible at overlaps.
        for key, frame in reference_catalogs.items():
            add_reference_layer(p, reg, key, frame, styles[key])

        for k, tbl in dfs.items():
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

            renderer = draw_sky_marker(
                p,
                reg,
                source,
                styles[k],
                k,
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

                chen_renderer = draw_sky_marker(
                    p,
                    reg,
                    chen_source,
                    styles["chen2024"],
                    "chen2024",
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

        p_list.append(p)

    script, divs = components(p_list)
    # Include only the core runtime used by these plots, not imported Panel extensions.
    resources = "\n".join(
        f"<script>{code}</script>"
        for code in Resources(mode="inline", components=["bokeh"]).js_raw
    )
    sections = []
    for title, indices in (("SPRING", (0, 1, 2)), ("FALL", (3, 4)), ("HECTOMAP", (5,))):
        panels = "".join(
            f'<div class="sky-panel">{divs[i]}'
            + build_panel_legend_html(p_list[i], layers, layer_order)
            + "</div>"
            for i in indices
        )
        sections.append(
            f'<section><h2>{title}</h2><div class="region-panels">{panels}</div></section>'
        )
    html_content = (
        """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Catalog comparison</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; background: #f5f6f8; color: #374151; }
body { font: 15px system-ui, sans-serif; }
main { width: 100%; margin: 0 auto; padding: 20px clamp(12px, 2vw, 40px); }
h1 { font-size: 19px; font-weight: normal; margin: 0 0 10px; }
header { margin-bottom: 24px; line-height: 1.6; }
header p { color: #64748b; font-size: 13px; }
.catalog-group { display: flex; gap: 12px; padding: 9px 0; border-top: 1px solid #e2e7ed; }
.group-label { width: 175px; flex-shrink: 0; color: #475569; font-size: 12px; }
.catalog-switches { display: flex; flex-wrap: wrap; gap: 6px 20px; }
.catalog-switches label { display: flex; align-items: center; gap: 5px; font-size: 13px; }
header label { cursor: pointer; }
.marker-swatch { width: 20px; height: 20px; flex-shrink: 0; }
.count { color: #84909e; font-size: 11px; font-variant-numeric: tabular-nums; }
.panel-legend { display: flex; justify-content: center; flex-wrap: wrap; gap: 3px 14px; padding: 8px 12px 14px; }
.panel-layer { display: flex; align-items: center; gap: 5px; border: 0; background: none; color: #475569; font: 12px system-ui, sans-serif; padding: 3px 0; cursor: pointer; }
.panel-layer[aria-pressed="false"] { opacity: 0.35; }
.panel-layer:hover { color: #111827; }
@media (max-width: 700px) { .catalog-group { flex-direction: column; gap: 5px; } }
section + section { border-top: 1px solid #cbd2da; margin-top: 28px; padding-top: 16px; }
h2 { font-size: 12px; font-weight: normal; color: #64748b; margin: 0 0 12px; }
.region-panels { display: flex; flex-direction: column; gap: 20px; }
.sky-panel { min-width: 0; width: 100%; background: #fff; }
.sky-panel > div { width: 100%; }
</style>
"""
        + resources
        + """</head><body><main>
"""
        + build_catalog_controls_html(layers, groups, redshift_range)
        + "".join(sections)
        + "</main>"
        + script
        + """<script>
function setCatalogVisibility(key, visible, field = null) {
    for (const doc of Bokeh.documents) {
        for (const name of ['GAMA15H', 'WIDE12H', 'GAMA09H', 'XMM', 'VVDS', 'HECTOMAP']) {
            if (field && name !== field) continue;
            const renderer = doc.get_model_by_name(`catalog_${key}_${name}`);
            if (renderer) renderer.visible = visible;
        }
    }
    document.querySelectorAll('.panel-layer').forEach(button => {
        if (button.dataset.catalog === key && (!field || button.dataset.field === field)) {
            button.setAttribute('aria-pressed', String(visible));
        }
    });
}
function syncCatalogControls() {
    document.querySelectorAll('.catalog-toggle').forEach(input => {
        const buttons = [...document.querySelectorAll('.panel-layer')].filter(button => button.dataset.catalog === input.dataset.catalog);
        input.checked = buttons.every(button => button.getAttribute('aria-pressed') === 'true');
        input.indeterminate = buttons.some(button => button.getAttribute('aria-pressed') === 'true') && !input.checked;
    });
    document.querySelectorAll('.group-toggle').forEach(input => {
        const members = [...input.closest('.catalog-group').querySelectorAll('.catalog-toggle')];
        input.checked = members.every(member => member.checked);
        input.indeterminate = members.some(member => member.checked || member.indeterminate) && !input.checked;
    });
}
document.querySelectorAll('.catalog-toggle').forEach(input => {
    input.addEventListener('change', () => {
        setCatalogVisibility(input.dataset.catalog, input.checked);
        syncCatalogControls();
    });
});
document.querySelectorAll('.group-toggle').forEach(input => {
    input.addEventListener('change', () => {
        input.closest('.catalog-group').querySelectorAll('.catalog-toggle').forEach(member => {
            member.checked = input.checked;
            setCatalogVisibility(member.dataset.catalog, input.checked);
        });
        syncCatalogControls();
    });
});
document.querySelectorAll('.panel-layer').forEach(button => {
    button.addEventListener('click', () => {
        setCatalogVisibility(button.dataset.catalog, button.getAttribute('aria-pressed') !== 'true', button.dataset.field);
        syncCatalogControls();
    });
});
syncCatalogControls();
</script></body></html>"""
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_content, encoding="utf-8")
    print(f"Interactive Bokeh HTML saved at {save_path}")


def match_reference_to_candidate(
    ref_table_or_df: pd.DataFrame | Table,
    cand_table_or_df: pd.DataFrame | Table,
    r_phys_mpc_h: float = 0.5,
) -> np.ndarray:
    """Cross-match reference clusters to candidate lenses within physical radius.

    When reference has valid redshift, evaluates physical transverse radius at
    reference redshift. When reference lacks redshift, falls back to matched
    candidate lens redshift.

    Parameters
    ----------
    ref_table_or_df : pd.DataFrame or Table
        Reference clusters containing coordinates and optional redshifts.
    cand_table_or_df : pd.DataFrame or Table
        Candidate lens catalog containing coordinates and redshifts.
    r_phys_mpc_h : float, default 0.5
        Physical transverse matching radius in Mpc/h.

    Returns
    -------
    np.ndarray of bool
        Boolean array indicating whether each reference cluster was matched.
    """
    n_ref = len(ref_table_or_df)
    if n_ref == 0:
        return np.zeros(0, dtype=bool)

    c_ref = SkyCoord(
        ra=np.asarray(ref_table_or_df["ra"], dtype=float) * u.deg,
        dec=np.asarray(ref_table_or_df["dec"], dtype=float) * u.deg,
    )
    c_cand = SkyCoord(
        ra=np.asarray(cand_table_or_df["ra"], dtype=float) * u.deg,
        dec=np.asarray(cand_table_or_df["dec"], dtype=float) * u.deg,
    )

    idx_match, d2d, _ = c_ref.match_to_catalog_sky(c_cand)

    has_ref_z = (
        "z" in ref_table_or_df.colnames
        if isinstance(ref_table_or_df, Table)
        else "z" in ref_table_or_df
    )
    ref_z = (
        np.asarray(ref_table_or_df["z"], dtype=float)
        if has_ref_z
        else np.full(n_ref, np.nan)
    )
    ref_z_valid = np.isfinite(ref_z) & (ref_z > 0)

    cand_z = np.asarray(cand_table_or_df["z"], dtype=float)
    matched_cand_z = cand_z[idx_match]

    z_eval = np.where(ref_z_valid, ref_z, matched_cand_z)
    z_eval = np.clip(z_eval, 1e-4, None)

    da = Planck18.angular_diameter_distance(z_eval).value  # Mpc
    h = Planck18.h
    r_deg = (r_phys_mpc_h / h) / da * (180.0 / np.pi)

    matched = d2d.deg < r_deg
    return matched


def compute_reference_matches_dict(
    reference_dict: dict[str, pd.DataFrame | Table],
    lens_dict: dict[str, Table],
    r_phys_mpc_h: float = 0.5,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute boolean match arrays for each reference catalog against candidate lenses."""
    matches = {}
    for ref_key, ref_df in reference_dict.items():
        matches[ref_key] = {}
        for cand_key, cand_df in lens_dict.items():
            matches[ref_key][cand_key] = match_reference_to_candidate(
                ref_df, cand_df, r_phys_mpc_h=r_phys_mpc_h
            )
    return matches


def compute_differential_advantage(
    matches_dict: dict[str, np.ndarray],
    first_class_keys: list[str],
    rz_keys: list[str],
    n_ref: int,
) -> pd.DataFrame:
    """Compute differential advantage matrix between rz_diff variants and 1st-class catalogs."""
    records = []
    for b_key in rz_keys:
        m_b = matches_dict[b_key]
        n_b = int(np.sum(m_b))
        for a_key in first_class_keys:
            m_a = matches_dict[a_key]
            n_a = int(np.sum(m_a))

            both = int(np.sum(m_a & m_b))
            only_b = int(np.sum(~m_a & m_b))
            only_a = int(np.sum(m_a & ~m_b))
            neither = int(np.sum(~m_a & ~m_b))

            delta_n = only_b - only_a
            discordant = only_b + only_a
            adv_ratio = (delta_n / discordant) if discordant > 0 else 0.0

            records.append(
                {
                    "rz_catalog": b_key,
                    "first_class_catalog": a_key,
                    "n_ref": n_ref,
                    "n_a": n_a,
                    "n_b": n_b,
                    "both": both,
                    "only_b": only_b,
                    "only_a": only_a,
                    "neither": neither,
                    "delta_n": delta_n,
                    "delta_pct": (delta_n / n_ref * 100.0) if n_ref > 0 else 0.0,
                    "adv_ratio": adv_ratio,
                }
            )
    return pd.DataFrame(records)


def compute_global_benchmark_scorecard(
    reference_dict: dict[str, pd.DataFrame | Table],
    all_matches: dict[str, dict[str, np.ndarray]],
    candidate_order: list[str],
    baseline_key: str = "camira_1bin",
    ref_metadata: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Compute benchmark completeness and delta vs baseline across all reference catalogs."""
    records = []
    meta = ref_metadata or {}
    for ref_key, ref_df in reference_dict.items():
        n_ref = len(ref_df)
        m_dict = all_matches[ref_key]
        baseline_m = m_dict.get(baseline_key, np.zeros(n_ref, dtype=bool))
        baseline_pct = (np.mean(baseline_m) * 100.0) if n_ref > 0 else 0.0

        ref_info = meta.get(ref_key, {})
        rec = {
            "ref_key": ref_key,
            "ref_name": ref_info.get("label", ref_key),
            "ref_type": ref_info.get("type", "General"),
            "ref_region": ref_info.get("region", "Survey"),
            "n_ref": n_ref,
        }
        for cand_key in candidate_order:
            m = m_dict[cand_key]
            n_match = int(np.sum(m))
            pct = (n_match / n_ref * 100.0) if n_ref > 0 else 0.0
            delta_pct = pct - baseline_pct
            rec[f"{cand_key}_count"] = n_match
            rec[f"{cand_key}_pct"] = pct
            rec[f"{cand_key}_delta_pct"] = delta_pct

        records.append(rec)
    return pd.DataFrame(records)


def print_scorecard_markdown_table(
    scorecard_df: pd.DataFrame,
    candidate_order: list[str],
    display_names: dict[str, str],
):
    """Print formatted markdown summary table to console."""
    headers = ["Reference", "Type", "Region", "N_ref"] + [
        display_names.get(k, k) for k in candidate_order
    ]
    sep = ["---"] * len(headers)
    print("\n| " + " | ".join(headers) + " |")
    print("| " + " | ".join(sep) + " |")
    for _, row in scorecard_df.iterrows():
        line = [
            str(row["ref_name"]),
            str(row["ref_type"]),
            str(row["ref_region"]),
            str(row["n_ref"]),
        ]
        for k in candidate_order:
            cnt = int(row[f"{k}_count"])
            pct = float(row[f"{k}_pct"])
            d_pct = float(row[f"{k}_delta_pct"])
            if k == "camira_1bin":
                line.append(f"{pct:.1f}% ({cnt})")
            else:
                line.append(f"{pct:.1f}% ({d_pct:+.1f}%)")
        print("| " + " | ".join(line) + " |")
    print()


def plot_differential_advantage_heatmaps(
    diff_dict: dict[str, pd.DataFrame],
    first_class_keys: list[str],
    rz_keys: list[str],
    save_path: Path,
    ref_keys: list[str] | None = None,
    display_names: dict[str, str] | None = None,
    ref_metadata: dict[str, dict[str, str]] | None = None,
):
    """Plot differential advantage heatmaps (rz_diff vs Class 1) across reference catalogs."""
    from matplotlib.colors import Normalize

    names_map = display_names or {}
    meta = ref_metadata or {}
    keys_to_plot = ref_keys or list(diff_dict.keys())
    n_plots = len(keys_to_plot)
    if n_plots == 0:
        return

    # Compute global maximum of absolute percentage across all references
    all_abs_pct = []
    for ref_k in keys_to_plot:
        df_d = diff_dict[ref_k]
        if len(df_d) > 0:
            all_abs_pct.extend(np.abs(df_d["delta_pct"].values))
    max_pct = max(float(np.max(all_abs_pct)), 1.0) if len(all_abs_pct) > 0 else 10.0
    vlim = float(np.ceil(max_pct / 5.0) * 5.0)
    norm = Normalize(vmin=-vlim, vmax=vlim)
    cmap = plt.colormaps["coolwarm"]

    ncols = 4 if n_plots >= 4 else n_plots
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols + 0.6, 4.4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    y_labels = [names_map.get(k, k) for k in rz_keys]
    x_labels = [names_map.get(k, k) for k in first_class_keys]

    im_ref = None

    for p_idx, ref_k in enumerate(keys_to_plot):
        ax = axes[p_idx]
        df_diff = diff_dict[ref_k]
        ref_info = meta.get(ref_k, {})
        ref_title = ref_info.get("label", ref_k)
        n_ref = df_diff["n_ref"].iloc[0] if len(df_diff) > 0 else 0

        delta_pct_mat = np.zeros((len(rz_keys), len(first_class_keys)), dtype=float)
        delta_n_mat = np.zeros((len(rz_keys), len(first_class_keys)), dtype=int)
        both_mat = np.zeros_like(delta_n_mat)
        only_b_mat = np.zeros_like(delta_n_mat)
        only_a_mat = np.zeros_like(delta_n_mat)

        for _, row in df_diff.iterrows():
            r_idx = rz_keys.index(row["rz_catalog"])
            c_idx = first_class_keys.index(row["first_class_catalog"])
            delta_pct_mat[r_idx, c_idx] = float(row["delta_pct"])
            delta_n_mat[r_idx, c_idx] = int(row["delta_n"])
            both_mat[r_idx, c_idx] = int(row["both"])
            only_b_mat[r_idx, c_idx] = int(row["only_b"])
            only_a_mat[r_idx, c_idx] = int(row["only_a"])

        im = ax.imshow(delta_pct_mat, cmap=cmap, norm=norm, aspect="auto")
        im_ref = im

        ax.set_xticks(np.arange(len(first_class_keys)))
        ax.set_yticks(np.arange(len(rz_keys)))
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9.0)
        ax.set_yticklabels(y_labels if (p_idx % ncols == 0) else [], fontsize=9.0)

        region_str = ref_info.get("region", "")
        reg_annot = f" ({region_str})" if region_str else ""
        ax.set_title(
            f"{ref_title}{reg_annot}\nN={n_ref} Clusters",
            fontsize=10.0,
            pad=8,
            fontweight="normal",
        )

        for r_i in range(len(rz_keys)):
            for c_j in range(len(first_class_keys)):
                d_pct = delta_pct_mat[r_i, c_j]
                d_val = delta_n_mat[r_i, c_j]
                b_val = only_b_mat[r_i, c_j]
                a_val = only_a_mat[r_i, c_j]
                both_val = both_mat[r_i, c_j]

                norm_val = norm(d_pct)
                text_color = (
                    "white" if (norm_val < 0.22 or norm_val > 0.78) else "black"
                )

                cell_txt = (
                    f"Δ={d_val:+d} ({d_pct:+.1f}%)\n"
                    f"+{b_val} / -{a_val}\n"
                    f"both:{both_val}"
                )
                ax.text(
                    c_j,
                    r_i,
                    cell_txt,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.0,
                    fontweight="normal",
                    linespacing=1.2,
                )

    for empty_idx in range(n_plots, len(axes)):
        axes[empty_idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.93, 0.98])
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.942, 0.18, 0.012, 0.64])
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        cbar.set_label("Net Differential Advantage Δ / N_ref (%)", fontsize=10.0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Differential advantage heatmap saved to {save_path}")
    plt.show()
    plt.close(fig)


def plot_benchmark_scorecard(
    scorecard_df: pd.DataFrame,
    candidate_keys: list[str],
    save_path: Path,
    baseline_key: str = "camira_1bin",
    display_names: dict[str, str] | None = None,
):
    """Plot public benchmark cluster recovery scorecard across references."""
    from matplotlib.colors import Normalize

    names_map = display_names or {}
    n_refs = len(scorecard_df)
    n_cands = len(candidate_keys)

    fig, ax = plt.subplots(figsize=(13.5, max(5.0, 0.75 * n_refs + 1.8)))

    pct_matrix = np.zeros((n_refs, n_cands), dtype=float)
    delta_matrix = np.zeros((n_refs, n_cands), dtype=float)
    counts_matrix = np.zeros((n_refs, n_cands), dtype=int)
    n_ref_list = scorecard_df["n_ref"].values

    for i, (_, row) in enumerate(scorecard_df.iterrows()):
        for j, c_key in enumerate(candidate_keys):
            pct_matrix[i, j] = float(row[f"{c_key}_pct"])
            delta_matrix[i, j] = float(row[f"{c_key}_delta_pct"])
            counts_matrix[i, j] = int(row[f"{c_key}_count"])

    norm = Normalize(vmin=0.0, vmax=max(80.0, float(np.max(pct_matrix))))
    cmap = plt.colormaps["YlGnBu"]

    im = ax.imshow(pct_matrix, cmap=cmap, norm=norm, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Match Fraction (%)", fontsize=10.0)

    row_labels = [
        f"{row['ref_name']} [{row['ref_type']}]\n({row['ref_region']}, N={row['n_ref']})"
        for _, row in scorecard_df.iterrows()
    ]
    col_labels = [names_map.get(k, k) for k in candidate_keys]

    ax.set_xticks(np.arange(n_cands))
    ax.set_yticks(np.arange(n_refs))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=9.5)
    ax.set_yticklabels(row_labels, fontsize=9.0)

    for i in range(n_refs):
        for j in range(n_cands):
            pct_val = pct_matrix[i, j]
            cnt_val = counts_matrix[i, j]
            tot_val = n_ref_list[i]
            d_val = delta_matrix[i, j]

            norm_val = norm(pct_val)
            text_color = "white" if norm_val > 0.58 else "black"

            if candidate_keys[j] == baseline_key:
                cell_txt = f"{pct_val:.1f}%\n({cnt_val}/{tot_val})\n[Baseline]"
            else:
                cell_txt = f"{pct_val:.1f}%\n({cnt_val}/{tot_val})\nΔ: {d_val:+.1f}%"

            ax.text(
                j,
                i,
                cell_txt,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="normal",
                linespacing=1.2,
            )

    ax.set_title(
        "Public Benchmark Cluster Recovery Scorecard across Overlapping HSC Footprints\n"
        "(Matching within 0.5 Mpc/h Physical Transverse Radius)",
        fontsize=11.0,
        pad=12,
        fontweight="normal",
    )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Benchmark scorecard plot saved to {save_path}")
    plt.show()
    plt.close(fig)


# %% Global Configuration

LABELS_TO_COMPARE = [
    "camira_1bin",
    "redm_r16_1bin",
    "amico_1bin",
    "rz_diff_1bin",
    "rz_diff_lum_1bin",
    "rz_diff_fixed_1bin",
    "rz_diff_fixed_lum_1bin",
    "rz_diff_no_bkg_1bin",
    "rz_diff_no_bkg_lum_1bin",
]

DISPLAY_NAMES = {
    "camira_1bin": "CAMIRA",
    "redm_r16_1bin": "redMaPPer R16",
    "amico_1bin": "AMICO",
    "rz_diff_1bin": "r-z Diff (Richness)",
    "rz_diff_lum_1bin": "r-z Diff (Luminosity)",
    "rz_diff_fixed_1bin": "r-z Diff (Fixed)",
    "rz_diff_fixed_lum_1bin": "r-z Diff (Fixed Lum)",
    "rz_diff_no_bkg_1bin": "r-z Diff (No Bkg)",
    "rz_diff_no_bkg_lum_1bin": "r-z Diff (No Bkg Lum)",
}

PALETTE = [
    "#EE6677",  # Red (CAMIRA)
    "#4477AA",  # Blue (redMaPPer R16)
    "#10B981",  # Green (AMICO)
    "#228833",  # Dark Green (r-z Richness)
    "#66CCEE",  # Cyan (r-z Luminosity)
    "#AA3377",  # Purple (r-z Fixed)
    "#CCBB44",  # Yellow (r-z Fixed Lum)
    "#EE7733",  # Orange (r-z No Bkg)
    "#332288",  # Navy (r-z No Bkg Lum)
]

MARKERS = ["s", "x", "o", "^", "D", "v", "<", ">", "p"]

REFERENCE_KEYS = tuple(REFERENCE_CATALOGS)
REDSHIFT_RANGE = (0.19, 0.52)
MATCH_RADIUS_MPC_H = 0.5

FIRST_CLASS_KEYS = [
    "camira_1bin",
    "redm_r16_1bin",
    "amico_1bin",
]

RZ_DIFF_KEYS = [
    "rz_diff_fixed_1bin",
    "rz_diff_fixed_lum_1bin",
    "rz_diff_1bin",
    "rz_diff_lum_1bin",
    "rz_diff_no_bkg_1bin",
    "rz_diff_no_bkg_lum_1bin",
]

REFERENCE_BENCHMARK_ORDER = [
    # X-ray
    "erass1",
    "efeds",
    "xxl_dr2",
    # SZ
    "act_dr6",
    # WL
    "chen2024",
    # Optical
    "des_y3_redmapper",
    "des_y6_wazp",
    "kids_dr3_amico",
]

REFERENCE_METADATA = {
    "erass1": {
        "label": "eRASS1 + eROMaPPer",
        "type": "X-ray",
        "region": "Spring",
    },
    "efeds": {
        "label": "eFEDS + MCMF",
        "type": "X-ray",
        "region": "GAMA09H",
    },
    "xxl_dr2": {
        "label": "XXL DR2 C1/C2",
        "type": "X-ray",
        "region": "XMM",
    },
    "act_dr6": {
        "label": "ACT DR6 SZ",
        "type": "SZ",
        "region": "Spring+Fall",
    },
    "chen2024": {
        "label": "Chen+2024 WL",
        "type": "WL Shear",
        "region": "Full Survey",
    },
    "des_y3_redmapper": {
        "label": "DES Y3 redMaPPer",
        "type": "Optical",
        "region": "Fall",
    },
    "des_y6_wazp": {
        "label": "DES Y6 WaZP",
        "type": "Optical",
        "region": "Fall",
    },
    "kids_dr3_amico": {
        "label": "KiDS DR3 AMICO",
        "type": "Optical",
        "region": "Spring",
    },
}

OUTPUT_SPATIAL_PNG = project_root / "output/plots_for_agents/spatial_distribution.png"
OUTPUT_BOKEH_HTML = project_root / "output/plots_for_agents/spatial_distribution.html"
OUTPUT_SCORECARD_PNG = (
    project_root / "output/plots_for_agents/reference_benchmark_scorecard.png"
)
OUTPUT_DIFF_HEATMAPS_ALL_PNG = (
    project_root / "output/plots_for_agents/differential_advantage_all_references.png"
)


# %% [Stage 1: Load Catalogs (Lenses, Chen+2024, References)]

dfs_dict = load_lens_data(LABELS_TO_COMPARE, project_root)
chen_tbl = load_chen2024_clusters(project_root, redshift_range=REDSHIFT_RANGE)
reference_dfs = load_reference_catalogs(project_root, REFERENCE_KEYS, REDSHIFT_RANGE)

print(
    f"\nLoaded Chen+2024 WL shear-selected clusters in full survey: N={len(chen_tbl)} (z in [0.19, 0.52], Y3 mask)"
)


# %% [Stage 2: Static Regional Spatial Distribution (3-panel PNG)]

plot_spatial_distribution_regions(
    dfs_dict,
    colors=PALETTE,
    markers=MARKERS,
    save_path=OUTPUT_SPATIAL_PNG,
    chen_table=chen_tbl,
    display_names=DISPLAY_NAMES,
)


# %% [Stage 3: Interactive Regional Spatial Web Visualizer (Bokeh HTML)]

HTML_MAIN_KEYS = (
    "camira_1bin",
    "redm_r16_1bin",
    "amico_1bin",
    "rz_diff_fixed_1bin",
    "rz_diff_fixed_lum_1bin",
    "rz_diff_1bin",
    "rz_diff_lum_1bin",
    "rz_diff_no_bkg_1bin",
    "rz_diff_no_bkg_lum_1bin",
)
HTML_GROUPS = {
    "CAMIRA / redMaPPer / AMICO": ("camira_1bin", "redm_r16_1bin", "amico_1bin"),
    "RZ diff": (
        "rz_diff_fixed_1bin",
        "rz_diff_fixed_lum_1bin",
        "rz_diff_1bin",
        "rz_diff_lum_1bin",
        "rz_diff_no_bkg_1bin",
        "rz_diff_no_bkg_lum_1bin",
    ),
    "Reference catalogs": (
        "act_dr6",
        "erass1",
        "efeds",
        "xxl_dr2",
        "des_y6_wazp",
        "des_y3_redmapper",
        "kids_dr3_amico",
        "chen2024",
    ),
}
HTML_STYLES = {
    "camira_1bin": dict(
        color="#286FA5",
        shape="circle",
        diameter=0.18,
        line_width=1.9,
        alpha=0.95,
        visible=True,
    ),
    "redm_r16_1bin": dict(
        color="#20466E",
        shape="square",
        diameter=0.23,
        line_width=1.9,
        alpha=0.95,
        visible=True,
    ),
    "amico_1bin": dict(
        color="#10B981",
        shape="diamond",
        diameter=0.20,
        line_width=1.9,
        alpha=0.95,
        visible=True,
    ),
    "rz_diff_fixed_1bin": dict(
        color="#C7682E",
        shape="inverted_triangle",
        diameter=0.24,
        line_width=1.9,
        alpha=0.95,
        visible=True,
    ),
    "rz_diff_fixed_lum_1bin": dict(
        color="#8B1E0F",
        shape="inverted_triangle",
        diameter=0.26,
        line_width=1.8,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_1bin": dict(
        color="#D97706",
        shape="inverted_triangle",
        diameter=0.21,
        line_width=1.7,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_lum_1bin": dict(
        color="#9A3412",
        shape="inverted_triangle",
        diameter=0.27,
        line_width=1.7,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_no_bkg_1bin": dict(
        color="#EA580C",
        shape="inverted_triangle",
        diameter=0.22,
        line_width=1.7,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_no_bkg_lum_1bin": dict(
        color="#7C2D12",
        shape="inverted_triangle",
        diameter=0.26,
        line_width=1.7,
        alpha=0.95,
        visible=False,
    ),
    "act_dr6": dict(
        color="#327D80",
        shape="square",
        diameter=0.12,
        line_width=1.6,
        alpha=0.8,
        visible=True,
    ),
    "erass1": dict(
        color="#527568",
        shape="circle",
        diameter=0.30,
        line_width=1.5,
        alpha=0.8,
        visible=True,
    ),
    "efeds": dict(
        color="#88768F",
        shape="diamond",
        diameter=0.14,
        line_width=1.5,
        alpha=0.8,
        visible=True,
    ),
    "xxl_dr2": dict(
        color="#66798C",
        shape="triangle",
        diameter=0.19,
        line_width=1.5,
        alpha=0.8,
        visible=True,
    ),
    "des_y6_wazp": dict(
        color="#7C8A58",
        shape="triangle",
        diameter=0.13,
        line_width=1.5,
        alpha=0.7,
        visible=True,
    ),
    "des_y3_redmapper": dict(
        color="#6A94BD",
        shape="diamond",
        diameter=0.28,
        line_width=1.7,
        alpha=0.9,
        visible=True,
    ),
    "kids_dr3_amico": dict(
        color="#8A99A7",
        shape="plus",
        diameter=0.09,
        line_width=1.2,
        alpha=0.65,
        visible=False,
    ),
    "chen2024": dict(
        color="#3B424A",
        shape="hexagon",
        diameter=0.36,
        line_width=1.8,
        alpha=0.95,
        visible=True,
    ),
}

plot_bokeh_spatial_regional(
    {key: dfs_dict[key] for key in HTML_MAIN_KEYS},
    save_path=OUTPUT_BOKEH_HTML,
    reference_catalogs=reference_dfs,
    redshift_range=REDSHIFT_RANGE,
    styles=HTML_STYLES,
    groups=HTML_GROUPS,
    chen_table=chen_tbl,
    display_names=DISPLAY_NAMES,
)


# %% [Stage 4: Reference Benchmark Matching & Global Scorecard]

raw_references_dict = {
    **reference_dfs,
    "chen2024": pd.DataFrame(
        {col: np.asarray(chen_tbl[col]) for col in chen_tbl.colnames}
    ),
}
all_references_dict = {
    k: raw_references_dict[k]
    for k in REFERENCE_BENCHMARK_ORDER
    if k in raw_references_dict
}

all_benchmark_matches = compute_reference_matches_dict(
    all_references_dict,
    dfs_dict,
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
)

scorecard_df = compute_global_benchmark_scorecard(
    all_references_dict,
    all_benchmark_matches,
    candidate_order=FIRST_CLASS_KEYS + RZ_DIFF_KEYS,
    baseline_key="camira_1bin",
    ref_metadata=REFERENCE_METADATA,
)

print_scorecard_markdown_table(
    scorecard_df,
    candidate_order=FIRST_CLASS_KEYS + RZ_DIFF_KEYS,
    display_names=DISPLAY_NAMES,
)

plot_benchmark_scorecard(
    scorecard_df,
    candidate_keys=FIRST_CLASS_KEYS + RZ_DIFF_KEYS,
    save_path=OUTPUT_SCORECARD_PNG,
    baseline_key="camira_1bin",
    display_names=DISPLAY_NAMES,
)


# %% [Stage 5: Differential Advantage Heatmaps (rz_diff vs Class 1)]

diff_matrices = {
    ref_k: compute_differential_advantage(
        all_benchmark_matches[ref_k],
        first_class_keys=FIRST_CLASS_KEYS,
        rz_keys=RZ_DIFF_KEYS,
        n_ref=len(all_references_dict[ref_k]),
    )
    for ref_k in all_references_dict
}

plot_differential_advantage_heatmaps(
    diff_matrices,
    first_class_keys=FIRST_CLASS_KEYS,
    rz_keys=RZ_DIFF_KEYS,
    save_path=OUTPUT_DIFF_HEATMAPS_ALL_PNG,
    ref_keys=list(all_references_dict.keys()),
    display_names=DISPLAY_NAMES,
    ref_metadata=REFERENCE_METADATA,
)
