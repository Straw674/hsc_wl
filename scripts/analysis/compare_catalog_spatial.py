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

from hsc_wl.prepare import load_stratified_candidates
from hsc_wl.reference_catalogs import (
    EXTERNAL_REFERENCE_METADATA,
    load_external_reference_catalogs,
)
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


def build_interactive_regions(
    dfs: dict[str, Table],
    reference_catalogs: dict[str, pd.DataFrame],
) -> list[dict]:
    """Use canonical field membership and padded catalog bounds for six panels."""
    from hsc_wl.coverage import in_field

    tables = list(dfs.values()) + list(reference_catalogs.values())
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
    spec = EXTERNAL_REFERENCE_METADATA[key]
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
        "erass3": "EXT_LIKE",
        "efeds": "EXT_LIKE",
        "des_y3_redmapper": "maskfrac",
        "xxl_dr2": "Bextlike",
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
        f"<p>Optical catalogs: {redshift_range[0]} ≤ z ≤ {redshift_range[1]}; X-ray / SZ / WL references: no redshift cuts</p>"
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
    region_defs = build_interactive_regions(dfs, reference_catalogs)
    layers = {
        key: dict(**styles[key], label=names_map.get(key, key), count=len(tbl))
        for key, tbl in dfs.items()
    }
    layers.update(
        {
            key: dict(
                **styles[key],
                label=EXTERNAL_REFERENCE_METADATA[key]["label"],
                count=len(frame),
            )
            for key, frame in reference_catalogs.items()
        }
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
    include_summary_rows: bool = True,
    summary_groups: tuple[str, ...] = ("non_optical", "optical", "overall"),
) -> pd.DataFrame:
    """Compute benchmark completeness and delta vs baseline across all reference catalogs."""
    records = []
    meta = ref_metadata or {}
    ref_records_by_key = {}

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
            m = m_dict.get(cand_key, np.zeros(n_ref, dtype=bool))
            n_match = int(np.sum(m))
            pct = (n_match / n_ref * 100.0) if n_ref > 0 else 0.0
            delta_pct = pct - baseline_pct
            rec[f"{cand_key}_count"] = n_match
            rec[f"{cand_key}_pct"] = pct
            rec[f"{cand_key}_delta_pct"] = delta_pct

        records.append(rec)
        ref_records_by_key[ref_key] = rec

    if include_summary_rows:
        group_specs = {
            "non_optical": {
                "name": "Mean (X-ray / SZ / WL · 5 refs)",
                "region": "5 Physical Probes",
                "keys": ("erass3", "efeds", "xxl_dr2", "act_dr6", "chen2024"),
            },
            "optical": {
                "name": "Mean (Optical · 3 refs)",
                "region": "3 Optical Surveys",
                "keys": ("des_y3_redmapper", "des_y6_wazp", "kids_dr3_amico"),
            },
            "overall": {
                "name": "Mean (Overall · 8 refs)",
                "region": "8 References",
                "keys": tuple(EXTERNAL_REFERENCE_METADATA),
            },
        }

        for grp_id in summary_groups:
            if grp_id not in group_specs:
                continue
            spec = group_specs[grp_id]
            target_keys = [k for k in spec["keys"] if k in ref_records_by_key]
            if not target_keys:
                continue

            baseline_mean = float(
                np.mean(
                    [ref_records_by_key[k][f"{baseline_key}_pct"] for k in target_keys]
                )
            )
            total_n_ref = sum(ref_records_by_key[k]["n_ref"] for k in target_keys)

            sum_rec = {
                "ref_key": f"summary_{grp_id}",
                "ref_name": spec["name"],
                "ref_type": "Summary",
                "ref_region": spec["region"],
                "n_ref": total_n_ref,
            }
            for cand_key in candidate_order:
                cand_mean_pct = float(
                    np.mean(
                        [ref_records_by_key[k][f"{cand_key}_pct"] for k in target_keys]
                    )
                )
                cand_total_count = sum(
                    ref_records_by_key[k][f"{cand_key}_count"] for k in target_keys
                )
                sum_rec[f"{cand_key}_count"] = cand_total_count
                sum_rec[f"{cand_key}_pct"] = cand_mean_pct
                sum_rec[f"{cand_key}_delta_pct"] = cand_mean_pct - baseline_mean

            records.append(sum_rec)

    return pd.DataFrame(records)


def print_scorecard_markdown_table(
    scorecard_df: pd.DataFrame,
    candidate_order: list[str],
    display_names: dict[str, str],
    baseline_key: str = "camira_1bin",
):
    """Print formatted markdown summary table to console."""
    headers = ["Reference", "Type", "Region", "N_ref"] + [
        display_names.get(k, k) for k in candidate_order
    ]
    sep = ["---"] * len(headers)
    print("\n| " + " | ".join(headers) + " |")
    print("| " + " | ".join(sep) + " |")
    for _, row in scorecard_df.iterrows():
        is_summary = str(row["ref_type"]) == "Summary"
        line = [
            str(row["ref_name"]),
            str(row["ref_type"]),
            str(row["ref_region"]),
            "—" if is_summary else str(row["n_ref"]),
        ]
        for k in candidate_order:
            cnt = int(row[f"{k}_count"])
            pct = float(row[f"{k}_pct"])
            d_pct = float(row[f"{k}_delta_pct"])
            if is_summary:
                if k == baseline_key:
                    line.append(f"{pct:.1f}%")
                else:
                    line.append(f"{pct:.1f}% ({d_pct:+.1f}%)")
            else:
                if k == baseline_key:
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
    suptitle: str | None = None,
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

    n_fc = len(first_class_keys)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    sub_w = max(5.0, 0.72 * n_fc + 1.2)
    sub_h = max(4.5, 0.65 * len(rz_keys) + 0.5)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(sub_w * ncols + 0.6, sub_h * nrows),
    )
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

        lbl_fs = 8.5 if n_fc <= 5 else 7.5
        ax.set_xticks(np.arange(len(first_class_keys)))
        ax.set_yticks(np.arange(len(rz_keys)))
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=lbl_fs)
        ax.set_yticklabels(y_labels if (p_idx % ncols == 0) else [], fontsize=9.0)

        region_str = ref_info.get("region", "")
        reg_annot = f" ({region_str})" if region_str else ""
        ax.set_title(
            f"{ref_title}{reg_annot}\nN={n_ref} Clusters",
            fontsize=10.0,
            pad=8,
            fontweight="normal",
        )

        cell_fs = 8.0 if n_fc <= 5 else 6.8
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
                    fontsize=cell_fs,
                    fontweight="normal",
                    linespacing=1.15,
                )

    for empty_idx in range(n_plots, len(axes)):
        axes[empty_idx].set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.93, 0.96 if suptitle else 0.98])
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.0, y=0.99, fontweight="normal")
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
    title: str | None = None,
):
    """Plot public benchmark cluster recovery scorecard across references."""
    from matplotlib.colors import Normalize

    names_map = display_names or {}
    n_refs = len(scorecard_df)
    n_cands = len(candidate_keys)

    fig, ax = plt.subplots(
        figsize=(max(13.5, 1.4 * n_cands), max(5.0, 0.75 * n_refs + 1.8))
    )

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

    is_summary_arr = (scorecard_df["ref_type"] == "Summary").to_numpy(bool)
    first_sum = np.where(is_summary_arr)[0]
    if len(first_sum) > 0:
        ax.axhline(first_sum[0] - 0.5, color="#374151", linewidth=1.5)

    row_labels = []
    for _, row in scorecard_df.iterrows():
        if row["ref_type"] == "Summary":
            row_labels.append(f"{row['ref_name']}\n({row['ref_region']})")
        else:
            row_labels.append(
                f"{row['ref_name']} [{row['ref_type']}]\n({row['ref_region']}, N={row['n_ref']})"
            )
    col_labels = [names_map.get(k, k) for k in candidate_keys]

    ax.set_xticks(np.arange(n_cands))
    ax.set_yticks(np.arange(n_refs))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=9.5)
    ax.set_yticklabels(row_labels, fontsize=9.0)

    # Separate adjacent catalog families without changing their order.
    for j in range(1, n_cands):
        if candidate_keys[j].startswith("rz_diff") != candidate_keys[j - 1].startswith(
            "rz_diff"
        ):
            ax.axvline(j - 0.5, color="#374151", linewidth=1.8)

    for i in range(n_refs):
        is_row_sum = is_summary_arr[i]
        for j in range(n_cands):
            pct_val = pct_matrix[i, j]
            cnt_val = counts_matrix[i, j]
            tot_val = n_ref_list[i]
            d_val = delta_matrix[i, j]

            norm_val = norm(pct_val)
            text_color = "white" if norm_val > 0.58 else "black"

            if is_row_sum:
                if candidate_keys[j] == baseline_key:
                    cell_txt = f"{pct_val:.1f}%\n[Baseline]"
                else:
                    cell_txt = f"{pct_val:.1f}%\nΔ: {d_val:+.1f}%"
            else:
                if candidate_keys[j] == baseline_key:
                    cell_txt = f"{pct_val:.1f}%\n({cnt_val}/{tot_val})\n[Baseline]"
                else:
                    cell_txt = (
                        f"{pct_val:.1f}%\n({cnt_val}/{tot_val})\nΔ: {d_val:+.1f}%"
                    )

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

    scorecard_title = title or (
        "Public Benchmark Cluster Recovery Scorecard across Overlapping HSC Footprints\n"
        "(Matching within 0.5 Mpc/h Physical Transverse Radius)"
    )
    ax.set_title(
        scorecard_title,
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


def plot_benchmark_release_table(
    scorecard_df: pd.DataFrame,
    candidate_keys: list[str],
    save_path: Path,
    r_phys_mpc_h: float,
    sample_label: str,
    baseline_key: str = "camira_1bin",
    display_names: dict[str, str] | None = None,
):
    """Render a release-style table, highlighting exact maxima including ties.

    Empty reference samples have undefined rates and receive no highlight.
    """
    from itertools import groupby
    from textwrap import fill

    from matplotlib.patches import Rectangle

    if scorecard_df.empty or not candidate_keys:
        return

    names = display_names or {}
    rates = scorecard_df[[f"{key}_pct" for key in candidate_keys]].to_numpy(float)
    totals = scorecard_df["n_ref"].to_numpy()
    valid = np.isfinite(rates) & (totals[:, None] > 0)
    maxima = np.max(np.where(valid, rates, -np.inf), axis=1)
    winners = valid & (rates == maxima[:, None])
    background, ink, muted = "#FAF9F6", "#272923", "#77786F"
    accent, highlight, rule = "#985738", "#EAD7C7", "#D9D7CF"
    label_width, col_width, row_height = 3.65, 1.65, 1.10
    width = label_width + col_width * len(candidate_keys)
    table_top = 3.05
    bottom = table_top + len(scorecard_df) * row_height
    height = bottom + 1.05

    with plt.rc_context(
        {"font.family": "DejaVu Sans", "font.weight": "normal", "text.usetex": False}
    ):
        fig, ax = plt.subplots(figsize=(width, height), facecolor=background)
        fig.subplots_adjust(left=0.035, right=0.965, bottom=0.035, top=0.965)
        ax.set(xlim=(0, width), ylim=(height, 0))
        ax.set_axis_off()

        def write(x, y, text, size=11, color=ink, ha="left", family="DejaVu Sans"):
            return ax.text(
                x,
                y,
                text,
                fontsize=size,
                color=color,
                ha=ha,
                va="center",
                fontweight="normal",
                fontfamily=family,
                linespacing=1.4,
            )

        ax.hlines(0.04, 0, width, color=ink, linewidth=1.1)
        write(0, 0.32, "HSC  /  CATALOG COMPARISON", size=9, color=accent)
        write(0, 0.91, "External catalog recovery", size=32, family="Source Serif 4")
        write(0, 1.44, sample_label, size=10, color=muted)
        write(width, 0.32, "MATCH FRACTION (%)", size=9, color=muted, ha="right")
        ax.add_patch(
            Rectangle(
                (width - 3.25, 1.32), 0.20, 0.20, facecolor=highlight, edgecolor="none"
            )
        )
        write(width - 2.93, 1.42, "Highest in row · ties included", size=9, color=muted)

        # Group contiguous catalog families without changing the scorecard order.
        grouped = groupby(
            enumerate(candidate_keys), key=lambda item: item[1].startswith("rz_diff")
        )
        for is_rz, members in grouped:
            indices = [index for index, _ in members]
            left = label_width + indices[0] * col_width
            right = label_width + (indices[-1] + 1) * col_width
            ax.add_patch(
                Rectangle(
                    (left, 1.91),
                    right - left,
                    1.14,
                    facecolor="#F0EEE7" if is_rz else "#F5F4EF",
                    edgecolor="none",
                )
            )
            write(
                (left + right) / 2,
                2.10,
                "r-z DIFF VARIANTS" if is_rz else "OPTICAL CATALOGS",
                size=9,
                color=accent if is_rz else muted,
                ha="center",
            )
            ax.hlines(2.30, left + 0.18, right - 0.18, color=rule, linewidth=0.65)
        write(0, 2.45, "External catalog", size=12)
        write(0, 2.79, "Selection / footprint / sample size", size=9, color=muted)
        for j, key in enumerate(candidate_keys):
            x = label_width + (j + 0.5) * col_width
            label = names.get(key, key)
            if key.startswith("rz_diff"):
                label = label.removeprefix("r-z Diff ").strip("()")
            label = fill(label, width=17)
            write(x, 2.59, label, size=11, ha="center")
            if key == baseline_key:
                write(x, 2.89, "Baseline", size=8, color=muted, ha="center")
        ax.hlines(table_top, 0, width, color=ink, linewidth=0.85)

        for i in range(len(scorecard_df)):
            top = table_top + i * row_height
            ref_type = str(scorecard_df["ref_type"].iloc[i])
            is_summary = ref_type == "Summary"
            prev_type = str(scorecard_df["ref_type"].iloc[i - 1]) if i > 0 else ""

            if i and is_summary and prev_type != "Summary":
                ax.hlines(top, 0, width, color=ink, linewidth=1.2)
            elif i and ref_type != prev_type:
                ax.hlines(top, 0, width, color="#A8A79D", linewidth=0.9)

            write(0, top + 0.23, ref_type.upper(), size=8, color=accent)
            write(0, top + 0.54, str(scorecard_df["ref_name"].iloc[i]), size=12)
            if is_summary:
                detail = f"{scorecard_df['ref_region'].iloc[i]} · unweighted mean"
            else:
                detail = (
                    f"{scorecard_df['ref_region'].iloc[i]}  ·  N = {int(totals[i]):,}"
                )
            write(0, top + 0.86, detail, size=9, color=muted)
            for j, key in enumerate(candidate_keys):
                left = label_width + j * col_width
                x = left + col_width / 2
                if winners[i, j]:
                    ax.add_patch(
                        Rectangle(
                            (left + 0.05, top + 0.06),
                            col_width - 0.10,
                            row_height - 0.12,
                            facecolor=highlight,
                            edgecolor="none",
                        )
                    )
                if not valid[i, j]:
                    write(x, top + 0.48, "—", size=18, color=muted, ha="center")
                    continue
                write(
                    x,
                    top + 0.43,
                    f"{rates[i, j]:.1f}%",
                    size=18,
                    color=ink,
                    ha="center",
                )
                delta = float(scorecard_df[f"{key}_delta_pct"].iloc[i])
                if is_summary:
                    if key == baseline_key:
                        detail = "Baseline"
                    else:
                        detail = f"{delta:+.1f} pp"
                else:
                    count = int(scorecard_df[f"{key}_count"].iloc[i])
                    detail = f"{count}/{int(totals[i])}"
                    if key != baseline_key:
                        detail += f" · {delta:+.1f} pp"
                write(
                    x,
                    top + 0.79,
                    detail,
                    size=8.5,
                    color="#675546" if winners[i, j] else muted,
                    ha="center",
                )
            ax.hlines(top + row_height, 0, width, color=rule, linewidth=0.55)
        ax.hlines(bottom, 0, width, color=ink, linewidth=0.85)

        baseline = names.get(baseline_key, baseline_key)
        write(
            0,
            bottom + 0.36,
            f"Physical transverse radius < {r_phys_mpc_h:g} Mpc/h. "
            f"Cell details: matched / reference count; percentage-point difference from {baseline}.",
            size=9,
            color=muted,
        )
        write(
            0,
            bottom + 0.70,
            "Highlights use unrounded match fractions. Empty reference samples are shown as —.",
            size=9,
            color=muted,
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor=background)
        print(f"Release-style benchmark table saved to {save_path}")
        plt.show()
        plt.close(fig)


def plot_benchmark_variant_delta(
    scorecard_df_a: pd.DataFrame,
    scorecard_df_b: pd.DataFrame,
    candidate_keys: list[str],
    save_path: Path,
    label_a: str = "Stratified",
    label_b: str = "Raw",
    display_names: dict[str, str] | None = None,
    title: str | None = None,
):
    """Plot heatmap of percentage-point shift between two benchmark variants (A - B)."""
    from matplotlib.colors import TwoSlopeNorm

    names_map = display_names or {}

    keys_b = set(scorecard_df_b["ref_key"].values)
    matched_keys = [k for k in scorecard_df_a["ref_key"].values if k in keys_b]
    if not matched_keys:
        return

    sub_a = scorecard_df_a.set_index("ref_key").loc[matched_keys]
    sub_b = scorecard_df_b.set_index("ref_key").loc[matched_keys]

    n_refs = len(matched_keys)
    n_cands = len(candidate_keys)

    delta_matrix = np.zeros((n_refs, n_cands), dtype=float)
    pct_a_mat = np.zeros((n_refs, n_cands), dtype=float)
    pct_b_mat = np.zeros((n_refs, n_cands), dtype=float)

    for i, r_k in enumerate(matched_keys):
        for j, c_k in enumerate(candidate_keys):
            p_a = float(sub_a.loc[r_k, f"{c_k}_pct"])
            p_b = float(sub_b.loc[r_k, f"{c_k}_pct"])
            delta_matrix[i, j] = p_a - p_b
            pct_a_mat[i, j] = p_a
            pct_b_mat[i, j] = p_b

    fig, ax = plt.subplots(
        figsize=(max(13.5, 1.4 * n_cands), max(5.0, 0.75 * n_refs + 1.8))
    )

    abs_max = max(5.0, float(np.nanmax(np.abs(delta_matrix))))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    cmap = plt.colormaps["coolwarm"]

    im = ax.imshow(delta_matrix, cmap=cmap, norm=norm, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(f"Recovery Shift ({label_a} − {label_b}, pp)", fontsize=10.0)

    is_summary_arr = (sub_a["ref_type"] == "Summary").values
    first_sum = np.where(is_summary_arr)[0]
    if len(first_sum) > 0:
        ax.axhline(first_sum[0] - 0.5, color="#374151", linewidth=1.4)

    row_labels = []
    for _, row in sub_a.iterrows():
        if row["ref_type"] == "Summary":
            row_labels.append(f"{row['ref_name']}\n({row['ref_region']})")
        else:
            row_labels.append(
                f"{row['ref_name']} [{row['ref_type']}]\n({row['ref_region']}, N={row['n_ref']})"
            )

    col_labels = [names_map.get(k, k) for k in candidate_keys]
    ax.set_xticks(np.arange(n_cands))
    ax.set_yticks(np.arange(n_refs))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=9.5)
    ax.set_yticklabels(row_labels, fontsize=9.0)

    for i in range(n_refs):
        for j in range(n_cands):
            d_val = delta_matrix[i, j]
            p_a = pct_a_mat[i, j]
            p_b = pct_b_mat[i, j]

            norm_val = norm(d_val)
            text_color = "white" if (norm_val < 0.20 or norm_val > 0.80) else "black"

            cell_txt = f"{d_val:+.1f} pp\n({p_a:.1f}% vs {p_b:.1f}%)"
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

    plt_title = title or (
        f"Benchmark Recovery Rate Shift: {label_a} vs. {label_b}\n"
        "(Positive values indicate net recovery gain after equalizing redshift distributions)"
    )
    ax.set_title(plt_title, fontsize=11.0, pad=12, fontweight="normal")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Variant delta plot saved to {save_path}")
    plt.show()
    plt.close(fig)


def run_benchmark_comparison_suite(
    reference_dict: dict[str, pd.DataFrame | Table],
    lens_dict: dict[str, Table],
    candidate_order: list[str],
    first_class_keys: list[str],
    rz_keys: list[str],
    output_scorecard_png: Path,
    output_release_table_png: Path,
    output_diff_heatmaps_png: Path | None = None,
    baseline_key: str = "camira_1bin",
    r_phys_mpc_h: float = 0.5,
    ref_metadata: dict[str, dict[str, str]] | None = None,
    display_names: dict[str, str] | None = None,
    scorecard_title: str | None = None,
    sample_label: str | None = None,
    diff_suptitle: str | None = None,
    release_table_candidate_order: list[str] | None = None,
    include_summary_rows: bool = True,
    summary_groups: tuple[str, ...] = ("non_optical", "optical", "overall"),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Execute complete benchmark suite: matching, scorecard, release table, and diff heatmaps."""
    all_matches = compute_reference_matches_dict(
        reference_dict,
        lens_dict,
        r_phys_mpc_h=r_phys_mpc_h,
    )

    scorecard_df = compute_global_benchmark_scorecard(
        reference_dict,
        all_matches,
        candidate_order=candidate_order,
        baseline_key=baseline_key,
        ref_metadata=ref_metadata,
        include_summary_rows=include_summary_rows,
        summary_groups=summary_groups,
    )

    print_scorecard_markdown_table(
        scorecard_df,
        candidate_order=candidate_order,
        display_names=display_names or {},
        baseline_key=baseline_key,
    )

    plot_benchmark_scorecard(
        scorecard_df,
        candidate_keys=candidate_order,
        save_path=output_scorecard_png,
        baseline_key=baseline_key,
        display_names=display_names,
        title=scorecard_title,
    )

    plot_benchmark_release_table(
        scorecard_df,
        candidate_keys=release_table_candidate_order or candidate_order,
        save_path=output_release_table_png,
        r_phys_mpc_h=r_phys_mpc_h,
        sample_label=sample_label
        or f"Overlapping HSC footprints · {REDSHIFT_RANGE[0]:.2f} ≤ z ≤ {REDSHIFT_RANGE[1]:.2f}",
        baseline_key=baseline_key,
        display_names=display_names,
    )

    diff_matrices = {
        ref_k: compute_differential_advantage(
            all_matches[ref_k],
            first_class_keys=first_class_keys,
            rz_keys=rz_keys,
            n_ref=len(reference_dict[ref_k]),
        )
        for ref_k in reference_dict
    }

    if output_diff_heatmaps_png is not None:
        plot_differential_advantage_heatmaps(
            diff_matrices,
            first_class_keys=first_class_keys,
            rz_keys=rz_keys,
            save_path=output_diff_heatmaps_png,
            ref_keys=list(reference_dict.keys()),
            display_names=display_names,
            ref_metadata=ref_metadata,
            suptitle=diff_suptitle,
        )

    return scorecard_df, diff_matrices


# %% Global Configuration

REDSHIFT_RANGE = (0.19, 0.52)
MATCH_RADIUS_MPC_H = 0.5
REFERENCE_SAMPLE_LABEL = f"Y3 · X-ray / SZ / WL: no z cuts · Optical: {REDSHIFT_RANGE[0]:.2f} ≤ z ≤ {REDSHIFT_RANGE[1]:.2f}"

FIRST_CLASS_KEYS = [
    "camira_1bin",
    "redm_r16_1bin",
    "wh24_1bin",
    "zou21_1bin",
    "yang21_mass_1bin",
    "yang21_richness_1bin",
    "clumpr_mass_1bin",
    "clumpr_richness_1bin",
    "amico_1bin",
]

RZ_DIFF_KEYS = [
    # Inner radius: 0.21-0.30 Mpc/h.
    "rz_diff_single_box_1bin",
    "rz_diff_1bin",
    "rz_diff_six_param_1bin",
    "rz_diff_preset_1bin",
    "rz_diff_two_box_red_1bin",
    "rz_diff_single_box_match_recovery_1bin",
    # Inner radius: 0.52-0.57 Mpc/h.
    "rz_diff_two_box_red_redmapper_matching_1bin",
    "rz_diff_six_param_match_recovery_1bin",
    "rz_diff_two_box_match_recovery_1bin",
    "rz_diff_two_box_red_match_recovery_1bin",
    "rz_diff_two_box_red_camira_matching_1bin",
]

LABELS_TO_COMPARE = FIRST_CLASS_KEYS + RZ_DIFF_KEYS

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

OUTPUT_BOKEH_HTML = project_root / "output/plots_for_agents/spatial_distribution.html"
OUTPUT_SCORECARD_PNG = (
    project_root / "output/plots_for_agents/reference_benchmark_scorecard.png"
)
OUTPUT_RELEASE_TABLE_PNG = (
    project_root / "output/plots_for_agents/reference_benchmark_release_table.png"
)
OUTPUT_DIFF_HEATMAPS_ALL_PNG = (
    project_root / "output/plots_for_agents/differential_advantage_all_references.png"
)
OUTPUT_STRATIFIED_SCORECARD_PNG = (
    project_root
    / "output/plots_for_agents/reference_benchmark_scorecard_stratified.png"
)
OUTPUT_STRATIFIED_RELEASE_TABLE_PNG = (
    project_root
    / "output/plots_for_agents/reference_benchmark_release_table_stratified.png"
)
OUTPUT_DIFF_HEATMAPS_STRATIFIED_PNG = (
    project_root / "output/plots_for_agents/differential_advantage_stratified.png"
)
OUTPUT_STRATIFIED_DELTA_PNG = (
    project_root
    / "output/plots_for_agents/reference_benchmark_stratified_vs_raw_delta.png"
)
# %% [Stage 1: Load Lens and External Catalogs]

dfs_dict = load_lens_data(LABELS_TO_COMPARE, project_root)
reference_dfs = load_external_reference_catalogs(project_root, REDSHIFT_RANGE)
print(
    f"Loaded {len(reference_dfs)} external catalogs: 5 physical probes and 3 optical catalogs"
)


# %% [Stage 2: Interactive Regional Spatial Web Visualizer (Bokeh HTML)]

HTML_MAIN_KEYS = tuple(LABELS_TO_COMPARE)
HTML_GROUPS = {
    "Optical candidate catalogs": (
        "camira_1bin",
        "redm_r16_1bin",
        "wh24_1bin",
        "zou21_1bin",
        "yang21_mass_1bin",
        "yang21_richness_1bin",
        "clumpr_mass_1bin",
        "clumpr_richness_1bin",
    ),
    "AMICO / RZ diff": ("amico_1bin", *RZ_DIFF_KEYS),
    "External reference catalogs": (
        "act_dr6",
        "erass3",
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
    "wh24_1bin": dict(
        color="#2E7D32",
        shape="triangle",
        diameter=0.19,
        line_width=1.8,
        alpha=0.9,
        visible=False,
    ),
    "zou21_1bin": dict(
        color="#CCBB44",
        shape="hexagon",
        diameter=0.20,
        line_width=1.8,
        alpha=0.9,
        visible=False,
    ),
    "yang21_mass_1bin": dict(
        color="#0288D1",
        shape="diamond",
        diameter=0.20,
        line_width=1.8,
        alpha=0.9,
        visible=False,
    ),
    "yang21_richness_1bin": dict(
        color="#303F9F",
        shape="diamond",
        diameter=0.20,
        line_width=1.8,
        alpha=0.9,
        visible=False,
    ),
    "clumpr_mass_1bin": dict(
        color="#8E24AA",
        shape="circle",
        diameter=0.20,
        line_width=1.8,
        alpha=0.9,
        visible=False,
    ),
    "clumpr_richness_1bin": dict(
        color="#E65100",
        shape="circle",
        diameter=0.20,
        line_width=1.8,
        alpha=0.9,
        visible=False,
    ),
    "amico_1bin": dict(
        color="#10B981",
        shape="diamond",
        diameter=0.20,
        line_width=1.9,
        alpha=0.95,
        visible=True,
    ),
    "rz_diff_1bin": dict(
        color="#D97706",
        shape="inverted_triangle",
        diameter=0.21,
        line_width=1.7,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_preset_1bin": dict(
        color="#C7682E",
        shape="inverted_triangle",
        diameter=0.24,
        line_width=1.9,
        alpha=0.95,
        visible=True,
    ),
    "rz_diff_single_box_1bin": dict(
        color="#EA580C",
        shape="inverted_triangle",
        diameter=0.22,
        line_width=1.7,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_two_box_red_1bin": dict(
        color="#B91C1C",
        shape="inverted_triangle",
        diameter=0.23,
        line_width=1.7,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_two_box_match_recovery_1bin": dict(
        color="#7B2CBF",
        shape="triangle",
        diameter=0.23,
        line_width=1.8,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_single_box_match_recovery_1bin": dict(
        color="#A16207",
        shape="triangle",
        diameter=0.23,
        line_width=1.8,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_two_box_red_match_recovery_1bin": dict(
        color="#0F766E",
        shape="triangle",
        diameter=0.23,
        line_width=1.8,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_two_box_red_redmapper_matching_1bin": dict(
        color="#CC79A7",
        shape="triangle",
        diameter=0.23,
        line_width=1.8,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_two_box_red_camira_matching_1bin": dict(
        color="#56B4E9",
        shape="triangle",
        diameter=0.23,
        line_width=1.8,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_six_param_1bin": dict(
        color="#2563EB",
        shape="triangle",
        diameter=0.23,
        line_width=1.8,
        alpha=0.95,
        visible=False,
    ),
    "rz_diff_six_param_match_recovery_1bin": dict(
        color="#334155",
        shape="triangle",
        diameter=0.23,
        line_width=1.8,
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
    "erass3": dict(
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
    display_names=DISPLAY_NAMES,
)


# %% [Stage 3: Reference Benchmark Suite (Raw Top 1020 Candidates)]

raw_scorecard_df, raw_diff_matrices = run_benchmark_comparison_suite(
    reference_dict=reference_dfs,
    lens_dict=dfs_dict,
    candidate_order=FIRST_CLASS_KEYS + RZ_DIFF_KEYS,
    first_class_keys=FIRST_CLASS_KEYS,
    rz_keys=RZ_DIFF_KEYS,
    output_scorecard_png=OUTPUT_SCORECARD_PNG,
    output_release_table_png=OUTPUT_RELEASE_TABLE_PNG,
    output_diff_heatmaps_png=OUTPUT_DIFF_HEATMAPS_ALL_PNG,
    baseline_key="camira_1bin",
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
    ref_metadata=EXTERNAL_REFERENCE_METADATA,
    display_names=DISPLAY_NAMES,
    sample_label=REFERENCE_SAMPLE_LABEL,
    scorecard_title="External Catalog Recovery (Raw Top 1020)",
    include_summary_rows=True,
    summary_groups=("non_optical", "optical", "overall"),
)


# %% [Stage 4: Stratified Redshift-Controlled Benchmark Suite (Equal P(z))]

N_STRATIFIED_BINS = 10
TOTAL_STRATIFIED_TOP_N = 1020

candidate_order = FIRST_CLASS_KEYS + RZ_DIFF_KEYS

stratified_lens_dict = load_stratified_candidates(
    candidate_keys=FIRST_CLASS_KEYS + RZ_DIFF_KEYS,
    root=project_root,
    redshift_range=REDSHIFT_RANGE,
    n_bins=N_STRATIFIED_BINS,
    total_top_n=TOTAL_STRATIFIED_TOP_N,
    ref_lens_table=dfs_dict.get("redm_r16_1bin"),
    include_all_candidates=False,
)

print(
    f"\n=== Stratified Redshift-Controlled Benchmark Suite (N_bins={N_STRATIFIED_BINS}, Matched to redMaPPer N(z), Top {TOTAL_STRATIFIED_TOP_N}) ==="
)

stratified_scorecard_df, stratified_diff_matrices = run_benchmark_comparison_suite(
    reference_dict=reference_dfs,
    lens_dict=stratified_lens_dict,
    candidate_order=candidate_order,
    release_table_candidate_order=candidate_order,
    first_class_keys=FIRST_CLASS_KEYS,
    rz_keys=RZ_DIFF_KEYS,
    output_scorecard_png=OUTPUT_STRATIFIED_SCORECARD_PNG,
    output_release_table_png=OUTPUT_STRATIFIED_RELEASE_TABLE_PNG,
    output_diff_heatmaps_png=OUTPUT_DIFF_HEATMAPS_STRATIFIED_PNG,
    baseline_key="camira_1bin",
    r_phys_mpc_h=MATCH_RADIUS_MPC_H,
    ref_metadata=EXTERNAL_REFERENCE_METADATA,
    display_names=DISPLAY_NAMES,
    scorecard_title=(
        f"Stratified Redshift-Controlled Benchmark Recovery Scorecard (Equal P(z), Top {TOTAL_STRATIFIED_TOP_N})\n"
        f"(Matching within {MATCH_RADIUS_MPC_H:g} Mpc/h Physical Transverse Radius)"
    ),
    sample_label=f"Stratified lens N(z) · {REFERENCE_SAMPLE_LABEL}",
    diff_suptitle="Stratified Redshift-Controlled Differential Advantage (rz_diff variants vs Class 1)",
    include_summary_rows=True,
    summary_groups=("non_optical", "optical", "overall"),
)


# %% [Stage 5: Cross-Variant Delta Analysis (Stratified vs. Raw)]

common_candidate_order = [
    k
    for k in FIRST_CLASS_KEYS + RZ_DIFF_KEYS
    if f"{k}_pct" in raw_scorecard_df.columns
    and f"{k}_pct" in stratified_scorecard_df.columns
]

plot_benchmark_variant_delta(
    scorecard_df_a=stratified_scorecard_df,
    scorecard_df_b=raw_scorecard_df,
    candidate_keys=common_candidate_order,
    save_path=OUTPUT_STRATIFIED_DELTA_PNG,
    label_a="Stratified",
    label_b="Raw",
    display_names=DISPLAY_NAMES,
)
