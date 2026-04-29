import colorsys

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


def load_result_tables(base_dir):
    tables = []
    for i in range(4):
        path = base_dir / f"hsc_hsc_lens{i}_lens.csv"
        tables.append(Table.read(path, format="csv"))
    return tables


def _label_variant_color(base_color, label_index, n_labels):
    from matplotlib.colors import to_hex, to_rgb

    lightness_spread = 0.20
    saturation_spread = 0.35

    if n_labels <= 1:
        return to_hex(to_rgb(base_color))

    rgb = to_rgb(base_color)
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)

    label_positions = np.linspace(-1.0, 1.0, n_labels)
    position = float(label_positions[label_index])

    # Keep the bin hue fixed, but vary brightness and saturation across labels.
    # The ranges are heuristic and chosen to stay readable on a white background.
    lightness = np.clip(lightness + lightness_spread * position, 0.16, 0.86)
    saturation = np.clip(
        saturation * (1.0 - saturation_spread * abs(position)), 0.22, 1.0
    )

    return to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))


def plot_radial_profile(
    tables,
    value_column,
    title_label,
    multiply_by_radius=True,
    use_spline=True,
    error_column="ds_err",
    use_log_y=None,
    reference_line_y=0.0,
    y_label=None,
    title_suffix=None,
    ax_list=None,
    label_text=None,
    label_index=0,
    n_labels=1,
    marker="o",
):
    from scipy.interpolate import make_interp_spline

    if ax_list is None:
        fig, axes = plt.subplots(4, 1, figsize=(8.6, 13.2), sharex=True, sharey=False)
        axes = np.atleast_1d(axes)
    else:
        axes = ax_list
        fig = axes[0].get_figure()

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if use_log_y is None:
        use_log_y = not multiply_by_radius

    for i, table in enumerate(tables):
        ax = axes[i]
        rp = np.asarray(table["rp"], dtype=float)
        value = np.asarray(table[value_column], dtype=float)
        value_err = np.asarray(table[error_column], dtype=float)

        if multiply_by_radius:
            plot_y = rp * value
            plot_yerr = rp * value_err
        else:
            plot_y = value
            plot_yerr = value_err

        order = np.argsort(rp)
        rp = rp[order]
        plot_y = plot_y[order]
        plot_yerr = plot_yerr[order]

        base_color = palette[i % len(palette)]
        current_color = _label_variant_color(base_color, label_index, n_labels)
        valid = (
            np.isfinite(rp) & np.isfinite(plot_y) & np.isfinite(plot_yerr) & (rp > 0)
        )
        rp = rp[valid]
        plot_y = plot_y[valid]
        plot_yerr = np.clip(plot_yerr[valid], 0.0, np.inf)

        if len(rp) == 0:
            if ax_list is None:
                ax.set_title(f"Lens bin {i} (no valid data)")
                ax.set_xscale("log")
                ax.grid(alpha=0.2, which="both")
            continue

        if not use_spline:
            if not use_log_y:
                if ax_list is None:
                    y_for_lim = np.concatenate([plot_y - plot_yerr, plot_y + plot_yerr])
                    y_for_lim = y_for_lim[np.isfinite(y_for_lim)]
                    if y_for_lim.size > 0:
                        y_min = float(np.nanmin(y_for_lim))
                        y_max = float(np.nanmax(y_for_lim))
                        if np.isclose(y_min, y_max):
                            pad = 0.08 * max(abs(y_min), 1.0)
                        else:
                            pad = 0.08 * (y_max - y_min)
                        ax.set_ylim(y_min - pad, y_max + pad)

                ax.errorbar(
                    rp,
                    plot_y,
                    yerr=plot_yerr,
                    fmt=marker,
                    ms=5.0,
                    ls="none",
                    elinewidth=1.2,
                    capsize=3.0,
                    color=current_color,
                    alpha=0.95,
                    label=label_text if i == 0 else None,
                )
                ax.axhline(
                    reference_line_y, color="0.35", lw=1.0, ls="--", zorder=0
                )
            else:
                pos = plot_y > 0
                rp_pos = rp[pos]
                y_pos = plot_y[pos]
                yerr_pos = plot_yerr[pos]
                if len(rp_pos) == 0:
                    if ax_list is None:
                        ax.set_title(f"Lens bin {i} (no positive values for log y)")
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        ax.grid(alpha=0.2, which="both")
                    continue

                yerr_pos = np.minimum(yerr_pos, 0.95 * y_pos)
                ax.errorbar(
                    rp_pos,
                    y_pos,
                    yerr=yerr_pos,
                    fmt=marker,
                    ms=5.0,
                    ls="none",
                    elinewidth=1.2,
                    capsize=3.0,
                    color=current_color,
                    alpha=0.95,
                    label=label_text if i == 0 else None,
                )
                ax.set_yscale("log")

            ax.set_xscale("log")
            ax.grid(alpha=0.2, which="both")
            if ax_list is None:
                ax.set_title(f"Lens bin {i}")
            continue

        if not use_log_y:
            x_dense = np.logspace(np.log10(rp.min()), np.log10(rp.max()), 300)
            x_log = np.log10(rp)
            if len(rp) >= 2:
                spline_k = min(3, len(rp) - 1)
                smooth = make_interp_spline(x_log, plot_y, k=spline_k)
                smooth_err = make_interp_spline(x_log, plot_yerr, k=spline_k)
                y_dense = smooth(np.log10(x_dense))
                yerr_dense = np.clip(smooth_err(np.log10(x_dense)), 0.0, np.inf)
            else:
                y_dense = np.full_like(x_dense, plot_y[0])
                yerr_dense = np.full_like(x_dense, plot_yerr[0])

            ax.fill_between(
                x_dense,
                y_dense - yerr_dense,
                y_dense + yerr_dense,
                color=current_color,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(x_dense, y_dense, color=current_color, lw=2.0)

            ax.plot(
                rp,
                plot_y,
                color=current_color,
                marker=marker,
                ms=4.5,
                lw=0.0,
                label=label_text if i == 0 else None,
            )

            if ax_list is None:
                y_for_lim = np.concatenate([plot_y - plot_yerr, plot_y + plot_yerr])
                y_for_lim = y_for_lim[np.isfinite(y_for_lim)]
                if y_for_lim.size > 0:
                    y_min = float(np.nanmin(y_for_lim))
                    y_max = float(np.nanmax(y_for_lim))
                    if np.isclose(y_min, y_max):
                        pad = 0.08 * max(abs(y_min), 1.0)
                    else:
                        pad = 0.08 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)

            ax.axhline(reference_line_y, color="0.35", lw=1.0, ls="--", zorder=0)
        else:
            pos = plot_y > 0
            rp_pos = rp[pos]
            y_pos = plot_y[pos]
            yerr_pos = plot_yerr[pos]

            if len(rp_pos) == 0:
                if ax_list is None:
                    ax.set_title(f"Lens bin {i} (no positive values for log y)")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.grid(alpha=0.2, which="both")
                continue

            x_dense = np.logspace(np.log10(rp_pos.min()), np.log10(rp_pos.max()), 300)
            x_log = np.log10(rp_pos)

            if len(rp_pos) >= 2:
                spline_k = min(3, len(rp_pos) - 1)
                y_log = np.log10(y_pos)
                smooth_log = make_interp_spline(x_log, y_log, k=spline_k)
                y_dense = np.power(10.0, smooth_log(np.log10(x_dense)))

                frac_err = np.divide(
                    yerr_pos, y_pos, out=np.zeros_like(yerr_pos), where=y_pos > 0
                )
                dlog = np.log10(1.0 + np.clip(frac_err, 0.0, np.inf))
                smooth_dlog = make_interp_spline(x_log, dlog, k=spline_k)
                dlog_dense = np.clip(smooth_dlog(np.log10(x_dense)), 0.0, np.inf)
                ylo_dense = y_dense / np.power(10.0, dlog_dense)
                yhi_dense = y_dense * np.power(10.0, dlog_dense)
            else:
                y_dense = np.full_like(x_dense, y_pos[0])
                frac_err0 = yerr_pos[0] / y_pos[0] if y_pos[0] > 0 else 0.0
                dlog0 = np.log10(1.0 + max(frac_err0, 0.0))
                ylo_dense = y_dense / np.power(10.0, dlog0)
                yhi_dense = y_dense * np.power(10.0, dlog0)

            ylo_dense = np.clip(ylo_dense, np.finfo(float).tiny, np.inf)
            yhi_dense = np.clip(yhi_dense, np.finfo(float).tiny, np.inf)

            ax.fill_between(
                x_dense,
                ylo_dense,
                yhi_dense,
                color=current_color,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(x_dense, y_dense, color=current_color, lw=2.0)

            ax.plot(
                rp_pos,
                y_pos,
                color=current_color,
                marker=marker,
                ms=4.5,
                lw=0.0,
                label=label_text if i == 0 else None,
            )

            ax.set_yscale("log")

        ax.set_xscale("log")
        ax.grid(alpha=0.2, which="both")
        if ax_list is None:
            ax.set_title(f"Lens bin {i}")

    if y_label is not None:
        ylabel = y_label
    elif multiply_by_radius:
        ylabel = r"$R \times \Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$"
    elif use_log_y:
        ylabel = r"$\Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$"
    else:
        ylabel = value_column

    if title_suffix is None:
        if multiply_by_radius:
            title_suffix = r"$R \times \Delta\Sigma$ Profiles"
        elif use_log_y:
            title_suffix = r"$\Delta\Sigma$ Profiles"
        else:
            title_suffix = f"{value_column} Profiles"

    for ax in axes:
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel(r"$R\ [\mathrm{Mpc}]$")

    if ax_list is None:
        fig.suptitle(rf"{title_label} {title_suffix}", y=0.996)
        fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.985])
    else:
        for i, ax in enumerate(axes):
            ax.set_title(f"Lens bin {i}")

    if ax_list is None:
        plt.show()

    return fig
