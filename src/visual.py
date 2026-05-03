import colorsys

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import re


def load_result_tables(base_dir):
    # Pattern to find hsc_hsc_lens[N]_lens.fits or .csv
    # We use a dict to store the best file path for each index
    # key: index (int), value: path (Path)
    bin_files = {}

    # Find all matching files in the directory
    for path in base_dir.glob("hsc_hsc_lens*_lens.*"):
        match = re.search(r"hsc_hsc_lens(\d+)_lens\.(fits|csv)$", path.name)
        if match:
            idx = int(match.group(1))
            ext = match.group(2).lower()

            # Priority: FITS > CSV
            if idx not in bin_files or ext == "fits":
                bin_files[idx] = path

    if not bin_files:
        raise FileNotFoundError(f"No lens bin files found in {base_dir}")

    # Sort indices to ensure order (0, 1, 2, ...)
    sorted_indices = sorted(bin_files.keys())
    tables = []
    for idx in sorted_indices:
        path = bin_files[idx]
        if path.suffix.lower() == ".fits":
            tables.append(Table.read(path, hdu="PROFILE"))
        else:
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


def _plot_errorbar_style(
    ax, rp, plot_y, plot_yerr, color, marker, label, use_log_y, reference_line_y=0.0
):
    if not use_log_y:
        ax.errorbar(
            rp,
            plot_y,
            yerr=plot_yerr,
            fmt=marker,
            ms=5.0,
            ls="none",
            elinewidth=1.2,
            capsize=3.0,
            color=color,
            alpha=0.95,
            label=label,
        )
    else:
        pos = plot_y > 0
        if not np.any(pos):
            return False
        # Use asymmetric error bars on log scale to avoid clipping the upper error.
        yerr_lower = np.minimum(plot_yerr[pos], 0.99 * plot_y[pos])
        yerr_upper = plot_yerr[pos]
        ax.errorbar(
            rp[pos],
            plot_y[pos],
            yerr=[yerr_lower, yerr_upper],
            fmt=marker,
            ms=5.0,
            ls="none",
            elinewidth=1.2,
            capsize=3.0,
            color=color,
            alpha=0.95,
            label=label,
        )
        ax.set_yscale("log")

    ax.axhline(reference_line_y, color="0.35", lw=1.0, ls="--", zorder=0)
    return True


def _plot_spline_style(
    ax, rp, plot_y, plot_yerr, color, marker, label, use_log_y, reference_line_y=0.0
):
    from scipy.interpolate import make_interp_spline

    if use_log_y:
        pos = plot_y > 0
        rp, plot_y, plot_yerr = rp[pos], plot_y[pos], plot_yerr[pos]
        if len(rp) == 0:
            return False

    x_dense = np.logspace(np.log10(rp.min()), np.log10(rp.max()), 300)
    x_log = np.log10(rp)

    if not use_log_y:
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
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(x_dense, y_dense, color=color, lw=2.0)
    else:
        if len(rp) >= 2:
            spline_k = min(3, len(rp) - 1)
            # Interpolate in log-y space for the mean.
            smooth_log = make_interp_spline(x_log, np.log10(plot_y), k=spline_k)
            y_dense = np.power(10.0, smooth_log(np.log10(x_dense)))

            # Interpolate the upper and lower boundaries separately.
            # This handles large linear errors more gracefully on a log scale.
            y_upper = plot_y + plot_yerr
            y_lower = np.maximum(plot_y - plot_yerr, 1e-12 * plot_y)

            smooth_upper_log = make_interp_spline(x_log, np.log10(y_upper), k=spline_k)
            smooth_lower_log = make_interp_spline(x_log, np.log10(y_lower), k=spline_k)

            yhi_dense = np.power(10.0, smooth_upper_log(np.log10(x_dense)))
            ylo_dense = np.power(10.0, smooth_lower_log(np.log10(x_dense)))
        else:
            y_dense = np.full_like(x_dense, plot_y[0])
            yhi_dense = np.full_like(x_dense, plot_y[0] + plot_yerr[0])
            ylo_dense = np.full_like(
                x_dense, max(plot_y[0] - plot_yerr[0], 1e-12 * plot_y[0])
            )

        ax.fill_between(
            x_dense, ylo_dense, yhi_dense, color=color, alpha=0.22, linewidth=0
        )
        ax.plot(x_dense, y_dense, color=color, lw=2.0)
        ax.set_yscale("log")

    ax.axhline(reference_line_y, color="0.35", lw=1.0, ls="--", zorder=0)
    ax.plot(rp, plot_y, color=color, marker=marker, ms=4.5, lw=0.0, label=label)
    return True


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
    if ax_list is None:
        n_bins = len(tables)
        fig, axes = plt.subplots(
            n_bins, 1, figsize=(8.6, 3.3 * n_bins), sharex=True, sharey=False
        )
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
            plot_y, plot_yerr = rp * value, rp * value_err
        else:
            plot_y, plot_yerr = value, value_err

        order = np.argsort(rp)
        rp, plot_y, plot_yerr = rp[order], plot_y[order], plot_yerr[order]

        valid = (
            np.isfinite(rp) & np.isfinite(plot_y) & np.isfinite(plot_yerr) & (rp > 0)
        )
        rp, plot_y, plot_yerr = rp[valid], plot_y[valid], plot_yerr[valid]

        if len(rp) == 0:
            if ax_list is None:
                ax.set_title(f"Lens bin {i} (no valid data)")
                ax.set_xscale("log")
                ax.grid(alpha=0.2, which="both")
            continue

        base_color = palette[i % len(palette)]
        current_color = _label_variant_color(base_color, label_index, n_labels)
        current_label = label_text if i == 0 else None

        if not use_spline:
            success = _plot_errorbar_style(
                ax,
                rp,
                plot_y,
                plot_yerr,
                current_color,
                marker,
                current_label,
                use_log_y,
                reference_line_y,
            )
        else:
            success = _plot_spline_style(
                ax,
                rp,
                plot_y,
                plot_yerr,
                current_color,
                marker,
                current_label,
                use_log_y,
                reference_line_y,
            )

        if not success:
            if ax_list is None:
                ax.set_title(f"Lens bin {i} (plotting failed)")
            continue

        if ax_list is None:
            ax.set_title(f"Lens bin {i}")
            if not use_log_y:
                y_for_lim = np.concatenate([plot_y - plot_yerr, plot_y + plot_yerr])
                y_for_lim = y_for_lim[np.isfinite(y_for_lim)]
                if y_for_lim.size > 0:
                    y_min, y_max = np.nanmin(y_for_lim), np.nanmax(y_for_lim)
                    pad = 0.08 * (
                        y_max - y_min if y_max != y_min else max(abs(y_min), 1)
                    )
                    ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_xscale("log")
        ax.grid(alpha=0.2, which="both")

    ylabel = y_label or (
        r"$R \times \Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$"
        if multiply_by_radius
        else r"$\Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$"
        if use_log_y
        else value_column
    )

    if title_suffix is None:
        title_suffix = (
            r"$R \times \Delta\Sigma$ Profiles"
            if multiply_by_radius
            else r"$\Delta\Sigma$ Profiles"
            if use_log_y
            else f"{value_column} Profiles"
        )

    for ax in axes:
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel(r"$R\ [\mathrm{Mpc}]$")

    if ax_list is None:
        fig.suptitle(rf"{title_label} {title_suffix}", y=0.996)
        fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.985])
        plt.show()
    else:
        for i, ax in enumerate(axes):
            ax.set_title(f"Lens bin {i}")

    return fig
