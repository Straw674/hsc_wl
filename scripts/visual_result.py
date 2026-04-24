# %%
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        print("Error: pyproject.toml not found in parent directories.")
        # Handle the error appropriately, maybe raise an exception or exit
        # For now, just break to avoid infinite loop if marker is truly missing
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        print(f"Project root found: {root_path}")  # Confirm the path found
        break
    else:
        current_dir = current_dir.parent

if root_path:
    root_path_str = str(root_path)

    if root_path_str not in sys.path:
        sys.path.append(root_path_str)

    try:
        from initial import *
    except ModuleNotFoundError as e:
        print(f"Error importing 'initial': {e}")

else:
    print("Could not proceed without finding the project root.")

# %%

label = "s16a_mass"

# Select which plot function to use for main visualization
MULTIPLY_BY_RADIUS = True

# False: keep current spline + shaded error band.
# True: disable spline/shaded band and draw only points with vertical error bars.
USE_POINT_ERRORBAR_MODE = True

RANDOM_SUBTRACTION = False


def load_result_tables(base_dir):
    tables = []
    for i in range(4):
        path = base_dir / f"hsc_hsc_lens{i}_lens.csv"
        tables.append(Table.read(path, format="csv"))
    return tables


def _pair_marker_sizes(n_pairs, min_size=30.0, max_size=190.0):
    n_pairs = np.asarray(n_pairs, dtype=float)
    safe = np.clip(n_pairs, 1.0, None)
    log_np = np.log10(safe)
    lo, hi = np.nanmin(log_np), np.nanmax(log_np)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return np.full_like(log_np, (min_size + max_size) * 0.5)
    return min_size + (log_np - lo) / (hi - lo) * (max_size - min_size)


def plot_radial_profile(
    tables,
    value_column,
    title_label,
    show_pair_sizes=False,
    multiply_by_radius=True,
    point_errorbar_mode=False,
):
    from scipy.interpolate import make_interp_spline

    fig, axes = plt.subplots(4, 1, figsize=(8.6, 13.2), sharex=True, sharey=False)
    axes = np.atleast_1d(axes)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, table in enumerate(tables):
        ax = axes[i]
        rp = np.asarray(table["rp"], dtype=float)
        value = np.asarray(table[value_column], dtype=float)
        ds_err = np.asarray(table["ds_err"], dtype=float)
        n_pairs = np.asarray(table["n_pairs"], dtype=float)

        if multiply_by_radius:
            plot_y = rp * value
            plot_yerr = rp * ds_err
        else:
            plot_y = value
            plot_yerr = ds_err

        order = np.argsort(rp)
        rp = rp[order]
        plot_y = plot_y[order]
        plot_yerr = plot_yerr[order]
        n_pairs = n_pairs[order]

        color = palette[i % len(palette)]
        valid = (
            np.isfinite(rp) & np.isfinite(plot_y) & np.isfinite(plot_yerr) & (rp > 0)
        )
        rp = rp[valid]
        plot_y = plot_y[valid]
        plot_yerr = np.clip(plot_yerr[valid], 0.0, np.inf)
        n_pairs = n_pairs[valid]

        if len(rp) == 0:
            ax.set_title(f"Lens bin {i} (no valid data)")
            ax.set_xscale("log")
            ax.grid(alpha=0.2, which="both")
            continue

        if point_errorbar_mode:
            if multiply_by_radius:
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
                    fmt="o",
                    ms=5.0,
                    ls="none",
                    elinewidth=1.2,
                    capsize=3.0,
                    color=color,
                    alpha=0.95,
                )
                ax.axhline(0.0, color="0.35", lw=1.0, ls="--", zorder=0)
            else:
                # For log-scale y, keep strictly positive points and enforce positive lower error bar.
                pos = plot_y > 0
                rp_pos = rp[pos]
                y_pos = plot_y[pos]
                yerr_pos = plot_yerr[pos]
                if len(rp_pos) == 0:
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
                    fmt="o",
                    ms=5.0,
                    ls="none",
                    elinewidth=1.2,
                    capsize=3.0,
                    color=color,
                    alpha=0.95,
                )
                ax.set_yscale("log")

            ax.set_xscale("log")
            ax.grid(alpha=0.2, which="both")
            ax.set_title(f"Lens bin {i}")
            continue

        if multiply_by_radius:
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
                color=color,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(x_dense, y_dense, color=color, lw=2.0)
            if show_pair_sizes:
                marker_sizes = _pair_marker_sizes(n_pairs)
                ax.scatter(
                    rp,
                    plot_y,
                    s=marker_sizes,
                    c=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
            else:
                ax.plot(rp, plot_y, color=color, marker="o", ms=4.5, lw=0.0)

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

            ax.axhline(0.0, color="0.35", lw=1.0, ls="--", zorder=0)
        else:
            # For log-scale y, only keep strictly positive measurements.
            pos = plot_y > 0
            rp_pos = rp[pos]
            y_pos = plot_y[pos]
            yerr_pos = plot_yerr[pos]
            n_pairs_pos = n_pairs[pos]

            if len(rp_pos) == 0:
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
                color=color,
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(x_dense, y_dense, color=color, lw=2.0)
            if show_pair_sizes:
                marker_sizes = _pair_marker_sizes(n_pairs_pos)
                ax.scatter(
                    rp_pos,
                    y_pos,
                    s=marker_sizes,
                    c=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
            else:
                ax.plot(rp_pos, y_pos, color=color, marker="o", ms=4.5, lw=0.0)

            ax.set_yscale("log")

        ax.set_xscale("log")
        ax.grid(alpha=0.2, which="both")
        ax.set_title(f"Lens bin {i}")

    if multiply_by_radius:
        ylabel = r"$R \times \Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$"
        title_suffix = r"$R \times \Delta\Sigma$ Profiles"
    else:
        ylabel = r"$\Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$"
        title_suffix = r"$\Delta\Sigma$ Profiles"

    for ax in axes:
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    if show_pair_sizes:
        fig.text(
            0.5,
            0.012,
            "Marker size scales with n_pairs in each rp bin",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.suptitle(rf"{title_label} {title_suffix}", y=0.996)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.985])

    plt.show()
    return fig


def plot_pair_counts_vs_rp(tables):
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, table in enumerate(tables):
        ax = axes[i]
        rp = np.asarray(table["rp"], dtype=float)
        n_pairs = np.asarray(table["n_pairs"], dtype=float)
        order = np.argsort(rp)
        rp = rp[order]
        n_pairs = n_pairs[order]

        ax.plot(
            rp, n_pairs, marker="o", ms=4.8, lw=1.3, color=palette[i % len(palette)]
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.set_title(f"Lens bin {i}")

    axes[0].set_ylabel("n_pairs")
    axes[2].set_ylabel("n_pairs")
    axes[2].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    axes[3].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    fig.suptitle("Pair-count diagnostics per radial bin", y=0.98)
    fig.tight_layout()
    plt.show()
    return fig


def plot_correction_factors_radial(tables):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, table in enumerate(tables):
        ax = axes[i]
        rp = np.asarray(table["rp"], dtype=float)
        one_plus_m = np.asarray(table["1+m"], dtype=float)
        two_r = np.asarray(table["2R"], dtype=float)
        one_plus_m_sel = np.asarray(table["1+m_sel"], dtype=float)

        order = np.argsort(rp)
        rp = rp[order]
        one_plus_m = one_plus_m[order]
        two_r = two_r[order]
        one_plus_m_sel = one_plus_m_sel[order]

        ax.plot(rp, one_plus_m, marker="o", ms=4.2, lw=1.5, label=r"$1+m$")
        ax.plot(rp, two_r, marker="s", ms=4.0, lw=1.3, label=r"$2R$")
        ax.plot(rp, one_plus_m_sel, marker="^", ms=4.0, lw=1.3, label=r"$1+m_{sel}$")

        ax.axhline(1.0, color="0.55", lw=0.9, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.set_title(f"Lens bin {i}")

    axes[0].set_ylabel("correction factor")
    axes[2].set_ylabel("correction factor")
    axes[2].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    axes[3].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Radial distribution of correction factors", y=0.98)
    fig.tight_layout()
    plt.show()
    return fig


# %%
result_dir = root_path / f"output/{label}/dsigma"
tables = load_result_tables(result_dir)


expected_cols = [
    "rp_min",
    "rp_max",
    "n_pairs",
    "rp",
    "ds_raw",
    "ds",
    "z_l",
    "z_s",
    "1+m",
    "2R",
    "1+m_sel",
    "ds_err",
]
if RANDOM_SUBTRACTION:
    expected_cols.append("ds_r")
print("Columns in result table:")
print(tables[0].colnames)
missing_cols = [c for c in expected_cols if c not in tables[0].colnames]
if missing_cols:
    raise KeyError(f"Missing expected columns: {missing_cols}")

column_meaning = {
    "rp_min": "radial bin lower edge",
    "rp_max": "radial bin upper edge",
    "n_pairs": "number of lens-source pairs",
    "rp": "radial bin center",
    "ds_raw": "raw DeltaSigma before all enabled corrections",
    "ds": "corrected DeltaSigma (main WL signal)",
    "z_l": "weighted mean lens redshift",
    "z_s": "weighted mean source redshift",
    "1+m": "multiplicative shear-bias factor",
    "2R": "shear responsivity factor",
    "1+m_sel": "selection-bias correction factor",
    "ds_r": "random-point DeltaSigma term",
    "ds_err": "jackknife error on ds",
}

print("\nColumn meanings:")
for k in expected_cols:
    print(f"- {k:8s}: {column_meaning[k]}")

basic_fig = plot_radial_profile(
    tables,
    value_column="ds",
    title_label=f"{label} - corrected (main)",
    show_pair_sizes=False,
    multiply_by_radius=MULTIPLY_BY_RADIUS,
    point_errorbar_mode=USE_POINT_ERRORBAR_MODE,
)
raw_fig = plot_radial_profile(
    tables,
    value_column="ds_raw",
    title_label=f"{label} - raw",
    multiply_by_radius=MULTIPLY_BY_RADIUS,
    point_errorbar_mode=USE_POINT_ERRORBAR_MODE,
)
if RANDOM_SUBTRACTION:
    rds_fig = plot_radial_profile(
        tables,
        value_column="ds_r",
        title_label=f"{label} - random",
        multiply_by_radius=MULTIPLY_BY_RADIUS,
        point_errorbar_mode=USE_POINT_ERRORBAR_MODE,
    )
# pair_fig = plot_pair_counts_vs_rp(tables)
# factor_fig = plot_correction_factors_radial(tables)
