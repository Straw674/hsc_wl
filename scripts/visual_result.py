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


def load_result_tables(base_dir):
    tables = []
    for i in range(4):
        path = base_dir / f"hsc_hsc_lens{i}_lens.csv"
        tables.append(Table.read(path, format="csv"))
    return tables


def summarize_result_table(table, lens_label):
    rp = np.asarray(table["rp"])
    ds = np.asarray(table["ds"])
    ds_err = np.asarray(table["ds_err"])
    ds_raw = np.asarray(table["ds_raw"])
    ds_r = np.asarray(table["ds_r"])
    r_ds = rp * ds
    snr = np.divide(ds, ds_err, out=np.full_like(ds, np.nan), where=ds_err > 0)
    random_snr = np.divide(
        ds_r, ds_err, out=np.full_like(ds_r, np.nan), where=ds_err > 0
    )

    return {
        "lens_bin": lens_label,
        "n_radial_bins": len(table),
        "rp_min": float(np.nanmin(rp)),
        "rp_max": float(np.nanmax(rp)),
        "sum_pairs": float(np.nansum(table["n_pairs"])),
        "min_pairs": float(np.nanmin(table["n_pairs"])),
        "max_pairs": float(np.nanmax(table["n_pairs"])),
        "finite_ds_frac": float(np.mean(np.isfinite(ds))),
        "positive_ds_err_frac": float(np.mean(ds_err > 0)),
        "max_abs_snr": float(np.nanmax(np.abs(snr))),
        "max_abs_rds": float(np.nanmax(np.abs(r_ds))),
        "max_abs_random_snr": float(np.nanmax(np.abs(random_snr))),
        "mean_abs_random_snr": float(np.nanmean(np.abs(random_snr))),
        "max_abs_ds_minus_raw": float(np.nanmax(np.abs(ds - ds_raw))),
        "chi2_null": float(
            np.nansum(
                (np.divide(ds, ds_err, out=np.zeros_like(ds), where=ds_err > 0)) ** 2
            )
        ),
    }


def _pair_marker_sizes(n_pairs, min_size=30.0, max_size=190.0):
    n_pairs = np.asarray(n_pairs, dtype=float)
    safe = np.clip(n_pairs, 1.0, None)
    log_np = np.log10(safe)
    lo, hi = np.nanmin(log_np), np.nanmax(log_np)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return np.full_like(log_np, (min_size + max_size) * 0.5)
    return min_size + (log_np - lo) / (hi - lo) * (max_size - min_size)


def plot_radial_profile(
    tables, value_column, title_label, show_pair_sizes=False, multiply_by_radius=True
):
    from scipy.interpolate import make_interp_spline

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
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
        x_dense = np.logspace(np.log10(rp.min()), np.log10(rp.max()), 300)
        x_log = np.log10(rp)

        spline_k = min(3, len(rp) - 1)
        smooth = make_interp_spline(x_log, plot_y, k=spline_k)
        smooth_err = make_interp_spline(x_log, plot_yerr, k=spline_k)
        y_dense = smooth(np.log10(x_dense))
        yerr_dense = np.clip(smooth_err(np.log10(x_dense)), 0.0, np.inf)

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
        ax.axhline(0.0, color="0.35", lw=1.0, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.grid(alpha=0.2, which="both")
        ax.set_title(f"Lens bin {i}")

    if multiply_by_radius:
        ylabel = r"$R \times \Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$"
        title_suffix = r"$R \times \Delta\Sigma$ Profiles"
    else:
        ylabel = r"$\Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$"
        title_suffix = r"$\Delta\Sigma$ Profiles"

    axes[0].set_ylabel(ylabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    axes[3].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    if show_pair_sizes:
        fig.text(
            0.5,
            0.012,
            "Marker size scales with n_pairs in each rp bin",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.suptitle(rf"{title_label} {title_suffix}", y=0.98)
    fig.tight_layout()

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
    "ds_r",
    "ds_err",
]

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
)
raw_fig = plot_radial_profile(
    tables,
    value_column="ds_raw",
    title_label=f"{label} - raw",
    multiply_by_radius=MULTIPLY_BY_RADIUS,
)
rds_fig = plot_radial_profile(
    tables,
    value_column="ds_r",
    title_label=f"{label} - random",
    multiply_by_radius=MULTIPLY_BY_RADIUS,
)
# pair_fig = plot_pair_counts_vs_rp(tables)
# factor_fig = plot_correction_factors_radial(tables)

summary = pd.DataFrame(
    [summarize_result_table(tables[i], f"lens{i}") for i in range(len(tables))]
)
print("\nPer-lens summary:")
print(summary.to_string(index=False))
