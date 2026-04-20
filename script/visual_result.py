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
result_dir = root_path / "output/pdr3/dsigma"


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
        "mean_z_l": float(np.nanmean(table["z_l"])),
        "mean_z_s": float(np.nanmean(table["z_s"])),
        "sum_pairs": float(np.nansum(table["n_pairs"])),
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


def plot_r_times_quantity(tables, value_column, title_label):
    from scipy.interpolate import make_interp_spline

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.8), sharex=True, sharey=True)
    axes = axes.ravel()
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, table in enumerate(tables):
        ax = axes[i]
        rp = np.asarray(table["rp"], dtype=float)
        value = np.asarray(table[value_column], dtype=float)
        ds_err = np.asarray(table["ds_err"], dtype=float)
        rds = rp * value
        rds_err = rp * ds_err

        order = np.argsort(rp)
        rp = rp[order]
        rds = rds[order]
        rds_err = rds_err[order]

        color = palette[i % len(palette)]
        x_dense = np.logspace(np.log10(rp.min()), np.log10(rp.max()), 300)
        x_log = np.log10(rp)

        spline_k = min(3, len(rp) - 1)
        smooth = make_interp_spline(x_log, rds, k=spline_k)
        smooth_err = make_interp_spline(x_log, rds_err, k=spline_k)
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
        ax.plot(rp, rds, color=color, marker="o", ms=4.5, lw=0.0)
        ax.axhline(0.0, color="0.35", lw=1.0, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.grid(alpha=0.2, which="both")
        ax.set_title(f"Lens bin {i}")

    axes[0].set_ylabel(r"$R \times \Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$")
    axes[2].set_ylabel(r"$R \times \Delta\Sigma\ [10^6\,M_{\odot}/\mathrm{pc}]$")
    axes[2].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    axes[3].set_xlabel(r"$R\ [\mathrm{Mpc}]$")
    fig.suptitle(rf"{title_label} $R \times \Delta\Sigma$ Profiles", y=0.98)
    fig.tight_layout()

    plt.show()


# %%
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

basic_fig = plot_r_times_quantity(
    tables,
    value_column="ds",
    title_label="corrected (main)",
)
raw_fig = plot_r_times_quantity(
    tables,
    value_column="ds_raw",
    title_label="raw",
)
rds_fig = plot_r_times_quantity(
    tables,
    value_column="ds_r",
    title_label="random",
)

summary = pd.DataFrame(
    [summarize_result_table(tables[i], f"lens{i}") for i in range(len(tables))]
)
print("\nPer-lens summary:")
print(summary.to_string(index=False))
