#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
import yaml
from astropy.cosmology import Planck15
from astropy.table import Table
from dsigma.helpers import dsigma_table
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density
from dsigma.surveys import hsc as hsc_survey


# ---------- runtime settings ----------
# Switch this label before each run when using profile-based YAML config.
RUN_PROFILE_LABEL = "s16a"


def find_one(path_or_pattern, description):
    paths = (
        sorted(glob.glob(path_or_pattern))
        if any(ch in path_or_pattern for ch in "*?[]")
        else [path_or_pattern]
    )
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {description}: {path_or_pattern}")


def pick_column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def pick_required_column(cols, candidates, description):
    col = pick_column(cols, candidates)
    if col is None:
        raise KeyError(f"Could not find {description}. Tried: {', '.join(candidates)}")
    return col


def assign_jackknife_fields_with_fallback(
    table_l, table_r, n_jk_requested, distance_threshold=1.0
):
    """Assign jackknife fields, reducing n_jk or relaxing connectivity as needed."""
    if len(table_l) < 2:
        raise ValueError(
            f"Need at least 2 lenses after precompute filtering for jackknife; got {len(table_l)}"
        )

    # Use only lenses with non-zero lens-source pair counts for clustering weights.
    weights = np.sum(table_l["sum 1"], axis=1)
    n_positive_weight = int(np.sum(weights > 0))
    if n_positive_weight < 2:
        raise ValueError(
            "Need at least 2 lenses with positive jackknife weights after filtering; "
            f"got {n_positive_weight}"
        )

    n_jk_start = min(int(n_jk_requested), len(table_l), n_positive_weight)
    last_error = None
    distance_thresholds = [float(distance_threshold)]
    while distance_thresholds[-1] < 180.0:
        next_threshold = min(distance_thresholds[-1] * 2.0, 180.0)
        if next_threshold == distance_thresholds[-1]:
            break
        distance_thresholds.append(next_threshold)

    for n_jk_try in range(n_jk_start, 1, -1):
        for distance_threshold_try in distance_thresholds:
            try:
                centers = compute_jackknife_fields(
                    table_l,
                    n_jk_try,
                    distance_threshold=distance_threshold_try,
                    weights=weights,
                )
                compute_jackknife_fields(table_r, centers)
                if n_jk_try < n_jk_requested or distance_threshold_try != float(
                    distance_threshold
                ):
                    print(
                        "[jackknife] requested n_jk="
                        f"{n_jk_requested}, using n_jk={n_jk_try}, "
                        f"distance_threshold={distance_threshold_try} deg"
                    )
                return centers, n_jk_try
            except ValueError as err:
                err_text = str(err)
                last_error = err
                if (
                    "larger sample than population" in err_text
                    or "0 sample(s)" in err_text
                ):
                    continue
                raise
            except RuntimeError as err:
                last_error = err
                continue

    raise RuntimeError(
        "Could not assign jackknife fields with n_jk >= 2 after filtering. "
        f"Last error: {last_error}"
    )


# ---------- core ----------
def run_from_config(config_path, run_label=RUN_PROFILE_LABEL):
    """Load and parse YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"[config] run_label: {run_label}")

    # Validate required top-level sections
    for sec in (
        "misc",
        "lens_galaxies",
        "source_galaxies",
        "label_paths",
        "corrections",
    ):
        if sec not in cfg:
            raise KeyError(f"Missing required section: {sec}")

    label_paths = cfg["label_paths"]
    if run_label not in label_paths:
        available = ", ".join(sorted(label_paths.keys()))
        raise KeyError(
            f"Unknown run profile label: {run_label}. Available labels: {available}"
        )
    run_paths = label_paths[run_label]

    for key in ("savepath", "lens_files", "random_files"):
        if key not in run_paths:
            raise KeyError(f"Missing required key in label_paths.{run_label}: {key}")

    # ---- [misc] (all required, no defaults)
    misc = cfg["misc"]
    savepath = run_paths["savepath"]
    os.makedirs(savepath, exist_ok=True)

    njobs = int(misc["njobs"])
    comoving = bool(misc["comoving"])
    lens_source_cut = float(misc["lens_source_cut"])
    n_jk = int(misc["njackknife"])

    # ---- [lens_galaxies]
    lenssec = cfg["lens_galaxies"]
    lens_survey = lenssec["surveys"].strip()
    z_bins = np.array(lenssec["z_bins"])
    rpmin = float(lenssec["rpmin"])
    rpmax = float(lenssec["rpmax"])
    n_rpbins = int(lenssec["n_rpbins"])
    linlog = lenssec["linlog"].lower()
    if linlog not in ("lin", "log"):
        raise ValueError('[lens_galaxies] linlog must be "lin" or "log"')

    lens_files = [find_one(p, "lens file") for p in run_paths["lens_files"]]
    rand_files = [find_one(p, "random file") for p in run_paths["random_files"]]
    lens_z_col = lenssec["z_col"]
    lens_ra_col = lenssec["ra_col"]
    lens_dec_col = lenssec["dec_col"]

    # ---- [source_galaxies]
    srcsec = cfg["source_galaxies"]
    src_survey = srcsec["surveys"].strip()
    src_file = find_one(srcsec["source_file"], "source catalog")
    nz_file = find_one(srcsec["nz_file"], "n(z) file")

    # ---- survey-specific corrections
    if "corrections" not in cfg:
        raise KeyError("Missing required section: corrections")
    corr_all = cfg["corrections"]
    if src_survey not in corr_all:
        raise KeyError(
            f"Missing required correction config for source survey: {src_survey}"
        )

    corr = {
        "boost_correction": bool(corr_all[src_survey]["boost_correction"]),
        "scalar_shear_response_correction": bool(
            corr_all[src_survey]["scalar_shear_response_correction"]
        ),
        "shear_responsivity_correction": bool(
            corr_all[src_survey]["shear_responsivity_correction"]
        ),
        "random_subtraction": bool(corr_all[src_survey]["random_subtraction"]),
        "selection_bias_correction": bool(
            corr_all[src_survey]["selection_bias_correction"]
        ),
    }

    # ---------------- load catalogs ----------------
    print(f"[load] sources: {src_file}")

    table_s = Table.read(src_file)

    source_cols = table_s.colnames
    source_ra_col = pick_required_column(
        source_cols, ["i_ra", "RA", "ra"], "source right ascension column"
    )
    source_dec_col = pick_required_column(
        source_cols,
        ["i_dec", "Dec", "DEC", "dec"],
        "source declination column",
    )
    source_e1_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_e1", "e_1", "e1"],
        "source e_1 column",
    )
    source_e2_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_e2", "e_2", "e2"],
        "source e_2 column",
    )
    source_w_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_derived_weight", "weight", "w"],
        "source weight column",
    )
    source_m_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_derived_shear_bias_m", "m_corr", "m"],
        "source shear bias column",
    )
    source_e_rms_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_derived_rms_e", "e_rms"],
        "source e_rms column",
    )
    source_r2_col = pick_required_column(
        source_cols,
        ["i_hsmshaperegauss_resolution", "resolution", "R_2"],
        "source resolution column",
    )
    source_zbin_col = pick_required_column(
        source_cols, ["hsc_y3_zbin", "z_bin"], "source redshift-bin column"
    )
    source_mag_col = pick_required_column(
        source_cols,
        ["i_apertureflux_10_mag", "aperture_mag", "mag_A"],
        "source aperture magnitude column",
    )

    # check whether the b-mode mask column exists, and if so, filter to only use sources that pass the mask
    if "b_mode_mask" in source_cols:
        table_s = table_s[table_s["b_mode_mask"] == 1]
    table_s = dsigma_table(
        table_s,
        "source",
        survey=src_survey.upper(),
        ra=source_ra_col,
        dec=source_dec_col,
        e_1=source_e1_col,
        e_2=source_e2_col,
        w=source_w_col,
        m=source_m_col,
        e_rms=source_e_rms_col,
        R_2=source_r2_col,
        z_bin=source_zbin_col,
        mag_A=source_mag_col,
    )

    table_s["m_sel"] = hsc_survey.multiplicative_selection_bias(table_s)
    # Remove galaxies with bimodal P(z)'s.
    table_s = table_s[table_s["z_bin"] > 0]
    # dsigma expects the first redshift bin to be 0, not 1.
    table_s["z_bin"] = table_s["z_bin"] - 1

    print(f"[load] n(z): {nz_file}")
    table_n = Table.read(nz_file)
    table_n.rename_column("Z_MID", "z")
    table_n["n"] = np.column_stack([table_n[f"BIN{i + 1}"] for i in range(4)])
    table_n.keep_columns(["z", "n"])

    # Assign each galaxy in the source catalog the mean redshift of the bin. This
    # is only used to determine which lens-source pairs to use.
    table_s["z"] = np.sum(table_n["z"][:, np.newaxis] * table_n["n"], axis=0)[
        table_s["z_bin"]
    ]

    rp_bins = np.logspace(np.log10(rpmin), np.log10(rpmax), n_rpbins + 1)

    for i, lens_file in enumerate(lens_files):
        print(f"[load] lenses: {i}; randoms: {i}")

        table_l = Table.read(lens_file)
        table_l = dsigma_table(
            table_l, "lens", z=lens_z_col, ra=lens_ra_col, dec=lens_dec_col, w_sys=1.0
        )
        table_r = Table.read(rand_files[i])
        table_r = dsigma_table(table_r, "lens", z="z", ra="ra", dec="dec", w_sys=1.0)

        print(f"lens file length: {len(table_l)}")
        print(f"random file length: {len(table_r)}")

        # ---------------- compute ----------------
        print("[precompute] lenses")
        precompute(
            table_l,
            table_s,
            rp_bins,
            cosmology=Planck15,
            comoving=comoving,
            table_n=table_n,
            lens_source_cut=lens_source_cut,
            progress_bar=True,
            n_jobs=njobs,
        )
        print("[precompute] randoms")
        precompute(
            table_r,
            table_s,
            rp_bins,
            cosmology=Planck15,
            comoving=comoving,
            table_n=table_n,
            lens_source_cut=lens_source_cut,
            progress_bar=True,
            n_jobs=njobs,
        )

        # Drop all lenses and randoms that did not have any nearby source.
        table_l = table_l[np.sum(table_l["sum 1"], axis=1) > 0]
        table_r = table_r[np.sum(table_r["sum 1"], axis=1) > 0]

        print("[jackknife] fields")
        centers, n_jk_use = assign_jackknife_fields_with_fallback(
            table_l, table_r, n_jk
        )

        print("[stack] ΔΣ by lens z-bin")

        # we only have one z bin
        # for i in range(len(z_bins) - 1):
        lo, hi = z_bins[0], z_bins[1]
        mL = (lo <= table_l["z"]) & (table_l["z"] < hi)
        mR = (lo <= table_r["z"]) & (table_r["z"] < hi)

        kwargs = corr.copy()
        kwargs["return_table"] = True
        kwargs["table_r"] = table_r[mR]

        # kwargs = dict(
        #     return_table=True,
        #     scalar_shear_response_correction=True,
        #     shear_responsivity_correction=True,
        #     selection_bias_correction=True,
        #     boost_correction=False,
        #     random_subtraction=True,
        #     table_r=table_r[mR],
        # )

        # kwargs = dict(return_table=False, table_r=table_r[mR], **corr)
        kwargs_summary = {k: v for k, v in kwargs.items() if k != "table_r"}
        kwargs_summary["table_r_rows"] = len(kwargs["table_r"])
        print("[stack] kwargs summary", kwargs_summary)

        result = excess_surface_density(table_l[mL], **kwargs)
        kwargs["return_table"] = False
        cov = jackknife_resampling(
            excess_surface_density,
            table_l[mL],
            **kwargs,
        )
        result["ds_err"] = np.sqrt(np.diag(cov))

        out_csv = os.path.join(
            savepath, f"{src_survey.lower()}_{lens_survey or 'lenses'}_lens{i}_lens.csv"
        )
        result.write(out_csv, overwrite=True)
        print(f"  wrote: {out_csv}")

    print("[done]")


# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run dsigma from YAML config.")
    ap.add_argument("config", help="Path to YAML configuration file.")
    args = ap.parse_args()
    run_from_config(args.config, run_label=RUN_PROFILE_LABEL)
