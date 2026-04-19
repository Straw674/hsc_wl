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


# ---------- core ----------
def run_from_config(config_path):
    """Load and parse YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validate required top-level sections
    for sec in ("misc", "lens_galaxies", "source_galaxies"):
        if sec not in cfg:
            raise KeyError(f"Missing required section: {sec}")

    # ---- [misc] (all required, no defaults)
    misc = cfg["misc"]
    savepath = misc["savepath"]
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

    lens_files = [find_one(p, "lens file") for p in lenssec["lens_files"]]
    rand_files = [find_one(p, "random file") for p in lenssec["random_files"]]
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

    # this procedure might have been done already
    # table_s = table_s[table_s["b_mode_mask"] == 1]
    table_s = dsigma_table(table_s, "source", survey=src_survey.upper())

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
        centers = compute_jackknife_fields(
            table_l, n_jk, weights=np.sum(table_l["sum 1"], axis=1)
        )
        compute_jackknife_fields(table_r, centers)

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

        print("kwargs", kwargs)

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
    ap = argparse.ArgumentParser(
        description="Run dsigma from YAML config."
    )
    ap.add_argument("config", help="Path to YAML configuration file.")
    args = ap.parse_args()
    run_from_config(args.config)
