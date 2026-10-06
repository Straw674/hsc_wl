# %% [Initialization]

import gzip
import logging
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

import numpy as np
import pandas as pd
from astropy.table import Table

from hsc_wl.config import RUN_REGISTRY
from hsc_wl.prepare import run_prepare_pipeline
from initial import *  # noqa: F401,F403
from src.data import download_public_catalog

# %% Local Functions


def build_wh24_parquet(raw_gz: Path, output_parquet: Path) -> Path:
    """Parse fixed-width table2 from Wen & Han (2024) and persist as Parquet."""
    if output_parquet.exists():
        logging.info("WH24 parquet already exists: %s", output_parquet)
        return output_parquet

    logging.info("Parsing WH24 table2 from %s...", raw_gz)
    with gzip.open(raw_gz, "rt", encoding="ascii") as handle:
        lines = handle.readlines()

    names = [line[11:27].strip() for line in lines]
    ra = np.array([float(line[28:37]) for line in lines], dtype=np.float64)
    dec = np.array([float(line[38:47]) for line in lines], dtype=np.float64)
    z = np.array([float(line[49:55]) for line in lines], dtype=np.float32)
    lam500 = np.array([float(line[87:93]) for line in lines], dtype=np.float32)
    m500 = np.array([float(line[95:100]) for line in lines], dtype=np.float32)
    ngal = np.array([int(line[101:104]) for line in lines], dtype=np.int32)

    df = pd.DataFrame(
        {
            "name": names,
            "ra": ra,
            "dec": dec,
            "z": z,
            "lam500": lam500,
            "m500": m500,
            "ngal": ngal,
        }
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logging.info("Saved %d WH24 clusters to %s", len(df), output_parquet)
    return output_parquet


def build_zou21_parquet(raw_fits_gz: Path, output_parquet: Path) -> Path:
    """Parse FITS cluster catalog from Zou et al. (2021) and persist as Parquet."""
    if output_parquet.exists():
        logging.info("Zou21 parquet already exists: %s", output_parquet)
        return output_parquet

    logging.info("Parsing Zou21 catalog from %s...", raw_fits_gz)
    table = Table.read(raw_fits_gz)
    df = table[
        "CLUSTER_ID",
        "RA_PEAK",
        "DEC_PEAK",
        "PZ_PEAK",
        "SZ_PEAK",
        "RICHNESS",
        "M_500",
        "L_1MPC",
        "N_1MPC",
    ].to_pandas()
    df = df.rename(
        columns={
            "CLUSTER_ID": "id",
            "RA_PEAK": "ra",
            "DEC_PEAK": "dec",
            "PZ_PEAK": "z",
            "RICHNESS": "richness",
            "M_500": "m500",
            "L_1MPC": "l1mpc",
            "N_1MPC": "ngal",
        }
    )
    # Prefer spectroscopic redshift when available
    sz_valid = df["SZ_PEAK"] > 0
    df.loc[sz_valid, "z"] = df.loc[sz_valid, "SZ_PEAK"]

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logging.info("Saved %d Zou21 clusters to %s", len(df), output_parquet)
    return output_parquet


def build_clumpr_parquet(raw_fits: Path, output_parquet: Path) -> Path:
    """Parse CluMPR cluster catalog (Yantovski-Barth et al. 2024) and persist as Parquet."""
    if output_parquet.exists():
        logging.info("CluMPR parquet already exists: %s", output_parquet)
        return output_parquet

    logging.info("Parsing CluMPR catalog from %s...", raw_fits)
    table = Table.read(raw_fits)
    df = table[
        "RA_central",
        "DEC_central",
        "z_median_central",
        "spec_z",
        "cluster_mass_onempc",
        "richness_onempc",
        "mass_central",
    ].to_pandas()
    df["z"] = df["z_median_central"]
    sz_valid = (df["spec_z"] > 0) & (df["spec_z"] < 5)
    df.loc[sz_valid, "z"] = df.loc[sz_valid, "spec_z"]
    df = df.rename(
        columns={
            "RA_central": "ra",
            "DEC_central": "dec",
            "cluster_mass_onempc": "mass",
            "richness_onempc": "richness",
        }
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logging.info("Saved %d CluMPR clusters to %s", len(df), output_parquet)
    return output_parquet


def build_yang21_parquet(raw_ngc: Path, raw_sgc: Path, output_parquet: Path) -> Path:
    """Parse Yang et al. (2021) group catalog text files into unified Parquet."""
    if output_parquet.exists():
        logging.info("Yang21 parquet already exists: %s", output_parquet)
        return output_parquet

    def parse_file(path: Path) -> pd.DataFrame:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                z = float(parts[4])
                if z < 0.15 or z > 0.55:
                    continue
                log_m_h = float(parts[5])
                if log_m_h < 12.0:
                    continue
                records.append(
                    (
                        int(parts[0]),
                        float(parts[2]),
                        float(parts[3]),
                        z,
                        float(parts[1]),
                        log_m_h,
                    )
                )
        return pd.DataFrame(
            records, columns=["id", "ra", "dec", "z", "richness", "log_m_h"]
        )

    logging.info("Parsing Yang21 NGC...")
    df_ngc = parse_file(raw_ngc)
    logging.info("Parsing Yang21 SGC...")
    df_sgc = parse_file(raw_sgc)

    df_combined = pd.concat([df_ngc, df_sgc], ignore_index=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(output_parquet, index=False)
    logging.info("Saved %d Yang21 groups to %s", len(df_combined), output_parquet)
    return output_parquet


# %% Global Configuration

RAW_DIR = project_root / "data/external_clusters/raw"
PARQUET_DIR = project_root / "data/external_clusters"

WH24_RAW = RAW_DIR / "wh24_table2.dat.gz"
ZOU21_RAW = RAW_DIR / "zou21_clusters.fits.gz"
CLUMPR_RAW = RAW_DIR / "clumpr_simple.fits"
YANG21_RAW_TAR = RAW_DIR / "yang21_groups.tar.gz"

LABELS_TO_PREPARE = [
    "wh24_1bin",
    "zou21_1bin",
    "clumpr_mass_1bin",
    "clumpr_richness_1bin",
    "yang21_mass_1bin",
    "yang21_richness_1bin",
]

# %% [Stage 1: Build Parquet Catalogs]

build_wh24_parquet(WH24_RAW, PARQUET_DIR / "wh24.parquet")
build_zou21_parquet(ZOU21_RAW, PARQUET_DIR / "zou21.parquet")
build_clumpr_parquet(CLUMPR_RAW, PARQUET_DIR / "clumpr.parquet")

# %% [Stage 2: Run Lens Preparation Pipeline (Top 1020 Selection)]

for label in LABELS_TO_PREPARE:
    print(f"\n--- Preparing lens catalog: {label} ---")
    run_prepare_pipeline(RUN_REGISTRY[label], root=project_root)
