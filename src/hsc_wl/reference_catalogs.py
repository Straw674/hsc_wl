"""Public cluster references clipped to the master HSC Y3 shape footprint."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hsc_wl.coverage import CANONICAL_FIELDS, Y3_MASK_PATH, in_field, load_y3_mask
from src.data import download_public_catalog, read_catalog_frame

PUBLIC_DOWNLOADS = (
    (
        "des_y6_wazp.fits",
        "https://datasets.linea.org.br/wazp/y6a2_dnf_wazp_v5.0.12.6801_clusters.fits",
    ),
    (
        "xxl_dr2.fits",
        "https://vizier.cds.unistra.fr/viz-bin/asu-binfits?-source=IX/52/xxl365gc&-out.all&-out.max=unlimited",
    ),
    (
        "act_dr6.fits",
        "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/actadv_dr6_cluster_cat/DR6_cluster-catalog_v1.0.fits",
    ),
    (
        "des_y3_redmapper.h5",
        "https://data.darkenergysurvey.org/fnalmisc/y3-clusters/y3_redmapper_v6.4.22+2_release.h5",
    ),
    (
        "erass1_primary.tgz",
        "https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/BulbulE_DR1/erass1cl_primary_v3.2.fits.tgz",
    ),
    (
        "erass1_optical.tgz",
        "https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/KlugeM_DR1/eRASS1_clusters_optical.fits.tgz",
    ),
    (
        "efeds_clusters.fits.gz",
        "https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/liuA/eFEDS_clusters_V3.2.fits.gz",
    ),
    (
        "efeds_optical.fits.gz",
        "https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/klein/eFEDS_c001_main_ctp_clus_v2.1.fits.gz",
    ),
    (
        "kids_dr3_amico.fits",
        "https://vizier.cds.unistra.fr/viz-bin/asu-binfits?-source=J/A%2BA/665/A100/ak3&-out.all&-out.max=unlimited",
    ),
)

REFERENCE_CATALOGS = {
    "des_y6_wazp": dict(
        label="DES Y6 WaZP",
        source_url="https://data.linea.org.br/en/sci_products/wazp.html",
        selection="v5.0.12.6801; IN_COSMO == True and NGALS >= 25; density-map centers; ZPHOT.",
    ),
    "xxl_dr2": dict(
        label="XXL DR2 C1/C2",
        source_url="https://cdsarc.cds.unistra.fr/viz-bin/cat/IX/52",
        selection="Updated CDS IX/52/xxl365gc confirmed C1/C2 sample; X-ray centers.",
    ),
    "act_dr6": dict(
        label="ACT DR6 SZ v1.0",
        source_url="https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_szcluster_catalog_info.html",
        selection="Full optically confirmed DR6 catalog; SZ centers; flags retained.",
    ),
    "erass1": dict(
        label="eRASS1 + eROMaPPer",
        source_url="https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/",
        selection="Primary v3.2; X-ray centers; optical properties joined by DETUID. eROMaPPer is optical follow-up of the same X-ray sample.",
    ),
    "efeds": dict(
        label="eFEDS + MCMF",
        source_url="https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/",
        selection="Liu v3.2 + Klein v2.1; X-ray centers; 0 <= F_CONT_BEST_COMB < 0.3.",
    ),
    "kids_dr3_amico": dict(
        label="KiDS DR3 AMICO",
        source_url="https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A%2BA/665/A100",
        selection="Published CDS J/A+A/665/A100 subset: intrinsic richness >= 15, S/N >= 3.5; corrected zfix; AMICO centers. This is not the full 7988-object DR3 sample.",
    ),
    "des_y3_redmapper": dict(
        label="DES Y3 redMaPPer",
        source_url="https://data.darkenergysurvey.org/fnalmisc/y3-clusters/",
        selection="v6.4.22+2; official index/redmapper/lgt20/select selection; optical centers; z_lambda.",
    ),
}


def standardize_reference(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Keep explicit native quantities and normalize coordinate and ID columns."""
    result = frame[list(columns)].rename(columns=columns).copy()
    result["name"] = result["name"].astype(str)
    for column in ("snr", "richness", "mass", "ra_opt", "dec_opt", "contamination"):
        if column not in result:
            result[column] = np.nan
    for column in ("snr", "richness", "mass"):
        result[column] = result[column].where(result[column] > 0)
    optical_valid = result["ra_opt"].between(0, 360, inclusive="left") & result[
        "dec_opt"
    ].between(-90, 90)
    result.loc[~optical_valid, ["ra_opt", "dec_opt"]] = np.nan
    if "quality" not in result:
        result["quality"] = ""
    if "mass_definition" not in result:
        result["mass_definition"] = ""
    return result


def read_public_references(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read current public releases and join optical follow-up by unique source ID."""
    wazp = read_catalog_frame(
        raw_dir / "des_y6_wazp.fits",
        (
            "NAME",
            "RA",
            "DEC",
            "ZPHOT",
            "SNR",
            "NGALS",
            "IN_COSMO",
            "COVER_FRAC_1MPC",
        ),
    )
    wazp = wazp.loc[wazp["IN_COSMO"] & (wazp["NGALS"] >= 25)]
    wazp_frame = standardize_reference(
        wazp,
        {
            "NAME": "name",
            "RA": "ra",
            "DEC": "dec",
            "ZPHOT": "z",
            "SNR": "snr",
            "NGALS": "richness",
            "COVER_FRAC_1MPC": "quality",
        },
    )
    xxl = read_catalog_frame(
        raw_dir / "xxl_dr2.fits", ("XLSSC", "RAJ2000", "DEJ2000", "z", "Class")
    )
    xxl_frame = standardize_reference(
        xxl,
        {
            "XLSSC": "name",
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "z": "z",
            "Class": "quality",
        },
    )
    xxl_frame["name"] = "XLSSC " + xxl_frame["name"]
    act = read_catalog_frame(raw_dir / "act_dr6.fits")
    act_frame = standardize_reference(
        act,
        {
            "name": "name",
            "RADeg": "ra",
            "decDeg": "dec",
            "redshift": "z",
            "SNR": "snr",
            "M500c": "mass",
            "opt_RADeg": "ra_opt",
            "opt_decDeg": "dec_opt",
            "flags": "quality",
        },
    )
    act_frame["mass_definition"] = "M500c [1e14 Msun]"
    erass = read_catalog_frame(
        raw_dir / "erass1_primary.tgz",
        (
            "DETUID",
            "NAME",
            "RA",
            "DEC",
            "BEST_Z",
            "EXT_LIKE",
            "M500",
            "PCONT",
        ),
    )
    erass_opt = read_catalog_frame(
        raw_dir / "erass1_optical.tgz",
        (
            "DETUID",
            "RA_OPT",
            "DEC_OPT",
            "LAMBDA_NORM",
        ),
    )
    erass = erass.merge(erass_opt, on="DETUID", how="left", validate="one_to_one")
    erass_frame = standardize_reference(
        erass,
        {
            "NAME": "name",
            "RA": "ra",
            "DEC": "dec",
            "BEST_Z": "z",
            "M500": "mass",
            "RA_OPT": "ra_opt",
            "DEC_OPT": "dec_opt",
            "LAMBDA_NORM": "richness",
            "EXT_LIKE": "quality",
            "PCONT": "contamination",
        },
    )
    erass_frame["mass_definition"] = "M500 [1e13 Msun]"
    efeds = read_catalog_frame(
        raw_dir / "efeds_clusters.fits.gz",
        (
            "ID_SRC",
            "RA",
            "DEC",
            "z",
            "SNR_MAX",
        ),
    )
    efeds_opt = read_catalog_frame(
        raw_dir / "efeds_optical.fits.gz",
        (
            "ID_SRC",
            "Name",
            "F_CONT_BEST_COMB",
            "LAMBDA_BEST_COMB",
            "RA_OPTCEN_BEST_COMB",
            "DEC_OPTCEN_BEST_COMB",
        ),
    )
    efeds = efeds.merge(efeds_opt, on="ID_SRC", how="left", validate="one_to_one")
    efeds = efeds.loc[efeds["F_CONT_BEST_COMB"].between(0, 0.3, inclusive="left")]
    efeds_frame = standardize_reference(
        efeds,
        {
            "Name": "name",
            "RA": "ra",
            "DEC": "dec",
            "z": "z",
            "SNR_MAX": "snr",
            "LAMBDA_BEST_COMB": "richness",
            "F_CONT_BEST_COMB": "contamination",
            "RA_OPTCEN_BEST_COMB": "ra_opt",
            "DEC_OPTCEN_BEST_COMB": "dec_opt",
        },
    )
    kids = read_catalog_frame(
        raw_dir / "kids_dr3_amico.fits",
        (
            "Name",
            "RAJ2000",
            "DEJ2000",
            "zfix",
            "lstar",
            "S_N",
        ),
    )
    kids_frame = standardize_reference(
        kids,
        {
            "Name": "name",
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "zfix": "z",
            "lstar": "richness",
            "S_N": "snr",
        },
    )
    des = read_catalog_frame(
        raw_dir / "des_y3_redmapper.h5",
        (
            "mem_match_id",
            "ra",
            "dec",
            "z_lambda",
            "lambda_chisq",
            "maskfrac",
        ),
        hdf5_dataset="catalog/cluster",
        hdf5_selection="index/redmapper/lgt20/select",
    )
    des_frame = standardize_reference(
        des,
        {
            "mem_match_id": "name",
            "ra": "ra",
            "dec": "dec",
            "z_lambda": "z",
            "lambda_chisq": "richness",
            "maskfrac": "quality",
        },
    )
    return dict(
        des_y6_wazp=wazp_frame,
        xxl_dr2=xxl_frame,
        act_dr6=act_frame,
        erass1=erass_frame,
        efeds=efeds_frame,
        kids_dr3_amico=kids_frame,
        des_y3_redmapper=des_frame,
    )


def clip_reference_to_mask(frame: pd.DataFrame, mask) -> pd.DataFrame:
    """Clip valid native catalog centers to the exact boolean Y3 footprint."""
    valid = np.isfinite(frame[["ra", "dec"]]).all(axis=1)
    valid &= frame["ra"].between(0, 360, inclusive="left") & frame["dec"].between(
        -90, 90
    )
    clean = frame.loc[valid]
    inside = mask.get_values_pos(
        clean["ra"].to_numpy(), clean["dec"].to_numpy(), lonlat=True
    )
    return clean.loc[np.asarray(inside, dtype=bool)].reset_index(drop=True)


def prepare_reference_catalogs(
    root: Path, downloads: tuple[tuple[str, str], ...]
) -> pd.DataFrame:
    """Download, normalize, mask once, and persist references with provenance."""
    directory = root / "data/reference_catalogs"
    for name, url in downloads:
        download_public_catalog(url, directory / "raw" / name)
    logging.info("Normalizing public cluster catalogs")
    frames = read_public_references(directory / "raw")
    mask = load_y3_mask(root)
    if mask.nside_sparse != 8192 or mask.dtype != np.dtype(bool):
        raise ValueError("Expected the boolean NSIDE=8192 HSC Y3 shape mask")
    records = []
    for key, frame in frames.items():
        clipped = clip_reference_to_mask(frame, mask)
        clipped.to_parquet(directory / f"{key}_y3.parquet", index=False)
        counts = {
            field: int(
                in_field(
                    clipped["ra"].to_numpy(), clipped["dec"].to_numpy(), field
                ).sum()
            )
            for field in CANONICAL_FIELDS
        }
        records.append(
            dict(catalog=key, source_rows=len(frame), mask_rows=len(clipped), **counts)
        )
        logging.info(
            "%s: %d selected source rows -> %d Y3 centers",
            key,
            len(frame),
            len(clipped),
        )
    manifest = dict(
        mask=Y3_MASK_PATH,
        nside=8192,
        mask_sha256=hashlib.sha256((root / Y3_MASK_PATH).read_bytes()).hexdigest(),
        downloads=[dict(file=name, url=url) for name, url in downloads],
        catalogs={
            key: dict(**REFERENCE_CATALOGS[key], **record)
            for key, record in zip(frames, records)
        },
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return pd.DataFrame(records)


def load_reference_catalogs(
    root: Path, keys: tuple[str, ...], redshift_range: tuple[float, float]
) -> dict[str, pd.DataFrame]:
    """Load masked Parquet references and apply the comparison redshift interval."""
    directory = root / "data/reference_catalogs"
    if not (directory / "manifest.json").exists():
        raise FileNotFoundError(
            "Run scripts/data_process/prepare_reference_catalogs.py first"
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    fingerprint = hashlib.sha256((root / Y3_MASK_PATH).read_bytes()).hexdigest()
    if manifest["mask_sha256"] != fingerprint:
        raise ValueError("Y3 mask changed; rerun prepare_reference_catalogs.py")
    catalogs = {}
    for key in keys:
        frame = read_catalog_frame(directory / f"{key}_y3.parquet")
        selected = frame.loc[frame["z"].between(*redshift_range)].reset_index(drop=True)
        selected.attrs = manifest["catalogs"][key]
        catalogs[key] = selected
        logging.info(
            "%s: %d Y3 references at z in %s", key, len(selected), redshift_range
        )
    return catalogs


PURE_REFERENCE_METADATA = {
    "erass1_ext3": {
        "label": "eRASS1 (EXT>=3 Complete)",
        "type": "X-ray",
        "region": "Spring",
    },
    "erass1_ext6": {
        "label": "eRASS1 (EXT>=6 Standard)",
        "type": "X-ray",
        "region": "Spring",
    },
    "erass1_ext10": {
        "label": "eRASS1 (EXT>=10 Pure)",
        "type": "X-ray",
        "region": "Spring",
    },
    "efeds_raw": {
        "label": "eFEDS Raw X-ray",
        "type": "X-ray",
        "region": "GAMA09H",
    },
    "xxl_dr2_pure": {
        "label": "XXL DR2 C1/C2",
        "type": "X-ray",
        "region": "XMM",
    },
    "act_dr6_pure": {
        "label": "ACT DR6 SZ All",
        "type": "SZ",
        "region": "Spring+Fall",
    },
    "act_dr6_snr55": {
        "label": "ACT DR6 SZ (SNR>=5.5)",
        "type": "SZ",
        "region": "Spring+Fall",
    },
    "chen2024_pure": {
        "label": "Chen+2024 WL (Pure)",
        "type": "WL Shear",
        "region": "Full Survey",
    },
}

PURE_REFERENCE_ORDER = [
    # X-ray
    "erass1_ext3",
    "erass1_ext6",
    "erass1_ext10",
    "efeds_raw",
    "xxl_dr2_pure",
    # SZ
    "act_dr6_pure",
    "act_dr6_snr55",
    # WL
    "chen2024_pure",
]


def load_pure_reference_catalogs(
    root: Path,
    keys: tuple[str, ...] | None = None,
    ext_like_thresholds: tuple[float, ...] = (3.0, 6.0, 10.0),
    act_snr_threshold: float = 5.5,
) -> dict[str, pd.DataFrame]:
    """Load pure X-ray, SZ, and WL references clipped to Y3 mask without optical or redshift cuts.

    Coordinates are native instrument/survey centers. The 'z' column is explicitly set
    to NaN so that downstream matching dynamically converts physical apertures using
    the candidate optical cluster's redshift.
    """
    raw_dir = root / "data/reference_catalogs/raw"
    mask = load_y3_mask(root)
    all_catalogs = {}

    # 1. Chen et al. (2024) blind WL shear peaks
    chen_path = root / "data/chen2024_shear_selected_clusters.parquet"
    if chen_path.exists():
        chen_df = read_catalog_frame(chen_path)
        chen_clipped = clip_reference_to_mask(chen_df, mask)
        chen_clipped["name"] = "Chen+2024 #" + chen_clipped["peak_id"].astype(str)
        chen_clipped["z"] = np.nan
        all_catalogs["chen2024_pure"] = chen_clipped

    # 2. eFEDS raw X-ray candidates (Liu et al. 2022, no MCMF cut)
    efeds_path = raw_dir / "efeds_clusters.fits.gz"
    if efeds_path.exists():
        ef_raw = read_catalog_frame(efeds_path, ("ID_SRC", "RA", "DEC", "SNR_MAX"))
        ef_std = standardize_reference(
            ef_raw,
            {"ID_SRC": "name", "RA": "ra", "DEC": "dec", "SNR_MAX": "snr"},
        )
        ef_clipped = clip_reference_to_mask(ef_std, mask)
        ef_clipped["name"] = "eFEDS #" + ef_clipped["name"].astype(str)
        ef_clipped["z"] = np.nan
        all_catalogs["efeds_raw"] = ef_clipped

    # 3. eRASS1 extended X-ray sources (Bulbul et al. 2024, no eROMaPPer optical cut)
    erass_path = raw_dir / "erass1_primary.tgz"
    if erass_path.exists():
        er_raw = read_catalog_frame(
            erass_path, ("DETUID", "NAME", "RA", "DEC", "EXT_LIKE", "DET_LIKE_0")
        )
        er_std = standardize_reference(
            er_raw,
            {"NAME": "name", "RA": "ra", "DEC": "dec", "DET_LIKE_0": "snr"},
        )
        er_std["EXT_LIKE"] = er_raw["EXT_LIKE"]
        er_clipped = clip_reference_to_mask(er_std, mask)
        er_clipped["z"] = np.nan
        for thr in ext_like_thresholds:
            thr_int = int(thr) if thr == int(thr) else thr
            sub = er_clipped.loc[er_clipped["EXT_LIKE"] >= thr].reset_index(drop=True)
            all_catalogs[f"erass1_ext{thr_int}"] = sub

    # 4. ACT DR6 SZ catalog (no redshift cut; all and high SNR)
    act_path = raw_dir / "act_dr6.fits"
    if act_path.exists():
        act_raw = read_catalog_frame(
            act_path, ("name", "RADeg", "decDeg", "SNR", "M500c", "flags")
        )
        act_std = standardize_reference(
            act_raw,
            {
                "name": "name",
                "RADeg": "ra",
                "decDeg": "dec",
                "SNR": "snr",
                "M500c": "mass",
                "flags": "quality",
            },
        )
        act_clipped = clip_reference_to_mask(act_std, mask)
        act_clipped["z"] = np.nan
        all_catalogs["act_dr6_pure"] = act_clipped
        all_catalogs["act_dr6_snr55"] = act_clipped.loc[
            act_clipped["snr"] >= act_snr_threshold
        ].reset_index(drop=True)

    # 5. XXL DR2 C1/C2 (no redshift cut)
    xxl_path = raw_dir / "xxl_dr2.fits"
    if xxl_path.exists():
        xxl_raw = read_catalog_frame(xxl_path, ("XLSSC", "RAJ2000", "DEJ2000", "Class"))
        xxl_std = standardize_reference(
            xxl_raw,
            {
                "XLSSC": "name",
                "RAJ2000": "ra",
                "DEJ2000": "dec",
                "Class": "quality",
            },
        )
        xxl_clipped = clip_reference_to_mask(xxl_std, mask)
        xxl_clipped["name"] = "XLSSC " + xxl_clipped["name"].astype(str)
        xxl_clipped["z"] = np.nan
        all_catalogs["xxl_dr2_pure"] = xxl_clipped

    target_keys = keys if keys is not None else tuple(all_catalogs.keys())
    result = {}
    for k in target_keys:
        if k in all_catalogs:
            df = all_catalogs[k]
            if k in PURE_REFERENCE_METADATA:
                df.attrs = PURE_REFERENCE_METADATA[k]
            result[k] = df
            logging.info("%s: %d Y3 pure references (no optical/z cuts)", k, len(df))
    return result
