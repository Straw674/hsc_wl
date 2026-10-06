"""Eight external cluster references using native centers and the HSC Y3 footprint."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hsc_wl.coverage import CANONICAL_FIELDS, Y3_MASK_PATH, in_field, load_y3_mask
from src.data import (
    download_archive_catalog,
    download_public_catalog,
    read_catalog_frame,
)

PUBLIC_DOWNLOADS = (
    (
        "erass3_main.fits.gz",
        "https://erosita.mpe.mpg.de/dr2/AllSkySurveyData_dr2/Catalogues_dr2/RamosM_DR2/eRASS3_Main_v1.3.fits.gz",
    ),
    (
        "efeds_clusters.fits.gz",
        "https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/liuA/eFEDS_clusters_V3.2.fits.gz",
    ),
    (
        "xxl_dr2_sources.fits",
        "https://vizier.cds.unistra.fr/viz-bin/asu-binfits?-source=IX/52/3xlss&-out.all&-out.max=unlimited",
    ),
    (
        "des_y6_wazp.fits",
        "https://datasets.linea.org.br/wazp/y6a2_dnf_wazp_v5.0.12.6801_clusters.fits",
    ),
    (
        "des_y3_redmapper.h5",
        "https://data.darkenergysurvey.org/fnalmisc/y3-clusters/y3_redmapper_v6.4.22+2_release.h5",
    ),
    (
        "kids_dr3_amico.fits",
        "https://vizier.cds.unistra.fr/viz-bin/asu-binfits?-source=J/A%2BA/665/A100/ak3&-out.all&-out.max=unlimited",
    ),
)

ACT_CANDIDATE_ARCHIVE = (
    "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/actadv_dr6_cluster_cat/DR6_nemo-products_v1.0.tgz",
    "DR6ClusterSearch/DR6ClusterSearch_optimalCatalog.fits",
    "act_dr6_candidates.fits",
)

OPTICAL_REFERENCE_KEYS = ("des_y3_redmapper", "des_y6_wazp", "kids_dr3_amico")

EXTERNAL_REFERENCE_METADATA = {
    "erass3": {
        "label": "eRASS:3 (EXT_LIKE >= 6)",
        "type": "X-ray",
        "region": "Spring",
        "selection": "Main v1.3; EXT_LIKE >= 6; native X-ray centers; no optical join.",
    },
    "efeds": {
        "label": "eFEDS (EXT_LIKE >= 6)",
        "type": "X-ray",
        "region": "GAMA09H",
        "selection": "Liu v3.2; EXT_LIKE >= 6; no MCMF selection.",
    },
    "xxl_dr2": {
        "label": "XXL DR2 (Bextlike >= 33)",
        "type": "X-ray",
        "region": "XMM",
        "selection": "3XLSS raw X-ray sources; Bextlike >= 33; no optical confirmation.",
    },
    "act_dr6": {
        "label": "ACT DR6 SZ (fixed S/N > 5.5)",
        "type": "SZ",
        "region": "Spring+Fall",
        "selection": "Nemo v1.0 optimal candidates; fixed_SNR > 5.5; no confirmation or post-flags cut.",
    },
    "chen2024": {
        "label": "Chen+2024 WL (S/N >= 4.7)",
        "type": "WL Shear",
        "region": "Full Survey",
        "selection": "Published shear peaks; snr >= 4.7; native peak centers.",
    },
    "des_y3_redmapper": {
        "label": "DES Y3 redMaPPer",
        "type": "Optical",
        "region": "Fall",
        "selection": "v6.4.22+2; official lgt20 selection; z_lambda in comparison interval.",
    },
    "des_y6_wazp": {
        "label": "DES Y6 WaZP",
        "type": "Optical",
        "region": "Fall",
        "selection": "v5.0.12.6801; IN_COSMO and NGALS >= 25; ZPHOT in comparison interval.",
    },
    "kids_dr3_amico": {
        "label": "KiDS DR3 AMICO",
        "type": "Optical",
        "region": "Spring",
        "selection": "Published richness >= 15 and S/N >= 3.5 subset; zfix in comparison interval.",
    },
}

# Exactly one likelihood or S/N threshold per physical probe.
EXTERNAL_REFERENCE_SELECTIONS = (
    (
        "erass3",
        "erass3_main.fits.gz",
        {"IAUNAME": "name", "RA": "ra", "DEC": "dec", "EXT_LIKE": "quality"},
        "quality",
        6.0,
        False,
    ),
    (
        "efeds",
        "efeds_clusters.fits.gz",
        {"ID_SRC": "name", "RA": "ra", "DEC": "dec", "EXT_LIKE": "quality"},
        "quality",
        6.0,
        False,
    ),
    (
        "xxl_dr2",
        "xxl_dr2_sources.fits",
        {"Xcatname": "name", "RABdeg": "ra", "DEBdeg": "dec", "Bextlike": "quality"},
        "quality",
        33.0,
        False,
    ),
    (
        "act_dr6",
        "act_dr6_candidates.fits",
        {"name": "name", "RADeg": "ra", "decDeg": "dec", "fixed_SNR": "snr"},
        "snr",
        5.5,
        True,
    ),
    (
        "chen2024",
        "chen2024_shear_selected_clusters.parquet",
        {"peak_id": "name", "ra": "ra", "dec": "dec", "snr": "snr"},
        "snr",
        4.7,
        False,
    ),
)


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


def read_optical_references(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the three optical releases with their published richness selections."""
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
        des_y3_redmapper=des_frame, des_y6_wazp=wazp_frame, kids_dr3_amico=kids_frame
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


def build_external_reference_catalogs(
    root: Path, redshift_range: tuple[float, float]
) -> dict[str, pd.DataFrame]:
    """Load five instrument-selected samples and three optical references.

    Physical probes retain unconfirmed detections and use candidate lens redshifts
    for matching apertures. Optical references retain the comparison z interval.
    No optical counterpart, richness, redshift, or post-flag is used to select the
    physical probes; their native X-ray, SZ, or shear peak centers are preserved.
    """
    raw_dir = root / "data/reference_catalogs/raw"
    for filename, url in PUBLIC_DOWNLOADS:
        download_public_catalog(url, raw_dir / filename)
    url, member, filename = ACT_CANDIDATE_ARCHIVE
    download_archive_catalog(url, member, raw_dir / filename)
    mask = load_y3_mask(root)
    if mask.nside_sparse != 8192 or mask.dtype != np.dtype(bool):
        raise ValueError("Expected the boolean NSIDE=8192 HSC Y3 shape mask")
    catalogs = {}
    for (
        key,
        filename,
        columns,
        metric,
        threshold,
        strict,
    ) in EXTERNAL_REFERENCE_SELECTIONS:
        directory = root / "data" if key == "chen2024" else raw_dir
        logging.info("Loading %s instrument detections", key)
        frame = read_catalog_frame(directory / filename, tuple(columns))
        standardized = standardize_reference(frame, columns)
        values = standardized[metric]
        selected = values > threshold if strict else values >= threshold
        clipped = clip_reference_to_mask(standardized.loc[selected], mask)
        if key in ("chen2024", "efeds"):
            clipped["name"] = (
                ("Chen+2024" if key == "chen2024" else "eFEDS") + " #" + clipped["name"]
            )
        clipped["z"] = np.nan
        clipped.attrs = EXTERNAL_REFERENCE_METADATA[key].copy()
        catalogs[key] = clipped
        logging.info(
            "%s: %d Y3 references; %s", key, len(clipped), clipped.attrs["selection"]
        )
    for key, frame in read_optical_references(raw_dir).items():
        selected = frame.loc[frame["z"].between(*redshift_range)]
        clipped = clip_reference_to_mask(selected, mask)
        clipped.attrs = EXTERNAL_REFERENCE_METADATA[key].copy()
        catalogs[key] = clipped
        logging.info(
            "%s: %d Y3 optical references at z in %s", key, len(clipped), redshift_range
        )
    return catalogs


def prepare_reference_catalogs(
    root: Path,
    redshift_range: tuple[float, float],
) -> pd.DataFrame:
    """Persist the current eight masked references and explicit selection provenance."""
    directory = root / "data/reference_catalogs"
    frames = build_external_reference_catalogs(root, redshift_range)
    records = []
    for key, frame in frames.items():
        frame.to_parquet(directory / f"{key}_y3.parquet", index=False)
        counts = {
            field: int(
                in_field(frame["ra"].to_numpy(), frame["dec"].to_numpy(), field).sum()
            )
            for field in CANONICAL_FIELDS
        }
        records.append(dict(catalog=key, mask_rows=len(frame), **counts))
    manifest = dict(
        mask=Y3_MASK_PATH,
        nside=8192,
        mask_sha256=hashlib.sha256((root / Y3_MASK_PATH).read_bytes()).hexdigest(),
        optical_redshift_range=redshift_range,
        downloads=[dict(file=name, url=url) for name, url in PUBLIC_DOWNLOADS],
        act_archive=dict(url=ACT_CANDIDATE_ARCHIVE[0], member=ACT_CANDIDATE_ARCHIVE[1]),
        catalogs={
            key: dict(**EXTERNAL_REFERENCE_METADATA[key], **record)
            for key, record in zip(frames, records)
        },
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return pd.DataFrame(records)


def load_external_reference_catalogs(
    root: Path, redshift_range: tuple[float, float]
) -> dict[str, pd.DataFrame]:
    """Load the eight prepared Parquet references and verify their selection provenance."""
    directory = root / "data/reference_catalogs"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Run scripts/data_process/prepare_reference_catalogs.py first"
        )
    manifest = json.loads(manifest_path.read_text())
    fingerprint = hashlib.sha256((root / Y3_MASK_PATH).read_bytes()).hexdigest()
    if manifest["mask_sha256"] != fingerprint:
        raise ValueError("Y3 mask changed; rerun prepare_reference_catalogs.py")
    if tuple(manifest["optical_redshift_range"]) != redshift_range:
        raise ValueError(
            "Optical redshift interval changed; rerun prepare_reference_catalogs.py"
        )
    if tuple(manifest["catalogs"]) != tuple(EXTERNAL_REFERENCE_METADATA):
        raise ValueError(
            "Expected five physical probes and three optical references; rerun preparation"
        )
    catalogs = {}
    for key, metadata in EXTERNAL_REFERENCE_METADATA.items():
        if manifest["catalogs"][key]["selection"] != metadata["selection"]:
            raise ValueError(
                f"{key} selection changed; rerun prepare_reference_catalogs.py"
            )
        frame = read_catalog_frame(directory / f"{key}_y3.parquet")
        if len(frame) != manifest["catalogs"][key]["mask_rows"]:
            raise ValueError(f"{key} cached row count changed; rerun preparation")
        frame.attrs = metadata.copy()
        catalogs[key] = frame
        logging.info(
            "%s: %d prepared Y3 references; %s", key, len(frame), metadata["selection"]
        )
    return catalogs
