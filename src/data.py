"""Catalog import, public downloads, and bounded-memory inspection samples."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_catalog_frame(
    path: Path,
    columns: tuple[str, ...] | None = None,
    hdf5_dataset: str | None = None,
    hdf5_selection: str | None = None,
) -> pd.DataFrame:
    """Read scalar catalog columns without coercing integer identifiers to floats."""
    import h5py
    from astropy.table import Table

    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=None if columns is None else list(columns))
    if path.suffix == ".h5":
        if hdf5_dataset is None:
            raise ValueError("An explicit HDF5 dataset is required")
        with h5py.File(path, "r") as handle:
            group = handle[hdf5_dataset]
            names = (
                columns
                if columns is not None
                else tuple(name for name, dataset in group.items() if dataset.ndim == 1)
            )
            table = Table({name: group[name][:] for name in names})
            if hdf5_selection is not None:
                table = table[handle[hdf5_selection][:]]
    elif path.suffix == ".tgz":
        import tarfile
        from io import BytesIO

        with tarfile.open(path) as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".fits")
            ]
            if len(members) != 1:
                raise ValueError(f"Expected one FITS catalog in {path}")
            table = Table.read(
                BytesIO(archive.extractfile(members[0]).read()), format="fits"
            )
    else:
        table = Table.read(path, format="fits")
    names = (
        columns
        if columns is not None
        else tuple(name for name in table.colnames if table[name].ndim == 1)
    )
    frame = table[list(names)].to_pandas()
    for name in frame.select_dtypes(include="object"):
        frame[name] = frame[name].map(
            lambda value: value.decode("utf-8").strip()
            if isinstance(value, bytes)
            else value
        )
    return frame


def download_public_catalog(url: str, destination: Path) -> Path:
    """Cache a public catalog atomically; leave no partial file after failure."""
    import requests

    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    logging.info("Downloading %s", destination.name)
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def load_s23b_scalar_sample(
    base_dir: Path,
    fields: tuple[str, ...],
    sample_size: int,
    seed: int,
    batch_size: int,
) -> dict[str, pd.DataFrame]:
    """Sample raw rows uniformly, then retain coordinates passing all three BSM flags.

    Only five columns are streamed. Sampling precedes BSM filtering to bound the
    plotted count; the retained rows remain an unbiased sample of clean objects.
    No Y3, magnitude, or photo-z selection is applied. Missing flags are excluded.
    """
    if sample_size <= 0 or batch_size <= 0:
        raise ValueError("Sample and batch sizes must be positive")
    flags = [f"i_mask_brightstar_{name}" for name in ("halo", "ghost", "blooming")]
    samples = {}
    for field in fields:
        name = field.lower()
        path = base_dir / f"s23b_photometry_{name}" / f"s23b_{name}_scalar.parquet"
        parquet = pq.ParquetFile(path)
        count = parquet.metadata.num_rows
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(count, size=min(sample_size, count), replace=False)
        )
        chunks = []
        offset = 0
        logging.info("Sampling %s: %d of %d raw rows", field, len(indices), count)
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["ra", "dec", *flags]
        ):
            end = offset + batch.num_rows
            selected = (
                indices[
                    np.searchsorted(indices, offset) : np.searchsorted(indices, end)
                ]
                - offset
            )
            if selected.size:
                frame = batch.take(selected).to_pandas()
                clean = frame[flags].eq("False").all(axis=1)
                coords = frame.loc[clean, ["ra", "dec"]]
                valid = np.isfinite(coords).all(axis=1) & coords["dec"].between(-90, 90)
                chunks.append(coords.loc[valid])
            offset = end
        samples[field] = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=["ra", "dec"], dtype=float)
        )
        logging.info(
            "%s: %d sampled rows pass BSM and coordinate checks",
            field,
            len(samples[field]),
        )
    return samples
