"""Bounded-memory catalog sampling for inspection plots."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
