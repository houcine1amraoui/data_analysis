"""File discovery, chunked CSV -> Parquet conversion, and per-column access."""
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("da")


def find_file(root, name):
    hits = sorted(Path(root).rglob(name))
    return hits[0] if hits else None


def find_misc_dir(cfg):
    if cfg["paths"].get("misc_dir"):
        return Path(cfg["paths"]["misc_dir"])
    hit = find_file(cfg["dataset_folder"], cfg["files"]["states_map"])
    return hit.parent if hit else None


def locate_masters(cfg):
    """Return list of master CSV paths (combined file preferred, otherwise fallback trio)."""
    if cfg["paths"].get("master_files"):
        return [Path(p) for p in cfg["paths"]["master_files"]]
    root = cfg["dataset_folder"]
    found = [find_file(root, n) for n in cfg["files"]["master"]]
    found = [f for f in found if f]
    if found:
        return found
    log.warning("Combined master not found; falling back to %s", cfg["files"]["fallback_masters"])
    found = [find_file(root, n) for n in cfg["files"]["fallback_masters"]]
    return [f for f in found if f]


def convert_csv_to_parquet(csv_path, pq_path, cfg):
    """Chunked CSV -> Parquet. Strips column names, coerces non-numeric cells to NaN (logged)."""
    ts_col = cfg["timestamp_col"]
    dtype = np.dtype(cfg["value_dtype"])
    tmp = str(pq_path) + ".tmp"
    writer, n_rows, coerced = None, 0, {}
    t = time.time()
    for chunk in pd.read_csv(csv_path, chunksize=int(cfg["chunk_rows"])):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk[ts_col] = pd.to_datetime(chunk[ts_col])
        for c in chunk.columns:
            if c == ts_col:
                continue
            if not pd.api.types.is_numeric_dtype(chunk[c]):
                num = pd.to_numeric(chunk[c], errors="coerce")
                coerced[c] = coerced.get(c, 0) + int((num.isna() & chunk[c].notna()).sum())
                chunk[c] = num
            chunk[c] = chunk[c].astype(dtype)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
        n_rows += len(chunk)
        log.info("  converted %s rows (%.0fs)", f"{n_rows:,}", time.time() - t)
    writer.close()
    os.replace(tmp, pq_path)
    coerced = {k: v for k, v in coerced.items() if v}
    if coerced:
        log.warning("Non-numeric cells coerced to NaN: %s", coerced)
    return n_rows


def prepare_store(cfg):
    masters = locate_masters(cfg)
    if not masters:
        raise FileNotFoundError(f"No master CSV found under {cfg['dataset_folder']}")
    cache = Path(cfg["paths"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    pq_paths = []
    for m in masters:
        p = cache / (m.stem + f"_{cfg['value_dtype']}.parquet")
        if p.exists() and not cfg.get("force_reconvert") and p.stat().st_mtime >= m.stat().st_mtime:
            log.info("Using cached %s", p.name)
        else:
            log.info("Converting %s -> %s", m, p.name)
            convert_csv_to_parquet(m, p, cfg)
        pq_paths.append(p)
    return ColumnStore(pq_paths, cfg["timestamp_col"])


class ColumnStore:
    """Column-at-a-time access to one or several aligned Parquet files."""

    def __init__(self, paths, ts_col):
        self.ts_col, self.paths, self.map = ts_col, [str(p) for p in paths], {}
        for p in self.paths:
            for name in pq.read_schema(p).names:
                if name == ts_col:
                    continue
                if name in self.map:
                    log.warning("Duplicate column %s in %s ignored", name, p)
                    continue
                self.map[name] = p
        self.columns = list(self.map)
        self._ts = None

    def timestamps(self):
        if self._ts is None:
            ts = pq.read_table(self.paths[0], columns=[self.ts_col]).column(0).to_pandas().to_numpy()
            self._ts = ts.astype("datetime64[ns]")
            for p in self.paths[1:]:
                n = pq.ParquetFile(p).metadata.num_rows
                if n != len(self._ts):
                    raise ValueError(f"{p} has {n} rows, expected {len(self._ts)} (files not aligned)")
        return self._ts

    def get(self, col):
        return pq.read_table(self.map[col], columns=[col]).column(0).to_pandas().to_numpy()
