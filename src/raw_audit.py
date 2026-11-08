"""Optional RAW-file audit (true event rates, duplicates, gaps). UNTESTED on the real RAW files:
column names follow the paper (state_id, entity_id, last_updated); BRE weekly files are auto-detected."""
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("da")


def audit_cu_raw(root, cfg, out):
    files = sorted(Path(root).rglob(cfg["raw_audit"]["cu_raw_glob"]))
    if not files:
        log.warning("No CU RAW files found for %s", cfg["raw_audit"]["cu_raw_glob"])
        return
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c.strip() in {"state_id", "entity_id", "state", "last_updated"})
        d.columns = [c.strip() for c in d.columns]
        d["file"] = f.name
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["last_updated"] = pd.to_datetime(d["last_updated"], errors="coerce")
    summ = dict(files=len(files), rows=len(d), duplicate_state_id=int(d.state_id.duplicated().sum()),
                bad_timestamps=int(d.last_updated.isna().sum()))
    d = d.drop_duplicates("state_id").sort_values(["entity_id", "last_updated"])
    d["dt"] = d.groupby("entity_id")["last_updated"].diff().dt.total_seconds()
    per = d.groupby("entity_id").agg(n_events=("state_id", "size"), first=("last_updated", "min"),
                                     last=("last_updated", "max"), median_dt_s=("dt", "median"),
                                     max_dt_s=("dt", "max"), n_states=("state", "nunique"))
    per.to_csv(Path(out) / "A_raw_cu_entity_event_stats.csv")
    pd.Series(summ).to_csv(Path(out) / "A_raw_cu_summary.csv")


def audit_bre_raw(root, cfg, out):
    files = sorted(Path(root).rglob(cfg["raw_audit"]["bre_raw_glob"]))
    if not files:
        log.warning("No BRE RAW files found for %s", cfg["raw_audit"]["bre_raw_glob"])
        return
    ts = []
    for f in files:
        d = pd.read_csv(f)
        for c in d.columns:
            t = pd.to_datetime(d[c], errors="coerce")
            if t.notna().mean() > 0.9:
                ts.append(t.dropna())
                break
    t = pd.concat(ts).sort_values().drop_duplicates()
    dt = t.diff().dt.total_seconds()
    thr = cfg["raw_audit"]["bre_gap_minutes"] * 60
    g = pd.DataFrame({"gap_start": t.shift(1)[dt > thr], "gap_end": t[dt > thr], "gap_h": dt[dt > thr] / 3600})
    g.sort_values("gap_h", ascending=False).to_csv(Path(out) / "A_raw_bre_gaps.csv", index=False)
