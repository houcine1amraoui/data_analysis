"""Single pass over every column: run-length encoding makes all stats cheap (data is a 1 Hz step function).

Produces, per column: quality stats (Stage A), informativeness stats (Stage B), valid change events
(for behaviour, Stage C), a 1-minute aggregated series and hourly event counts (for Stage D).
"""
import logging
import re
import time

import numpy as np
import pandas as pd

log = logging.getLogger("da")
FIELDS = ["window_days", "unavailable_share", "n_unique", "top_share", "top_value", "mean", "std", "min", "max",
          "n_changes", "changes_per_day"]


def to_sec(x, t0):
    return float((pd.Timestamp(x) - t0).total_seconds())


def period_bounds(cfg, t0):
    return {k: (to_sec(v[0], t0), to_sec(v[1], t0)) for k, v in cfg["periods"].items()}


def _clip(rt, rend, a, b):
    return np.clip(np.minimum(rend, b) - np.maximum(rt, a), 0, None)


def _wstats(rt, rend, rv, valid, ev, a, b):
    out = dict.fromkeys(FIELDS, np.nan)
    out["n_changes"] = 0
    if b <= a:
        return out
    d = _clip(rt, rend, a, b)
    tot = d.sum()
    if tot <= 0:
        return out
    out["window_days"] = tot / 86400
    m = valid & (d > 0)
    dv = d[m].sum()
    out["unavailable_share"] = 1 - dv / tot
    nch = int(((ev >= a) & (ev < b)).sum())
    out["n_changes"], out["changes_per_day"] = nch, nch / (tot / 86400)
    if dv <= 0:
        out["n_unique"] = 0
        return out
    vals, w = rv[m].astype(np.float64), d[m]
    uv, inv = np.unique(vals, return_inverse=True)
    ws = np.bincount(inv, weights=w)
    out.update(n_unique=len(uv), top_share=ws.max() / ws.sum(), top_value=uv[ws.argmax()], min=uv[0], max=uv[-1])
    mu = np.average(vals, weights=w)
    out.update(mean=mu, std=float(np.sqrt(np.average((vals - mu) ** 2, weights=w))))
    return out


def _range_violation(col, rv, cls, rdur, rules):
    low = col.lower()
    for r in rules:
        if re.search(r["pattern"], low):
            vals = rv.astype(np.float64)
            bad = ~np.isin(vals, [0.0, 1.0]) if r.get("binary") else (vals < r["lo"]) | (vals > r["hi"])
            m = cls == "valid"
            den = rdur[m].sum()
            return (float(rdur[m & bad].sum() / den) if den > 0 else np.nan), r["pattern"]
    return np.nan, "no_rule"


def timestamp_report(ts, grid):
    tsec = (ts - ts[0]) / np.timedelta64(1, "s")
    d = np.diff(tsec)
    rep = dict(n_rows=len(ts), start=str(ts[0]), end=str(ts[-1]), monotonic=bool((d >= 0).all()),
               duplicate_timestamps=int((d == 0).sum()), expected_grid_s=grid,
               share_on_grid=float(np.mean(np.isclose(d, grid))) if len(d) else np.nan,
               min_step_s=float(d.min()), median_step_s=float(np.median(d)), max_step_s=float(d.max()),
               n_gaps_gt_5x_grid=int((d > 5 * grid).sum()))
    gaps = pd.DataFrame({"gap_start": ts[:-1][d > 5 * grid], "gap_end": ts[1:][d > 5 * grid],
                         "gap_s": d[d > 5 * grid]}).sort_values("gap_s", ascending=False).head(50)
    return rep, gaps


def profile_columns(store, meta, decoder, cfg):
    ts = store.timestamps()
    t0 = pd.Timestamp(ts[0])
    tsec = (ts - ts[0]) / np.timedelta64(1, "s")
    grid = float(cfg["grid_seconds"])
    end_sec = float(tsec[-1] + grid)
    n = len(ts)
    minute_id = np.floor(tsec / 60).astype(np.int64)
    nmin, nhours = int(minute_id[-1]) + 1, int(np.ceil(end_sec / 3600))
    bounds = period_bounds(cfg, t0)
    bounds["ALL"] = (0.0, end_sec)
    vr = {s: (to_sec(a, t0), to_sec(b, t0)) for s, (a, b) in cfg.get("source_valid_ranges", {}).items()}
    cc = cfg["classification"]
    hk = re.compile(cc["housekeeping_regex"])

    rep, gaps = timestamp_report(ts, grid)
    records, events, minute, hourly = [], {}, {}, {}
    t_start = time.time()
    for i, col in enumerate(store.columns):
        v = store.get(col)
        src = meta.loc[col, "source"]
        nanv = np.isnan(v)
        same = (v[1:] == v[:-1]) | (nanv[1:] & nanv[:-1])
        starts = np.r_[0, np.flatnonzero(~same) + 1]
        rv, rt = v[starts], tsec[starts]
        rend = np.r_[rt[1:], end_sec]
        rdur = rend - rt
        lens = np.diff(np.r_[starts, n])
        cls = decoder.classify_values(rv) if src == "CU" else np.where(np.isnan(rv), "nan", "valid").astype(object)
        valid = np.isin(cls, ["valid", "state"])
        k = np.flatnonzero(valid[1:] & valid[:-1]) + 1
        ev = rt[k]
        events[col] = ev
        tot = rdur.sum()
        sh = lambda c: float(rdur[cls == c].sum() / tot)

        rec = dict(column=col, source=src, n_runs=len(rv), nan_share=sh("nan"), missing_marker_share=sh("missing"),
                   state_code_share=sh("state"), nonbehavioral_share=sh("nonbehavioral"),
                   temporal_proxy_share=sh("temporal"), n_availability_changes=int((valid[1:] != valid[:-1]).sum()))
        vals_all = rv[valid].astype(np.float64)
        uv = np.unique(vals_all)
        rec["n_unique_all"] = len(uv)
        if len(uv) == 0:
            rec["data_type"] = "empty"
        elif np.isin(uv, [0.0, 1.0]).all():
            rec["data_type"] = "binary"
        elif rec["state_code_share"] > 0.01:
            rec["data_type"] = "categorical"
        elif len(uv) <= 1:
            rec["data_type"] = "constant"
        elif len(uv) <= 12:
            rec["data_type"] = "discrete"
        else:
            rec["data_type"] = "continuous"
        rec["invalid_share"], rec["range_rule"] = _range_violation(col, rv, cls, rdur, cfg["range_rules"])
        first_valid = float(rt[valid][0]) if valid.any() else np.nan
        rec["first_valid"] = str(t0 + pd.Timedelta(seconds=first_valid)) if valid.any() else ""
        if len(ev):
            gp = np.diff(np.r_[0.0, ev, end_sec])
            pos = np.r_[0.0, ev]
            rec.update(first_change=str(t0 + pd.Timedelta(seconds=float(ev[0]))),
                       last_change=str(t0 + pd.Timedelta(seconds=float(ev[-1]))), longest_flat_h=gp.max() / 3600,
                       longest_flat_start=str(t0 + pd.Timedelta(seconds=float(pos[gp.argmax()]))))
            last_ev = float(ev[-1])
        else:
            rec.update(first_change="", last_change="", longest_flat_h=end_sec / 3600, longest_flat_start="")
            last_ev = first_valid
        va_ = vr.get(src)
        eff_end = min(end_sec, va_[1]) if va_ else end_sec          # padding after the real recording is not "frozen"
        rec["tail_flat_h"] = max(eff_end - last_ev, 0.0) / 3600 if valid.any() else np.nan
        thr_h = cfg["freeze"]["thresholds_h"].get(src, cfg["freeze"]["thresholds_h"].get("OTHER", 48))
        frozen = valid.any() and rec["data_type"] in cfg["freeze"]["types"] and rec["tail_flat_h"] >= thr_h
        rec["freeze_start_sec"] = last_ev if frozen else np.nan
        rec["frozen_tail"] = bool(frozen)
        rec["housekeeping"] = bool(hk.search(col.lower()))
        va = vr.get(src)
        rec["valid_range_share"] = ((min(va[1], end_sec) - max(va[0], 0)) / end_sec) if va else 1.0
        for pname, (a, b) in bounds.items():
            if va:
                a, b = max(a, va[0]), min(b, va[1])
            st = _wstats(rt, rend, rv, valid, ev, a, b)
            if rec["data_type"] == "categorical":
                st["mean"] = st["std"] = np.nan
            for f in FIELDS:
                rec[f"{pname.lower()}_{f}"] = st[f]
        records.append(rec)

        # 1-minute series (mean for numeric/binary, last value for categorical)
        valid_el = np.repeat(valid, lens)
        if rec["data_type"] == "categorical":
            last = np.r_[np.flatnonzero(np.diff(minute_id)), n - 1]
            ser = np.full(nmin, np.nan, dtype=np.float32)
            ser[minute_id[last]] = np.where(valid_el[last], v[last], np.nan)
        else:
            s = np.bincount(minute_id, weights=np.where(valid_el, v, 0).astype(np.float64), minlength=nmin)
            c = np.bincount(minute_id, weights=valid_el.astype(np.float64), minlength=nmin)
            with np.errstate(invalid="ignore", divide="ignore"):
                ser = (s / c).astype(np.float32)
        minute[col] = ser
        hourly[col] = np.bincount((ev // 3600).astype(np.int64), minlength=nhours)[:nhours]
        if (i + 1) % 50 == 0 or i + 1 == len(store.columns):
            log.info("  profiled %d/%d columns (%.0fs)", i + 1, len(store.columns), time.time() - t_start)

    prof = pd.DataFrame(records).set_index("column", drop=False).rename_axis(None)
    mdf = pd.DataFrame(minute, index=t0 + pd.to_timedelta(np.arange(nmin) * 60, unit="s"))
    hdf = pd.DataFrame(hourly, index=t0.floor("h") + pd.to_timedelta(np.arange(nhours), unit="h")).T

    # Joint inactivity per source (catches outages that forward-fill turned into "flat" data)
    jg = []
    for src, thr in cfg["joint_gap_hours"].items():
        cols = [c for c in store.columns if meta.loc[c, "source"] == src and len(events[c])]
        if not cols:
            continue
        e = np.sort(np.concatenate([events[c] for c in cols]))
        g = np.diff(np.r_[0.0, e, end_sec])
        pos = np.r_[0.0, e]
        for idx in np.flatnonzero(g / 3600 >= thr):
            jg.append(dict(source=src, start=t0 + pd.Timedelta(seconds=float(pos[idx])),
                           end=t0 + pd.Timedelta(seconds=float(pos[idx] + g[idx])), hours=g[idx] / 3600,
                           threshold_h=thr))
    jg = pd.DataFrame(jg).sort_values(["source", "start"]) if jg else pd.DataFrame(
        columns=["source", "start", "end", "hours", "threshold_h"])
    # Joint unavailability per source (e.g. Home Assistant offline -> every CU column 'unknown')
    ua, ucfg = [], cfg["unavailable_runs"]
    for src in sorted(set(meta.loc[store.columns, "source"])):
        cols = [c for c in store.columns if meta.loc[c, "source"] == src]
        share = mdf[cols].isna().mean(axis=1).to_numpy() >= ucfg["min_share"]
        edges = np.flatnonzero(np.diff(np.r_[0, share.astype(int), 0]))
        for a_, b_ in zip(edges[::2], edges[1::2]):
            if b_ - a_ >= ucfg["min_minutes"]:
                ua.append(dict(source=src, start=mdf.index[a_], end=mdf.index[b_ - 1], hours=(b_ - a_) / 60,
                               n_columns=len(cols)))
    ua = pd.DataFrame(ua) if ua else pd.DataFrame(columns=["source", "start", "end", "hours", "n_columns"])
    return dict(profile=prof, events=events, minute=mdf, hourly=hdf, t0=t0, end_sec=end_sec,
                ts_report=rep, ts_gaps=gaps, joint_gaps=jg, joint_unavailable=ua, bounds=bounds)
