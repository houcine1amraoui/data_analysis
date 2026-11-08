"""Stage C: within-user consistency, between-user distinguishability, temporal stability, label validation.

All standardisation / sensor selection is fitted on T1 only (no test-period leakage).
Results are exploratory: user identity is confounded with time (T2 sits between T1 and T3).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

log = logging.getLogger("da")


def load_logs(misc_dir, cfg):
    out = {}
    for key, name in [("R1", cfg["files"]["activity_r1"]), ("R2", cfg["files"]["activity_r2"])]:
        p = Path(misc_dir) / name if misc_dir else None
        if p is not None and p.exists():
            d = pd.read_csv(p, encoding="utf-8-sig")
            d["start"], d["end"] = pd.to_datetime(d["StartTime"]), pd.to_datetime(d["EndTime"])
            out[key] = d
    return out


def _day_period(day_start, cfg):
    for k, (a, b) in cfg["periods"].items():
        if pd.Timestamp(a) <= day_start < pd.Timestamp(b):
            return k
    return None


def daily_counts(events, cols, sod0, ndays, h0, h1):
    D = np.zeros((ndays, len(cols)))
    for j, c in enumerate(cols):
        e = events[c] + sod0
        day, hr = (e // 86400).astype(int), (e % 86400) / 3600
        m = (hr >= h0) & (hr < h1) & (day < ndays)
        D[:, j] = np.bincount(day[m], minlength=ndays)[:ndays]
    return D


def _pair(Dm, ga, gb):
    ga, gb = np.asarray(ga), np.asarray(gb)
    if len(ga) == 0 or len(gb) == 0:
        return np.nan, np.nan, 0
    if np.array_equal(ga, gb):
        v = Dm[np.ix_(ga, ga)][np.triu_indices(len(ga), 1)]
    else:
        v = Dm[np.ix_(ga, gb)].ravel()
    return (float(v.mean()), float(v.std()), len(v)) if len(v) else (np.nan, np.nan, 0)


def _perm_test(Dm, ga, gb, n_perm, rng):
    idx = np.r_[ga, gb]
    na = len(ga)
    if len(ga) < 2 or len(gb) < 1:
        return np.nan

    def stat(a, b):
        w = [_pair(Dm, a, a)[0], _pair(Dm, b, b)[0] if len(b) > 1 else np.nan]
        return _pair(Dm, a, b)[0] - np.nanmean(w)
    obs = stat(np.asarray(ga), np.asarray(gb))
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(idx)
        ge += stat(p[:na], p[na:]) >= obs
    return (ge + 1) / (n_perm + 1)


def _loo_centroid(Z, labels, a, b):
    ia, ib = np.flatnonzero(labels == a), np.flatnonzero(labels == b)
    if len(ia) < 2 or len(ib) < 2:
        return np.nan
    correct = []
    for grp, other in [(ia, ib), (ib, ia)]:
        hit = []
        for i in grp:
            same = np.setdiff1d(grp, [i])
            hit.append(np.linalg.norm(Z[i] - Z[same].mean(0)) < np.linalg.norm(Z[i] - Z[other].mean(0)))
        correct.append(np.mean(hit))
    return float(np.mean(correct))


def run_behavior(prof, cls, misc_dir, cfg, variant, scope, out_dir):
    bc = cfg["behavior"]
    rng = np.random.default_rng(bc["seed"])
    t0, end_sec = prof["t0"], prof["end_sec"]
    events = prof["events"]
    h0, h1 = cfg["daytime"]
    cand = cls[cls.behavioral_candidate]
    if variant == "no_phone":
        cand = cand[~cand.is_phone.astype(bool)]
    cols = list(cand.index)
    res = dict(variant=variant, scope=scope, n_candidates=len(cols))
    if len(cols) < 3:
        log.warning("[%s/%s] <3 candidate sensors, skipping", variant, scope)
        return res
    sod0 = (t0 - t0.floor("D")).total_seconds()
    ndays = int(np.ceil((end_sec + sod0) / 86400))
    dates = pd.date_range(t0.floor("D"), periods=ndays)
    D = daily_counts(events, cols, sod0, ndays, h0, h1)
    period = np.array([_day_period(d, cfg) or "" for d in dates])
    logs = load_logs(misc_dir, cfg)
    logged = set()
    for d in logs.values():
        logged |= set(d["start"].dt.normalize())
    keep = (D.sum(1) > 0) & (period != "")
    if scope == "logged_days":
        keep &= np.array([d in logged for d in dates])
    di = np.flatnonzero(keep)
    lab = period[di].copy()
    split = pd.Timestamp(cfg["t1_split"])
    sub = np.where((lab == "T1") & (dates[di] < split), "T1a", np.where(lab == "T1", "T1b", lab))
    res["days_per_group"] = {k: int((lab == k).sum()) for k in ["T1", "T2", "T3"]}
    res["days_per_group"].update({k: int((sub == k).sum()) for k in ["T1a", "T1b"]})
    if res["days_per_group"]["T1"] < 3:
        log.warning("[%s/%s] fewer than 3 T1 days: skipping", variant, scope)
        return res

    X = np.log1p(D[di])
    t1 = lab == "T1"
    mu, sd = X[t1].mean(0), np.maximum(X[t1].std(0), bc["std_floor"])
    Z = (X - mu) / sd
    p = Z.shape[1]
    Dm = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2) / np.sqrt(p)
    share = D[di] / D[di].sum(1, keepdims=True)
    Js = np.array([[jensenshannon(a, b) for b in share] for a in share])
    out_dir = Path(out_dir)
    tag = f"{variant}__{scope}"

    rows = []
    for a, b in [("T1", "T1"), ("T1a", "T1b"), ("T1", "T3"), ("T3", "T3"), ("T1", "T2"), ("T3", "T2"), ("T2", "T2")]:
        for name, M in [("euclid_z", Dm), ("js", Js)]:
            ga = np.flatnonzero((sub == a) if a.startswith("T1") and len(a) > 2 else (lab == a))
            gb = np.flatnonzero((sub == b) if b.startswith("T1") and len(b) > 2 else (lab == b))
            m, s, k = _pair(M, ga, gb)
            rows.append(dict(group_a=a, group_b=b, metric=name, mean=m, sd=s, n_pairs=k))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out_dir / f"behavior_pair_distances__{tag}.csv", index=False)

    def idx(g):
        return np.flatnonzero(sub == g) if g in ("T1a", "T1b") else np.flatnonzero(lab == g)
    tests = []
    for a, b in [("T1", "T2"), ("T1", "T3"), ("T1a", "T1b")]:
        ga, gb = idx(a), idx(b)
        w = pairs[(pairs.group_a == "T1") & (pairs.group_b == "T1") & (pairs.metric == "euclid_z")]["mean"].iloc[0]
        c = pairs[(pairs.group_a == a) & (pairs.group_b == b) & (pairs.metric == "euclid_z")]["mean"].iloc[0]
        tests.append(dict(comparison=f"{a} vs {b}", n_a=len(ga), n_b=len(gb), cross_over_within_T1=c / w if w else np.nan,
                          perm_p=_perm_test(Dm, ga, gb, bc["n_perm"], rng),
                          loo_centroid_balanced_acc=_loo_centroid(Z, sub if "T1a" in (a, b) else lab, a, b)))
    tests = pd.DataFrame(tests)
    tests.to_csv(out_dir / f"behavior_separation__{tag}.csv", index=False)
    res["separation"] = tests

    # per-sensor effect sizes (Cohen's d on log daily counts, T1 as reference)
    es = []
    for g in ("T2", "T3"):
        ig = lab == g
        if ig.sum() >= 2:
            sp = np.sqrt(((X[t1].var(0, ddof=1) * (t1.sum() - 1)) + (X[ig].var(0, ddof=1) * (ig.sum() - 1)))
                         / (t1.sum() + ig.sum() - 2))
            d = (X[ig].mean(0) - X[t1].mean(0)) / np.maximum(sp, bc["std_floor"])
            es.append(pd.Series(d, index=cols, name=f"d_{g}_vs_T1"))
    if es:
        es = pd.concat(es, axis=1)
        es.to_csv(out_dir / f"behavior_effect_sizes__{tag}.csv")
        res["frac_abs_d_gt_0.8"] = {c: float((es[c].abs() > 0.8).mean()) for c in es.columns}

    # drift: distance of each day to the T1 centroid (leave-self-out for T1 days)
    drift = []
    for k, i in enumerate(di):
        ref = Z[t1 & (np.arange(len(di)) != k)].mean(0)
        drift.append(np.linalg.norm(Z[k] - ref) / np.sqrt(p))
    drift = pd.DataFrame({"date": dates[di], "group": sub, "dist_to_T1_centroid": drift})
    drift.to_csv(out_dir / f"behavior_drift__{tag}.csv", index=False)

    # hour-of-day profiles on selected days
    prof_h = {}
    dayset = {g: set(dates[di][sub == g]) for g in ["T1", "T2", "T3"]}
    for g in ["T1", "T2", "T3"]:
        dayset[g] = set(dates[di][lab == g])
    for g in ["T1", "T2", "T3"]:
        H = np.zeros(24)
        for c in cols:
            e = events[c] + sod0
            day = pd.DatetimeIndex(dates[np.minimum((e // 86400).astype(int), ndays - 1)])
            m = np.asarray(day.isin(list(dayset[g])))
            H += np.bincount(((e[m] % 86400) // 3600).astype(int), minlength=24)[:24]
        prof_h[g] = H / max(len(dayset[g]), 1)
    hp = pd.DataFrame(prof_h)
    hp.to_csv(out_dir / f"behavior_hourly_profile__{tag}.csv")
    res["hourly_js"] = {f"{a}-{b}": float(jensenshannon(hp[a] + 1e-12, hp[b] + 1e-12))
                        for a, b in [("T1", "T3"), ("T1", "T2"), ("T3", "T2")] if hp[a].sum() and hp[b].sum()}

    # co-activation structure (minute-level activity), sensors selected on T1
    nmin = int(np.ceil((end_sec + sod0) / 60))
    A = np.zeros((nmin, len(cols)), dtype=bool)
    for j, c in enumerate(cols):
        A[np.minimum(((events[c] + sod0) // 60).astype(int), nmin - 1), j] = True
    mins = pd.date_range(t0.floor("D"), periods=nmin, freq="min")
    day_of = mins.normalize()
    hr = mins.hour
    inday = (hr >= h0) & (hr < h1)
    gmask = {}
    for g, ii in [("T1", idx("T1")), ("T1a", idx("T1a")), ("T1b", idx("T1b")), ("T2", idx("T2")), ("T3", idx("T3"))]:
        gmask[g] = np.asarray(day_of.isin(list(dates[di][ii]))) & inday
    sel = A[gmask["T1"]].sum(0) >= bc["min_active_minutes"]
    if sel.sum() >= 3:
        def corr(g):
            M = A[gmask[g]][:, sel].astype(float)
            if len(M) < 30:
                return None
            with np.errstate(invalid="ignore", divide="ignore"):
                C = np.corrcoef(M.T)
            return np.nan_to_num(C)
        Cs = {g: corr(g) for g in gmask}
        q = int(sel.sum())
        co = []
        for a, b in [("T1a", "T1b"), ("T1", "T3"), ("T1", "T2"), ("T3", "T2")]:
            if Cs.get(a) is not None and Cs.get(b) is not None:
                co.append(dict(group_a=a, group_b=b, n_sensors=q,
                               frobenius_per_pair=float(np.linalg.norm(Cs[a] - Cs[b]) / np.sqrt(q * (q - 1)))))
        co = pd.DataFrame(co)
        co.to_csv(out_dir / f"behavior_coactivation__{tag}.csv", index=False)
        res["coactivation"] = co
    res["_artifacts"] = dict(Dm=Dm, Js=Js, dates=dates[di], sub=sub, drift=drift, hourly=hp, Z=Z)
    return res


def label_alignment(prof, cls, misc_dir, cfg, out_dir):
    """Does the sensor set listed per ActID respond during logged windows? Also tests log/clock offset."""
    import json
    logs = load_logs(misc_dir, cfg)
    lp = Path(misc_dir) / cfg["files"]["activity_labels"] if misc_dir else None
    if "R1" not in logs or lp is None or not lp.exists():
        return None
    labels = json.load(open(lp))
    t0, events = prof["t0"], prof["events"]
    r1 = logs["R1"].copy()
    r1["act"] = r1["ActID"].astype(float).round(1).astype(str)
    cu_pid = cls[cls.source == "CU"].groupby("pid").groups
    bre_loc = cls[cls.source == "BRE"].groupby("device").groups
    phy_dev = cls[cls.source == "PHY"].groupby("pid").groups
    h0, h1 = cfg["daytime"]

    def sensors_for(act):
        d = labels.get(act, {}).get("Databases", {})
        cols = []
        for pid in d.get("CU_CyPhy", []):
            cols += [c for c in cu_pid.get(pid, []) if not cls.loc[c, "housekeeping"]]
        for loc in d.get("BRE_CyPhy", []):
            code = str(loc).split("_")[-1]
            cols += [c for c in bre_loc.get(code, []) if not cls.loc[c, "housekeeping"]]
        for dev in d.get("Physical", []):
            cols += list(phy_dev.get(dev, []))
        return [c for c in dict.fromkeys(cols) if c in events and len(events[c])]

    def rate(ev, starts, ends):
        return np.searchsorted(ev, ends) - np.searchsorted(ev, starts)

    results = []
    dst = pd.Timestamp(cfg["dst_change"])
    for off_h in cfg["behavior"]["offsets_hours"]:
        shift = off_h * 3600.0
        for act, g in r1.groupby("act"):
            cols = sensors_for(act)
            if not cols:
                continue
            s = ((g["start"] - t0).dt.total_seconds().to_numpy() + shift)
            e = ((g["end"] - t0).dt.total_seconds().to_numpy() + shift)
            pre = (g["start"] < dst).to_numpy()
            dur_min = (e - s) / 60
            # baseline: daytime events on logged days outside windows (approximate: all-day-time rate per minute)
            for era, m in [("pre_dst", pre), ("post_dst", ~pre)]:
                if m.sum() == 0:
                    continue
                enr = []
                for c in cols:
                    ev = events[c]
                    inn = rate(ev, s[m], e[m]).sum() / max(dur_min[m].sum(), 1e-9)
                    days = pd.date_range(g["start"][m].min().normalize(), g["start"][m].max().normalize())
                    day_s = ((days - t0).total_seconds().to_numpy() + h0 * 3600)
                    day_e = ((days - t0).total_seconds().to_numpy() + h1 * 3600)
                    base_cnt = rate(ev, day_s, day_e).sum() - rate(ev, s[m], e[m]).sum()
                    base_min = ((h1 - h0) * 60 * len(days)) - dur_min[m].sum()
                    base = max(base_cnt, 0) / max(base_min, 1e-9)
                    enr.append((inn + 1e-6) / (base + 1e-6))
                results.append(dict(offset_h=off_h, era=era, act_id=act,
                                    name=labels.get(act, {}).get("Name", ""), n_windows=int(m.sum()),
                                    n_sensor_cols=len(cols), median_enrichment=float(np.median(enr)),
                                    frac_cols_enriched_gt1_5=float(np.mean(np.array(enr) > 1.5))))
    if not results:
        return None
    r = pd.DataFrame(results)
    r.to_csv(Path(out_dir) / "activity_label_validation.csv", index=False)
    summ = (r.groupby(["era", "offset_h"]).apply(lambda x: np.average(np.log(x.median_enrichment),
                                                                   weights=x.n_windows), include_groups=False)
            .rename("weighted_mean_log_enrichment").reset_index())
    summ.to_csv(Path(out_dir) / "activity_clock_offset_check.csv", index=False)
    return summ
