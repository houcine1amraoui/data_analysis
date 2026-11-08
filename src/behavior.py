"""Stage C: within-user consistency, between-user distinguishability, temporal stability, label validation.

Feature families are kept apart on purpose:
  events : binary sensors -> log1p(daytime state-change count)      (behaviour-like signals)
  levels : continuous/discrete channels -> daytime mean and std      (environment-like signals)
Never count "changes" of noisy continuous channels: that measures sensor chatter, not behaviour.

Frozen (terminal-flatline) sensors are masked per day. Standardisation is fitted on T1 only.
Results are exploratory: user identity is confounded with time (T2 sits between T1 and T3).
"""
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("da")
warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices are expected after masking


# ---------------------------------------------------------------- helpers
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


def _mdist(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return (np.sqrt(np.mean((a[m] - b[m]) ** 2)), int(m.sum())) if m.any() else (np.nan, 0)


def masked_distance_matrix(Z, min_overlap):
    n = len(Z)
    V = ~np.isnan(Z)
    Z0 = np.nan_to_num(Z)
    Dm, Ov = np.full((n, n), np.nan), np.zeros((n, n), int)
    for i in range(n):
        both = V[i] & V
        cnt = both.sum(1)
        d = np.sqrt((((Z0[i] - Z0) ** 2) * both).sum(1) / np.maximum(cnt, 1))
        d[cnt < min_overlap] = np.nan
        Dm[i], Ov[i] = d, cnt
    return Dm, Ov


def _pair(Dm, ga, gb):
    ga, gb = np.asarray(ga, int), np.asarray(gb, int)
    if len(ga) == 0 or len(gb) == 0:
        return np.nan, np.nan, 0
    v = Dm[np.ix_(ga, ga)][np.triu_indices(len(ga), 1)] if np.array_equal(ga, gb) else Dm[np.ix_(ga, gb)].ravel()
    v = v[~np.isnan(v)]
    return (float(v.mean()), float(v.std()), len(v)) if len(v) else (np.nan, np.nan, 0)


def _perm_test(Dm, ga, gb, n_perm, rng):
    if len(ga) < 2 or len(gb) < 2:
        return np.nan
    idx, na = np.r_[ga, gb], len(ga)

    def stat(a, b):
        return _pair(Dm, a, b)[0] - np.nanmean([_pair(Dm, a, a)[0], _pair(Dm, b, b)[0]])
    obs = stat(np.asarray(ga), np.asarray(gb))
    if np.isnan(obs):
        return np.nan
    ge = sum(stat(p[:na], p[na:]) >= obs for p in (rng.permutation(idx) for _ in range(n_perm)))
    return (ge + 1) / (n_perm + 1)


def _loo_centroid(Z, labels, a, b):
    ia, ib = np.flatnonzero(labels == a), np.flatnonzero(labels == b)
    if len(ia) < 2 or len(ib) < 2:
        return np.nan
    accs = []
    for grp, other in [(ia, ib), (ib, ia)]:
        hit = []
        for i in grp:
            ca = np.nanmean(Z[np.setdiff1d(grp, [i])], axis=0)
            cb = np.nanmean(Z[other], axis=0)
            hit.append(_mdist(Z[i], ca)[0] < _mdist(Z[i], cb)[0])
        accs.append(np.mean(hit))
    return float(np.mean(accs))


# ---------------------------------------------------------------- features
def daily_features(prof, cls, cols_ev, cols_lv, sod0, ndays, h0, h1, cfg):
    F, names, kinds, until = [], [], [], []
    for c in cols_ev:
        e = prof["events"][c] + sod0
        day, hr = (e // 86400).astype(int), (e % 86400) / 3600
        m = (hr >= h0) & (hr < h1) & (day < ndays)
        F.append(np.log1p(np.bincount(day[m], minlength=ndays)[:ndays]).astype(float))
        names.append(f"{c}|events"), kinds.append("events"), until.append(c)
    if cols_lv:
        M = prof["minute"][cols_lv]
        k = np.arange(len(M))
        t = k * 60.0 + sod0
        day, hr = (t // 86400).astype(int), (t % 86400) / 3600
        m = (hr >= h0) & (hr < h1) & (day < ndays)
        g = M[m].groupby(day[m])
        mean, std, cnt = g.mean().reindex(range(ndays)), g.std().reindex(range(ndays)), g.count().reindex(range(ndays))
        for c in cols_lv:
            ok = (cnt[c] >= 60).to_numpy()
            F.append(np.where(ok, mean[c].to_numpy(), np.nan))
            F.append(np.where(ok, std[c].to_numpy(), np.nan))
            names.extend([f"{c}|mean", f"{c}|std"])
            kinds.extend(["levels", "levels"])
            until.extend([c, c])
    return np.array(F).T if F else np.zeros((ndays, 0)), names, kinds, until


def usable_mask(until_cols, cls, cfg, t0, sod0, ndays):
    U = np.ones((ndays, len(until_cols)), bool)
    day0 = np.arange(ndays) * 86400.0 - sod0
    day1 = day0 + 86400.0
    for j, c in enumerate(until_cols):
        u = cls.loc[c, "usable_until_sec"]
        if pd.notna(u):
            U[:, j] &= day1 <= u + 1e-6
        va = cfg.get("source_valid_ranges", {}).get(cls.loc[c, "source"])
        if va:
            a, b = [(pd.Timestamp(x) - t0).total_seconds() for x in va]
            U[:, j] &= (day0 >= a) & (day1 <= b + 86400)
    return U


# ---------------------------------------------------------------- main analysis
def run_behavior(prof, cls, misc_dir, cfg, variant, fset, scope, out_dir):
    bc = cfg["behavior"]
    rng = np.random.default_rng(bc["seed"])
    t0, end_sec = prof["t0"], prof["end_sec"]
    h0, h1 = cfg["daytime"]
    cand = cls[cls.behavioral_candidate]
    if variant == "no_phone":
        if not cand.is_phone.astype(bool).any():
            return None
        cand = cand[~cand.is_phone.astype(bool)]
    elif variant in ("CU", "BRE", "PHY"):
        cand = cand[cand.source == variant]
    cols_ev = list(cand.index[cand.data_type == "binary"]) if fset in ("events", "combined") else []
    cols_lv = list(cand.index[cand.data_type.isin(["continuous", "discrete"])]) if fset in ("levels", "combined") else []
    res = dict(variant=variant, fset=fset, scope=scope)
    if len(cols_ev) + len(cols_lv) < 3:
        return None
    sod0 = (t0 - t0.floor("D")).total_seconds()
    ndays = int(np.ceil((end_sec + sod0) / 86400))
    dates = pd.date_range(t0.floor("D"), periods=ndays)
    F, names, kinds, until = daily_features(prof, cls, cols_ev, cols_lv, sod0, ndays, h0, h1, cfg)
    U = usable_mask(until, cls, cfg, t0, sod0, ndays)
    F = np.where(U, F, np.nan)
    period = np.array([_day_period(d, cfg) or "" for d in dates])
    logged = set()
    for d in load_logs(misc_dir, cfg).values():
        logged |= set(d["start"].dt.normalize())
    keep = (period != "") & (np.isfinite(F).sum(1) >= bc["min_overlap_features"])
    if scope == "logged_days":
        keep &= np.array([d in logged for d in dates])
    di = np.flatnonzero(keep)
    lab = period[di]
    split = pd.Timestamp(cfg["t1_split"])
    sub = np.where((lab == "T1") & (dates[di] < split), "T1a", np.where(lab == "T1", "T1b", lab))
    res["n_features"] = F.shape[1]
    res["days"] = {k: int((lab == k).sum()) for k in ["T1", "T2", "T3"]}
    t1 = lab == "T1"
    if t1.sum() < 4:
        return None
    X = F[di]
    ok_t1 = np.isfinite(X[t1]).sum(0) >= 3
    mu = np.nanmean(np.where(ok_t1, X, np.nan)[t1], axis=0)
    sd = np.nanstd(np.where(ok_t1, X, np.nan)[t1], axis=0)
    isev = np.array([k == "events" for k in kinds])
    floor = np.where(isev, bc["std_floor"], np.maximum(0.02 * np.abs(mu), 1e-6))
    Z = (X - mu) / np.maximum(sd, floor)
    Z[:, ~ok_t1] = np.nan
    res["n_features_used"] = int(ok_t1.sum())
    Dm, Ov = masked_distance_matrix(Z, bc["min_overlap_features"])
    res["median_feature_overlap"] = float(np.median(Ov[np.triu_indices(len(Z), 1)]))
    out_dir = Path(out_dir)
    tag = f"{variant}__{fset}__{scope}"

    def idx(g):
        return np.flatnonzero(sub == g) if g in ("T1a", "T1b") else np.flatnonzero(lab == g)
    pairs = []
    for a, b in [("T1", "T1"), ("T1a", "T1b"), ("T1", "T3"), ("T3", "T3"), ("T1", "T2"), ("T3", "T2"), ("T2", "T2")]:
        m, s, k = _pair(Dm, idx(a), idx(b))
        pairs.append(dict(group_a=a, group_b=b, mean=m, sd=s, n_pairs=k))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out_dir / f"behavior_pair_distances__{tag}.csv", index=False)
    w = pairs[(pairs.group_a == "T1") & (pairs.group_b == "T1")]["mean"].iloc[0]
    rows = []
    for a, b in [("T1", "T2"), ("T1", "T3"), ("T2", "T3"), ("T1a", "T1b")]:
        ga, gb = idx(a), idx(b)
        c = _pair(Dm, ga, gb)[0]
        rows.append(dict(comparison=f"{a} vs {b}", n_a=len(ga), n_b=len(gb),
                         cross_over_within_T1=c / w if w else np.nan, perm_p=_perm_test(Dm, ga, gb, bc["n_perm"], rng),
                         loo_balanced_acc=_loo_centroid(Z, sub if "T1a" in (a, b) else lab, a, b)))
    res["separation"] = pd.DataFrame(rows)
    res["separation"].to_csv(out_dir / f"behavior_separation__{tag}.csv", index=False)

    # effect sizes vs T1 (Cohen d on the standardised features)
    es = {}
    for g in ("T2", "T3"):
        ig = lab == g
        if ig.sum() >= 2:
            a1, a2 = Z[t1], Z[ig]
            n1, n2 = np.isfinite(a1).sum(0), np.isfinite(a2).sum(0)
            with np.errstate(all="ignore"):
                sp = np.sqrt((np.nanvar(a1, 0, ddof=1) * (n1 - 1) + np.nanvar(a2, 0, ddof=1) * (n2 - 1)) / (n1 + n2 - 2))
                es[f"d_{g}_vs_T1"] = (np.nanmean(a2, 0) - np.nanmean(a1, 0)) / np.maximum(sp, 0.25)
    if es:
        es = pd.DataFrame(es, index=names)
        es.to_csv(out_dir / f"behavior_effect_sizes__{tag}.csv")
        res["frac_abs_d_gt_0.8"] = {k: float((v.abs() > 0.8).mean()) for k, v in es.items()}
        res["_effect"] = es

    # drift: distance to T1 centroid (leave-self-out)
    drift = []
    for k in range(len(di)):
        ref = np.nanmean(Z[t1 & (np.arange(len(di)) != k)], axis=0)
        drift.append(_mdist(Z[k], ref)[0])
    drift = pd.DataFrame({"date": dates[di], "group": sub, "dist_to_T1_centroid": drift})
    drift.to_csv(out_dir / f"behavior_drift__{tag}.csv", index=False)

    # hourly profile + co-activation: binary event sensors only
    hp, cols_c = None, [c for c in cols_ev if pd.isna(cls.loc[c, "usable_until_sec"])]  # never frozen
    if len(cols_c) >= 3 and fset in ("events", "combined"):
        dayset = {g: list(dates[di][lab == g]) for g in ["T1", "T2", "T3"]}
        prof_h = {}
        for g in dayset:
            H = np.zeros(24)
            for c in cols_c:
                e = prof["events"][c] + sod0
                dd = pd.DatetimeIndex(dates[np.minimum((e // 86400).astype(int), ndays - 1)])
                msk = np.asarray(dd.isin(dayset[g]))
                H += np.bincount(((e[msk] % 86400) // 3600).astype(int), minlength=24)[:24]
            prof_h[g] = H / max(len(dayset[g]), 1)
        hp = pd.DataFrame(prof_h)
        hp.to_csv(out_dir / f"behavior_hourly_profile__{tag}.csv")
        nmin = int(np.ceil((end_sec + sod0) / 60))
        A = np.zeros((nmin, len(cols_c)), bool)
        for j, c in enumerate(cols_c):
            A[np.minimum(((prof["events"][c] + sod0) // 60).astype(int), nmin - 1), j] = True
        mins = pd.date_range(t0.floor("D"), periods=nmin, freq="min")
        inday = (mins.hour >= h0) & (mins.hour < h1)
        gm = {g: np.asarray(mins.normalize().isin(list(dates[di][ii]))) & inday
              for g, ii in [("T1", idx("T1")), ("T1a", idx("T1a")), ("T1b", idx("T1b")), ("T2", idx("T2")), ("T3", idx("T3"))]}
        sel = A[gm["T1"]].sum(0) >= bc["min_active_minutes"]
        if sel.sum() >= 3:
            def corr(g):
                M = A[gm[g]][:, sel].astype(float)
                if len(M) < 30:
                    return None
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.nan_to_num(np.corrcoef(M.T))
            Cs, q, co = {g: corr(g) for g in gm}, int(sel.sum()), []
            for a, b in [("T1a", "T1b"), ("T1", "T3"), ("T1", "T2"), ("T3", "T2")]:
                if Cs.get(a) is not None and Cs.get(b) is not None:
                    co.append(dict(group_a=a, group_b=b, n_sensors=q,
                                   frobenius_per_pair=float(np.linalg.norm(Cs[a] - Cs[b]) / np.sqrt(q * (q - 1)))))
            res["coactivation"] = pd.DataFrame(co)
            res["coactivation"].to_csv(out_dir / f"behavior_coactivation__{tag}.csv", index=False)
    res["_artifacts"] = dict(Dm=Dm, dates=dates[di], sub=sub, drift=drift, hourly=hp, Z=np.nan_to_num(Z))
    return res


def summarize(results):
    rows = []
    for r in results:
        s = r["separation"].set_index("comparison")
        row = dict(variant=r["variant"], features=r["fset"], scope=r["scope"], n_feat=r["n_features_used"],
                   days_T1=r["days"]["T1"], days_T2=r["days"]["T2"], days_T3=r["days"]["T3"])
        for k in ["T1 vs T2", "T1 vs T3", "T2 vs T3", "T1a vs T1b"]:
            if k in s.index:
                tag = k.replace(" vs ", "_")
                row[f"ratio_{tag}"], row[f"p_{tag}"] = s.loc[k, "cross_over_within_T1"], s.loc[k, "perm_p"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- label validation
def label_alignment(prof, cls, misc_dir, cfg, out_dir):
    """Do the binary sensors listed per ActID fire inside logged windows more often than in random
    same-length daytime windows on the same days? Also scans a clock offset (time-zone check)."""
    logs = load_logs(misc_dir, cfg)
    lp = Path(misc_dir) / cfg["files"]["activity_labels"] if misc_dir else None
    if "R1" not in logs or lp is None or not lp.exists():
        return None
    labels = json.load(open(lp))
    t0, events = prof["t0"], prof["events"]
    r1 = logs["R1"].copy()
    r1["act"] = r1["ActID"].astype(float).round(1).astype(str)
    binc = cls[(cls.data_type == "binary") & ~cls.housekeeping & (cls.category != "constant")]
    cu_pid = binc[binc.source == "CU"].groupby("pid").groups
    bre_dev = binc[binc.source == "BRE"].groupby("device").groups
    h0, h1 = cfg["daytime"]
    rng = np.random.default_rng(cfg["behavior"]["seed"])
    ndraw = cfg["behavior"]["n_draw_random_windows"]

    def sensors_for(act):
        d = labels.get(act, {}).get("Databases", {})
        cols = []
        for pid in d.get("CU_CyPhy", []):
            cols += list(cu_pid.get(pid, []))
        for loc in d.get("BRE_CyPhy", []):
            cols += list(bre_dev.get(str(loc).split("_")[-1], []))
        return [c for c in dict.fromkeys(cols) if c in events and len(events[c])]

    dst = pd.Timestamp(cfg["dst_change"])
    out = []
    for off_h in cfg["behavior"]["offsets_hours"]:
        for act, g in r1.groupby("act"):
            cols = sensors_for(act)
            if not cols:
                continue
            ev = np.sort(np.concatenate([events[c] for c in cols]))
            s0 = (g["start"] - t0).dt.total_seconds().to_numpy() + off_h * 3600.0
            e0 = (g["end"] - t0).dt.total_seconds().to_numpy() + off_h * 3600.0
            dur = np.maximum(e0 - s0, 1.0)
            day_start = ((g["start"].dt.normalize() - t0).dt.total_seconds().to_numpy())
            for era, m in [("pre_dst", (g["start"] < dst).to_numpy()), ("post_dst", (g["start"] >= dst).to_numpy())]:
                if m.sum() < 3:
                    continue
                hit = np.mean((np.searchsorted(ev, e0[m]) - np.searchsorted(ev, s0[m])) > 0)
                base = []
                for _ in range(ndraw):
                    lo = day_start[m] + h0 * 3600
                    span = np.maximum((h1 - h0) * 3600 - dur[m], 1.0)
                    rs = lo + rng.random(m.sum()) * span
                    base.append(np.mean((np.searchsorted(ev, rs + dur[m]) - np.searchsorted(ev, rs)) > 0))
                b = float(np.mean(base))
                out.append(dict(offset_h=off_h, era=era, act_id=act, name=labels.get(act, {}).get("Name", ""),
                                n_windows=int(m.sum()), n_sensor_cols=len(cols), hit_rate=float(hit),
                                random_window_hit_rate=b, lift=(hit + 0.01) / (b + 0.01)))
    if not out:
        return None
    r = pd.DataFrame(out)
    r.to_csv(Path(out_dir) / "activity_label_validation.csv", index=False)
    summ = (r.groupby(["era", "offset_h"]).apply(lambda x: np.average(np.log(x.lift), weights=x.n_windows),
                                              include_groups=False).rename("weighted_mean_log_lift").reset_index())
    summ.to_csv(Path(out_dir) / "activity_clock_offset_check.csv", index=False)
    return summ
