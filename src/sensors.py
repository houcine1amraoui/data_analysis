"""Stage B: qualitative sensor classification, decided on the T1 (training) window only."""
import numpy as np
import pandas as pd


def classify_sensors(prof, meta, cfg):
    cc = cfg["classification"]
    df = prof.copy()
    for c in ["pid", "device", "location", "floor", "kind", "network_guess", "is_phone", "confidence"]:
        df[c] = meta.loc[df.index, c]
    cats, reasons, decisions = [], [], []
    for _, r in df.iterrows():
        nb = r.nonbehavioral_share + r.temporal_proxy_share
        ref = "t1" if (pd.notna(r.get("t1_window_days")) and r.t1_window_days > 0) else "all"
        nun, unav = r[f"{ref}_n_unique"], r[f"{ref}_unavailable_share"]
        if nb >= cc["nonbehavioral_share"]:
            cat, why, dec = "non_behavioral", f"{nb:.0%} of time holds ids/addresses/date strings", "exclude"
        elif r["data_type"] == "empty" or (pd.notna(nun) and nun == 0):
            cat, why, dec = "unreliable", "no valid observations in reference window", "investigate/exclude"
        elif (pd.notna(unav) and unav > cc["unreliable_unavailable_share"]) or (
                pd.notna(r.invalid_share) and r.invalid_share > cc["invalid_share"]):
            cat, why, dec = "unreliable", f"unavailable={unav:.0%}, invalid={r.invalid_share}", "investigate"
        elif nun <= 1:
            cat, why, dec = "constant", "single value in reference window", "exclude (document)"
        elif (r["data_type"] in ("continuous", "discrete") and r[f"{ref}_top_share"] >= cc["nearly_constant_top_share"]) \
                or r[f"{ref}_changes_per_day"] < cc["nearly_constant_changes_per_day"]:
            cat, why, dec = "nearly_constant", (f"top_share={r[f'{ref}_top_share']:.4f}, "
                                                f"changes/day={r[f'{ref}_changes_per_day']:.2f}"), "investigate"
        else:
            cat, why, dec = "informative", "variation/events in reference window", "retain"
        cats.append(cat), reasons.append(why), decisions.append(dec)
    df["category"], df["reason"], df["suggested_decision"] = cats, reasons, decisions
    ok = df.category.isin(["informative"] + (["nearly_constant"] if cc["include_nearly_constant_in_candidates"] else []))
    # --- freeze propagation: binary sensors inherit the freeze of their device only when ALL of the device's
    # continuous channels are frozen (device dead). Continuous channels keep their own freeze time.
    key = df.source.astype(str) + "|" + df.pid.astype(str)
    key = np.where(df.source == "BRE", df.source + "|" + df.device.astype(str) + "|" + df.network_guess.astype(str), key)
    df["_gk"] = key
    cont = df.data_type.isin(cfg["freeze"]["types"])
    g = df[cont].groupby("_gk").agg(n=("freeze_start_sec", "size"), nf=("freeze_start_sec", "count"),
                                    t=("freeze_start_sec", "median"))
    dead = g[(g.n > 0) & (g.n == g.nf)]["t"]
    df["usable_until_sec"] = df.freeze_start_sec
    if cfg["freeze"]["propagate_to_device"]:
        inh = df["_gk"].map(dead)
        binary = df.data_type == "binary"
        df.loc[binary, "usable_until_sec"] = inh[binary]
    df = df.drop(columns="_gk")
    df["frozen_any"] = df.usable_until_sec.notna()
    df["behavioral_candidate"] = ok & ~df.housekeeping & (df["data_type"] != "categorical")
    df["has_temporal_proxy"] = df.temporal_proxy_share > 0
    return df


def summaries(cls):
    by = cls.groupby(["source", "category"]).size().unstack(fill_value=0)
    by["total"] = by.sum(axis=1)
    types = cls.groupby(["source", "data_type"]).size().unstack(fill_value=0)
    return by, types
