"""Assemble the Phase 1 suitability report (Markdown). Reports evidence; does not issue a verdict."""
import json

import numpy as np
import pandas as pd

from .sensors import summaries


def md(df, floatfmt=".3f", max_rows=40):
    if df is None or len(df) == 0:
        return "_none_\n"
    df = df.head(max_rows)
    try:
        return "\n" + df.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except Exception:
        return "\n```\n" + df.to_string(index=False) + "\n```\n"


def write_report(prof, cls, decoder, behavior, align, cfg, out_dir):
    rep = prof["ts_report"]
    by, types = summaries(cls)
    L = []
    L.append(f"# Dataset Suitability Report (Phase 1) - run `{cfg['run_name']}`\n")
    L.append("_Auto-generated. Every number is exploratory evidence, not a verdict; thresholds are provisional._\n")
    L.append("## 1. Dataset overview\n")
    L.append(md(pd.DataFrame([rep]).T.reset_index().rename(columns={"index": "item", 0: "value"})))
    L.append("Columns per source and inferred data type:\n")
    L.append(md(types.reset_index()))
    L.append("## 2. Data quality (Master file; inflated to the grid, so see caveats)\n")
    L.append("- Missing/duplicate/interval checks on an inflated master are trivially clean; real event rates need the RAW audit.\n"
             "- 'unavailable' = NaN + missing markers (unknown/unavailable/None/never) + id/date strings.\n")
    q = cls[["source", "nan_share", "missing_marker_share", "nonbehavioral_share", "temporal_proxy_share",
             "invalid_share", "valid_range_share"]].groupby("source").mean().reset_index()
    L.append("Mean shares of time per source:\n" + md(q))
    L.append("Largest joint-inactivity stretches (no change in ANY column of a source; outage/padding candidates):\n")
    L.append(md(prof["joint_gaps"].sort_values("hours", ascending=False)))
    L.append("Sentinel code classification is in `tables/sentinel_classification.csv` (review it).\n")
    tp = cls[cls.has_temporal_proxy]
    L.append(f"Columns containing date/clock-change strings (temporal proxies): **{len(tp)}**\n")
    L.append("## 3. Sensor analysis (decided on the T1 window only)\n")
    L.append(md(by.reset_index()))
    L.append(f"Behavioural candidates: **{int(cls.behavioral_candidate.sum())}** of {len(cls)} columns "
             f"(excluding housekeeping, categorical, phone-only variants handled in Stage C).\n")
    L.append("Full table: `tables/sensor_classification.csv`.\n")
    L.append("## 4. Behavioural analysis\n")
    for r in behavior:
        L.append(f"### variant `{r['variant']}` / scope `{r['scope']}` "
                 f"({r['n_candidates']} sensors; days {r.get('days_per_group')})\n")
        if "separation" in r:
            L.append("Separation (cross-group distance / within-T1 distance; permutation p; leave-one-day-out accuracy):\n")
            L.append(md(r["separation"]))
        if "frac_abs_d_gt_0.8" in r:
            L.append(f"Share of sensors with |Cohen d| > 0.8 vs T1: {r['frac_abs_d_gt_0.8']}\n")
        if "hourly_js" in r:
            L.append(f"Hourly-profile JS distances: {r['hourly_js']}\n")
        if "coactivation" in r:
            L.append("Co-activation structure distance (T1a vs T1b is the within-user baseline):\n")
            L.append(md(r["coactivation"]))
    L.append("**Caveats.** User identity is confounded with time; T2 is a handful of days; "
             "Actor 2 often carried no phone (phone ablation = `no_phone`); results are exploratory.\n")
    if align is not None:
        L.append("### Log / clock alignment check (higher = stronger sensor response inside logged windows)\n")
        L.append(md(align))
        L.append("If the best offset is not 0 h (esp. before the 30 Oct clock change), logs and sensor "
                 "timestamps may differ by a time-zone offset.\n")
    L.append("## 5. Preliminary modelling\n_Not part of this run (Stage E: Persistence/VAR/LSTM/LSTM-AE/GDN)._\n")
    L.append("## 6. Suitability conclusion (to be written by you)\n")
    L.append("Map the evidence above to: Continue / Continue with restrictions / Change research scope / "
             "Consider another dataset. Document what the dataset cannot establish.\n")
    (out_dir / "suitability_report.md").write_text("\n".join(L))
