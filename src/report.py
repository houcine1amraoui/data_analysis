"""Assemble the Phase 1 suitability report (Markdown). Reports evidence; does not issue a verdict."""
import numpy as np
import pandas as pd

from .behavior import summarize
from .sensors import summaries


def md(df, floatfmt=".3f", max_rows=40):
    if df is None or len(df) == 0:
        return "\n_none_\n"
    try:
        return "\n" + df.head(max_rows).to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except Exception:
        return "\n```\n" + df.head(max_rows).to_string(index=False) + "\n```\n"


def write_report(prof, cls, decoder, behavior, align, cfg, out_dir):
    rep = prof["ts_report"]
    by, types = summaries(cls)
    t0 = prof["t0"]
    L = [f"# Dataset Suitability Report (Phase 1) - run `{cfg['run_name']}`\n",
         "_Auto-generated. Every number is exploratory evidence, not a verdict; thresholds are provisional._\n"]
    L.append("## 1. Dataset overview\n")
    L.append(md(pd.DataFrame([rep]).T.reset_index().rename(columns={"index": "item", 0: "value"})))
    L.append("Columns per source and inferred data type:\n" + md(types.reset_index()))
    L.append("## 2. Data quality (master file is inflated to the grid; see caveats)\n")
    L.append("- Missing/duplicate/interval checks on an inflated master are trivially clean; real event rates need the RAW audit.\n"
             "- 'unavailable' = NaN + missing markers (unknown/unavailable/None/never) + id/date strings.\n")
    q = cls[["source", "nan_share", "missing_marker_share", "nonbehavioral_share", "temporal_proxy_share",
             "invalid_share", "valid_range_share"]].groupby("source").mean().reset_index()
    L.append("Mean shares of time per source:\n" + md(q))
    L.append("### 2a. Joint unavailability (>= 80% of a source's columns unknown/NaN)\n" + md(prof["joint_unavailable"]))
    L.append("### 2b. Joint inactivity (no change in ANY column of a source)\n" + md(
        prof["joint_gaps"].sort_values("hours", ascending=False)))
    fr = cls[cls.frozen_any]
    L.append("### 2c. Frozen sensors (stop changing, then forward-filled: look valid but are stale)\n")
    L.append(f"**{len(fr)} of {len(cls)} columns** are frozen until the end of the recording (flagged from continuous "
             "channels and propagated to the binary sensors of the same device).\n")
    if len(fr):
        sm = fr.assign(freeze_at=[t0 + pd.Timedelta(seconds=float(u)) for u in fr.usable_until_sec]).groupby(
            "source").agg(n_columns=("column", "size"), first_freeze=("freeze_at", "min"),
                          last_freeze=("freeze_at", "max")).reset_index()
        L.append(md(sm))
        L.append("Per-column list: `tables/sensor_classification.csv` (`usable_until_sec`, `frozen_any`). "
                 "Figure: `D1b_frozen_tails.png`.\n")
    tp = cls[cls.has_temporal_proxy]
    L.append(f"Columns containing date/clock-change strings (temporal proxies): **{len(tp)}**. "
             "Sentinel classes: `tables/sentinel_classification.csv` (review).\n")
    L.append("## 3. Sensor analysis (decided on the T1 window only)\n" + md(by.reset_index()))
    L.append(f"Behavioural candidates: **{int(cls.behavioral_candidate.sum())}** of {len(cls)} columns "
             "(informative, not housekeeping, not categorical). Full table: `tables/sensor_classification.csv`.\n")
    L.append("## 4. Behavioural analysis\n")
    L.append("Day-level feature vectors, standardised on T1 days, frozen sensors masked per day. "
             "`ratio_X_Y` = mean distance between groups X and Y divided by the mean within-T1 distance "
             "(about 1 = indistinguishable; permutation p in `p_X_Y`). `events` = binary change counts, "
             "`levels` = daytime mean/std of continuous channels.\n")
    if behavior:
        sm = summarize(behavior)
        L.append(md(sm, floatfmt=".2f", max_rows=80))
        L.append("**How to read this.** If T1-vs-T2 and T1-vs-T3 ratios are similar, differences reflect "
                 "time/environment, not the user. User-specific signal requires T1-vs-T2 clearly above "
                 "T1-vs-T3 and T2-vs-T3 clearly above 1, in the `events` features, with T1a-vs-T1b near 1.\n")
        for r in behavior:
            if r["variant"] == "all_sensors" and r["scope"] == "logged_days":
                L.append(f"### detail: `{r['variant']}` / `{r['fset']}` / `{r['scope']}` "
                         f"({r['n_features_used']} features, days {r['days']})\n")
                L.append(md(r["separation"]))
                if "coactivation" in r:
                    L.append("Co-activation structure distance (T1a vs T1b = within-user baseline):\n" + md(r["coactivation"]))
    L.append("**Caveats.** User identity is confounded with time; T2 is a few days; results are exploratory.\n")
    if align is not None:
        L.append("### Log / clock alignment check (weighted mean log-lift of listed binary sensors inside logged windows "
                 "vs random same-length daytime windows; higher = better aligned)\n")
        L.append(md(align))
    L.append("## 5. Preliminary modelling\n_Not part of this run (Stage E)._\n")
    L.append("## 6. Suitability conclusion (to be written by you)\n"
             "Map the evidence to: Continue / Continue with restrictions / Change research scope / Consider another dataset.\n")
    (out_dir / "suitability_report.md").write_text("\n".join(L))
