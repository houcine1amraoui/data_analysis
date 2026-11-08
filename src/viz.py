"""Stage D: static figures (matplotlib)."""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

log = logging.getLogger("da")
SRC_ORDER = ["CU", "BRE", "PHY", "OTHER"]


def _save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _order(cls):
    return cls.assign(_o=cls.source.map({s: i for i, s in enumerate(SRC_ORDER)})).sort_values(["_o", "column"]).index


def _period_lines(ax, cfg, t0, unit_s):
    for k, (a, b) in cfg["periods"].items():
        for x in (a, b):
            ax.axvline((pd.Timestamp(x) - t0).total_seconds() / unit_s, color="k", lw=0.6, ls="--")


def fig_missing_matrix(prof, cls, cfg, out, dpi):
    cols = _order(cls)
    m = prof["minute"][cols].isna()
    h = m.groupby(m.index.floor("h")).mean().T
    fig, ax = plt.subplots(figsize=(14, max(4, 0.045 * len(cols) + 2)))
    im = ax.imshow(h.values, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Unavailable/missing share per hour (black = no valid data)")
    ax.set_xlabel("hours since start"), ax.set_ylabel("sensors (grouped by source)")
    _period_lines(ax, cfg, prof["t0"].floor("h"), 3600)
    fig.colorbar(im, ax=ax, fraction=0.02)
    _save(fig, out / "D1_missing_matrix.png", dpi)


def fig_activity_heatmap(prof, cls, cfg, out, dpi):
    cols = [c for c in _order(cls) if c in prof["hourly"].index]
    H = np.log1p(prof["hourly"].loc[cols].values.astype(float))
    fig, ax = plt.subplots(figsize=(14, max(4, 0.045 * len(cols) + 2)))
    im = ax.imshow(H, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title("Sensor activity heatmap: log(1 + state changes per hour)")
    ax.set_xlabel("hours since start"), ax.set_ylabel("sensors (grouped by source)")
    _period_lines(ax, cfg, prof["t0"].floor("h"), 3600)
    fig.colorbar(im, ax=ax, fraction=0.02)
    _save(fig, out / "D2_sensor_activity_heatmap.png", dpi)


def fig_daily_activity(prof, cls, cfg, out, dpi):
    h = prof["hourly"]
    d = h.T.groupby(h.columns.floor("D")).sum().T
    fig, ax = plt.subplots(figsize=(13, 4))
    for s in SRC_ORDER:
        cols = [c for c in cls.index[cls.source == s] if c in d.index]
        if cols:
            ax.plot(d.columns, d.loc[cols].sum(), marker="o", ms=3, label=f"{s} ({len(cols)} cols)")
    for k, (a, b) in cfg["periods"].items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=0.07, label=k)
    ax.set_yscale("log"), ax.set_title("State changes per day by source (flat/zero stretches = outage or padding)")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, out / "D2b_daily_activity_by_source.png", dpi)


def _pick(cls, n):
    """Round-robin over sources so noisy physical channels do not crowd everything else out."""
    cand = cls[cls.behavioral_candidate].sort_values("t1_changes_per_day", ascending=False)
    groups = [list(cand.index[cand.source == s]) for s in SRC_ORDER]
    out = []
    while len(out) < n and any(groups):
        for g in groups:
            if g and len(out) < n:
                out.append(g.pop(0))
    return out


def fig_freeze_map(prof, cls, cfg, out, dpi):
    cols = [c for c in _order(cls)]
    end_h = prof["end_sec"] / 3600
    fig, ax = plt.subplots(figsize=(14, max(4, 0.045 * len(cols) + 2)))
    for i, c in enumerate(cols):
        u = cls.loc[c, "usable_until_sec"]
        if pd.notna(u):
            ax.barh(i, end_h - u / 3600, left=u / 3600, color="firebrick", height=1.0)
    ax.set_ylim(len(cols), -1), ax.set_xlim(0, end_h)
    ax.set_title("Frozen tails: red = sensor (or its device) no longer changes, values are forward-filled")
    ax.set_xlabel("hours since start"), ax.set_ylabel("sensors (grouped by source)")
    _period_lines(ax, cfg, prof["t0"].floor("h"), 3600)
    _save(fig, out / "D1b_frozen_tails.png", dpi)


def _slice(df, a, b):
    return df[(df.index >= pd.Timestamp(a)) & (df.index < pd.Timestamp(b))]


def fig_distributions(prof, cls, cfg, out, dpi):
    cols = _pick(cls, cfg["viz"]["n_dist_plots"])
    if not cols:
        return
    n = len(cols)
    nc = 4
    fig, axs = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(16, 2.6 * np.ceil(n / nc)), squeeze=False)
    for ax, c in zip(axs.ravel(), cols):
        for k, (a, b) in cfg["periods"].items():
            v = _slice(prof["minute"][[c]], a, b)[c].dropna()
            if len(v) > 1:
                ax.hist(v, bins=30, density=True, histtype="step", label=k)
        ax.set_title(c[-38:], fontsize=7), ax.tick_params(labelsize=6)
    axs.ravel()[0].legend(fontsize=6)
    fig.suptitle("Per-minute value distributions by period (top T1-active candidate sensors)")
    _save(fig, out / "D3_distributions_by_period.png", dpi)


def fig_corr(prof, cls, cfg, out, dpi):
    cols = _pick(cls, cfg["viz"]["max_corr_features"])
    if len(cols) < 3:
        return
    M = prof["minute"][cols]
    ref = _slice(M, *cfg["periods"]["T1"]).corr(min_periods=30).fillna(0)
    order = leaves_list(linkage(1 - ref.abs().values[np.triu_indices(len(cols), 1)], "average"))
    cols = [cols[i] for i in order]
    pairs = []
    for name, rng in [("ALL", None)] + list(cfg["periods"].items()):
        sub = M[cols] if rng is None else _slice(M[cols], *rng)
        C = sub.corr(min_periods=30)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(C.fillna(0).values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"Correlation of per-minute values ({name}); order fixed from T1 clustering")
        fig.colorbar(im, ax=ax, fraction=0.04)
        _save(fig, out / f"D4_correlation_{name}.png", dpi)
        if name == "T1":
            iu = np.triu_indices(len(cols), 1)
            df = pd.DataFrame({"a": np.array(cols)[iu[0]], "b": np.array(cols)[iu[1]], "r": C.values[iu]})
            df[df.r.abs() >= 0.95].sort_values("r", key=abs, ascending=False).to_csv(
                out.parent / "tables" / "redundant_pairs_T1_abs_r_ge_0.95.csv", index=False)


def fig_hourly_profiles(res, out, dpi):
    a = res.get("_artifacts")
    if not a or a.get("hourly") is None:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for g in a["hourly"].columns:
        ax.plot(a["hourly"].index, a["hourly"][g] / max(a["hourly"][g].sum(), 1), marker="o", ms=3, label=g)
    ax.set_xlabel("hour of day"), ax.set_ylabel("share of state changes"), ax.legend()
    ax.set_title(f"Hourly event profile ({res['variant']} / {res['fset']} / {res['scope']})")
    _save(fig, out / f"D5_hourly_profile__{res['variant']}__{res['fset']}__{res['scope']}.png", dpi)


def fig_behavior(res, out, dpi):
    a = res.get("_artifacts")
    if not a:
        return
    tag = f"{res['variant']}__{res['fset']}__{res['scope']}"
    order = np.argsort(a["dates"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    im = axs[0].imshow(a["Dm"][np.ix_(order, order)], cmap="magma_r")
    axs[0].set_xticks(range(len(order)))
    axs[0].set_xticklabels([f"{a['sub'][i]} {pd.Timestamp(a['dates'][i]).strftime('%m-%d')}" for i in order],
                           rotation=90, fontsize=6)
    axs[0].set_yticks(range(len(order)))
    axs[0].set_yticklabels([f"{a['sub'][i]}" for i in order], fontsize=6)
    axs[0].set_title("Day-to-day distance (z-scored log counts)")
    fig.colorbar(im, ax=axs[0], fraction=0.04)
    U, S, _ = np.linalg.svd(a["Z"] - a["Z"].mean(0), full_matrices=False)
    P = U[:, :2] * S[:2]
    for g in sorted(set(a["sub"])):
        m = a["sub"] == g
        axs[1].scatter(P[m, 0], P[m, 1], label=g, s=40)
    axs[1].legend(), axs[1].set_title("PCA of daily behaviour vectors")
    d = a["drift"]
    for g in sorted(set(d.group)):
        m = d.group == g
        axs[2].scatter(d.date[m], d.dist_to_T1_centroid[m], label=g)
    axs[2].legend(), axs[2].set_title("Distance to T1 centroid (drift)"), axs[2].tick_params(axis="x", rotation=45)
    _save(fig, out / f"D6_behavior__{tag}.png", dpi)


def make_figures(prof, cls, behavior_results, cfg, out_dir):
    out = Path(out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "tables").mkdir(exist_ok=True)
    dpi = cfg["viz"]["dpi"]
    for fn in (fig_missing_matrix, fig_freeze_map, fig_activity_heatmap, fig_daily_activity, fig_distributions, fig_corr):
        try:
            fn(prof, cls, cfg, out, dpi)
        except Exception as e:  # keep going: one broken figure must not kill the run
            log.exception("figure %s failed: %s", fn.__name__, e)
    for r in behavior_results:
        if r["scope"] != "logged_days" and r["variant"] not in ("all_sensors",):
            continue   # keep the number of figures manageable
        for fn in (fig_hourly_profiles, fig_behavior):
            try:
                fn(r, out, dpi)
            except Exception as e:
                log.exception("figure %s failed: %s", fn.__name__, e)
