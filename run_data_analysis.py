"""Phase 1 dataset suitability audit.

  python -m run_data_analysis --project_root_dir ROOT --dataset_folder DATA \
      [--config extra.yaml] [--set periods.T2='["2022-11-08","2022-11-11"]' grid_seconds=1] \
      [--stages convert,profile,behavior,viz,report] [--force]
"""
import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from src.config import load_config
from src.io_utils import find_misc_dir, prepare_store
from src.metadata import build_metadata
from src.sentinels import StateDecoder

log = logging.getLogger("da")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root_dir", required=True)
    ap.add_argument("--dataset_folder", required=True)
    ap.add_argument("--config", default=None, help="extra YAML merged over configs/config.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. grid_seconds=30")
    ap.add_argument("--stages", default=None, help="comma list; default from config")
    ap.add_argument("--force", action="store_true", help="redo conversion and profiling")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)

    cfg = load_config(a.project_root_dir, a.dataset_folder, a.config, a.set)
    if a.force:
        cfg["force_reconvert"] = True
    stages = a.stages.split(",") if a.stages else cfg["stages"]
    out = Path(cfg["paths"]["output_dir"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    cache = Path(cfg["paths"]["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    (out / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    t = time.time()

    misc = find_misc_dir(cfg)
    log.info("Misc dir: %s", misc)
    decoder = StateDecoder(Path(misc) / cfg["files"]["states_map"] if misc else None, cfg["state_class_overrides"])
    decoder.table().to_csv(out / "tables" / "sentinel_classification.csv", index=False)

    store = prepare_store(cfg) if ("convert" in stages or "profile" in stages) else None
    meta_path = cache / "metadata.pkl"
    prof_path = cache / "profile.pkl"

    if "profile" in stages:
        meta = build_metadata(store.columns, misc, cfg)
        meta.to_csv(out / "tables" / "sensor_metadata.csv", index=False)
        from src.profile import profile_columns
        log.info("Profiling %d columns", len(store.columns))
        prof = profile_columns(store, meta, decoder, cfg)
        with open(prof_path, "wb") as f:
            pickle.dump((prof, meta), f)
        if cfg["raw_audit"]["enabled"]:
            from src.raw_audit import audit_bre_raw, audit_cu_raw
            audit_cu_raw(cfg["dataset_folder"], cfg, out / "tables")
            audit_bre_raw(cfg["dataset_folder"], cfg, out / "tables")
    else:
        with open(prof_path, "rb") as f:
            prof, meta = pickle.load(f)

    from src.sensors import classify_sensors
    cls = classify_sensors(prof["profile"], meta, cfg)
    cls.to_csv(out / "tables" / "sensor_classification.csv", index=False)
    pd.DataFrame([prof["ts_report"]]).T.to_csv(out / "tables" / "A_timestamp_report.csv", header=["value"])
    prof["ts_gaps"].to_csv(out / "tables" / "A_timestamp_gaps.csv", index=False)
    prof["joint_gaps"].to_csv(out / "tables" / "A_joint_inactivity_gaps.csv", index=False)
    prof["joint_unavailable"].to_csv(out / "tables" / "A_joint_unavailable_runs.csv", index=False)
    log.info("Classification: %s", cls.category.value_counts().to_dict())

    results, align = [], None
    if "behavior" in stages:
        from src.behavior import label_alignment, run_behavior, summarize
        for variant in cfg["behavior"]["variants"]:
            for fset in cfg["behavior"]["feature_sets"]:
                for scope in cfg["behavior"]["scopes"]:
                    r = run_behavior(prof, cls, misc, cfg, variant, fset, scope, out / "tables")
                    log.info("Behaviour %s/%s/%s: %s", variant, fset, scope, "skipped" if r is None else "ok")
                    if r is not None:
                        results.append(r)
        if results:
            summarize(results).to_csv(out / "tables" / "behavior_summary.csv", index=False)
        try:
            align = label_alignment(prof, cls, misc, cfg, out / "tables")
        except Exception as e:
            log.exception("label alignment failed: %s", e)
        with open(cache / "behavior.pkl", "wb") as f:
            pickle.dump((results, align), f)
    elif (cache / "behavior.pkl").exists():
        with open(cache / "behavior.pkl", "rb") as f:
            results, align = pickle.load(f)

    if "viz" in stages:
        from src.viz import make_figures
        make_figures(prof, cls, results, cfg, out)
    if "report" in stages:
        from src.report import write_report
        write_report(prof, cls, decoder, results, align, cfg, out)
        log.info("Report: %s", out / "suitability_report.md")
    log.info("Done in %.0fs. Outputs: %s", time.time() - t, out)


if __name__ == "__main__":
    main()
