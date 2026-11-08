"""Config loading: defaults.yaml <- config.yaml <- --config file <- --set key=value."""
import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def deep_update(base, new):
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _load(p):
    p = Path(p)
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def parse_set(items):
    """['a.b=3', 'periods.T2=["2022-11-08","2022-11-11"]'] -> nested dict (values parsed as YAML)."""
    out = {}
    for it in items or []:
        key, _, val = it.partition("=")
        cur = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return out


def load_config(project_root_dir, dataset_folder, extra_config=None, sets=None):
    cfg = _load(REPO / "configs" / "defaults.yaml")
    deep_update(cfg, _load(REPO / "configs" / "config.yaml"))
    if extra_config:
        deep_update(cfg, _load(extra_config))
    deep_update(cfg, parse_set(sets))
    root = Path(project_root_dir)
    cfg["project_root_dir"] = str(root)
    cfg["dataset_folder"] = str(dataset_folder)
    p = cfg["paths"]
    p["output_dir"] = p.get("output_dir") or str(root / "outputs" / cfg["run_name"])
    p["cache_dir"] = p.get("cache_dir") or str(root / "cache" / "data_analysis")
    return cfg
