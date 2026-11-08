"""Decode the -9.x numeric codes that replaced string states in the CU dataset (CUStatesMap.json)."""
import json
import re

import numpy as np
import pandas as pd

MISSING = {"unknown", "unavailable", "none", "never"}
TEMPORAL_NAMES = {"greenwich mean time", "british summer time"}
RE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")
RE_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
RE_MAC = re.compile(r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$", re.I)
RE_VER = re.compile(r"^\d{4}\.\d+\.\d+")
RE_ADDR = re.compile(r",\s*(UK|United Kingdom)$|[A-Z]{1,2}\d{1,2}[A-Z]? \d[A-Z]{2}")


def classify_state(s):
    """-> 'missing' | 'temporal' | 'nonbehavioral' | 'state'."""
    low = s.strip().lower()
    if low in MISSING:
        return "missing"
    if RE_ISO.match(s.strip()) or low in TEMPORAL_NAMES:
        return "temporal"   # date strings / clock-change names: act as time proxies
    if RE_IP.match(s) or RE_MAC.match(s) or RE_VER.match(s) or RE_ADDR.search(s) or low == "shtb":
        return "nonbehavioral"
    return "state"


class StateDecoder:
    def __init__(self, states_map_path=None, overrides=None):
        self.code2class, self.rows = {}, []
        if states_map_path:
            # the shipped file contains an invalid JSON escape (\\' in "Queen\'s Rd"); repair before parsing
            raw = open(states_map_path, encoding="utf-8-sig").read().replace("\\'", "'")
            for s, c in json.loads(raw).items():
                cl = (overrides or {}).get(s) or classify_state(s)
                self.code2class[round(float(c), 2)] = cl
                self.rows.append({"state": s, "code": c, "class": cl})

    def table(self):
        return pd.DataFrame(self.rows).sort_values("code", ascending=False) if self.rows else pd.DataFrame()

    def classify_values(self, rv):
        """Vector of classes: 'nan' | 'valid' | 'missing' | 'state' | 'nonbehavioral' | 'temporal'."""
        out = np.where(np.isnan(rv), "nan", "valid").astype(object)
        if not self.code2class:
            return out
        r = np.round(rv.astype(np.float64), 2)
        for i in np.flatnonzero((r <= -9.09) & (r >= -9.9)):
            cl = self.code2class.get(float(r[i]))
            if cl:
                out[i] = cl
        return out
