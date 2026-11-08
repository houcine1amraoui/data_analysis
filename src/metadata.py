"""Infer a sensor metadata table (source, device, room, floor, kind) from column names + Misc files."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# PID -> (device, location). From Table 2 of the paper. 147 is listed in Table 11 only.
PID_INFO = {
    4: ("SmartThings Contact", "Oven"), 3: ("SmartThings Contact", "Fridge"), 6: ("SmartThings Contact", "Entrance"),
    138: ("Xiaomi Motion", "Kitchen"), 139: ("Xiaomi Motion", "Workarea"), 136: ("Xiaomi Contact", "Bedroom"),
    81: ("Aqara Switch", "NA"), 82: ("Aqara Switch", "NA"), 77: ("Aqara Vibration", "Kitchen"),
    72: ("Aqara TH", "Bedroom"), 74: ("Aqara TH", "Workarea"), 71: ("Aqara TH", "Kitchen"), 73: ("Aqara TH", "Washroom"),
    79: ("Aqara Vibration", "Bedroom"), 78: ("Aqara Vibration", "Stairs"), 75: ("Aqara Motion", "Entrance"),
    40: ("Hue Bulb", "Kids Room"), 145: ("Hue LightStrip", "Workarea"), 146: ("Hue LightStrip", "Stairs"),
    147: ("Hue LightStrip", "Unknown"), 62: ("Hue Motion", "Bedroom"),   # paper text says Workarea (inconsistent)
    83: ("Aqara Water Leak", "Kitchen"), 84: ("Aqara Water Leak", "Washroom"), 114: ("Coffee Machine", "Kitchen"),
    109: ("Kettle", "Kitchen"), 107: ("Humidifier", "Kitchen"), 113: ("Mill Radiator", "Workarea"),
    -1: ("Samsung TV", "Workarea"), 108: ("Air Purifier", "Workarea"), 143: ("IP Camera", "Kitchen"),
    115: ("Projector", "Study Room"), 125: ("Pixel 5 phone", "Mobile"), 94: ("Smart Scale", "Washroom"),
    20: ("Kasa Plug", "Workarea"), 21: ("Kasa Plug", "Workarea"), 22: ("Kasa Plug", "Workarea"),
    91: ("Roborock", "Entrance"), 24: ("Kasa Plug", "Entrance"), 53: ("Lumiman Bulb", "Hallway"),
    13: ("Echo Dot", "Kitchen"), 23: ("Kasa Plug", "Kitchen"), 18: ("Kasa Plug", "Kitchen"),
    19: ("Kasa Plug", "Kitchen"), 17: ("Kasa Plug", "Bedroom"), 63: ("Google Speaker", "Study Room"),
}
ROOM_FLOOR = {"Kitchen": "GF", "Workarea": "GF", "Entrance": "GF", "Oven": "GF", "Fridge": "GF",
              "Living Room": "GF", "Stairs": "GF/FF", "Bedroom": "FF", "Washroom": "FF", "Bathroom": "FF",
              "Kids Room": "FF", "Study Room": "FF", "Hallway": "FF", "Solar Loft": "SF"}
RPI_ROOM = {1: "Entrance", 2: "Workarea", 3: "Kitchen", 4: "Bathroom", 5: "Bedroom"}
BRE_AREA = {"K": "Kitchen", "B": "Bedroom", "H": "Hallway", "L": "Living Room", "S": "Solar Loft", "W": "Water Closet"}
FLOOR_CODE = {"00": "GF", "01": "FF", "02": "SF"}

RE_CU = re.compile(r"^(binary_sensor|sensor|switch|light|climate|vacuum|fan|number|select|humidifier|"
                   r"media_player|device_tracker|update|input_\w+|automation|weather|person|camera)\.")
RE_BRE = re.compile(r"^([A-Z0-9]{2})-(\d{2})-([A-Z0-9]+)_(.+)$")
KINDS = ["battery_low", "mass concentration", "number concentration", "particle size", "ultraviolet", "battery", "power_outage", "update_available", "temperature", "temp", "humidity", "illuminance",
         "lux", "pressure", "noise", "masspm", "numpm", "tpm", "prox", "motion", "occupancy", "contact",
         "vibration", "tamper", "moving", "leak", "voltage", "consumption", "current", "light", "switch",
         "media_player", "climate", "vacuum", "fan", "humidifier", "kettle", "tracker", "wifi", "volume",
         "pm2_5", "click", "action", "angle", "strength"]


def kind_of(name):
    low = name.lower()
    for k in KINDS:
        if k in low:
            return k
    return "other"


def _reverse_pid_map(pid_map_path):
    rev = {}
    if pid_map_path and Path(pid_map_path).exists():
        for pid, d in json.load(open(pid_map_path)).items():
            for ent in d.get("entityids", {}):
                rev[ent.strip()] = int(pid)
    return rev


def build_metadata(columns, misc_dir, cfg):
    pid_map = (Path(misc_dir) / cfg["files"]["pid_map"]) if misc_dir else None
    rev = _reverse_pid_map(pid_map)
    rows = []
    for col in columns:
        r = dict(column=col, source="PHY", pid=np.nan, device="", location="", floor="", kind=kind_of(col),
                 network_guess="", is_phone=False, confidence="guessed")
        m = RE_BRE.match(col)
        if RE_CU.match(col):
            r["source"] = "CU"
            pid = rev.get(col.strip())
            if pid is None:
                mm = re.search(r"pid_0*(\d+)|kasa_0*(\d+)", col)
                if mm:
                    pid = int(mm.group(1) or mm.group(2))
                elif "pixel_5" in col:
                    pid = 125
            if pid is not None:
                r["pid"] = pid
                dev, loc = PID_INFO.get(pid, ("", ""))
                r.update(device=dev, location=loc, floor=ROOM_FLOOR.get(loc, ""), is_phone=(pid == 125),
                         confidence="parsed")
        elif m:
            code, fl, area, rest = m.groups()
            net = "ZB?" if rest.endswith("_x") else "WB?" if rest.endswith("_y") else ""
            rest = re.sub(r"_[xy]$", "", rest)
            r.update(source="BRE", device=f"{code}-{fl}-{area}", location=BRE_AREA.get(area[0], area),
                     floor=FLOOR_CODE.get(fl, ""), kind=kind_of(rest), network_guess=net, confidence="parsed")
        else:
            r["source"] = "PHY"
            mm = re.match(r"^Phy_(\d)_(.+)$", col) or re.search(r"(?:rpi|dev(?:id)?)[_\-]?(\d)", col.lower()) \
                or re.search(r"_(\d)$", col) or re.search(r"^(\d)_", col)
            if mm and int(mm.group(1)) in RPI_ROOM:
                did = int(mm.group(1))
                r.update(pid=did, device=f"RPi-{did}", location=RPI_ROOM[did],
                         floor=ROOM_FLOOR.get(RPI_ROOM[did], ""), confidence="parsed")
        rows.append(r)
    meta = pd.DataFrame(rows).set_index("column", drop=False).rename_axis(None)
    ov = (Path(misc_dir) / cfg["files"]["metadata_overrides"]) if misc_dir else None
    if ov is not None and ov.exists():
        o = pd.read_csv(ov).set_index("column")
        for c in o.columns:
            meta.loc[o.index.intersection(meta.index), c] = o.loc[o.index.intersection(meta.index), c]
    return meta
