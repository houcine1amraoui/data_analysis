"""Synthetic stand-in for Phy_CU_BRE_Master.csv (same quirks: sentinel codes, BRE _x/_y, padding, outage week).
usage: python tests/make_synthetic_data.py OUT_DIR MISC_SRC_DIR"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out, misc_src = Path(sys.argv[1]), Path(sys.argv[2])
(out / "Master").mkdir(parents=True, exist_ok=True)
shutil.copytree(misc_src, out / "Misc", dirs_exist_ok=True)
rng = np.random.default_rng(1)
GRID = 30
ts = pd.date_range("2022-10-18 00:00:00", "2022-11-17 21:59:30", freq=f"{GRID}s")
n = len(ts)
sec = (ts - ts[0]).total_seconds().to_numpy()
day = ((sec) // 86400).astype(int)
hour = (sec % 86400) / 3600
r1 = pd.read_csv(out / "Misc/ActivityLogR1.csv", encoding="utf-8-sig")
r2 = pd.read_csv(out / "Misc/ActivityLogR2.csv", encoding="utf-8-sig")
d1 = set(pd.to_datetime(r1.StartTime).dt.normalize()); d2 = set(pd.to_datetime(r2.StartTime).dt.normalize())
dates = ts.normalize()
u1 = np.array([d in d1 for d in dates]); u2 = np.array([d in d2 for d in dates])
active = (hour >= 8) & (hour < 19) & (u1 | u2)


def binary_events(rate1, rate2):
    """Binary step signal toggling at Poisson rate (per active hour), user-dependent."""
    p = np.where(u2, rate2, rate1) * active * GRID / 3600
    tog = rng.random(n) < p
    return (np.cumsum(tog) % 2).astype(float)


cols = {}
# --- CU (entity-id names; sentinel codes) ---
for i, (nm, r1_, r2_) in enumerate([("binary_sensor.xiaomi_motion_sensor_kitchen_occupancy", 6, 2),
                                    ("binary_sensor.xiaomi_motion_sensor_workarea_occupancy", 3, 7),
                                    ("binary_sensor.smartthings_contact_sensor_pid_003_contact", 1.5, 1.5),
                                    ("binary_sensor.smartthings_contact_sensor_pid_006_contact", 1.0, 2.5),
                                    ("switch.kasa_020", 0.6, 0.6), ("switch.kasa_023", 0.5, 1.5),
                                    ("binary_sensor.aqara_motion_sensor_pid_075_occupancy", 4, 4),
                                    ("light.hue_light_bulb_pid_040", 0.4, 0.2),
                                    ("binary_sensor.aqara_vibration_sensor_pid_077_vibration", 5, 1),
                                    ("switch.kettle_start", 0.3, 0.9)]):
    s = binary_events(r1_, r2_)
    s[: 40] = -9.7                      # 'unknown' before first report
    if "kasa_023" in nm:
        s[: 2000] = -9.21               # 'unavailable' (offline) at start
    cols[nm] = s
cols["sensor.aqara_temperature_and_humidity_pid_071_temperature"] = (
    21 + 2 * np.sin(sec / 86400 * 2 * np.pi) + np.cumsum(rng.normal(0, 0.01, n))).round(1)
cols["sensor.mi_air_purifier_3_3h_pm2_5"] = np.maximum(0, 8 + np.cumsum(rng.normal(0, 0.3, n)) % 30).round(0)
st = np.full(n, -9.27); st[(hour > 10) & (hour < 12) & u1] = -9.5      # idle / cleaning
cols["vacuum.roborock_s5_max"] = st
cols["switch.kettle_led_constant"] = np.ones(n)                          # constant
cols["light.philips_hue_lightstrip_pid_146"] = np.full(n, np.nan)        # all NaN
# phone (PID 125): address codes, BST/GMT names, a date string only on Nov 10, plus some real signal
ph = np.full(n, -9.55); cols["sensor.pixel_5_geocoded_location"] = ph
tz = np.where(sec < (pd.Timestamp("2022-10-30 02:00") - ts[0]).total_seconds(), -9.37, -9.75)
cols["sensor.pixel_5_current_time_zone"] = tz
dstr = np.full(n, -9.7); dstr[(dates == pd.Timestamp("2022-11-10"))] = -9.39
cols["sensor.pixel_5_next_alarm"] = dstr
cols["binary_sensor.pixel_5_interactive"] = binary_events(8, 1)
cols["sensor.pixel_5_battery_level"] = (100 - (sec / 3600 % 20) * 4).round(0)
# --- BRE (location codes, _x/_y collisions, outage week flat) ---
outage = (sec >= 6 * 86400) & (sec < 13 * 86400)
def bre(sig):
    s = sig.copy(); first = np.argmax(outage); s[outage] = s[first - 1]; return s
for nm, r in [("OC-00-H1_contact_x", (2, 3)), ("OC-00-H1_contact_y", (2, 3)), ("A6-00-K1_motion_x", (5, 1)),
              ("A6-00-K1_motion_y", (5, 1)), ("A6-01-B1_motion_x", (0.5, 4)), ("OC-01-HB_contact", (1, 1))]:
    cols[nm] = bre(binary_events(*r))
for nm in ["A6-00-K1_temperature_x", "A6-01-B1_temperature_x", "A6-00-L1_humidity"]:
    base = 20 + np.cumsum(rng.normal(0, 0.02, n))
    cols[nm] = bre((base * 10).round() / 10) if "humidity" not in nm else bre(np.clip(base + 25, 0, 100).round(1))
cols["OC-02-S1_tamper"] = np.zeros(n)
# --- Physical (padding before first real reading) ---
pad = sec < (pd.Timestamp("2022-10-19 13:27:15") - ts[0]).total_seconds()
for did in (1, 3):
    t = 21 + np.cumsum(rng.normal(0, 0.01, n)); t[pad] = t[~pad][0]
    cols[f"Temp_{did}"] = t.round(2)
    h = 45 + np.cumsum(rng.normal(0, 0.05, n)); h[pad] = h[~pad][0]; cols[f"Humidity_{did}"] = np.clip(h, 0, 100).round(2)
    l = np.where(active, 300 + 50 * rng.standard_normal(n), 2).round(1); l[pad] = l[~pad][0]; cols[f"Lux_{did}"] = l
    cols[f"MassPM2.5_{did}"] = np.maximum(0, 5 + 3 * rng.standard_normal(n) * active).round(2)
cols["Temp_2"] = np.full(n, 22.0)    # stuck sensor (constant)
df = pd.DataFrame(cols); df.insert(0, "Timestamp", ts)
df.to_csv(out / "Master" / "Phy_CU_BRE_Master.csv", index=False)
print("wrote", df.shape, "grid", GRID, "s")
