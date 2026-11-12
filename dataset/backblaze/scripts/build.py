"""Build a landmark table from the consolidated Backblaze data.

For each landmark date L: drives alive on L, their SMART values on L, the 7-day
change of the error counters, and labels y_t (failure in (L, L+t]) for each
horizon t. y_t is NaN when the drive left the data before L+t without failing.
Usage: python -I build.py <data dir with panel/ and drives.parquet> <output parquet>
   ex: python -I build.py data out/landmarks.parquet
"""
import glob
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

data_dir, out_path = sys.argv[1], sys.argv[2]
panel = ds.dataset(sorted(glob.glob(os.path.join(data_dir, "panel", "*.parquet"))), format="parquet")
drives = pd.read_parquet(os.path.join(data_dir, "drives.parquet"))
days = sorted(d.date() for d in pd.to_datetime(
    panel.to_table(columns=["date"]).column("date").unique().to_pylist()))
day_set = set(days)
print("days", len(days), days[0], "->", days[-1], flush=True)

ERR = [5, 187, 188, 197, 198]  # counters Backblaze uses to judge failure
OTHER = [1, 7, 9, 10, 12, 183, 184, 189, 190, 192, 193, 194, 199, 240, 241, 242]
SMART = [f"smart_{i}_raw" for i in ERR + OTHER]
HORIZONS = [7, 30, 60]


def read(d, cols):
    have = [c for c in cols if c in panel.schema.names]  # dropped near-empty SMART -> NaN
    t = panel.to_table(columns=have, filter=ds.field("date") == d).to_pandas()
    for c in cols:
        if c not in t:
            t[c] = np.nan
    t = t[cols]
    if "model" in t:
        t["model"] = t["model"].astype(str)
    return t


last_seen = dict(zip(drives.serial_number, drives.last_seen.dt.date))
fail_day = dict(zip(drives.serial_number[drives.failed == 1], drives.fail_date[drives.failed == 1].dt.date))
print("drives", len(last_seen), "failures", len(fail_day), flush=True)
end = days[-1]

# Pass 2: landmark snapshots.
landmarks = [d for d in days if d.weekday() == days[0].weekday()]
landmarks = [d for d in landmarks if d - timedelta(7) in day_set and d + timedelta(max(HORIZONS)) <= end]
print("landmarks", landmarks, flush=True)
base_cols = ["serial_number", "model", "capacity_bytes", "failure"] + SMART
rows = []
for L in landmarks:
    now = read(L, base_cols)
    prev = read(L - timedelta(7), ["serial_number"] + [f"smart_{i}_raw" for i in ERR])
    now = now[now.failure == 0].drop(columns="failure")
    now = now[~now.serial_number.map(lambda s: s in fail_day and fail_day[s] <= L)]
    now = now.merge(prev, on="serial_number", how="left", suffixes=("", "_prev"))
    for i in ERR:
        c = f"smart_{i}_raw"
        now[f"d7_{i}"] = now[c] - now[c + "_prev"]
        now = now.drop(columns=c + "_prev")
    now["landmark"] = pd.Timestamp(L)
    fd = now.serial_number.map(fail_day)
    ls = now.serial_number.map(last_seen)
    now["t_obs"] = [((f if not pd.isna(f) else l) - L).days for f, l in zip(fd, ls)]
    now["event"] = fd.notna().astype(int)
    for h in HORIZONS:
        ev = now.event.eq(1) & now.t_obs.le(h)
        known = ev | now.t_obs.ge(h)
        now[f"y_{h}"] = np.where(known, ev.astype(float), np.nan)
    rows.append(now)
    print(L, len(now), {h: int(now[f"y_{h}"].sum()) for h in HORIZONS}, flush=True)

df = pd.concat(rows, ignore_index=True)
df.to_parquet(out_path, index=False)
print("saved", df.shape, flush=True)
