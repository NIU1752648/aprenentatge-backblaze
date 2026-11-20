"""Daily operations series and failure histories from the consolidated panel.

Walks the panel once, day by day (one row group = one day), keeping a small
vectorised state per drive: a 64-day presence bitmask, the last sum of the
error counters and the days of their changes. Writes small tables to <out>:

  daily_stats.parquet      one row per day: active drives, first failures,
                           drives that appear / come back / leave without
                           failing, drives whose error counters go up.
  landmark_stats.parquet   one row per Wednesday: share of active drives with
                           absent days or error increases in the previous
                           7 / 30 days (feature distributions per date).
  failure_history.parquet  one row per failing drive (first failure): absent
                           days and error-counter timing before the failure.
  drive_onsets.parquet     one row per drive whose error counters go from 0 to
                           > 0 while observed: onset date (join with
                           drives.parquet to see how long until removal).

Usage: python -I daily_stats.py <data dir with panel/ and drives.parquet> <out dir>
   ex: python -I daily_stats.py data out
"""
import glob
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

data_dir, out_dir = sys.argv[1], sys.argv[2]
ERR = [f"smart_{i}_raw" for i in (5, 187, 188, 197, 198)]
EPOCH = date(1970, 1, 1)

serials = pd.read_parquet(os.path.join(data_dir, "drives.parquet"), columns=["serial_number"]).serial_number
index = pd.Index(serials.astype(str).to_numpy())
n = len(index)
print("drives", n, flush=True)

NONE = np.int32(-(2**30))
mask = np.zeros(n, np.uint64)            # bit k = present k days before the current day
first_day = np.full(n, NONE, np.int32)   # days since epoch
last_day = np.full(n, NONE, np.int32)
fail_day = np.full(n, NONE, np.int32)
err_last = np.full(n, np.nan)            # last observed sum of error counters
onset_day = np.full(n, NONE, np.int32)   # first observed 0 -> >0 transition
err_at_first = np.zeros(n, bool)         # counters already > 0 on first day seen
last_inc = np.full(n, NONE, np.int32)    # last day the error sum went up
n_inc = np.zeros(n, np.int32)            # number of days with an increase


def popcount(x):
    return np.bitwise_count(x).astype(np.int32)


def absent_last(ids, d, w):
    """Absent days among the previous w days (excluding today), only since first seen."""
    win = np.where(first_day[ids] == NONE, 0, np.clip(d - first_day[ids], 0, w))  # unseen: no window
    bits = (mask[ids] >> np.uint64(1)) & ((np.uint64(1) << win.astype(np.uint64)) - np.uint64(1))
    return win - popcount(bits), win


daily, marks, fails = [], [], []
prev_d = None
t0 = time.time()
files = sorted(glob.glob(os.path.join(data_dir, "panel", "*.parquet")))
for f in files:
    pf = pq.ParquetFile(f)
    have = set(pf.schema_arrow.names)
    err_cols = [c for c in ERR if c in have]
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["date", "serial_number", "failure"] + err_cols)
        day = t["date"][0].as_py()
        d = (day - EPOCH).days
        ids = index.get_indexer(pc.cast(t["serial_number"], "string").to_numpy(zero_copy_only=False))
        assert (ids >= 0).all(), "serial missing from drives.parquet: rebuild it with consolidate.py"
        failure = t["failure"].to_numpy(zero_copy_only=False) == 1
        err = np.zeros(len(t))
        for c in err_cols:
            err += np.nan_to_num(t[c].to_numpy(zero_copy_only=False).astype(float))

        delta = 1 if prev_d is None else d - prev_d
        mask <<= np.uint64(delta)
        seen_before = first_day[ids] != NONE
        on_prev = (last_day == prev_d) if prev_d is not None else np.zeros(n, bool)
        today = np.zeros(n, bool)
        today[ids] = True
        failed_before = fail_day != NONE

        # leavers: present on the previous day, missing today, never failed
        n_gone = int((on_prev & ~today & ~failed_before).sum())
        n_new = int((~seen_before).sum())
        n_return = int((seen_before & ~on_prev[ids]).sum())

        # error counters: compare with the drive's previous observation
        prev_err = err_last[ids]
        inc = seen_before & ~np.isnan(prev_err) & (err > prev_err)
        new_onset = inc & (prev_err == 0) & (onset_day[ids] == NONE)
        onset_day[ids[new_onset]] = d
        last_inc[ids[inc]] = d
        n_inc[ids[inc]] += 1
        err_at_first[ids[~seen_before & (err > 0)]] = True

        # first failures: record history before updating presence state
        first_fail = failure & (fail_day[ids] == NONE)
        if first_fail.any():
            fi = ids[first_fail]
            a7, w7 = absent_last(fi, d, 7)
            a30, w30 = absent_last(fi, d, 30)
            fails.append(pd.DataFrame({
                "serial_number": index[fi],
                "fail_date": day,
                "days_observed": np.where(first_day[fi] == NONE, 0, d - first_day[fi]),
                "gap_before": np.where(last_day[fi] == NONE, -1, d - last_day[fi]),
                "absent_7": a7, "window_7": w7,
                "absent_30": a30, "window_30": w30,
                "err_at_fail": err[first_fail],
                "err_at_first_seen": err_at_first[fi],
                "onset_day": onset_day[fi],
                "last_inc_day": last_inc[fi],
                "n_inc_days": n_inc[fi],
            }))
            fail_day[fi] = d

        first_day[ids[~seen_before]] = d
        last_day[ids] = d
        mask[ids] |= np.uint64(1)
        err_last[ids] = err

        daily.append({
            "date": day, "n_active": len(ids), "n_fail": int(first_fail.sum()),
            "n_fail_rows": int(failure.sum()), "n_new": n_new, "n_return": n_return,
            "n_gone": n_gone, "n_err_any": int((err > 0).sum()), "n_err_inc": int(inc.sum()),
            "n_err_onset": int(new_onset.sum()),
        })

        old = ids[(d - first_day[ids]) >= 30]
        if day.weekday() == 2 and len(old):  # Wednesday landmarks, as in build.py
            a7, _ = absent_last(old, d, 7)
            a30, _ = absent_last(old, d, 30)
            e0 = err_last[old] > 0
            marks.append({
                "date": day, "n_drives_30d": len(old),
                "share_absent_7": float((a7 > 0).mean()), "share_absent_30": float((a30 > 0).mean()),
                "mean_absent_30": float(a30.mean()),
                "share_err_any": float(e0.mean()),
                "share_inc_7": float((last_inc[old] > d - 7).mean()),
                "share_inc_30": float((last_inc[old] > d - 30).mean()),
            })
        prev_d = d
        if len(daily) % 30 == 0:
            print(f"  {day} active {len(ids)} ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(out_dir, exist_ok=True)


def to_date(col):
    return pd.to_datetime(np.where(col == NONE, np.nan, col.astype(float)), unit="D")


pd.DataFrame(daily).assign(date=lambda x: pd.to_datetime(x.date)).to_parquet(
    os.path.join(out_dir, "daily_stats.parquet"), index=False)
pd.DataFrame(marks).assign(date=lambda x: pd.to_datetime(x.date)).to_parquet(
    os.path.join(out_dir, "landmark_stats.parquet"), index=False)
fh = pd.concat(fails, ignore_index=True)
fh["fail_date"] = pd.to_datetime(fh.fail_date)
fh["onset_date"] = to_date(fh.pop("onset_day").to_numpy())
fh["last_inc_date"] = to_date(fh.pop("last_inc_day").to_numpy())
fh.to_parquet(os.path.join(out_dir, "failure_history.parquet"), index=False)
has = onset_day != NONE
pd.DataFrame({"serial_number": index[has], "onset_date": to_date(onset_day[has])}).to_parquet(
    os.path.join(out_dir, "drive_onsets.parquet"), index=False)
print("days", len(daily), "failures", len(fh), "onsets", int(has.sum()), f"{time.time() - t0:.0f}s", flush=True)
