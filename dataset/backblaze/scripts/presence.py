"""Per-drive daily presence table and per-day reporting outages.

Walks the panel once, day by day (one row group = one day), reading only
`date` and `serial_number`, and sets one bit per drive and day. Writes to
<data dir>:

  presence.parquet       one row per drive, same drives and order as
                         drives.parquet: serial_number, bits. `bits` is a
                         fixed_size_binary of ceil(n_days/8) bytes made with
                         np.packbits(present_row, bitorder="little"): bit i
                         (byte i // 8, bit i % 8) = the drive has a row on
                         day index i (0 = first panel day). Decode with
                           np.unpackbits(np.frombuffer(col, np.uint8)
                                         .reshape(n, n_bytes), axis=1,
                                         bitorder="little")[:, :n_days]
  presence_days.parquet  one row per day: day_idx, date, n_present,
                         n_expected (first_seen <= date <= last_seen and not
                         after fail_date), absent_share, outage
                         (absent_share > OUTAGE_THRESHOLD: a general
                         reporting outage, not a per-drive absence). The
                         threshold and first day are stored in the file
                         metadata ("outage_threshold", "first_date").

Usage: python -I presence.py <data dir with panel/ and drives.parquet>
   ex: python -I presence.py data
"""
import glob
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

data_dir = sys.argv[1]
OUTAGE_THRESHOLD = 0.05
t0 = time.time()

drives = pd.read_parquet(os.path.join(data_dir, "drives.parquet"),
                         columns=["serial_number", "first_seen", "last_seen", "n_days", "fail_date"])
index = pd.Index(drives.serial_number.astype(str).to_numpy())
assert index.is_unique
n = len(index)

files = sorted(glob.glob(os.path.join(data_dir, "panel", "*.parquet")))
# Day range from the row-group statistics (one row group = one day)
day_list = []
for f in files:
    md = pq.ParquetFile(f).metadata
    col = md.schema.to_arrow_schema().get_field_index("date")
    for g in range(md.num_row_groups):
        st = md.row_group(g).column(col).statistics
        assert st.min == st.max, f"row group {g} of {f} spans several days"
        day_list.append(st.min)
first_date, last_date = min(day_list), max(day_list)
n_total = (last_date - first_date).days + 1
n_bytes = (n_total + 7) // 8
print("drives", n, "days", n_total, f"{first_date} -> {last_date}", "bytes per drive", n_bytes, flush=True)
assert len(day_list) == len(set(day_list)), "a day appears in more than one row group"

bits = np.zeros((n, n_bytes), np.uint8)
n_present = np.zeros(n_total, np.int32)
seen_day = np.zeros(n_total, bool)
for f in files:
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["date", "serial_number"])
        day = t["date"][0].as_py()
        i = (day - first_date).days
        ids = index.get_indexer(pc.cast(t["serial_number"], "string").to_numpy(zero_copy_only=False))
        assert (ids >= 0).all(), "serial missing from drives.parquet: rebuild it with consolidate.py"
        assert len(np.unique(ids)) == len(ids), f"duplicate serials on {day}"
        bits[ids, i // 8] |= np.uint8(1 << (i % 8))
        n_present[i] = len(ids)
        seen_day[i] = True
        if i % 60 == 0:
            print(f"  {day} present {len(ids)} ({time.time() - t0:.0f}s)", flush=True)
missing_days = [first_date + timedelta(int(k)) for k in np.flatnonzero(~seen_day)]
print("panel days without data:", missing_days or "none", flush=True)

# Expected drives per day: first_seen <= date <= min(last_seen, fail_date)
dates = pd.date_range(first_date, periods=n_total, freq="D")
start = (drives.first_seen - pd.Timestamp(first_date)).dt.days.to_numpy()
end_ts = drives.last_seen.where(drives.fail_date.isna() | (drives.last_seen <= drives.fail_date), drives.fail_date)
end = (end_ts - pd.Timestamp(first_date)).dt.days.to_numpy()
diff = np.zeros(n_total + 1, np.int64)
np.add.at(diff, start, 1)
np.add.at(diff, end + 1, -1)
n_expected = np.cumsum(diff)[:n_total].astype(np.int32)
absent_share = 1 - n_present / n_expected
outage = absent_share > OUTAGE_THRESHOLD

days = pa.table({
    "day_idx": pa.array(np.arange(n_total, dtype=np.int16)),
    "date": pa.array(dates.date, pa.date32()),
    "n_present": pa.array(n_present),
    "n_expected": pa.array(n_expected),
    "absent_share": pa.array(absent_share.astype(np.float64)),
    "outage": pa.array(outage),
}).replace_schema_metadata({"outage_threshold": str(OUTAGE_THRESHOLD), "first_date": first_date.isoformat()})
pq.write_table(days, os.path.join(data_dir, "presence_days.parquet"))

pres = pa.table({
    "serial_number": pa.array(index.to_numpy(), pa.string()),
    "bits": pa.FixedSizeBinaryArray.from_buffers(pa.binary(n_bytes), n, [None, pa.py_buffer(bits.tobytes())]),
}).replace_schema_metadata({"first_date": first_date.isoformat(), "n_days": str(n_total)})
pres_path = os.path.join(data_dir, "presence.parquet")
pq.write_table(pres, pres_path, compression="zstd")
print(f"written ({time.time() - t0:.0f}s)", flush=True)

# ---- checks (on the file as read back) ----
back = pq.read_table(pres_path)
assert back["serial_number"].to_pylist()[:5] == list(index[:5]) and back.num_rows == n
raw = np.frombuffer(back["bits"].combine_chunks().buffers()[1], np.uint8)[: n * n_bytes].reshape(n, n_bytes)
mat = np.unpackbits(raw, axis=1, bitorder="little")[:, :n_total].astype(bool)
assert not np.unpackbits(raw, axis=1, bitorder="little")[:, n_total:].any(), "padding bits set"
pop = mat.sum(1)
bad_n = pop != drives.n_days.to_numpy()
print("check popcount == n_days: mismatches", int(bad_n.sum()), flush=True)
if bad_n.any():
    print(drives.loc[bad_n, ["serial_number", "n_days"]].assign(popcount=pop[bad_n]).head(20).to_string(), flush=True)
first_bit = mat.argmax(1)
last_bit = n_total - 1 - mat[:, ::-1].argmax(1)
bad_f = first_bit != start
bad_l = last_bit != (drives.last_seen - pd.Timestamp(first_date)).dt.days.to_numpy()
print("check first bit == first_seen: mismatches", int(bad_f.sum()), flush=True)
print("check last bit == last_seen: mismatches", int(bad_l.sum()), flush=True)
print("check column sums == n_present:", bool((mat.sum(0) == n_present).all()), flush=True)
# rows after fail_date make n_present exceed n_expected slightly
after_fail = mat & (np.arange(n_total)[None, :] > end[:, None])
print("drive-days present after min(last_seen, fail_date):", int(after_fail.sum()), flush=True)
del mat, after_fail

d = days.to_pandas()
print(f"\noutage days (absent_share > {OUTAGE_THRESHOLD}):", int(d.outage.sum()), flush=True)
print(d.loc[d.outage, ["date", "n_present", "n_expected", "absent_share"]].to_string(index=False), flush=True)
ok = d.loc[~d.outage, "absent_share"]
print("\nabsent_share on non-outage days:", flush=True)
print(f"  median {ok.median():.5f}  mean {ok.mean():.5f}  p90 {ok.quantile(.9):.5f}  "
      f"p99 {ok.quantile(.99):.5f}  max {ok.max():.5f}  min {ok.min():.5f}", flush=True)
near = d[(d.absent_share > OUTAGE_THRESHOLD / 5) & (d.absent_share < OUTAGE_THRESHOLD * 4)]
print(f"days with absent_share in ({OUTAGE_THRESHOLD / 5:.3f}, {OUTAGE_THRESHOLD * 4:.2f}):", flush=True)
print(near[["date", "n_present", "n_expected", "absent_share", "outage"]].to_string(index=False) if len(near) else "  none",
      flush=True)
print("top 10 absent_share:", flush=True)
print(d.nlargest(10, "absent_share")[["date", "absent_share", "outage"]].to_string(index=False), flush=True)

for p in ("presence.parquet", "presence_days.parquet"):
    print(f"{p}: {os.path.getsize(os.path.join(data_dir, p)) / 2**20:.1f} MB", flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
