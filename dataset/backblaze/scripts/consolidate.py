"""Consolidate one quarter of Backblaze daily CSVs into Parquet.

Layer 1, panel:  <out>/panel/<label>.parquet  one row per drive and day,
                 only *_raw SMART columns that are not (almost) empty,
                 one row group per day (filter by date is cheap).
Layer 2, drives: <out>/drives_q/<label>.parquet  per-quarter drive aggregate;
                 <out>/drives.parquet  one row per drive, combined from every
                 file in <out>/drives_q/, so adding quarters just works and
                 only one quarter is ever held in memory.
Readers of panel/ must unify the schemas of the quarters (the kept SMART
columns can differ between quarters): see open_panel() in build.py.

Usage: python -I consolidate.py <dir with daily csv files> <out dir> <label>
   ex: python -I consolidate.py data/q4 data 2025Q4
"""
import glob
import os
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

csv_dir, out_dir, label = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_NULL = 0.999  # drop SMART columns missing in more than this share of rows

files = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
files = [f for f in files if os.path.basename(f)[:4].isdigit()]
print("files", len(files), os.path.basename(files[0]), "->", os.path.basename(files[-1]), flush=True)

BASE = {
    "date": pa.date32(),
    "serial_number": pa.string(),
    "model": pa.string(),
    "capacity_bytes": pa.int64(),
    "failure": pa.int8(),
    "datacenter": pa.string(),
    "cluster_id": pa.int32(),
    "vault_id": pa.int32(),
    "pod_id": pa.int32(),
    "pod_slot_num": pa.int32(),
    "is_legacy_format": pa.string(),
}
DICT = ["model", "datacenter", "is_legacy_format"]


def header(f):
    with open(f, encoding="utf-8-sig") as fh:
        return fh.readline().strip().split(",")


def smart_key(c):
    return int(c.split("_")[1])


# Union of SMART raw columns over all days (headers only, cheap).
smart = sorted({c for f in files for c in header(f) if c.startswith("smart_") and c.endswith("_raw")}, key=smart_key)
schema = pa.schema([(c, t) for c, t in BASE.items()] + [(c, pa.float64()) for c in smart])
print("smart raw columns in csv:", len(smart), flush=True)


def read_day(f):
    h = set(header(f))
    use = [c for c in schema.names if c in h]
    t = pv.read_csv(
        f,
        convert_options=pv.ConvertOptions(
            include_columns=use,
            column_types={c: schema.field(c).type for c in use},
            strings_can_be_null=True,
        ),
    )
    cols = [t[c] if c in h else pa.nulls(len(t), schema.field(c).type) for c in schema.names]
    return pa.Table.from_arrays(cols, schema=schema)


# Pass 1: all raw SMART columns to a temporary file, counting nulls.
os.makedirs(os.path.join(out_dir, "panel"), exist_ok=True)
tmp = os.path.join(out_dir, "panel", f".{label}.tmp.parquet")
nulls = dict.fromkeys(smart, 0)
n_rows = 0
t0 = time.time()
with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
    for i, f in enumerate(files):
        t = read_day(f)
        t = t.sort_by("serial_number")
        n_rows += len(t)
        for c in smart:
            nulls[c] += t[c].null_count
        w.write_table(t, row_group_size=len(t))
        if i % 10 == 0:
            print(f"  {os.path.basename(f)} rows {len(t)} ({time.time() - t0:.0f}s)", flush=True)
share = {c: nulls[c] / n_rows for c in smart}
keep = [c for c in smart if share[c] <= MAX_NULL]
drop = [c for c in smart if share[c] > MAX_NULL]
print(f"rows {n_rows}; keep {len(keep)} SMART columns, drop {len(drop)} (>{MAX_NULL:.1%} missing)", flush=True)

# Pass 2: rewrite with the kept columns, dictionary-encoded strings.
final_cols = list(BASE) + keep
panel_path = os.path.join(out_dir, "panel", f"{label}.parquet")
out_schema = None
with_dict = None
with pq.ParquetFile(tmp) as src:  # must be closed before removing it (Windows)
    for g in range(src.num_row_groups):
        t = src.read_row_group(g, columns=final_cols)
        for c in DICT:
            t = t.set_column(t.schema.get_field_index(c), c, pc.dictionary_encode(t[c]))
        if with_dict is None:
            out_schema = t.schema
            with_dict = pq.ParquetWriter(panel_path, out_schema, compression="zstd")
        with_dict.write_table(t.cast(out_schema), row_group_size=len(t))
with_dict.close()
os.remove(tmp)

pa.Table.from_pydict({
    "column": smart,
    "missing_share": [round(share[c], 5) for c in smart],
    "kept": [c in keep for c in smart],
}).to_pandas().to_csv(os.path.join(out_dir, "panel", f"{label}_columns.csv"), index=False)
print("panel saved", panel_path, f"{os.path.getsize(panel_path) / 2**20:.0f} MB", flush=True)

# Layer 2a: per-quarter drive aggregate (one quarter in memory at a time).
cols = ["date", "serial_number", "model", "capacity_bytes", "datacenter", "failure", "smart_9_raw"]
t = pq.read_table(panel_path, columns=cols)
t = t.cast(pa.schema([
    ("date", pa.date32()), ("serial_number", pa.string()), ("model", pa.string()),
    ("capacity_bytes", pa.int64()), ("datacenter", pa.string()), ("failure", pa.int8()),
    ("smart_9_raw", pa.float64())]))
t = t.sort_by([("serial_number", "ascending"), ("date", "ascending")])
t = t.append_column("fail_d", pc.if_else(pc.equal(t["failure"], 1), t["date"], pa.scalar(None, pa.date32())))
g = t.group_by("serial_number", use_threads=False).aggregate([
    ("model", "last"),
    ("capacity_bytes", "max"),
    ("datacenter", "last"),
    ("date", "min"),
    ("date", "max"),
    ("date", "count"),
    ("fail_d", "min"),
    ("smart_9_raw", "first"),
    ("smart_9_raw", "last"),
])
del t
g = g.rename_columns(["serial_number", "model", "capacity_bytes", "datacenter", "first_seen", "last_seen",
                      "n_days", "fail_date", "power_on_hours_first", "power_on_hours_last"])
os.makedirs(os.path.join(out_dir, "drives_q"), exist_ok=True)
pq.write_table(g, os.path.join(out_dir, "drives_q", f"{label}.parquet"), compression="zstd")
print("quarter drives", g.num_rows, flush=True)
del g

# Layer 2b: drive table, combining the aggregates of every quarter in drives_q/.
import pandas as pd  # noqa: E402

parts = sorted(glob.glob(os.path.join(out_dir, "drives_q", "*.parquet")))
q = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
for c in ["first_seen", "last_seen", "fail_date"]:
    q[c] = pd.to_datetime(q[c])
q = q.sort_values(["serial_number", "first_seen"])
d = q.groupby("serial_number", sort=True).agg(
    model=("model", "last"),
    capacity_bytes=("capacity_bytes", "max"),
    datacenter=("datacenter", "last"),
    first_seen=("first_seen", "min"),
    last_seen=("last_seen", "max"),
    n_days=("n_days", "sum"),
    fail_date=("fail_date", "min"),
    power_on_hours_first=("power_on_hours_first", "first"),
    power_on_hours_last=("power_on_hours_last", "last"),
    n_quarters=("first_seen", "size"),
).reset_index()
data_end = d.last_seen.max()
d["failed"] = d.fail_date.notna().astype("int8")
d["end_date"] = d.fail_date.fillna(d.last_seen)
d["duration_days"] = (d.end_date - d.first_seen).dt.days
# days inside [first_seen, last_seen] without a row (drive temporarily out of the data)
d["absent_days"] = ((d.last_seen - d.first_seen).dt.days + 1 - d.n_days).astype("int32")
d["censoring"] = "failed"
d.loc[(d.failed == 0) & (d.last_seen == data_end), "censoring"] = "alive_at_end"
d.loc[(d.failed == 0) & (d.last_seen < data_end), "censoring"] = "left_data"
d["model"] = d.model.astype("category")
d["datacenter"] = d.datacenter.astype("category")
d = d[["serial_number", "model", "capacity_bytes", "datacenter", "first_seen", "last_seen", "n_days",
       "absent_days", "n_quarters", "fail_date", "failed", "end_date", "duration_days", "censoring",
       "power_on_hours_first", "power_on_hours_last"]]
drives_path = os.path.join(out_dir, "drives.parquet")
d.to_parquet(drives_path, index=False)
print("quarters combined:", [os.path.basename(p) for p in parts], flush=True)
print("drives saved", drives_path, d.shape, "failed", int(d.failed.sum()), flush=True)
print(d.censoring.value_counts().to_string(), flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
