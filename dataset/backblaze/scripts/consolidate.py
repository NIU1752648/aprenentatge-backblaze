"""Consolidate one quarter of Backblaze daily CSVs into Parquet.

Layer 1, panel:  <out>/panel/<label>.parquet  one row per drive and day,
                 only *_raw SMART columns that are not (almost) empty,
                 one row group per day (filter by date is cheap).
Layer 2, drives: <out>/drives.parquet  one row per drive, rebuilt from every
                 file in <out>/panel/, so adding quarters just works.

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
    with open(f) as fh:
        return fh.readline().strip().lstrip("﻿").split(",")


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
src = pq.ParquetFile(tmp)
out_schema = None
with_dict = None
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

# Layer 2: drive table from every quarter in panel/.
panels = sorted(glob.glob(os.path.join(out_dir, "panel", "*.parquet")))
cols = ["date", "serial_number", "model", "capacity_bytes", "datacenter", "failure", "smart_9_raw"]
t = pa.concat_tables(
    [pq.read_table(p, columns=cols).cast(pa.schema([
        ("date", pa.date32()), ("serial_number", pa.string()), ("model", pa.string()),
        ("capacity_bytes", pa.int64()), ("datacenter", pa.string()), ("failure", pa.int8()),
        ("smart_9_raw", pa.float64())])) for p in panels]
)
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
d = g.to_pandas().rename(columns={
    "model_last": "model",
    "capacity_bytes_max": "capacity_bytes",
    "datacenter_last": "datacenter",
    "date_min": "first_seen",
    "date_max": "last_seen",
    "date_count": "n_days",
    "fail_d_min": "fail_date",
    "smart_9_raw_first": "power_on_hours_first",
    "smart_9_raw_last": "power_on_hours_last",
})
import pandas as pd  # noqa: E402

for c in ["first_seen", "last_seen", "fail_date"]:
    d[c] = pd.to_datetime(d[c])
data_end = d.last_seen.max()
d["failed"] = d.fail_date.notna().astype("int8")
d["end_date"] = d.fail_date.fillna(d.last_seen)
d["duration_days"] = (d.end_date - d.first_seen).dt.days
d["censoring"] = "failed"
d.loc[(d.failed == 0) & (d.last_seen == data_end), "censoring"] = "alive_at_end"
d.loc[(d.failed == 0) & (d.last_seen < data_end), "censoring"] = "left_data"
d["model"] = d.model.astype("category")
d["datacenter"] = d.datacenter.astype("category")
d = d[["serial_number", "model", "capacity_bytes", "datacenter", "first_seen", "last_seen", "n_days",
       "fail_date", "failed", "end_date", "duration_days", "censoring",
       "power_on_hours_first", "power_on_hours_last"]]
drives_path = os.path.join(out_dir, "drives.parquet")
d.to_parquet(drives_path, index=False)
print("drives saved", drives_path, d.shape, "failed", int(d.failed.sum()), flush=True)
print(d.censoring.value_counts().to_string(), flush=True)
print(f"total {time.time() - t0:.0f}s", flush=True)
