"""Build the landmark tables (layer 3) from the consolidated Backblaze data.

Landmarks L: every Wednesday from 2025-02-05 to 2026-04-29 (L-30 and L+60 are
inside the data). One file per landmark, <out dir>/L=YYYY-MM-DD.parquet, with a
`split` column: "dev" (2025-02-05 .. 2025-11-05, 40 dates), "gap"
(2025-11-12 .. 2025-12-31, 8) and "test" (2026-01-07 .. 2026-04-29, 17).

Population at L: drives seen in [L-7, L] (decision 19) that have not failed on or
before L. Values are those of the drive's latest row in that window (carried
forward); `days_since_seen` = L - that row's day (0 if the drive has a row on L).
Drives that are absent on L and never come back without failing are dropped:
their labels would be unknown at every horizon. model and capacity_bytes come
from drives.parquet.

Columns (one row per drive and L):
  ids       date (= L), serial_number, split
  G0        model (raw), manufacturer, capacity_bytes, age_years (SMART 9 / 8766 h)
  G1        smart_<i>_raw for i in 1 7 10 12 183 184 189 190 192 193 194 199 240
            241 242, plus d30_241, d30_242
  G2        smart_<i>_raw for the error counters 5 187 188 197 198 (raw, NaN kept)
  G3        d1_<i>, d7_<i>, d30_<i> for each error counter
  GO        absent_30, days_since_return, days_since_first, seen_at_start,
            days_since_seen
  labels    event, t_obs, y_7, y_30, y_60

Lag rule (d<k>_<i>): latest value (as of L) minus the value of the drive's most
recent row in the window [L-k-7, L-k]. NaN when the drive has no row in that window or
either value is NaN (a row with the counter missing counts as the latest row).

GO (from data/presence.parquet and presence_days.parquet):
  absent_30          days in [L-30, L-1] without a row, counting only days
                     >= first_seen and excluding fleet-wide outage days.
  days_since_return  L minus the day the drive came back from its most recent
                     non-outage absence (first present day after the gap);
                     equal to days_since_first if never absent since first_seen.
  days_since_first   L - first_seen (lower bound when seen_at_start).
  seen_at_start      first_seen == 2025-01-01 (left-truncated).

Labels: event = 1 if the drive fails after L; t_obs = fail_date - L if event,
else last_seen - L (known alive up to last_seen over the whole period, so gaps
are not censoring). y_t = 1 if event and t_obs <= t, 0 if t_obs >= t otherwise,
NaN if the drive left before L+t without failing. No holiday censoring,
sampling or imputation here (applied when reading).

The panel is walked once, day by day (one row group = one day), keeping the
latest SMART values and last-seen day per drive; the lagged columns of that
state are snapshotted at L-1, L-7 and L-30.

Usage: python -I build.py <data dir> <out dir> [--no-presence]
   ex: python -I build.py data out/landmarks
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

args = [a for a in sys.argv[1:] if not a.startswith("--")]
data_dir, out_dir = args[0], args[1]
use_presence = "--no-presence" not in sys.argv
os.makedirs(out_dir, exist_ok=True)
log_file = open(os.path.join(os.path.dirname(os.path.abspath(out_dir)), "build_log.txt"), "w", encoding="utf-8")
t0 = time.time()


def log(*parts):
    msg = " ".join(str(p) for p in parts)
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()


DAY0 = date(2025, 1, 1)
N_DAYS = 546
ERR = [5, 187, 188, 197, 198]  # counters Backblaze uses to judge failure
G1 = [1, 7, 10, 12, 183, 184, 189, 190, 192, 193, 194, 199, 240, 241, 242]
LAG_COLS = ERR + [241, 242]    # columns kept in the per-drive state
LAGS = [1, 7, 30]
LAG_WINDOW = 7                 # look-back: latest row in [L-k-7, L-k]
SEEN_WINDOW = 7                # population at L: drives seen in [L-7, L]
HORIZONS = [7, 30, 60]
# Raw values above 2^24 lose integer precision in float32: keep these in float64.
WIDE = {1, 7, 10, 188, 192, 193, 241, 242}
NONE = np.int16(-1)


def didx(d):
    return (d - DAY0).days


# --- Landmarks and splits -----------------------------------------------------
landmarks = []
L = date(2025, 2, 5)
while L <= date(2026, 4, 29):
    landmarks.append(L)
    L += timedelta(7)


def split_of(L):
    if L <= date(2025, 11, 5):
        return "dev"
    return "gap" if L <= date(2025, 12, 31) else "test"


splits = {L: split_of(L) for L in landmarks}
counts = pd.Series(splits).value_counts().to_dict()
assert all(L.weekday() == 2 for L in landmarks)
assert counts == {"dev": 40, "gap": 8, "test": 17}, counts
# (the d30 window of 2025-02-05 is truncated to [2025-01-01, L-30])
assert didx(landmarks[0]) - 30 >= 0 and didx(landmarks[-1]) + 60 <= N_DAYS - 1
log("landmarks", len(landmarks), counts, landmarks[0], "->", landmarks[-1])

# --- Drives -------------------------------------------------------------------
drives = pd.read_parquet(os.path.join(data_dir, "drives.parquet"),
                         columns=["serial_number", "model", "capacity_bytes", "first_seen", "last_seen",
                                  "fail_date"])
index = pd.Index(drives.serial_number.astype(str).to_numpy())
assert index.is_unique
n = len(index)


def to_idx(col):
    v = (col.dt.normalize() - pd.Timestamp(DAY0)).dt.days
    return v.fillna(-1).astype(np.int16).to_numpy()


first_idx = to_idx(drives.first_seen)
last_idx = to_idx(drives.last_seen)
fail_idx = to_idx(drives.fail_date)  # -1 if no failure
log("drives", n, "failures", int((fail_idx >= 0).sum()))
drive_model = drives.model.astype(str).to_numpy()
drive_capacity = drives.capacity_bytes.to_numpy()


def manufacturer(m):
    u = m.upper()
    if u.startswith("ST") or u.startswith("SEAGATE"):
        return "Seagate"
    if u.startswith("HGST") or u.startswith(("HUH", "HUS", "HMS")):
        return "HGST"
    if u.startswith("TOSHIBA") or u.startswith(("MG0", "MG1")):
        return "Toshiba"
    if u.startswith(("WDC", "WUH", "WD ")):
        return "WDC"
    if u.startswith("CT"):
        return "Crucial"
    if u.startswith(("MTFD", "MICRON")):
        return "Micron"
    if u.startswith("SAMSUNG"):
        return "Samsung"
    if u.startswith("SSDSC"):
        return "Intel"
    if u.startswith("DELLBOSS"):
        return "Dell"
    return "other"


mf = pd.DataFrame({"model": drives.model.astype(str)})
mf["manufacturer"] = mf.model.map({m: manufacturer(m) for m in mf.model.unique()})
log("manufacturer (drives, models):\n" + mf.groupby("manufacturer").agg(
    drives=("model", "size"), models=("model", "nunique")).to_string())
log("models -> 'other':", sorted(mf.model[mf.manufacturer == "other"].unique()))
del mf

# --- Presence -----------------------------------------------------------------
if use_presence:
    pres = pq.read_table(os.path.join(data_dir, "presence.parquet"))
    assert pres.num_rows == n
    assert (pc.cast(pres["serial_number"], "string").to_numpy(zero_copy_only=False) == index.to_numpy()).all()
    raw = pres["bits"].combine_chunks()
    nbytes = (N_DAYS + 7) // 8
    buf = np.frombuffer(raw.buffers()[-1], np.uint8)[raw.offset * nbytes:(raw.offset + n) * nbytes]
    present = np.unpackbits(buf.reshape(n, nbytes), axis=1, bitorder="little")[:, :N_DAYS].astype(bool)
    del pres, raw, buf
    pdays = pd.read_parquet(os.path.join(data_dir, "presence_days.parquet")).sort_values("day_idx")
    assert (pdays.day_idx.to_numpy() == np.arange(N_DAYS)).all()
    outage = pdays.outage.to_numpy(bool)
    log("presence", present.shape, "outage days", [str(d)[:10] for d in pdays.date[outage]])
else:
    present, outage = None, np.zeros(N_DAYS, bool)
    log("presence: SKIPPED (--no-presence), GO columns are NaN")


def go_features(ids, Li):
    """absent_30, days_since_return, days_since_first for drives ids at day Li."""
    fs = first_idx[ids].astype(np.int32)
    since_first = Li - fs
    if present is None:
        nan = np.full(len(ids), np.nan, np.float32)
        return nan, nan, since_first
    days = np.arange(Li)
    absent = ~present[ids, :Li] & (days[None, :] >= fs[:, None]) & ~outage[None, :Li]
    absent_30 = absent[:, Li - 30:].sum(axis=1).astype(np.int16)
    has = absent.any(axis=1)
    last_abs = Li - 1 - np.argmax(absent[:, ::-1], axis=1)
    since_return = np.where(has, Li - (last_abs + 1), since_first)
    return absent_30, since_return, since_first


# --- Panel walk ---------------------------------------------------------------
files = sorted(glob.glob(os.path.join(data_dir, "panel", "*.parquet")))
day_groups = {}  # day index -> (file, row group)
for f in files:
    md = pq.ParquetFile(f).metadata
    j = [md.row_group(0).column(c).path_in_schema for c in range(md.num_columns)].index("date")
    for g in range(md.num_row_groups):
        st = md.row_group(g).column(j).statistics
        assert st.min == st.max, (f, g)
        day_groups[didx(st.min)] = (f, g)
assert sorted(day_groups) == list(range(N_DAYS)), "expected one row group per day"

SMART_L = ERR + G1 + [9]
smart_name = [f"smart_{i}_raw" for i in SMART_L]
lag_pos = [SMART_L.index(i) for i in LAG_COLS]      # lagged columns inside the state
state_val = np.full((n, len(SMART_L)), np.nan)      # latest row's values
state_day = np.full(n, NONE, np.int16)              # latest day with a row
targets = {}  # snapshot day -> landmarks needing it
for L in landmarks:
    for k in LAGS:
        targets.setdefault(didx(L) - k, []).append(didx(L))
snaps = {}
land_idx = {didx(L): L for L in landmarks}
summary = []
files_cache = {}


def read_group(f, g, cols):
    pf = files_cache.setdefault(f, pq.ParquetFile(f))
    have = set(pf.schema_arrow.names)
    t = pf.read_row_group(g, columns=[c for c in cols if c in have])
    for c in cols:  # columns dropped in some quarter read as null
        if c not in have:
            t = t.append_column(c, pa.nulls(t.num_rows, pa.float64()))
    return t


def col(t, c):
    return t[c].to_numpy(zero_copy_only=False).astype(np.float64)


panel_after_fail = 0
mismatch = 0
for di in range(didx(landmarks[-1]) + 1):
    f, g = day_groups[di]
    t = read_group(f, g, ["serial_number"] + smart_name)
    ids = index.get_indexer(pc.cast(t["serial_number"], "string").to_numpy(zero_copy_only=False))
    assert (ids >= 0).all(), "serial missing from drives.parquet: rebuild it with consolidate.py"
    panel_after_fail += int(((fail_idx[ids] >= 0) & (fail_idx[ids] < di)).sum())
    if present is not None:
        today = np.zeros(n, bool)
        today[ids] = True
        mismatch += int((today != present[:, di]).sum())
    state_val[ids] = np.column_stack([col(t, c) for c in smart_name])
    state_day[ids] = di

    if di in land_idx:
        L = land_idx[di]
        seen = (state_day >= di - SEEN_WINDOW) & ~((fail_idx >= 0) & (fail_idx <= di))
        gone = seen & (last_idx < di)   # absent on L, never back, no failure: labels unknown
        r = np.flatnonzero(seen & ~gone)
        out = {"date": pa.array(np.full(len(r), np.datetime64(L, "D"))).cast(pa.date32()),
               "serial_number": index.to_numpy()[r],
               "split": splits[L]}
        model = drive_model[r]
        out["model"] = model
        out["manufacturer"] = pd.Series(model).map({m: manufacturer(m) for m in set(model)}).to_numpy()
        out["capacity_bytes"] = drive_capacity[r]
        sv_now = state_val[r]
        out["age_years"] = (sv_now[:, SMART_L.index(9)] / (24 * 365.25)).astype(np.float32)
        for i in G1 + ERR:
            out[f"smart_{i}_raw"] = sv_now[:, SMART_L.index(i)].astype(np.float64 if i in WIDE else np.float32)
        now = sv_now[:, lag_pos]
        for k in LAGS:
            sv, sd = snaps[di - k]
            ok = sd[r] >= di - k - LAG_WINDOW
            diff = np.where(ok[:, None], now - sv[r], np.nan)
            for j, i in enumerate(LAG_COLS):
                if i in ERR or k == 30:
                    out[f"d{k}_{i}"] = diff[:, j].astype(np.float64 if i in WIDE else np.float32)
        a30, dsr, dsf = go_features(r, di)
        out["absent_30"] = a30
        out["days_since_return"] = dsr.astype(np.int16) if present is not None else dsr
        out["days_since_first"] = dsf.astype(np.int16)
        out["seen_at_start"] = first_idx[r] == 0
        out["days_since_seen"] = (di - state_day[r]).astype(np.int16)
        fi = fail_idx[r].astype(np.int32)
        event = fi > di
        t_obs = np.where(event, fi - di, last_idx[r].astype(np.int32) - di).astype(np.int16)
        assert (t_obs >= 0).all() and (~event | (t_obs > 0)).all()
        out["event"] = event.astype(np.int8)
        out["t_obs"] = t_obs
        for h in HORIZONS:
            pos = event & (t_obs <= h)
            out[f"y_{h}"] = np.where(pos, 1.0, np.where(t_obs >= h, 0.0, np.nan)).astype(np.float32)
        df = pd.DataFrame({c: v for c, v in out.items() if c != "date"})
        cols_order = (["date", "serial_number", "split", "model", "manufacturer", "capacity_bytes", "age_years"]
                      + [f"smart_{i}_raw" for i in G1] + ["d30_241", "d30_242"]
                      + [f"smart_{i}_raw" for i in ERR]
                      + [f"d{k}_{i}" for i in ERR for k in LAGS]
                      + ["absent_30", "days_since_return", "days_since_first", "seen_at_start",
                         "days_since_seen", "event", "t_obs"] + [f"y_{h}" for h in HORIZONS])
        df = df[cols_order[1:]]
        table = pa.Table.from_pandas(df, preserve_index=False).add_column(0, "date", out["date"])
        pq.write_table(table,
                       os.path.join(out_dir, f"L={L}.parquet"), compression="zstd")
        row = {"date": L, "split": splits[L], "rows": len(df),
               "absent_on_L": int((out["days_since_seen"] > 0).sum()), "gone": int(gone.sum()),
               **{f"pos_{h}": int(np.nansum(df[f"y_{h}"])) for h in HORIZONS},
               **{f"nan_{h}": float(df[f"y_{h}"].isna().mean()) for h in HORIZONS},
               "d30_nan": float(df["d30_5"].isna().mean()), "d7_nan": float(df["d7_5"].isna().mean()),
               "d1_nan": float(df["d1_5"].isna().mean()),
               "absent30_mean": float(np.mean(a30)), "absent30_any": float(np.mean(a30 > 0)),
               "fails_band7": int(((event) & (t_obs <= 7)).sum())}
        summary.append(row)
        log(f"{L} {splits[L]:4s} rows {len(df)} y7/30/60 {row['pos_7']}/{row['pos_30']}/{row['pos_60']} "
            f"absent30>0 {row['absent30_any']:.4f} ({time.time() - t0:.0f}s)")
        for s in [s for s in snaps if max(targets[s]) <= di]:  # no later landmark needs it
            del snaps[s]

    if di in targets:
        snaps[di] = (state_val[:, lag_pos].copy(), state_day.copy())

# --- Summary ------------------------------------------------------------------
S = pd.DataFrame(summary)
S.to_csv(os.path.join(os.path.dirname(os.path.abspath(out_dir)), "build_summary.csv"), index=False)
log("\nper split:")
log(S.groupby("split").agg(dates=("rows", "size"), rows=("rows", "sum"),
                           pos_7=("pos_7", "sum"), pos_30=("pos_30", "sum"), pos_60=("pos_60", "sum"),
                           nan_7=("nan_7", "mean"), nan_30=("nan_30", "mean"), nan_60=("nan_60", "mean"),
                           d30_nan=("d30_nan", "mean")).to_string())
# Each failure on day in (L0, L_last+7] falls in band (0, 7] of exactly one Wednesday.
lo, hi = didx(landmarks[0]), didx(landmarks[-1]) + 7
in_window = int(((fail_idx > lo) & (fail_idx <= hi)).sum())
log(f"band (0,7] check: {int(S.fails_band7.sum())} failures in band over all dates vs "
    f"{in_window} first failures in ({landmarks[0]}, {landmarks[-1] + timedelta(7)}]")
log("panel rows after the drive's first failure:", panel_after_fail)
if present is not None:
    log("presence vs panel mismatches:", mismatch)
log(f"done {time.time() - t0:.0f}s")
