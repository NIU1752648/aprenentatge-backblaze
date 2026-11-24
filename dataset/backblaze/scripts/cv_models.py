"""Grouped cross-validation of gradient boosting and logistic regression (development period).

What it does
------------
1. Loads only the 40 development landmarks (split == "dev", 2025-02-05 .. 2025-11-05). Gap and
   test files are filtered out by file name before anything is read, and every row read is
   asserted to be "dev".
2. Recomputes the labels strictly, with no information after CUTOFF = 2025-11-16 (holiday cut,
   decision 3): a failure counts if fail_date <= CUTOFF, and a drive is known alive through day D
   only if it has a row on some day in [D, CUTOFF] (presence bits), never from last_seen / t_obs of
   the whole period. Targets:
     (a)  y_7, y_30, y_60: failure in (L, L+t]; only dates with L+t <= CUTOFF.
     (b1) bands b_0_7, b_7_30, b_30_60: for band (a, b], rows at risk at L+a (known alive at L+a,
          not failed by then); label = failure in (L+a, L+b]; unknown -> excluded; only L+b <= CUTOFF.
   The number of labels that differ from the stored y_t is logged and saved in run_info.json.
3. Folds: K = 3 by drive, fold = crc32("fold:" + serial_number) % 3 (fixed, identical for every
   model, configuration and target).
4. Training sample, per target and training fold: all positives + every row (all dates) of the
   negative drives with crc32("negative-sample:" + serial) / 2^32 < --neg-drive-frac (default 2 %,
   pi ~ 2 %). Trained unweighted; probabilities corrected with p = q*pi / (q*pi + 1 - q), with
   pi = kept negatives / negatives of the training fold. Negatives are sampled by DRIVE, never by
   date alone: keeping positives on all dates and negatives on a subset of dates leaks the label,
   because date proxies (days_since_first + seen_at_start) identify the landmark. The optional
   --date-step k keeps 1 of every k dates per drive for positives and negatives alike.
   Sanity check: every fit logs mean uncorrected q on its training sample vs the sample's positive
   rate, and mean corrected p vs observed rate on the held-out fold; ratios outside [0.5, 2] are
   flagged with "WARNING calibration".
5. Evaluation on the held-out fold: every held-out row with a known label (no sampling). Alert
   budget per date (top k % of that date's rows by score, exact expected value under random
   tie-breaking), k in {0.01, 0.1, 0.5, 1, 2, 5, 10} %. Recall and precision at each k, PR-AUC,
   ROC-AUC, log loss and mean predicted vs observed rate (after the pi correction).
6. Grid (pla_projecte.md 3.1): HistGradientBoosting (learning_rate x max_depth x min_samples_leaf,
   trees by early stopping on a drive-grouped inner validation split) and logistic regression
   (L2 x 5 C, L1 x 2 C). Choice per model and target: best mean recall@1 % and the one-standard-
   deviation rule (3.1: simplest config whose mean is within one fold sd of the best).
7. Final stage: refits the chosen (1-SD) config of each target and fold, scores ALL held-out rows,
   writes out-of-fold predictions, the budget curve, the band cumulative probability
   1 - prod(1 - h_k) evaluated at 7/30/60 on the same rows as (a) ("b1_cum"), and the share of
   rows / drives whose cumulative curve decreases (E4 preview).

Every (stage, model, config, target, fold) result is appended to cv_folds.csv as soon as it is
computed; a restart skips finished rows (and finished final-stage folds, cached in tmp/).

Outputs in <out dir> (meant to be committed: the CSV / JSON / log; git-ignored: the parquet files)
  cv_folds.csv, cv_summary.csv, best_configs.json, budget_curve.csv, run_info.json, cv_log.txt
  oof_<model>_<a|b1>.parquet, tmp/*.parquet   (big, not committed)

Importable: the script has no side effects on import (everything runs under __main__), so other
scripts (learning_curve.py) can load it by file path and reuse load_drive_status,
load_dev_landmarks, strict_interval_labels, strict_labels, feature_columns, FEATURE_GROUPS,
drive_hashes, sample_training_rows, correct_probability, frequent_model_map, design_matrix,
make_hgb, fit_hgb, make_logistic, fit_model, predict_proba_rows, budget_metrics, evaluate_scores.

Usage (from dataset/backblaze/):
  uv run python -I scripts/cv_models.py <landmarks dir> <data dir> <out dir>
        [--jobs N] [--models gb,lr] [--neg-drive-frac 0.02] [--date-step 1] [--quick]
Homelab command (full grid, 8 threads):
  uv run python -I scripts/cv_models.py out/landmarks data out/cv --jobs 8
Smoke test (3 dates, 1 config per model; use another out dir):
  uv run python -I scripts/cv_models.py out/landmarks data out/cv_quick --quick --jobs 8
Expected runtime (8 threads, measured on an 8-thread laptop with the full development data):
grid 180 GB fits x ~70 s (fit ~10 s, scoring ~4 M held-out rows ~55 s) + 105 logistic fits
(L2 ~17 s, L1/saga ~90-170 s) ~ 4.5-5 h; final stage ~0.6 h; total ~5-6 h (b_0_7 reuses the y_7
fits: same labels). The estimate printed at start uses these numbers and the log shows an ETA.
Memory: features 2.3 GB in float32 + per-chunk scoring; peak ~4-5 GB.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
import zlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --- Fixed design -----------------------------------------------------------------------------
DAY0 = date(2025, 1, 1)               # day index 0 of the presence bits
N_DAYS = 546
DEV_FIRST, DEV_LAST = date(2025, 2, 5), date(2025, 11, 5)
N_DEV_DATES = 40
CUTOFF = date(2025, 11, 16)           # holiday cut for development labels (decision 3)
K_FOLDS = 3
SEED = 0
FOLD_RULE = "fold = zlib.crc32(b'fold:' + serial_number.encode()) % 3"
SAMPLE_RULE = ("negatives: drives with crc32('negative-sample:'+serial)/2**32 < neg_drive_frac, all their "
               "dates; all positives kept. Optional date thinning (date_step > 1): rows with "
               "(date_pos + crc32('date-phase:'+serial) % date_step) % date_step == 0, applied to positives "
               "and negatives alike (thinning negatives only leaks the label through date proxies)")
INNER_RULE = "GB early-stopping validation = training-sample drives with crc32('inner-validation:'+serial) % 10 == 0"
DEFAULT_NEG_DRIVE_FRAC = 0.02
DEFAULT_DATE_STEP = 1

HORIZONS = (7, 30, 60)
BANDS = ((0, 7), (7, 30), (30, 60))
# target name -> (formulation, a, b): label = failure in (L+a, L+b], at risk at L+a
TARGETS = {f"y_{t}": ("a", 0, t) for t in HORIZONS}
TARGETS.update({f"b_{a}_{b}": ("b1", a, b) for a, b in BANDS})
BUDGETS = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)   # % of the rows of each date
SELECTION_BUDGET = 1.0

ERR = (5, 187, 188, 197, 198)
G1_SMART = (1, 7, 10, 12, 183, 184, 189, 190, 192, 193, 194, 199, 240, 241, 242)
GROUP_ORDER = ("G0", "G1", "GO", "G2", "G3")
# Feature groups of the plan (1.3 / E1). "manufacturer" and "model" are categorical; "model" is
# reduced per training fold to the frequent models (rest -> "other").
FEATURE_GROUPS = {
    "G0": ["manufacturer", "model", "capacity_tb", "age_years"],
    "G1": [f"smart_{i}_raw" for i in G1_SMART] + ["d30_241", "d30_242"],
    "GO": ["absent_30", "days_since_return", "days_since_first", "seen_at_start", "days_since_seen"],
    "G2": [f"smart_{i}_raw" for i in ERR],
    "G3": [f"d{k}_{i}" for i in ERR for k in (1, 7, 30)],
}
CATEGORICAL = ("manufacturer", "model")
NUMERIC_FEATURES = [c for g in GROUP_ORDER for c in FEATURE_GROUPS[g] if c not in CATEGORICAL]
# SMART levels and deltas are stored as sign(x) * log1p(|x|) (= log1p for x >= 0): monotone, so
# irrelevant for trees, and the logistic transform of the plan. Keeps float32 exact enough.
LOG_FEATURES = set(FEATURE_GROUPS["G1"] + FEATURE_GROUPS["G2"] + FEATURE_GROUPS["G3"])
# columns that also get a "> 0" indicator in the logistic model (counter level / counter went up)
POSITIVE_INDICATORS = FEATURE_GROUPS["G2"] + FEATURE_GROUPS["G3"]

GB_GRID = [{"learning_rate": lr, "max_depth": d, "min_samples_leaf": m}
           for lr in (0.05, 0.1) for d in (3, 5) for m in (500, 2000, 5000)]
LR_GRID = ([{"penalty": "l2", "C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0)]
           + [{"penalty": "l1", "C": c} for c in (0.01, 0.1)])
QUICK_GRID = {"gb": [{"learning_rate": 0.1, "max_depth": 3, "min_samples_leaf": 2000}],
              "lr": [{"penalty": "l2", "C": 0.1}]}
QUICK_DATES = (date(2025, 3, 5), date(2025, 6, 4), date(2025, 9, 3))
GB_FIXED = {"max_iter": 1000, "l2_regularization": 1.0, "n_iter_no_change": 15, "tol": 1e-7,
            "max_bins": 255}
LR_MAX_ITER = 3000
MIN_MODEL_DRIVES = 500        # frequent model: >= 500 distinct drives in the training fold
MAX_MODEL_LEVELS = 60
# Seconds per fit (fit + scoring of one held-out fold) measured on the full development data with
# 8 threads on the development laptop; used only for the estimate printed at start.
EST_SECONDS = {"gb": 70.0, "lr": 50.0}   # lr: mean of L2 (~17 s) and L1/saga (~130 s)


def day_index(d):
    """Day index from 2025-01-01 (index of the presence bits)."""
    return (d - DAY0).days


def config_id(model, params):
    """Stable short name of a configuration."""
    if model == "gb":
        return f"gb_lr{params['learning_rate']}_d{params['max_depth']}_m{params['min_samples_leaf']}"
    return f"lr_{params['penalty']}_C{params['C']}"


def complexity_key(model, params, mean_n_iter=0.0):
    """Sort key: smaller = simpler (one-SD rule). GB: shallower, larger leaves, fewer trees.
    Logistic: stronger regularisation (smaller C), L1 before L2 at equal C."""
    if model == "gb":
        return (params["max_depth"], -params["min_samples_leaf"], mean_n_iter)
    return (params["C"], 0 if params["penalty"] == "l1" else 1)


# --- Hashes, folds, sampling ------------------------------------------------------------------
def stable_hash(keys, salt):
    """crc32 of 'salt:key' for every key (uint32), identical on every OS and Python version."""
    pre = f"{salt}:".encode()
    return np.fromiter((zlib.crc32(pre + str(k).encode()) for k in keys), np.uint32, len(keys))


@dataclass
class DriveHashes:
    """Deterministic per-drive hashes, in the order of drives.parquet."""
    fold: np.ndarray       # int8, 0..K-1
    sample_u: np.ndarray   # float64 uniform in [0, 1): negative drive sample
    phase: np.ndarray      # uint32 raw hash: date phase = phase % date_step
    inner: np.ndarray      # bool: GB early-stopping validation drives


def drive_hashes(serials, k=K_FOLDS):
    """Fold assignment (FOLD_RULE), negative-sample uniform, date phase and inner-validation flag."""
    return DriveHashes(
        fold=(stable_hash(serials, "fold") % k).astype(np.int8),
        sample_u=stable_hash(serials, "negative-sample") / 2.0 ** 32,
        phase=stable_hash(serials, "date-phase"),
        inner=stable_hash(serials, "inner-validation") % 10 == 0,
    )


def sample_training_rows(y, train_mask, drive, date_pos, hashes, neg_drive_frac=DEFAULT_NEG_DRIVE_FRAC,
                         date_step=DEFAULT_DATE_STEP):
    """Training rows among rows with train_mask and a known y: all positives + the negatives of a
    deterministic drive sample (all their dates).

    Optional date thinning (date_step > 1) keeps 1 of every date_step dates of each drive (per-drive
    phase) and is applied to positives AND negatives alike: thinning only the negatives would let
    date proxies (days_since_first, seen_at_start, ...) reveal "this date has no negatives" and
    leak the label. Thinning both does not change the class ratio, so only the drive sample enters pi.

    Returns (rows, pi): row indices (int64, sorted) and pi = kept negatives / negatives among the
    date-selected rows (1.0 when no drive is dropped). Correct with correct_probability(q, pi).
    """
    known = train_mask & ~np.isnan(y)
    if date_step > 1:
        known = known & ((date_pos.astype(np.int64) + hashes.phase[drive] % date_step) % date_step == 0)
    pos = known & (y == 1)
    neg = known & (y == 0)
    keep = neg & (hashes.sample_u[drive] < neg_drive_frac) if neg_drive_frac < 1 else neg
    n_neg = int(neg.sum())
    pi = float(keep.sum()) / n_neg if n_neg else 1.0
    return np.flatnonzero(pos | keep), pi


def correct_probability(q, pi):
    """Undo negative sampling at rate pi: p = q*pi / (q*pi + 1 - q)."""
    q = np.asarray(q, np.float64)
    return q * pi / (q * pi + 1.0 - q)


# --- Drives, presence and strict labels -------------------------------------------------------
@dataclass
class DriveStatus:
    """Per drive (order of drives.parquet), using only information up to `cutoff`."""
    serial: np.ndarray        # str
    index: pd.Index           # serial -> position
    fail_cut: np.ndarray      # int32 day index of the first failure if <= cutoff, else -1
    last_pres_cut: np.ndarray  # int32 last day index <= cutoff with a row, -1 if none
    cutoff_idx: int


def load_drive_status(data_dir, cutoff=CUTOFF):
    """Read drives.parquet and presence.parquet and keep only what is known on `cutoff`."""
    drives = pd.read_parquet(os.path.join(data_dir, "drives.parquet"), columns=["serial_number", "fail_date"])
    serial = drives.serial_number.astype(str).to_numpy()
    index = pd.Index(serial)
    assert index.is_unique
    n = len(serial)
    cut = day_index(cutoff)
    fail = (drives.fail_date.dt.normalize() - pd.Timestamp(DAY0)).dt.days.fillna(-1).astype(np.int32).to_numpy()
    fail_cut = np.where((fail >= 0) & (fail <= cut), fail, -1).astype(np.int32)

    pres = pq.read_table(os.path.join(data_dir, "presence.parquet"))
    assert pres.num_rows == n
    assert (pc.cast(pres["serial_number"], "string").to_numpy(zero_copy_only=False) == serial).all()
    raw = pres["bits"].combine_chunks()
    nbytes = (N_DAYS + 7) // 8
    buf = np.frombuffer(raw.buffers()[-1], np.uint8)[raw.offset * nbytes:(raw.offset + n) * nbytes]
    present = np.unpackbits(buf.reshape(n, nbytes), axis=1, bitorder="little")[:, :cut + 1].astype(bool)
    rev = present[:, ::-1]
    last = np.where(rev.any(axis=1), cut - np.argmax(rev, axis=1), -1).astype(np.int32)
    # a failure day always has a row
    assert (last[fail_cut >= 0] >= fail_cut[fail_cut >= 0]).all()
    return DriveStatus(serial=serial, index=index, fail_cut=fail_cut, last_pres_cut=last, cutoff_idx=cut)


def strict_interval_labels(day, drive, a, b, status):
    """Strict label of 'failure in (L+a, L+b]' with information up to status.cutoff_idx only.

    day: landmark day index per row; drive: position in drives.parquet per row.
    Returns float32: 1 failure in (L+a, L+b]; 0 not failed by L+b and present on some day in
    [L+b, cutoff]; NaN when L+b > cutoff, when (a > 0) the drive is not at risk at L+a (failed by
    L+a, or not seen on any day in [L+a, cutoff]), or when the outcome is unknown.
    With a = 0 this is y_t (the row population is alive at L by construction); with a > 0 it is
    the discrete-time hazard label of band (a, b].
    """
    L = np.asarray(day, np.int32)
    f = status.fail_cut[drive]
    lp = status.last_pres_cut[drive]
    assert not ((f >= 0) & (f <= L)).any(), "row of a drive that failed on or before L"
    failed_by_b = (f >= 0) & (f <= L + b)
    pos = (f > L + a) & (f <= L + b)
    neg = ~failed_by_b & (lp >= L + b)
    y = np.where(pos, 1.0, np.where(neg, 0.0, np.nan)).astype(np.float32)
    scope = L + b <= status.cutoff_idx
    if a > 0:
        scope &= ~((f >= 0) & (f <= L + a)) & (lp >= L + a)
    y[~scope] = np.nan
    return y


def strict_labels(data, status, targets=None):
    """Dict target name -> strict label array (float32, NaN = excluded) for every row of `data`."""
    targets = TARGETS if targets is None else targets
    return {name: strict_interval_labels(data.day, data.drive, a, b, status)
            for name, (_, a, b) in targets.items()}


def label_differences(data, labels, status):
    """Compare strict y_t with the stored y_t (dates with L+t <= cutoff); counts per horizon."""
    out = {}
    for t in HORIZONS:
        scope = data.day.astype(np.int32) + t <= status.cutoff_idx
        s = data.stored_labels[t][scope]
        n = labels[f"y_{t}"][scope]
        cat = lambda v: np.where(np.isnan(v), "nan", np.where(v == 1, "1", "0"))
        tab = pd.crosstab(cat(s), cat(n))
        out[f"y_{t}"] = {
            "rows_out_of_scope_L+t>cutoff": int((~scope).sum()),
            "rows_in_scope": int(scope.sum()),
            "stored_vs_strict": {f"{r}->{c}": int(tab.loc[r, c]) for r in tab.index for c in tab.columns},
            "differ": int((cat(s) != cat(n)).sum()),
        }
    return out


# --- Landmarks --------------------------------------------------------------------------------
@dataclass
class DevData:
    """Development landmark rows. X columns = numeric_features (float32, NaN kept)."""
    X: np.ndarray
    numeric_features: list
    manufacturer: np.ndarray      # int16 codes into manufacturer_names
    manufacturer_names: list
    model: np.ndarray             # int16 codes into model_names
    model_names: list
    drive: np.ndarray             # int32 position in drives.parquet
    day: np.ndarray               # int16 landmark day index
    date_pos: np.ndarray          # int8 position among the 40 development dates
    dates: list                   # landmark dates loaded
    stored_labels: dict = field(default_factory=dict)  # t -> stored y_t (float32)


def dev_landmark_files(landmarks_dir, dates=None):
    """[(L, path)] of the development landmarks, selected by FILE NAME (gap/test never opened)."""
    found = {}
    for name in os.listdir(landmarks_dir):
        if name.startswith("L=") and name.endswith(".parquet"):
            L = date.fromisoformat(name[2:12])
            if DEV_FIRST <= L <= DEV_LAST:
                found[L] = os.path.join(landmarks_dir, name)
    assert len(found) == N_DEV_DATES and all(L.weekday() == 2 for L in found), sorted(found)
    all_dates = sorted(found)
    if dates is not None:
        assert set(dates) <= set(all_dates), dates
        all_dates = sorted(dates)
    return [(L, found[L]) for L in all_dates]


def load_dev_landmarks(landmarks_dir, drive_index, dates=None, log=print):
    """Load development landmarks (all 40, or `dates`) into a DevData, reading only needed columns.

    drive_index: pd.Index of serial numbers in drives.parquet order (DriveStatus.index).
    SMART levels/deltas -> sign*log1p (float32); capacity in TB (NaN if <= 0); seen_at_start 0/1.
    """
    files = dev_landmark_files(landmarks_dir, dates)
    pos_of = {L: i for i, (L, _) in enumerate(dev_landmark_files(landmarks_dir))}
    sizes = [pq.ParquetFile(p).metadata.num_rows for _, p in files]
    n = int(sum(sizes))
    p = len(NUMERIC_FEATURES)
    X = np.empty((n, p), np.float32)
    drive = np.empty(n, np.int32)
    day = np.empty(n, np.int16)
    date_pos = np.empty(n, np.int8)
    manuf = np.empty(n, np.int16)
    model = np.empty(n, np.int16)
    stored = {t: np.empty(n, np.float32) for t in HORIZONS}
    manuf_codes, model_codes = {}, {}
    source = [c for c in NUMERIC_FEATURES if c != "capacity_tb"] + ["capacity_bytes"]
    cols = ["date", "serial_number", "split", "model", "manufacturer"] + source + [f"y_{t}" for t in HORIZONS]
    off = 0
    for (L, path), m in zip(files, sizes):
        t = pq.read_table(path, columns=cols)
        assert t.num_rows == m
        split = pc.unique(t["split"]).to_pylist()
        assert split == ["dev"], (path, split)
        dts = pc.unique(t["date"]).to_pylist()
        assert dts == [L], (path, dts)
        sl = slice(off, off + m)
        ser = pc.cast(t["serial_number"], "string").to_numpy(zero_copy_only=False)
        ids = drive_index.get_indexer(ser)
        assert (ids >= 0).all(), "serial missing from drives.parquet"
        drive[sl] = ids
        day[sl] = day_index(L)
        date_pos[sl] = pos_of[L]
        for name, codes, dest in (("manufacturer", manuf_codes, manuf), ("model", model_codes, model)):
            v = pc.cast(t[name], "string").to_numpy(zero_copy_only=False)
            u, inv = np.unique(v, return_inverse=True)
            for s in u:
                codes.setdefault(str(s), len(codes))
            dest[sl] = np.array([codes[str(s)] for s in u], np.int16)[inv]
        for j, c in enumerate(NUMERIC_FEATURES):
            if c == "capacity_tb":
                v = t["capacity_bytes"].to_numpy(zero_copy_only=False).astype(np.float64)
                v = np.where(v > 0, v / 1e12, np.nan)
            else:
                v = t[c].to_numpy(zero_copy_only=False).astype(np.float64)
                if c in LOG_FEATURES:
                    v = np.sign(v) * np.log1p(np.abs(v))
            X[sl, j] = v
        for h in HORIZONS:
            stored[h][sl] = t[f"y_{h}"].to_numpy(zero_copy_only=False)
        off += m
        log(f"  loaded {L} rows {m}")
    return DevData(X=X, numeric_features=list(NUMERIC_FEATURES),
                   manufacturer=manuf, manufacturer_names=list(manuf_codes),
                   model=model, model_names=list(model_codes),
                   drive=drive, day=day, date_pos=date_pos, dates=[L for L, _ in files],
                   stored_labels=stored)


# --- Features and design matrices -------------------------------------------------------------
def feature_columns(groups=GROUP_ORDER):
    """Feature names of the given groups, numeric first and categorical last (design order)."""
    names = [c for g in groups for c in FEATURE_GROUPS[g]]
    return [c for c in names if c not in CATEGORICAL] + [c for c in CATEGORICAL if c in names]


def frequent_model_map(data, rows, min_drives=MIN_MODEL_DRIVES, max_levels=MAX_MODEL_LEVELS):
    """Map model code -> level computed on the training rows only: models with >= min_drives
    distinct drives (at most max_levels, most frequent first) keep a level, the rest share the
    last level ("other"). Returns (mapping int16 array, list of level names)."""
    n_models = len(data.model_names)
    key = data.model[rows].astype(np.int64) * (2 ** 31) + data.drive[rows]
    pairs = np.unique(key)
    counts = np.bincount((pairs // (2 ** 31)).astype(np.int64), minlength=n_models)
    order = [c for c in np.argsort(-counts, kind="stable") if counts[c] >= min_drives][:max_levels]
    mapping = np.full(n_models, len(order), np.int16)
    mapping[order] = np.arange(len(order), dtype=np.int16)
    return mapping, [data.model_names[c] for c in order] + ["other"]


def design_matrix(data, rows, model_map, features=None):
    """float32 design matrix of `rows` for `features` (default: all groups), in feature_columns
    order; manufacturer and model (frequent-model level) as integer codes in the last columns."""
    features = feature_columns() if features is None else features
    num = [c for c in features if c not in CATEGORICAL]
    j = [data.numeric_features.index(c) for c in num]
    blocks = [data.X[rows][:, j]] if len(j) < data.X.shape[1] else [data.X[rows]]
    if "manufacturer" in features:
        blocks.append(data.manufacturer[rows].astype(np.float32)[:, None])
    if "model" in features:
        blocks.append(model_map[data.model[rows]].astype(np.float32)[:, None])
    return np.hstack(blocks) if len(blocks) > 1 else blocks[0]


def make_hgb(params, features, random_state=SEED):
    """HistGradientBoostingClassifier for a design from design_matrix(features): NaN native,
    manufacturer/model native categorical, early stopping on an explicit validation set."""
    cat_mask = np.array([c in CATEGORICAL for c in features])
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=params["learning_rate"], max_depth=params["max_depth"],
        max_leaf_nodes=None, min_samples_leaf=params["min_samples_leaf"],
        categorical_features=cat_mask if cat_mask.any() else None,
        early_stopping=True, scoring="loss", random_state=random_state,
        **{**GB_FIXED, **{k: v for k, v in params.items()
                          if k not in ("learning_rate", "max_depth", "min_samples_leaf")}})


def fit_hgb(model, X, y, val_mask):
    """Fit with early stopping on the rows of val_mask (drive-grouped split, see INNER_RULE)."""
    return model.fit(X[~val_mask], y[~val_mask], X_val=X[val_mask], y_val=y[val_mask])


def _positive_indicator(X):
    return (X > 0).astype(np.float32)


def make_logistic(params, features, random_state=SEED):
    """Pipeline for a design from design_matrix(features): numeric -> median imputation + missing
    indicators -> standardisation; '> 0' indicators for error-counter levels and deltas; one-hot
    manufacturer and frequent model; L1 (saga) or L2 (lbfgs) logistic regression."""
    num = [i for i, c in enumerate(features) if c not in CATEGORICAL]
    ind = [i for i, c in enumerate(features) if c in POSITIVE_INDICATORS]
    cat = [i for i, c in enumerate(features) if c in CATEGORICAL]
    parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                               ("sc", StandardScaler())]), num)]
    if ind:
        parts.append(("gt0", FunctionTransformer(_positive_indicator), ind))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cat))
    l1 = params["penalty"] == "l1"
    lr = LogisticRegression(C=params["C"], l1_ratio=1.0 if l1 else 0.0, solver="saga" if l1 else "lbfgs",
                            max_iter=LR_MAX_ITER, tol=1e-4, random_state=random_state)
    return Pipeline([("prep", ColumnTransformer(parts, sparse_threshold=0)), ("lr", lr)])


def fit_model(model_name, params, X, y, val_mask, features, random_state=SEED):
    """Build and fit one model. Returns (fitted model, info dict with n_iter / convergence)."""
    if model_name == "gb":
        m = fit_hgb(make_hgb(params, features, random_state), X, y, val_mask)
        return m, {"n_iter": int(m.n_iter_), "converged": True, "warnings": ""}
    m = make_logistic(params, features, random_state)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(X, y)
    conv = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    return m, {"n_iter": int(np.max(m[-1].n_iter_)), "converged": not conv,
               "warnings": str(conv[0].message)[:120] if conv else ""}


def predict_proba_rows(model, data, rows, model_map, features, chunk=500_000):
    """Uncorrected P(y=1) for `rows`, scored in chunks to bound memory (float64)."""
    out = np.empty(len(rows), np.float64)
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        out[s:s + len(r)] = model.predict_proba(design_matrix(data, r, model_map, features))[:, 1]
    return out


# --- Metrics ----------------------------------------------------------------------------------
def budget_metrics(y, score, group, budgets=BUDGETS):
    """Recall and precision with an alert budget per group (date): the top round(k% * n_group)
    rows of each group (at least 1) by score. Ties at the threshold get fractional credit (exact
    expected value under random tie-breaking). Pooled over groups.
    Returns {k: {"recall", "precision", "alerts", "tp"}}."""
    y = np.asarray(y, np.float64)
    score = np.asarray(score, np.float64)
    order = np.lexsort((-score, group))
    g, s, yy = np.asarray(group)[order], score[order], y[order]
    bounds = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    res = {k: [0.0, 0] for k in budgets}
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ss = s[lo:hi]
        cum = np.r_[0.0, np.cumsum(yy[lo:hi])]
        neg_s = -ss  # ascending
        n = hi - lo
        for k in budgets:
            m = max(1, int(np.floor(k / 100.0 * n + 0.5)))
            thr = ss[m - 1]
            above = int(np.searchsorted(neg_s, -thr, side="left"))   # count s > thr
            tied_end = int(np.searchsorted(neg_s, -thr, side="right"))
            tied = tied_end - above
            tp = cum[above] + (cum[tied_end] - cum[above]) * (m - above) / tied
            res[k][0] += tp
            res[k][1] += m
    total_pos = y.sum()
    return {k: {"recall": tp / total_pos if total_pos else np.nan, "precision": tp / a,
                "alerts": a, "tp": tp} for k, (tp, a) in res.items()}


def budget_key(k, what):
    return f"{what}@{k:g}%"


def evaluate_scores(y, p, group, budgets=BUDGETS):
    """All metrics for corrected probabilities p of labelled rows (y 0/1, group = date)."""
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    out = {}
    bm = budget_metrics(y, p, group, budgets)
    for k in budgets:
        out[budget_key(k, "recall")] = bm[k]["recall"]
        out[budget_key(k, "precision")] = bm[k]["precision"]
    if 0 < y.sum() < len(y):
        out["pr_auc"] = average_precision_score(y, p)
        out["roc_auc"] = roc_auc_score(y, p)
    else:
        out["pr_auc"] = out["roc_auc"] = np.nan
    out["log_loss"] = log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])
    out["mean_pred"] = float(p.mean())
    out["obs_rate"] = float(y.mean())
    out["calib_ratio"] = out["mean_pred"] / out["obs_rate"] if out["obs_rate"] > 0 else np.nan
    return out


METRIC_COLS = ([budget_key(k, w) for w in ("recall", "precision") for k in BUDGETS]
               + ["pr_auc", "roc_auc", "log_loss", "mean_pred", "obs_rate", "calib_ratio"])
KEY_COLS = ["stage", "model", "config_id", "formulation", "target", "fold"]
FOLD_COLS = (KEY_COLS + ["params", "neg_drive_frac", "date_step", "pi", "n_train", "n_train_pos",
                         "n_val", "n_eval", "n_eval_pos", "n_eval_dates", "fit_s", "predict_s",
                         "n_iter", "converged", "warnings", "train_mean_q", "train_pos_rate"] + METRIC_COLS)


# --- Runner (only under __main__) -------------------------------------------------------------
class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, *parts):
        msg = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{(time.time() - self.t0) / 60:7.1f} min] " + " ".join(
            str(p) for p in parts)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def _ram_gb():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as fh:
                return round(int(fh.readline().split()[1]) / 2 ** 20, 1)
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("t", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            s = MS()
            s.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return round(s.t / 2 ** 30, 1)
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
        return round(int(out) / 2 ** 30, 1)
    except Exception:
        return None


def _peak_rss_gb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (2 ** 30 if sys.platform == "darwin" else 2 ** 20), 2)
    except Exception:
        return None


def _git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here,
                               capture_output=True, text=True).stdout.strip()
        return {"commit": h or None, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def _versions():
    from importlib.metadata import version
    out = {"python": sys.version.split()[0]}
    for p in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "threadpoolctl"):
        try:
            out[p] = version(p)
        except Exception:
            out[p] = None
    return out


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


class Results:
    """cv_folds.csv with append-only checkpoints and resume."""

    def __init__(self, path):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.done = {self.key(r) for r in df[KEY_COLS].astype(str).to_dict("records")}

    @staticmethod
    def key(r):
        return tuple(str(r[c]) for c in KEY_COLS)

    def has(self, **k):
        return self.key(k) in self.done

    def add(self, row):
        df = pd.DataFrame([{c: row.get(c, np.nan) for c in FOLD_COLS}])
        df.to_csv(self.path, mode="a", header=not os.path.exists(self.path), index=False)
        self.done.add(self.key(row))

    def frame(self):
        return pd.read_csv(self.path) if os.path.exists(self.path) else pd.DataFrame(columns=FOLD_COLS)


def summarise(df):
    """Mean and sd over folds per (stage, model, config, formulation, target)."""
    g = df.groupby(["stage", "model", "config_id", "formulation", "target"], sort=False)
    mean = g[METRIC_COLS + ["fit_s", "n_iter", "pi", "n_train"]].mean()
    sd = g[METRIC_COLS].std(ddof=1).add_suffix("_sd")
    out = pd.concat([g.size().rename("n_folds"), g["params"].first(), mean, sd], axis=1).reset_index()
    return out


def select_configs(summary, models, k=K_FOLDS):
    """Best mean recall@1 % and the one-SD choice (3.1) per model and target, among grid configs
    with all K folds."""
    sel = budget_key(SELECTION_BUDGET, "recall")
    choice = {}
    grid = summary[(summary.stage == "grid") & (summary.n_folds == k)]
    for model in models:
        choice[model] = {}
        for target in TARGETS:
            s = grid[(grid.model == model) & (grid.target == target)]
            if s.empty:
                continue
            best = s.loc[s[sel].idxmax()]
            thr = best[sel] - (best[f"{sel}_sd"] if np.isfinite(best[f"{sel}_sd"]) else 0.0)
            cand = s[s[sel] >= thr]
            keyed = sorted(cand.itertuples(), key=lambda r: complexity_key(model, json.loads(r.params), r.n_iter))
            one = keyed[0]
            choice[model][target] = {
                "best": {"config_id": best.config_id, "params": json.loads(best.params),
                         sel: best[sel], f"{sel}_sd": best[f"{sel}_sd"], "mean_n_iter": best.n_iter},
                "one_sd": {"config_id": one.config_id, "params": json.loads(one.params),
                           sel: float(s.loc[s.config_id == one.config_id, sel].iloc[0]),
                           "mean_n_iter": float(one.n_iter)},
                "threshold": thr, "n_candidates": int(len(cand)),
            }
    return choice


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("landmarks_dir")
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--models", default="gb,lr")
    ap.add_argument("--neg-drive-frac", type=float, default=DEFAULT_NEG_DRIVE_FRAC)
    ap.add_argument("--date-step", type=int, default=DEFAULT_DATE_STEP)
    ap.add_argument("--quick", action="store_true", help="3 dates, 1 config per model (smoke test)")
    args = ap.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    assert set(models) <= {"gb", "lr"}, models
    assert 0 < args.neg_drive_frac <= 1 and args.date_step >= 1

    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=args.jobs)

    out = args.out_dir
    os.makedirs(os.path.join(out, "tmp"), exist_ok=True)
    log = Logger(os.path.join(out, "cv_log.txt"))
    info_path = os.path.join(out, "run_info.json")
    settings = {"quick": args.quick, "neg_drive_frac": args.neg_drive_frac, "date_step": args.date_step,
                "cutoff": str(CUTOFF), "k_folds": K_FOLDS, "fold_rule": FOLD_RULE, "seed": SEED}
    info = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as fh:
            info = json.load(fh)
        if info.get("settings") != settings:
            sys.exit(f"{out} holds a run with other settings {info.get('settings')}; use another out dir")
    grids = QUICK_GRID if args.quick else {"gb": GB_GRID, "lr": LR_GRID}
    info.setdefault("runs", []).append({"start": datetime.now().isoformat(timespec="seconds"),
                                        "argv": sys.argv, "args": vars(args)})
    info.update({
        "settings": settings, "git": _git_info(), "versions": _versions(),
        "machine": {"os": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                    "ram_gb": _ram_gb()},
        "seeds": {"random_state": SEED, "hashes": "zlib.crc32 with salts fold / negative-sample / "
                                                  "date-phase / inner-validation"},
        "sample_rule": SAMPLE_RULE, "inner_validation_rule": INNER_RULE,
        "budgets_pct": BUDGETS, "selection": f"recall@{SELECTION_BUDGET:g}% mean over folds; one-SD rule "
                                             "(simplest config with mean >= best mean - best fold sd)",
        "targets": {k: list(v) for k, v in TARGETS.items()}, "feature_groups": FEATURE_GROUPS,
        "grids": {m: grids[m] for m in models}, "gb_fixed": GB_FIXED, "lr_max_iter": LR_MAX_ITER,
        "frequent_models": f">= {MIN_MODEL_DRIVES} drives in the training fold, at most {MAX_MODEL_LEVELS}",
    })
    _write_json(info_path, info)
    log(f"start: models {models}, jobs {args.jobs}, quick {args.quick}, neg_drive_frac {args.neg_drive_frac}, "
        f"date_step {args.date_step}, cutoff {CUTOFF}")

    # --- data
    status = load_drive_status(args.data_dir, CUTOFF)
    log(f"drives {len(status.serial)}, failures <= cutoff {int((status.fail_cut >= 0).sum())}")
    data = load_dev_landmarks(args.landmarks_dir, status.index, QUICK_DATES if args.quick else None, log)
    assert data.day.max() <= day_index(DEV_LAST) and data.day.min() >= day_index(DEV_FIRST)
    log(f"dev rows {len(data.day)}, dates {len(data.dates)} ({data.dates[0]} .. {data.dates[-1]}), "
        f"X {data.X.shape} {data.X.nbytes / 2 ** 30:.2f} GB, models {len(data.model_names)}")
    labels = strict_labels(data, status)
    diffs = label_differences(data, labels, status)
    for t, d in diffs.items():
        log(f"labels {t}: in scope {d['rows_in_scope']}, out of scope {d['rows_out_of_scope_L+t>cutoff']}, "
            f"differ from stored {d['differ']}: {d['stored_vs_strict']}")
    counts = {name: {"known": int((~np.isnan(y)).sum()), "pos": int(np.nansum(y)),
                     "dates": int(len(np.unique(data.day[~np.isnan(y)])))} for name, y in labels.items()}
    for name, c in counts.items():
        log(f"target {name}: known {c['known']}, positives {c['pos']}, dates {c['dates']}")
    hashes = drive_hashes(status.serial)
    fold = hashes.fold[data.drive]
    info.update({"label_differences": diffs, "target_counts": counts,
                 "rows": int(len(data.day)), "dates": [str(L) for L in data.dates],
                 "fold_sizes": {int(k): int((fold == k).sum()) for k in range(K_FOLDS)}})
    _write_json(info_path, info)
    features = feature_columns()

    res = Results(os.path.join(out, "cv_folds.csv"))
    jobs_all = [(f, t, m, p) for f in range(K_FOLDS) for t in TARGETS for m in models for p in grids[m]]
    todo = [j for j in jobs_all if not res.has(stage="grid", model=j[2], config_id=config_id(j[2], j[3]),
                                                formulation=TARGETS[j[1]][0], target=j[1], fold=j[0])]
    n_by = {m: sum(1 for j in todo if j[2] == m and j[1] != "b_0_7") for m in models}
    est = sum(n_by[m] * EST_SECONDS[m] for m in models) * 8 / max(1, min(args.jobs, 8))
    est_final = sum((len(TARGETS) - 1) * K_FOLDS * EST_SECONDS[m] * 1.2 for m in models) * 8 / max(1, min(args.jobs, 8))
    log(f"grid fits: {len(jobs_all)} total, {len(todo)} to do {n_by}; estimate {est / 3600:.1f} h grid + "
        f"{est_final / 3600:.1f} h final stage (full data, 8 threads calibration; quick runs are much faster)")

    # --- grid
    times = {m: [] for m in models}

    def final_tmp_path(m, cid, t, f):
        return os.path.join(out, "tmp", f"final_{cid}_{t}_f{f}.parquet")
    # band (0, 7] has the same labels as y_7: its fits are copied from y_7 within a fold
    same_as_y7 = bool(np.array_equal(labels["y_7"], labels["b_0_7"], equal_nan=True))
    log(f"b_0_7 labels identical to y_7: {same_as_y7} (fits reused)")
    y7_cache = {}

    def run_fold_jobs(stage, f, job_list, chosen=None):
        """Fit/score jobs [(target, model, params)] of fold f. stage 'final' also caches full
        held-out predictions in tmp/."""
        train = fold != f
        held = np.flatnonzero(fold == f)
        model_map, levels = frequent_model_map(data, np.flatnonzero(train))
        log(f"fold {f}: train rows {int(train.sum())}, held-out rows {len(held)}, frequent models "
            f"{len(levels) - 1} + other")
        by_target = {}
        for t, m, p in job_list:
            by_target.setdefault(t, []).append((m, p))
        for t, mp in by_target.items():
            y = labels[t]
            rows, pi = sample_training_rows(y, train, data.drive, data.date_pos, hashes,
                                            args.neg_drive_frac, args.date_step)
            Xtr = design_matrix(data, rows, model_map, features)
            ytr = y[rows].astype(np.int8)
            val = hashes.inner[data.drive[rows]]
            ev = held[~np.isnan(y[held])]
            yev = y[ev]
            for m, p in mp:
                cid = config_id(m, p)
                tmp_path = final_tmp_path(m, cid, t, f)
                reuse = same_as_y7 and t == "b_0_7" and (m, cid) in y7_cache
                if reuse:  # identical labels and sample: identical model, no refit
                    model, finfo, fit_s, pred_s, q, q_all = None, *y7_cache[(m, cid)]
                else:
                    t0 = time.time()
                    model, finfo = fit_model(m, p, Xtr, ytr, val, features)
                    fit_s = time.time() - t0
                    t1 = time.time()
                    q_all = predict_proba_rows(model, data, held, model_map, features) if stage == "final" else None
                    q = q_all[~np.isnan(y[held])] if stage == "final" else predict_proba_rows(
                        model, data, ev, model_map, features)
                    pred_s = time.time() - t1
                    # calibration-in-the-large on the training sample (leak / overfit check)
                    finfo["train_mean_q"] = float(predict_proba_rows(model, data, rows, model_map, features).mean())
                    finfo["train_pos_rate"] = float(ytr.mean())
                    if t == "y_7" and same_as_y7:
                        y7_cache[(m, cid)] = (finfo, fit_s, pred_s, q, q_all)
                if stage == "final":
                    pd.DataFrame({"row": held.astype(np.int32), "q": q_all.astype(np.float32),
                                  "pi": np.float64(pi)}).to_parquet(tmp_path)
                pcor = correct_probability(q, pi)
                met = evaluate_scores(yev, pcor, data.day[ev])
                row = {"stage": stage, "model": m, "config_id": cid, "formulation": TARGETS[t][0], "target": t,
                       "fold": f, "params": json.dumps(p, sort_keys=True), "neg_drive_frac": args.neg_drive_frac,
                       "date_step": args.date_step, "pi": pi, "n_train": len(rows), "n_train_pos": int(ytr.sum()),
                       "n_val": int(val.sum()), "n_eval": len(ev), "n_eval_pos": int(yev.sum()),
                       "n_eval_dates": int(len(np.unique(data.day[ev]))), "fit_s": round(fit_s, 2),
                       "predict_s": round(pred_s, 2), **finfo, **met}
                res.add(row)
                if not reuse:
                    times[m].append(fit_s + pred_s)
                r_train = finfo["train_mean_q"] / finfo["train_pos_rate"]
                if not (0.5 <= met["calib_ratio"] <= 2.0 and 0.5 <= r_train <= 2.0):
                    log(f"WARNING calibration {stage} f{f} {t} {cid}: train mean q {finfo['train_mean_q']:.5f} vs "
                        f"rate {finfo['train_pos_rate']:.5f}; held-out mean p {met['mean_pred']:.6f} vs observed "
                        f"{met['obs_rate']:.6f} (ratio {met['calib_ratio']:.2f})")
                done_now = sum(len(v) for v in times.values())
                left = {mm: max(0, n_by.get(mm, 0) - len(times[mm])) for mm in models} if stage == "grid" else {}
                eta = sum(left[mm] * (np.mean(times[mm]) if times[mm] else EST_SECONDS[mm]) for mm in left)
                log(f"{stage} f{f} {t:8s} {cid:24s} n_train {len(rows)} (pos {int(ytr.sum())}, pi {pi:.4f}) "
                    f"fit {fit_s:.0f}s pred {pred_s:.0f}s n_iter {finfo['n_iter']}"
                    f"{'' if finfo['converged'] else ' NOT CONVERGED'} | rec@1% "
                    f"{met[budget_key(1.0, 'recall')]:.3f} rec@0.1% {met[budget_key(0.1, 'recall')]:.3f} "
                    f"PR {met['pr_auc']:.4f} calib held-out {met['calib_ratio']:.2f} "
                    f"train {finfo['train_mean_q'] / finfo['train_pos_rate']:.2f}"
                    + (f" | {done_now} done, ETA grid {eta / 3600:.2f} h" if stage == "grid" else ""))
                del model
            del Xtr
        y7_cache.clear()
        log(f"fold {f} done, peak RSS {_peak_rss_gb()} GB")

    for f in range(K_FOLDS):
        jl = [(t, m, p) for (ff, t, m, p) in todo if ff == f]
        if jl:
            run_fold_jobs("grid", f, jl)

    # --- selection
    df = res.frame()
    summary = summarise(df)
    choice = select_configs(summary, models)
    _write_json(os.path.join(out, "best_configs.json"),
                {"selection": info["selection"], "neg_drive_frac": args.neg_drive_frac,
                 "date_step": args.date_step, "choices": choice})
    for m in models:
        for t, c in choice[m].items():
            log(f"choice {m} {t}: best {c['best']['config_id']} ({c['best']['recall@1%']:.3f} "
                f"+- {c['best']['recall@1%_sd']:.3f}), one-SD {c['one_sd']['config_id']} "
                f"({c['one_sd']['recall@1%']:.3f})")

    # --- final stage: refit the one-SD choice, score all held-out rows
    for f in range(K_FOLDS):
        jl = []
        for m in models:
            for t in TARGETS:
                if t not in choice[m]:
                    continue
                c = choice[m][t]["one_sd"]
                if os.path.exists(final_tmp_path(m, c["config_id"], t, f)) and res.has(
                        stage="final", model=m, config_id=c["config_id"], formulation=TARGETS[t][0], target=t, fold=f):
                    continue
                jl.append((t, m, c["params"]))
        if jl:
            run_fold_jobs("final", f, jl)

    budget_rows, mono = [], {}
    for m in models:
        if any(t not in choice[m] for t in TARGETS):
            log(f"final {m}: grid incomplete, skipped")
            continue
        q = {}
        for t in TARGETS:
            arr = np.full(len(data.day), np.nan, np.float64)
            for f in range(K_FOLDS):
                tp = pd.read_parquet(final_tmp_path(m, choice[m][t]["one_sd"]["config_id"], t, f))
                arr[tp.row.to_numpy()] = correct_probability(tp.q.to_numpy(), float(tp.pi.iloc[0]))
            assert not np.isnan(arr).any()
            q[t] = arr
        cum = {7: q["b_0_7"]}
        cum[30] = 1 - (1 - q["b_0_7"]) * (1 - q["b_7_30"])
        cum[60] = 1 - (1 - cum[30]) * (1 - q["b_30_60"])
        band_cid = "+".join(choice[m][b]["one_sd"]["config_id"] for b in ("b_0_7", "b_7_30", "b_30_60"))
        for t in HORIZONS:
            y = labels[f"y_{t}"]
            for f in range(K_FOLDS):
                if res.has(stage="final", model=m, config_id=band_cid, formulation="b1_cum", target=f"y_{t}", fold=f):
                    continue
                ev = np.flatnonzero((fold == f) & ~np.isnan(y))
                met = evaluate_scores(y[ev], cum[t][ev], data.day[ev])
                res.add({"stage": "final", "model": m, "config_id": band_cid, "formulation": "b1_cum",
                         "target": f"y_{t}", "fold": f, "params": "{}", "neg_drive_frac": args.neg_drive_frac,
                         "date_step": args.date_step, "n_eval": len(ev), "n_eval_pos": int(y[ev].sum()),
                         "n_eval_dates": int(len(np.unique(data.day[ev]))), **met})
        # E4 preview: decreasing cumulative curves on every development row
        for name, (c7, c30, c60) in {"a": (q["y_7"], q["y_30"], q["y_60"]), "b1": (cum[7], cum[30], cum[60])}.items():
            dec = (c7 > c30) | (c30 > c60)
            by_drive = pd.Series(dec).groupby(data.drive).any()
            mono[f"{m}_{name}"] = {"share_rows_decreasing": float(dec.mean()),
                                   "share_drives_decreasing": float(by_drive.mean()), "rows": int(len(dec))}
            log(f"E4 {m} {name}: rows with decreasing cumulative {dec.mean():.4f}, drives {by_drive.mean():.4f}")
        # out-of-fold predictions
        base = {"date": pd.to_datetime(DAY0) + pd.to_timedelta(data.day.astype(np.int64), unit="D"),
                "serial_number": status.serial[data.drive], "fold": fold}
        oof_a = pd.DataFrame({**base, **{f"y_{t}": labels[f"y_{t}"] for t in HORIZONS},
                              **{f"p_{t}": q[f"y_{t}"].astype(np.float32) for t in HORIZONS}})
        oof_a.to_parquet(os.path.join(out, f"oof_{m}_a.parquet"), compression="zstd", index=False)
        oof_b = pd.DataFrame({**base, **{b: labels[b] for b in ("b_0_7", "b_7_30", "b_30_60")},
                              **{f"h_{b[2:]}": q[b].astype(np.float32) for b in ("b_0_7", "b_7_30", "b_30_60")},
                              **{f"cum_{t}": cum[t].astype(np.float32) for t in HORIZONS},
                              **{f"y_{t}": labels[f"y_{t}"] for t in HORIZONS}})
        oof_b.to_parquet(os.path.join(out, f"oof_{m}_b1.parquet"), compression="zstd", index=False)
        del oof_a, oof_b, base
        log(f"wrote oof_{m}_a.parquet and oof_{m}_b1.parquet")

    # --- summaries
    df = res.frame()
    summary = summarise(df)
    summary["is_best"] = False
    summary["is_one_sd"] = False
    for m, ct in choice.items():
        for t, c in ct.items():
            sel = (summary.stage == "grid") & (summary.model == m) & (summary.target == t)
            summary.loc[sel & (summary.config_id == c["best"]["config_id"]), "is_best"] = True
            summary.loc[sel & (summary.config_id == c["one_sd"]["config_id"]), "is_one_sd"] = True
    summary.to_csv(os.path.join(out, "cv_summary.csv"), index=False)
    chosen = set()
    for m, ct in choice.items():
        for t, c in ct.items():
            chosen.add((m, TARGETS[t][0], t, c["one_sd"]["config_id"]))
        if all(b in ct for b in ("b_0_7", "b_7_30", "b_30_60")):
            cid = "+".join(ct[b]["one_sd"]["config_id"] for b in ("b_0_7", "b_7_30", "b_30_60"))
            chosen |= {(m, "b1_cum", f"y_{t}", cid) for t in HORIZONS}
    fin = df[(df.stage == "final").to_numpy()
             & np.array([k in chosen for k in zip(df.model, df.formulation, df.target, df.config_id)], bool)]
    for k in BUDGETS:
        part = fin[["model", "formulation", "target", "config_id", "fold"]].copy()
        part["budget_pct"] = k
        part["recall"] = fin[budget_key(k, "recall")].to_numpy()
        part["precision"] = fin[budget_key(k, "precision")].to_numpy()
        budget_rows.append(part)
    if budget_rows and len(fin):
        B = pd.concat(budget_rows, ignore_index=True)
        M = B.groupby(["model", "formulation", "target", "config_id", "budget_pct"], as_index=False)[
            ["recall", "precision"]].mean()
        M["fold"] = "mean"
        pd.concat([B, M], ignore_index=True).to_csv(os.path.join(out, "budget_curve.csv"), index=False)

    info["runs"][-1]["end"] = datetime.now().isoformat(timespec="seconds")
    info["runs"][-1]["peak_rss_gb"] = _peak_rss_gb()
    info["e4_monotonicity"] = mono
    _write_json(info_path, info)
    log("done")


if __name__ == "__main__":
    main()
