"""Learning curve for the negative-sampling decision (decision 9 of pla_projecte.md).

Question: how does cross-validated performance change with the amount of training data,
and where does adding data stop paying off (given fit time)? Each drive appears in up to
40 weekly development landmarks, so the rows are redundant; two axes are varied separately:

  * main axis: fraction of NEGATIVE drives kept (deterministic drive hash), with all their
    landmarks: 1, 2, 5, 15, 30, 50, 100 %; all positive rows are kept;
  * landmark subset applied to positives AND negatives alike: every landmark, 1 in 3, 1 in
    10 (date_pos % step == 0, the same dates for every drive).
Thinning the dates of negatives only is NOT done: days_since_first with seen_at_start
identifies the landmark date, so the model would learn "date outside the negative sample =>
positive" (a prototype gave mean q 0.067 on the training sample and 0.72 on held-out rows).
As a guard, every fit logs calibration-in-the-large on the held-out fold (mean corrected p /
observed rate, flagged outside 0.67-1.5) and the mean uncorrected q on the training sample
vs its positive rate and vs the held-out rows.

Shared definitions. Everything about data, labels, folds, sampling, models and metrics is
imported from cv_models.py (loaded by file path), so both scripts agree exactly:
  load_drive_status, load_dev_landmarks (split == "dev" only, asserted), strict_labels (holiday
  cut 2025-11-16, no information after it), TARGETS, drive_hashes (FOLD_RULE: 3 folds by
  crc32 of the serial), sample_training_rows (drive hash only, called with date_step=1) and
  correct_probability
  (p = q pi / (q pi + 1 - q), pi = kept / available negatives), frequent_model_map,
  design_matrix, feature_columns (all groups G0 G1 GO G2 G3), make_hgb / fit_hgb
  (early stopping on the drive-grouped inner split, INNER_RULE), fit_model (logistic),
  predict_proba_rows, evaluate_scores (per-date alert budgets).

Targets (--targets): bands b_0_7, b_7_30, b_30_60 (main formulation, aliases B1 B2 B3) and
the separate horizon y_30 as a reference. y_7 and y_60 are also accepted.

Models with FIXED hyperparameters (not tuned here; --hgb-* / --lr-C):
  hgb         HistGradientBoosting, learning rate 0.1, max_leaf_nodes 31 (no depth limit),
              min_samples_leaf 2000; trees by early stopping (cv_models GB_FIXED: cap 1000,
              patience 15, l2 1.0) on the inner validation drives.
  hgb_scaled  same, but min_samples_leaf = 2000 * n_train / n_ref (>= 20), n_ref = training
              rows of 15 % of negative drives x all dates for the same target and fold: the
              leaf holds the same share of the data (about the same expected positives) at
              every size, whereas a fixed leaf size is weaker regularisation at larger n.
              The value used is in lc_folds.csv.
  logreg      cv_models logistic pipeline, L2, C = 0.1 (signed log1p applied at load,
              median imputation + missing indicators, standardised, "> 0" indicators of the
              error counters, one-hot manufacturer / frequent model).

Validation: the 3 grouped folds of cv_models.py. Each fit uses the sample drawn from 2 folds
and is scored on ALL known-label rows of the held-out fold (no sampling), date by date.
Metrics per (model, target, size, fold): recall and precision with per-date budgets of 0.1 %
and 1 % (1 % selects), PR-AUC, ROC-AUC, log loss, mean predicted vs observed rate (after the
pi correction), fit / predict time, process peak memory, training rows / positives /
negatives, n_iter.

"Enough data" (per model / target): the smallest mean training size whose mean recall@1 % is
within one fold-sd of the best size (rule of 3.1 of the plan). "overall" = the smallest size
that satisfies this for every model / target evaluated. Sizes whose estimated fit time
exceeds --max-fit-minutes are skipped and flagged (status "skipped_time"), except the two
corners 15 % x all dates and 100 % x all dates (all data), which always run.

Outputs in <out dir> (small, meant to be committed):
  lc_folds.csv            one row per (model, target, size, fold); checkpoint for resume
  lc_summary.csv          mean and sd over folds per (model, target, size)
  lc_recommendation.json  chosen size per model / target and overall, with the numbers
  lc_curve.png            recall@1 % and PR-AUC vs training rows (log x), one column per target
  run_info.json           commit, machine, versions, args, seeds, cut, fold / sample rules
  lc_log.txt              log (appended across restarts)
Restarting with the same arguments skips the rows already in lc_folds.csv (and retries the
skipped ones). Other settings in the same out dir are refused. --quick (3 dates, 2 sizes)
writes to <out dir>/quick/. --report-only rebuilds the summary, JSON and figure.

Usage (from dataset/backblaze/):
  uv run python -I scripts/learning_curve.py <landmarks dir> <data dir> <out dir>
      [--jobs N] [--quick] [--models hgb,hgb_scaled,logreg] [--targets b_0_7,b_7_30,b_30_60,y_30]
      [--max-fit-minutes 30] [--cv-module scripts/cv_models.py] [--report-only]

Homelab command (unattended; run it before cv_models.py):
  cd dataset/backblaze && mkdir -p out/lc
  nohup uv run python -I scripts/learning_curve.py out/landmarks data out/lc --jobs $(nproc) \
      > out/lc/stdout.txt 2>&1 &
  (stdout.txt duplicates lc_log.txt plus any traceback; it need not be committed.)
Expected runtime: the script prints its own estimate at start and an ETA after every fit. The
full grid is 3 models x 4 targets x 21 sizes x 3 folds = 756 fits, ~279 M training rows per
model in total (largest sample 8.6 M rows). From local measurements (i5-8300H, 4 threads:
HGB ~10-19 us per training row, logistic ~18 us per row, ~20 s to score one held-out fold of
4.3 M rows): about 8 h with 4 threads, 5-6 h with 8; scoring the held-out folds is about half.
Memory: dev table ~2.3 GB (process ~3 GB after loading); full-data HGB fits peak ~9-10 GB;
the full-data logistic fit needs ~13-14 GB, so with the default --max-mem-gb 12 the 9
logistic fits at 100 % x all landmarks of b_0_7, b_7_30 and y_30 are skipped (status
skipped_memory). On a machine with >= 24 GB RAM add --max-mem-gb 16 to include them.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

FRACS = [0.01, 0.02, 0.05, 0.15, 0.30, 0.50, 1.00]   # share of negative drives kept
STEPS = [1, 3, 10]                     # landmark subset for ALL rows: date_pos % step == 0
QUICK_SIZES = [(0.15, 1), (1.00, 1)]
CORNERS = [(0.15, 1), (1.00, 1)]       # 15 % of negative drives, all data: always run
REF_SIZE = (0.15, 1)                   # n_ref of hgb_scaled
CALIB_FLAG = (0.67, 1.5)               # calibration-in-the-large outside this range is flagged
BUDGETS = (0.1, 1.0)                   # % of the rows of each date
MODELS_ALL = ["hgb", "hgb_scaled", "logreg"]
TARGET_ALIASES = {"B1": "b_0_7", "B2": "b_7_30", "B3": "b_30_60"}
DEFAULT_TARGETS = "b_0_7,b_7_30,b_30_60,y_30"

# Reference timing (i5-8300H, 4 cores, HGB 200 trees, y_30): 205 k rows 7 s, 1.9 M 45 s,
# 12.8 M 243 s  ->  ~3 s + 19 us per row. Only for the first estimate; refined on the fly.
REF_CORES = 4
# Full dev data probe (7.9 M rows, 4 threads): HGB 38 s (early stop at 15 trees), logistic 138 s.
REF_FIT = {"hgb": (3.0, 19e-6), "hgb_scaled": (3.0, 19e-6), "logreg": (3.0, 18e-6)}
REF_EVAL = {"hgb": 15.0, "hgb_scaled": 15.0, "logreg": 30.0}   # scoring one held-out fold
# Extra memory per training row during a fit, measured on the full dev data (7.9 M training
# rows, y_30): HGB +6.2 GB (incl. a copy that is no longer made here), logistic +9.8 GB
# (ColumnTransformer output plus its parts at the hstack). Used by the --max-mem-gb guard.
MEM_PER_ROW = {"hgb": 700.0, "hgb_scaled": 700.0, "logreg": 1300.0}   # bytes


# ---------------------------------------------------------------- utilities

class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *parts):
        msg = f"{datetime.now():%Y-%m-%d %H:%M:%S} " + " ".join(str(p) for p in parts)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def fmt_s(s):
    if s is None or not np.isfinite(s):
        return "?"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def peak_rss_gb():
    """Process peak resident memory so far (monotone over the run), NaN if unavailable."""
    try:
        import resource  # POSIX
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / (2**30 if sys.platform == "darwin" else 2**20), 2)
    except ImportError:
        pass
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)
    except ImportError:
        return float("nan")


def current_rss_gb():
    """Current resident memory of the process, NaN if unavailable (memory guard disabled)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except ImportError:
        pass
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, ValueError, AttributeError):
        return float("nan")


def load_module(path):
    """Import a module from an explicit file path (sys.path is not used under -I)."""
    spec = importlib.util.spec_from_file_location("cv_models", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cv_models"] = mod   # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def size_label(frac, step):
    return f"{round(frac * 100)}%x1/{step}"


def sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("landmarks_dir")
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="threads for HGB / BLAS (fits run one at a time to bound memory)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 dates, 2 sizes, out_dir/quick")
    ap.add_argument("--models", default=",".join(MODELS_ALL))
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--max-fit-minutes", type=float, default=30.0,
                    help="skip non-corner sizes whose estimated fit time exceeds this")
    ap.add_argument("--max-mem-gb", type=float, default=12.0,
                    help="skip fits (corners included) whose estimated peak memory exceeds this")
    ap.add_argument("--cv-module", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        "cv_models.py"))
    ap.add_argument("--report-only", action="store_true",
                    help="only rebuild lc_summary.csv, lc_recommendation.json and lc_curve.png")
    ap.add_argument("--hgb-lr", type=float, default=0.1)
    ap.add_argument("--hgb-leaves", type=int, default=31)
    ap.add_argument("--hgb-min-leaf", type=int, default=2000)
    ap.add_argument("--hgb-max-iter", type=int, default=None, help="default: cv_models GB_FIXED")
    ap.add_argument("--lr-C", type=float, default=0.1)
    a = ap.parse_args(argv)
    a.models = [m.strip() for m in a.models.split(",") if m.strip()]
    a.targets = [TARGET_ALIASES.get(t.strip(), t.strip()) for t in a.targets.split(",") if t.strip()]
    if bad := [m for m in a.models if m not in MODELS_ALL]:
        ap.error(f"unknown models {bad}; choose from {MODELS_ALL}")
    return a


# ---------------------------------------------------------------- timing

class TimeModel:
    """Per-model estimate of fit and scoring time. Reference timings until fits are observed,
    then a least-squares line fit_s = a + b n (b > 0) on the observed fits of that model."""

    def __init__(self, jobs):
        self.scale = REF_CORES / max(1, min(jobs, os.cpu_count() or jobs))
        self.fit, self.eval = {}, {}

    def add(self, model, n, fit_s, eval_s):
        self.fit.setdefault(model, []).append((n, fit_s))
        self.eval.setdefault(model, []).append(eval_s)

    def ref(self, model, n):
        a, b = REF_FIT[model]
        return (a + b * n) * self.scale

    def predict_fit(self, model, n):
        pts = self.fit.get(model, [])
        ns = [p[0] for p in pts]
        if pts and max(ns) >= 2 * min(ns):      # a line only over a wide enough range of n
            x, t = np.array(pts, dtype=float).T
            b, a = np.polyfit(x, t, 1)
            if b > 0:
                return max(a, 0.0) + b * n
        if pts:
            n1, t1 = max(pts)
            return t1 * self.ref(model, n) / self.ref(model, n1)
        return self.ref(model, n)

    def predict_eval(self, model):
        e = self.eval.get(model)
        return float(np.mean(e)) if e else REF_EVAL[model] * self.scale


# ---------------------------------------------------------------- reports

METRICS = ["n_train", "n_pos", "n_neg", "pi", "min_samples_leaf", "n_iter", "fit_s", "predict_s",
           "peak_rss_gb", "recall_01", "recall_1", "precision_01", "precision_1", "pr_auc",
           "roc_auc", "log_loss", "mean_pred", "obs_rate", "calib_ratio", "calib_flag",
           "mean_q_train", "train_pos_rate", "mean_q_eval"]
KEYS = ["model", "target", "frac", "step"]
FOLD_COLS = (KEYS + ["size", "fold", "corner", "status", "est_fit_s", "est_mem_gb", "n_train", "n_pos",
                     "n_neg", "n_val", "pi", "pi_nominal", "min_samples_leaf", "n_iter", "converged",
                     "fit_s", "predict_s", "peak_rss_gb", "n_eval", "n_eval_pos", "n_eval_dates",
                     "recall_01", "recall_1", "precision_01", "precision_1", "pr_auc", "roc_auc",
                     "log_loss", "mean_pred", "obs_rate", "calib_ratio", "calib_flag", "mean_q_train",
                     "train_pos_rate", "mean_q_eval", "finished"])


def append_row(path, row):
    """Append one result row with the fixed columns (checkpoint)."""
    assert set(row) <= set(FOLD_COLS), set(row) - set(FOLD_COLS)
    pd.DataFrame([{c: row.get(c, np.nan) for c in FOLD_COLS}]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False)


def summarize(folds, k_folds):
    ok = folds[folds.status == "ok"]
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby(KEYS, sort=False)
    s = g[METRICS].mean().add_suffix("_mean").join(g[METRICS].std(ddof=1).add_suffix("_sd"))
    s["n_folds"] = g.size()
    s = s.reset_index()
    s["size"] = [size_label(f, st) for f, st in zip(s.frac, s.step)]
    s["corner"] = [(f, st) in CORNERS for f, st in zip(s.frac, s.step)]
    skipped = folds[folds.status != "ok"].groupby(KEYS).size().rename("n_skipped").reset_index()
    s = s.merge(skipped, on=KEYS, how="left").fillna({"n_skipped": 0})
    # sizes skipped in every fold still get a line, so the gap is visible
    miss = skipped.merge(s[KEYS], on=KEYS, how="left", indicator=True)
    miss = miss[miss._merge == "left_only"].drop(columns="_merge")
    if len(miss):
        miss["size"] = [size_label(f, st) for f, st in zip(miss.frac, miss.step)]
        miss["corner"] = False
        miss["n_folds"] = 0
        s = pd.concat([s, miss], ignore_index=True)
    s["complete"] = s.n_folds == k_folds
    front = KEYS + ["size", "corner", "n_folds", "n_skipped", "complete"]
    s = s[front + [c for c in s.columns if c not in front]]
    return s.sort_values(["model", "target", "n_train_mean"]).reset_index(drop=True)


def _brief(r):
    return {"size": r["size"], "frac": float(r.frac), "step": int(r.step),
            "n_train": round(float(r.n_train_mean)), "recall_1": float(r.recall_1_mean),
            "recall_1_sd": float(r.recall_1_sd), "recall_01": float(r.recall_01_mean),
            "pr_auc": float(r.pr_auc_mean), "pr_auc_sd": float(r.pr_auc_sd),
            "log_loss": float(r.log_loss_mean), "calib_ratio": float(r.calib_ratio_mean),
            "fit_s": float(r.fit_s_mean), "n_iter": float(r.n_iter_mean),
            "share_folds_calib_flag": float(r.calib_flag_mean)}


def recommend(summary):
    full = summary[summary.complete]
    out = {"rule": "per model/target: smallest mean training rows whose mean recall@1% (per-date "
                   "budget) is >= best mean - fold sd of the best (3 drive-grouped folds); overall: "
                   "smallest size satisfying it for every model/target",
           "per_model_target": [], "overall": None,
           "skipped_sizes": summary.loc[~summary.complete, ["model", "target", "size"]]
                                   .to_dict("records")}
    within_sets = []
    for (m, t), sub in full.groupby(["model", "target"], sort=False):
        best = sub.loc[sub.recall_1_mean.idxmax()]
        thr = best.recall_1_mean - (best.recall_1_sd if np.isfinite(best.recall_1_sd) else 0.0)
        within = sub[sub.recall_1_mean >= thr].sort_values("n_train_mean")
        chosen = within.iloc[0]
        rec = {"model": m, "target": t, "threshold_recall_1": float(thr),
               "best": _brief(best), "chosen": _brief(chosen),
               "sizes_within": list(within["size"]),
               "fit_s_saved_vs_best": float(best.fit_s_mean - chosen.fit_s_mean)}
        for name, (f, st) in (("negative_drives_15pct", CORNERS[0]), ("all_data", CORNERS[1])):
            r = sub[(sub.frac == f) & (sub.step == st)]
            if len(r):
                rec[name] = _brief(r.iloc[0]) | {"within_threshold": bool(r.iloc[0].recall_1_mean >= thr)}
        out["per_model_target"].append(rec)
        within_sets.append(set(zip(within.frac, within.step)))
    if within_sets:
        common = set.intersection(*within_sets)
        if common:
            size_n = full.groupby(["frac", "step"]).n_train_mean.mean()
            f, st = min(common, key=lambda fs: size_n.loc[fs])
            fit = full[(full.frac == f) & (full.step == st)].groupby("model").fit_s_mean.mean()
            out["overall"] = {"size": size_label(f, st), "frac": float(f), "step": int(st),
                              "mean_n_train": round(float(size_n.loc[(f, st)])),
                              "mean_fit_s_by_model": {k: float(v) for k, v in fit.items()}}
        else:
            out["overall"] = {"size": None, "note": "no single size is within one fold-sd of the "
                                                    "best for every model/target"}
    return out


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    if summary.empty:
        return
    s = summary[summary.n_folds > 0]
    if s.empty:
        return
    targets = list(dict.fromkeys(s.target))
    colors = {"hgb": "#2a78d6", "hgb_scaled": "#eb6834", "logreg": "#1baf7a"}
    markers = {10: "o", 3: "s", 1: "^"}
    styles = {10: ":", 3: "--", 1: "-"}
    fig, axes = plt.subplots(2, len(targets), figsize=(3.4 * len(targets) + 1.0, 6.6),
                             squeeze=False, sharex=True)
    for j, t in enumerate(targets):
        for i, (metric, label) in enumerate([("recall_1", "recall, 1 % alerts per date"),
                                             ("pr_auc", "PR-AUC")]):
            ax = axes[i, j]
            for (m, st), sub in s[s.target == t].groupby(["model", "step"]):
                sub = sub.sort_values("n_train_mean")
                ax.errorbar(sub.n_train_mean, sub[f"{metric}_mean"], yerr=sub[f"{metric}_sd"].fillna(0),
                            color=colors.get(m, "#555555"), marker=markers.get(st, "o"), ms=5,
                            ls=styles.get(st, "-"), lw=1.5, capsize=2,
                            label=f"{m}, {'all landmarks' if st == 1 else f'1 in {st} landmarks'}")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(True, color="#dddddd", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(t)
            else:
                ax.set_xlabel("training rows")
            if j == 0:
                ax.set_ylabel(label)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    ncol = 3
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=ncol, frameon=False, fontsize=8)
    fig.suptitle("Learning curve: development landmarks, 3 drive-grouped folds (mean and fold sd)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.03 + 0.03 * math.ceil(len(handles) / ncol), 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_reports(out_dir, k_folds, log):
    folds = pd.read_csv(os.path.join(out_dir, "lc_folds.csv"))
    ok = folds[folds.status == "ok"]
    flagged = ok[ok.calib_flag.astype(bool)]
    log(f"calibration-in-the-large: {len(flagged)} of {len(ok)} fits outside {CALIB_FLAG} "
        f"(mean corrected p / observed rate on the held-out fold)"
        + ("" if flagged.empty else "; e.g. " + ", ".join(
            f"{r.model} {r.target} {r.size} f{r.fold}: {r.calib_ratio:.2f}" for r in flagged.head(8).itertuples())))
    folds["calib_flag"] = folds["calib_flag"].astype(float)
    summary = summarize(folds, k_folds)
    summary.to_csv(os.path.join(out_dir, "lc_summary.csv"), index=False, float_format="%.6g")
    rec = recommend(summary) if not summary.empty else {}
    write_json(os.path.join(out_dir, "lc_recommendation.json"), rec)
    plot(summary, os.path.join(out_dir, "lc_curve.png"))
    for r in rec.get("per_model_target", []):
        log(f"  {r['model']:<10} {r['target']:<8} best {r['best']['size']:<9} rec@1% "
            f"{r['best']['recall_1']:.3f} (sd {r['best']['recall_1_sd']:.3f}, fit {r['best']['fit_s']:.0f}s)"
            f" -> enough {r['chosen']['size']:<9} rec@1% {r['chosen']['recall_1']:.3f} "
            f"(fit {r['chosen']['fit_s']:.0f}s)")
    if rec.get("overall"):
        log("  overall:", rec["overall"].get("size"))
    return summary, rec


# ---------------------------------------------------------------- machine / provenance

def machine_info():
    info = {"os": platform.platform(), "python": sys.version.split()[0],
            "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                info["cpu"] = next(x.split(":", 1)[1].strip() for x in f if x.startswith("model name"))
        except (OSError, StopIteration):
            pass
    return info


def git_info(start):
    def run(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=start, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("rev-parse", "HEAD") or None, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked": run("status", "--porcelain", "--untracked-files=no").splitlines()}


def versions():
    from importlib.metadata import version
    out = {"python": sys.version.split()[0]}
    for p in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "matplotlib", "threadpoolctl"):
        try:
            out[p] = version(p)
        except Exception:
            out[p] = None
    return out


# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cv = load_module(args.cv_module)
    if bad := [t for t in args.targets if t not in cv.TARGETS]:
        sys.exit(f"unknown targets {bad}; choose from {list(cv.TARGETS)} or {list(TARGET_ALIASES)}")
    out = os.path.join(args.out_dir, "quick") if args.quick else args.out_dir
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "lc_log.txt"))
    folds_path = os.path.join(out, "lc_folds.csv")
    if args.report_only:
        write_reports(out, cv.K_FOLDS, log)
        return

    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=args.jobs)

    hgb_fixed = dict(cv.GB_FIXED)
    if args.hgb_max_iter is not None:
        hgb_fixed["max_iter"] = args.hgb_max_iter
    hgb_params = {"learning_rate": args.hgb_lr, "max_leaf_nodes": args.hgb_leaves, "max_depth": None,
                  "min_samples_leaf": args.hgb_min_leaf, **hgb_fixed}
    lr_params = {"penalty": "l2", "C": args.lr_C}
    sizes = QUICK_SIZES if args.quick else [(f, s) for s in STEPS for f in FRACS]
    # settings that define the results: a resume must use the same ones
    settings = {"quick": args.quick, "cutoff": str(cv.CUTOFF), "k_folds": cv.K_FOLDS,
                "fold_rule": cv.FOLD_RULE, "inner_rule": cv.INNER_RULE,
                "sample_rule": "landmarks with date_pos % date_step == 0 for positives AND negatives; "
                               "on them all positives + negatives of drives with "
                               "crc32('negative-sample:'+serial)/2**32 < neg_drive_frac "
                               "(cv_models.sample_training_rows with date_step=1)",
                "hgb": hgb_params, "hgb_scaled": f"min_samples_leaf = {args.hgb_min_leaf} * n_train / "
                                                 f"n_train({size_label(*REF_SIZE)}), >= 20",
                "logreg": lr_params, "budgets_pct": list(BUDGETS),
                "quick_dates": [str(d) for d in cv.QUICK_DATES] if args.quick else None}
    info_path = os.path.join(out, "run_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        if info.get("settings") != settings:
            sys.exit(f"{out} holds a run with other settings; use another out dir.\n"
                     f"old: {info.get('settings')}\nnew: {settings}")
    here = os.path.dirname(os.path.abspath(__file__))
    info.setdefault("runs", []).append({"start": datetime.now().isoformat(timespec="seconds"),
                                        "argv": sys.argv, "args": vars(args)})
    info.update({
        "settings": settings, "git": git_info(here), "machine": machine_info() | {"ram_gb": cv._ram_gb()},
        "versions": versions(), "cv_module": {"path": os.path.abspath(args.cv_module),
                                              "sha256": sha256(args.cv_module)},
        "seeds": {"random_state": cv.SEED, "hashes": "zlib.crc32 with salts fold / negative-sample / "
                                                     "date-phase / inner-validation (cv_models.py)"},
        "grid": {"neg_drive_frac": FRACS, "date_step": STEPS, "sizes": [size_label(*s) for s in sizes],
                 "corners_always_run": [size_label(*c) for c in CORNERS]},
        "max_fit_minutes": args.max_fit_minutes, "targets": args.targets, "models": args.models,
        "selection": "recall@1% mean over folds; enough data = smallest size within one fold sd of the best",
        "peak_rss_note": "peak_rss_gb is the process peak so far (monotone), not per fit",
    })
    write_json(info_path, info)
    log(f"start: models {args.models}, targets {args.targets}, jobs {args.jobs}, quick {args.quick}, "
        f"cutoff {cv.CUTOFF}, cv_models {args.cv_module}")

    # --- data (all definitions from cv_models.py)
    t_load = time.time()
    status = cv.load_drive_status(args.data_dir, cv.CUTOFF)
    data = cv.load_dev_landmarks(args.landmarks_dir, status.index,
                                 cv.QUICK_DATES if args.quick else None, log=lambda *a: None)
    first, last = cv.day_index(cv.DEV_FIRST), cv.day_index(cv.DEV_LAST)
    assert first <= data.day.min() and data.day.max() <= last, "rows outside the dev period"
    labels = cv.strict_labels(data, status, {t: cv.TARGETS[t] for t in args.targets})
    hashes = cv.drive_hashes(status.serial)
    fold = hashes.fold[data.drive]
    features = cv.feature_columns()
    counts = {t: {"known": int((~np.isnan(y)).sum()), "pos": int(np.nansum(y)),
                  "dates": int(len(np.unique(data.day[~np.isnan(y)])))} for t, y in labels.items()}
    log(f"loaded {len(data.day)} dev rows, {len(data.dates)} dates ({data.dates[0]} .. {data.dates[-1]}), "
        f"X {data.X.shape} {data.X.nbytes / 2**30:.2f} GB in {time.time() - t_load:.0f}s; peak RSS "
        f"{peak_rss_gb()} GB")
    for t, c in counts.items():
        log(f"  target {t}: known {c['known']}, positives {c['pos']}, dates {c['dates']}")
    info.update({"rows": int(len(data.day)), "dates": [str(d) for d in data.dates],
                 "target_counts": counts, "features": features,
                 "fold_sizes": {int(k): int((fold == k).sum()) for k in range(cv.K_FOLDS)}})
    write_json(info_path, info)

    # --- training sizes of every task (cheap: masks only), for n_ref and the estimate
    n_train = {}
    def sample(y, k, f, st):
        """Training rows of fold k: landmark subset (date_pos % st == 0) for positives AND
        negatives, then all positives + negatives of the drives kept by cv_models' hash
        (date_step=1 there: thinning negatives only by date leaks the label through the date
        features). pi = kept / available negatives on those landmarks."""
        mask = (fold != k) & (data.date_pos.astype(np.int64) % st == 0)
        return cv.sample_training_rows(y, mask, data.drive, data.date_pos, hashes,
                                       neg_drive_frac=f, date_step=1)

    for t in args.targets:
        for k in range(cv.K_FOLDS):
            for f, st in set(sizes) | {REF_SIZE}:
                n_train[(t, k, f, st)] = len(sample(labels[t], k, f, st)[0])
    tasks = []
    for m in args.models:
        for t in args.targets:
            for f, st in sorted(sizes, key=lambda s: n_train[(t, 0, *s)]):
                for k in range(cv.K_FOLDS):
                    tasks.append((m, t, f, st, k))

    # --- resume
    done = set()
    if os.path.exists(folds_path):
        prev = pd.read_csv(folds_path)
        keep = prev[prev.status == "ok"]
        if len(keep) < len(prev):          # retry skipped rows (the limit may have changed)
            keep.to_csv(folds_path + ".tmp", index=False)
            os.replace(folds_path + ".tmp", folds_path)
        done = {(r.model, r.target, float(r.frac), int(r.step), int(r.fold)) for r in keep.itertuples()}
    todo = [x for x in tasks if (x[0], x[1], float(x[2]), int(x[3]), int(x[4])) not in done]

    tm = TimeModel(args.jobs)
    if os.path.exists(folds_path) and done:
        for r in pd.read_csv(folds_path).itertuples():
            tm.add(r.model, r.n_train, r.fit_s, r.predict_s)
    limit_s = args.max_fit_minutes * 60
    base_gb = current_rss_gb()

    def estimate(task):
        m, t, f, st, k = task
        return tm.predict_fit(m, n_train[(t, k, f, st)])

    def mem_estimate(task):
        m, t, f, st, k = task
        return base_gb + MEM_PER_ROW[m] * n_train[(t, k, f, st)] / 2**30

    def why_skip(task):
        if not mem_estimate(task) <= args.max_mem_gb and np.isfinite(base_gb):
            return "skipped_memory"
        if (task[2], task[3]) not in CORNERS and estimate(task) > limit_s:
            return "skipped_time"
        return None

    def runs(task):
        return why_skip(task) is None

    def eta(rest):
        return sum(estimate(x) + tm.predict_eval(x[0]) for x in rest if runs(x))

    n_skip_t = sum(why_skip(x) == "skipped_time" for x in todo)
    n_skip_m = sum(why_skip(x) == "skipped_memory" for x in todo)
    log(f"fits: {len(tasks)} total, {len(todo)} to do; expected skips: {n_skip_t} over the "
        f"{args.max_fit_minutes:g}-min limit, {n_skip_m} over the {args.max_mem_gb:g}-GB limit "
        f"(RSS now {base_gb:.1f} GB); estimate {fmt_s(eta(todo))} "
        f"(reference timings scaled to {args.jobs} threads; refined after each fit)")
    big = max(n_train.values())
    log(f"largest training sample {big} rows; all data corner per target: "
        + ", ".join(f"{t} {n_train[(t, 0, 1.0, 1)]}" for t in args.targets if (t, 0, 1.0, 1) in n_train))

    # --- fits
    model_maps = {}
    t_run = time.time()
    for i, task in enumerate(todo):
        m, t, f, st, k = task
        n = n_train[(t, k, f, st)]
        row = {"model": m, "target": t, "frac": f, "step": st, "size": size_label(f, st), "fold": k,
               "corner": (f, st) in CORNERS, "est_fit_s": round(estimate(task), 1)}
        row["est_mem_gb"] = round(mem_estimate(task), 1)
        if (why := why_skip(task)) is not None:
            row.update(status=why, n_train=n)
            log(f"[{i + 1}/{len(todo)}] {m} {t} {row['size']} f{k}: SKIPPED ({why}): estimated fit "
                f"{fmt_s(row['est_fit_s'])} (limit {args.max_fit_minutes:g} min), memory "
                f"{row['est_mem_gb']} GB (limit {args.max_mem_gb:g} GB), n_train {n}")
            append_row(folds_path, row)
            continue
        if k not in model_maps:
            model_maps[k] = cv.frequent_model_map(data, np.flatnonzero(fold != k))[0]
        mmap = model_maps[k]
        y = labels[t]
        rows, pi = sample(y, k, f, st)
        assert len(rows) == n
        ytr = y[rows].astype(np.int8)
        val = hashes.inner[data.drive[rows]]
        msl = args.hgb_min_leaf
        if m == "hgb_scaled":
            msl = max(20, int(round(args.hgb_min_leaf * n / n_train[(t, k, *REF_SIZE)])))
        t0 = time.time()
        if m.startswith("hgb"):
            model = cv.make_hgb({"learning_rate": args.hgb_lr, "max_depth": None, "min_samples_leaf": msl},
                                features)
            model.set_params(max_leaf_nodes=args.hgb_leaves, **hgb_fixed)
            # same as cv.fit_hgb(model, X, y, val), but the two design matrices are built
            # directly instead of slicing a full one (saves one copy of the training matrix)
            Xa = cv.design_matrix(data, rows[~val], mmap, features)
            Xv = cv.design_matrix(data, rows[val], mmap, features)
            model.fit(Xa, ytr[~val], X_val=Xv, y_val=ytr[val])
            del Xa, Xv
            finfo = {"n_iter": int(model.n_iter_), "converged": True, "warnings": ""}
        else:
            msl = np.nan
            Xtr = cv.design_matrix(data, rows, mmap, features)
            model, finfo = cv.fit_model("lr", lr_params, Xtr, ytr, val, features)
            del Xtr
        fit_s = time.time() - t0
        # leak check: mean uncorrected q on (up to 200 k rows of) the training sample vs held out
        sub = np.sort(np.random.default_rng(cv.SEED).permutation(n)[:200_000])
        mean_q_train = float(model.predict_proba(cv.design_matrix(data, rows[sub], mmap, features))[:, 1].mean())
        t1 = time.time()
        ev = np.flatnonzero((fold == k) & ~np.isnan(y))
        q = cv.predict_proba_rows(model, data, ev, mmap, features)
        del model
        p = cv.correct_probability(q, pi)
        met = cv.evaluate_scores(y[ev], p, data.day[ev], budgets=BUDGETS)
        predict_s = time.time() - t1
        n_pos = int(ytr.sum())
        row.update(status="ok", n_train=n, n_pos=n_pos, n_neg=n - n_pos, n_val=int(val.sum()),
                   pi=pi, pi_nominal=f, min_samples_leaf=msl, n_iter=finfo["n_iter"],
                   converged=finfo["converged"], fit_s=round(fit_s, 2), predict_s=round(predict_s, 2),
                   peak_rss_gb=peak_rss_gb(), n_eval=len(ev), n_eval_pos=int(y[ev].sum()),
                   n_eval_dates=int(len(np.unique(data.day[ev]))),
                   recall_01=met[cv.budget_key(0.1, "recall")], recall_1=met[cv.budget_key(1.0, "recall")],
                   precision_01=met[cv.budget_key(0.1, "precision")],
                   precision_1=met[cv.budget_key(1.0, "precision")],
                   pr_auc=met["pr_auc"], roc_auc=met["roc_auc"], log_loss=met["log_loss"],
                   mean_pred=met["mean_pred"], obs_rate=met["obs_rate"], calib_ratio=met["calib_ratio"],
                   mean_q_train=mean_q_train, train_pos_rate=n_pos / n, mean_q_eval=float(q.mean()),
                   finished=datetime.now().isoformat(timespec="seconds"))
        cr = row["calib_ratio"]
        row["calib_flag"] = bool(not np.isfinite(cr) or cr < CALIB_FLAG[0] or cr > CALIB_FLAG[1])
        append_row(folds_path, row)
        tm.add(m, n, fit_s, predict_s)
        rest = todo[i + 1:]
        log(f"[{i + 1}/{len(todo)}] {m} {t} {row['size']} f{k}: n_train {n} (pos {n_pos}, pi {pi:.4f}"
            f"{'' if np.isnan(msl) else f', leaf {msl}'}) fit {fit_s:.0f}s (est {row['est_fit_s']:.0f}s) "
            f"pred {predict_s:.0f}s n_iter {finfo['n_iter']}{'' if finfo['converged'] else ' NOT CONVERGED'}"
            f" | rec@1% {row['recall_1']:.3f} rec@0.1% {row['recall_01']:.3f} PR {row['pr_auc']:.4f} "
            f"calib {row['calib_ratio']:.2f}{' FLAG' if row['calib_flag'] else ''} (q train "
            f"{mean_q_train:.4f} vs pos rate {n_pos / n:.4f}, q held-out {row['mean_q_eval']:.4f}) | RSS peak {row['peak_rss_gb']} GB | "
            f"elapsed {fmt_s(time.time() - t_run)}, ETA {fmt_s(eta(rest))}")

    # --- reports
    log("all fits done; writing reports")
    write_reports(out, cv.K_FOLDS, log)
    info["runs"][-1]["end"] = datetime.now().isoformat(timespec="seconds")
    info["runs"][-1]["peak_rss_gb"] = peak_rss_gb()
    write_json(info_path, info)
    log("done")


if __name__ == "__main__":
    main()
