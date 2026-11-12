"""Per-horizon failure models on the Backblaze landmark table.

Training: all failures plus a random sample of non-failures (fraction r); the
probabilities are corrected back with odds * r. Evaluation: GroupKFold by drive,
on all test failures plus a weighted sample of test non-failures.
Usage: python -I model.py <landmarks.parquet> <output dir>
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
d = pd.read_parquet(sys.argv[1])
out = sys.argv[2]
H = [7, 30, 60]
ERR = [5, 187, 188, 197, 198]

# Features
smart = [c for c in d.columns if c.startswith("smart_") and d[c].isna().mean() < 0.9]
for c in smart:
    d[c] = np.log1p(d[c].clip(lower=0))
for i in ERR:
    d[f"d7_{i}"] = np.sign(d[f"d7_{i}"]) * np.log1p(d[f"d7_{i}"].abs())
d["capacity_tb"] = d.capacity_bytes.where(d.capacity_bytes > 0) / 1e12
top = d.model.value_counts().index[:15]
d["model_g"] = d.model.where(d.model.isin(top), "other")
num_all = smart + [f"d7_{i}" for i in ERR] + ["capacity_tb"]
err_cols = [f"smart_{i}_raw" for i in ERR] + [f"d7_{i}" for i in ERR]
d["rule"] = (d[[f"smart_{i}_raw" for i in ERR]].fillna(0) > 0).any(axis=1).astype(float)


def prep(num):
    return ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("c", OneHotEncoder(handle_unknown="ignore"), ["model_g"]),
    ], sparse_threshold=0)


MODELS = {
    "logistic": lambda: LogisticRegression(C=0.1, max_iter=3000),
    "knn50": lambda: KNeighborsClassifier(50),
    "gaussNB": lambda: GaussianNB(),
    "tree": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=0),
    "rf": lambda: RandomForestClassifier(300, min_samples_leaf=3, n_jobs=4, random_state=0),
    "svm_rbf": lambda: SVC(C=1.0, random_state=0),
}
NEG_TRAIN = {"svm_rbf": 15000}
NEG_TRAIN_DEFAULT = 40000
NEG_TEST = 100000


def score(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def top_budget(y, s, w, frac=0.01):
    o = np.argsort(-s)
    cw = np.cumsum(w[o])
    k = np.searchsorted(cw, frac * w.sum()) + 1
    sel = o[:k]
    tp = (y[sel] * w[sel]).sum()
    return tp / (y * w).sum(), tp / w[sel].sum()


res, oof = [], {h: {} for h in H}
gkf = GroupKFold(3)
for h in H:
    k = d[d[f"y_{h}"].notna()].reset_index(drop=True)
    y = k[f"y_{h}"].to_numpy()
    for fold, (tr, te) in enumerate(gkf.split(k, y, groups=k.serial_number)):
        pos_te, neg_te = te[y[te] == 1], te[y[te] == 0]
        neg_te_s = rng.choice(neg_te, min(NEG_TEST, len(neg_te)), replace=False)
        ev = np.concatenate([pos_te, neg_te_s])
        w = np.where(y[ev] == 1, 1.0, len(neg_te) / len(neg_te_s))
        rr, rp = top_budget(y[ev], k.rule.to_numpy()[ev] + rng.random(len(ev)) * 1e-6, w)
        r_flag = k.rule.to_numpy()[ev]
        res.append(dict(h=h, fold=fold, model="rule_err>0", features="all",
                        roc=roc_auc_score(y[ev], r_flag, sample_weight=w),
                        pr=average_precision_score(y[ev], r_flag, sample_weight=w),
                        rec1=rr, prec1=rp, prev=(y[ev] * w).sum() / w.sum()))
        pos_tr, neg_tr = tr[y[tr] == 1], tr[y[tr] == 0]
        for name, mk in MODELS.items():
            for feats, num in [("all", num_all), ("no_err", [c for c in num_all if c not in err_cols])]:
                if feats == "no_err" and name not in ("logistic", "rf"):
                    continue
                n_neg = NEG_TRAIN.get(name, NEG_TRAIN_DEFAULT)
                neg_s = rng.choice(neg_tr, min(n_neg, len(neg_tr)), replace=False)
                fit = np.concatenate([pos_tr, neg_s])
                r = len(neg_s) / len(neg_tr)
                est = make_pipeline(prep(num), mk()).fit(k.loc[fit, num + ["model_g"]], y[fit])
                s = score(est, k.loc[ev, num + ["model_g"]])
                rec, prec = top_budget(y[ev], s, w)
                res.append(dict(h=h, fold=fold, model=name, features=feats,
                                roc=roc_auc_score(y[ev], s, sample_weight=w),
                                pr=average_precision_score(y[ev], s, sample_weight=w),
                                rec1=rec, prec1=prec, prev=(y[ev] * w).sum() / w.sum()))
                if feats == "all" and name in ("logistic", "rf"):
                    q = score(est, d.loc[d.serial_number.isin(k.serial_number[te]), num + ["model_g"]])
                    idx = d.index[d.serial_number.isin(k.serial_number[te])]
                    p = q * r / (q * r + 1 - q)
                    oof[h].setdefault(name, pd.Series(np.nan, index=d.index)).loc[idx] = p
                if feats == "all" and name == "rf" and h == 30 and fold == 0:
                    sub = rng.choice(ev, min(30000, len(ev)), replace=False)
                    sub = np.union1d(sub, pos_te)
                    Xs = k.loc[sub, num + ["model_g"]]
                    pi = permutation_importance(est, Xs, y[sub], scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=1)
                    imp = pd.Series(pi.importances_mean, index=Xs.columns).sort_values(ascending=False)
                    imp.to_csv(f"{out}/perm_importance_rf_t30.csv")
                print(h, fold, name, feats, f"roc {res[-1]['roc']:.3f} pr {res[-1]['pr']:.4f} rec@1% {res[-1]['rec1']:.3f}", flush=True)

R = pd.DataFrame(res)
R.to_csv(f"{out}/results_folds.csv", index=False)
S = R.groupby(["h", "model", "features"])[["roc", "pr", "rec1", "prec1", "prev"]].agg(["mean", "std"]).round(4)
S.to_csv(f"{out}/results_summary.csv")
print(S.to_string())

# Cumulative probability vs Kaplan-Meier, per landmark cohort
def km(t_obs, event, t):
    s = 1.0
    for u in np.unique(t_obs[(event == 1) & (t_obs <= t)]):
        at_risk = (t_obs >= u).sum()
        s *= 1 - ((t_obs == u) & (event == 1)).sum() / at_risk
    return 1 - s


rows = []
for L, g in d.groupby("landmark"):
    for h in H:
        row = dict(landmark=str(L.date()), h=h, km=km(g.t_obs.to_numpy(), g.event.to_numpy(), h),
                   naive=g[f"y_{h}"].mean())
        for m in ("logistic", "rf"):
            row[m] = oof[h][m].loc[g.index].mean()
        rows.append(row)
C = pd.DataFrame(rows)
C.to_csv(f"{out}/cumulative_vs_km.csv", index=False)
print(C.groupby("h")[["km", "naive", "logistic", "rf"]].mean().round(5).to_string())
for m in ("logistic", "rf"):
    P = pd.concat([oof[h][m] for h in H], axis=1)
    viol = ((P.iloc[:, 0] > P.iloc[:, 1]) | (P.iloc[:, 1] > P.iloc[:, 2])).mean()
    print(m, "share of drives with non-monotonic cumulative curve:", round(viol, 3))
