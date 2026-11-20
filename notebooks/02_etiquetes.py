import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    import time

    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return (
        FunctionTransformer,
        GroupKFold,
        HistGradientBoostingClassifier,
        LogisticRegression,
        Pipeline,
        SimpleImputer,
        StandardScaler,
        average_precision_score,
        ds,
        glob,
        mdates,
        mo,
        np,
        pa,
        pd,
        plt,
        pq,
        roc_auc_score,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 · L'etiqueta és una decisió humana: què vol dir i quant ens costa

    Continuació de `01_dades.py`. Allà vam veure que les fallades registrades cauen per festes
    mentre els discs es continuen deteriorant igual. Aquí:

    1. Expliquem què vol dir cada casella de la matriu de confusió en aquest problema.
    2. Repassem les proves que l'etiqueta la posa una persona i com evitem arrossegar l'error.
    3. **Ho mesurem** amb models simples (regressió logística i gradient boosting) en quatre
       experiments.

    **Regla:** com al notebook 01, només fem servir dades fins al **4-1-2026**. Les dates de test
    (gener–juny de 2026) no es toquen.

    *Temps d'execució:* la primera vegada construeix la mostra llegint uns 120 dies del panel
    (1–2 min) i la desa a `out/label_study.parquet`. Els models triguen uns minuts més.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Què vol dir cada casella

    El model dona una puntuació de risc a cada disc en una data de referència L. Amb un pressupost
    d'alertes (aquí l'**1 %** dels discs de cada data) avisem dels discs amb més risc. **Positiu = el
    model avisa.** L'etiqueta és «el disc consta com a fallat en els t dies següents».

    | | **Consta com a fallat dins de t dies** | **No consta com a fallat** |
    | --- | --- | --- |
    | **Alerta** | **TP:** l'alerta era bona; s'hauria pogut canviar a temps | **FP:** alerta per a un disc que no consta com a fallat. Cost: una revisió o un canvi innecessari |
    | **Sense alerta** | **FN:** no l'hem vist venir. Cost: reconstrucció d'urgència, risc de perdre dades | **TN:** cap alerta i el disc continua bé (el 99,8 % dels casos) |

    Per això no fem servir l'accuracy: no avisar mai encerta el 99,8 %. Mesurem **quina part de les
    fallades capturem amb l'1 % d'alertes** (*recall@1 %*), la PR-AUC i la ROC-AUC.

    ### El problema: «consta com a fallat» no és «s'ha trencat»

    Backblaze marca un disc com a fallat quan el retira, i en part ho decideix mirant els comptadors
    SMART 5, 187, 188, 197 i 198, que també són variables del model. Per tant:

    - **Alguns FP són avisos bons:** el disc es retira més tard (fora de l'horitzó o després de festes).
    - **Alguns TP són fàcils:** el model endevina la decisió del tècnic mirant els mateixos comptadors.
    - **Els FN durs** són els discs que moren de cop, sense errors previs.

    ### Proves que l'etiqueta és humana (notebook 01)

    - Del 17-11-2025 al 4-1-2026 es registren un 57 % menys de fallades, però les pujades de comptadors
      d'error augmenten un 10 %.
    - En diumenge i dilluns es registren un 35 % menys de fallades; les pujades de comptadors són
      iguals tots els dies.
    - Al desembre, només un 1 % dels discs amb errors nous consten com a fallats en 14 dies (2 % a la
      tardor).

    ### Com evitem arrossegar l'error

    1. **Treure l'error conegut:** no entrenar amb etiquetes que passen per festes (censura al 16-11-2025).
    2. **Acceptar el que és inevitable:** el model aprèn «probabilitat que Backblaze el retiri». Si el
       criteri no canvia, és exactament el que passarà al futur, i per al manteniment ja és útil.
    3. **Mesurar quant depèn de la regla humana:** fallades netes (sense errors) i brutes, i un
       model sense els 5 comptadors.
    4. **Comprovar-ho al futur:** test separat en el temps (al final del projecte).
    5. **Vigilar la calibració:** les probabilitats es poden desajustar si canvia la taxa de fallades.

    Els experiments següents posen números als punts 1 i 3, i a dues preguntes més: quants FP són
    avisos anticipats, i quant error d'etiqueta tolera el model.
    """)
    return


@app.cell
def _(mo, pd):
    BASE = mo.notebook_dir().parent / "dataset" / "backblaze"
    CUTOFF = pd.Timestamp("2026-01-04")     # last day used by development labels
    CLEAN_END = pd.Timestamp("2025-11-16")  # holiday season starts on 17-11-2025 (notebook 01)
    DEV_LANDMARKS = pd.date_range("2025-02-05", "2025-11-05", freq="7D")
    HORIZONS = [7, 30, 60]
    ERR = [5, 187, 188, 197, 198]  # counters Backblaze uses to judge failure
    OTHER = [1, 7, 9, 10, 12, 183, 184, 189, 190, 192, 193, 194, 199, 240, 241, 242]
    NEG_RATE = 0.05   # share of never-failing drives kept in the study sample
    BUDGET = 0.01     # alert budget: 1 % of the drives of each landmark
    STUDY_PATH = BASE / "out" / "label_study.parquet"
    return (
        BASE,
        BUDGET,
        CLEAN_END,
        CUTOFF,
        DEV_LANDMARKS,
        ERR,
        HORIZONS,
        NEG_RATE,
        OTHER,
        STUDY_PATH,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Mostra d'estudi

    - **Dates de referència:** els 40 dimecres de desenvolupament (5-2 … 5-11-2025).
    - **Discs:** tots els que fallen fins al 4-1-2026 i un 5 % de la resta, triats amb un hash del
      número de sèrie (igual a qualsevol màquina). Els discs de la mostra pesen 1/0,05 = 20; els que
      fallen, 1. Amb aquests pesos, mètriques i probabilitats representen el parc sencer.
    - **Variables:** els 21 SMART de `build.py`, els canvis dels 5 comptadors d'error a 7 i 30 dies, la
      capacitat i el model.
    - **Etiquetes:** `y_t` = 1 si falla dins de (L, L+t]; NaN si surt de les dades abans sense fallar.
      Una etiqueta és **neta** si la finestra acaba abans de les festes (L + t ≤ 16-11-2025).
    - **Fallada neta o bruta:** bruta si el dia de la fallada algun dels 5 comptadors és > 0
      (probablement retirada proactiva), neta si no (probablement el disc va deixar de funcionar).
    """)
    return


@app.cell
def _(
    BASE,
    CUTOFF,
    DEV_LANDMARKS,
    ERR,
    HORIZONS,
    NEG_RATE,
    OTHER,
    STUDY_PATH,
    ds,
    glob,
    np,
    pa,
    pd,
    pq,
    time,
):
    def build_study():
        _files = sorted(glob.glob(str(BASE / "data" / "panel" / "*.parquet")))
        _panel = ds.dataset(_files, format="parquet",
                            schema=pa.unify_schemas([pq.read_schema(f) for f in _files]))
        _drives = pd.read_parquet(BASE / "data" / "drives.parquet",
                                  columns=["serial_number", "fail_date", "last_seen"])
        _drives["serial_number"] = _drives.serial_number.astype(str)
        # sampling decided with information up to the cutoff only
        _fails_dev = _drives.fail_date.notna() & (_drives.fail_date <= CUTOFF)
        _h = pd.util.hash_pandas_object(_drives.serial_number, index=False).to_numpy() % 10_000
        _keep = _fails_dev.to_numpy() | (_h < NEG_RATE * 10_000)
        _sel = _drives[_keep].copy()
        _sel["weight"] = np.where(_fails_dev[_keep], 1.0, 1.0 / NEG_RATE)
        _serials = pa.array(_sel.serial_number.tolist())
        _smart = [f"smart_{i}_raw" for i in ERR + OTHER]
        _errc = [f"smart_{i}_raw" for i in ERR]

        def _read(day, cols):
            _t = _panel.to_table(columns=cols, filter=(ds.field("date") == day.date())
                                 & ds.field("serial_number").isin(_serials))
            _d = _t.to_pandas()
            _d["serial_number"] = _d.serial_number.astype(str)
            return _d

        _rows = []
        _t0 = time.time()
        for _L in DEV_LANDMARKS:
            _now = _read(_L, ["serial_number", "model", "capacity_bytes", "failure"] + _smart)
            _now["model"] = _now.model.astype(str)
            _now = _now[_now.failure == 0].drop(columns="failure")
            for _lag in (7, 30):
                _prev = _read(_L - pd.Timedelta(days=_lag), ["serial_number"] + _errc)
                _now = _now.merge(_prev, on="serial_number", how="left", suffixes=("", "_prev"))
                for _i in ERR:
                    _c = f"smart_{_i}_raw"
                    _now[f"d{_lag}_{_i}"] = _now[_c] - _now[_c + "_prev"]
                    _now = _now.drop(columns=_c + "_prev")
            _now = _now.merge(_sel, on="serial_number", how="left")
            _now = _now[~(_now.fail_date <= _L)]  # already failed
            _now["landmark"] = _L
            for _t in HORIZONS + [90]:
                _end = _L + pd.Timedelta(days=_t)
                _ev = (_now.fail_date > _L) & (_now.fail_date <= _end)
                _known = _ev | (_now.last_seen >= _end)
                _now[f"y_{_t}"] = np.where(_known, _ev.astype(float), np.nan)
            _rows.append(_now)
        _study = pd.concat(_rows, ignore_index=True)
        # never use failures after the cutoff (test period) as labels
        _study.loc[_study.fail_date > CUTOFF, "fail_date"] = pd.NaT
        _study.to_parquet(STUDY_PATH, index=False)
        print(f"study built: {len(_study):,} rows in {time.time() - _t0:.0f}s")
        return _study

    study_raw = pd.read_parquet(STUDY_PATH) if STUDY_PATH.exists() else build_study()
    return (study_raw,)


@app.cell
def _(BASE, CLEAN_END, ERR, HORIZONS, OTHER, np, pd, study_raw):
    study = study_raw.copy()
    # net / dirty failure: error counters at the failure day (daily_stats.py)
    _fh = pd.read_parquet(BASE / "out" / "failure_history.parquet", columns=["serial_number", "err_at_fail"])
    study = study.merge(_fh, on="serial_number", how="left")
    study["fail_kind"] = np.select([study.err_at_fail > 0, study.err_at_fail == 0], ["bruta", "neta"], "cap")
    for _t in HORIZONS:
        study[f"clean_{_t}"] = study.landmark + pd.Timedelta(days=_t) <= CLEAN_END
    # top models as categories, the rest as "other" (chosen on the development sample only)
    _top = study.model.value_counts().index[:30]
    study["model_cat"] = pd.Categorical(study.model.where(study.model.isin(_top), "other"))
    study["err_sum"] = study[[f"smart_{i}_raw" for i in ERR]].fillna(0).sum(axis=1)
    study["n_rising"] = (study[[f"d30_{i}" for i in ERR]] > 0).sum(axis=1)
    study["n_positive"] = (study[[f"smart_{i}_raw" for i in ERR]] > 0).sum(axis=1)

    F_ERR = [f"smart_{i}_raw" for i in ERR] + [f"d{l}_{i}" for l in (7, 30) for i in ERR]
    F_OTHER = [f"smart_{i}_raw" for i in OTHER] + ["capacity_bytes"]
    F_ALL = F_ERR + F_OTHER + ["model_cat"]
    F_NO_ERR = F_OTHER + ["model_cat"]
    return F_ALL, F_NO_ERR, study


@app.cell
def _(HORIZONS, mo, pd, study):
    _rows = []
    for _t in HORIZONS:
        _k = study[study[f"y_{_t}"].notna()]
        _c = _k[_k[f"clean_{_t}"]]
        _rows.append({
            "horitzó": _t,
            "files": len(_k),
            "discs": _k.serial_number.nunique(),
            "positives": int(_k[f"y_{_t}"].sum()),
            "positives netes (fallada sense errors)": int(_k.loc[_k.fail_kind == "neta", f"y_{_t}"].sum()),
            "dates amb etiqueta neta": _c.landmark.nunique(),
            "taxa al parc (‰)": round(1000 * (_k[f"y_{_t}"] * _k.weight).sum() / _k.weight.sum(), 2),
        })
    mo.vstack([mo.md("**Mida de la mostra per horitzó.** La taxa al parc ja té en compte els pesos."),
               pd.DataFrame(_rows)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Eines comunes

    - **Validació creuada per disc** (`GroupKFold`, 3 grups per `serial_number`): el mateix disc mai
      no és a entrenament i avaluació alhora.
    - **Mètriques ponderades** amb els pesos de mostreig. El *recall@1 %* tria, a cada data, l'1 % del
      parc amb més risc i compta quina part de les fallades hi cau.
    - **Calibració:** fallades previstes / fallades observades (1 = ben calibrat; < 1 = el model
      infravalora el risc).
    - **Intervals de confiança:** bootstrap per disc (pesos de Poisson per número de sèrie), 100
      repeticions.
    - **Models:** regressió logística (log dels SMART, imputació, estandardització) i
      `HistGradientBoostingClassifier` (tracta els absents i el model de disc directament).

    **Avís: sobreajust per disc.** Cada disc apareix a unes 40 dates amb valors gairebé iguals. Amb
    la configuració habitual del GB (fulles de 200 files), el model memoritzava els discs: a
    l'entrenament la calibració era 1,00, però amb discs nous preveia **2,2 vegades** les fallades
    reals. Amb fulles de 2.000 files i més regularització, l'excés baixa a un 25–30 % i la PR-AUC
    amb discs nous **millora** (0,115 → 0,172, prova feta amb t = 60). És la mateixa raó per la qual
    validem separant per disc. L'excés que queda ve de tenir pocs discs sans diferents a la mostra
    (5 %). Per això, quan cal una xifra absoluta (E3), recalibrem amb les dates netes, i a E4 comparem
    calibracions relatives.
    """)
    return


@app.cell
def _(
    FunctionTransformer,
    GroupKFold,
    HistGradientBoostingClassifier,
    LogisticRegression,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    average_precision_score,
    np,
    pd,
    roc_auc_score,
):
    def _log_features(X):
        X = X.astype(float).copy()
        for _c in X.columns:
            if _c.startswith("d7_") or _c.startswith("d30_"):
                X[_c] = np.sign(X[_c]) * np.log1p(np.abs(X[_c]))
            else:
                X[_c] = np.log1p(X[_c].clip(lower=0))
        return X

    def make_lr():
        return Pipeline([
            ("log", FunctionTransformer(_log_features)),
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=1000)),
        ]), "lr__sample_weight"

    def make_hgb():
        # strongly regularised: each drive repeats over ~40 nearly identical landmarks (see note above)
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=2000,
            l2_regularization=10.0, early_stopping=False, categorical_features="from_dtype", random_state=0,
        ), "sample_weight"

    def oof_predict(df, y, features, make_model, train_mask=None, eval_mask=None):
        """Out-of-fold predictions with GroupKFold by drive. y can differ from the true label (noise)."""
        _train = np.ones(len(df), bool) if train_mask is None else np.asarray(train_mask)
        _eval = np.ones(len(df), bool) if eval_mask is None else np.asarray(eval_mask)
        _y = np.asarray(y)
        _w = df.weight.to_numpy()
        _pred = np.full(len(df), np.nan)
        _X = df[features]
        if "model_cat" in features and make_model is make_lr:
            raise ValueError("logistic regression takes numeric features only")
        for _tr, _te in GroupKFold(n_splits=3).split(_X, groups=df.serial_number):
            _tr = _tr[_train[_tr]]
            _te = _te[_eval[_te]]
            _m, _wname = make_model()
            _m.fit(_X.iloc[_tr], _y[_tr], **{_wname: _w[_tr]})
            _pred[_te] = _m.predict_proba(_X.iloc[_te])[:, 1]
        return _pred

    def alerts_at_budget(p, w, landmark, budget):
        """Boolean mask: top `budget` share of the (weighted) fleet of each landmark."""
        _d = pd.DataFrame({"p": p, "w": w, "lm": landmark}).sort_values(["lm", "p"], ascending=[True, False])
        _cum = _d.groupby("lm").w.cumsum()
        _tot = _d.groupby("lm").w.transform("sum")
        _alert = pd.Series(False, index=_d.index)
        _alert[_d.index] = (_cum <= budget * _tot).to_numpy()
        return _alert.sort_index().to_numpy()

    def metrics(y, p, w, landmark, budget, alert=None):
        y, p, w = np.asarray(y), np.asarray(p), np.asarray(w)
        _a = alerts_at_budget(p, w, landmark, budget) if alert is None else alert
        return {
            "PR-AUC": average_precision_score(y, p, sample_weight=w),
            "ROC-AUC": roc_auc_score(y, p, sample_weight=w),
            "recall@1%": (w * y * _a).sum() / (w * y).sum(),
            "calibració": (w * p).sum() / (w * y).sum(),
        }

    def boot_compare(y, preds, w, landmark, serial, budget, n=100, seed=0):
        """Bootstrap by drive: mean and 95 % interval of PR-AUC and recall@1 % for each prediction."""
        _rng = np.random.default_rng(seed)
        _codes, _u = pd.factorize(serial)
        y, w = np.asarray(y), np.asarray(w)
        _alerts = {k: alerts_at_budget(p, w, landmark, budget) for k, p in preds.items()}
        _out = {k: {"PR-AUC": [], "recall@1%": []} for k in preds}
        for _ in range(n):
            _w2 = w * _rng.poisson(1.0, len(_u))[_codes]
            for _k, _p in preds.items():
                _out[_k]["PR-AUC"].append(average_precision_score(y, _p, sample_weight=_w2))
                # alerts kept fixed: the budget is a property of each landmark, not of the resample
                _out[_k]["recall@1%"].append((_w2 * y * _alerts[_k]).sum() / (_w2 * y).sum())
        return {k: {m: np.array(v) for m, v in d.items()} for k, d in _out.items()}

    def ci(a):
        return f"{np.mean(a):.3f} [{np.quantile(a, 0.025):.3f}, {np.quantile(a, 0.975):.3f}]"

    return alerts_at_budget, boot_compare, ci, make_hgb, make_lr, metrics, oof_predict


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E1 · Quant depèn el model de la regla humana? (t = 30, etiquetes netes)

    Comparem quatre maneres d'ordenar els discs:

    - **Regla:** primer els discs amb més comptadors d'error que han pujat els últims 30 dies, i
      després els que en tenen més per sobre de 0. És el que faria un tècnic.
    - **Logística** i **gradient boosting (GB)** amb totes les variables.
    - **GB sense comptadors:** sense els 5 comptadors ni els seus canvis. El que encerti aquest model
      no pot venir de copiar la regla de Backblaze.

    Mirem el recall@1 % per separat per a les fallades **brutes** (amb errors el dia de la fallada,
    probablement retirades proactives) i **netes** (sense errors, probablement el disc va deixar de
    funcionar).
    """)
    return


@app.cell
def _(F_ALL, F_NO_ERR, make_hgb, make_lr, oof_predict, study):
    d30 = study[study.y_30.notna() & study.clean_30].reset_index(drop=True)
    _num = [f for f in F_ALL if f != "model_cat"]
    preds30 = {
        # technician rule: how many counters went up in the last 30 days, then how many are > 0
        "Regla (comptadors)": (10 * d30.n_rising + d30.n_positive).to_numpy(),
        "Logística": oof_predict(d30, d30.y_30, _num, make_lr),
        "GB": oof_predict(d30, d30.y_30, F_ALL, make_hgb),
        "GB sense comptadors": oof_predict(d30, d30.y_30, F_NO_ERR, make_hgb),
    }
    return d30, preds30


@app.cell
def _(BUDGET, alerts_at_budget, d30, metrics, np, pd, preds30):
    _rows = []
    for _name, _p in preds30.items():
        _a = alerts_at_budget(_p, d30.weight, d30.landmark, BUDGET)
        _m = metrics(d30.y_30, _p, d30.weight, d30.landmark, BUDGET, alert=_a)
        if _name.startswith("Regla"):
            _m["calibració"] = np.nan  # a counter sum is not a probability
        for _kind in ["bruta", "neta"]:
            _pos = (d30.y_30 == 1) & (d30.fail_kind == _kind)
            _m[f"recall@1% {_kind}"] = (d30.weight * _pos * _a).sum() / (d30.weight * _pos).sum()
        _rows.append({"model": _name, **_m})
    e1_table = pd.DataFrame(_rows).set_index("model").round(3)
    _pos = d30[d30.y_30 == 1]
    e1_share_net = (_pos.fail_kind == "neta").mean()
    e1_table
    return e1_share_net, e1_table


@app.cell
def _(BUDGET, boot_compare, ci, d30, mo, pd, preds30):
    _keys = ["Regla (comptadors)", "GB", "GB sense comptadors"]
    _b = boot_compare(d30.y_30, {k: preds30[k] for k in _keys}, d30.weight, d30.landmark,
                      d30.serial_number, BUDGET)
    _rows = [{"model": k, "PR-AUC": ci(_b[k]["PR-AUC"]), "recall@1%": ci(_b[k]["recall@1%"])} for k in _keys]
    _diff = lambda a, b, m: ci(_b[a][m] - _b[b][m])  # noqa: E731
    _rows.append({"model": "GB − Regla", "PR-AUC": _diff("GB", "Regla (comptadors)", "PR-AUC"),
                  "recall@1%": _diff("GB", "Regla (comptadors)", "recall@1%")})
    _rows.append({"model": "GB − GB sense comptadors", "PR-AUC": _diff("GB", "GB sense comptadors", "PR-AUC"),
                  "recall@1%": _diff("GB", "GB sense comptadors", "recall@1%")})
    e1_boot = pd.DataFrame(_rows).set_index("model")
    mo.vstack([mo.md("**Intervals del 95 % (bootstrap per disc).**"), e1_boot])
    return (e1_boot,)


@app.cell
def _(e1_boot, e1_share_net, e1_table, mo):
    _r = e1_table["recall@1%"]
    _rb, _rn = e1_table["recall@1% bruta"], e1_table["recall@1% neta"]
    mo.md(f"""
    **Lectura.**

    - **La regla del tècnic ja fa molta feina.** Amb l'1 % d'alertes captura un {_r.iloc[0]:.0%} de les
      fallades a 30 dies. El GB en captura un {_r['GB']:.0%}: una millora real
      ({e1_boot.loc['GB − Regla', 'recall@1%']} en recall), però modesta.
    - **Tots dos caçen gairebé només fallades brutes.** De les fallades **netes** (sense errors el dia
      de la fallada, un {e1_share_net:.0%} de les positives), la regla en captura un {_rn.iloc[0]:.1%} i el
      GB un {_rn['GB']:.1%}. El model complet aprèn sobretot la mateixa informació que fa servir
      Backblaze per decidir, i les alertes se les emporten els discs que ja tenen errors.
    - **Sense els comptadors, el GB troba les netes:** un {_rn['GB sense comptadors']:.0%}, però només
      un {_r['GB sense comptadors']:.0%} del total. Les fallades netes sí que tenen senyal en altres
      variables (edat, temperatura, model de disc...), però el model complet no hi arriba perquè gasta
      el pressupost abans.

    **Conseqüència per al projecte:** si l'objectiu és avisar dels discs que **sorprendrien** el tècnic,
    cal un model o una part del pressupost per als discs sense errors. Si l'objectiu és ordenar millor
    la feina que el tècnic ja faria, el model complet serveix, però s'ha de presentar al costat de la
    regla.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### La matriu de confusió en xifres

    El GB a 30 dies, amb l'1 % d'alertes, en una data de referència mitjana. Els recomptes estan escalats
    al parc sencer amb els pesos.
    """)
    return


@app.cell
def _(BUDGET, alerts_at_budget, d30, mo, pd, preds30):
    _a = alerts_at_budget(preds30["GB"], d30.weight, d30.landmark, BUDGET)
    _y = d30.y_30.to_numpy() == 1
    _w = d30.weight.to_numpy()
    _n = d30.landmark.nunique()
    _cm = {k: (_w * m).sum() / _n for k, m in
           {"TP": _a & _y, "FP": _a & ~_y, "FN": ~_a & _y, "TN": ~_a & ~_y}.items()}
    confusion30 = pd.DataFrame(
        [[round(_cm["TP"]), round(_cm["FP"])], [round(_cm["FN"]), round(_cm["TN"])]],
        index=["alerta", "sense alerta"], columns=["consta com a fallat (30 dies)", "no consta"])
    mo.vstack([confusion30, mo.md(f"""
    De cada **{_cm['TP'] + _cm['FP']:,.0f} alertes**, **{_cm['TP']:,.0f}** són discs que fallen en 30
    dies (precisió del {_cm['TP'] / (_cm['TP'] + _cm['FP']):.1%}) i en deixem escapar **{_cm['FN']:,.0f}**.
    La precisió sembla baixa, però la taxa base és d'un {(_cm['TP'] + _cm['FN']) / sum(_cm.values()):.2%}:
    una alerta és unes {(_cm['TP'] / (_cm['TP'] + _cm['FP'])) / ((_cm['TP'] + _cm['FN']) / sum(_cm.values())):.0f}
    vegades més probable que sigui fallada que un disc a l'atzar. L'experiment següent mira què són
    realment els FP.
    """)])
    return (confusion30,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E2 · Quants falsos positius són avisos anticipats?

    Un FP a 30 dies pot ser un disc que **sí que acaba fallant**, però més tard. Per a les dates amb
    L + 90 ≤ 16-11-2025 (així no hi entren les festes) mirem, entre els FP del GB amb l'1 % d'alertes:

    - quina part falla entre 30 i 90 dies després (**FP anticipat**);
    - quina part té algun comptador d'error > 0 a la data L (un tècnic també el tindria al punt de mira).

    Ho comparem amb els discs negatius sense alerta.
    """)
    return


@app.cell
def _(BUDGET, CLEAN_END, alerts_at_budget, d30, mo, pd, preds30):
    _a = alerts_at_budget(preds30["GB"], d30.weight, d30.landmark, BUDGET)
    _ok = (d30.landmark + pd.Timedelta(days=90) <= CLEAN_END) & d30.y_90.notna()
    _neg = (d30.y_30 == 0) & _ok
    _later = (d30.y_90 == 1).to_numpy()
    _err = (d30.err_sum > 0).to_numpy()
    _w = d30.weight.to_numpy()

    def _share(mask, cond):
        return (_w * mask * cond).sum() / (_w * mask).sum()

    _fp, _tn = (_neg & _a).to_numpy(), (_neg & ~_a).to_numpy()
    e2_table = pd.DataFrame({
        "falla entre 30 i 90 dies": [_share(_fp, _later), _share(_tn, _later)],
        "algun comptador > 0 a L": [_share(_fp, _err), _share(_tn, _err)],
    }, index=["FP (alerta, no falla en 30 dies)", "TN (sense alerta)"]).round(4)
    _tp = ((d30.y_30 == 1) & _ok & _a).to_numpy()
    _prec = (_w * _tp).sum() / (_w * (_ok.to_numpy() & _a)).sum()
    _prec90 = (_w * (_tp | (_fp & _later))).sum() / (_w * (_ok.to_numpy() & _a)).sum()
    e2_precision = {"30 dies": _prec, "90 dies (TP + FP anticipats)": _prec90}
    mo.vstack([e2_table, mo.md(f"""
    Un **{e2_table.iloc[0, 0]:.1%}** dels FP fallen entre 30 i 90 dies després, contra un
    {e2_table.iloc[1, 0]:.2%} dels TN ({e2_table.iloc[0, 0] / max(e2_table.iloc[1, 0], 1e-9):.0f} vegades
    més). I un **{e2_table.iloc[0, 1]:.0%}** dels FP ja tenen errors a la data de l'alerta. Si comptem
    els FP anticipats com a encerts, la precisió de les alertes passa del {_prec:.1%} al {_prec90:.1%}.
    Part de l'«error» del model és l'horitzó, no el model: són avisos que arriben massa d'hora.
    """)])
    return e2_precision, e2_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E3 · Què passa si entrenem amb les etiquetes contaminades per festes? (t = 60)

    És l'horitzó més afectat: les dates del 24-9 al 5-11-2025 tenen finestres que passen per les
    festes. Dues versions del GB, totes dues avaluades **només amb etiquetes netes** (dates fins al
    17-9) i amb validació creuada per disc:

    - **A · censura (proposta):** només s'entrena amb etiquetes netes.
    - **B · ingènua:** també s'entrena amb les 7 dates contaminades.

    A més, el model A fa una predicció per a les dates contaminades. La diferència entre les fallades
    que preveu i les que consten és una **estimació, feta pel model, de les fallades que falten a
    l'etiqueta**.
    """)
    return


@app.cell
def _(F_ALL, make_hgb, oof_predict, study):
    d60 = study[study.y_60.notna()].reset_index(drop=True)
    _clean = d60.clean_60.to_numpy()
    preds60 = {
        "A · censura": oof_predict(d60, d60.y_60, F_ALL, make_hgb, train_mask=_clean),
        "B · ingènua": oof_predict(d60, d60.y_60, F_ALL, make_hgb),
    }
    return d60, preds60


@app.cell
def _(BUDGET, boot_compare, ci, d60, metrics, mo, pd, preds60):
    _c = d60.clean_60.to_numpy()
    _dc = d60[_c]
    _rows = [{"entrenament": k, **metrics(_dc.y_60, p[_c], _dc.weight, _dc.landmark, BUDGET)}
             for k, p in preds60.items()]
    e3_table = pd.DataFrame(_rows).set_index("entrenament").round(3)
    _b = boot_compare(_dc.y_60, {k: p[_c] for k, p in preds60.items()}, _dc.weight, _dc.landmark,
                      _dc.serial_number, BUDGET)
    e3_boot = pd.DataFrame({m: [ci(_b["B · ingènua"][m] - _b["A · censura"][m])] for m in ["PR-AUC", "recall@1%"]},
                           index=["B − A (95 %)"])
    mo.vstack([mo.md("**Avaluació amb etiquetes netes (dates fins al 17-9-2025).**"), e3_table, e3_boot])
    return e3_boot, e3_table


@app.cell
def _(d60, pd, preds60):
    _d = d60.assign(pA=preds60["A · censura"], pB=preds60["B · ingènua"])
    _d["wy"] = _d.weight * _d.y_60
    _d["wpA"] = _d.weight * _d.pA
    _d["wpB"] = _d.weight * _d.pB
    e3_by_lm = _d.groupby("landmark").agg(n=("weight", "sum"), obs=("wy", "sum"), predA=("wpA", "sum"),
                                         predB=("wpB", "sum"), clean=("clean_60", "first")).reset_index()
    _aff = ~e3_by_lm.clean
    # calibration-in-the-large with the clean landmarks only (A and B separately)
    e3_factor = {k: e3_by_lm.loc[~_aff, "obs"].sum() / e3_by_lm.loc[~_aff, k].sum() for k in ["predA", "predB"]}
    for _k, _f in e3_factor.items():
        e3_by_lm[_k + "_cal"] = e3_by_lm[_k] * _f
    for _c in ["obs", "predA", "predB", "predA_cal", "predB_cal"]:
        e3_by_lm[f"{_c}_rate"] = e3_by_lm[_c] / e3_by_lm.n * 1000
    _last = e3_by_lm[~_aff].tail(6)  # the six clean landmarks right before the affected ones
    e3_missing = {
        "factor de recalibració (A)": e3_factor["predA"],
        "dates afectades": int(_aff.sum()),
        "fallades observades": float(e3_by_lm.loc[_aff, "obs"].sum()),
        "fallades previstes (A recalibrat)": float(e3_by_lm.loc[_aff, "predA_cal"].sum()),
    }
    e3_missing["fallades que falten (A − obs)"] = e3_missing["fallades previstes (A recalibrat)"] - e3_missing["fallades observades"]
    e3_missing["% de l'etiqueta que falta"] = 100 * e3_missing["fallades que falten (A − obs)"] / e3_missing["fallades previstes (A recalibrat)"]
    # control: on the last clean landmarks the recalibrated model should track the labels
    e3_missing["control: (previstes − obs.) / obs., 6 últimes dates netes (%)"] = float(
        100 * (_last.predA_cal.sum() - _last.obs.sum()) / _last.obs.sum())
    e3_missing_table = pd.Series(e3_missing).round(1).to_frame("valor (parc sencer)")
    e3_missing_table
    return e3_by_lm, e3_missing


@app.cell
def _(e3_by_lm, mdates, pd, plt):
    _C = {"obs": "#0b0b0b", "A": "#2a78d6", "B": "#eb6834", "shade": "#ecebe6", "ink2": "#52514e"}
    fig_e3, _ax = plt.subplots(figsize=(9, 3.6))
    _aff = e3_by_lm[~e3_by_lm.clean]
    _ax.axvspan(_aff.landmark.min() - pd.Timedelta(days=3),
                _aff.landmark.max() + pd.Timedelta(days=3), color=_C["shade"], lw=0,
                label="finestra que toca les festes")
    _ax.plot(e3_by_lm.landmark, e3_by_lm.obs_rate, color=_C["obs"], marker="o", ms=3.5, lw=2, label="observada (etiqueta)")
    _ax.plot(e3_by_lm.landmark, e3_by_lm.predA_cal_rate, color=_C["A"], marker="o", ms=3.5, lw=2, label="prevista, A · censura")
    _ax.plot(e3_by_lm.landmark, e3_by_lm.predB_cal_rate, color=_C["B"], marker="o", ms=3.5, lw=2, ls="--", label="prevista, B · ingènua")
    _ax.fill_between(_aff.landmark, _aff.obs_rate, _aff.predA_cal_rate, color=_C["A"], alpha=0.15, lw=0)
    _ax.set_title("Fallades a 60 dies per 1.000 discs: observades i previstes (discs no vistos, recalibrat amb dates netes)",
                  loc="left", fontsize=10)
    _ax.set_ylim(bottom=0)
    _ax.grid(color="#e6e5e0")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.xaxis.set_major_locator(mdates.MonthLocator())
    _ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig_e3.tight_layout()
    fig_e3
    return (fig_e3,)


@app.cell
def _(e3_boot, e3_missing, e3_table, mo):
    _cal = e3_table["calibració"]
    mo.md(f"""
    **Lectura.** Per **ordenar** discs, la diferència de recall@1 % entre les dues versions és
    {e3_boot.iloc[0, 1]} (B − A, interval del 95 %). Si l'interval conté el 0, entrenar amb etiquetes
    contaminades no canvia de manera apreciable quins discs avisem. La **probabilitat**, en canvi, sí
    que canvia: amb etiquetes netes, B preveu un {100 * (_cal.iloc[1] / _cal.iloc[0] - 1):+.0f} % de
    fallades respecte d'A (calibració {_cal.iloc[0]:.2f} i {_cal.iloc[1]:.2f}). Un valor negatiu vol dir que
    el model ingenu infravalora el risc, perquè ha après que a la tardor hi ha menys fallades.

    Recalibrat amb les dates netes (factor {e3_missing['factor de recalibració (A)']:.2f}), el model A
    estima que a les {e3_missing['dates afectades']} dates contaminades falten unes
    **{e3_missing['fallades que falten (A − obs)']:,.0f} fallades** (un
    {e3_missing["% de l'etiqueta que falta"]:.0f} % del que caldria esperar). L'estimació sense model del
    notebook 01 (unes 1.000) suposava una taxa plana igual a la mitjana de les dates netes. El model, en
    canvi, té en compte l'estat dels discs a cada data, i a finals d'estiu el risc del parc era més alt
    que la mitjana. Per això en dona més. Com a control, a les sis últimes dates netes
    la previsió s'aparta de l'etiqueta un
    {e3_missing['control: (previstes − obs.) / obs., 6 últimes dates netes (%)']:+.0f} %.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E4 · Quant error d'etiqueta tolera el model?

    Simulem retirades que no es registren: a l'entrenament, una proporció *p* dels discs que fallen
    passen a constar com a «no ha fallat» (totes les seves files). L'avaluació es fa sempre amb les
    etiquetes reals netes. GB, t = 60, dates netes.

    Referències del que hem mesurat: un **3 %** de les positives a 60 dies falten en conjunt per les
    festes, i fins a un **45 %** a la pitjor data (5-11-2025).

    *Limitació:* aquí l'error és a l'atzar. Les retirades que es retarden de veritat són les menys
    evidents (notebook 01, N3), i això pot fer més mal que l'atzar a les fallades netes.
    """)
    return


@app.cell
def _(BUDGET, F_ALL, d60, make_hgb, metrics, np, oof_predict, pd):
    _c = d60.clean_60.to_numpy()
    _dc = d60[_c].reset_index(drop=True)
    _fail_serials = _dc.loc[_dc.y_60 == 1, "serial_number"].unique()
    _rng = np.random.default_rng(0)
    _order = _rng.permutation(len(_fail_serials))
    _rows = []
    for _p in [0.0, 0.1, 0.25, 0.5]:
        _flip = set(_fail_serials[_order[: int(round(_p * len(_fail_serials)))]])
        _ynoisy = np.where(_dc.serial_number.isin(_flip), 0.0, _dc.y_60)
        _pred = oof_predict(_dc, _ynoisy, F_ALL, make_hgb)
        _rows.append({"p (positives amagades)": _p, **metrics(_dc.y_60, _pred, _dc.weight, _dc.landmark, BUDGET)})
    e4_table = pd.DataFrame(_rows).set_index("p (positives amagades)")
    e4_table["calibració relativa"] = e4_table["calibració"] / e4_table["calibració"].iloc[0]
    e4_table = e4_table.round(3)
    e4_table
    return (e4_table,)


@app.cell
def _(e4_table, plt):
    fig_e4, _axes = plt.subplots(1, 2, figsize=(9, 3.2))
    _x = e4_table.index.to_numpy()
    for _ax, _cols, _title in [
        (_axes[0], [("recall@1%", "#2a78d6"), ("PR-AUC", "#eb6834")], "Ordenació"),
        (_axes[1], [("calibració relativa", "#1baf7a")], "Calibració relativa a p = 0"),
    ]:
        for _c, _color in _cols:
            _ax.plot(_x, e4_table[_c], marker="o", color=_color, lw=2, label=_c)
        _ax.axvline(0.033, color="#8a8984", lw=1, ls=":")
        _ax.axvline(0.45, color="#8a8984", lw=1, ls=":")
        _ax.set_title(_title, loc="left", fontsize=10)
        _ax.set_xlabel("proporció de discs que fallen amagats a l'entrenament")
        _ax.set_ylim(bottom=0)
        _ax.grid(color="#e6e5e0")
        _ax.spines[["top", "right"]].set_visible(False)
        _ax.legend(frameon=False, fontsize=8)
    _axes[1].plot(_x, 1 - _x, color="#8a8984", lw=1, ls="--")
    _axes[1].annotate("1 − p", (0.3, 0.72), color="#52514e", fontsize=8)
    _axes[0].annotate("3 % (total)     45 % (pitjor data)", (0.04, 0.03), xycoords=("data", "axes fraction"),
                      color="#52514e", fontsize=7)
    fig_e4.tight_layout()
    fig_e4
    return (fig_e4,)


@app.cell
def _(e4_table, mo):
    _r = e4_table["recall@1%"]
    _cal = e4_table["calibració relativa"]
    mo.md(f"""
    **Lectura.** Amagant un 10 % de les fallades, el recall@1 % passa de {_r.iloc[0]:.3f} a
    {_r.iloc[1]:.3f}, i amagant-ne la meitat, a {_r.iloc[-1]:.3f}. L'ordenació aguanta força bé. La
    calibració relativa, en canvi, cau ({_cal.iloc[0]:.2f} → {_cal.iloc[-1]:.2f}; la línia 1 − p és la referència): el model
    aprèn una taxa base més baixa. Al nivell real del conjunt (3 %), l'efecte és petit. El problema serien
    les dates concretes amb fins a un 45 %, i per això les traiem.
    """)
    return


@app.cell
def _(
    confusion30,
    e1_table,
    e2_precision,
    e2_table,
    e3_missing,
    e3_table,
    e4_table,
    mo,
):
    _r = e1_table["recall@1%"]
    _rn = e1_table["recall@1% neta"]
    mo.vstack([mo.md(f"""
    ## Conclusions

    | Pregunta | Resultat | Què en fem |
    | --- | --- | --- |
    | Quant depèn el model de la regla humana? | Molt. La regla del tècnic ja captura un {_r.iloc[0]:.0%} de les fallades amb l'1 % d'alertes; el GB, un {_r['GB']:.0%}. De les netes, només un {_rn.iloc[0]:.1%} i un {_rn['GB']:.1%}. Sense comptadors, el GB en troba un {_rn['GB sense comptadors']:.0%} de netes | Presentar sempre el model al costat de la regla, separant netes i brutes. Provar un model per als discs sense errors |
    | Què són els FP? | Un {e2_table.iloc[0, 0]:.0%} fallen en 30–90 dies i un {e2_table.iloc[0, 1]:.0%} ja tenen errors. Precisió: {e2_precision['30 dies']:.1%} a 30 dies, {e2_precision['90 dies (TP + FP anticipats)']:.1%} comptant els anticipats | Explicar-ho a l'aplicació. Lliga amb el model per franges: un disc amb risc alt però tardà és una alerta «no urgent» |
    | Quant error hi ha a l'etiqueta per festes? | El model estima que falten unes {e3_missing['fallades que falten (A − obs)']:,.0f} fallades a 60 dies a les 7 dates afectades ({e3_missing["% de l'etiqueta que falta"]:.0f} %) | Censura de les etiquetes de desenvolupament al 16-11-2025 |
    | Fa mal entrenar-hi igualment? | Per ordenar, gairebé no. Sense censura, el model preveu un {100 * (e3_table['calibració'].iloc[1] / e3_table['calibració'].iloc[0] - 1):+.0f} % de fallades respecte del model amb censura | La censura protegeix sobretot la probabilitat (pregunta Q1 del pla) |
    | Quant error tolera el model? | Recall@1 % de {e4_table['recall@1%'].iloc[0]:.2f} sense soroll i {e4_table['recall@1%'].iloc[-1]:.2f} amagant la meitat de les fallades; calibració relativa {e4_table['calibració relativa'].iloc[-1]:.2f} | L'error d'etiqueta que no podem treure afecta més la probabilitat que el rànquing: recalibrar amb dades recents |

    **Resposta curta a «arrosseguem l'error?»:** per decidir quins discs canviar (rànquing amb pressupost),
    molt poc. Per dir «aquest disc té un 3 % de probabilitat de fallar», sí, si no traiem les etiquetes
    contaminades. Per això la censura i la recalibració formen part del pla. I el que el model aprèn
    sempre és «Backblaze el retirarà», no «es trencarà»: la separació neta/bruta diu fins a quin punt
    són el mateix.

    *Matriu de confusió de referència (GB, 30 dies, 1 % d'alertes, data mitjana):*
    """), confusion30])
    return


@app.cell
def _(BASE, fig_e3, fig_e4):
    # PNG copies of the figures for the slides
    _dir = BASE / "out" / "figures"
    _dir.mkdir(parents=True, exist_ok=True)
    fig_e3.savefig(_dir / "etiqueta_festes_prevista.png", dpi=150, bbox_inches="tight")
    fig_e4.savefig(_dir / "etiqueta_soroll.png", dpi=150, bbox_inches="tight")
    return


if __name__ == "__main__":
    app.run()
