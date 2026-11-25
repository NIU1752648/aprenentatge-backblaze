import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    return (
        FixedLocator,
        FuncFormatter,
        LinearSegmentedColormap,
        NullLocator,
        Path,
        TwoSlopeNorm,
        mo,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 · Corba d'aprenentatge: per què entrenem amb el 15 % dels discs negatius

    Decisió 9. Volem saber quantes files d'entrenament calen abans de fer la cerca
    d'hiperparàmetres (`cv_models.py`), que es repeteix moltes vegades i és cara.

    **Experiment** (`scripts/learning_curve.py`, sortida a `out/lc/`):

    - 40 dates de referència de desenvolupament, etiquetes estrictes (tall del 16-11-2025).
    - Validació en 3 folds separats per disc (hash del `serial_number`). Els folds són els
      mateixos per a totes les mides, així que les diferències entre mides es poden aparellar per fold.
    - Hiperparàmetres fixos. Tres models:
      - `hgb`: gradient boosting amb `min_samples_leaf = 2000`;
      - `hgb_scaled`: igual, però `min_samples_leaf` creix proporcionalment a la mida
        (2000 al 15 %, ~13.000 al 100 %);
      - `logreg`: regressió logística (C = 0,1).
    - Dos eixos de mida:
      - **fracció de discs negatius** {1, 2, 5, 15, 30, 50, 100} %, triats per hash i amb totes les
        seves dates. Els positius es queden sempre tots;
      - **aprimament de dates** {1, 1/3, 1/10}, que treu **positius i negatius** alhora.
    - Quatre objectius: les franges `b_0_7`, `b_7_30`, `b_30_60` i l'acumulat `y_30`.
    - Mètrica principal: **recall amb l'1 % d'alertes per data** (pressupost d'alertes de
      l'aplicació). També mirem el PR-AUC i el calibratge (probabilitat mitjana corregida / taxa observada).

    El notebook només llegeix `out/lc/lc_summary.csv` i `out/lc/lc_folds.csv`.
    Es pot passar un altre directori amb `-- --data <dir>`.
    """)
    return


@app.cell
def _(Path, mo, pd):
    BASE = Path(mo.cli_args().get("data") or mo.notebook_dir().parent / "dataset" / "backblaze")
    LC = BASE / "out" / "lc"
    summary = pd.read_csv(LC / "lc_summary.csv")
    folds = pd.read_csv(LC / "lc_folds.csv")

    MODELS = ["hgb_scaled", "hgb", "logreg"]
    TARGETS = ["b_0_7", "b_7_30", "b_30_60", "y_30"]
    TARGET_LABEL = {"b_0_7": "Falla en 0–7 dies", "b_7_30": "Falla en 7–30 dies",
                    "b_30_60": "Falla en 30–60 dies", "y_30": "Falla en ≤ 30 dies"}
    MODEL_LABEL = {"hgb_scaled": "GB, fulla escalada", "hgb": "GB, fulla fixa", "logreg": "Logística"}
    FRACS = [0.01, 0.02, 0.05, 0.15, 0.30, 0.50, 1.00]
    CHOSEN = 0.15
    return (
        CHOSEN,
        FRACS,
        MODELS,
        MODEL_LABEL,
        TARGETS,
        TARGET_LABEL,
        folds,
        summary,
    )


@app.cell
def _(plt):
    # Reference palette (dataviz skill): first three categorical slots, validated all-pairs
    # for colour-vision deficiency. Aqua is below 3:1 on the surface, so every line is
    # direct-labelled.
    COLOR = {"hgb_scaled": "#eb6834", "hgb": "#2a78d6", "logreg": "#1baf7a"}
    INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    SURFACE = "#fcfcfb"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 6, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 9.5, "figure.dpi": 110,
    })
    return COLOR, INK, INK2, MUTED


@app.cell
def _(CHOSEN, FRACS, FixedLocator, FuncFormatter, MUTED, NullLocator):
    def pct_axis(ax, mark_chosen=True):
        """Log x axis over the negative-drive fractions, labelled in %, chosen size marked."""
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(FRACS))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
        ax.set_xlim(FRACS[0] / 1.25, FRACS[-1] * 1.25)
        ax.grid(axis="x", visible=False)
        if mark_chosen:
            ax.axvspan(CHOSEN / 1.12, CHOSEN * 1.12, color=MUTED, alpha=0.12, lw=0, zorder=0)

    return (pct_axis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Quantes dades calen: el recall s'estanca al 15 %

    Només l'eix de la fracció de negatius, amb totes les dates. La línia és la mitjana dels 3 folds
    i l'ombra, ± una desviació estàndard entre folds. La franja grisa marca el 15 %.
    """)
    return


@app.cell
def _(
    COLOR,
    INK2,
    MODELS,
    MODEL_LABEL,
    TARGETS,
    TARGET_LABEL,
    pct_axis,
    plt,
    summary,
):
    def curve_grid(metric, ylabel, title):
        _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
        _s = summary[summary["step"] == 1]
        for _ax, _t in zip(_axes, TARGETS):
            for _m in MODELS:
                _d = _s[(_s.model == _m) & (_s.target == _t)].sort_values("frac").dropna(
                    subset=[f"{metric}_mean"])
                _y, _sd = _d[f"{metric}_mean"].to_numpy(), _d[f"{metric}_sd"].to_numpy()
                _ls = "--" if _m == "hgb" else "-"
                _ax.fill_between(_d.frac, _y - _sd, _y + _sd, color=COLOR[_m], alpha=0.12, lw=0)
                _ax.plot(_d.frac, _y, _ls, color=COLOR[_m], lw=2 if _m != "hgb" else 1.5,
                         marker="o", ms=5, mec="#fcfcfb", mew=1.2, label=MODEL_LABEL[_m])
            _ax.set_title(TARGET_LABEL[_t])
            pct_axis(_ax)
            _ax.set_xlabel("% de discs negatius a l'entrenament")
        _axes[0].set_ylabel(ylabel)
        # direct labels on the right-most panel, at each line's end
        _last = _axes[-1]
        for _m in MODELS:
            _d = _s[(_s.model == _m) & (_s.target == TARGETS[-1])].dropna(subset=[f"{metric}_mean"])
            _end = _d.sort_values("frac").iloc[-1]
            _last.annotate(MODEL_LABEL[_m], (_end.frac, _end[f"{metric}_mean"]),
                           xytext=(8, 0), textcoords="offset points", va="center",
                           fontsize=9, color=INK2)
        _fig.legend(*_axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                    bbox_to_anchor=(0.5, -0.06))
        _fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
        _fig.tight_layout(rect=(0, 0.03, 0.97, 1))
        return _fig

    return (curve_grid,)


@app.cell
def _(curve_grid):
    curve_grid("recall_1", "recall amb l'1 % d'alertes",
               "El GB guanya a la logística a totes les franges, i a partir del 15 % gairebé no millora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **GB, fulla escalada** (taronja) puja fins al 15 % i després queda pla: totes les mides
      següents estan dins de l'ombra.
    - **GB, fulla fixa** (blau, discontinu) coincideix amb l'escalada fins al 15 % (al 15 % els dos
      models són idèntics) i **s'enfonsa** al 50–100 % en tres de les quatre franges. No és que
      més dades facin mal, és una fallada de l'early stopping (secció 5).
    - **Logística** (aqua): ja és plana des de l'1 % i baixa una mica amb més negatius. Queda
      5–6 punts per sota del GB a totes les franges.

    El PR-AUC continua pujant una mica més enllà del 15 %, sobretot a `y_30`. Mesura tota la
    llista ordenada; a l'aplicació només ens importa el principi de la llista (l'1 %).
    """)
    return


@app.cell
def _(curve_grid):
    curve_grid("pr_auc", "PR-AUC",
               "El PR-AUC encara puja una mica després del 15 %, però amb molta variació entre folds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Quant perdem respecte de fer servir totes les dades

    Com que els folds són els mateixos discs a totes les mides, podem comparar cada mida amb la
    millor **fold a fold**. La cel·la és la diferència mitjana de recall@1 % (en punts) respecte de
    la millor mida del mateix model i objectiu, amb totes les dates. El símbol `·` indica que la
    diferència és més petita que una desviació estàndard de la millor mida (la regla de
    `lc_recommendation.json`).
    """)
    return


@app.cell
def _(FRACS, MODELS, TARGETS, folds, pd):
    _f = folds[(folds.step == 1) & (folds.status == "ok")]
    _rows = []
    for _m in MODELS:
        for _t in TARGETS:
            _d = _f[(_f.model == _m) & (_f.target == _t)].pivot(index="fold", columns="frac",
                                                                values="recall_1")
            _best = _d.mean().idxmax()
            _thr = _d[_best].std()
            for _fr in FRACS:
                if _fr not in _d:
                    _rows.append(dict(model=_m, target=_t, frac=_fr, gap=float("nan"),
                                      within=False, best=False))
                    continue
                _diff = _d[_fr] - _d[_best]
                _rows.append(dict(model=_m, target=_t, frac=_fr, gap=100 * _diff.mean(),
                                  gap_sd=100 * _diff.std(), within=-_diff.mean() <= _thr,
                                  best=_fr == _best))
    gaps = pd.DataFrame(_rows)
    return (gaps,)


@app.cell
def _(
    CHOSEN,
    FRACS,
    INK,
    LinearSegmentedColormap,
    MODELS,
    MODEL_LABEL,
    TARGETS,
    TARGET_LABEL,
    TwoSlopeNorm,
    gaps,
    np,
    plt,
):
    # Diverging: red-ish loss -> neutral gray at 0 -> blue gain (gain only exists vs a noisy best,
    # so it is always small)
    _cmap = LinearSegmentedColormap.from_list(
        "gap", [(0, "#b8322f"), (0.25, "#e98a87"), (0.5, "#f0efec"), (1, "#86b6ef")])
    _cmap.set_bad("#fcfcfb")
    _norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=2)
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 3.0), sharey=True)
    for _ax, _m in zip(_axes, MODELS):
        _g = gaps[gaps.model == _m]
        _mat = np.array([[_g[(_g.target == _t) & (_g.frac == _fr)].gap.iloc[0] for _fr in FRACS]
                         for _t in TARGETS])
        _ax.imshow(np.clip(_mat, -20, 2), cmap=_cmap, norm=_norm, aspect="auto")
        for _i, _t in enumerate(TARGETS):
            for _j, _fr in enumerate(FRACS):
                _r = _g[(_g.target == _t) & (_g.frac == _fr)].iloc[0]
                if np.isnan(_r.gap):
                    _ax.text(_j, _i, "memòria", ha="center", va="center", fontsize=7.5, color="#8a8984")
                    continue
                _txt = "millor" if _r.best else f"{_r.gap:+.1f}".replace(".", ",") + (" ·" if _r.within else "")
                _ax.text(_j, _i, _txt, ha="center", va="center", fontsize=8.5,
                         color="white" if _r.gap < -10 else INK,
                         fontweight="bold" if _fr == CHOSEN else "normal")
        _ax.set_xticks(range(len(FRACS)), [f"{f * 100:.0f}" for f in FRACS])
        _ax.set_yticks(range(len(TARGETS)), [TARGET_LABEL[t] for t in TARGETS])
        _ax.grid(False)
        for _sp in _ax.spines.values():
            _sp.set_visible(False)
        _j = FRACS.index(CHOSEN)
        _ax.add_patch(plt.Rectangle((_j - 0.5, -0.5), 1, len(TARGETS), fill=False, ec=INK, lw=1.5))
        _ax.set_title(MODEL_LABEL[_m])
        _ax.set_xlabel("% de discs negatius")
    _fig.suptitle("Punts de recall@1 % perduts respecte de la millor mida (· = dins d'1 desv. est.)",
                  x=0.01, ha="left", fontsize=12, fontweight="bold")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Al 15 % (recuadre), el GB amb fulla escalada perd com a molt 1,8 punts (`y_30`) i a les tres
    franges queda a menys d'un punt. Fins i tot quan la diferència supera una desviació estàndard, és
    més petita que la que separa el GB de la logística. La regla automàtica no triava cap mida global
    perquè la desviació de referència és la de la millor mida, que de vegades és molt petita (0,2
    punts a `b_0_7`), i perquè les mides grans de `hgb` estan trencades.

    La taula resumeix el 15 % contra el 100 % (diferència aparellada per fold, en punts):
    """)
    return


@app.cell
def _(MODEL_LABEL, TARGET_LABEL, folds, mo, pd, summary):
    _f = folds[(folds.step == 1) & (folds.status == "ok")]
    _s = summary[summary.step == 1].set_index(["model", "target", "frac"])
    _rows = []
    for (_m, _t), _d in _f.groupby(["model", "target"]):
        _p = _d.pivot(index="fold", columns="frac", values="recall_1")
        _ref = 1.0 if 1.0 in _p else 0.5
        _diff = 100 * (_p[0.15] - _p[_ref])
        _rows.append({
            "model": MODEL_LABEL[_m], "objectiu": TARGET_LABEL[_t],
            "comparat amb": f"{_ref * 100:.0f} %",
            "recall@1 % al 15 %": round(_s.loc[(_m, _t, 0.15), "recall_1_mean"], 3),
            "recall@1 % comparat": round(_s.loc[(_m, _t, _ref), "recall_1_mean"], 3),
            "Δ punts (mitjana)": round(_diff.mean(), 1),
            "Δ punts (desv. est.)": round(_diff.std(), 1),
            "temps entrenament ×": round(_s.loc[(_m, _t, _ref), "fit_s_mean"]
                                         / _s.loc[(_m, _t, 0.15), "fit_s_mean"], 1),
        })
    mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Quant costa: el temps creix amb les files, el recall no

    Cada punt és una mida (totes les dates). Eix x: segons per entrenar un fold (escala
    logarítmica); eix y: recall@1 %. A partir del 15 %, la corba és horitzontal: multiplicar el
    temps per 6–8 no compra res que es pugui distingir del soroll entre folds. A la cerca
    d'hiperparàmetres, aquest cost es multiplica per cada combinació de la graella i cada fold.
    """)
    return


@app.cell
def _(COLOR, INK2, MODELS, MODEL_LABEL, TARGETS, TARGET_LABEL, plt, summary):
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.4), sharex=True)
    _s = summary[(summary.step == 1) & (summary.model != "hgb")]
    for _ax, _t in zip(_axes, TARGETS):
        for _m in [m for m in MODELS if m != "hgb"]:
            _d = _s[(_s.model == _m) & (_s.target == _t)].dropna(subset=["recall_1_mean"]).sort_values("frac")
            _ax.plot(_d.fit_s_mean, _d.recall_1_mean, "-o", color=COLOR[_m], ms=5,
                     mec="#fcfcfb", mew=1.2, label=MODEL_LABEL[_m])
            for _, _r in _d.iterrows():
                if _r.frac in (0.01, 0.15, 1.0) or (_m == "logreg" and _r.frac == 0.5):
                    _ax.annotate(f"{_r.frac * 100:.0f} %", (_r.fit_s_mean, _r.recall_1_mean),
                                 xytext=(0, 7), textcoords="offset points", ha="center",
                                 fontsize=8, color=INK2,
                                 fontweight="bold" if _r.frac == 0.15 else "normal")
        _ax.set_xscale("log")
        _ax.set_title(TARGET_LABEL[_t])
        _ax.set_xlabel("segons d'entrenament per fold")
        _ax.margins(y=0.15)
    _axes[0].set_ylabel("recall amb l'1 % d'alertes")
    _fig.legend(*_axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.06))
    _fig.suptitle("Del 15 % al 100 %: entre 6 i 8 vegades més temps per a la mateixa recall",
                  x=0.01, ha="left", fontsize=12, fontweight="bold")
    _fig.tight_layout(rect=(0, 0.03, 1, 1))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. I aprimar dates en lloc de negatius?

    Una alternativa per tenir menys files seria quedar-se amb 1 de cada 3 o 1 de cada 10 dates.
    Però això també treu **positius**, que són l'escàs: amb 1/10 passem de ~2.300 a ~200 positius
    per fold a `b_0_7`. Per al GB amb fulla escalada, cada línia és un grau d'aprimament de dates
    i els punts són les fraccions de negatius (1 → 100 %, d'esquerra a dreta).
    """)
    return


@app.cell
def _(COLOR, INK2, TARGETS, TARGET_LABEL, plt, summary):
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.6))
    _shade = {1: 1.0, 3: 0.55, 10: 0.28}
    _lab = {1: "totes les dates", 3: "1 de cada 3 dates", 10: "1 de cada 10 dates"}
    for _ax, _t in zip(_axes, TARGETS):
        for _st in (1, 3, 10):
            _d = summary[(summary.model == "hgb_scaled") & (summary.target == _t)
                         & (summary.step == _st)].sort_values("n_train_mean")
            _ax.plot(_d.n_train_mean, _d.recall_1_mean, "-o", color=COLOR["hgb_scaled"],
                     alpha=_shade[_st], ms=5, mec="#fcfcfb", mew=1.2, label=_lab[_st])
            _c = _d[_d.frac == 0.15].iloc[0]
            _ax.annotate("15 %", (_c.n_train_mean, _c.recall_1_mean), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=8, color=INK2)
        _ax.set_xscale("log")
        _ax.set_title(TARGET_LABEL[_t])
        _ax.set_xlabel("files d'entrenament")
        _ax.margins(y=0.15)
    _axes[0].set_ylabel("recall amb l'1 % d'alertes")
    _fig.legend(*_axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                bbox_to_anchor=(0.5, -0.06))
    _fig.suptitle("Amb 1 de cada 10 dates es perd molt; el màxim de cada franja s'assoleix amb totes les dates",
                  x=0.01, ha="left", fontsize=12, fontweight="bold")
    _fig.tight_layout(rect=(0, 0.03, 1, 1))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **1 de cada 10 dates** (línia més clara) queda clarament per sota a `b_0_7` (−10 punts al 15 %)
      i `b_7_30`, i no arriba mai al nivell de totes les dates.
    - **1 de cada 3 dates** és més subtil: a igual nombre de files, sovint guanya a "totes les dates
      amb menys negatius", perquè amb pocs negatius el que limita és la varietat de discs, no les
      dates repetides. Però el seu màxim queda per sota del de totes les dates a `b_0_7`
      (0,632 contra 0,657) i `b_7_30`, i al 15 % perd entre 0 i 2,7 punts.
    - El valor més alt de cada franja s'obté sempre amb totes les dates.

    Per tant, mantenim totes les dates (i tots els positius) i reduïm només els discs negatius.
    1 de cada 3 dates queda com a opció si la cerca d'hiperparàmetres no cap en temps.

    ## 5. Calibratge: amb el 15 % la correcció ja funciona

    Entrenem amb una part dels negatius i corregim la probabilitat amb la proporció de mostreig.
    El quocient (probabilitat mitjana corregida / taxa observada al fold de validació) hauria de
    ser 1. La franja grisa és l'interval (0,67; 1,5) que el script fa servir com a alarma.
    """)
    return


@app.cell
def _(COLOR, INK2, MODELS, MODEL_LABEL, MUTED, TARGETS, TARGET_LABEL, pct_axis, plt, summary):
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.3), sharey=True)
    _s = summary[(summary.step == 1) & (summary.model != "hgb")]
    for _ax, _t in zip(_axes, TARGETS):
        _ax.axhspan(0.67, 1.5, color=MUTED, alpha=0.10, lw=0, zorder=0)
        _ax.axhline(1, color=MUTED, lw=1)
        for _m in [m for m in MODELS if m != "hgb"]:
            _d = _s[(_s.model == _m) & (_s.target == _t)].dropna(subset=["calib_ratio_mean"]).sort_values("frac")
            _ax.plot(_d.frac, _d.calib_ratio_mean, "-o", color=COLOR[_m], ms=5, mec="#fcfcfb",
                     mew=1.2, label=MODEL_LABEL[_m])
        pct_axis(_ax)
        _ax.set_yscale("log")
        _ax.set_yticks([0.67, 1, 1.5, 2], ["0,67", "1", "1,5", "2"])
        _ax.minorticks_off()
        _ax.set_title(TARGET_LABEL[_t])
        _ax.set_xlabel("% de discs negatius")
    _axes[0].set_ylabel("p corregida / taxa observada")
    _axes[-1].annotate("calibratge perfecte", (0.012, 1), xytext=(0, -10),
                       textcoords="offset points", fontsize=8, color=INK2)
    _fig.legend(*_axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.07))
    _fig.suptitle("Amb molt pocs negatius, la probabilitat corregida es dispara; al 15 % ja és ~1,1",
                  x=0.01, ha="left", fontsize=12, fontweight="bold")
    _fig.tight_layout(rect=(0, 0.04, 1, 1))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Amb l'1–2 % de negatius, les probabilitats corregides surten entre 1,3 i 2,2 vegades massa altes:
    la correcció per la proporció de mostreig és exacta en teoria, però el model veu tan pocs
    negatius que sobreestima el risc. Al 15 % el quocient és 1,1–1,2 per a tots dos models. Per a
    la probabilitat acumulada que volem donar, això és un altre motiu per no baixar del 15 %.

    ## 6. Per què el GB amb fulla fixa s'enfonsa amb moltes dades

    Cada punt és un fold. `n_iter` és el nombre d'arbres que deixa l'early stopping (paciència
    de 15 iteracions). Els folds que s'aturen a **15–18** arbres no han millorat la pèrdua de
    validació després dels primers arbres: el model resultant és gairebé constant.
    """)
    return


@app.cell
def _(COLOR, FRACS, INK2, MUTED, TARGETS, TARGET_LABEL, folds, np, pct_axis, plt):
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.3), sharey=True)
    _f = folds[(folds.step == 1) & (folds.status == "ok")]
    _rng = np.random.default_rng(0)
    for _ax, _t in zip(_axes, TARGETS):
        _ax.axhline(15, color=MUTED, lw=1)
        for _m, _off in (("hgb", 0.94), ("hgb_scaled", 1.06)):
            _d = _f[(_f.model == _m) & (_f.target == _t)]
            _x = _d.frac * _off * np.exp(_rng.uniform(-0.03, 0.03, len(_d)))
            _ax.scatter(_x, _d.n_iter, s=30, color=COLOR[_m], edgecolor="#fcfcfb", lw=1,
                        label={"hgb": "GB, fulla fixa", "hgb_scaled": "GB, fulla escalada"}[_m],
                        zorder=3)
        pct_axis(_ax, mark_chosen=False)
        _ax.set_yscale("log")
        _ax.set_yticks([15, 30, 100, 300], ["15", "30", "100", "300"])
        _ax.minorticks_off()
        _ax.set_title(TARGET_LABEL[_t])
        _ax.set_xlabel("% de discs negatius")
    _axes[0].set_ylabel("arbres (n_iter)")
    _axes[0].annotate("aturat al mínim (15)", (FRACS[0], 15), xytext=(0, 4),
                      textcoords="offset points", fontsize=8, color=INK2)
    _fig.legend(*_axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.07))
    _fig.suptitle("Amb fulla fixa, molts folds del 50–100 % s'aturen a ~15 arbres; amb fulla escalada, mai",
                  x=0.01, ha="left", fontsize=12, fontweight="bold")
    _fig.tight_layout(rect=(0, 0.04, 1, 1))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El col·lapse passa només amb la fulla fixa de 2000 i moltes files. Amb la fulla escalada no
    passa mai, i al 15 % els dos models coincideixen. Per tant, el problema del `hgb` no es pot fer
    servir com a argument a favor ni en contra d'una mida, però sí que ens avisa que, a la
    cerca d'hiperparàmetres, cal escalar `min_samples_leaf` amb la mida de la mostra.

    ## Conclusió

    **Entrenem amb el 15 % dels discs negatius (triats per hash), totes les dates i tots els positius.**

    1. **Rendiment:** el GB queda a menys d'un punt de recall@1 % de la millor mida a les tres
       franges, i a 1,8 punts a `y_30`. La logística no millora amb més negatius.
    2. **Cost:** cada entrenament és 6–8 vegades més ràpid que amb totes les dades, i ocupa menys
       memòria (la logística amb el 100 % ni tan sols cabia en 12 GB).
    3. **Calibratge:** la correcció de probabilitats ja funciona (quocient ~1,1), cosa que no passa
       per sota del 5 %.
    4. **Com reduir:** mostrejar discs negatius i mantenir totes les dates. Aprimar 1 de cada 10
       dates treu massa positius; 1 de cada 3 costa poc i queda de reserva si falta temps.

    **Limitacions:**

    - Només 3 folds: la desviació estàndard entre folds és una estimació grollera. Els intervals
      definitius s'han de fer amb bootstrap per disc sobre les prediccions.
    - Hiperparàmetres fixos: amb el 15 %, la cerca de `cv_models.py` podria trobar models que
      aprofitin més dades. Si el millor model de la cerca és a la vora de la graella, convindria
      repetir aquest punt amb el 30 %.
    - Les etiquetes depenen en part de SMART 5, 187, 188, 197 i 198 (vegeu
      `dataset/estudi_backblaze.md`, secció 7), així que el recall absolut s'ha de llegir amb
      cautela. La comparació entre mides no hi queda afectada.
    """)
    return


if __name__ == "__main__":
    app.run()
