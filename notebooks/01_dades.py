import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return Path, mdates, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 · Dades i efecte de Nadal

    Anàlisi preliminar de la secció 1.5 bis de `pla_projecte.md`: quina mida i forma té la caiguda de
    fallades per Nadal (N1), si hi ha acumulació després de festes (N2), si és retard en la retirada o
    menys fallades reals (N3), quines etiquetes de desenvolupament toca (N4) i si afecta les variables
    de les primeres dates de test (N5).

    **Regla per no contaminar el test:** tot el que hi ha aquí fa servir dades fins al **4-1-2026**,
    l'últim dia que fan servir les etiquetes de desenvolupament.

    Fonts (totes ja processades, el notebook no llegeix el panel):

    - `data/drives.parquet` (`consolidate.py`): una fila per disc.
    - `out/daily_stats.parquet`, `out/landmark_stats.parquet`, `out/failure_history.parquet`,
      `out/drive_onsets.parquet` (`daily_stats.py`).

    Es pot passar un altre directori de dades amb `-- --data <dir>`.
    """)
    return


@app.cell
def _(Path, mo, pd):
    BASE = Path(mo.cli_args().get("data") or mo.notebook_dir().parent / "dataset" / "backblaze")
    CUTOFF = pd.Timestamp("2026-01-04")
    # Christmas windows (both pass through the New Year)
    XMAS = [(pd.Timestamp("2024-12-20"), pd.Timestamp("2025-01-05")),
            (pd.Timestamp("2025-12-20"), pd.Timestamp("2026-01-04"))]
    # Holiday season seen in the data: weekly failures start to fall the week of
    # Thanksgiving (N1). Labels of development landmarks that reach into it are affected (N4).
    SEASON = (pd.Timestamp("2025-11-17"), pd.Timestamp("2026-01-04"))
    # Reporting outage: 29 % of the drives are missing on this single day
    OUTAGE = pd.Timestamp("2025-05-12")
    # US federal holidays in the period (Backblaze data centres are mostly in the US)
    US_HOLIDAYS = pd.to_datetime([
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-07-04", "2025-09-01",
        "2025-11-27", "2025-11-28", "2025-12-25", "2026-01-01",
    ])
    # Development landmarks of the plan (section 1.5)
    DEV_LANDMARKS = pd.date_range("2025-02-05", "2025-11-05", freq="7D")
    HORIZONS = [7, 30, 60]
    return BASE, CUTOFF, DEV_LANDMARKS, HORIZONS, OUTAGE, SEASON, US_HOLIDAYS, XMAS


@app.cell
def _(BASE, CUTOFF, pd):
    drives = pd.read_parquet(BASE / "data" / "drives.parquet")
    daily_all = pd.read_parquet(BASE / "out" / "daily_stats.parquet")
    daily = daily_all[daily_all.date <= CUTOFF].reset_index(drop=True)
    marks = pd.read_parquet(BASE / "out" / "landmark_stats.parquet").query("date <= @CUTOFF")
    fails = pd.read_parquet(BASE / "out" / "failure_history.parquet").query("fail_date <= @CUTOFF")
    onsets = pd.read_parquet(BASE / "out" / "drive_onsets.parquet").query("onset_date <= @CUTOFF")
    return daily, daily_all, drives, fails, marks, onsets


@app.cell(hide_code=True)
def _(daily_all, drives, mo):
    _n_fail = int(drives.failed.sum())
    mo.md(f"""
    ## Les dades

    | | |
    | --- | --- |
    | Dies al panel | {len(daily_all)} ({daily_all.date.min():%d-%m-%Y} → {daily_all.date.max():%d-%m-%Y}) |
    | Discs diferents | {len(drives):,} |
    | Discs amb fallada | {_n_fail:,} |
    | Discs actius per dia | {daily_all.n_active.min():,} – {daily_all.n_active.max():,} |
    | Censura | {", ".join(f"{k}: {v:,}" for k, v in drives.censoring.value_counts().items())} |

    Totes les xifres següents es limiten a dates fins al 4-1-2026.
    """)
    return


@app.cell
def _(mdates, plt):
    # Reference palette (dataviz skill), light mode
    C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6",
         "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8984", "grid": "#e6e5e0",
         "shade": "#ecebe6"}
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
        "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlesize": 11,
        "axes.titlecolor": C["ink"], "font.size": 9, "lines.linewidth": 2, "legend.frameon": False,
    })

    def shade_holidays(ax, xmas, holidays, label=True):
        for i, (a, b) in enumerate(xmas):
            ax.axvspan(a, b, color=C["shade"], zorder=0, lw=0,
                       label="Nadal" if (label and i == 0) else None)
        for h in holidays:
            ax.axvline(h, color=C["muted"], lw=0.6, ls=":", zorder=0)

    def month_axis(ax):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))

    return C, month_axis, shade_holidays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N1 · Mida i forma de la caiguda

    Tres sèries setmanals normalitzades per 100.000 discs-dia (així les setmanes incompletes i els
    canvis de mida del parc no hi influeixen):

    1. **Fallades** (primera fallada de cada disc): és l'etiqueta del model.
    2. **Pujades d'un comptador d'error** (SMART 5, 187, 188, 197, 198 respecte de l'observació
       anterior del mateix disc): un senyal físic que no depèn de cap decisió humana.
    3. **Discs que surten sense fallar**: retirades i migracions, també una decisió operativa.

    Si les fallades cauen per Nadal però les pujades de comptadors no, la caiguda és operativa.
    """)
    return


@app.cell
def _(daily):
    weekly = (daily.set_index("date")
              .resample("W-SUN")
              .agg({"n_active": "sum", "n_fail": "sum", "n_err_inc": "sum", "n_err_onset": "sum",
                    "n_gone": "sum", "n_return": "sum", "n_new": "sum"}))
    weekly["n_days"] = daily.set_index("date").resample("W-SUN").size()
    for _c in ["n_fail", "n_err_inc", "n_err_onset", "n_gone"]:
        weekly[f"rate_{_c[2:]}"] = weekly[_c] / weekly.n_active * 1e5
    weekly = weekly.reset_index().rename(columns={"date": "week_end"})
    # the first day has no previous observation: no increases, every drive is "new"
    weekly = weekly.iloc[1:].reset_index(drop=True) if weekly.n_days.iloc[0] < 7 else weekly
    return (weekly,)


@app.cell
def _(C, OUTAGE, SEASON, US_HOLIDAYS, XMAS, month_axis, pd, plt, shade_holidays, weekly):
    _w = weekly.copy()
    _x = _w.week_end - pd.Timedelta(days=3)  # centre of the week
    fig_weekly, _axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    _series = [
        ("rate_fail", "Fallades per 100.000 discs-dia", C["blue"]),
        ("rate_err_inc", "Discs amb pujada d'un comptador d'error, per 100.000 discs-dia", C["orange"]),
        ("rate_gone", "Discs que surten sense fallar, per 100.000 discs-dia", C["aqua"]),
    ]
    for _ax, (_col, _title, _color) in zip(_axes, _series):
        shade_holidays(_ax, XMAS, US_HOLIDAYS, label=_col == "rate_fail")
        _ax.plot(_x, _w[_col], color=_color, marker="o", ms=3.5)
        _part = _w.n_days < 7
        _ax.plot(_x[_part], _w[_col][_part], ls="none", marker="o", ms=6, mfc="white", mec=_color)
        _ax.set_title(_title)
        _ax.set_ylim(bottom=0)
    _axes[0].axvline(SEASON[0], color=C["ink2"], lw=1, ls="--", label="inici de la baixada (17-11)")
    _axes[0].legend(loc="lower left")
    _top = 4 * _w.rate_gone.median()
    _axes[2].set_ylim(0, _top)
    _axes[2].annotate(f"{OUTAGE:%d-%m}: caiguda de reporting\n(fora d'escala: {_w.rate_gone.max():,.0f})",
                      (OUTAGE, _top * 0.97), xytext=(8, -4), textcoords="offset points", va="top",
                      fontsize=8, color=C["ink2"])
    _axes[-1].set_xlim(_x.min() - pd.Timedelta(days=5), _x.max() + pd.Timedelta(days=5))
    month_axis(_axes[-1])
    _axes[-1].annotate("cercles buits: setmanes incompletes · línies puntejades: festius federals dels EUA",
                       (0, -0.32), xycoords="axes fraction", color=C["ink2"], fontsize=8)
    fig_weekly.tight_layout()
    fig_weekly
    return (fig_weekly,)


@app.cell
def _(SEASON, XMAS, pd, weekly):
    def period(d):
        if XMAS[0][0] <= d <= XMAS[0][1]:
            return "Nadal 2024 (final)"
        if SEASON[0] <= d <= SEASON[1]:
            return "festes 2025 (17-11 – 4-1)"
        if pd.Timestamp("2025-09-29") <= d < SEASON[0]:
            return "29-9 – 16-11-2025"
        if XMAS[0][1] < d <= pd.Timestamp("2025-03-31"):
            return "gen–mar 2025 (després de Nadal)"
        return "resta de 2025"

    _w = weekly.assign(period=weekly.week_end.map(lambda e: period(e - pd.Timedelta(days=3))))
    n1_table = (_w.groupby("period")
                .agg(setmanes=("week_end", "size"), discs_dia=("n_active", "sum"), fallades=("n_fail", "sum"),
                     pujades=("n_err_inc", "sum"), sortides=("n_gone", "sum")))
    for _c, _n in [("fallades", "fallades_100k"), ("pujades", "pujades_100k"), ("sortides", "sortides_100k")]:
        n1_table[_n] = (n1_table[_c] / n1_table.discs_dia * 1e5).round(2)
    _ref = n1_table.loc["resta de 2025"]
    for _n in ["fallades_100k", "pujades_100k", "sortides_100k"]:
        n1_table[_n.replace("_100k", "_vs_resta")] = (n1_table[_n] / _ref[_n]).round(2)
    n1_table = n1_table.drop(columns=["discs_dia"])
    n1_table
    return n1_table, period


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dia de la setmana

    Si l'etiqueta depèn de quan algú retira el disc, s'hauria de veure també el cap de setmana. Taxa
    per dia de la setmana, relativa a la mitjana, fora de Nadal.
    """)
    return


@app.cell
def _(C, XMAS, daily, np, plt):
    _d = daily.iloc[1:].copy()
    _xm = np.zeros(len(_d), bool)
    for _a, _b in XMAS:
        _xm |= (_d.date >= _a).to_numpy() & (_d.date <= _b).to_numpy()
    _d = _d[~_xm]
    _d["dow"] = _d.date.dt.day_of_week
    dow = _d.groupby("dow")[["n_active", "n_fail", "n_err_inc", "n_gone"]].sum()
    for _c in ["n_fail", "n_err_inc", "n_gone"]:
        _r = dow[_c] / dow.n_active
        dow[_c + "_rel"] = _r / _r.mean()
    fig_dow, _ax = plt.subplots(figsize=(7.5, 3.2))
    _lab = ["dl", "dt", "dc", "dj", "dv", "ds", "dg"]
    _wd = 0.26
    for _k, (_c, _name, _color) in enumerate([("n_fail", "Fallades", C["blue"]),
                                              ("n_err_inc", "Pujades de comptador", C["orange"]),
                                              ("n_gone", "Sortides sense fallar", C["aqua"])]):
        _ax.bar(np.arange(7) + (_k - 1) * (_wd + 0.02), dow[_c + "_rel"], width=_wd, color=_color, label=_name)
    _ax.axhline(1, color=C["ink2"], lw=1)
    _ax.set_xticks(range(7), _lab)
    _ax.set_title("Taxa per dia de la setmana, relativa a la mitjana (fora de Nadal)")
    _ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig_dow.tight_layout()
    fig_dow
    return dow, fig_dow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N2 · Acumulació després de festes (gener de 2025)

    Les dades comencen l'1-1-2025, així que del Nadal de 2024 només se'n veu el final. Si la caiguda és
    retard en la retirada, les primeres setmanes de gener haurien de tenir **més** fallades que la
    mitjana del trimestre (els discs que no es van retirar per festes).
    """)
    return


@app.cell
def _(pd, weekly):
    _q1 = weekly[(weekly.week_end > pd.Timestamp("2025-01-01")) & (weekly.week_end <= pd.Timestamp("2025-04-06"))]
    _mean = _q1.n_fail.sum() / _q1.n_active.sum() * 1e5
    n2_table = _q1[["week_end", "n_days", "n_fail", "rate_fail", "rate_err_inc"]].assign(
        rate_fail_vs_q1=lambda x: x.rate_fail / _mean).round(
        {"rate_fail": 2, "rate_err_inc": 1, "rate_fail_vs_q1": 2}).head(8)
    n2_table
    return (n2_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N3 · Retard en la retirada o menys fallades reals?

    Per a cada disc que falla, `daily_stats.py` guarda:

    - `gap_before`: dies entre l'observació anterior i el dia de la fallada (1 = hi era el dia abans).
    - `absent_30`: dies absents dels 30 anteriors.
    - Dies entre la primera aparició d'errors (`onset_date`, pas de 0 a > 0 observat) i la fallada.

    Si per Nadal es retarda la retirada, dels discs que comencen a tenir errors en una setmana una
    proporció menor hauria de constar com a fallada en els 14 dies següents, i les fallades que sí
    consten haurien de ser les més evidents (amb errors) o de discs que ja feia dies que fallaven
    (absències abans de la fallada).

    *Atenció:* els dies entre la primera pujada i la fallada no es poden comparar entre mesos, perquè
    el panel comença l'1-1-2025 i el retard màxim observable creix amb la data (truncament per
    l'esquerra). Per això la figura fa servir proporcions.
    """)
    return


@app.cell
def _(fails, pd, period):
    _f = fails.copy()
    _f["onset_to_fail"] = (_f.fail_date - _f.onset_date).dt.days
    _f["since_inc"] = (_f.fail_date - _f.last_inc_date).dt.days
    _f["period"] = _f.fail_date.map(period)
    _f["month"] = _f.fail_date.dt.to_period("M")
    n3_month = _f.groupby("month").agg(
        fallades=("serial_number", "size"),
        amb_errors=("err_at_fail", lambda s: round((s > 0).mean(), 3)),
        absent_dia_abans=("gap_before", lambda s: round((s > 1).mean(), 3)),
        absent_30_algun=("absent_30", lambda s: round((s > 0).mean(), 3)),
        onset_observat=("onset_to_fail", lambda s: round(s.notna().mean(), 3)),
        pujada_ult_30_dies=("since_inc", lambda s: round((s <= 30).mean(), 3)),
    )
    n3_month.index = n3_month.index.astype(str)
    fails_ext = _f
    n3_month
    return fails_ext, n3_month


@app.cell
def _(CUTOFF, drives, onsets, pd):
    # Of the drives whose error counters first go up in a given week, which share
    # is recorded as failed within 14 days? Needs 14 days of follow-up before the cutoff.
    FOLLOW = 14
    _o = onsets.merge(drives[["serial_number", "fail_date"]], on="serial_number", how="left")
    _o = _o[_o.onset_date <= CUTOFF - pd.Timedelta(days=FOLLOW)]
    _o["fail_in"] = ((_o.fail_date > _o.onset_date) & (_o.fail_date <= _o.onset_date + pd.Timedelta(days=FOLLOW)))
    _o["week"] = _o.onset_date.dt.to_period("W-SUN").dt.end_time.dt.normalize()
    conv = _o.groupby("week").agg(onsets=("fail_in", "size"), fail_14=("fail_in", "sum"))
    conv["share"] = conv.fail_14 / conv.onsets
    conv = conv.reset_index()
    return FOLLOW, conv


@app.cell
def _(C, FOLLOW, US_HOLIDAYS, XMAS, conv, fails_ext, month_axis, np, pd, plt, shade_holidays):
    fig_n3, _axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    _ax = _axes[0]
    shade_holidays(_ax, XMAS, US_HOLIDAYS)
    _c = conv[conv.week > conv.week.min()]  # first week: onsets of drives already broken
    _x = _c.week - pd.Timedelta(days=3)
    _se = np.sqrt(_c.share * (1 - _c.share) / _c.onsets)
    _ax.fill_between(_x, (_c.share - 1.96 * _se).clip(0), _c.share + 1.96 * _se, color=C["blue"], alpha=0.15, lw=0)
    _ax.plot(_x, _c.share, color=C["blue"], marker="o", ms=3.5)
    _ax.set_title(f"Discs amb primera pujada d'errors aquella setmana que consten com a fallada en {FOLLOW} dies")
    _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _ax.set_ylim(bottom=0)

    _ax = _axes[1]
    shade_holidays(_ax, XMAS, US_HOLIDAYS, label=False)
    _f = fails_ext[fails_ext.fail_date < pd.Timestamp("2026-01-01")]
    _m = _f.groupby(_f.fail_date.dt.to_period("M")).agg(
        err=("err_at_fail", lambda s: (s > 0).mean()),
        abs30=("absent_30", lambda s: (s > 0).mean()),
        gap=("gap_before", lambda s: (s > 1).mean()))
    _xm = _m.index.to_timestamp() + pd.Timedelta(days=14)
    for _c, _lab, _color in [("err", "amb algun comptador d'error > 0", C["orange"]),
                             ("abs30", "amb algun dia absent als 30 dies anteriors", C["aqua"]),
                             ("gap", "absent el dia abans de la fallada", C["violet"])]:
        _ax.plot(_xm, _m[_c], color=_color, marker="o", ms=4, label=_lab)
    _ax.set_title("Fallades de cada mes: proporció...")
    _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _ax.set_ylim(0, 1)
    _ax.legend(loc="center left", fontsize=8)
    month_axis(_ax)
    fig_n3.tight_layout()
    fig_n3
    return (fig_n3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N4 · Etiquetes de desenvolupament afectades

    Per a cada data de referència de desenvolupament L (5-2 … 5-11-2025): taxa de fallada a t dies
    entre els discs actius a L (vistos abans i després de L, sense fallada fins a L). La línia de
    referència és la mitjana de les dates la finestra de les quals no toca la temporada de festes
    (17-11-2025 – 4-1-2026); la diferència dona les fallades que «falten».
    """)
    return


@app.cell
def _(DEV_LANDMARKS, HORIZONS, SEASON, drives, np, pd):
    _fs = drives.first_seen.to_numpy()
    _ls = drives.last_seen.to_numpy()
    _fd = drives.fail_date.to_numpy()
    _rows = []
    for _L in DEV_LANDMARKS:
        _L64 = np.datetime64(_L)
        _act = (_fs <= _L64) & (_ls >= _L64) & ~(_fd <= _L64)
        _r = {"landmark": _L, "n": int(_act.sum())}
        for _t in HORIZONS:
            _end = np.datetime64(_L + pd.Timedelta(days=_t))
            _r[f"fail_{_t}"] = int((_act & (_fd > _L64) & (_fd <= _end)).sum())
            # days of the window (L, L+t] inside the holiday season
            _r[f"xmas_days_{_t}"] = max(0, (min(_L + pd.Timedelta(days=_t), SEASON[1])
                                            - max(_L, SEASON[0] - pd.Timedelta(days=1))).days)
        _rows.append(_r)
    n4 = pd.DataFrame(_rows)
    for _t in HORIZONS:
        n4[f"rate_{_t}"] = n4[f"fail_{_t}"] / n4.n * 1e3
        _clean = n4[f"xmas_days_{_t}"] == 0
        _base = n4.loc[_clean, f"fail_{_t}"].sum() / n4.loc[_clean, "n"].sum()
        n4[f"missing_{_t}"] = (_base * n4.n - n4[f"fail_{_t}"]).where(~_clean, 0).round(1)
    n4_summary = pd.DataFrame({
        "horitzó": HORIZONS,
        "dates_afectades": [int((n4[f"xmas_days_{t}"] > 0).sum()) for t in HORIZONS],
        "fallades_etiqueta": [int(n4[f"fail_{t}"].sum()) for t in HORIZONS],
        "fallades_que_falten": [round(n4[f"missing_{t}"].sum(), 1) for t in HORIZONS],
    })
    n4_summary["%_que_falten"] = (100 * n4_summary.fallades_que_falten
                                  / (n4_summary.fallades_etiqueta + n4_summary.fallades_que_falten)).round(1)
    n4_summary
    return n4, n4_summary


@app.cell
def _(C, HORIZONS, mdates, n4, plt):
    fig_n4, _axes = plt.subplots(1, 3, figsize=(10, 3.3), sharex=True)
    for _ax, _t, _color in zip(_axes, HORIZONS, [C["blue"], C["orange"], C["aqua"]]):
        _clean = n4[f"xmas_days_{_t}"] == 0
        _base = n4.loc[_clean, f"fail_{_t}"].sum() / n4.loc[_clean, "n"].sum() * 1e3
        _ax.axhline(_base, color=C["ink2"], lw=1, ls="--")
        _ax.plot(n4.landmark, n4[f"rate_{_t}"], color=_color, marker="o", ms=3)
        _ax.plot(n4.landmark[~_clean], n4[f"rate_{_t}"][~_clean], ls="none", marker="o", ms=6,
                 mfc="white", mec=_color)
        _ax.set_title(f"t = {_t} dies")
        _ax.set_ylim(bottom=0)
        _ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 5, 8, 11]))
        _ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    _axes[0].set_ylabel("fallades per 1.000 discs")
    fig_n4.suptitle("Taxa de l'etiqueta per data de desenvolupament · cercles buits: finestra que toca les festes (17-11 – 4-1)",
                    x=0.01, ha="left", fontsize=10)
    fig_n4.tight_layout()
    fig_n4
    return (fig_n4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N5 · Variables de les primeres dates de test

    Les dates de test de gener llegeixen 7 i 30 dies de desembre. Proporció de discs actius (vistos fa
    30 dies o més) amb algun dia absent i amb alguna pujada de comptador en la finestra anterior, per a
    cada dimecres. Si al desembre canvien, el model veurà al test una distribució diferent de la
    d'entrenament. El salt de maig–juny és la caiguda de reporting del 12-5-2025, no un canvi real.
    """)
    return


@app.cell
def _(C, US_HOLIDAYS, XMAS, marks, month_axis, pd, plt, shade_holidays):
    _m = marks[marks.date >= marks.date.min() + pd.Timedelta(days=30)]
    fig_n5, _axes = plt.subplots(1, 2, figsize=(10, 3.3), sharex=True)
    for _ax, _cols, _title in [
        (_axes[0], [("share_absent_7", "7 dies", C["blue"]), ("share_absent_30", "30 dies", C["orange"])],
         "Discs amb algun dia absent a la finestra"),
        (_axes[1], [("share_inc_7", "7 dies", C["blue"]), ("share_inc_30", "30 dies", C["orange"])],
         "Discs amb alguna pujada de comptador a la finestra"),
    ]:
        shade_holidays(_ax, XMAS, US_HOLIDAYS, label=False)
        for _c, _lab, _color in _cols:
            _ax.plot(_m.date, _m[_c], color=_color, marker="o", ms=3, label=_lab)
        _ax.set_title(_title)
        _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        _ax.set_ylim(bottom=0)
        _ax.legend(loc="upper left")
        month_axis(_ax)
    fig_n5.tight_layout()
    fig_n5
    return (fig_n5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions (20-11-2026, dades fins al 4-1-2026)

    **N1. La caiguda és gran i comença per Thanksgiving, no per Nadal.** Del 17-11-2025 al 4-1-2026 la
    taxa de fallada és d'1,80 per 100.000 discs-dia, contra 4,14 les set setmanes anteriors (−57 %).
    Les dues últimes setmanes de desembre baixa un 90 %. En canvi, les pujades de comptadors d'error
    **augmenten** un 10 % i les aparicions noves d'errors es mantenen (unes 10 per 100.000 discs-dia
    cada setmana). Els discs es continuen deteriorant igual: el que baixa és quants se'n declaren com a fallats.

    **Dia de la setmana.** Les fallades es registren un 35–38 % menys en diumenge i dilluns i un 33 % més en
    divendres. Les sortides sense fallada es concentren en dilluns (×4,7). Les pujades de comptadors són
    planes (0,98–1,02). L'etiqueta segueix l'horari del personal. Amb dates de referència setmanals
    sempre el mateix dia, cada finestra conté el mateix nombre de caps de setmana i això no esbiaixa res.

    **N2. No es veu acumulació clara després de festes al gener de 2025.** Les dues primeres setmanes
    senceres estan un 6 % i un 10 % per sobre de la mitjana del Q1, dins de la variació setmanal habitual.
    Només en veiem el final (les dades comencen l'1-1-2025), així que és una prova feble.

    **N3. És retard en la retirada.** Dels discs que comencen a tenir errors al desembre, només un 1,0 %
    consten com a fallats en 14 dies, contra un 1,8–2,2 % de setembre a novembre. Les fallades que sí
    es registren al desembre són més «evidents»: un 82 % tenen algun comptador d'error > 0 (contra un 69–73 %),
    un 38 % han estat absents algun dia del mes anterior (contra un 28–31 %) i un 23 % ho estaven el dia
    abans (contra un 16–19 %). Els discs dubtosos es deixen per després de festes.

    **N4. Etiquetes de desenvolupament afectades.** A 7 dies, cap. A 30 dies, 3 dates (22-10 a 5-11) i unes
    150 fallades que falten (1 % de les positives). A 60 dies, 7 dates (24-9 a 5-11) i unes 1.000 fallades
    que falten (3,3 % de totes les positives, però fins a un 45 % a la data del 5-11). Aquestes fallades no
    són negatius reals: són discs deteriorats que es van retirar més tard.

    **N5. Variables del primer test.** La proporció de discs amb algun dia absent als 30 dies anteriors
    passa d'un 1 % a un 1,5 % durant les festes. Les pujades de comptadors no canvien. El desplaçament és
    petit. El problema més gran és la caiguda de reporting del **12-5-2025** (29 % dels discs absents un
    sol dia): s'han d'ignorar els dies d'absència causats per caigudes generals.

    ### Com incorporem l'efecte als models

    1. **Censura administrativa al 16-11-2025 per al desenvolupament** (variant de l'opció O2). Per a
       l'horitzó t només s'entrenen les dates L amb L + t ≤ 16-11-2025. Al model per franges, les franges
       posteriors al 16-11 es tracten com a censurades. La censura depèn només del calendari, no de l'estat del
       disc, així que no esbiaixa el risc. Cost: 0, 3 i 7 de les 40 dates a 7, 30 i 60 dies. Treure només
       les positives que falten seria pitjor: la regla ha de treure la data sencera.
    2. **No fem servir una variable de calendari (O4).** Les finestres de test (gener–juny de 2026) no
       toquen cap festa, així que la variable seria constant al test. Entrenant sense etiquetes de festes,
       la probabilitat ja queda calibrada per a l'operativa normal.
    3. **El test comença el 7-1 com estava previst (no O3).** El gener de 2025 no mostra acumulació. Es
       manté com a comprovació de sensibilitat treure les dues primeres dates de test.
    4. **Dies d'absència com a variable:** es compten sense els dies de caiguda general (dies en què falta
       més del 5 % del parc, com el 12-5-2025).
    5. **Per a l'aplicació:** l'etiqueta és una decisió operativa. La figura setmanal va a la diapositiva 2.
    """)
    return


@app.cell
def _(BASE, fig_dow, fig_n3, fig_n4, fig_n5, fig_weekly):
    # PNG copies of the figures for the slides
    _dir = BASE / "out" / "figures"
    _dir.mkdir(parents=True, exist_ok=True)
    for _name, _fig in [("nadal_setmanal", fig_weekly), ("dia_setmana", fig_dow), ("nadal_retard", fig_n3),
                        ("nadal_etiquetes", fig_n4), ("nadal_variables", fig_n5)]:
        _fig.savefig(_dir / f"{_name}.png", dpi=150, bbox_inches="tight")
    return


if __name__ == "__main__":
    app.run()
