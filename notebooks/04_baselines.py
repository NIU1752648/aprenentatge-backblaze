import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import marimo as mo

    return Path, mo, np, pd, plt


@app.cell
def _(Path, pd):
    _csv_path = Path("dataset/backblaze/out/baselines_dev.csv")
    if not _csv_path.exists():
        _csv_path = Path("out/baselines_dev.csv")

    df_baselines = pd.read_csv(_csv_path)
    _fig_dir = _csv_path.parent / "figures"
    _fig_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = _fig_dir
    return df_baselines, fig_dir


@app.cell
def _(df_baselines, mo):
    mo.md(
        """
        # Referències sense aprenentatge (`baselines_dev.csv`)

        Comparació de les referències inicials sobre el conjunt de desenvolupament (`split == "dev"`):
        detecció per pressupost d'alertes (0,1 % i 1 %) i mètriques globals d'ordenació (PR-AUC i ROC-AUC)[cite: 2, 6].
        """
    )
    mo.ui.table(df_baselines)
    return


@app.cell
def _(df_baselines, mo):
    _targets = ["Tots"] + sorted(df_baselines["objectiu"].unique().tolist())
    target_selector = mo.ui.dropdown(
        options=_targets,
        value="Tots",
        label="Filtra per objectiu / franja",
    )
    target_selector
    return (target_selector,)


@app.cell
def _(df_baselines, target_selector):
    if target_selector.value == "Tots":
        df_plot = df_baselines.copy()
    else:
        df_plot = df_baselines[df_baselines["objectiu"] == target_selector.value].copy()
    return (df_plot,)


@app.cell
def _(df_plot, fig_dir, np, plt, target_selector):
    _refs = df_plot["referencia"].unique().tolist()
    _targets = df_plot["objectiu"].unique().tolist()

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    _metrics = [
        ("recall_0.1pct", "Detecció amb pressupost del 0,1 % (fracció)", _axes[0]),
        ("recall_1pct", "Detecció amb pressupost de l'1 % (fracció)", _axes[1]),
    ]

    _x = np.arange(len(_targets))
    _n_refs = max(len(_refs), 1)
    _width = 0.8 / _n_refs

    for _col, _ylabel, _ax in _metrics:
        for _i, _ref in enumerate(_refs):
            _sub = df_plot[df_plot["referencia"] == _ref].set_index("objectiu").reindex(_targets)
            _offset = (_i - (_n_refs - 1) / 2.0) * _width
            _ax.bar(
                _x + _offset,
                _sub[_col].to_numpy(dtype=float),
                width=_width,
                label=_ref,
            )
        _ax.set_title(_ylabel)
        _ax.set_xlabel("Objectiu / franja temporal")
        _ax.set_ylabel("Detecció (fracció de fallades)")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_targets, rotation=20, ha="right")
        _ax.set_ylim(0, max(0.05, float(df_plot[_col].max()) * 1.18))
        _ax.grid(axis="y", linestyle="--", alpha=0.4)
        if len(_refs) > 1:
            _ax.legend(title="Referència", fontsize=8, title_fontsize=8)

    _fig.suptitle(f"Detecció per pressupost d'alertes — {target_selector.value}")
    _fig.tight_layout()

    _suffix = "tots" if target_selector.value == "Tots" else target_selector.value
    _out_file = fig_dir / f"baselines_deteccio_{_suffix}.png"
    _fig.savefig(_out_file, dpi=150)
    _fig
    return


@app.cell
def _(df_plot, fig_dir, np, plt, target_selector):
    _refs = df_plot["referencia"].unique().tolist()
    _targets = df_plot["objectiu"].unique().tolist()

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    _metrics = [
        ("pr_auc", "PR-AUC (àrea sota la corba precisió-detecció)", _axes[0]),
        ("roc_auc", "ROC-AUC (àrea sota la corba ROC)", _axes[1]),
    ]

    _x = np.arange(len(_targets))
    _n_refs = max(len(_refs), 1)
    _width = 0.8 / _n_refs

    for _col, _ylabel, _ax in _metrics:
        for _i, _ref in enumerate(_refs):
            _sub = df_plot[df_plot["referencia"] == _ref].set_index("objectiu").reindex(_targets)
            _offset = (_i - (_n_refs - 1) / 2.0) * _width
            _ax.bar(
                _x + _offset,
                _sub[_col].to_numpy(dtype=float),
                width=_width,
                label=_ref,
            )
        _ax.set_title(_ylabel)
        _ax.set_xlabel("Objectiu / franja temporal")
        _ax.set_ylabel("Puntuació (adimensional)")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_targets, rotation=20, ha="right")
        if _col == "roc_auc":
            _ax.set_ylim(0.5, 1.0)
        else:
            _ax.set_ylim(0, max(0.05, float(df_plot[_col].max()) * 1.18))
        _ax.grid(axis="y", linestyle="--", alpha=0.4)
        if len(_refs) > 1:
            _ax.legend(title="Referència", fontsize=8, title_fontsize=8)

    _fig.suptitle(f"Mètriques globals d'ordenació — {target_selector.value}")
    _fig.tight_layout()

    _suffix = "tots" if target_selector.value == "Tots" else target_selector.value
    _out_file = fig_dir / f"baselines_auc_{_suffix}.png"
    _fig.savefig(_out_file, dpi=150)
    _fig
    return


if __name__ == "__main__":
    app.run()
