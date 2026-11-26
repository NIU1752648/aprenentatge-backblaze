from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, _tree
from matplotlib.patches import FancyBboxPatch

# Importar funcions del marc d'avaluació existent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cv_models import CUTOFF, DAY0, load_drive_status, strict_labels

# Comptadors clau de Backblaze (G2 i G3)
ERR_IDS = ["5", "187", "188", "197", "198"]
COUNTERS = [f"smart_{i}_raw" for i in ERR_IDS]
D7_COUNTERS = [f"d7_{i}" for i in ERR_IDS]
ALL_COUNTER_COLS = COUNTERS + D7_COUNTERS

# Columnes de metadades/etiquetes que NO són predictors
META_COLS = {
    "date",
    "serial_number",
    "split",
    "t_obs",
    "event",
    "y_7",
    "y_30",
    "y_60",
    "y_0_7",
    "y_7_30",
    "y_30_60",
}


def load_dev_sample( landmarks_dir: Path, drives_path: Path, neg_frac: float = 0.05, random_state: int = 0 ) -> pd.DataFrame:
    
    """
    
    Llegeix els fitxers L=YYYY-MM-DD.parquet del conjunt de desenvolupament (split == 'dev'),
    calcula les etiquetes estrictes a 30 dies (y_30) i retorna totes les fallades (y_30 == 1)
    més una mostra aleatòria del 5 % dels discs sans (y_30 == 0).
    
    """

    files = sorted(landmarks_dir.glob("L=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No s'han trobat fitxers L=*.parquet a {landmarks_dir.resolve()}")

    status = load_drive_status(drives_path, cutoff=CUTOFF)
    rng = np.random.default_rng(random_state)

    pos_parts: list[pd.DataFrame] = []
    neg_parts: list[pd.DataFrame] = []

    print(f"[arbre] Processant {len(files)} fitxers de landmarks...", flush=True)
    for f in files:
        df = pd.read_parquet(f)
        if "split" not in df.columns:
            continue

        dev_df = df[df["split"] == "dev"].copy()
        if dev_df.empty:
            continue

        # Data de referència (del fitxer L=YYYY-MM-DD.parquet o de la columna 'date')
        if "date" in dev_df.columns:
            ref_dates = pd.to_datetime(dev_df["date"]).dt.date
        else:
            d_file = date.fromisoformat(f.stem.split("=")[-1])
            ref_dates = pd.Series([d_file] * len(dev_df), index=dev_df.index)
            dev_df["date"] = ref_dates

        day_arr = np.fromiter(((d - DAY0).days for d in ref_dates), dtype=np.int32, count=len(dev_df))
        drive_arr = status.index.get_indexer(dev_df["serial_number"].astype(str)).astype(np.int32)
        if (drive_arr < 0).any():
            raise ValueError(f"Hi ha serial_number no trobats a drives.parquet al fitxer {f.name}")

        data_ns = SimpleNamespace(day=day_arr, drive=drive_arr)
        labels = strict_labels(data_ns, status)
        dev_df["y_30"] = labels["y_30"]

        # Descartar files censurades o fora d'abast (y_30 == NaN)
        dev_df = dev_df[dev_df["y_30"].notna()].copy()
        dev_df["y_30"] = dev_df["y_30"].astype(int)

        pos = dev_df[dev_df["y_30"] == 1]
        neg = dev_df[dev_df["y_30"] == 0]

        if not pos.empty:
            pos_parts.append(pos)
        if not neg.empty:
            seed = int(rng.integers(0, 2**32 - 1))
            neg_parts.append(neg.sample(frac=neg_frac, random_state=seed))

    if not pos_parts and not neg_parts:
        raise ValueError("No s'han trobat files vàlides amb split == 'dev'.")

    sample_df = pd.concat(pos_parts + neg_parts, ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sample_df


def prepare_features( df: pd.DataFrame, exclude_counters: bool = False ) -> tuple[pd.DataFrame, list[str]]:
    
    """
    
    Prepara la matriu X per a DecisionTreeClassifier:
    - Exclou metadades i etiquetes.
    - Opcionalment exclou els 5 comptadors i les seves variacions d7_*.
    - Codifica columnes categòriques (manufacturer, model) amb codis numèrics.
    - Imputa NaN amb 0 als comptadors i amb la mediana a la resta.

    """

    candidate_cols = [c for c in df.columns if c not in META_COLS]
    if exclude_counters:
        candidate_cols = [c for c in candidate_cols if c not in ALL_COUNTER_COLS]

    X = df[candidate_cols].copy()

    # Codificar categòriques (p. ex. manufacturer, model) si n'hi ha
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].fillna("DESCONEGUT").astype("category").cat.codes

    # Omplir NaN: 0 per als comptadors, mediana per a la resta
    for col in X.columns:
        if col in ALL_COUNTER_COLS:
            X[col] = X[col].fillna(0.0)
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(0.0 if pd.isna(median_val) else median_val)

    return X, list(X.columns)


def train_and_plot_tree( X: pd.DataFrame, y: pd.Series,feature_names: list[str], output_path: Path, title: str, max_depth: int = 3, min_samples_leaf: int = 100, random_state: int = 0 ) -> DecisionTreeClassifier:
    
    """
    
    Entrena un arbre poc profund, imprimeix les regles per consola i desa el PNG.
    
    """

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X, y)

    print(f"\n--- Regles de l'arbre: {title} ---")
    print(export_text(clf, feature_names=feature_names, decimals=3))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(20, 10), dpi=300)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Sa", "Fallada 30d"],
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        fontsize=10,
    )
    plt.title(title, fontsize=14, pad=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[arbre] Figura desada a: {output_path}", flush=True)
    return clf


def _short_question(name: str, thresh: float) -> str:
    """Formula una pregunta en dues línies perquè càpiga bé dins la caixa."""
    smart_labels = {
        "5": "SMART 5\n(recol·locats)",
        "187": "SMART 187\n(no corregits)",
        "188": "SMART 188\n(timeouts)",
        "197": "SMART 197\n(pendents)",
        "198": "SMART 198\n(offline)",
        "192": "SMART 192",
    }
    short_code = {
        "5": "SMART 5",
        "187": "SMART 187",
        "188": "SMART 188",
        "197": "SMART 197",
        "198": "SMART 198",
        "192": "SMART 192",
    }

    is_zero_cut = abs(thresh - 0.5) < 0.12

    # 1. Variacions a 7d o 30d (d7_197, d7_smart_197_raw, d30_187...)
    for prefix, period in [
        ("d7_smart_", "canvi 7d"),
        ("d7_", "canvi 7d"),
        ("d30_smart_", "canvi 30d"),
        ("d30_", "canvi 30d"),
    ]:
        if name.startswith(prefix):
            code = name.removeprefix(prefix).removesuffix("_raw")
            s_lbl = short_code.get(code, f"SMART {code}")
            if is_zero_cut:
                return f"Sense {period}\n{s_lbl}?"
            return f"{period.capitalize()}\n{s_lbl} ≤ {thresh:.0f}?"

    # 2. Comptadors directes (smart_197_raw, etc.)
    if name.startswith("smart_") and name.endswith("_raw"):
        code = name.removeprefix("smart_").removesuffix("_raw")
        if is_zero_cut:
            return f"Sense errors\n{short_code.get(code, f'SMART {code}')}?"
        return f"{short_code.get(code, f'SMART {code}')}\n≤ {thresh:.0f}?"

    # 3. Qualsevol altra variable
    clean = name.replace("_", " ")
    return f"{clean}\n≤ {thresh:.1f}?"


def plot_clean_presentation_tree(
    clf: DecisionTreeClassifier,
    feature_names: list[str],
    output_path: Path,
    title: str,
    subtitle: str = "",
) -> None:
    """Dibuixa un arbre minimalista amb text ben encaixat dins cada node."""
    tree_ = clf.tree_
    feat_list = list(feature_names)

    nodes_by_depth: dict[int, list[int]] = {}

    def walk(node_id: int, depth: int) -> None:
        if depth > 3 or node_id == _tree.TREE_LEAF:
            return
        nodes_by_depth.setdefault(depth, []).append(node_id)
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED and depth < 3:
            walk(int(tree_.children_left[node_id]), depth + 1)
            walk(int(tree_.children_right[node_id]), depth + 1)

    walk(0, 0)

    coords: dict[int, tuple[float, float]] = {}
    max_d = max(nodes_by_depth.keys(), default=0)
    for d, n_ids in nodes_by_depth.items():
        y = 0.86 - d * 0.245
        n = len(n_ids)
        for i, nid in enumerate(n_ids):
            x = (i + 0.5) / n
            coords[nid] = (x, y)

    fig, ax = plt.subplots(figsize=(19, 9.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 1.05)
    ax.axis("off")

    box_h = 0.135
    box_w = 0.21 if max_d <= 2 else 0.114

    # Connexions netes (SÍ / NO)
    for nid, (x1, y1) in coords.items():
        if tree_.feature[nid] == _tree.TREE_UNDEFINED:
            continue
        for child_id, ans, col, dx_txt in [
            (int(tree_.children_left[nid]), "SÍ", "#2e7d32", -0.012),
            (int(tree_.children_right[nid]), "NO", "#c62828", 0.012),
        ]:
            if child_id not in coords:
                continue
            x2, y2 = coords[child_id]
            ax.annotate(
                "",
                xy=(x2, y2 + box_h / 2 + 0.008),
                xytext=(x1, y1 - box_h / 2 - 0.008),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col, mutation_scale=14),
            )
            mx, my = (x1 + x2) / 2 + dx_txt, (y1 + y2) / 2
            ax.text(
                mx,
                my,
                ans,
                fontsize=10.5,
                fontweight="bold",
                color=col,
                ha="right" if ans == "SÍ" else "left",
                va="center",
            )

    # Caixes amb text en 2 línies + percentatge a sota
    for nid, (x, y) in coords.items():
        counts = tree_.value[nid][0]
        tot = counts.sum()
        p_fail = (counts[1] / tot) if tot > 0 else 0.0
        is_leaf = tree_.feature[nid] == _tree.TREE_UNDEFINED

        if p_fail >= 0.50:
            fc, ec, tc = "#fdecea", "#c62828", "#9e1b1b"
        elif p_fail >= 0.20:
            fc, ec, tc = "#fff8e1", "#ef6c00", "#b25000"
        else:
            fc, ec, tc = "#edf7ed", "#2e7d32", "#1b5e20"

        patch = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.005,rounding_size=0.018",
            facecolor=fc,
            edgecolor=ec,
            linewidth=2.2 if not is_leaf else 1.8,
        )
        ax.add_patch(patch)

        if not is_leaf:
            q = _short_question(feat_list[tree_.feature[nid]], float(tree_.threshold[nid]))
            ax.text(
                x,
                y + 0.022,
                q,
                fontsize=9.2 if max_d > 2 else 11,
                fontweight="bold",
                color="#111111",
                ha="center",
                va="center",
                linespacing=1.25,
            )
            ax.text(
                x,
                y - 0.042,
                f"{p_fail * 100:.0f}% fallades",
                fontsize=9.0 if max_d > 2 else 10.5,
                fontweight="bold",
                color=tc,
                ha="center",
                va="center",
            )
        else:
            estat = "ALT RISC" if p_fail >= 0.50 else ("RISC MODERAT" if p_fail >= 0.20 else "BAIX RISC")
            ax.text(
                x,
                y + 0.022,
                estat,
                fontsize=9.5 if max_d > 2 else 11,
                fontweight="bold",
                color=tc,
                ha="center",
                va="center",
            )
            ax.text(
                x,
                y - 0.026,
                f"{p_fail * 100:.0f}% fallades",
                fontsize=9.5 if max_d > 2 else 11,
                color="#222222",
                ha="center",
                va="center",
            )

    ax.text(0.5, 1.00, title, fontsize=16, fontweight="bold", ha="center", va="center", color="#111111")
    if subtitle:
        ax.text(0.5, 0.955, subtitle, fontsize=11, ha="center", va="center", color="#555555")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[arbre] Figura neta desada a: {output_path}", flush=True)



def main() -> None:
    t0 = time.perf_counter()

    base_dir = Path(__file__).resolve().parent.parent
    landmarks_dir = base_dir / "out/landmarks"
    drives_path = base_dir / "data"
    figures_dir = base_dir / "out/figures"

    print("[arbre] Carregant mostra de desenvolupament...", flush=True)
    df_sample = load_dev_sample(
        landmarks_dir=landmarks_dir,
        drives_path=drives_path,
        neg_frac=0.05,
        random_state=42,
    )

    n_pos = int((df_sample["y_30"] == 1).sum())
    n_neg = int((df_sample["y_30"] == 0).sum())
    print(
        f"[arbre] Mostra carregada: {len(df_sample):,} files "
        f"({n_pos:,} fallades, {n_neg:,} sans).",
        flush=True,
    )

    y = df_sample["y_30"]

    # 1. Arbre complet (totes les variables)
    X_all, feat_all = prepare_features(df_sample, exclude_counters=False)
    clf_all = train_and_plot_tree(
        X=X_all,
        y=y,
        feature_names=feat_all,
        output_path=figures_dir / "arbre_poc_profund_raw.png",
        title="Arbre de decisió (max_depth=3) — Totes les variables",
        max_depth=3,
        min_samples_leaf=100,
        random_state=0,
    )
    plot_clean_presentation_tree(
        clf=clf_all,
        feature_names=feat_all,
        output_path=figures_dir / "arbre_poc_profund.png",
        title="Regles apreses amb totes les variables (profunditat màxima 3)",
        subtitle="Mostra dev (100% fallades 30d + 5% sans) · Llindars en escala log(1 + |x|)",
    )

    # 2. Opcional: Arbre sense els 5 comptadors (G0 + G1 + GO)
    X_no_cnt, feat_no_cnt = prepare_features(df_sample, exclude_counters=True)
    clf_no_cnt = train_and_plot_tree(
        X=X_no_cnt,
        y=y,
        feature_names=feat_no_cnt,
        output_path=figures_dir / "arbre_sense_comptadors_raw.png",
        title="Arbre de decisió (max_depth=3) — Sense els 5 comptadors (G0 + G1 + GO)",
        max_depth=3,
        min_samples_leaf=100,
        random_state=0,
    )
    plot_clean_presentation_tree(
        clf=clf_no_cnt,
        feature_names=feat_no_cnt,
        output_path=figures_dir / "arbre_sense_comptadors.png",
        title="Regles sense els 5 comptadors clau (G0 + G1 + GO)",
        subtitle="Mostra dev (100% fallades 30d + 5% sans) · Què mira quan falten els comptadors?",
    )

    elapsed = time.perf_counter() - t0
    print(f"\n[arbre] Completat en {elapsed:.1f} s.", flush=True)

if __name__ == "__main__":
    main()