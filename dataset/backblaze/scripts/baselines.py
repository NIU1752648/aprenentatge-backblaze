from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import sys
from pathlib import Path
from datetime import date

# Importar funcions del marc d'avaluació existent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cv_models import CUTOFF, budget_metrics, load_drive_status, strict_labels

ERR_IDS = ["5", "187", "188", "197", "198"]
COUNTERS = [f"smart_{i}_raw" for i in ERR_IDS]
D7_COUNTERS = [f"d7_{i}" for i in ERR_IDS]

def load_dev_data(landmarks_dir: Path) -> pd.DataFrame:

    """

    Llegeix els fitxers .parquet de dates de referència, filtra les files del
    conjunt de desenvolupament (split == "dev") i les concatena en un sol DataFrame.
    
    """

    files = sorted(landmarks_dir.glob("*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df = df[df["split"] == "dev"].copy()
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def compute_model_age_baseline(df: pd.DataFrame, target_col: str = "y_30", n_splits: int = 3) -> np.ndarray:
    
    """
    Calcula una puntuació de risc basada en la taxa de fallada per model i tram d'edat
    mitjançant validació creuada agrupada per disc (GroupKFold sobre serial_number),
    amb suavitzat cap a la taxa global per evitar soroll en combinacions amb poques dades.

    """

    scores = np.zeros(len(df), dtype=np.float64)
    age_bin = np.floor(df["age_years"].fillna(0)).astype(int)
    
    temp = pd.DataFrame({
        "model": df["model"],
        "age_bin": age_bin,
        "target": df[target_col]
    })
    
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(df, groups=df["serial_number"]):
        train_df = temp.iloc[train_idx]
        val_df = temp.iloc[val_idx]
        
        global_rate = train_df["target"].mean()
        # Taxa amb suavitzat lleuger (p. ex. m=10) per evitar soroll en combinacions rares
        stats = train_df.groupby(["model", "age_bin"])["target"].agg(["count", "sum"])
        stats["rate"] = (stats["sum"] + 10 * global_rate) / (stats["count"] + 10)
        
        val_merged = val_df.merge(stats["rate"], on=["model", "age_bin"], how="left")
        scores[val_idx] = val_merged["rate"].fillna(global_rate).to_numpy()
        
    return scores

import time


def main():
    t0 = time.perf_counter()
    print("[baselines] Iniciant càlcul de referències sense aprenentatge...", flush=True)

    rng = np.random.default_rng(42)
    landmarks_dir = Path("out/landmarks")
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    print(f"[baselines] Carregant dades de desenvolupament des de {landmarks_dir}...", flush=True)
    df = load_dev_data(landmarks_dir)
    print(f"[baselines] Dades carregades: {len(df):,} files ({time.perf_counter() - t0:.1f} s)", flush=True)

    DAY0 = date(2025, 1, 1)

    d_col = pd.to_datetime(df["date"]).dt.date
    df["day"] = np.fromiter(((d - DAY0).days for d in d_col), dtype=np.int32, count=len(df))

    data_dir = Path("data")
    print(f"[baselines] Carregant estat dels discs des de {data_dir}...", flush=True)
    status = load_drive_status(data_dir, CUTOFF)

    drive_idx = status.index.get_indexer(df["serial_number"].astype(str))
    assert (drive_idx >= 0).all(), "Hi ha números de sèrie que no són a drives.parquet"
    df["drive"] = drive_idx.astype(np.int32)

    print("[baselines] Calculant etiquetes estrictes...", flush=True)
    labels = strict_labels(df, status)
    for col, arr in labels.items():
        df[col] = arr

    # Omplir NaNs dels comptadors amb 0
    for c in COUNTERS + D7_COUNTERS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    print("[baselines] Calculant puntuacions R1, R2 i R3...", flush=True)
    r1 = (df[COUNTERS] > 0).sum(axis=1).to_numpy(dtype=np.float64)
    r2 = (df[D7_COUNTERS] > 0).sum(axis=1).to_numpy(dtype=np.float64)
    r3 = np.log(1.0 + df[COUNTERS].clip(lower=0)).sum(axis=1).to_numpy(dtype=np.float64)

    print("[baselines] Calculant taxa per model i edat (3-fold GroupKFold)...", flush=True)
    t_ma = time.perf_counter()
    model_age = compute_model_age_baseline(df, target_col="y_30", n_splits=3)
    print(f"[baselines] Taxa per model i edat calculada ({time.perf_counter() - t_ma:.1f} s)", flush=True)

    raw_scores = {
        "R1_regla_backblaze": r1,
        "R2_pujada_7d": r2,
        "R3_suma_log1p": r3,
        "Taxa_model_edat": model_age,
    }

    # Afegir soroll infinitesimal per desempatar a l'atzar de manera reproduïble
    scores = {
        name: s + rng.uniform(0, 1e-9, size=len(s))
        for name, s in raw_scores.items()
    }

    targets = [k for k in labels.keys() if k.startswith("y_")]
    print(f"[baselines] Avaluant {len(scores)} referències en {len(targets)} objectius...", flush=True)

    results = []
    for ref_name, score_arr in scores.items():
        print(f"[baselines]   -> Avaluant {ref_name}...", flush=True)
        df["_score"] = score_arr
        for target in targets:
            # Filtrar files vàlides per a aquesta franja/horitzó (si hi ha NaNs per censura)
            valid = df[df[target].notna()]
            y_true = valid[target].to_numpy(dtype=int)
            y_score = valid["_score"].to_numpy()

            # Mètriques per dimecres (0.1% i 1%)
            b_metrics = budget_metrics(y_true, y_score, valid["date"], budgets=[0.1, 1.0])  

            # Mètriques globals d'ordenació
            pr_auc = average_precision_score(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)

            results.append({
                "referencia": ref_name,
                "objectiu": target,
                "recall_0.1pct": b_metrics[0.1]["recall"],
                "recall_1pct": b_metrics[1.0]["recall"],
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            })

    out_path = out_dir / "baselines_dev.csv"
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)
    print(f"[baselines] Fet! Resultats desats a {out_path} en {time.perf_counter() - t0:.1f} s", flush=True)
    print(res_df.to_string(index=False), flush=True)
    

if __name__ == "__main__":
    main()


