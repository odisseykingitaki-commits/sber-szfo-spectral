"""KMeans(K=2) vs threshold (основной) AND vs Louvain (labels_louvain.npy).

Features: no-income (expect p=17); income cols excluded if present.
Writes: results/kmeans_vs_ours_no_income.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import DATA_PROC, RESULTS, feature_cols  # noqa: E402

FEAT_PATH = DATA_PROC / "features_szfo_v2_final.csv"
THRESH_CSV = DATA_PROC / "clusters_final_threshold.csv"
THRESH_LAB = RESULTS / "labels_threshold.npy"
LOUVAIN_LAB = RESULTS / "labels_louvain.npy"
_LEGACY_LOUVAIN = RESULTS / "labels_final.npy"  # deprecated
OUT_JSON = RESULTS / "kmeans_vs_ours_no_income.json"

INCOME_COL_SUBSTR = ("income", "urov", "зарплат", "доход")
C_REG = 0.2


def exclude_income(cols: list[str]) -> list[str]:
    kept = []
    dropped = []
    for c in cols:
        low = c.lower()
        if any(s in low for s in INCOME_COL_SUBSTR):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


def metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "SW": float(silhouette_score(X, labels)),
        "CH": float(calinski_harabasz_score(X, labels)),
        "DB": float(davies_bouldin_score(X, labels)),
        "sizes": np.bincount(labels.astype(int)).tolist(),
        "n_unique": int(len(np.unique(labels))),
    }


def recompute_threshold(X: np.ndarray) -> np.ndarray:
    """Same as 11_final_clusters.py: median threshold on U1 = X @ v1."""
    from scipy.linalg import eigh
    from utils import build_J

    X_bin = (X > 0).astype(int)
    J = build_J(X_bin, C_reg=C_REG)
    evals, evecs = eigh(J)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    v1 = evecs[:, 0]
    U1 = X @ v1
    return (U1 > np.median(U1)).astype(int)


def load_threshold_labels(N: int, X: np.ndarray) -> tuple[np.ndarray, str]:
    if THRESH_LAB.exists():
        lab = np.load(THRESH_LAB)
        if len(lab) != N:
            raise SystemExit(f"{THRESH_LAB.name} len={len(lab)} != N={N}")
        return lab.astype(int), f"npy:{THRESH_LAB.name}"
    if THRESH_CSV.exists():
        tdf = pd.read_csv(THRESH_CSV, encoding="utf-8-sig")
        for col in ("cluster", "new_cluster", "threshold", "labels"):
            if col in tdf.columns:
                lab = tdf[col].to_numpy()
                if len(lab) != N:
                    raise SystemExit(
                        f"threshold CSV col={col} len={len(lab)} != N={N}"
                    )
                return lab.astype(int), f"csv:{THRESH_CSV.name}:{col}"
        raise SystemExit(
            f"No cluster column in {THRESH_CSV}; cols={list(tdf.columns)}"
        )
    print("WARN: threshold labels missing — recomputing like 11")
    return recompute_threshold(X), "recomputed:median_threshold_U1"


def load_louvain_labels(N: int) -> tuple[np.ndarray, str]:
    path = LOUVAIN_LAB if LOUVAIN_LAB.exists() else _LEGACY_LOUVAIN
    if not path.exists():
        raise SystemExit(f"Missing {LOUVAIN_LAB} (and legacy {_LEGACY_LOUVAIN})")
    louvain = np.load(path)
    if len(louvain) != N:
        raise SystemExit(f"{path.name} len={len(louvain)} != N={N}")
    return louvain.astype(int), path.name


def print_block(title: str, block: dict) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"N={block['N']}  p={block['p']}  feature_list_len={block['feature_list_len']}")
    print(f"features={block['features']}")
    km = block["kmeans"]
    ours = block["ours"]
    print(f"KMeans K=2:  SW={km['SW']:+.6f}  CH={km['CH']:.4f}  DB={km['DB']:.6f}  sizes={km['sizes']}")
    print(
        f"{block['ours_name']}: SW={ours['SW']:+.6f}  CH={ours['CH']:.4f}  "
        f"DB={ours['DB']:.6f}  sizes={ours['sizes']}"
    )
    print(f"ARI(ours, KMeans)={block['ARI']:.6f}")


def main() -> None:
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig")
    feat_raw = feature_cols(df)
    feat, dropped = exclude_income(feat_raw)
    if dropped:
        print(f"Excluded income-like cols: {dropped}")
    else:
        print("No income-like cols in feature list (good, no-income lock).")

    X_raw = df[feat].values
    N, p = X_raw.shape
    X = StandardScaler().fit_transform(X_raw)
    print(f"N={N}, p={p}, feature_list_len={len(feat)}")
    assert p == 17, f"Expected p=17 no-income, got p={p}: {feat}"

    # --- KMeans baseline (same for both comparisons) ---
    km = KMeans(n_clusters=2, n_init=30, random_state=42).fit_predict(X)
    km_m = metrics(X, km)

    # --- 1) Threshold (основной) ---
    thr, thr_src = load_threshold_labels(N, X)
    # sanity: recompute and compare
    thr_re = recompute_threshold(X)
    ari_thr_re = float(adjusted_rand_score(thr, thr_re))
    print(f"Threshold source={thr_src}; ARI(csv, recompute)={ari_thr_re:.6f}")

    thr_m = metrics(X, thr)
    ari_thr_km = float(adjusted_rand_score(thr, km))
    block_thr = {
        "ours_name": "threshold_median_U1 (ОСНОВНОЙ)",
        "ours_source": thr_src,
        "N": int(N),
        "p": int(p),
        "feature_list_len": int(len(feat)),
        "features": feat,
        "kmeans": km_m,
        "ours": thr_m,
        "ARI": ari_thr_km,
        "ARI_csv_vs_recompute": ari_thr_re,
    }
    print_block("1) THRESHOLD (основной) vs KMeans K=2", block_thr)

    # --- 2) Louvain labels_louvain.npy ---
    louvain, lv_src = load_louvain_labels(N)
    print(f"\nLouvain labels: {lv_src}")
    print(f"  len={len(louvain)}  N={N}  match={len(louvain) == N}")
    print(f"  unique={np.unique(louvain).tolist()}  sizes={np.bincount(louvain.astype(int)).tolist()}")

    ari_lv_thr = float(adjusted_rand_score(louvain, thr))
    identical_to_thr = bool(np.array_equal(louvain.astype(int), thr.astype(int)))
    print(f"  ARI(Louvain, threshold)={ari_lv_thr:.6f}  identical_to_threshold={identical_to_thr}")
    print("  NOTE: Louvain from 06_network_louvain.py (NOT основной метод).")

    lv_m = metrics(X, louvain)
    ari_lv_km = float(adjusted_rand_score(louvain, km))
    block_lv = {
        "ours_name": "Louvain labels_louvain.npy (НЕ основной)",
        "ours_source": lv_src,
        "N": int(N),
        "p": int(p),
        "feature_list_len": int(len(feat)),
        "features": feat,
        "kmeans": km_m,
        "ours": lv_m,
        "ARI": ari_lv_km,
        "ARI_vs_threshold": ari_lv_thr,
        "identical_to_threshold": identical_to_thr,
        "len_matches_N": True,
        "verified_as": "Louvain from 06 (labels_louvain.npy), after no-income p=17",
    }
    print_block("2) LOUVAIN (labels_louvain.npy) vs KMeans K=2", block_lv)

    out = {
        "note": (
            "Основной метод = median threshold on U1 (11_final_clusters.py) → labels_threshold.npy. "
            "Louvain = labels_louvain.npy from 06_network_louvain.py. "
            "Старое имя labels_final.npy было Louvain (не threshold)."
        ),
        "features_path": str(FEAT_PATH),
        "N": int(N),
        "p": int(p),
        "feature_list_len": int(len(feat)),
        "features": feat,
        "income_cols_dropped": dropped,
        "kmeans": {
            "K": 2,
            "n_init": 30,
            "random_state": 42,
            **km_m,
        },
        "comparison_threshold_vs_kmeans": {
            "method": "median_threshold_U1_PLM_v1",
            "is_primary": True,
            "label_source": thr_src,
            "ours": thr_m,
            "kmeans": km_m,
            "ARI": ari_thr_km,
            "ARI_csv_vs_recompute": ari_thr_re,
        },
        "comparison_louvain_vs_kmeans": {
            "method": "Louvain_labels_louvain_npy",
            "is_primary": False,
            "label_source": lv_src,
            "ours": lv_m,
            "kmeans": km_m,
            "ARI": ari_lv_km,
            "ARI_vs_threshold": ari_lv_thr,
            "identical_to_threshold": identical_to_thr,
            "len_matches_N": True,
        },
        "verification": {
            "labels_louvain_len": int(len(louvain)),
            "N": int(N),
            "len_matches_N": True,
            "p_is_17_no_income": p == 17,
            "labels_louvain_is_not_threshold": not identical_to_thr,
            "ARI_louvain_vs_threshold": ari_lv_thr,
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
