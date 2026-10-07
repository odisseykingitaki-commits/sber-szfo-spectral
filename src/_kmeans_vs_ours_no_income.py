"""KMeans(K=2) vs threshold (основной) AND vs Louvain (labels_louvain.npy).

Основной метод = median threshold on U1 (11_final_clusters.py) → labels_threshold.npy.
Louvain = labels_louvain.npy (06). Старое labels_final.npy = Louvain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
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
from utils import DATA_PROC, RESULTS, build_J, feature_cols  # noqa: E402

FEAT_PATH = DATA_PROC / "features_szfo_v2_final.csv"
THRESH_CSV = DATA_PROC / "clusters_final_threshold.csv"
THRESH_LAB = RESULTS / "labels_threshold.npy"
LOUVAIN_LAB = RESULTS / "labels_louvain.npy"
_LEGACY_LOUVAIN = RESULTS / "labels_final.npy"
OUT_JSON = RESULTS / "kmeans_vs_ours_no_income.json"

C_REG = 0.2
INCOME_SUBSTR = ("income", "dohod", "зарплат", "wage", "salary")


def exclude_income(cols: list[str]) -> list[str]:
    kept, dropped = [], []
    for c in cols:
        cl = c.lower()
        if any(s in cl for s in INCOME_SUBSTR):
            dropped.append(c)
        else:
            kept.append(c)
    if dropped:
        print(f"Excluded income-like cols: {dropped}")
    return kept


def metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "SW": float(silhouette_score(X, labels)),
        "CH": float(calinski_harabasz_score(X, labels)),
        "DB": float(davies_bouldin_score(X, labels)),
        "n_clusters": int(len(np.unique(labels))),
        "sizes": np.bincount(labels.astype(int)).tolist(),
    }


def recompute_threshold(X: np.ndarray) -> np.ndarray:
    X_bin = (X > 0).astype(int)
    J = build_J(X_bin, C_reg=C_REG)
    evals, evecs = eigh(J)
    idx = np.argsort(evals)[::-1]
    v1 = evecs[:, idx][:, 0]
    U1 = X @ v1
    return (U1 > np.median(U1)).astype(int)


def load_threshold_labels(df: pd.DataFrame, X: np.ndarray) -> tuple[np.ndarray, str]:
    if THRESH_LAB.exists():
        lab = np.load(THRESH_LAB)
        if len(lab) == len(df):
            print(f"Threshold labels from {THRESH_LAB.name}")
            return lab.astype(int), f"npy:{THRESH_LAB.name}"
    if THRESH_CSV.exists():
        tdf = pd.read_csv(THRESH_CSV, encoding="utf-8-sig")
        cand = [
            c
            for c in tdf.columns
            if c.lower() in ("cluster", "new_cluster", "threshold", "label", "labels")
            or "cluster" in c.lower()
        ]
        for preferred in ("cluster", "new_cluster", "threshold"):
            if preferred in tdf.columns:
                lab = tdf[preferred].to_numpy()
                if len(lab) == len(df):
                    print(f"Threshold labels from {THRESH_CSV.name} col='{preferred}'")
                    return lab.astype(int), f"csv:{preferred}"
        print(f"Threshold CSV cluster-like cols: {cand}")
    print("Recomputing median threshold on U1 (same as 11_final_clusters.py)")
    return recompute_threshold(X), "recomputed_U1_median"


def compare_block(name: str, our: np.ndarray, km: np.ndarray, X: np.ndarray) -> dict:
    m_our = metrics(X, our)
    m_km = metrics(X, km)
    ari = float(adjusted_rand_score(our, km))
    block = {
        "method_name": name,
        "our": m_our,
        "kmeans": m_km,
        "ARI_our_vs_kmeans": ari,
    }
    print(f"\n===== {name} vs KMeans K=2 =====")
    print(f"  our:    SW={m_our['SW']:+.6f}  CH={m_our['CH']:.3f}  DB={m_our['DB']:.6f}  "
          f"K={m_our['n_clusters']} sizes={m_our['sizes']}")
    print(f"  KMeans: SW={m_km['SW']:+.6f}  CH={m_km['CH']:.3f}  DB={m_km['DB']:.6f}  "
          f"K={m_km['n_clusters']} sizes={m_km['sizes']}")
    print(f"  ARI(our, KMeans) = {ari:.6f}")
    return block


def main() -> None:
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig")
    feat = exclude_income(feature_cols(df))
    X_raw = df[feat].values.astype(float)
    N, p = X_raw.shape
    print(f"N={N}, p={p}, feature_list_len={len(feat)}")
    print(f"features: {feat}")
    assert p == 17, f"Expected p=17 no-income, got p={p}"
    assert len(feat) == p

    X = StandardScaler().fit_transform(X_raw)
    km = KMeans(n_clusters=2, n_init=30, random_state=42).fit_predict(X)

    # --- threshold (основной) ---
    thr, thr_src = load_threshold_labels(df, X)
    assert len(thr) == N, f"threshold len {len(thr)} != N={N}"

    # --- Louvain labels_louvain.npy ---
    lv_path = LOUVAIN_LAB if LOUVAIN_LAB.exists() else _LEGACY_LOUVAIN
    assert lv_path.exists(), f"missing {LOUVAIN_LAB}"
    louvain = np.load(lv_path)
    print(f"\n{lv_path.name}: len={len(louvain)}, unique={np.unique(louvain).tolist()}, "
          f"sizes={np.bincount(louvain.astype(int)).tolist()}")
    assert len(louvain) == N, (
        f"{lv_path.name} len={len(louvain)} != N={N} — not aligned with no-income features"
    )
    louvain_is_balanced_median = (
        len(np.unique(louvain)) == 2
        and sorted(np.bincount(louvain.astype(int)).tolist()) == [N // 2, N - N // 2]
        and float(adjusted_rand_score(thr, louvain)) > 0.999
    )
    louvain_note = (
        "WARNING: Louvain looks identical to threshold bipartition"
        if louvain_is_balanced_median
        else "OK: Louvain differs from threshold — treat as Louvain from 06"
    )
    print(f"Louvain check vs threshold: ARI={adjusted_rand_score(thr, louvain):.6f}")
    print(louvain_note)

    b_thr = compare_block("threshold_U1_median (ОСНОВНОЙ)", thr, km, X)
    b_lv = compare_block("Louvain labels_louvain.npy (НЕ основной)", louvain, km, X)

    ari_thr_lv = float(adjusted_rand_score(thr, louvain))
    print(f"\nARI(threshold, Louvain) = {ari_thr_lv:.6f}")

    out = {
        "note": (
            "Основной метод = median threshold on U1 (11_final_clusters.py) → labels_threshold.npy. "
            "Louvain = labels_louvain.npy from 06 — NOT основной. "
            "Старое labels_final.npy было Louvain."
        ),
        "features": {
            "N": int(N),
            "p": int(p),
            "feature_list_length": int(len(feat)),
            "feature_list": feat,
            "source": str(FEAT_PATH),
        },
        "labels_louvain_npy": {
            "path": str(lv_path),
            "length": int(len(louvain)),
            "matches_N": bool(len(louvain) == N),
            "n_clusters": int(len(np.unique(louvain))),
            "sizes": np.bincount(louvain.astype(int)).tolist(),
            "interpreted_as": "Louvain (06)",
            "verification": louvain_note,
        },
        "threshold_labels": {
            "source": thr_src,
            "n_clusters": int(len(np.unique(thr))),
            "sizes": np.bincount(thr.astype(int)).tolist(),
            "interpreted_as": "median_threshold_U1 (основной)",
        },
        "kmeans": {
            "K": 2,
            "n_init": 30,
            "random_state": 42,
            "sizes": np.bincount(km.astype(int)).tolist(),
        },
        "comparison_threshold_vs_kmeans": b_thr,
        "comparison_louvain_vs_kmeans": b_lv,
        "ARI_threshold_vs_louvain": ari_thr_lv,
        "paste": {
            "N": int(N),
            "p": int(p),
            "feature_list_length": int(len(feat)),
            "threshold": {
                "SW": b_thr["our"]["SW"],
                "CH": b_thr["our"]["CH"],
                "DB": b_thr["our"]["DB"],
                "KMeans_SW": b_thr["kmeans"]["SW"],
                "KMeans_CH": b_thr["kmeans"]["CH"],
                "KMeans_DB": b_thr["kmeans"]["DB"],
                "ARI": b_thr["ARI_our_vs_kmeans"],
            },
            "louvain": {
                "SW": b_lv["our"]["SW"],
                "CH": b_lv["our"]["CH"],
                "DB": b_lv["our"]["DB"],
                "KMeans_SW": b_lv["kmeans"]["SW"],
                "KMeans_CH": b_lv["kmeans"]["CH"],
                "KMeans_DB": b_lv["kmeans"]["DB"],
                "ARI": b_lv["ARI_our_vs_kmeans"],
            },
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
