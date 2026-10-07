"""Полный ICVI → results/icvi_full.csv

«Наш» = median threshold on U1 (labels_threshold.npy / 11_final_clusters.py).
Louvain = labels_louvain.npy (06_network_louvain.py) — отдельная строка сравнения.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             pairwise_distances)

from utils import DATA_PROC, RESULTS, feature_cols

FEAT_PATH = DATA_PROC / 'features_szfo_v2_final.csv'
THRESH_CSV = DATA_PROC / 'clusters_final_threshold.csv'
THRESH_LAB = RESULTS / 'labels_threshold.npy'  # основной метод (11)
LOUVAIN_LAB = RESULTS / 'labels_louvain.npy'   # сетевой анализ (06)
# backward compat: старое имя было Louvain
_LEGACY_LOUVAIN = RESULTS / 'labels_final.npy'
OUT = RESULTS / 'icvi_full.csv'
C_REG = 0.2

def s_dbw(X, labels):
    K = len(np.unique(labels))
    if K < 2:
        return np.inf
    sigma_all = np.sqrt(np.mean(np.sum((X - X.mean(axis=0))**2, axis=1)))
    clusters = [X[labels == k] for k in range(K)]
    stdevs = []
    for c in clusters:
        if len(c) > 1:
            stdev = np.sqrt(np.mean(np.sum((c - c.mean(axis=0))**2, axis=1)))
            stdevs.append(stdev)
    avg_stdev = np.mean(stdevs) if stdevs else 0
    scatter = avg_stdev / (sigma_all + 1e-12)

    def density(c1, c2, sigma):
        mid = (c1.mean(axis=0) + c2.mean(axis=0)) / 2
        n_mid = np.sum(np.all(np.abs(X - mid) < sigma, axis=1))
        return n_mid / (max(len(c1), len(c2)) + 1e-12)

    db = 0
    count = 0
    for i in range(K):
        for j in range(i+1, K):
            db += density(clusters[i], clusters[j], avg_stdev)
            count += 1
    db = db / count if count else 0
    return scatter + db

def avi(X, labels):
    K = len(np.unique(labels))
    D = pairwise_distances(X)
    clusters = [np.where(labels == k)[0] for k in range(K)]
    intra = []
    for idx in clusters:
        if len(idx) > 1:
            sub = D[np.ix_(idx, idx)]
            intra.append(sub[np.triu_indices(len(idx), 1)].mean())
    intra = np.mean(intra) if intra else 0
    inter = []
    for i in range(K):
        for j in range(i+1, K):
            inter.append(D[np.ix_(clusters[i], clusters[j])].mean())
    inter = np.mean(inter) if inter else 1
    return intra / (inter + 1e-12)

def avu(X, labels):
    K = len(np.unique(labels))
    var = []
    for k in range(K):
        c = X[labels == k]
        if len(c) > 0:
            var.append(c.var(axis=0).mean())
    return np.mean(var) if var else np.inf

def mq(X, labels):
    K = len(np.unique(labels))
    errors = []
    for k in range(K):
        c = X[labels == k]
        if len(c) > 0:
            cent = c.mean(axis=0)
            errors.append(np.mean(np.linalg.norm(c - cent, axis=1)))
    return np.mean(errors) if errors else np.inf


def metrics_row(X, labels):
    return {
        'SW': silhouette_score(X, labels),
        'CH': calinski_harabasz_score(X, labels),
        'S_Dbw': s_dbw(X, labels),
        'AVI': avi(X, labels),
        'AVU': avu(X, labels),
        'MQ': mq(X, labels),
    }


def load_louvain():
    if LOUVAIN_LAB.exists():
        return np.load(LOUVAIN_LAB), LOUVAIN_LAB.name
    if _LEGACY_LOUVAIN.exists():
        print(f"WARN: {LOUVAIN_LAB.name} отсутствует, используем legacy {_LEGACY_LOUVAIN.name}")
        return np.load(_LEGACY_LOUVAIN), _LEGACY_LOUVAIN.name
    return None, None


def load_threshold(X):
    """Primary: labels_threshold.npy; else CSV; else recompute like 11."""
    if THRESH_LAB.exists():
        return np.load(THRESH_LAB), THRESH_LAB.name
    if THRESH_CSV.exists():
        thr_df = pd.read_csv(THRESH_CSV, encoding='utf-8-sig')
        if 'cluster' in thr_df.columns:
            lab = thr_df['cluster'].values.astype(int)
            print(f"WARN: {THRESH_LAB.name} отсутствует — берём cluster из CSV")
            return lab, 'clusters_final_threshold.csv:cluster'
    from scipy.linalg import eigh
    from utils import build_J
    X_bin = (X > 0).astype(int)
    J = build_J(X_bin, C_reg=C_REG)
    evals, evecs = eigh(J)
    v1 = evecs[:, np.argsort(evals)[::-1][0]]
    U1 = X @ v1
    lab = (U1 > np.median(U1)).astype(int)
    print("WARN: threshold labels missing — recomputed median U1 (как 11)")
    return lab, 'recomputed_median_U1'


def main():
    df = pd.read_csv(FEAT_PATH, encoding='utf-8-sig')
    feat = feature_cols(df)
    X = StandardScaler().fit_transform(df[feat].values)

    labels_thr, thr_src = load_threshold(X)
    labels_lv, lv_src = load_louvain()
    print(f"Threshold source: {thr_src}")

    print(f"Threshold labels: {THRESH_LAB.name} sizes={np.bincount(labels_thr.astype(int)).tolist()}")
    if labels_lv is not None:
        print(f"Louvain labels:   {lv_src} sizes={np.bincount(labels_lv.astype(int)).tolist()}")

    km = KMeans(n_clusters=2, n_init=30, random_state=42).fit_predict(X)

    compare = [('Наш (threshold)', labels_thr), ('KMeans', km)]
    if labels_lv is not None:
        compare.append(('Louvain', labels_lv))

    print(f"\n{'Метрика':<20} {'SW':>10} {'CH':>12}")
    print("-" * 45)
    for name, lab in compare:
        m = metrics_row(X, lab)
        print(f"{name:<20} {m['SW']:>10.3f} {m['CH']:>12.1f}")

    print(f"\n{'='*60}")
    print("ICVI по K (KMeans) + наши методы")
    print(f"{'='*60}")
    print(f"{'K':>18} {'SW':>8} {'CH':>8} {'S_Dbw':>8} {'AVI':>8} {'AVU':>8} {'MQ':>8}")
    print("-" * 70)

    all_rows = []
    for K in [2, 3, 4, 5]:
        lab = KMeans(n_clusters=K, n_init=30, random_state=42).fit_predict(X)
        row = {'K': K, **metrics_row(X, lab)}
        all_rows.append(row)
        print(f"{K:>18} {row['SW']:>8.3f} {row['CH']:>8.1f} "
              f"{row['S_Dbw']:>8.3f} {row['AVI']:>8.3f} "
              f"{row['AVU']:>8.3f} {row['MQ']:>8.3f}")

    # Основной метод — threshold
    thr_m = metrics_row(X, labels_thr)
    all_rows.append({'K': 'threshold', **thr_m})
    print(f"{'threshold':>18} {thr_m['SW']:>8.3f} {thr_m['CH']:>8.1f} "
          f"{thr_m['S_Dbw']:>8.3f} {thr_m['AVI']:>8.3f} "
          f"{thr_m['AVU']:>8.3f} {thr_m['MQ']:>8.3f}")

    if labels_lv is not None:
        lv_m = metrics_row(X, labels_lv)
        all_rows.append({'K': 'Louvain', **lv_m})
        print(f"{'Louvain':>18} {lv_m['SW']:>8.3f} {lv_m['CH']:>8.1f} "
              f"{lv_m['S_Dbw']:>8.3f} {lv_m['AVI']:>8.3f} "
              f"{lv_m['AVU']:>8.3f} {lv_m['MQ']:>8.3f}")

    pd.DataFrame(all_rows).to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f"\n✓ Сохранено: {OUT}")

if __name__ == '__main__':
    main()
