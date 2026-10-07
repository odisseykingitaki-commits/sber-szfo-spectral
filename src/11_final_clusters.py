"""Финальная кластеризация: median threshold на U1 = X @ v1 (C_reg=0.2).

Основной метод проекта. Пишет:
  - data/processed/clusters_final_threshold.csv
  - results/labels_threshold.npy
  - results/final_method_summary.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.linalg import eigh

from utils import DATA_PROC, RESULTS, build_J, spectral_metrics, feature_cols

FEAT_PATH = DATA_PROC / 'features_szfo_v2_final.csv'
OUT_CSV = DATA_PROC / 'clusters_final_threshold.csv'
OUT_LAB = RESULTS / 'labels_threshold.npy'
OUT_SUM = RESULTS / 'final_method_summary.json'
LOUVAIN_LAB = RESULTS / 'labels_louvain.npy'
_LEGACY_LOUVAIN = RESULTS / 'labels_final.npy'  # deprecated alias

C_REG = 0.2


def load_louvain():
    if LOUVAIN_LAB.exists():
        return np.load(LOUVAIN_LAB), LOUVAIN_LAB.name
    if _LEGACY_LOUVAIN.exists():
        print(f"WARN: legacy {_LEGACY_LOUVAIN.name} — предпочтителен {LOUVAIN_LAB.name}")
        return np.load(_LEGACY_LOUVAIN), _LEGACY_LOUVAIN.name
    return None, None


def main():
    df = pd.read_csv(FEAT_PATH, encoding='utf-8-sig')
    feat = feature_cols(df)
    X_raw = df[feat].values
    N, p = X_raw.shape
    print(f"N={N}, p={p}")

    X = StandardScaler().fit_transform(X_raw)
    X_bin = (X > 0).astype(int)

    J = build_J(X_bin, C_reg=C_REG)
    evals, evecs = eigh(J)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    v1 = evecs[:, 0]
    U1 = X @ v1
    labels = (U1 > np.median(U1)).astype(int)

    sizes = np.bincount(labels).tolist()
    sw = float(silhouette_score(X, labels))
    PR, frustr, n_pos, n_neg = spectral_metrics(evals, p)

    print(f"\n=== MAIN METHOD: median threshold on U1 (C_reg={C_REG}) ===")
    print(f"λ_max={evals[0]:+.3f}  PR₊={PR:.2f}  Frustration={frustr:.2f}")
    print(f"SW={sw:+.3f}  sizes={sizes}")

    # KMeans on raw standardized features
    km = KMeans(n_clusters=2, n_init=30, random_state=42).fit_predict(X)
    ari_km = float(adjusted_rand_score(labels, km))
    print(f"ARI vs KMeans(n=2) on X: {ari_km:.3f}")

    ari_lv = None
    louvain, lv_src = load_louvain()
    if louvain is not None:
        ari_lv = float(adjusted_rand_score(labels, louvain))
        print(f"ARI vs Louvain ({lv_src}): {ari_lv:.3f}")
    else:
        print("labels_louvain.npy не найден — ARI vs Louvain пропущен")

    df_out = df.copy()
    df_out['cluster'] = labels
    df_out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    np.save(OUT_LAB, labels)

    summary = {
        'method': 'median_threshold_U1_PLM_v1',
        'C_reg': C_REG,
        'N': int(N),
        'p': int(p),
        'SW': sw,
        'sizes': sizes,
        'labels_file': str(OUT_LAB.name),
        'ARI_vs_KMeans': ari_km,
        'ARI_vs_Louvain': ari_lv,
        'lambda_max': float(evals[0]),
        'PR_plus': float(PR),
        'Frustration': float(frustr),
        'n_pos': int(n_pos),
        'n_neg': int(n_neg),
    }
    with open(OUT_SUM, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Сохранено: {OUT_CSV}, {OUT_LAB}, {OUT_SUM}")


if __name__ == '__main__':
    main()
