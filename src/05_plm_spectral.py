"""PLM + спектр + кластеризация → data/processed/final_clusters_v2.csv"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)
from scipy.linalg import eigh

from utils import DATA_PROC, RESULTS, build_J, spectral_metrics, feature_cols

FEAT_PATH = DATA_PROC / 'features_szfo_v2_final.csv'
OUT_CSV = DATA_PROC / 'final_clusters_v2.csv'
OUT_J = RESULTS / 'J_v2.npy'
OUT_EVALS = RESULTS / 'evals_v2.npy'
OUT_EVECS = RESULTS / 'evecs_v2.npy'
OUT_SUM = RESULTS / 'spectral_summary_v2.json'

C_REG = 0.2

def main():
    df = pd.read_csv(FEAT_PATH, encoding='utf-8-sig')
    mo_names = df['mo_norm'].values
    feat = feature_cols(df)
    X_raw = df[feat].values
    N, p = X_raw.shape
    print(f"N={N}, p={p}")
    print(f"Признаки: {feat}")

    X = StandardScaler().fit_transform(X_raw)
    X_bin = (X > 0).astype(int)

    print(f"\n=== PLM (C={C_REG}, p={p}) ===")
    J = build_J(X_bin, C_reg=C_REG)
    print(f"J: min={J.min():.3f}, max={J.max():.3f}, std={J.std():.3f}")
    print(f"|J|>0.2: {(np.abs(J) > 0.2).mean():.1%}")

    print(f"\n=== СПЕКТР ===")
    evals, evecs = eigh(J)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    for k in range(min(10, p)):
        print(f"  λ_{k+1:<2} = {evals[k]:+.4f}")

    PR_plus, frustration, n_pos, n_neg = spectral_metrics(evals, p)
    print(f"\nPR₊ = {PR_plus:.2f}")
    print(f"Frustration = {frustration:.2f} ({n_neg}/{p})")
    print(f"λ_max / Σλ₊ = {evals[0]/evals[evals>0].sum():.1%}")

    print(f"\n=== ТОП-5 МОД ===")
    for k in range(min(5, p)):
        v = evecs[:, k]
        top = np.argsort(np.abs(v))[::-1][:5]
        print(f"\nМода {k+1} (λ={evals[k]:+.3f}):")
        for i in top:
            print(f"   {feat[i]:<22} {v[i]:+.3f}")

    print(f"\n=== КЛАСТЕРИЗАЦИЯ (моды 1-4) ===")
    V = evecs[:, :4]
    U = X @ V
    U_n = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

    results = {}
    for K in range(2, 9):
        lab = KMeans(n_clusters=K, n_init=30, random_state=42).fit_predict(U_n)
        sizes = np.bincount(lab)
        if sizes.min() < 2:
            continue
        sw = silhouette_score(X, lab)
        ch = calinski_harabasz_score(X, lab)
        db = davies_bouldin_score(X, lab)
        results[K] = {'SW': sw, 'CH': ch, 'DB': db, 'labels': lab,
                      'sizes': sizes.tolist()}
        print(f"  K={K}: SW={sw:+.3f} CH={ch:6.1f} DB={db:.3f} "
              f"sizes={sizes.tolist()}")

    best_K = max(results, key=lambda k: results[k]['SW'])
    best = results[best_K]
    print(f"\n✅ Лучший K={best_K}, SW={best['SW']:+.3f}")

    np.save(OUT_J, J)
    np.save(OUT_EVALS, evals)
    np.save(OUT_EVECS, evecs)

    df_out = df.copy()
    df_out['new_cluster'] = best['labels']
    df_out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    summary = {
        'N': int(N), 'p': int(p), 'C_reg': C_REG,
        'PR_plus': float(PR_plus), 'frustration': float(frustration),
        'n_pos': n_pos, 'n_neg': n_neg,
        'best_K': int(best_K), 'best_SW': float(best['SW']),
        'top_evals': evals[:10].tolist(),
    }
    with open(OUT_SUM, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Сохранено: {OUT_CSV}, {OUT_J}, {OUT_EVALS}, {OUT_EVECS}, {OUT_SUM}")

if __name__ == '__main__':
    main()
