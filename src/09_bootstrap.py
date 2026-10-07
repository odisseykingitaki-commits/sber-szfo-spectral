"""Bootstrap stability: 100 итераций → results/bootstrap_stability.csv"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

from utils import DATA_PROC, RESULTS, FIGURES, build_J, feature_cols

FEAT_PATH = DATA_PROC / 'features_szfo_v2_final.csv'
OUT_CSV = RESULTS / 'bootstrap_stability.csv'
OUT_FIG = FIGURES / 'fig_bootstrap.png'

C_REG = 0.2
N_BOOT = 100

def threshold_labels(X, X_bin):
    """Пороговое разделение по главной моде J."""
    from scipy.linalg import eigh
    J = build_J(X_bin, C_reg=C_REG)
    evals, evecs = eigh(J)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    v1 = evecs[:, 0]
    U1 = X @ v1
    return (U1 > np.median(U1)).astype(int)

def main():
    df = pd.read_csv(FEAT_PATH, encoding='utf-8-sig')
    feat = feature_cols(df)
    X = StandardScaler().fit_transform(df[feat].values)
    X_bin = (X > 0).astype(int)
    N, p = X.shape
    print(f"N={N}, p={p}")

    # Полный набор
    labels_full = threshold_labels(X, X_bin)
    print(f"Полный набор: размеры={np.bincount(labels_full).tolist()}")

    # Bootstrap
    print(f"\n=== BOOTSTRAP ({N_BOOT} итераций) ===")
    np.random.seed(42)
    aris = []
    for i in range(N_BOOT):
        idx_b = resample(np.arange(N), n_samples=N, replace=True, random_state=i)
        Xb = X[idx_b]
        Xb_bin = X_bin[idx_b]
        lab_b = threshold_labels(Xb, Xb_bin)
        aris.append(adjusted_rand_score(labels_full[idx_b], lab_b))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N_BOOT}, mean ARI = {np.mean(aris):.3f}")

    aris = np.array(aris)
    print(f"\nBootstrap ARI: {aris.mean():.3f} ± {aris.std():.3f}")
    print(f"min={aris.min():.3f}, max={aris.max():.3f}")

    # Сохранение
    pd.DataFrame({'ari': aris}).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    # Визуализация
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(aris, bins=20, color='tab:blue', alpha=0.7, edgecolor='k')
    ax.axvline(aris.mean(), color='red', ls='--', lw=2,
               label=f'mean={aris.mean():.3f}')
    ax.set_xlabel('ARI vs полный набор')
    ax.set_ylabel('Частота')
    ax.set_title(f'Bootstrap stability (n={N_BOOT})\n'
                 f'{aris.mean():.3f} ± {aris.std():.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Сохранено: {OUT_CSV}, {OUT_FIG}")

if __name__ == '__main__':
    main()