"""Robustness check по C_reg → results/robustness.csv + figures/fig_final.png"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from utils import DATA_PROC, RESULTS, FIGURES, build_J, spectral_metrics, feature_cols

FEAT_PATH = DATA_PROC / 'features_szfo_v2_final.csv'
OUT_CSV = RESULTS / 'robustness.csv'
OUT_FIG = FIGURES / 'fig_final.png'

C_REG_LIST = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

def cluster_with_C(X, X_bin, C_reg):
    """Пороговое разделение по главной моде для данного C_reg."""
    J = build_J(X_bin, C_reg=C_reg)
    evals, evecs = eigh(J)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    v1 = evecs[:, 0]
    U1 = X @ v1
    labels = (U1 > np.median(U1)).astype(int)
    return labels, evals, evecs, J

def main():
    df = pd.read_csv(FEAT_PATH, encoding='utf-8-sig')
    feat = feature_cols(df)
    X = StandardScaler().fit_transform(df[feat].values)
    X_bin = (X > 0).astype(int)
    N, p = X.shape
    print(f"N={N}, p={p}")

    # Базовая конфигурация
    base_labels, _, _, _ = cluster_with_C(X, X_bin, 0.2)

    print(f"\n{'='*80}")
    print("ROBUSTNESS по C_reg")
    print(f"{'='*80}")
    print(f"{'C_reg':>8} {'ARI':>8} {'λ_max':>10} {'λ_2':>10} "
          f"{'PR₊':>8} {'Frustr':>8} {'SW':>8}")
    print("-" * 80)

    rows = []
    for C in C_REG_LIST:
        lab, ev, _, _ = cluster_with_C(X, X_bin, C)
        ari = adjusted_rand_score(base_labels, lab)
        PR, frustr, _, _ = spectral_metrics(ev, p)
        sw = silhouette_score(X, lab)
        rows.append({
            'C_reg': C, 'ARI': ari, 'lambda_max': ev[0], 'lambda_2': ev[1],
            'PR_plus': PR, 'Frustration': frustr, 'SW': sw,
        })
        print(f"{C:>8.2f} {ari:>8.3f} {ev[0]:>+10.3f} {ev[1]:>+10.3f} "
              f"{PR:>8.2f} {frustr:>8.2f} {sw:>+8.3f}")

    robust_df = pd.DataFrame(rows)
    robust_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    # Финальная конфигурация C=0.2
    print(f"\n{'='*80}")
    print("ФИНАЛЬНАЯ СВОДКА (C=0.2)")
    print(f"{'='*80}")
    _, final_evals, _, _ = cluster_with_C(X, X_bin, 0.2)
    pos = final_evals[final_evals > 0]
    neg = final_evals[final_evals < 0]
    PR = (pos.sum())**2 / (pos**2).sum()
    print(f"λ_max           = {final_evals[0]:+.3f}")
    print(f"PR₊             = {PR:.2f}")
    print(f"Frustration     = {len(neg)}/{p} = {len(neg)/p:.2f}")
    print(f"SW (K=2)        = {silhouette_score(X, base_labels):+.3f}")
    print(f"Robustness min  = {robust_df['ARI'].min():.3f}")

    # Визуализация
    bootstrap = pd.read_csv(RESULTS / 'bootstrap_stability.csv')
    bootstrap_aris = bootstrap['ari'].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Спектр
    colors = ['tab:blue' if e > 0 else 'tab:red' for e in final_evals]
    axes[0].bar(range(1, len(final_evals)+1), final_evals,
                color=colors, edgecolor='k', lw=0.5)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].set_xlabel('Номер моды', fontsize=12)
    axes[0].set_ylabel('λ', fontsize=12)
    axes[0].set_title('Спектр J', fontsize=13)
    axes[0].text(0.97, 0.95,
                 f'λ_max={final_evals[0]:.2f}\nPR₊={PR:.2f}\n'
                 f'Frustration={len(neg)/p:.2f}',
                 transform=axes[0].transAxes, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # 2. Robustness
    axes[1].plot(robust_df['C_reg'], robust_df['ARI'], 'o-',
                 color='tab:green', lw=2)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('C_reg (log)', fontsize=12)
    axes[1].set_ylabel('ARI vs C=0.2', fontsize=12)
    axes[1].set_title('Устойчивость к гиперпараметру', fontsize=13)
    axes[1].axhline(0.95, color='orange', ls='--', lw=0.8)
    axes[1].set_ylim(0.9, 1.02)
    axes[1].grid(True, alpha=0.3)

    # 3. Bootstrap
    axes[2].hist(bootstrap_aris, bins=20, color='tab:blue',
                 alpha=0.7, edgecolor='k')
    axes[2].axvline(bootstrap_aris.mean(), color='red', ls='--', lw=2,
                    label=f'mean={bootstrap_aris.mean():.3f}')
    axes[2].set_xlabel('ARI', fontsize=12)
    axes[2].set_ylabel('Частота', fontsize=12)
    axes[2].set_title(f'Bootstrap\n{bootstrap_aris.mean():.3f} ± '
                      f'{bootstrap_aris.std():.3f}', fontsize=13)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Сохранено: {OUT_CSV}, {OUT_FIG}")

if __name__ == '__main__':
    main()