"""Динамическая сеть J(t) + траектории → data/processed/trajectories.csv"""
import re
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from scipy.linalg import eigh
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt

from utils import (DATA_INT, DATA_PROC, RESULTS, FIGURES,
                   norm_name, build_J, spectral_metrics)

SPEND_PATH = DATA_INT / 'spend.parquet'
FEAT_PATH = DATA_PROC / 'features_szfo_v2_final.csv'
OUT_TRAJ = DATA_PROC / 'trajectories.csv'
OUT_SUM = RESULTS / 'dynamic_summary.json'
OUT_FIG = FIGURES / 'fig_dynamic.png'

CATS = ['Здоровье', 'Маркетплейсы', 'Общественное питание',
        'Продовольствие', 'Транспорт']
SLUG = {'Здоровье': 'health', 'Маркетплейсы': 'marketplace',
        'Общественное питание': 'food', 'Продовольствие': 'grocery',
        'Транспорт': 'transport'}

WINDOW_SIZE = 6

def build_features_window(df_w, mo_list, share_cols):
    rows = []
    for mo in mo_list:
        g = df_w[df_w['mo_norm'] == mo].sort_values('period')
        if len(g) < 3:
            rows.append(None)
            continue
        row = {}
        mean_w = g[share_cols].mean()
        for c in share_cols:
            row[c] = mean_w[c]
        row['log_total'] = np.log1p(g['Все категории'].mean())
        first3 = g.head(3)[share_cols].mean()
        last3 = g.tail(3)[share_cols].mean()
        for c in share_cols:
            row[f'growth_{c}'] = (last3[c] - first3[c]) / (first3[c] + 1e-9)
        for c in share_cols:
            vals = g[c].values
            m = vals.mean()
            row[f'cv_{c}'] = vals.std() / m if m > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    spend = pd.read_parquet(SPEND_PATH)
    spend['mo_norm'] = spend['mo'].apply(norm_name)
    spend['period'] = pd.to_datetime(spend['period'])

    final = pd.read_csv(FEAT_PATH, encoding='utf-8-sig')
    szfo_mos = set(final['mo_norm'])
    spend_sz = spend[spend['mo_norm'].isin(szfo_mos)].copy()
    print(f"Строк: {len(spend_sz)}, МО: {spend_sz['mo_norm'].nunique()}")

    wide = spend_sz.pivot_table(
        index=['mo_norm', 'period'],
        columns='category_15',
        values='value',
        aggfunc='mean'
    ).reset_index().dropna(subset=['Все категории'])

    for c in CATS:
        wide[f'share_{SLUG[c]}'] = wide[c] / wide['Все категории'].replace(0, np.nan)
    share_cols = [f'share_{SLUG[c]}' for c in CATS]
    wide = wide.dropna(subset=share_cols)
    print(f"Wide: {wide.shape}")

    months = sorted(wide['period'].unique())
    windows = []
    for s in range(0, len(months) - WINDOW_SIZE + 1, 1):
        w = months[s:s+WINDOW_SIZE]
        windows.append((w[0], w[-1]))
    print(f"Окон: {len(windows)}")

    results = []
    for wi, (start, end) in enumerate(windows):
        df_w = wide[(wide['period'] >= start) & (wide['period'] <= end)]
        counts = df_w.groupby('mo_norm').size()
        good_mos = sorted(counts[counts >= 4].index)
        df_w = df_w[df_w['mo_norm'].isin(good_mos)]
        if len(good_mos) < 50:
            continue

        X_df = build_features_window(df_w, good_mos, share_cols)
        mask = X_df.notna().all(axis=1).values
        good_mos = [m for m, v in zip(good_mos, mask) if v]
        X_df = X_df[mask].reset_index(drop=True)

        X = StandardScaler().fit_transform(X_df.values)
        X_bin = (X > 0).astype(int)
        p = X.shape[1]

        J_t = build_J(X_bin, C_reg=0.2)
        evals, evecs = eigh(J_t)
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
        PR, frustr, _, _ = spectral_metrics(evals, p)

        n_modes = min(5, p)
        V_k = evecs[:, :n_modes]
        U = X @ V_k
        D = pairwise_distances(U)
        eps = np.quantile(D[D > 0], 0.30)
        A_t = (D < eps).astype(int)
        np.fill_diagonal(A_t, 0)

        G_t = nx.from_numpy_array(A_t)
        try:
            coms = louvain_communities(G_t, seed=42, resolution=0.5)
            coms = sorted(coms, key=len, reverse=True)
            labels = np.full(len(good_mos), -1)
            for i, c in enumerate(coms):
                if len(c) >= 10:
                    for node in c:
                        labels[node] = i
            if (labels == -1).any():
                n_big = labels.max() + 1
                labels[labels == -1] = n_big
            n_com = int(labels.max()) + 1
        except Exception:
            labels = np.zeros(len(good_mos), dtype=int)
            n_com = 1

        results.append({
            'window': wi, 'start': str(start.date()), 'end': str(end.date()),
            'n_mo': len(good_mos), 'PR_plus': float(PR),
            'frustration': float(frustr), 'n_communities': n_com,
            'evals': evals.tolist(), 'labels': labels.tolist(),
            'mos': good_mos,
        })
        print(f"Окно {wi+1:2d} ({start.date()}..{end.date()}): "
              f"n={len(good_mos)}, PR={PR:.2f}, Frustr={frustr:.2f}, "
              f"сообществ={n_com}")

    # Траектории
    all_mos = set(results[0]['mos'])
    for r in results[1:]:
        all_mos &= set(r['mos'])
    all_mos = sorted(all_mos)
    print(f"\nМО во всех окнах: {len(all_mos)}")

    traj = np.full((len(all_mos), len(results)), -1, dtype=int)
    for wi, r in enumerate(results):
        for mi, mo in enumerate(all_mos):
            if mo in r['mos']:
                traj[mi, wi] = r['labels'][r['mos'].index(mo)]

    switches = (traj[:, 1:] != traj[:, :-1]).sum(axis=1)
    print(f"Перебежчиков: {(switches>0).sum()} / {len(all_mos)}")
    print(f"Среднее переключений: {switches.mean():.2f}")

    # ARI(t, t+1)
    aris = []
    for i in range(len(results) - 1):
        r1, r2 = results[i], results[i+1]
        common = sorted(set(r1['mos']) & set(r2['mos']))
        if len(common) < 10:
            continue
        idx1 = [r1['mos'].index(m) for m in common]
        idx2 = [r2['mos'].index(m) for m in common]
        ari = adjusted_rand_score(np.array(r1['labels'])[idx1],
                                  np.array(r2['labels'])[idx2])
        aris.append({'from': i+1, 'to': i+2, 'ari': float(ari), 'n': len(common)})

    aris_df = pd.DataFrame(aris)
    print(f"\nСредний ARI: {aris_df['ari'].mean():.3f}")
    print(f"Min ARI: {aris_df['ari'].min():.3f}")

    # Сохранение
    traj_df = pd.DataFrame(traj, index=all_mos,
                            columns=[f"w{i+1}" for i in range(len(results))])
    traj_df['n_switches'] = switches
    traj_df.to_csv(OUT_TRAJ, encoding='utf-8-sig')

    summary = []
    for r in results:
        summary.append({
            'window': r['window'] + 1,
            'start': r['start'], 'end': r['end'],
            'n_mo': r['n_mo'],
            'PR_plus': r['PR_plus'],
            'frustration': r['frustration'],
            'n_communities': r['n_communities'],
            'top_evals': r['evals'][:5],
        })
    with open(OUT_SUM, 'w', encoding='utf-8') as f:
        json.dump({'windows': summary, 'aris': aris}, f,
                  indent=2, ensure_ascii=False)

    # Визуализация
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    x = [r['window'] + 1 for r in results]

    ax = axes[0]
    ax.plot(x, [r['PR_plus'] for r in results], 'o-', color='tab:blue')
    ax.set_ylabel('PR₊', color='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(x, [r['frustration'] for r in results], 's-', color='tab:red')
    ax2.set_ylabel('Frustration', color='tab:red')
    ax.set_title('Эмерджентные метрики во времени')

    axes[1].bar(x, [r['n_communities'] for r in results],
                color='tab:green', edgecolor='k')
    axes[1].set_ylabel('Сообществ')
    axes[1].set_title('Число сообществ в A(t)')

    axes[2].bar([a['from'] for a in aris], [a['ari'] for a in aris],
                color='tab:purple', edgecolor='k')
    axes[2].axhline(0.7, color='r', ls='--')
    axes[2].set_ylabel('ARI')
    axes[2].set_xlabel('Окно')
    axes[2].set_title('Стабильность между окнами')

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Сохранено: {OUT_TRAJ}, {OUT_SUM}, {OUT_FIG}")

if __name__ == '__main__':
    main()
