"""
Вторая задача: обнаружение структурных сдвигов.
9 методов на многомерном сигнале [λ1..λ5, PR₊].
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import json
from collections import Counter
import ruptures as rpt
import matplotlib.pyplot as plt
from utils import RESULTS, FIGURES
import warnings
warnings.filterwarnings('ignore')

def main():
    with open(RESULTS / 'dynamic_summary.json', encoding='utf-8') as f:
        dyn = json.load(f)
    spectral = pd.DataFrame(dyn['windows'])
    spectral['date'] = pd.to_datetime(spectral['end'])  # ← дата = КОНЕЦ окна
    for k in range(5):
        spectral[f'lambda_{k+1}'] = spectral['top_evals'].apply(
            lambda x: x[k] if k < len(x) else np.nan
        )

    # Сигнал: [λ1..λ5, PR_plus]. Frustration НЕ включена
    MULTI_COLS = ['lambda_1', 'lambda_2', 'lambda_3', 'lambda_4',
                  'lambda_5', 'PR_plus']
    X = spectral[MULTI_COLS].values
    print(f"Signal: {X.shape}, окон: {len(spectral)}")

    methods = {}

    for model in ['rbf', 'l2', 'normal']:
        try:
            cps = rpt.Pelt(model=model).fit(X).predict(pen=2)
            methods[f'PELT_{model}_pen2'] = [c for c in cps if c < len(spectral)]
        except Exception: pass

    for model in ['rbf', 'l2']:
        try:
            cps = rpt.Binseg(model=model).fit(X).predict(n_bkps=3)
            methods[f'Binseg_{model}_3'] = [c for c in cps if c < len(spectral)]
        except Exception: pass

    try:
        cps = rpt.BottomUp(model='rbf').fit(X).predict(n_bkps=3)
        methods['BottomUp_rbf_3'] = [c for c in cps if c < len(spectral)]
    except Exception: pass

    try:
        cps = rpt.Window(width=5, model='rbf').fit(X).predict(n_bkps=3)
        methods['Window_rbf_3'] = [c for c in cps if c < len(spectral)]
    except Exception: pass

    try:
        cps = rpt.Dynp(model='rbf').fit(X).predict(n_bkps=3)
        methods['Dynp_rbf_3'] = [c for c in cps if c < len(spectral)]
    except Exception: pass

    # CUSUM на ARI(t)
    try:
        aris_data = dyn.get('aris', [])
        if aris_data:
            aris_df = pd.DataFrame(aris_data)
            ari_series = np.full(len(spectral), np.nan)
            for _, row in aris_df.iterrows():
                idx = int(row['from']) - 1
                if 0 <= idx < len(ari_series):
                    ari_series[idx] = row['ari']
            valid = ~np.isnan(ari_series)
            if valid.sum() > 3:
                cusum = np.cumsum(ari_series[valid] - np.nanmean(ari_series))
                signs = np.sign(cusum)
                changes = np.where(np.diff(signs) != 0)[0] + 1
                methods['CUSUM_ARI'] = [int(i) for i in changes if i < len(spectral)]
    except Exception: pass

    # Таблица методов
    print(f"\n{'='*80}")
    print("МЕТОДЫ ОБНАРУЖЕНИЯ СДВИГОВ")
    print(f"{'='*80}")
    for name, cps in methods.items():
        dates = [str(spectral.iloc[c]['date'].date()) for c in cps]
        print(f"{name:<22} {len(cps):>3}  {dates}")

    # Консенсус
    all_dates = []
    for cps in methods.values():
        for c in cps:
            if c < len(spectral):
                all_dates.append(str(spectral.iloc[c]['date'].date()))
    counter = Counter(all_dates)
    n_methods = len(methods)

    print(f"\n{'='*80}")
    print("КОНСЕНСУС")
    print(f"{'='*80}")
    consensus = []
    for date, count in counter.most_common():
        pct = count / n_methods * 100
        marker = '⭐' if count >= n_methods * 0.5 else ''
        print(f"  {date}: {count}/{n_methods} ({pct:.0f}%) {marker}")
        consensus.append({
            'date': date, 'count': count, 'pct': pct,
            'consensus': count >= n_methods * 0.5
        })

    # Сохранение
    rows = [{'method': n, 'window_idx': int(c),
             'date': str(spectral.iloc[c]['date'].date())}
            for n, cps in methods.items() for c in cps if c < len(spectral)]
    pd.DataFrame(rows).to_csv(RESULTS / 'changepoints_all_methods.csv',
                               index=False, encoding='utf-8-sig')
    pd.DataFrame(consensus).to_csv(RESULTS / 'changepoints_consensus.csv',
                                    index=False, encoding='utf-8-sig')
    print(f"\n✓ Сохранено: changepoints_all_methods.csv, changepoints_consensus.csv")

    # Визуализация
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax = axes[0]
    ax.plot(spectral['date'], spectral['PR_plus'], 'o-', color='tab:blue')
    for date, count in counter.most_common():
        if count >= n_methods * 0.5:
            ax.axvline(pd.to_datetime(date), color='red', ls='--', alpha=0.7)
    ax.set_title('Точки сдвигов на PR₊(t). Красные — консенсус')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    dates_sorted = [d for d, _ in counter.most_common()]
    counts_sorted = [c for _, c in counter.most_common()]
    colors = ['tab:red' if c >= n_methods * 0.5 else 'tab:gray'
              for c in counts_sorted]
    ax.bar(range(len(dates_sorted)), counts_sorted, color=colors, edgecolor='k')
    ax.axhline(n_methods * 0.5, color='red', ls='--', label='50% порог')
    ax.set_xticks(range(len(dates_sorted)))
    ax.set_xticklabels(dates_sorted, rotation=45, ha='right')
    ax.set_ylabel('Число методов')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_changepoints_consensus.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Сохранено: fig_changepoints_consensus.png")

if __name__ == '__main__':
    main()
