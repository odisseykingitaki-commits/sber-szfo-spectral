"""Сборка признаков → data/processed/features_szfo_v2_final.csv

Финальная матрица для PLM: **17 признаков, без доходов Росстат**:
  5 share + log_total + 5 growth + 5 cv + mob_logratio = 17.

Отрицательный результат (эксперимент, отклонён):
  Добавляли log_income_2024 / income_growth / income_cv (+ Variant C OLS-proxy
  и q75-импутацию для ~103 МО СПб без районной разбивки в urov).
  corr(log_total, log_income)≈0.75 на matched; SW без доходов ≈ SW с proxy
  (не улучшило кластеризацию). Фиксируем no-income как основной метод.
  Ablation: results/variant_c_comparison.json (q75 / Variant C / residual / no_income).
"""
import numpy as np
import pandas as pd
from utils import DATA_INT, DATA_PROC, norm_name, feature_cols

SPEND_PATH = DATA_INT / 'spend.parquet'
MOB_PATH = DATA_INT / 'mobility.parquet'
OUT_PATH = DATA_PROC / 'features_szfo_v2_final.csv'

CATS = ['Здоровье', 'Маркетплейсы', 'Общественное питание',
        'Продовольствие', 'Транспорт']
SLUG = {'Здоровье': 'health', 'Маркетплейсы': 'marketplace',
        'Общественное питание': 'food', 'Продовольствие': 'grocery',
        'Транспорт': 'transport'}


def main():
    print("=== 1. Загрузка ===")
    spend = pd.read_parquet(SPEND_PATH)
    mob = pd.read_parquet(MOB_PATH)

    spend['mo_norm'] = spend['mo'].apply(norm_name)
    mob['mo_norm'] = mob['ref_area'].apply(norm_name)

    # Только СЗФО (МО из мобильности)
    szfo_mos = set(mob['mo_norm'])
    spend_sz = spend[spend['mo_norm'].isin(szfo_mos)].copy()
    print(f"МО СЗФО в расходах: {spend_sz['mo_norm'].nunique()}")

    print("\n=== 2. Пивот по категориям ===")
    spend_sz['period'] = pd.to_datetime(spend_sz['period'])
    wide = spend_sz.pivot_table(
        index=['mo_norm', 'period'],
        columns='category_15',
        values='value',
        aggfunc='mean'
    ).reset_index()

    # Оставляем МО с >= 18 месяцев
    months_per_mo = wide.groupby('mo_norm').size()
    good_mos = months_per_mo[months_per_mo >= 18].index
    wide = wide[wide['mo_norm'].isin(good_mos)].copy()
    print(f"МО с >=18 месяцев: {len(good_mos)}")

    wide['year'] = wide['period'].dt.year

    print("\n=== 3. Считаем признаки ===")
    features = []
    for mo, g in wide.groupby('mo_norm'):
        g = g.sort_values('period')
        row = {'mo_norm': mo}

        g24 = g[g['year'] == 2024]
        if len(g24) < 6:
            continue
        mean24 = g24[CATS + ['Все категории']].mean()
        total24 = mean24['Все категории']
        if total24 <= 0 or pd.isna(total24):
            continue

        # Доли
        for c in CATS:
            row[f'share_{SLUG[c]}'] = mean24[c] / total24

        # Размер
        row['log_total'] = np.log1p(total24)

        # Рост
        g23 = g[g['year'] == 2023]
        if len(g23) >= 6:
            mean23 = g23[CATS].mean()
            for c in CATS:
                m23 = mean23[c]
                m24 = mean24[c]
                row[f'growth_{SLUG[c]}'] = (m24 - m23) / m23 if m23 > 0 else 0.0
        else:
            for c in CATS:
                row[f'growth_{SLUG[c]}'] = 0.0

        # Волатильность
        for c in CATS:
            vals = g24[c].values
            m = vals.mean()
            row[f'cv_{SLUG[c]}'] = vals.std() / m if m > 0 else 0.0

        features.append(row)

    feat_df = pd.DataFrame(features)
    print(f"Признаков на МО: {feat_df.shape[1] - 1}")
    print(f"Всего МО: {len(feat_df)}")

    print("\n=== 4. Мобильность ===")
    mob_piv = mob.pivot_table(
        index='mo_norm', columns='period', values='value'
    ).reset_index()
    date_cols = [c for c in mob_piv.columns if c != 'mo_norm']

    if len(date_cols) == 2:
        d1, d2 = sorted(date_cols)
        mob_piv['mob_logratio'] = np.log1p(mob_piv[d2]) - np.log1p(mob_piv[d1])
    else:
        mob_piv['mob_logratio'] = 0.0

    feat_df = feat_df.merge(
        mob_piv[['mo_norm', 'mob_logratio']], on='mo_norm', how='left'
    )
    feat_df['mob_logratio'] = feat_df['mob_logratio'].fillna(0.0)

    # Агрегация дубликатов (если есть)
    dup_cols = feat_df[feat_df.duplicated('mo_norm', keep=False)]['mo_norm'].unique()
    if len(dup_cols):
        print(f"Дубликатов: {len(dup_cols)} — агрегирую средним")
        num_cols = feature_cols(feat_df)
        feat_df = feat_df.groupby('mo_norm', as_index=False)[num_cols].mean()

    fcols = feature_cols(feat_df)
    print(f"\nФинал: {feat_df.shape}")
    print(f"p = {len(fcols)} (ожидаем 17, no-income)")
    print(f"Пропусков: {feat_df.isna().sum().sum()}")
    print(f"Признаки: {fcols}")
    assert len(fcols) == 17, f"Expected p=17, got {len(fcols)}: {fcols}"

    feat_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n✓ Сохранено: {OUT_PATH}")


if __name__ == '__main__':
    main()
