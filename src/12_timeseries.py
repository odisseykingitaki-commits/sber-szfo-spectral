"""
Вторая задача: прогнозирование расходов.
Naive, Seasonal, Mean, Prophet, LightGBM, CatBoost.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import json
from utils import DATA_INT, DATA_PROC, RESULTS, norm_name
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

HORIZONS = [1, 3, 6, 12]
CLUSTER_NAMES = {0: 'сельские', 1: 'городские'}

def load_data():
    spend = pd.read_parquet(DATA_INT / 'spend.parquet')
    spend['period'] = pd.to_datetime(spend['period'])
    spend['mo_norm'] = spend['mo'].apply(norm_name)
    all_cat = spend[spend['category_15'] == 'Все категории'].copy()
    all_cat = all_cat.groupby(['mo_norm', 'period'])['value'].mean().reset_index()

    labels_thr = np.load(RESULTS / 'labels_threshold.npy')
    final = pd.read_csv(DATA_PROC / 'features_szfo_v2_final.csv',
                        encoding='utf-8-sig')
    final['cluster'] = labels_thr
    mo_to_cluster = dict(zip(final['mo_norm'], final['cluster']))
    all_cat['cluster'] = all_cat['mo_norm'].map(mo_to_cluster)
    all_cat = all_cat.dropna(subset=['cluster'])
    all_cat['cluster'] = all_cat['cluster'].astype(int)

    agg = all_cat.groupby(['cluster', 'period'])['value'].mean().reset_index()
    agg = agg.sort_values(['cluster', 'period'])

    with open(RESULTS / 'dynamic_summary.json', encoding='utf-8') as f:
        dyn = json.load(f)
    spectral = pd.DataFrame(dyn['windows'])
    for k in range(5):
        spectral[f'lambda_{k+1}'] = spectral['top_evals'].apply(
            lambda x: x[k] if k < len(x) else np.nan
        )
    spectral['end_dt'] = pd.to_datetime(spectral['end'])

    # Google Trends
    try:
        trends = pd.read_csv(DATA_PROC / 'google_trends_szfo.csv',
                             encoding='utf-8-sig')
        trends['period'] = pd.to_datetime(trends['year_month'] + '-01')
        queries_keep = [q for q in trends['query'].unique()
                        if trends[trends['query'] == q]['interest'].sum() > 100]
        trends = trends[trends['query'].isin(queries_keep)]
        trends_agg = trends.groupby(['period', 'query'])['interest'].mean().reset_index()
        trends_pivot = trends_agg.pivot(index='period', columns='query',
                                          values='interest').reset_index()
        trends_pivot.columns.name = None
        trends_cols = [c for c in trends_pivot.columns if c != 'period']
    except Exception:
        trends_pivot = None
        trends_cols = []

    return agg, spectral, trends_pivot, trends_cols

def predict_naive(y, H): return np.full(H, y[-1])
def predict_seasonal(y, H):
    if len(y) < 12: return np.full(H, y[-1])
    return np.array([y[-12 + (i % 12)] for i in range(H)])
def predict_mean(y, H): return np.full(H, y.mean())

def predict_prophet(ds_train, y_train, ds_test, H):
    try:
        from prophet import Prophet
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.05)
        m.fit(pd.DataFrame({'ds': pd.to_datetime(ds_train), 'y': y_train}))
        fc = m.predict(pd.DataFrame({'ds': pd.to_datetime(ds_test)}))
        return fc['yhat'].values
    except Exception:
        return np.full(H, y_train[-1])

def make_features(sub, spectral, trends_pivot, trends_cols, use_trends, use_spectral):
    """Возвращает df с признаками. ВАЖНО: только прошлые данные (no leakage)."""
    df = sub.copy()
    df['month'] = df['ds'].dt.month
    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['ma_3'] = df['y'].shift(1).rolling(3).mean()
    df['ma_6'] = df['y'].shift(1).rolling(6).mean()

    if use_spectral:
        # ФИКС LEAKAGE: используем окно, которое ЗАКАНЧИВАЕТСЯ до текущей даты
        for idx, row in df.iterrows():
            mask = spectral['end_dt'] < row['ds']
            if mask.any():
                last = spectral[mask].iloc[-1]
                for col in ['PR_plus', 'frustration', 'lambda_1', 'lambda_2']:
                    df.loc[idx, col] = last[col]

    if use_trends and trends_pivot is not None:
        df = df.merge(trends_pivot[['period'] + trends_cols],
                      left_on='ds', right_on='period', how='left')
        df = df.drop(columns=['period'])
    return df.dropna()

def predict_lgbm(df, H):
    import lightgbm as lgb
    tr, te = df.iloc[:-H], df.iloc[-H:]
    fcols = [c for c in df.columns if c not in ('ds', 'y')]
    m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.05,
                           min_child_samples=3, reg_alpha=0.5, reg_lambda=0.5,
                           verbose=-1, random_state=42)
    m.fit(tr[fcols].values, tr['y'].values)
    return m.predict(te[fcols].values), te['y'].values

def predict_catboost(df, H):
    from catboost import CatBoostRegressor
    tr, te = df.iloc[:-H], df.iloc[-H:]
    fcols = [c for c in df.columns if c not in ('ds', 'y')]
    m = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.05,
                           verbose=0, random_seed=42)
    m.fit(tr[fcols].values, tr['y'].values)
    return m.predict(te[fcols].values), te['y'].values

def main():
    agg, spectral, trends_pivot, trends_cols = load_data()
    print(f"Aggregated: {agg.shape}")
    print(f"Trends: {trends_pivot.shape if trends_pivot is not None else 'None'}")

    results = []
    for cluster in [0, 1]:
        sub = agg[agg['cluster'] == cluster].sort_values('period').copy()
        sub = sub.rename(columns={'period': 'ds', 'value': 'y'})[['ds', 'y']]
        label = CLUSTER_NAMES[cluster]
        print(f"\n=== {label} (n={len(sub)}) ===")

        for H in HORIZONS:
            train, test = sub.iloc[:-H], sub.iloc[-H:]
            y_true = test['y'].values
            y_train = train['y'].values
            ds_test = test['ds'].values

            row = {'cluster': cluster, 'cluster_name': label, 'horizon': H}

            for name, pred in [
                ('naive', predict_naive(y_train, H)),
                ('seasonal', predict_seasonal(y_train, H)),
                ('mean', predict_mean(y_train, H)),
                ('prophet', predict_prophet(train['ds'].values, y_train, ds_test, H)),
            ]:
                row[f'MAE_{name}'] = mean_absolute_error(y_true, pred)
                row[f'R2_{name}'] = r2_score(y_true, pred) if H > 1 else np.nan

            for name, pred_fn in [('lgbm', predict_lgbm),
                                    ('catboost', predict_catboost)]:
                # без трендов
                try:
                    df = make_features(sub, spectral, trends_pivot, trends_cols,
                                        use_trends=False, use_spectral=True)
                    if len(df) >= 10:
                        p, yt = pred_fn(df, H)
                        row[f'MAE_{name}'] = mean_absolute_error(yt, p)
                        row[f'R2_{name}'] = r2_score(yt, p) if H > 1 else np.nan
                except Exception:
                    pass
                # с трендами
                try:
                    df = make_features(sub, spectral, trends_pivot, trends_cols,
                                        use_trends=True, use_spectral=True)
                    if len(df) >= 10:
                        p, yt = pred_fn(df, H)
                        row[f'MAE_{name}_trends'] = mean_absolute_error(yt, p)
                        row[f'R2_{name}_trends'] = r2_score(yt, p) if H > 1 else np.nan
                except Exception:
                    pass

            results.append(row)

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS / 'forecast_all_models.csv',
                  index=False, encoding='utf-8-sig')
    print(f"\n✓ Сохранено: forecast_all_models.csv ({len(res_df)} строк)")

    # Сводка
    print(f"\n{'='*100}")
    print("СВОДКА MAE")
    print(f"{'='*100}")
    models = ['naive', 'seasonal', 'mean', 'prophet',
              'lgbm', 'lgbm_trends', 'catboost', 'catboost_trends']
    print(f"{'Cl':>10} {'H':>3} " + "".join(f"{m:>14}" for m in models))
    for cluster in [0, 1]:
        for H in HORIZONS:
            s = res_df[(res_df['cluster'] == cluster) & (res_df['horizon'] == H)]
            if len(s) == 0: continue
            line = f"{CLUSTER_NAMES[cluster]:>10} {H:>3} "
            for m in models:
                col = f'MAE_{m}'
                if col in s.columns and s[col].notna().any():
                    line += f"{s[col].mean():>14.0f}"
                else:
                    line += f"{'—':>14}"
            print(line)

if __name__ == '__main__':
    main()
