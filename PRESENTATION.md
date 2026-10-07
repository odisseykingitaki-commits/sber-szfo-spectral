# Фрагмент для презентации / ответа жюри

## Основной метод

Медианный **threshold** по главной моде PLM:  
`U1 = X @ v1`, метка `(U1 > median(U1))`, `C_reg = 0.2`, **без** доходов Росстат (**p = 17**).

- Метки: `results/labels_threshold.npy` — **140 / 140**
- CSV: `data/processed/clusters_final_threshold.csv`
- Скрипт: `src/11_final_clusters.py`

## Ключевые цифры (только threshold)

| | |
|--|--|
| SW | 0.346 |
| Bootstrap ARI (n=100) | 0.897 ± 0.070 |
| Robustness min ARI (C_reg) | ≥ 0.972 |
| ARI vs KMeans / Louvain | ≈ 0.835 |

## Что не основной метод

- **Louvain** (`labels_louvain.npy`, 149/129/2) — сетевой анализ, SW≈0.348  
- **KMeans на PLM-модах** (`spectral_summary_v2.json` из `05`) — разведочная кластеризация, не финал  
- Старое имя `labels_final.npy` = Louvain (deprecated)

## Income

Добавление доходов Росстат не улучшило SW (0.346 → 0.345 Variant C) — честный отрицательный результат; в финале income нет.

## Задача 2 (прогноз + сдвиги)

**Прогноз (после leakage fix `end_dt < ds`) — смешанный итог, нет доминирующей модели:**
- Городские: Prophet лучший на H=1/3/6; H=12 — naive.
- Сельские: seasonal (H=1), lgbm (H=3), naive (H=6), prophet (H=12).
- Google Trends не стабильно помогают; LGBM+Trends часто хуже LGBM без Trends.
- Канон MAE: `results/forecast_all_models.csv`. Ориентир — **MAE**, не R².

**Структурные сдвиги — положительный результат:**
- 9 методов; консенсус (≥50%): **2023-11, 2024-04, 2024-09**.
