# Спектральная синергетика экономических систем

Применение PLM-спектрального анализа к данным СберИндекса
для кластеризации 280 МО Северо-Западного ФО.

## Быстрый старт

```bash
conda activate sber
pip install -r requirements.txt
python run_all.py
```

Подробнее: [`INSTALL.md`](INSTALL.md).  
Метод-отчёт (Task 1): [`docs/TASK1_METHOD_REPORT.md`](docs/TASK1_METHOD_REPORT.md) · PDF: [`docs/TASK1_method_report.pdf`](docs/TASK1_method_report.pdf).  
Слайды: [`docs/TASK1_slides.md`](docs/TASK1_slides.md) · [`docs/TASK1_presentation.pdf`](docs/TASK1_presentation.pdf).  
Конфиги (source of truth для жюри): `configs/config.yaml`, `configs/methods.yaml`.  
Репозиторий: (локально / будет на GitHub).

## Основной метод

**Медианный threshold по главной моде PLM:**  
`U1 = X @ v1`, метки `(U1 > median(U1))`, `C_reg = 0.2`.

Скрипт: `src/11_final_clusters.py` →  
- `data/processed/clusters_final_threshold.csv` (колонка `cluster`)  
- **`results/labels_threshold.npy`** — канонические метки основного метода (размеры **140 / 140**)  
- `results/final_method_summary.json`

Матрица признаков: **17** (без доходов Росстат) —
5 share + `log_total` + 5 growth + 5 cv + `mob_logratio`.

### Именование меток (важно)

| Файл | Метод | Скрипт | Размеры |
|------|-------|--------|---------|
| `results/labels_threshold.npy` | **основной** median threshold U1 | `11_final_clusters.py` | **140 / 140** |
| `results/labels_louvain.npy` | Louvain (сеть) | `06_network_louvain.py` | 149 / 129 / 2 |
| `results/labels_final.npy` | **deprecated** = копия Louvain | `06` (legacy alias) | 149 / 129 / 2 |

Старое имя `labels_final.npy` **не** является основным методом — это Louvain.  
Не путать с threshold.

Два других подхода — для сопоставления, не как основной результат:
- **KMeans (K=2)** на стандартизованных признаках — baseline;
- **Louvain** на сети близости в пространстве PLM-мод — сетевой анализ.

## Ключевые результаты (threshold, C=0.2, no-income, p=17)

Все метрики ниже — **только threshold** (не Louvain), кроме явно помеченных строк.

| Метрика | Значение | Источник |
|---------|----------|----------|
| N МО | 280 | features |
| Признаков | **17** | features |
| λ_max | +5.541 | 11 / спектр J |
| PR₊ | 3.44 | 11 |
| Frustration | 0.59 (10/17) | 11 |
| SW (threshold) | **0.346** | 11, `labels_threshold.npy` |
| Bootstrap ARI (threshold, n=100) | 0.897 ± 0.070 | `09_bootstrap.py` |
| Robustness ARI (по C_reg ∈ [0.02, 2]) | ≥ 0.972 | `10_robustness.py` |
| ARI vs KMeans | 0.835 | 11 |
| ARI vs Louvain | 0.835 | 11 |
| Размеры кластеров | **140 / 140** | `labels_threshold.npy` |

Сводка: `results/final_no_income_summary.json`, `results/final_method_summary.json`.

### Сетевой анализ (Louvain, отдельно)

| Метрика | Значение |
|---------|----------|
| Файл меток | `labels_louvain.npy` |
| Размеры | 149 / 129 / 2 |
| SW (Louvain) | 0.348 |
| Согласованность с threshold | ARI ≈ 0.835 |

### ICVI (`08_icvi.py`)

Строка **«Наш» / `threshold`** = `labels_threshold.npy`.  
Строка **Louvain** = `labels_louvain.npy` (сравнение, не основной метод).

### `spectral_summary_v2.json` (не threshold!)

Пишется **`05_plm_spectral.py`**: спектр J + **KMeans на нормированных PLM-модах** (U_n).  
`best_SW` ≈ 0.345 — это SW лучшего KMeans-на-модах, **не** median-threshold из 11.  
Спектральные величины (λ_max, PR₊, Frustration) совпадают с 11, потому что J тот же.

### Robustness по C_reg (threshold)

При C_reg от 0.02 до 2.0: PR₊ ∈ [1.75, 5.11], Frustration ∈ [0.47, 0.71];  
разбиение почти не меняется (min ARI vs C=0.2 ≈ 0.972).

## Эксперимент с доходами Росстат (отрицательный результат)

Мы провели эксперимент с добавлением данных Росстата о доходах населения (2018–2024). Однако из-за отсутствия районной разбивки Санкт-Петербурга (103 из 280 МО) и высокой корреляции доходов с потребительскими расходами (corr≈0.75 на matched), это не улучшило кластеризацию: **SW = 0.346 без доходов vs 0.345 с Variant C proxy**. Мы фиксируем это как честный отрицательный результат и оставляем основную матрицу **без income-признаков** (p=17).

Кратко об ablation (см. `results/variant_c_comparison.json`):

| Вариант | p | SW threshold |
|---------|---|--------------|
| **no-income (основной)** | 17 | **0.346** |
| Variant C (OLS-proxy для unmatched) | 20 | 0.345 |
| Residual income | 18 | 0.330 |
| q75-константа (старое) | 20 | 0.331 |

Variant C / q75 — только эксперименты; в финальный CSV income не входит.

## Ограничения

1. **Мобильность: всего 2 даты.** `mob_logratio` = log1p(d2)−log1p(d1); нет сезонной корректировки.

2. **Бинаризация X>0 после StandardScaler.** Порог — ноль *стандартизованного* признака, не медиана сырых данных.

3. **ARI vs KMeans при K=2** — почти тавтология: оба метода делят по главной оси структуры расходов; ARI≈0.84 слабо доказывает независимость PLM.

4. **Louvain eps** выбирается ad-hoc (часто несвязный граф → q=0.40); ARI ≈ 0.835 — согласованность, не независимая валидация.

5. **Bootstrap optimism:** ресемплинг строк после fit StandardScaler на полном N.

6. **Выборка МО = список мобильности СЗФО**, не официальный справочник ОКТМО.

7. **name_norm:** 2 коллизии (`заполярный`, `нарьян-мар`) при join с внешними источниками (для no-income join с urov в основном пайплайне не используется).

## Ответ жюри (кратко)

**Основной метод кластеризации** — медианный threshold по проекции на главную моду PLM (`U1 = X @ v1`), `C_reg=0.2`, без income-признаков (p=17). Результат: два кластера по **140 МО**, SW≈0.346, bootstrap ARI≈0.90, устойчивость по C_reg (min ARI≥0.97).  
Louvain — вспомогательный сетевой анализ (ARI≈0.84 с threshold); KMeans K=2 — baseline.  
Доходы Росстат не улучшили качество (отрицательный ablation).

Подробный аудит меток: `results/AUDIT_LABELS.md`.

## Вторая задача: прогнозирование + структурные сдвиги

Те же 280 МО СЗФО × 24 месяца (2023–2024), метки threshold **140 / 140**,
спектральные метрики J(t) по 19 окнам, плюс Google Trends
(`data/processed/google_trends_szfo.csv`).

Скрипты: `src/12_timeseries.py`, `src/13_changepoints.py` (в `run_all.py` после 01–11).  
Зависимости: см. `requirements.txt` (`prophet`, `lightgbm`, `catboost`, `ruptures`, `pytrends`).

**Leakage fix:** спектральные признаки только из окон с `end_dt < ds`
(раньше ошибочно `start_dt <= ds`). Канон MAE: **`results/forecast_all_models.csv`**
(черновики `forecast_metrics*.csv` игнорировать). Подробности: `results/TASK2_NOTES.md`,
`results/forecast_mae_post_leakage.md`.

### A. Прогноз расходов — смешанный / отрицательный итог

Модели: Naive, Seasonal, Mean, Prophet (`yearly_seasonality=False`),
LightGBM / CatBoost (лаги + MA + PR₊/Frustration/λ ± Trends).

**Нет единой победившей модели.** По горизонтам и кластерам лучшие разные:
prophet (городские H=1/3/6, сельские H=12), lgbm только сельские H=3,
naive/seasonal конкурентоспособны на остальных. Google Trends **не** дают
стабильного выигрыша. H=12 для LGBM/CatBoost пуст (мало точек после lag_12).

**MAE — сельские (cluster 0)** — округление из `forecast_all_models.csv`:

| H | naive | seasonal | mean | Prophet | LGBM | LGBM+Trends | winner |
|---|------:|---------:|-----:|--------:|-----:|------------:|--------|
| 1 | 5198 | **4314** | 7179 | 4375 | 5223 | 4930 | seasonal |
| 3 | 3133 | 3862 | 3903 | 1839 | **1761** | 1802 | lgbm |
| 6 | **1497** | 3663 | 3617 | 1618 | 1649 | 1680 | naive |
| 12 | 1889 | 2997 | 3369 | **1514** | — | — | prophet |

**MAE — городские (cluster 1):**

| H | naive | seasonal | mean | Prophet | LGBM | LGBM+Trends | winner |
|---|------:|---------:|-----:|--------:|-----:|------------:|--------|
| 1 | 7222 | 7283 | 12388 | **5642** | 8479 | 8926 | prophet |
| 3 | 4713 | 6386 | 7897 | **2653** | 5241 | 5356 | prophet |
| 6 | 2168 | 6416 | 7122 | **2040** | 2783 | 2889 | prophet |
| 12 | **2525** | 6609 | 6675 | 3416 | — | — | naive |

Ключевые находки (post-leakage):
1. Prophet силён на городских; LGBM — только сельские H=3; naive конкурирует.
2. Trends часто хуже no-trends после закрытия утечки спектра (`end_dt < ds`).
3. Старые «огромные» MAE Prophet (десятки тысяч) — артефакт до пересчёта, не итог.
4. Ориентир — **MAE** (R² часто < 0 на 24 точках).

### B. Структурные сдвиги — положительный результат

Сигнал: **[λ₁…λ₅, PR₊]** (без Frustration), дата = конец окна. **9 методов.**

**Консенсус (≥50%):** **2023-11-01** (5/9), **2024-04-01** (5/9), **2024-09-01** (7/9).

Файлы: `changepoints_all_methods.csv`, `changepoints_consensus.csv`,
`figures/fig_changepoints_consensus.png`.

## Структура

- `src/` — пайплайн `01`→`11` (задача 1; 11 перед 08) + `12`/`13` (задача 2); cwd=`src`
- `data/raw/` — исходные zip/xlsx
- `data/intermediate/` — промежуточные parquet
- `data/processed/` — финальные CSV (`features_szfo_v2_final.csv` — 17 признаков; `google_trends_szfo.csv`)
- `figures/` — картинки (в т.ч. `fig_aggregated.png`, `fig_changepoints_consensus.png`)
- `results/` — таблицы и метрики (`labels_threshold.npy`, `forecast_*.csv`, `changepoints_*.csv`, …)
