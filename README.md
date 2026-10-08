# Спектральная синергетика экономических систем

PLM-спектральный анализ данных СберИндекса: устойчивая типология
**280 муниципальных образований** Северо-Западного ФО по структуре расходов.

| | |
|--|--|
| Метод | медианный threshold по главной моде PLM |
| Результат | два кластера **140 / 140**, SW ≈ **0.346** |
| Отчёт | [`docs/TASK1_METHOD_REPORT.md`](docs/TASK1_METHOD_REPORT.md) · [PDF](docs/TASK1_method_report.pdf) |
| Слайды | [`docs/TASK1_slides.md`](docs/TASK1_slides.md) · [PDF](docs/TASK1_presentation.pdf) |
| Гайд жюри | [`JURY_GUIDE.md`](JURY_GUIDE.md) |
| Установка | [`INSTALL.md`](INSTALL.md) |
| Репозиторий | https://github.com/odisseykingitaki-commits/sber-szfo-spectral |

## В чём задача

Нужна не «просто кластеризация похожих МО», а **читаемая типология по структуре признаков расходов**: какие направления согласованы, насколько структура напряжена, и можно ли из главной моды честно разрезать карту на два устойчивых типа поведения.

**PLM** (Pairwise Logistic Model) — для каждого признака обучается L2-логистическая регрессия остальных на его бинаризованное значение; коэффициенты собираются в симметричную матрицу зависимостей **J**. Спектр J даёт моды структуры; проекция на первую моду — ось типологии.

## Что сделали

1. Собрали матрицу **p = 17** признаков (доли категорий, лог суммы, рост, вариативность, мобильность) — **без** доходов Росстата.
2. Построили PLM → матрицу **J** (`C_reg = 0.2`) и её спектр (λ_max, PR₊, Frustration).
3. **Канон типологии:** `U1 = X @ v1`, метка `(U1 > median(U1))` → **140 / 140**.
4. Сверили с KMeans (K=2) и Louvain (сеть близости в пространстве мод) — сравнение, не замена канона.
5. Проверили устойчивость: bootstrap, перебор `C_reg`, динамика скользящих окон.

## Быстрый старт

```bash
conda activate sber
pip install -r requirements.txt
# положить сырые данные в data/raw/  (см. INSTALL.md)
python run_all.py
```

Канонические метки: `results/labels_threshold.npy`.  
Сводка чисел: `results/final_no_income_summary.json`.  
Конфиги (параметры для жюри): `configs/config.yaml`, `configs/methods.yaml`.

## Ключевые результаты (threshold, C=0.2, p=17)

| Метрика | Значение |
|---------|----------|
| N МО | 280 |
| Признаков | **17** (no-income) |
| λ_max | ≈ **5.541** |
| λ_max / Σλ₊ | ≈ **47.1%** |
| PR₊ | ≈ **3.44** |
| Frustration F | ≈ **0.59** (10/17) |
| SW | ≈ **0.346** |
| Bootstrap ARI (n=100) | **0.897 ± 0.070** |
| Robustness min ARI (C_reg) | ≈ **0.972** |
| Размеры кластеров | **140 / 140** |
| ARI vs KMeans / Louvain | ≈ 0.835 |

Источники: `results/final_no_income_summary.json`, `results/final_method_summary.json`.

### Именование меток

| Файл | Что это | Размеры |
|------|---------|---------|
| `results/labels_threshold.npy` | **основной** метод | **140 / 140** |
| `results/labels_louvain.npy` | Louvain (сеть) | 149 / 129 / 2 |
| `results/labels_final.npy` | **deprecated** = копия Louvain | 149 / 129 / 2 |

Не путать: связные компоненты графа **[278, 1, 1]** ≠ сообщества Louvain после merge **[149, 129, 2]**.

### Бинаризация (важно)

`StandardScaler` → `X_bin = (X > 0)`. Порог — ноль *после* z-score (эквивалент порога по среднему стандартизованного признака), **не** медиана сырых данных.

### Ablation: доходы Росстата

Добавление income **не** улучшило SW (0.346 no-income vs 0.345 Variant C). В финале income нет. См. `results/variant_c_comparison.json`.

## Ограничения (честно)

1. Мобильность — по сути **две даты** → грубый `mob_logratio`.
2. Bootstrap: scaler на полном N, затем ресемпл строк (лёгкий optimism).
3. Выборка МО = покрытие мобильности СЗФО, не полный справочник ОКТМО.
4. ARI≈0.84 vs KMeans(K=2) слабо доказывает «уникальность» PLM — оба делят по главной оси.
5. `name_norm`: 2 коллизии (`заполярный`, `нарьян-мар`) при внешних join; на no-income threshold не влияют.

## Структура репозитория

```
src/01…11     пайплайн (11 = threshold до ICVI)
configs/      config.yaml, methods.yaml
docs/         метод-отчёт + презентация (+ PDF)
figures/      fig_final, fig_network, fig_dynamic, fig_bootstrap
results/      метки, сводки, ICVI, bootstrap, …
data/         raw / intermediate / processed
```

Подробный аудит меток: `results/AUDIT_LABELS.md`.
