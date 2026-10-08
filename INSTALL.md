# Установка и воспроизведение (Task 1 + Task 2)

Аудитория: ML-инженеры / проверяющие. Цель — поднять окружение и прогнать пайплайн без сюрпризов.

> **Task 2:** прогноз расходов по двум режимам потребления + changepoints.  
> Канон метрик: `results/forecast_all_models.csv` (после фикса `end_dt < ds`).  
> Пакет документов: `docs/TASK2_METHOD_REPORT.md`, `docs/TASK2_slides.md`, PDF в `docs/`.  
> Task 2 можно понять и запустить после появления features/labels; жюри, читающее только материалы Task 2, должно понять сюжет из `TASK2_METHOD_REPORT.md` (§0) без отчёта Task 1.

## 1. Окружение

```bash
conda create -n sber python=3.12 -y
conda activate sber
pip install -r requirements.txt
```

Опционально для сборки PDF из этого пакета:

```bash
pip install reportlab pillow pyyaml
# для презентации PPTX (если нужен редактируемый файл):
pip install python-pptx
```

Конфиги `configs/*.yaml` — source of truth для жюри. Скрипты в `src/` пока используют те же константы в коде; лёгкая загрузка YAML доступна через `utils.load_config()` / `utils.load_methods()` без обязательного перевода всего пайплайна на YAML.

## 2. Данные

Положить исходники в `data/raw/` (zip/xlsx конкурса). Крупные raw zip уже в `.gitignore`.

Ожидаемые промежуточные/обработанные артефакты после парсинга:

| Путь | Назначение |
|------|------------|
| `data/intermediate/spend.parquet` | расходы |
| `data/intermediate/mobility.parquet` | мобильность |
| `data/processed/features_szfo_v2_final.csv` | **17 признаков, без income** |
| `results/labels_threshold.npy` | канонические метки 140/140 |

`*.npy` в `.gitignore` — после клона пересоберите пайплайном или скопируйте локальные `results/*.npy`. JSON/CSV в `results/` обычно коммитятся как сводки.

## 3. Запуск

Из корня репозитория:

```bash
conda activate sber
python run_all.py
```

Порядок внутри `run_all.py`:

`01` → `02` → `03` → `04` → `05` → `06` → **`11` (threshold)** → `07` → `08` → `09` → `10` → `12` → `13`

### Про шаг `03` (urov / Росстат)

- `03_parse_urov.py` **оставлен** в `run_all` (удобно для ablation).
- Финальные признаки собирает `04_features.py` **без доходов** (`include_rosstat_income: false` в `configs/config.yaml`).
- Даже если `03` отработал, матрица для PLM остаётся **p=17, no-income**.
- Ablation с income: см. `results/variant_c_comparison.json` (SW 0.346 vs 0.345). Данные urov — **доходы**, не занятость/зарплаты из brief.

Если `03` падает из‑за отсутствия файла urov, можно временно убрать его из списка в `run_all.py` — на Task 1 no-income это не влияет, пока есть `features_szfo_v2_final.csv`.

### Только Task 1 (без прогноза)

```bash
cd src
python 01_parse_spend.py
python 02_parse_mobility.py
# python 03_parse_urov.py   # опционально
python 04_features.py
python 05_plm_spectral.py
python 06_network_louvain.py
python 11_final_clusters.py
python 07_dynamics.py
python 08_icvi.py
python 09_bootstrap.py
python 10_robustness.py
```

## 4. Фигуры

После прогона появляются / обновляются:

| Файл | Скрипт |
|------|--------|
| `figures/fig_network.png` | `06_network_louvain.py` |
| `figures/fig_dynamic.png` | `07_dynamics.py` |
| `figures/fig_bootstrap.png` | `09_bootstrap.py` |
| `figures/fig_final.png` | `10_robustness.py` |
| `figures/fig_changepoints_consensus.png` | `13_changepoints.py` |

Пересобрать PDF метод-отчёта и презентации **Task 1**:

```bash
python scripts/make_pitch_assets.py      # PNG → docs/presentation_assets/
python scripts/build_task1_pdfs.py       # или build_method_pdf / build_slides_pdf
```

Выход: `docs/TASK1_method_report.pdf`, `docs/TASK1_presentation.pdf`.

## 4b. Task 2 — прогноз и changepoints

### Зависимости (уже в `requirements.txt`)

| Пакет | Назначение |
|-------|------------|
| `prophet` (+ `cmdstanpy`, `holidays`) | baseline / сильный классический прогноз |
| `lightgbm` | ML-прогноз + спектральные фичи (± Trends) |
| `catboost` | ML-прогноз (сравнение с LGBM) |
| `ruptures` | 9 методов поиска структурных сдвигов |
| `pytrends` | выгрузка Google Trends (если нужно пересобрать CSV) |

```bash
conda activate sber
pip install -r requirements.txt
# PDF-пакет Task 2:
pip install reportlab pillow pyyaml
```

Конфиги жюри: `configs/task2_config.yaml`, `configs/task2_methods.yaml`.

### Только Task 2 (нужны готовые признаки/метки)

Сначала должны существовать:

- `results/labels_threshold.npy` (Тип A / Тип B, 140/140)
- `results/dynamic_summary.json` (скользящие окна PLM → спектральный сигнал)
- `data/intermediate/spend.parquet`
- опционально `data/processed/google_trends_szfo.csv`

```bash
cd src
python 12_timeseries.py    # → results/forecast_all_models.csv
python 13_changepoints.py  # → changepoints_*.csv, figures/fig_changepoints_consensus.png
```

**Правило анти-утечки (обязательно):** спектральные признаки только из окон с `end_dt < ds`  
(в коде: `mask = spectral['end_dt'] < row['ds']`). Старые черновики `forecast_metrics*.csv` — **игнорировать**.

### PDF и pitch-ассеты Task 2

```bash
python scripts/make_task2_pitch_assets.py   # PNG → docs/presentation_assets_task2/
python scripts/build_task2_pdfs.py          # PDF отчёт + презентация
```

Выход:

- `docs/TASK2_method_report.pdf`
- `docs/TASK2_presentation.pdf`
- `docs/TASK2_METHOD_REPORT.md`, `docs/TASK2_slides.md`, `docs/TASK2_GAPS.md`

Честные пробелы критериев (foundation models, news): см. `docs/TASK2_GAPS.md`.

## 5. Git / GitHub

Публичный репозиторий: **https://github.com/odisseykingitaki-commits/sber-szfo-spectral**

```bash
git remote add origin https://github.com/odisseykingitaki-commits/sber-szfo-spectral.git
git push -u origin HEAD
```

## 6. Канон vs legacy

| Файл | Смысл |
|------|--------|
| `results/labels_threshold.npy` | **основной** метод |
| `results/labels_louvain.npy` | сеть Louvain |
| `results/labels_final.npy` | **deprecated** = копия Louvain |
