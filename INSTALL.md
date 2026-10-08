# Установка и воспроизведение

Аудитория: ML-инженеры / проверяющие. Цель — поднять окружение и прогнать пайплайн типологии без сюрпризов.

## 1. Окружение

```bash
conda create -n sber python=3.12 -y
conda activate sber
pip install -r requirements.txt
```

Опционально для пересборки PDF:

```bash
pip install reportlab pillow pyyaml
```

Конфиги `configs/config.yaml` и `configs/methods.yaml` — source of truth по параметрам для жюри. Скрипты в `src/` используют те же константы в коде; загрузка YAML доступна через `utils.load_config()` / `utils.load_methods()`.

## 2. Данные

Положить исходники конкурса в `data/raw/` (zip/xlsx). Крупные raw zip уже в `.gitignore`.

Ожидаемые артефакты после парсинга / фич:

| Путь | Назначение |
|------|------------|
| `data/intermediate/spend.parquet` | расходы |
| `data/intermediate/mobility.parquet` | мобильность |
| `data/processed/features_szfo_v2_final.csv` | **17 признаков, без income** |
| `results/labels_threshold.npy` | канонические метки **140 / 140** |

Большинство `*.npy` в `.gitignore` — после клона пересоберите пайплайном (исключения: `labels_threshold.npy`, `labels_louvain.npy`). JSON/CSV в `results/` обычно уже в репозитории как сводки.

## 3. Запуск

Из корня репозитория:

```bash
conda activate sber
python run_all.py
```

Порядок внутри `run_all.py`:

`01` → `02` → `03` → `04` → `05` → `06` → **`11` (threshold)** → `07` → `08` → `09` → `10`

### Про шаг `03` (urov / Росстат)

- `03_parse_urov.py` **оставлен** в `run_all` (удобно для ablation).
- Финальные признаки собирает `04_features.py` **без доходов** (`include_rosstat_income: false` в `configs/config.yaml`).
- Даже если `03` отработал, матрица для PLM остаётся **p=17, no-income**.
- Ablation с income: `results/variant_c_comparison.json` (SW 0.346 vs 0.345).

Если `03` падает из‑за отсутствия файла urov, временно уберите его из списка в `run_all.py` — на no-income результат это не влияет, пока есть `features_szfo_v2_final.csv`.

### По шагам (эквивалент `run_all`)

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

## 4. Фигуры и PDF

| Файл | Скрипт |
|------|--------|
| `figures/fig_network.png` | `06_network_louvain.py` |
| `figures/fig_dynamic.png` | `07_dynamics.py` |
| `figures/fig_bootstrap.png` | `09_bootstrap.py` |
| `figures/fig_final.png` | `10_robustness.py` |

Пересобрать PDF метод-отчёта и презентации:

```bash
python scripts/make_pitch_assets.py      # PNG → docs/presentation_assets/
python scripts/build_jury_slides.py     # chart-first 11 slides PDF
# или вместе с method report:
python scripts/build_task1_pdfs.py
```

Выход: `docs/TASK1_method_report.pdf`, `docs/TASK1_presentation.pdf`.

## 5. Репозиторий

https://github.com/odisseykingitaki-commits/sber-szfo-spectral

## 6. Канон vs legacy

| Файл | Смысл |
|------|--------|
| `results/labels_threshold.npy` | **основной** метод |
| `results/labels_louvain.npy` | сеть Louvain |
| `results/labels_final.npy` | **deprecated** = копия Louvain |
