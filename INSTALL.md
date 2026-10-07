# Установка и воспроизведение (Task 1 + Task 2)

Аудитория: ML-инженеры / проверяющие. Цель — поднять окружение и прогнать пайплайн без сюрпризов.

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

Пересобрать PDF метод-отчёта и презентации:

```bash
python scripts/build_method_pdf.py
python scripts/build_slides_pdf.py
```

Выход: `docs/TASK1_method_report.pdf`, `docs/TASK1_presentation.pdf`.

## 5. Git / GitHub

Локальный репозиторий инициализируется командой `git init` (если ещё нет).  
**Публичный remote не создаётся автоматически.** Когда будет готов GitHub:

```bash
git remote add origin https://github.com/<org>/<repo>.git
git push -u origin HEAD
```

В текстах для жюри до появления remote: **«репозиторий: (локально / будет на GitHub)»**.

## 6. Канон vs legacy

| Файл | Смысл |
|------|--------|
| `results/labels_threshold.npy` | **основной** метод |
| `results/labels_louvain.npy` | сеть Louvain |
| `results/labels_final.npy` | **deprecated** = копия Louvain |
