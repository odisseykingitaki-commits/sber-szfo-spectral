# Как запустить анализ

## 1. Окружение

```bash
conda create -n sber python=3.12 -y
conda activate sber
pip install -r requirements.txt
```

Параметры: `configs/config.yaml`, `configs/methods.yaml`.

## 2. Данные

Положить исходники конкурса в `data/raw/` (zip/xlsx).

Уже есть готовые артефакты для проверки результата:

| Путь | Назначение |
|------|------------|
| `data/processed/features_szfo_v2_final.csv` | 17 признаков |
| `results/labels_threshold.npy` | итоговые метки **140 / 140** |

## 3. Запуск всего пайплайна

```bash
conda activate sber
python run_all.py
```

Порядок скриптов: `01` → `02` → `03` → `04` → `05` → `06` → **`11` (основной метод)** → `07` → `08` → `09` → `10`.

- Скрипт `03` (Росстат) нужен только для проверки «с доходами / без»; в финале доходов нет.
- Если `03` падает из‑за отсутствия файла urov — уберите его из списка в `run_all.py`. На итоговый результат это не влияет, если уже есть features CSV.

### По шагам (по желанию)

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

## 4. Где смотреть результат

| Файл | Смысл |
|------|--------|
| `results/labels_threshold.npy` | основной результат |
| `results/labels_louvain.npy` | сетевой вариант (для сравнения) |
| `figures/` | картинки к отчёту |

Репозиторий: https://github.com/odisseykingitaki-commits/sber-szfo-spectral
