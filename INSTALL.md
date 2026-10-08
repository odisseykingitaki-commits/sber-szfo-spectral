# Установка и воспроизведение

## 1. Окружение

```bash
conda create -n sber python=3.12 -y
conda activate sber
pip install -r requirements.txt
```

Конфиги `configs/config.yaml` и `configs/methods.yaml` — параметры для жюри.

## 2. Данные

Положить исходники конкурса в `data/raw/` (zip/xlsx). Крупные raw zip уже в `.gitignore`.

| Путь | Назначение |
|------|------------|
| `data/processed/features_szfo_v2_final.csv` | **17 признаков, без income** |
| `results/labels_threshold.npy` | канонические метки **140 / 140** |

## 3. Запуск

```bash
conda activate sber
python run_all.py
```

Порядок: `01` → `02` → `03` → `04` → `05` → `06` → **`11` (threshold)** → `07` → `08` → `09` → `10`

- `03_parse_urov.py` оставлен для ablation; финальные признаки **без доходов** (`include_rosstat_income: false`).
- Если `03` падает из‑за отсутствия urov — временно уберите его из `run_all.py`; на no-income результат это не влияет при наличии features CSV.

### По шагам

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

## 4. Канон меток

| Файл | Смысл |
|------|--------|
| `results/labels_threshold.npy` | **основной** метод |
| `results/labels_louvain.npy` | сеть Louvain |
| `results/labels_final.npy` | **deprecated** = копия Louvain |

Фигуры: `figures/fig_final.png`, `fig_network.png`, `fig_dynamic.png`, `fig_bootstrap.png`.

Репозиторий: https://github.com/odisseykingitaki-commits/sber-szfo-spectral

---

## Пересборка PDF (не нужно жюри)

```bash
pip install reportlab pillow pyyaml
python scripts/make_pitch_assets.py
python scripts/build_jury_slides.py
# или: python scripts/build_task1_pdfs.py
```

Выход: `для_жюри/TASK1_method_report.pdf`, `для_жюри/TASK1_presentation.pdf`.
