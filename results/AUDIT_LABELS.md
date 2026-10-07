# AUDIT: threshold vs Louvain labels

Дата: 2026-10-03. Env: `conda run -n sber`.

## Что было не так

1. **`results/labels_final.npy` = Louvain (149/129/2), не threshold.**  
   Писался `06_network_louvain.py`, имя «final» вводило в заблуждение.

2. **`08_icvi.py` раньше грузил `labels_final.npy` как «Наш».**  
   В CSV строка называлась `Louvain` (SW≈0.348), но в print/смысле это подавалось как наш метод.  
   Основной метод (threshold, SW≈0.346, 140/140) в ICVI **отсутствовал**.

3. **Путаница в интерпретации файлов для DeepSeek/жюри**, если брать `labels_final.npy` за финал.

## Что уже было корректно (не трогали полный bootstrap)

| Артефакт | Скрипт | Метод | Вердикт |
|----------|--------|-------|---------|
| `clusters_final_threshold.csv` | 11 | threshold U1 | OK, cluster 140/140 |
| `final_method_summary.json` | 11 | threshold | OK |
| `final_no_income_summary.json` | (сводка) | threshold + boot/rob | OK |
| `bootstrap_stability.csv` | 09 | **threshold on U1** | OK, mean ARI≈0.897 |
| `robustness.csv` | 10 | **threshold on U1** | OK, min ARI≈0.972 |
| `spectral_summary_v2.json` | 05 | спектр + **KMeans на модах** | OK, но это **не** threshold |
| `adjacency_final.npy` / Louvain CSV | 06 | Louvain | OK как сеть |
| `kmeans_vs_ours_no_income.json` | cmp | уже различал thr vs Louvain | OK |

**Bootstrap / robustness перезапускать не нужно** — они уже считаются на threshold.

## Что исправлено

### Именование меток

| Файл | Содержание |
|------|------------|
| `results/labels_threshold.npy` | основной метод, 140/140 (пишет 11) |
| `results/labels_louvain.npy` | Louvain, 149/129/2 (пишет 06) |
| `results/labels_final.npy` | **deprecated** alias = копия Louvain (06) |

### Скрипты

- **06** → `np.save(labels_louvain.npy)` + legacy `labels_final.npy`
- **11** → `np.save(labels_threshold.npy)`; ARI vs Louvain из `labels_louvain.npy`
- **08** → «Наш» = threshold; отдельная строка Louvain; в CSV: `threshold` + `Louvain`
- **run_all.py** → 11 перед 08 (чтобы ICVI видел `labels_threshold.npy`)
- cmp / `_kmeans_vs_ours_*` → пути на новые имена

### Перезапуск

Перезапущены: **06, 11, 08**. Подтверждено:

```
labels_threshold.npy  sizes = [140, 140]
labels_louvain.npy    sizes = [149, 129, 2]
ARI(threshold, Louvain) ≈ 0.835
ICVI: threshold SW=0.346, Louvain SW=0.348, KMeans-2 SW=0.347
```

## Provenance краткая карта

```
05_plm_spectral.py  → spectral_summary_v2.json, J_v2, final_clusters_v2 (KMeans на модах)
06_network_louvain  → labels_louvain.npy, clusters_final_louvain.csv
09_bootstrap        → bootstrap_stability.csv          [threshold U1]
10_robustness       → robustness.csv                   [threshold U1]
11_final_clusters   → labels_threshold.npy, clusters_final_threshold.csv, final_method_summary.json
08_icvi             → icvi_full.csv                    [«Наш»=threshold; +Louvain]
```

## README

Обновлён: основной метод = threshold; таблица только threshold-метрик;  
явное предупреждение про `spectral_summary_v2` (05, не threshold);  
секция ответа жюри; ссылка на этот аудит.
