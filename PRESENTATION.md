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
| λ_max | ≈ 5.541 |
| λ_max / Σλ₊ | ≈ 47.1% |
| PR₊ | ≈ 3.44 |
| Frustration | ≈ 0.59 |
| Bootstrap ARI (n=100) | 0.897 ± 0.070 |
| Robustness min ARI (C_reg) | ≥ 0.972 |
| ARI vs KMeans / Louvain | ≈ 0.835 |

## Что не основной метод

- **Louvain** (`labels_louvain.npy`, 149/129/2) — сетевой анализ, SW≈0.348  
- **KMeans на PLM-модах** (`spectral_summary_v2.json` из `05`) — разведочная кластеризация, не финал  
- Старое имя `labels_final.npy` = Louvain (deprecated)

## Income

Добавление доходов Росстат не улучшило SW (0.346 → 0.345 Variant C) — честный отрицательный результат; в финале income нет.

## Материалы

- Отчёт: `docs/TASK1_METHOD_REPORT.md` / `docs/TASK1_method_report.pdf`
- Слайды: `docs/TASK1_slides.md` / `docs/TASK1_presentation.pdf`
- Гайд: `JURY_GUIDE.md`
