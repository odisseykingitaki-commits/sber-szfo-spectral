# Task 1 — слайды (8–10)

Источник чисел: `results/TASK1_GUIDE_DUMP.md`, `final_no_income_summary.json`.  
Репозиторий: (локально / будет на GitHub).

---

## Слайд 1 — Титул

**Спектральная типология расходов МО СЗФО**  
PLM → спектр \(J\) → median threshold  
Задача 1 конкурса · 280 МО · 2023–2024  
Репозиторий: локально / будет на GitHub

---

## Слайд 2 — Проблема / почему не только KMeans

- KMeans: *похожи ли объекты* в \(X\)
- Нам нужно: *какие признаки согласованы*, Frustration / PR₊, устойчивый разрез
- Канон: **threshold по главной моде**, не «ещё один KMeans»

---

## Слайд 3 — Данные + 17 признаков

- N = 280 МО СЗФО (список мобильности)
- 5 share + `log_total` (`log1p`) + 5 growth + 5 cv + `mob_logratio`
- **Без** доходов Росстата в финале
- Бинаризация: z-score, затем \(X>0\)

---

## Слайд 4 — Метод: PLM → спектр → threshold (+ Louvain)

1. PLM: \(C_{\mathrm{reg}}=0.2\) → матрица \(J\)
2. Спектр: \(\lambda_{\max}\approx 5.54\), PR₊≈3.44, F≈0.59
3. **Канон:** \(U_1=X v_1\), label = выше медианы → **140/140**
4. **Сеть:** ε=q₀.₄₀; компоненты [278,1,1]; Louvain на полном графе → после merge [149,129,2]

---

## Слайд 5 — Спектр и мода 1

- \(\lambda_{\max}/\Sigma\lambda_+ \approx 47.1\%\)
- Мода 1: food (+), grocery (−), transport (+), log_total (+), health (+)
- Интерпретация: городской / сельский **паттерн расходов** (не юрстатус)

---

## Слайд 6 — Кластеры и метрики

| | Threshold | Louvain |
|--|----------:|--------:|
| Размеры | **140/140** | 149/129/2 |
| SW | **0.346** | 0.348 |
| CH | **145.25** | **80.94** |

ICVI полный набор — `icvi_full.csv` (строка `threshold`).

---

## Слайд 7 — Устойчивость и динамика

- Bootstrap ARI (n=100): **0.897 ± 0.070**
- Robustness \(C_{\mathrm{reg}}\): **min ARI ≈ 0.972** (0.97153)
- Окна: перебежчики ≈ 17.7% — структура живая, не хаос
- Фигуры: `fig_bootstrap`, `fig_final`, `fig_dynamic`

---

## Слайд 8 — Сравнение и честные пределы

- ARI threshold vs KMeans / Louvain ≈ **0.835**
- **Rosstat ablation:** SW 0.346 (no-income) vs 0.345 (Variant C) — `variant_c_comparison.json`
- `urov` = доходы ≠ занятость/зарплаты брифа; СПб без районной разбивки
- Louvain ≠ канон; `labels_final.npy` = legacy Louvain

---

## Слайд 9 — Практика + воспроизводимость

- Региональный мониторинг паттернов расходов / риск смены режима / окна \(J(t)\)
- `conda activate sber` → `pip install -r requirements.txt` → `python run_all.py`
- Конфиги: `configs/config.yaml`, `methods.yaml`
- Инструкция: `INSTALL.md`

---

## Слайд 10 — Takeaways

1. Канон Task 1 = **median threshold U1**, 140/140, SW≈0.346  
2. Устойчивость: bootstrap≈0.90, min ARI≈0.972  
3. Louvain — сеть (CH=80.94); income — отрицательный ablation  
4. Путь: `docs/TASK1_METHOD_REPORT.md` + `figures/`  
5. Репозиторий: **локально / будет на GitHub**
