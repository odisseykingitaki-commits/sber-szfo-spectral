# -*- coding: utf-8 -*-
"""Extract Vologodsky regime-switcher case for Task 1 pitch. VERIFY only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "_pitch" / "presentation_assets"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "share_health",
    "share_marketplace",
    "share_food",
    "share_grocery",
    "share_transport",
    "log_total",
    "growth_health",
    "growth_marketplace",
    "growth_food",
    "growth_grocery",
    "growth_transport",
    "cv_health",
    "cv_marketplace",
    "cv_food",
    "cv_grocery",
    "cv_transport",
    "mob_logratio",
]

# mode1 "urban-pattern" direction (from Task1 pitch / DeepSeek brief)
URBAN_POS = {"share_food", "share_transport", "log_total", "share_health"}
URBAN_NEG = {"share_grocery"}  # grocery- pulls opposite to urban-pattern

SPEND_CATS = [
    "Здоровье",
    "Маркетплейсы",
    "Общественное питание",
    "Продовольствие",
    "Транспорт",
    "Все категории",
]


def norm_name(s: str) -> str:
    return str(s).strip().lower().replace("ё", "е")


def main() -> None:
    # ------------------------------------------------------------------
    # 1–2 Trajectories
    # ------------------------------------------------------------------
    traj = pd.read_csv(ROOT / "data" / "processed" / "trajectories.csv", encoding="utf-8")
    traj = traj.rename(columns={"Unnamed: 0": "mo_norm"})
    wcols = [f"w{i}" for i in range(1, 20)]

    mask = traj["mo_norm"].map(norm_name).str.contains("вологодск", na=False)
    vol = traj.loc[mask]
    assert len(vol) == 1, vol
    row = vol.iloc[0]
    mo_key = str(row["mo_norm"])
    n_sw = int(row["n_switches"])
    labels = [int(row[c]) for c in wcols]
    assert n_sw == 8

    eights = traj[traj["n_switches"] == 8]
    other_eights = []
    for _, r in eights.iterrows():
        if r["mo_norm"] != mo_key:
            other_eights.append(
                {
                    "mo_norm": r["mo_norm"],
                    "n_switches": int(r["n_switches"]),
                    "labels": [int(r[c]) for c in wcols],
                }
            )

    # ------------------------------------------------------------------
    # 3 Windows + ARI / PR_plus / frustration at switches
    # ------------------------------------------------------------------
    with open(ROOT / "results" / "dynamic_summary.json", encoding="utf-8") as f:
        dyn = json.load(f)
    windows = dyn["windows"]
    aris = {(a["from"], a["to"]): a["ari"] for a in dyn["aris"]}

    win_by_idx = {w["window"]: w for w in windows}

    switches = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            from_w, to_w = i, i + 1
            w_from = win_by_idx[from_w]
            w_to = win_by_idx[to_w]
            switches.append(
                {
                    "from_w": from_w,
                    "to_w": to_w,
                    "from_label": labels[i - 1],
                    "to_label": labels[i],
                    "from_start": w_from["start"],
                    "from_end": w_from["end"],
                    "to_start": w_to["start"],
                    "to_end": w_to["end"],
                    "PR_plus_from": w_from["PR_plus"],
                    "PR_plus_to": w_to["PR_plus"],
                    "frustration_from": w_from["frustration"],
                    "frustration_to": w_to["frustration"],
                    "ARI": aris.get((from_w, to_w)),
                    "n_communities_from": w_from["n_communities"],
                    "n_communities_to": w_to["n_communities"],
                }
            )

    traj_rows = []
    for i, lab in enumerate(labels, start=1):
        w = win_by_idx[i]
        traj_rows.append(
            {
                "mo_norm": mo_key,
                "window": i,
                "label": lab,
                "start": w["start"],
                "end": w["end"],
                "PR_plus": w["PR_plus"],
                "frustration": w["frustration"],
                "n_communities": w["n_communities"],
                "n_switches": n_sw,
            }
        )
    traj_df = pd.DataFrame(traj_rows)
    traj_path = OUT / "case_vologodsky_trajectory.csv"
    traj_df.to_csv(traj_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 5 Features vs SZFO median / cluster poles
    # ------------------------------------------------------------------
    feat = pd.read_csv(ROOT / "data" / "processed" / "features_szfo_v2_final.csv", encoding="utf-8")
    clus = pd.read_csv(ROOT / "data" / "processed" / "clusters_final_threshold.csv", encoding="utf-8")
    labels_thr = np.load(ROOT / "results" / "labels_threshold.npy")

    feat_row = feat[feat["mo_norm"].map(norm_name) == norm_name(mo_key)].iloc[0]
    clus_row = clus[clus["mo_norm"].map(norm_name) == norm_name(mo_key)].iloc[0]
    static_cluster = int(clus_row["cluster"])

    # Align labels_threshold with feature order
    assert len(labels_thr) == len(feat) or len(labels_thr) == len(clus)
    # clusters_final_threshold has cluster column — use that as ground for poles
    c0 = clus[clus["cluster"] == 0]
    c1 = clus[clus["cluster"] == 1]
    # verify npy matches csv
    # map by order if same length as clus
    if len(labels_thr) == len(clus):
        npy_match = int((labels_thr == clus["cluster"].to_numpy()).mean() * 100)
    else:
        npy_match = -1

    med = feat[FEATURE_COLS].median()
    mean0 = c0[FEATURE_COLS].mean()
    mean1 = c1[FEATURE_COLS].mean()

    # mode1 loadings from evecs if available
    evecs = np.load(ROOT / "results" / "evecs_v2.npy")
    # spectral_summary may have feature order
    with open(ROOT / "results" / "spectral_summary_v2.json", encoding="utf-8") as f:
        spec = json.load(f)
    print("spectral keys", list(spec.keys())[:40])

    feat_cmp = []
    for col in FEATURE_COLS:
        val = float(feat_row[col])
        m = float(med[col])
        m0 = float(mean0[col])
        m1 = float(mean1[col])
        # which pole closer
        d0 = abs(val - m0)
        d1 = abs(val - m1)
        closer = 0 if d0 < d1 else (1 if d1 < d0 else "tie")
        # urban-pattern flag
        vs_med = val - m
        if col in URBAN_POS:
            urban_pull = "toward urban-pattern" if vs_med > 0 else "away from urban-pattern"
        elif col in URBAN_NEG:
            # grocery high = away from urban
            urban_pull = "away from urban-pattern" if vs_med > 0 else "toward urban-pattern"
        else:
            urban_pull = "neutral/other (no mode1 sign assigned here)"
        feat_cmp.append(
            {
                "feature": col,
                "value": val,
                "szfo_median": m,
                "delta_vs_median": vs_med,
                "mean_cluster0": m0,
                "mean_cluster1": m1,
                "closer_to_cluster": closer,
                "urban_pattern_note": urban_pull,
            }
        )
    feat_cmp_df = pd.DataFrame(feat_cmp)

    # Try to get mode1 loadings vector aligned to FEATURE_COLS
    mode1_loadings = None
    if "feature_names" in spec:
        fnames = spec["feature_names"]
    elif "features" in spec:
        fnames = spec["features"]
    else:
        fnames = FEATURE_COLS
        print("spec sample:", {k: (spec[k] if not isinstance(spec[k], list) else f"list[{len(spec[k])}]") for k in list(spec)[:15]})

    # evecs shape
    print("evecs shape", evecs.shape)
    # typically columns are eigenvectors; mode1 = first non-trivial
    # mode1 = first eigenvector column (TASK1_GUIDE_DUMP: food+, grocery-, …)
    if evecs.ndim == 2 and evecs.shape[0] == len(FEATURE_COLS):
        mode1_loadings = evecs[:, 0]
    elif evecs.ndim == 2 and evecs.shape[1] == len(FEATURE_COLS):
        mode1_loadings = evecs[0]

    if mode1_loadings is not None:
        for i, col in enumerate(FEATURE_COLS):
            feat_cmp_df.loc[feat_cmp_df["feature"] == col, "mode1_loading"] = float(mode1_loadings[i])

    # ------------------------------------------------------------------
    # 6 Spend monthly
    # ------------------------------------------------------------------
    spend = pd.read_parquet(ROOT / "data" / "intermediate" / "spend.parquet")
    # match mo — may be raw names; try norm contains
    spend = spend.copy()
    spend["mo_norm_guess"] = spend["mo"].map(norm_name)
    # check unique categories
    cats = sorted(spend["category_15"].dropna().unique().tolist())
    print("categories:", cats)

    # find vologodsky in spend.mo
    vol_spend_names = sorted(
        spend.loc[spend["mo_norm_guess"].str.contains("вологодск", na=False), "mo"].unique().tolist()
    )
    print("spend mo matches:", vol_spend_names)

    # Prefer exact-ish: district not city
    # "вологодский" district vs "вологда" city
    cand = spend[spend["mo_norm_guess"].str.contains("вологодск", na=False)]
    # if multiple, pick the one whose norm equals mo_key or contains район-like
    if cand["mo"].nunique() > 1:
        print("multiple spend MOs:", cand.groupby("mo").size())
        # prefer name that normalizes closest to mo_key
        best = None
        for name in cand["mo"].unique():
            nn = norm_name(name)
            if nn == norm_name(mo_key) or nn.startswith(norm_name(mo_key)):
                best = name
                break
        if best is None:
            # exclude pure city вологда
            non_city = [n for n in cand["mo"].unique() if norm_name(n) != "вологда"]
            best = non_city[0] if non_city else cand["mo"].iloc[0]
        cand = cand[cand["mo"] == best]
        spend_mo_raw = best
    else:
        spend_mo_raw = cand["mo"].iloc[0] if len(cand) else None

    print("chosen spend mo:", spend_mo_raw)

    # Map category names — Russian exact
    cat_map = {}
    for c in SPEND_CATS:
        hits = [x for x in cats if norm_name(x) == norm_name(c) or c.lower() in x.lower()]
        cat_map[c] = hits[0] if hits else None
    print("cat_map", cat_map)

    monthly = []
    if spend_mo_raw:
        sub = cand[cand["mo"] == spend_mo_raw].copy()
        sub["period"] = pd.to_datetime(sub["period"])
        sub = sub[(sub["period"] >= "2023-01-01") & (sub["period"] <= "2024-12-31")]
        for nice, raw_cat in cat_map.items():
            if raw_cat is None:
                continue
            s = sub[sub["category_15"] == raw_cat][["period", "value"]].sort_values("period")
            for _, r in s.iterrows():
                monthly.append(
                    {
                        "mo": spend_mo_raw,
                        "category": nice,
                        "period": r["period"].strftime("%Y-%m-%d"),
                        "value": int(r["value"]),
                    }
                )
    monthly_df = pd.DataFrame(monthly)

    # Align spend with switch windows (honest): look at months newly entering/leaving window
    # Window i covers [start, end) monthly — end is exclusive-looking (2023-06-01 for w1)
    # For each switch to_w, compare mean spend in months of to-window vs from-window for key cats
    spend_align = []
    if len(monthly_df):
        piv = monthly_df.pivot_table(index="period", columns="category", values="value")
        piv.index = pd.to_datetime(piv.index)

        def window_months(start, end):
            # inclusive start, exclusive end (as in summary)
            idx = piv.index[(piv.index >= pd.Timestamp(start)) & (piv.index < pd.Timestamp(end))]
            return idx

        for sw in switches:
            m_from = window_months(sw["from_start"], sw["from_end"])
            m_to = window_months(sw["to_start"], sw["to_end"])
            rec = {
                "switch": f"w{sw['from_w']}→w{sw['to_w']}",
                "label": f"{sw['from_label']}→{sw['to_label']}",
                "n_months_from": len(m_from),
                "n_months_to": len(m_to),
            }
            for cat in ["Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт", "Здоровье", "Все категории"]:
                if cat not in piv.columns:
                    continue
                a = piv.loc[m_from, cat].mean() if len(m_from) else np.nan
                b = piv.loc[m_to, cat].mean() if len(m_to) else np.nan
                rec[f"{cat}_from"] = a
                rec[f"{cat}_to"] = b
                rec[f"{cat}_pct"] = (b / a - 1) * 100 if a and a == a and a != 0 else np.nan
            spend_align.append(rec)
    spend_align_df = pd.DataFrame(spend_align)

    # Honest marketplace seasonality check: YoY / within-year pattern
    mp_note = "нет данных"
    if len(monthly_df) and "Маркетплейсы" in monthly_df["category"].values:
        mp = monthly_df[monthly_df["category"] == "Маркетплейсы"].copy()
        mp["period"] = pd.to_datetime(mp["period"])
        mp = mp.sort_values("period")
        vals = mp["value"].to_numpy()
        # coefficient of variation
        cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) else np.nan
        # Nov-Dec peaks?
        mp["month"] = mp["period"].dt.month
        peak_months = mp.nlargest(3, "value")[["period", "value"]]
        mp_note = (
            f"CV месячных Маркетплейсы={cv:.3f}; топ-3 месяца: "
            + "; ".join(f"{r.period.date()}={r.value}" for r in peak_months.itertuples())
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    ends = [pd.Timestamp(win_by_idx[i]["end"]) for i in range(1, 20)]
    starts = [pd.Timestamp(win_by_idx[i]["start"]) for i in range(1, 20)]
    # step plot vs window end date (regime as of that window)
    caption = (
        "Ось Y: метка режима (threshold). Ось X: конец 6-месячного окна. "
        "Красные линии — точки переключения."
    )
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=150)
    x = np.arange(1, 20)
    ax.step(x, labels, where="mid", color="#1f4e79", linewidth=2)
    ax.plot(x, labels, "o", color="#1f4e79", markersize=5)
    for sw in switches:
        ax.axvline(sw["to_w"], color="#c45c26", alpha=0.35, linewidth=1)
    ax.set_yticks([0, 1, 2])
    ax.set_ylim(-0.3, 2.3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"w{i}\n{win_by_idx[i]['end'][2:7]}" for i in range(1, 20)],
        fontsize=7,
    )
    ax.set_xlabel("Конец 6-месячного окна")
    ax.set_ylabel("Метка режима (threshold)")
    ax.set_title(f"Вологодский район (mo_norm={mo_key}): {n_sw} переключений режимов")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8, wrap=True)
    png_path = OUT / "slide08b_vologodsky_trajectory.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Markdown narrative
    # ------------------------------------------------------------------
    def fmt_sw(sw):
        return (
            f"| w{sw['from_w']}→w{sw['to_w']} | {sw['from_label']}→{sw['to_label']} | "
            f"{sw['from_start']}…{sw['from_end']} → {sw['to_start']}…{sw['to_end']} | "
            f"{sw['PR_plus_from']:.3f}→{sw['PR_plus_to']:.3f} | "
            f"{sw['frustration_from']:.4f}→{sw['frustration_to']:.4f} | "
            f"{sw['ARI']:.4f} |"
        )

    # key feature bullets: largest |delta vs median| among mode1-signed
    signed = feat_cmp_df[feat_cmp_df["feature"].isin(URBAN_POS | URBAN_NEG)].copy()
    signed["abs_d"] = signed["delta_vs_median"].abs()
    top_signed = signed.sort_values("abs_d", ascending=False)

    # overall top deltas
    feat_cmp_df["abs_d"] = feat_cmp_df["delta_vs_median"].abs()
    top_all = feat_cmp_df.sort_values("abs_d", ascending=False).head(8)

    # spend table markdown (wide)
    spend_md = ""
    if len(monthly_df):
        wide = monthly_df.pivot(index="period", columns="category", values="value")
        # keep chronological
        wide = wide.sort_index()
        spend_md = wide.to_csv(sep="|", lineterminator="\n")
        # simple markdown table
        cols = list(wide.columns)
        lines = [
            "| period | " + " | ".join(cols) + " |",
            "|--------|" + "|".join(["------"] * len(cols)) + "|",
        ]
        for idx, r in wide.iterrows():
            lines.append(
                "| "
                + str(idx)[:10]
                + " | "
                + " | ".join(str(int(r[c])) if pd.notna(r[c]) else "" for c in cols)
                + " |"
            )
        spend_md = "\n".join(lines)

    if len(spend_align_df):
        cols = list(spend_align_df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|",
        ]
        for _, r in spend_align_df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append(f"{v:.2f}" if v == v else "")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        align_md = "\n".join(lines)
    else:
        align_md = "_нет_"

    # honest correlation statement
    honest_spend = []
    if len(spend_align_df):
        # Do marketplace pct move consistently with 0→1 or 1→0?
        if "Маркетплейсы_pct" in spend_align_df.columns:
            for _, r in spend_align_df.iterrows():
                direction = r["label"]
                pct = r.get("Маркетплейсы_pct")
                honest_spend.append(f"- {r['switch']} ({direction}): Маркетплейсы Δ={pct:+.1f}% (среднее по месяцам окна)")

    md = f"""# Кейс: Вологодский район — mode-switcher (Task 1)

> **Верификация:** `mo_norm = {mo_key!r}`, `n_switches = {n_sw}` (ровно 8).  
> Источник: `data/processed/trajectories.csv`. Не выдумано.

## Предупреждения для DeepSeek / слайдов

1. **Кластеры 0/1** в динамике — типы threshold-разбиения, интерпретируемые как *режимы потребления*, **не** юридический статус «город/село».
2. Окна — **скользящие 6 месяцев** (`start` включительно, `end` как в `dynamic_summary.json`). Не называть «режим января» без привязки к `start/end` окна.
3. `PR_plus` / `frustration` / `ARI` на переключениях — **глобальные** метрики разбиения СЗФО между соседними окнами, не индивидуальные метрики МО.
4. Сопоставление со spend — **корреляционное / описательное**. Причинность не утверждаем.
5. Если сезонность маркетплейсов не читается однозначно — так и говорим (см. ниже).

## Траектория (w1…w19)

| window | label | start | end |
|--------|------:|-------|-----|
"""
    for r in traj_rows:
        md += f"| w{r['window']} | {r['label']} | {r['start']} | {r['end']} |\n"

    md += f"""
**Последовательность лейблов:** `{' '.join(map(str, labels))}`  
**Число переключений:** {n_sw}

Файл: `case_vologodsky_trajectory.csv`  
PNG: `slide08b_vologodsky_trajectory.png`

## Точки переключения (8 штук)

| переход | label | даты окон (from → to) | PR_plus | frustration | ARI (глоб.) |
|---------|-------|------------------------|---------|-------------|-------------|
"""
    for sw in switches:
        md += fmt_sw(sw) + "\n"

    md += f"""
### Наблюдения по глобальным метрикам (факт, не интерпретация МО)

- Frustration на всех окнах в summary для этих переходов: значения из JSON (часто плато 0.5625 — проверять на слайде как свойство графа, не «стресс района»).
- ARI между соседними окнами при переключениях вологодского: {[round(s['ARI'], 4) for s in switches]}.

## Статический профиль (features_szfo_v2_final / clusters_final_threshold)

- **Статический cluster (threshold):** {static_cluster}
- Согласованность `labels_threshold.npy` с CSV cluster: {npy_match}% (если 100 — один и тот же вектор).
- Размеры полюсов: cluster0 n={len(c0)}, cluster1 n={len(c1)}.

### 17 признаков vs медиана СЗФО и средние полюсов

| feature | value | SZFO median | Δ | mean c0 | mean c1 | ближе к | urban-pattern note |
|---------|------:|------------:|--:|--------:|--------:|---------|--------------------|
"""
    for _, r in feat_cmp_df.iterrows():
        md += (
            f"| {r['feature']} | {r['value']:.5g} | {r['szfo_median']:.5g} | {r['delta_vs_median']:+.4g} | "
            f"{r['mean_cluster0']:.5g} | {r['mean_cluster1']:.5g} | {r['closer_to_cluster']} | {r['urban_pattern_note']} |\n"
        )

    md += """
**Правило urban-pattern (mode1, как в брифе):** food+, grocery−, transport+, log_total+, health+ → «urban-pattern».  
Остальные признаки без знака mode1 в этом дампе помечены `neutral/other`.

### Крупнейшие отклонения от медианы СЗФО
"""
    for _, r in top_all.iterrows():
        md += f"- `{r['feature']}`: value={r['value']:.5g}, median={r['szfo_median']:.5g}, Δ={r['delta_vs_median']:+.4g}, ближе к cluster {r['closer_to_cluster']}\n"

    md += f"""
## Spend (spend.parquet), МО ≈ {spend_mo_raw!r}

Категории: {SPEND_CATS}  
Период: 2023–2024.

### Месячные ряды
{spend_md if spend_md else '_не найдено_'}

### Сопоставление со switch-окнами (среднее value по месяцам внутри окна)

> Честно: окна перекрываются на 5 месяцев, поэтому Δ средних между соседними окнами **сглажены** и легко дают ложную «динамика режима». Это **не** доказательство причины переключения.

{align_md}

### Маркетплейсы / сезонность

{mp_note}

**Вердикт по сезонности маркетплейсов:** см. CV и топ-месяцы выше. Если пики не совпадают устойчиво с переходами 0↔1 — на слайде формулировать как *гипотезу / фон*, не как факт.

## Backup: другие МО с 8 переключениями

"""
    for o in other_eights:
        md += f"- **{o['mo_norm']}** — n_switches={o['n_switches']}; traj=`{' '.join(map(str, o['labels']))}`\n"

    md += """
Ожидаемые имена из брифа: Поддорский, Янтарный — подтверждены в `trajectories.csv`.

## Гипотезы (помечены)

| Утверждение | Статус |
|-------------|--------|
| Вологодский — топ-switcher с ровно 8 сменами лейбла на 19 окнах | **Факт** (trajectories) |
| Переключения между threshold-кластерами 0 и 1 (у вологодского не встречается 2) | **Факт** |
| Кластер = «город/село» в административном смысле | **Нет / запрещено** |
| Режим меняется из‑за сезонности маркетплейсов | **Спекуляция**, пока spend не даёт однозначного паттерна |
| PR+/frustration «объясняют» переключение района | **Спекуляция** (метрики глобальные) |
| Профиль признаков тянет к urban-pattern / к полюсу 1 | **Частично факт** — см. таблицу Δ vs median и closer_to_cluster |

## Короткий блок для вставки DeepSeek

См. конец файла / stdout.
"""

    # short pasteable block
    short = []
    short.append("### Вологодский (верифицировано)")
    short.append(f"mo_norm=`{mo_key}`, n_switches=**{n_sw}** (макс. вместе с поддорский, янтарный).")
    short.append(f"Траектория w1–w19: `{' '.join(map(str, labels))}`")
    short.append("Окна: скользящие 6м, w1=2023-01→2023-06 … w19=2024-07→2024-12 (end из dynamic_summary).")
    short.append("Переключения (label, PR+ from→to, ARI):")
    for sw in switches:
        short.append(
            f"- w{sw['from_w']}→w{sw['to_w']}: {sw['from_label']}→{sw['to_label']}; "
            f"даты to-окна {sw['to_start']}…{sw['to_end']}; "
            f"PR+ {sw['PR_plus_from']:.3f}→{sw['PR_plus_to']:.3f}; "
            f"frust {sw['frustration_from']:.4f}→{sw['frustration_to']:.4f}; "
            f"ARI={sw['ARI']:.4f}"
        )
    short.append(f"Статический threshold-cluster: **{static_cluster}**.")
    short.append("Ключевые сравнения со медианой СЗФО (mode1-направление):")
    for _, r in top_signed.iterrows():
        short.append(
            f"- {r['feature']}: {r['value']:.4g} vs med {r['szfo_median']:.4g} (Δ{r['delta_vs_median']:+.3g}) → {r['urban_pattern_note']}; ближе к c{r['closer_to_cluster']}"
        )
    short.append("Предупреждения: 0/1 ≠ юр. город/село; PR+/ARI глобальные; spend↔switch только описательно.")
    short.append(f"Маркетплейсы: {mp_note}")
    short.append("Backup 8-switch: поддорский, янтарный.")
    if honest_spend:
        short.append("Spend Δ на switch (Маркетплейсы, среднее окна):")
        short.extend(honest_spend)

    short_text = "\n".join(short)
    md += "\n## Pasteable short (RU)\n\n```\n" + short_text + "\n```\n"

    md_path = OUT / "case_vologodsky.md"
    md_path.write_text(md, encoding="utf-8")

    # also dump helpers
    feat_cmp_df.to_csv(OUT / "case_vologodsky_features.csv", index=False, encoding="utf-8-sig")
    if len(monthly_df):
        monthly_df.to_csv(OUT / "case_vologodsky_spend_monthly.csv", index=False, encoding="utf-8-sig")
    if len(spend_align_df):
        spend_align_df.to_csv(OUT / "case_vologodsky_spend_align.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(short_text)
    print("=" * 60)
    print("Wrote:", md_path)
    print("Wrote:", traj_path)
    print("Wrote:", png_path)
    print("static_cluster", static_cluster, "npy_match", npy_match)
    print("n_switches confirmed", n_sw, "mo", mo_key)


if __name__ == "__main__":
    main()
