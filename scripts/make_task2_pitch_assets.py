"""Generate one PNG per Task-2 pitch slide → docs/presentation_assets_task2/.

Run from repo root:
    python scripts/make_task2_pitch_assets.py

Reads post-leakage forecast_all_models.csv, changepoints_consensus.csv,
dynamic_summary.json — no model re-training.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "presentation_assets_task2"
RESULTS = ROOT / "results"

C_BG = "#f7f5f1"
C_INK = "#1a2332"
C_ACCENT = "#1e3a5f"
C_TEAL = "#2a6f6a"
C_CORAL = "#c45c26"
C_MUTED = "#6b7280"
C_CARD = "#ffffff"
C_LINE = "#d4cfc4"
C_GOLD = "#b08900"


def _style():
    plt.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor": C_BG,
        "axes.edgecolor": C_LINE,
        "axes.labelcolor": C_INK,
        "text.color": C_INK,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 140,
    })


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  → {path.name}")
    return path


def load_data():
    fc = pd.read_csv(RESULTS / "forecast_all_models.csv")
    cons = pd.read_csv(RESULTS / "changepoints_consensus.csv")
    dyn = json.loads((RESULTS / "dynamic_summary.json").read_text(encoding="utf-8"))
    models = [
        "naive", "seasonal", "mean", "prophet",
        "lgbm", "lgbm_trends", "catboost", "catboost_trends",
    ]
    mae_cols = [f"MAE_{m}" for m in models]
    winners = []
    for _, row in fc.iterrows():
        best_m, best_v = None, np.inf
        for m in models:
            v = row.get(f"MAE_{m}")
            if pd.notna(v) and v < best_v:
                best_v, best_m = float(v), m
        winners.append({
            "cluster": int(row["cluster"]),
            "name": row["cluster_name"],
            "H": int(row["horizon"]),
            "winner": best_m,
            "MAE": best_v,
        })
    return {
        "fc": fc, "cons": cons, "dyn": dyn,
        "models": models, "mae_cols": mae_cols, "winners": winners,
    }


# ─── slides ──────────────────────────────────────────────────────────────────

def slide01_title(d):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.05, 0.12), 0.9, 0.76,
                                boxstyle="round,pad=0.02", facecolor=C_CARD,
                                edgecolor=C_ACCENT, linewidth=2))
    ax.text(0.5, 0.72, "Спектральная синергетика\nэкономических систем",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=C_ACCENT, linespacing=1.35)
    ax.text(0.5, 0.50, "Задача 2 · прогноз + структурные сдвиги СЗФО",
            ha="center", fontsize=14, color=C_INK)
    chips = [
        (0.20, "H = 1,3,6,12"),
        (0.40, "MAE канон"),
        (0.60, "9 CP-методов"),
        (0.80, "post-leakage"),
    ]
    for x, t in chips:
        ax.add_patch(FancyBboxPatch((x - 0.09, 0.28), 0.18, 0.1,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#e8eef5", edgecolor=C_LINE))
        ax.text(x, 0.33, t, ha="center", va="center", fontsize=10, color=C_ACCENT)
    ax.text(0.5, 0.18, "СберИндекс · кластеры Task 1 · репозиторий: локально",
            ha="center", fontsize=10, color=C_MUTED)
    return _save(fig, "slide01_title_card.png")


def slide02_two_levels(d):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Два уровня одной экономики СЗФО", color=C_ACCENT, pad=8)
    boxes = [
        (0.08, 0.35, 0.38, 0.45, C_TEAL,
         "A · ПРОГНОЗ",
         "кластер × месяц\nгоризонты 1/3/6/12\nnaive · Prophet · LGBM…\n→ MAE-таблица"),
        (0.54, 0.35, 0.38, 0.45, C_CORAL,
         "B · CHANGEPOINTS",
         "сигнал [λ1..λ5, PR₊]\n9 методов ruptures\nконсенсус ≥ 50%\n→ 2023-11 / 2024-04 / 2024-09"),
    ]
    for x, y, w, h, col, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=C_CARD, edgecolor=col, linewidth=2.5))
        ax.text(x + w / 2, y + h - 0.08, title, ha="center", fontsize=13,
                fontweight="bold", color=col)
        ax.text(x + w / 2, y + 0.18, body, ha="center", va="center",
                fontsize=11, color=C_INK, linespacing=1.45)
    ax.text(0.5, 0.18, "Вход: метки Task 1 + dynamic_summary + spend",
            ha="center", fontsize=11, color=C_MUTED)
    return _save(fig, "slide02_two_levels.png")


def slide03_architecture(d):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Архитектура Task 2", color=C_ACCENT)

    def box(x, y, w, h, text, fc=C_CARD, ec=C_ACCENT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=fc, edgecolor=ec, linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, color=C_INK, linespacing=1.3)

    box(3.5, 5.5, 5, 1.0, "Task 1: labels 0/1 + окна (λ, PR₊)", fc="#e8eef5")
    box(0.5, 2.8, 5.0, 1.8,
        "12_timeseries.py\nagg spend · end_dt < ds\n± Trends → MAE",
        ec=C_TEAL)
    box(6.5, 2.8, 5.0, 1.8,
        "13_changepoints.py\n[λ1..λ5, PR₊]\n9 методов → консенсус",
        ec=C_CORAL)
    box(0.5, 0.6, 5.0, 1.2, "forecast_all_models.csv", ec=C_TEAL, fc="#eef6f5")
    box(6.5, 0.6, 5.0, 1.2, "changepoints_*.csv", ec=C_CORAL, fc="#f8efe9")

    for x0, x1 in [(6.0, 3.0), (6.0, 9.0)]:
        ax.annotate("", xy=(x1, 4.6), xytext=(x0, 5.5),
                    arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.5))
    ax.annotate("", xy=(3.0, 1.8), xytext=(3.0, 2.8),
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.5))
    ax.annotate("", xy=(9.0, 1.8), xytext=(9.0, 2.8),
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.5))
    return _save(fig, "slide03_architecture.png")


def slide04_leakage(d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, title, body, col in [
        (axes[0], "ДО (баг)", "start_dt ≤ ds\nокна могли\n«видеть» будущее\nMAE искажены", C_CORAL),
        (axes[1], "ПОСЛЕ (канон)", "end_dt < ds\nокно закончилось\nдо даты прогноза\nчестное сравнение", C_TEAL),
    ]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.08, 0.15), 0.84, 0.7,
                                    boxstyle="round,pad=0.03",
                                    facecolor=C_CARD, edgecolor=col, linewidth=2.5))
        ax.text(0.5, 0.72, title, ha="center", fontsize=14, fontweight="bold", color=col)
        ax.text(0.5, 0.42, body, ha="center", va="center", fontsize=12,
                color=C_INK, linespacing=1.5)
    fig.suptitle("Leakage fix: спектральные фичи без будущего", color=C_ACCENT, fontsize=14)
    return _save(fig, "slide04_leakage_note.png")


def slide05_mae_heatmap(d):
    fc = d["fc"]
    models = ["naive", "seasonal", "prophet", "lgbm", "lgbm_trends", "catboost"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, cl, title in [(axes[0], 0, "Сельские"), (axes[1], 1, "Городские")]:
        sub = fc[fc["cluster"] == cl].sort_values("horizon")
        mat = []
        for _, row in sub.iterrows():
            mat.append([row.get(f"MAE_{m}", np.nan) for m in models])
        mat = np.array(mat, dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"H={int(h)}" for h in sub["horizon"]], fontsize=10)
        ax.set_title(title, color=C_ACCENT)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt = "—" if np.isnan(v) else f"{v:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black" if np.isnan(v) or v < np.nanmax(mat) * 0.65 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("MAE по моделям × горизонтам (post-leakage)", color=C_ACCENT)
    fig.tight_layout()
    return _save(fig, "slide05_mae_heatmap.png")


def slide06_winner_map(d):
    winners = d["winners"]
    # grid: rows = clusters, cols = horizons
    Hs = [1, 3, 6, 12]
    names = {0: "сельские", 1: "городские"}
    color_map = {
        "prophet": C_ACCENT, "lgbm": C_TEAL, "naive": C_CORAL,
        "seasonal": C_GOLD, "mean": C_MUTED, "catboost": "#5c6bc0",
        "lgbm_trends": "#4db6ac", "catboost_trends": "#7986cb",
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"H={h}" for h in Hs])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["сельские", "городские"])
    ax.set_title("Карта победителей (min MAE)", color=C_ACCENT)
    for w in winners:
        xi = Hs.index(w["H"])
        yi = w["cluster"]
        col = color_map.get(w["winner"], C_MUTED)
        ax.add_patch(FancyBboxPatch((xi - 0.4, yi - 0.35), 0.8, 0.7,
                                    boxstyle="round,pad=0.02",
                                    facecolor=col, edgecolor="white", linewidth=1,
                                    alpha=0.9))
        ax.text(xi, yi + 0.05, w["winner"], ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(xi, yi - 0.18, f"{w['MAE']:.0f}", ha="center", va="center",
                fontsize=9, color="white")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # legend
    handles = [mpatches.Patch(color=c, label=m) for m, c in
               [("prophet", C_ACCENT), ("lgbm", C_TEAL), ("naive", C_CORAL), ("seasonal", C_GOLD)]]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    return _save(fig, "slide06_winner_map.png")


def slide07_rural_h3(d):
    fc = d["fc"]
    row = fc[(fc["cluster"] == 0) & (fc["horizon"] == 3)].iloc[0]
    models = ["naive", "seasonal", "prophet", "lgbm", "lgbm_trends", "catboost"]
    vals = [row[f"MAE_{m}"] for m in models]
    colors = [C_MUTED, C_GOLD, C_ACCENT, C_TEAL, "#4db6ac", "#5c6bc0"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.7)
    # highlight winner
    bars[3].set_edgecolor(C_TEAL)
    bars[3].set_linewidth(2.5)
    ax.set_ylabel("MAE")
    ax.set_title("Кейс: сельские · H=3 — LGBM vs Prophet", color=C_ACCENT)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:.0f}",
                ha="center", fontsize=10, color=C_INK)
    ax.axhline(row["MAE_prophet"], color=C_ACCENT, ls="--", alpha=0.5, label="Prophet")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.95, "LGBM 1761 < Prophet 1839\nTrends не помогают",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=C_TEAL, linespacing=1.3)
    return _save(fig, "slide07_rural_h3_case.png")


def slide08_consensus(d):
    cons = d["cons"].sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = [C_TEAL if bool(r.consensus) else C_MUTED for _, r in cons.iterrows()]
    ax.barh(cons["date"].astype(str), cons["count"], color=colors, edgecolor="white")
    ax.axvline(4.5, color=C_CORAL, ls="--", lw=1.5, label="порог 50% (4.5/9)")
    ax.set_xlabel("число методов (из 9)")
    ax.set_title("Консенсус changepoints ≥ 50%", color=C_ACCENT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for y, (_, r) in enumerate(cons.iterrows()):
        ax.text(r["count"] + 0.1, y, f"{r['count']}/9", va="center", fontsize=10)
    ax.legend(frameon=False)
    return _save(fig, "slide08_changepoint_consensus.png")


def slide09_cp_example(d):
    # PR+ / Frustration around 2024-09 consensus
    windows = d["dyn"]["windows"]
    sel = [w for w in windows if w["window"] in range(14, 19)]
    xs = [w["end"][:7] for w in sel]
    pr = [w["PR_plus"] for w in sel]
    fr = [w["frustration"] for w in sel]
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(xs, pr, "o-", color=C_TEAL, lw=2, markersize=8, label="PR₊")
    ax2.plot(xs, fr, "s--", color=C_CORAL, lw=1.5, markersize=7, label="Frustration")
    # mark consensus date
    if "2024-09" in xs:
        i = xs.index("2024-09")
        ax1.axvline(i, color=C_ACCENT, ls=":", lw=2, alpha=0.8)
        ax1.annotate("консенсус 7/9", xy=(i, pr[i]), xytext=(i + 0.3, pr[i] + 0.15),
                     fontsize=10, color=C_ACCENT,
                     arrowprops=dict(arrowstyle="->", color=C_ACCENT))
    ax1.set_ylabel("PR₊", color=C_TEAL)
    ax2.set_ylabel("Frustration", color=C_CORAL)
    ax1.set_title("Вокруг сдвига 2024-09 (dynamic_summary)", color=C_ACCENT)
    ax1.spines["top"].set_visible(False)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower left", frameon=False)
    ax1.text(0.02, 0.98,
             "Структурный сигнал спектра, не новостной заголовок",
             transform=ax1.transAxes, va="top", fontsize=9, color=C_MUTED)
    return _save(fig, "slide09_cp_example.png")


def slide10_trends(d):
    fc = d["fc"]
    # compare lgbm vs lgbm_trends where both exist
    pairs = []
    for _, row in fc.iterrows():
        a, b = row.get("MAE_lgbm"), row.get("MAE_lgbm_trends")
        if pd.notna(a) and pd.notna(b):
            pairs.append((f"{row['cluster_name'][:4]} H{int(row['horizon'])}", a, b))
    labels = [p[0] for p in pairs]
    no_t = [p[1] for p in pairs]
    with_t = [p[2] for p in pairs]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, no_t, w, label="lgbm", color=C_TEAL)
    ax.bar(x + w / 2, with_t, w, label="lgbm_trends", color=C_CORAL, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE")
    ax.set_title("Trends — proxy, не news: часто хуже после leakage-fix", color=C_ACCENT)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.95, "News NLP не строили\n→ критерий 15% = gap",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=C_MUTED)
    return _save(fig, "slide10_trends_not_news.png")


def slide11_gaps(d):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Пробелы критериев (честно)", color=C_ACCENT, pad=12)
    items = [
        (0.08, 0.55, "Foundation TS (15%)",
         "Chronos / TimesFM / MOIRAI\nне использовали\nряд ~24 точки\n«преимущество не закрыто»", C_CORAL),
        (0.52, 0.55, "News (15%)",
         "Есть Google Trends\n(прокси интереса)\nНет NLP новостей\npipeline = future work", C_GOLD),
        (0.08, 0.08, "Закрыто",
         "Метод · MAE · сравнение\nпрогнозов · CP-консенсус\nвоспроизводимость 12/13", C_TEAL),
        (0.52, 0.08, "Next steps",
         "zero-shot foundation\nв той же MAE-таблице\nnews: дата+лаг+no leak", C_ACCENT),
    ]
    for x, y, title, body, col in items:
        ax.add_patch(FancyBboxPatch((x, y), 0.40, 0.38,
                                    boxstyle="round,pad=0.02",
                                    facecolor=C_CARD, edgecolor=col, linewidth=2))
        ax.text(x + 0.20, y + 0.30, title, ha="center", fontsize=12,
                fontweight="bold", color=col)
        ax.text(x + 0.20, y + 0.12, body, ha="center", va="center",
                fontsize=10, color=C_INK, linespacing=1.35)
    return _save(fig, "slide11_gaps.png")


def slide12_takeaways(d):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.06, 0.1), 0.88, 0.78,
                                boxstyle="round,pad=0.02",
                                facecolor=C_CARD, edgecolor=C_ACCENT, linewidth=2))
    ax.text(0.5, 0.82, "Takeaways · Задача 2", ha="center",
            fontsize=16, fontweight="bold", color=C_ACCENT)
    points = [
        "1. Два уровня: прогноз расходов + консенсус changepoints",
        "2. Post-leakage: нет единого winner (Prophet / LGBM / naive / seasonal)",
        "3. Консенсус сдвигов: 2023-11 · 2024-04 · 2024-09",
        "4. Trends ≠ news; foundation models — документированный gap",
        "5. Воспроизведение: src/12, src/13 · configs/task2_*.yaml",
    ]
    for i, t in enumerate(points):
        ax.text(0.12, 0.68 - i * 0.11, t, ha="left", fontsize=12, color=C_INK)
    return _save(fig, "slide12_takeaways.png")


def main():
    _style()
    print("Loading Task 2 artifacts…")
    d = load_data()
    print("Rendering slides…")
    slide01_title(d)
    slide02_two_levels(d)
    slide03_architecture(d)
    slide04_leakage(d)
    slide05_mae_heatmap(d)
    slide06_winner_map(d)
    slide07_rural_h3(d)
    slide08_consensus(d)
    slide09_cp_example(d)
    slide10_trends(d)
    slide11_gaps(d)
    slide12_takeaways(d)
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
