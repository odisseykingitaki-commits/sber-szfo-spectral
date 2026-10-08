"""Generate one PNG per Task-2 pitch slide → docs/presentation_assets_task2/.

Run from repo root:
    python scripts/make_task2_pitch_assets.py

Reads post-leakage forecast_all_models.csv, backup pre-leakage CSVs,
changepoints_consensus.csv, forecast_foundation.csv, dynamic_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "presentation_assets_task2"
RESULTS = ROOT / "results"
BACKUP = RESULTS / "_backup_pre_leakage_fix"

C_BG = "#f7f5f1"
C_INK = "#1a2332"
C_ACCENT = "#1e3a5f"
C_TEAL = "#2a6f6a"
C_CORAL = "#c45c26"
C_MUTED = "#6b7280"
C_CARD = "#ffffff"
C_LINE = "#d4cfc4"
C_GOLD = "#b08900"
C_OK = "#2e7d4f"
C_BAD = "#a33b2b"


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
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


def _card(ax, x, y, w, h, ec=C_ACCENT, fc=C_CARD, lw=2.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    ))


def load_data():
    fc = pd.read_csv(RESULTS / "forecast_all_models.csv")
    cons = pd.read_csv(RESULTS / "changepoints_consensus.csv")
    dyn = json.loads((RESULTS / "dynamic_summary.json").read_text(encoding="utf-8"))
    foundation = pd.read_csv(RESULTS / "forecast_foundation.csv")

    old_agg = pd.read_csv(BACKUP / "forecast_aggregated.csv")
    old_tr = pd.read_csv(BACKUP / "forecast_with_trends.csv")

    models = [
        "naive", "seasonal", "mean", "prophet",
        "lgbm", "lgbm_trends", "catboost", "catboost_trends",
    ]
    winners = []
    for _, row in fc.iterrows():
        best_m, best_v = None, np.inf
        for m in models:
            v = row.get(f"MAE_{m}")
            if pd.notna(v) and float(v) < best_v:
                best_v, best_m = float(v), m
        winners.append({
            "cluster": int(row["cluster"]),
            "name": str(row["cluster_name"]),
            "H": int(row["horizon"]),
            "winner": best_m,
            "MAE": best_v,
        })
    return {
        "fc": fc, "cons": cons, "dyn": dyn, "foundation": foundation,
        "old_agg": old_agg, "old_tr": old_tr,
        "models": models, "winners": winners,
    }


# ─── slides ──────────────────────────────────────────────────────────────────

def slide01_title(d):
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # full-bleed dark header band
    ax.add_patch(FancyBboxPatch(
        (0, 0.55), 1.0, 0.45, boxstyle="square,pad=0",
        facecolor=C_ACCENT, edgecolor="none",
    ))
    ax.text(
        0.5, 0.78,
        "Можно ли предсказать экономику —\nи понять, когда она меняет режим?",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color="white", linespacing=1.35,
    )
    ax.text(
        0.5, 0.60,
        "280 МО СЗФО  ·  2 режима потребления  ·  ≈24 месяца",
        ha="center", va="center", fontsize=13, color="#c8d4e4",
    )

    # two question cards
    for x, title, sub, col in [
        (0.08, "Прогноз", "что будет дальше?", C_TEAL),
        (0.52, "Changepoints", "когда структура изменилась?", C_CORAL),
    ]:
        _card(ax, x, 0.12, 0.40, 0.34, ec=col, lw=2.5)
        ax.text(x + 0.20, 0.36, title, ha="center", fontsize=15,
                fontweight="bold", color=col)
        ax.text(x + 0.20, 0.24, sub, ha="center", fontsize=12, color=C_INK)

    ax.text(
        0.5, 0.04,
        "Задача 2  ·  канон после фикса end_dt < ds",
        ha="center", fontsize=10, color=C_MUTED,
    )
    return _save(fig, "slide01_title_hook.png")


def slide02_idea(d):
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Два типа территорий по структуре расходов — затем прогноз и сдвиги",
        color=C_ACCENT, fontsize=13, pad=10, fontweight="bold",
    )

    # Self-contained grouping (no Task 1 prerequisite)
    _card(ax, 0.18, 0.70, 0.64, 0.18, ec=C_ACCENT, fc="#e8eef5", lw=2)
    ax.text(0.50, 0.84, "2 режима потребления · 140 / 140",
            ha="center", fontsize=12, fontweight="bold", color=C_ACCENT)
    ax.text(
        0.50, 0.76,
        "Тип A ↑ grocery   ·   Тип B ↑ food / transport",
        ha="center", fontsize=10, color=C_INK,
    )

    # arrows down
    ax.annotate("", xy=(0.28, 0.56), xytext=(0.42, 0.70),
                arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.8))
    ax.annotate("", xy=(0.72, 0.56), xytext=(0.58, 0.70),
                arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.8))

    # A / B
    _card(ax, 0.06, 0.26, 0.40, 0.30, ec=C_TEAL, lw=2.5)
    ax.text(0.26, 0.49, "A · ПРОГНОЗ", ha="center", fontsize=13,
            fontweight="bold", color=C_TEAL)
    ax.text(0.26, 0.36,
            "агрегаты расходов\nH = 1 / 3 / 6 / 12\nnaive · Prophet · LGBM…\n→ MAE-таблица",
            ha="center", va="center", fontsize=10, color=C_INK, linespacing=1.4)

    _card(ax, 0.54, 0.26, 0.40, 0.30, ec=C_CORAL, lw=2.5)
    ax.text(0.74, 0.49, "B · CHANGEPOINTS", ha="center", fontsize=13,
            fontweight="bold", color=C_CORAL)
    ax.text(0.74, 0.36,
            "сигнал [λ₁…λ₅, PR₊]\n9 методов ruptures\nконсенсус ≥ 50%\n→ даты сдвигов",
            ha="center", va="center", fontsize=10, color=C_INK, linespacing=1.4)

    # leakage badge
    _card(ax, 0.12, 0.05, 0.76, 0.14, ec=C_GOLD, fc="#fbf6e8", lw=2)
    ax.text(
        0.50, 0.12,
        "Все прогнозные фичи — только из прошлого  (end_dt < ds)",
        ha="center", va="center", fontsize=12, fontweight="bold", color=C_GOLD,
    )
    return _save(fig, "slide02_idea.png")


def slide03_leakage(d):
    """Before / after with concrete MAE cells from backup vs current."""
    old_agg = d["old_agg"]
    old_tr = d["old_tr"]
    fc = d["fc"]

    def _old(cl, h, col):
        return float(old_agg[(old_agg.cluster == cl) & (old_agg.horizon == h)][col].iloc[0])

    def _old_tr(cl, h, col):
        return float(old_tr[(old_tr.cluster == cl) & (old_tr.horizon == h)][col].iloc[0])

    def _new(cl, h, col):
        return float(fc[(fc.cluster == cl) & (fc.horizon == h)][col].iloc[0])

    pairs = [
        ("Городские H=3\nLGBM + Trends",
         _old_tr(1, 3, "MAE_lgbm_trends"), _new(1, 3, "MAE_lgbm_trends"),
         "утечка завышала выгоду Trends"),
        ("Сельские H=3\nProphet",
         _old(0, 3, "MAE_prophet"), _new(0, 3, "MAE_prophet"),
         "катастрофа Prophet исправлена"),
        ("Городские H=3\nProphet",
         _old(1, 3, "MAE_prophet"), _new(1, 3, "MAE_prophet"),
         "катастрофа Prophet исправлена"),
    ]

    fig = plt.figure(figsize=(12.5, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.4], hspace=0.25)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    ax_top.set_title(
        "Утечка будущего искажала сравнение. Правило исправлено.",
        color=C_ACCENT, fontsize=14, fontweight="bold", pad=6,
    )

    _card(ax_top, 0.04, 0.15, 0.42, 0.70, ec=C_CORAL, lw=2)
    ax_top.text(0.25, 0.65, "БЫЛО", ha="center", fontsize=12,
                fontweight="bold", color=C_CORAL)
    ax_top.text(0.25, 0.38, "start_dt ≤ ds\nокна «видели» будущее",
                ha="center", va="center", fontsize=11, color=C_INK, linespacing=1.4)

    _card(ax_top, 0.54, 0.15, 0.42, 0.70, ec=C_TEAL, lw=2)
    ax_top.text(0.75, 0.65, "СТАЛО", ha="center", fontsize=12,
                fontweight="bold", color=C_TEAL)
    ax_top.text(0.75, 0.38, "end_dt < ds\nокно закончилось до ds",
                ha="center", va="center", fontsize=11, color=C_INK, linespacing=1.4)
    ax_top.annotate("", xy=(0.52, 0.50), xytext=(0.48, 0.50),
                    arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2.5))

    ax_bot.set_xlim(0, 1)
    ax_bot.set_ylim(0, 1)
    ax_bot.axis("off")

    for i, (label, old_v, new_v, note) in enumerate(pairs):
        x0 = 0.04 + i * 0.32
        worse = new_v > old_v * 1.05  # trends worsened; prophet "catastrophe" improved
        # for prophet, new is much better — green; for trends, worse — coral accent on new
        _card(ax_bot, x0, 0.08, 0.30, 0.84, ec=C_LINE, lw=1.5)
        ax_bot.text(x0 + 0.15, 0.82, label, ha="center", va="top",
                    fontsize=10, fontweight="bold", color=C_ACCENT, linespacing=1.25)

        ax_bot.text(x0 + 0.07, 0.55, "до", ha="center", fontsize=9, color=C_MUTED)
        ax_bot.text(x0 + 0.07, 0.42, f"{old_v:.0f}", ha="center", fontsize=16,
                    fontweight="bold", color=C_CORAL if old_v > 10000 else C_INK)

        ax_bot.annotate("", xy=(x0 + 0.20, 0.48), xytext=(x0 + 0.12, 0.48),
                        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.5))

        ax_bot.text(x0 + 0.24, 0.55, "после", ha="center", fontsize=9, color=C_MUTED)
        new_col = C_TEAL if new_v < old_v else C_CORAL
        ax_bot.text(x0 + 0.24, 0.42, f"{new_v:.0f}", ha="center", fontsize=16,
                    fontweight="bold", color=new_col)

        ax_bot.text(x0 + 0.15, 0.18, note, ha="center", fontsize=8.5,
                    color=C_MUTED, style="italic")

    return _save(fig, "slide03_leakage.png")


def slide04_mae_winners(d):
    winners = d["winners"]
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "После честного теста универсального победителя нет",
        color=C_ACCENT, fontsize=14, fontweight="bold", pad=8,
    )

    horizons = [1, 3, 6, 12]
    # header
    _card(ax, 0.04, 0.72, 0.70, 0.14, ec=C_LINE, fc="#e8eef5", lw=1)
    ax.text(0.10, 0.79, "Тип", ha="center", fontsize=11, fontweight="bold", color=C_ACCENT)
    for j, h in enumerate(horizons):
        ax.text(0.22 + j * 0.14, 0.79, f"H={h}", ha="center",
                fontsize=11, fontweight="bold", color=C_ACCENT)

    win_colors = {
        "prophet": C_ACCENT, "lgbm": C_TEAL, "naive": C_MUTED,
        "seasonal": C_GOLD, "catboost": "#5c6bc0",
    }

    for i, (cl, tname) in enumerate([(0, "A · сельские"), (1, "B · городские")]):
        y = 0.52 - i * 0.22
        _card(ax, 0.04, y, 0.70, 0.18, ec=C_LINE, lw=1.2)
        ax.text(0.10, y + 0.09, tname, ha="center", va="center",
                fontsize=10, fontweight="bold", color=C_INK)
        for j, h in enumerate(horizons):
            w = next(x for x in winners if x["cluster"] == cl and x["H"] == h)
            col = win_colors.get(w["winner"], C_INK)
            ax.text(
                0.22 + j * 0.14, y + 0.09,
                f"{w['winner']}\n{w['MAE']:.0f}",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=col, linespacing=1.25,
            )

    # side callout
    _card(ax, 0.78, 0.28, 0.20, 0.58, ec=C_CORAL, fc="#f8efe9", lw=2)
    ax.text(
        0.88, 0.57,
        "Прогноз\nзависит от\nтипа системы\nи горизонта.",
        ha="center", va="center", fontsize=11,
        fontweight="bold", color=C_CORAL, linespacing=1.45,
    )

    ax.text(
        0.39, 0.10,
        "Источник: results/forecast_all_models.csv  ·  min MAE среди доступных моделей",
        ha="center", fontsize=9, color=C_MUTED,
    )
    return _save(fig, "slide04_mae_winners.png")


def slide05_rural_h3(d):
    fc = d["fc"]
    row_a = fc[(fc["cluster"] == 0) & (fc["horizon"] == 3)].iloc[0]
    row_b = fc[(fc["cluster"] == 1) & (fc["horizon"] == 3)].iloc[0]

    fig = plt.figure(figsize=(12.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    models = ["naive", "prophet", "lgbm"]
    labels = ["Naive", "Prophet", "LGBM"]
    vals = [float(row_a[f"MAE_{m}"]) for m in models]
    colors = [C_MUTED, C_ACCENT, C_TEAL]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.65)
    bars[2].set_edgecolor(C_TEAL)
    bars[2].set_linewidth(3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 50, f"{v:.0f}",
                ha="center", fontsize=13, fontweight="bold", color=C_INK)
    ax.set_ylabel("MAE")
    ax.set_title("H=3 · Тип A (сельские)", color=C_TEAL, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.18)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    _card(ax2, 0.05, 0.55, 0.90, 0.38, ec=C_TEAL, fc="#eef6f5", lw=2)
    ax2.text(0.50, 0.82, "Локальный выигрыш ML", ha="center",
             fontsize=12, fontweight="bold", color=C_TEAL)
    ax2.text(
        0.50, 0.68,
        f"LGBM {vals[2]:.0f}  <  Prophet {vals[1]:.0f}\n"
        f"и существенно лучше Naive {vals[0]:.0f}",
        ha="center", va="center", fontsize=11, color=C_INK, linespacing=1.4,
    )

    _card(ax2, 0.05, 0.08, 0.90, 0.40, ec=C_CORAL, fc="#f8efe9", lw=2)
    ax2.text(0.50, 0.38, "На типе B при H=3 — наоборот",
             ha="center", fontsize=11, fontweight="bold", color=C_CORAL)
    ax2.text(
        0.50, 0.20,
        f"Prophet {float(row_b['MAE_prophet']):.0f}\n"
        f"vs LGBM {float(row_b['MAE_lgbm']):.0f}",
        ha="center", va="center", fontsize=12, color=C_INK, linespacing=1.35,
    )

    fig.suptitle(
        "ML выигрывает — но только там, где это действительно подтверждается",
        color=C_ACCENT, fontsize=13, fontweight="bold", y=0.98,
    )
    return _save(fig, "slide05_rural_h3.png")


def slide06_changepoints(d):
    cons = d["cons"].copy()
    cons["date_s"] = cons["date"].astype(str).str[:7]
    dyn = d["dyn"]
    windows = dyn["windows"]
    sel = [w for w in windows if w["window"] in (15, 16, 17)]
    pr_vals = [w["PR_plus"] for w in sel]

    fig = plt.figure(figsize=(12.5, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.35)
    ax_tl = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax_tl.set_xlim(0, 1)
    ax_tl.set_ylim(0, 1)
    ax_tl.axis("off")
    ax_tl.set_title(
        "Мы не доверяем одному алгоритму — ищем консенсус",
        color=C_ACCENT, fontsize=14, fontweight="bold", pad=4,
    )

    # method funnel badges
    for x, t, col in [
        (0.12, "9 методов", C_MUTED),
        (0.40, "5/9  сигнал", C_TEAL),
        (0.72, "7/9  сильный", C_CORAL),
    ]:
        _card(ax_tl, x - 0.10, 0.72, 0.22, 0.18, ec=col, lw=2)
        ax_tl.text(x + 0.01, 0.81, t, ha="center", fontsize=11,
                   fontweight="bold", color=col)

    # timeline
    events = [
        ("2023-11", "5/9", C_TEAL, False),
        ("2024-04", "5/9", C_TEAL, False),
        ("2024-09", "7/9", C_CORAL, True),
    ]
    ax_tl.plot([0.12, 0.88], [0.35, 0.35], color=C_LINE, lw=4, solid_capstyle="round")
    xs = [0.20, 0.50, 0.80]
    for x, (date, votes, col, hot) in zip(xs, events):
        ax_tl.plot(x, 0.35, "o", markersize=22 if hot else 16, color=col, zorder=5)
        if hot:
            ax_tl.text(x, 0.35, "!", ha="center", va="center",
                       fontsize=12, color="white", fontweight="bold", zorder=6)
        ax_tl.text(x, 0.55, date, ha="center", fontsize=13,
                   fontweight="bold", color=col)
        ax_tl.text(x, 0.18, votes, ha="center", fontsize=12, color=C_INK)
    ax_tl.text(
        0.80, 0.05,
        "2024-09 — самая согласованная точка",
        ha="center", fontsize=10, color=C_CORAL, fontweight="bold",
    )

    # consensus bar + PR
    cons_plot = cons.sort_values("count", ascending=True)
    colors = [C_CORAL if (r.consensus and r["count"] >= 7)
              else (C_TEAL if r.consensus else C_MUTED)
              for _, r in cons_plot.iterrows()]
    ax_bar.barh(cons_plot["date_s"], cons_plot["count"], color=colors, edgecolor="white")
    ax_bar.axvline(4.5, color=C_GOLD, ls="--", lw=1.4, label="порог 50% (4.5/9)")
    ax_bar.set_xlabel("число методов (из 9)")
    ax_bar.set_title(
        f"PR₊ вокруг 2024-09:  {pr_vals[0]:.3f} → {pr_vals[1]:.3f} → {pr_vals[2]:.3f}",
        color=C_ACCENT, fontsize=11,
    )
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.legend(frameon=False, loc="lower right")
    for y, (_, r) in enumerate(cons_plot.iterrows()):
        ax_bar.text(r["count"] + 0.12, y, f"{int(r['count'])}/9",
                    va="center", fontsize=9)
    return _save(fig, "slide06_changepoints.png")


def slide07_external(d):
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Внешний сигнал: что удалось, а что нет",
        color=C_ACCENT, fontsize=14, fontweight="bold", pad=8,
    )

    # Trends card
    _card(ax, 0.05, 0.28, 0.55, 0.58, ec=C_TEAL, lw=2.5)
    ax.text(0.325, 0.78, "Google Trends", ha="center",
            fontsize=14, fontweight="bold", color=C_TEAL)
    items = [
        ("[OK]", "выровнен по дате с расходами", C_OK),
        ("[OK]", "без заглядывания вперёд", C_OK),
        ("[!]", "один ряд на оба режима потребления", C_GOLD),
        ("[!]", "часто ухудшает MAE после фикса", C_GOLD),
    ]
    for i, (mark, text, col) in enumerate(items):
        y = 0.65 - i * 0.09
        ax.text(0.12, y, mark, ha="center", fontsize=11,
                fontweight="bold", color=col)
        ax.text(0.18, y, text, ha="left", va="center", fontsize=11, color=C_INK)

    # News NLP
    _card(ax, 0.64, 0.52, 0.31, 0.34, ec=C_BAD, lw=2.5)
    ax.text(0.795, 0.76, "News NLP", ha="center",
            fontsize=13, fontweight="bold", color=C_BAD)
    ax.text(0.795, 0.62, "[нет]  не реализован",
            ha="center", fontsize=12, color=C_INK)

    # GDELT
    _card(ax, 0.64, 0.28, 0.31, 0.20, ec=C_MUTED, lw=1.5)
    ax.text(0.795, 0.42, "GDELT events", ha="center",
            fontsize=11, fontweight="bold", color=C_MUTED)
    ax.text(0.795, 0.33, "без систематического win",
            ha="center", fontsize=9.5, color=C_INK)

    # bottom line
    _card(ax, 0.10, 0.06, 0.80, 0.14, ec=C_ACCENT, fc="#e8eef5", lw=2)
    ax.text(
        0.50, 0.13,
        "Мы не выдаём Trends за новости.",
        ha="center", va="center", fontsize=14, fontweight="bold", color=C_ACCENT,
    )
    return _save(fig, "slide07_external.png")


def slide08_chronos(d):
    fnd = d["foundation"]
    winners = d["winners"]

    # Chronos wins: 0/8
    chronos_wins = 0
    rows = []
    for _, row in fnd.iterrows():
        cl, h = int(row["cluster"]), int(row["horizon"])
        mae_c = float(row["MAE_chronos"])
        w = next(x for x in winners if x["cluster"] == cl and x["H"] == h)
        best = w["MAE"]
        best_name = w["winner"]
        if mae_c < best:
            chronos_wins += 1
        name = "сельские" if cl == 0 else "городские"
        rows.append((f"{name} H={h}", mae_c, best, best_name))

    fig = plt.figure(figsize=(12.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2], wspace=0.25)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.axis("off")
    _card(ax_l, 0.05, 0.15, 0.90, 0.70, ec=C_CORAL, fc="#f8efe9", lw=2.5)
    ax_l.text(0.50, 0.72, "Chronos-bolt-tiny", ha="center",
              fontsize=14, fontweight="bold", color=C_CORAL)
    ax_l.text(0.50, 0.55, "0 / 8", ha="center",
              fontsize=36, fontweight="bold", color=C_CORAL)
    ax_l.text(0.50, 0.38, "горизонтов выиграны",
              ha="center", fontsize=12, color=C_INK)
    ax_l.text(
        0.50, 0.22,
        "ряд ≈24 точки\nслишком короткий\nдля zero-shot foundation",
        ha="center", fontsize=10, color=C_MUTED, linespacing=1.35,
    )

    # comparison bars for a few key cells
    show = [r for r in rows if r[0] in ("сельские H=3", "городские H=3", "городские H=12", "сельские H=12")]
    y = np.arange(len(show))
    chronos_v = [r[1] for r in show]
    classic_v = [r[2] for r in show]
    labels = [f"{r[0]}\n({r[3]})" for r in show]
    h = 0.35
    ax_r.barh(y + h / 2, chronos_v, h, label="Chronos", color=C_CORAL, alpha=0.85)
    ax_r.barh(y - h / 2, classic_v, h, label="лучший классический", color=C_TEAL)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels(labels, fontsize=9)
    ax_r.set_xlabel("MAE")
    ax_r.set_title("Chronos  vs  лучший классический", color=C_ACCENT, fontsize=12)
    ax_r.legend(frameon=False, loc="lower right", fontsize=9)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    for yi, cv, bv in zip(y, chronos_v, classic_v):
        ax_r.text(cv + 80, yi + h / 2, f"{cv:.0f}", va="center", fontsize=8, color=C_CORAL)
        ax_r.text(bv + 80, yi - h / 2, f"{bv:.0f}", va="center", fontsize=8, color=C_TEAL)

    fig.suptitle(
        "Foundation-модель не дала преимущества — и это тоже результат",
        color=C_ACCENT, fontsize=13, fontweight="bold", y=0.98,
    )
    return _save(fig, "slide08_chronos.png")


def slide09_final(d):
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.50, 0.90,
        "Что мы узнали о системе",
        ha="center", fontsize=16, fontweight="bold", color=C_ACCENT,
    )

    points = [
        ("01", "Нет единого победителя прогноза"),
        ("02", "Спектральный сигнал для сдвигов\nнезависим от конкретной модели прогноза"),
        ("03", "Три точки ≥50%; сильнейшая — сентябрь 2024"),
    ]
    for i, (num, text) in enumerate(points):
        y = 0.72 - i * 0.14
        ax.add_patch(Circle((0.12, y), 0.035, facecolor=C_ACCENT, edgecolor="none"))
        ax.text(0.12, y, num, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(0.20, y, text, ha="left", va="center",
                fontsize=13, color=C_INK, linespacing=1.3)

    # huge closing line
    _card(ax, 0.08, 0.08, 0.84, 0.28, ec=C_ACCENT, fc=C_ACCENT, lw=0)
    ax.text(
        0.50, 0.22,
        "Не всегда важно точно предсказать число.\n"
        "Иногда важнее вовремя заметить,\n"
        "что система стала другой.",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color="white", linespacing=1.4,
    )
    return _save(fig, "slide09_final.png")


def main():
    _style()
    print("Loading Task 2 artifacts…")
    d = load_data()
    print("Rendering slides…")
    slide01_title(d)
    slide02_idea(d)
    slide03_leakage(d)
    slide04_mae_winners(d)
    slide05_rural_h3(d)
    slide06_changepoints(d)
    slide07_external(d)
    slide08_chronos(d)
    slide09_final(d)
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
