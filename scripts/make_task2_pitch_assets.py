"""Generate one strong PNG per Task-2 pitch slide → docs/presentation_assets_task2/.

Run from repo root:
    python scripts/make_task2_pitch_assets.py

Reads post-leakage forecast_all_models.csv, changepoints_consensus.csv,
forecast_foundation.csv, dynamic_summary.json, and pre-leakage backup MAE.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "presentation_assets_task2"
RESULTS = ROOT / "results"
BACKUP = RESULTS / "_backup_pre_leakage_fix"

C_BG = "#f4f1ea"
C_INK = "#1a2332"
C_ACCENT = "#1e3a5f"
C_TEAL = "#2a6f6a"
C_CORAL = "#c45c26"
C_MUTED = "#6b7280"
C_CARD = "#ffffff"
C_LINE = "#d4cfc4"
C_GOLD = "#b08900"
C_SOFT = "#e8eef5"
C_SOFT_TEAL = "#eef6f5"
C_SOFT_CORAL = "#f8efe9"
C_RED = "#b33a3a"


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
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=C_BG, pad_inches=0.15)
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


def _card(ax, x, y, w, h, ec=C_ACCENT, fc=C_CARD, lw=2.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    ))


def load_data():
    fc = pd.read_csv(RESULTS / "forecast_all_models.csv")
    cons = pd.read_csv(RESULTS / "changepoints_consensus.csv")
    dyn = json.loads((RESULTS / "dynamic_summary.json").read_text(encoding="utf-8"))
    foundation = pd.read_csv(RESULTS / "forecast_foundation.csv")
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
            "name": row["cluster_name"],
            "H": int(row["horizon"]),
            "winner": best_m,
            "MAE": best_v,
        })

    # Key before/after cells: documented defaults + live backup/current CSV when present
    specs = [
        (1, 3, "Городские H=3\nLGBM+Trends", "trends", "MAE_lgbm_trends", 3878.0, 5356.18),
        (0, 3, "Сельские H=3\nProphet", "agg", "MAE_prophet", 22807.8, 1838.56),
        (1, 3, "Городские H=3\nProphet", "agg", "MAE_prophet", 39107.5, 2653.36),
        (0, 12, "Сельские H=12\nProphet", "agg", "MAE_prophet", 78996.7, 1514.08),
    ]
    leakage_pairs = []
    agg = tr = None
    try:
        if (BACKUP / "forecast_aggregated.csv").exists():
            agg = pd.read_csv(BACKUP / "forecast_aggregated.csv")
        if (BACKUP / "forecast_with_trends.csv").exists():
            tr = pd.read_csv(BACKUP / "forecast_with_trends.csv")
    except Exception as e:
        print(f"  (backup CSV soft-fail: {e})")

    for cl, H, name, src_kind, model_col, old_fb, new_fb in specs:
        old_v, new_v = old_fb, new_fb
        src = tr if src_kind == "trends" else agg
        if src is not None and {"cluster", "horizon", model_col}.issubset(src.columns):
            r = src[(src["cluster"] == cl) & (src["horizon"] == H)]
            if len(r) and pd.notna(r.iloc[0][model_col]):
                old_v = float(r.iloc[0][model_col])
        new_r = fc[(fc["cluster"] == cl) & (fc["horizon"] == H)]
        if len(new_r) and pd.notna(new_r.iloc[0].get(model_col)):
            new_v = float(new_r.iloc[0][model_col])
        leakage_pairs.append((name, old_v, new_v, ""))

    return {
        "fc": fc, "cons": cons, "dyn": dyn, "foundation": foundation,
        "models": models, "winners": winners, "leakage_pairs": leakage_pairs,
    }


# ─── slides ──────────────────────────────────────────────────────────────────

def slide01_title(d):
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # left accent bar
    ax.add_patch(Rectangle((0, 0), 0.018, 1, facecolor=C_ACCENT, edgecolor="none"))
    ax.text(0.06, 0.86, "ЗАДАЧА 2  ·  СЗФО", fontsize=11, color=C_MUTED,
            fontweight="bold")
    ax.text(
        0.06, 0.68,
        "Можно ли предсказать экономику —\nи понять, когда она меняет режим?",
        ha="left", va="center", fontsize=22, fontweight="bold",
        color=C_ACCENT, linespacing=1.35,
    )
    ax.text(
        0.06, 0.48,
        "280 МО СЗФО  ·  2 спектральных типа  ·  ~24 месяца",
        ha="left", fontsize=13, color=C_INK,
    )
    # two question cards
    for x, title, body, col in [
        (0.06, "Прогноз", "Что будет дальше?", C_TEAL),
        (0.52, "Changepoints", "Когда структура изменилась?", C_CORAL),
    ]:
        _card(ax, x, 0.14, 0.40, 0.24, ec=col, lw=2.2)
        ax.text(x + 0.02, 0.30, title, fontsize=11, fontweight="bold", color=col)
        ax.text(x + 0.02, 0.21, body, fontsize=14, color=C_INK)
    return _save(fig, "slide01_title_card.png")


def slide02_idea(d):
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.93, "Одна типология — два способа смотреть в будущее",
            ha="center", fontsize=16, fontweight="bold", color=C_ACCENT)

    # Task1 source
    _card(ax, 0.32, 0.68, 0.36, 0.16, ec=C_ACCENT, fc=C_SOFT, lw=2)
    ax.text(0.5, 0.78, "Типология Task 1", ha="center", fontsize=13,
            fontweight="bold", color=C_ACCENT)
    ax.text(0.5, 0.72, "тип A · сельские   |   тип B · городские",
            ha="center", fontsize=10, color=C_MUTED)

    # arrows down
    ax.annotate("", xy=(0.28, 0.58), xytext=(0.42, 0.68),
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=2))
    ax.annotate("", xy=(0.72, 0.58), xytext=(0.58, 0.68),
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=2))

    # A / B
    _card(ax, 0.06, 0.22, 0.40, 0.34, ec=C_TEAL, lw=2.5)
    ax.text(0.26, 0.48, "A  ·  FORECAST", ha="center", fontsize=13,
            fontweight="bold", color=C_TEAL)
    ax.text(0.26, 0.36,
            "Прогноз расходов\nпо типу × горизонту\nnaive · Prophet · LGBM…\n→ таблица MAE",
            ha="center", va="center", fontsize=11, color=C_INK, linespacing=1.4)

    _card(ax, 0.54, 0.22, 0.40, 0.34, ec=C_CORAL, lw=2.5)
    ax.text(0.74, 0.48, "B  ·  CHANGEPOINTS", ha="center", fontsize=13,
            fontweight="bold", color=C_CORAL)
    ax.text(0.74, 0.36,
            "Структурные сдвиги\nсигнала [λ₁…λ₅, PR₊]\n9 методов → консенсус\n→ даты режима",
            ha="center", va="center", fontsize=11, color=C_INK, linespacing=1.4)

    # badge
    _card(ax, 0.18, 0.04, 0.64, 0.12, ec=C_GOLD, fc="#fff8e8", lw=1.8)
    ax.text(0.5, 0.10,
            "Все прогнозные фичи — только из прошлого  (end_dt < ds)",
            ha="center", va="center", fontsize=12, fontweight="bold", color=C_GOLD)
    return _save(fig, "slide02_idea.png")


def slide03_architecture(d):
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(6, 6.6, "Архитектура Task 2", ha="center", fontsize=16,
            fontweight="bold", color=C_ACCENT)

    def box(x, y, w, h, text, fc=C_CARD, ec=C_ACCENT, fs=10):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.8,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=C_INK, linespacing=1.35)

    box(3.2, 5.2, 5.6, 1.0, "Task 1: labels 0/1 + скользящие окна (λ, PR₊)", fc=C_SOFT)
    box(0.4, 2.6, 5.2, 1.9,
        "Уровень A · прогноз\nagg spend · end_dt < ds\n± Trends → MAE",
        ec=C_TEAL, fc=C_SOFT_TEAL)
    box(6.4, 2.6, 5.2, 1.9,
        "Уровень B · changepoints\n[λ₁…λ₅, PR₊]\n9 методов → консенсус ≥50%",
        ec=C_CORAL, fc=C_SOFT_CORAL)
    box(0.4, 0.5, 5.2, 1.2, "forecast_all_models.csv", ec=C_TEAL, fc=C_SOFT_TEAL)
    box(6.4, 0.5, 5.2, 1.2, "changepoints_consensus.csv", ec=C_CORAL, fc=C_SOFT_CORAL)

    ax.annotate("", xy=(3.0, 4.5), xytext=(5.5, 5.2),
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.8))
    ax.annotate("", xy=(9.0, 4.5), xytext=(6.5, 5.2),
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.8))
    ax.annotate("", xy=(3.0, 1.7), xytext=(3.0, 2.6),
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.6))
    ax.annotate("", xy=(9.0, 1.7), xytext=(9.0, 2.6),
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.6))
    return _save(fig, "slide03_architecture.png")


def slide04_leakage(d):
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 2.4], hspace=0.28)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")
    ax_top.text(0.5, 0.72, "Утечка будущего искажала сравнение. Правило исправлено.",
                ha="center", fontsize=15, fontweight="bold", color=C_ACCENT)
    # before / after rule chips
    _card(ax_top, 0.04, 0.08, 0.44, 0.45, ec=C_CORAL, fc=C_SOFT_CORAL)
    ax_top.text(0.26, 0.38, "ДО", ha="center", fontsize=11, fontweight="bold", color=C_CORAL)
    ax_top.text(0.26, 0.20, "start_dt ≤ ds  →  окна видели будущее",
                ha="center", fontsize=11, color=C_INK)
    _card(ax_top, 0.52, 0.08, 0.44, 0.45, ec=C_TEAL, fc=C_SOFT_TEAL)
    ax_top.text(0.74, 0.38, "ПОСЛЕ", ha="center", fontsize=11, fontweight="bold", color=C_TEAL)
    ax_top.text(0.74, 0.20, "end_dt < ds  →  только прошлое",
                ha="center", fontsize=11, color=C_INK)

    pairs = d["leakage_pairs"]
    labels = [p[0] for p in pairs]
    olds = [p[1] for p in pairs]
    news = [p[2] for p in pairs]
    x = np.arange(len(labels))
    w = 0.36
    # log scale helps show Prophet catastrophe vs normal
    bars1 = ax.bar(x - w / 2, olds, w, label="до фикса", color=C_CORAL, alpha=0.85)
    bars2 = ax.bar(x + w / 2, news, w, label="после фикса", color=C_TEAL, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE (лог. шкала)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars1, olds):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8, color=C_CORAL)
    for b, v in zip(bars2, news):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.0f}",
                ha="center", va="bottom", fontsize=8, color=C_TEAL)
    ax.set_title("Ключевой пример: городские H=3 LGBM+Trends  3878 → 5356; Prophet-катастрофа устранена",
                 color=C_MUTED, fontsize=10, pad=8)
    return _save(fig, "slide04_leakage.png")


def slide05_mae_winners(d):
    winners = d["winners"]
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.94, "После честного теста универсального победителя нет",
            ha="center", fontsize=15, fontweight="bold", color=C_ACCENT)

    horizons = [1, 3, 6, 12]
    # header
    col_x = [0.28, 0.46, 0.64, 0.82]
    ax.text(0.08, 0.82, "Тип", fontsize=11, fontweight="bold", color=C_MUTED)
    for hx, H in zip(col_x, horizons):
        ax.text(hx, 0.82, f"H = {H}", ha="center", fontsize=12,
                fontweight="bold", color=C_ACCENT)

    rows = [
        (0, "A · сельские\n(cluster 0)", 0.52),
        (1, "B · городские\n(cluster 1)", 0.22),
    ]
    win_map = {(w["cluster"], w["H"]): w for w in winners}
    accent_cells = {(0, 3), (1, 1), (1, 3), (1, 6)}  # highlight strong story cells

    for cl, label, y0 in rows:
        _card(ax, 0.03, y0, 0.94, 0.26, ec=C_LINE, fc=C_CARD, lw=1.4)
        ax.text(0.08, y0 + 0.13, label, ha="left", va="center",
                fontsize=12, fontweight="bold", color=C_INK, linespacing=1.3)
        for hx, H in zip(col_x, horizons):
            w = win_map[(cl, H)]
            highlight = (cl, H) in accent_cells
            ec = C_TEAL if highlight else C_LINE
            fc = C_SOFT_TEAL if highlight else "#faf9f6"
            _card(ax, hx - 0.08, y0 + 0.04, 0.16, 0.18, ec=ec, fc=fc, lw=1.6 if highlight else 1.0)
            ax.text(hx, y0 + 0.15, w["winner"], ha="center", fontsize=11,
                    fontweight="bold", color=C_TEAL if highlight else C_ACCENT)
            ax.text(hx, y0 + 0.08, f"{w['MAE']:.0f}", ha="center",
                    fontsize=12, color=C_INK)

    _card(ax, 0.15, 0.04, 0.70, 0.10, ec=C_ACCENT, fc=C_SOFT, lw=1.6)
    ax.text(0.5, 0.09, "Прогноз зависит от типа системы и горизонта.",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_ACCENT)
    return _save(fig, "slide05_mae_winners.png")


def slide06_rural_h3(d):
    fc = d["fc"]
    row_a = fc[(fc["cluster"] == 0) & (fc["horizon"] == 3)].iloc[0]
    row_b = fc[(fc["cluster"] == 1) & (fc["horizon"] == 3)].iloc[0]
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    models = ["naive", "prophet", "lgbm"]
    labels = ["Naive", "Prophet", "LGBM"]
    vals = [float(row_a[f"MAE_{m}"]) for m in models]
    colors = [C_MUTED, C_ACCENT, C_TEAL]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.62)
    bars[2].set_linewidth(2.8)
    bars[2].set_edgecolor(C_TEAL)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 50, f"{v:.0f}",
                ha="center", fontsize=13, fontweight="bold", color=C_INK)
    ax.set_ylabel("MAE")
    ax.set_title("Тип A · H = 3", color=C_TEAL, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.18)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(0.05, 0.92, "ML выигрывает —\nно только там,\nгде это подтверждается",
             fontsize=14, fontweight="bold", color=C_ACCENT, linespacing=1.35, va="top")
    _card(ax2, 0.05, 0.42, 0.90, 0.32, ec=C_TEAL, fc=C_SOFT_TEAL)
    ax2.text(0.5, 0.66, "Тип A · H=3", ha="center", fontsize=11,
             fontweight="bold", color=C_TEAL)
    ax2.text(0.5, 0.52, f"LGBM  {vals[2]:.0f}   ·   Prophet  {vals[1]:.0f}\nNaive  {vals[0]:.0f}",
             ha="center", va="center", fontsize=12, color=C_INK, linespacing=1.45)
    _card(ax2, 0.05, 0.08, 0.90, 0.28, ec=C_CORAL, fc=C_SOFT_CORAL)
    pb, lb = float(row_b["MAE_prophet"]), float(row_b["MAE_lgbm"])
    ax2.text(0.5, 0.28, "Тип B · H=3 — картина обратная",
             ha="center", fontsize=11, fontweight="bold", color=C_CORAL)
    ax2.text(0.5, 0.16, f"Prophet  {pb:.0f}   vs   LGBM  {lb:.0f}",
             ha="center", fontsize=12, color=C_INK)
    fig.suptitle("Локальный выигрыш, не универсальное превосходство",
                 color=C_MUTED, fontsize=11, y=0.02)
    return _save(fig, "slide06_rural_h3.png")


def slide07_changepoints(d):
    cons = d["cons"].copy()
    cons["date_s"] = pd.to_datetime(cons["date"]).dt.strftime("%Y-%m")
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.35)
    ax_t = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")
    ax_t.text(0.5, 0.88, "Мы не доверяем одному алгоритму — ищем консенсус",
              ha="center", fontsize=15, fontweight="bold", color=C_ACCENT)
    # pipeline chips
    for i, (txt, col) in enumerate([
        ("9 методов", C_ACCENT),
        ("5/9 сигнал", C_TEAL),
        ("7/9 сильный сигнал", C_CORAL),
    ]):
        x = 0.08 + i * 0.31
        _card(ax_t, x, 0.35, 0.28, 0.35, ec=col, fc=C_CARD)
        ax_t.text(x + 0.14, 0.52, txt, ha="center", va="center",
                  fontsize=13, fontweight="bold", color=col)
    ax_t.text(0.5, 0.12, "Порог консенсуса ≥ 50%  (не менее 5 из 9)",
              ha="center", fontsize=11, color=C_MUTED)

    # timeline of consensus dates
    timeline = [
        ("2023-11", 5, False),
        ("2024-04", 5, False),
        ("2024-09", 7, True),
    ]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.12, 0.88], [0.55, 0.55], color=C_LINE, lw=4, solid_capstyle="round")
    xs = [0.20, 0.50, 0.80]
    for x, (date, cnt, strong) in zip(xs, timeline):
        col = C_CORAL if strong else C_TEAL
        ax.plot(x, 0.55, "o", markersize=22 if strong else 16, color=col, zorder=3)
        label = f"{'* ' if strong else ''}{date}"
        ax.text(x, 0.78, label, ha="center", fontsize=14 if strong else 13,
                fontweight="bold", color=col)
        ax.text(x, 0.32, f"{cnt}/9", ha="center", fontsize=16,
                fontweight="bold", color=C_INK)
        ax.text(x, 0.18, "сильный" if strong else "сигнал",
                ha="center", fontsize=10, color=C_MUTED)
    ax.text(0.5, 0.02, "2024-09 — самая согласованная точка   ·   PR₊: 4.303 → 4.267 → 4.162",
            ha="center", fontsize=11, color=C_MUTED)
    return _save(fig, "slide07_changepoints.png")


def slide08_external(d):
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.93, "Внешний сигнал: что удалось, а что нет",
            ha="center", fontsize=16, fontweight="bold", color=C_ACCENT)

    # Trends block
    _card(ax, 0.05, 0.38, 0.55, 0.48, ec=C_TEAL, lw=2.2)
    ax.text(0.32, 0.78, "Google Trends", ha="center", fontsize=14,
            fontweight="bold", color=C_TEAL)
    items = [
        ("[+]", "выровнены по дате (aligned)", C_TEAL),
        ("[+]", "без заглядывания в будущее", C_TEAL),
        ("[!]", "один ряд на оба типа", C_GOLD),
        ("[!]", "часто ухудшают MAE", C_GOLD),
    ]
    for i, (mark, txt, col) in enumerate(items):
        ax.text(0.10, 0.66 - i * 0.08, f"{mark}  {txt}",
                fontsize=12, color=col)

    # News block
    _card(ax, 0.64, 0.38, 0.31, 0.48, ec=C_RED, lw=2.2, fc=C_SOFT_CORAL)
    ax.text(0.795, 0.72, "News NLP", ha="center", fontsize=14,
            fontweight="bold", color=C_RED)
    ax.text(0.795, 0.55, "[x]\nне реализован", ha="center", va="center",
            fontsize=14, color=C_INK, linespacing=1.4)
    ax.text(0.795, 0.42, "GDELT — без\nсистематического win",
            ha="center", fontsize=10, color=C_MUTED, linespacing=1.3)

    _card(ax, 0.12, 0.08, 0.76, 0.20, ec=C_ACCENT, fc=C_SOFT, lw=2)
    ax.text(0.5, 0.18, "Мы не выдаём Trends за новости.",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_ACCENT)
    return _save(fig, "slide08_external.png")


def slide09_chronos(d):
    found = d["foundation"]
    winners = { (w["cluster"], w["H"]): w for w in d["winners"] }
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 2.6], hspace=0.25)
    ax_t = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")
    ax_t.text(0.5, 0.65, "Foundation-модель не дала преимущества — и это тоже результат",
              ha="center", fontsize=15, fontweight="bold", color=C_ACCENT)
    ax_t.text(0.5, 0.25, "0 / 8 горизонтов выиграны Chronos   ·   ряд ≈ 24 точки слишком короткий",
              ha="center", fontsize=12, color=C_MUTED)

    labels, chronos_vals, classic_vals, classic_names = [], [], [], []
    for _, row in found.sort_values(["cluster", "horizon"]).iterrows():
        cl, H = int(row["cluster"]), int(row["horizon"])
        name = "A" if cl == 0 else "B"
        labels.append(f"{name} H{H}")
        chronos_vals.append(float(row["MAE_chronos"]))
        w = winners[(cl, H)]
        classic_vals.append(w["MAE"])
        classic_names.append(w["winner"])

    x = np.arange(len(labels))
    wbar = 0.38
    ax.bar(x - wbar / 2, chronos_vals, wbar, label="Chronos-bolt-tiny", color=C_CORAL, alpha=0.88)
    ax.bar(x + wbar / 2, classic_vals, wbar, label="лучший классический", color=C_TEAL, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAE")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, name in enumerate(classic_names):
        ax.text(x[i] + wbar / 2, classic_vals[i] * 1.03, name,
                ha="center", va="bottom", fontsize=7, color=C_TEAL, rotation=0)
    ax.set_title("Слева Chronos · справа лучший classic baseline по клетке",
                 color=C_MUTED, fontsize=10)
    return _save(fig, "slide09_chronos.png")


def slide10_final(d):
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 0.018, 1, facecolor=C_ACCENT, edgecolor="none"))
    ax.text(0.06, 0.90, "Что мы узнали о системе",
            fontsize=18, fontweight="bold", color=C_ACCENT)

    points = [
        ("01", "Нет единого победителя прогноза"),
        ("02", "Спектральный сигнал для сдвигов независим\nот конкретной модели прогноза"),
        ("03", "Три точки ≥50%, сильнейшая — сентябрь 2024"),
    ]
    for i, (num, txt) in enumerate(points):
        y = 0.72 - i * 0.14
        ax.text(0.08, y, num, fontsize=20, fontweight="bold", color=C_TEAL, va="center")
        ax.text(0.18, y, txt, fontsize=14, color=C_INK, va="center", linespacing=1.3)

    _card(ax, 0.06, 0.08, 0.88, 0.22, ec=C_ACCENT, fc=C_SOFT, lw=2.2)
    ax.text(
        0.5, 0.19,
        "Не всегда важно точно предсказать число.\n"
        "Иногда важнее вовремя заметить, что система стала другой.",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=C_ACCENT, linespacing=1.4,
    )
    return _save(fig, "slide10_final.png")


# Legacy aliases kept so older PDF builders / reports don't break if referenced
LEGACY_ALIASES = {
    "slide02_two_levels.png": "slide02_idea.png",
    "slide04_leakage_note.png": "slide04_leakage.png",
    "slide05_mae_grid.png": "slide05_mae_winners.png",
    "slide06_rural_h3_case.png": "slide06_rural_h3.png",
    "slide07_changepoint_consensus.png": "slide07_changepoints.png",
    "slide08_cp_example.png": "slide07_changepoints.png",
    "slide09_trends_not_news.png": "slide08_external.png",
    "slide10_gaps.png": "slide09_chronos.png",
    "slide11_takeaways.png": "slide10_final.png",
}


def _write_legacy_aliases():
    import shutil
    for old, new in LEGACY_ALIASES.items():
        src, dst = OUT / new, OUT / old
        if src.exists():
            shutil.copy2(src, dst)


def main():
    _style()
    print("Loading Task 2 artifacts…")
    d = load_data()
    print("Rendering slides…")
    slide01_title(d)
    slide02_idea(d)
    slide03_architecture(d)
    slide04_leakage(d)
    slide05_mae_winners(d)
    slide06_rural_h3(d)
    slide07_changepoints(d)
    slide08_external(d)
    slide09_chronos(d)
    slide10_final(d)
    _write_legacy_aliases()
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
