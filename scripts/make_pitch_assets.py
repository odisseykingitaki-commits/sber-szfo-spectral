"""Generate one PNG per Task-1 pitch slide → docs/presentation_assets/.

Run from repo root (conda env sber):
    python scripts/make_pitch_assets.py

Uses current no-income artifacts: features p=17, evals/evecs, labels_threshold,
bootstrap_stability, robustness, dynamic_summary / trajectories.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import feature_cols, DATA_PROC, RESULTS, FIGURES  # noqa: E402

OUT = ROOT / "docs" / "presentation_assets"
FEAT = DATA_PROC / "features_szfo_v2_final.csv"
C_REG = 0.2

# Visual theme (presentation-friendly, not purple-AI default)
C_BG = "#f7f5f1"
C_INK = "#1a2332"
C_ACCENT = "#1e3a5f"
C_TEAL = "#2a6f6a"
C_CORAL = "#c45c26"
C_MUTED = "#6b7280"
C_CARD = "#ffffff"
C_LINE = "#d4cfc4"


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


def load_canon():
    df = pd.read_csv(FEAT, encoding="utf-8-sig")
    feat = feature_cols(df)
    X = StandardScaler().fit_transform(df[feat].values)
    evals = np.load(RESULTS / "evals_v2.npy")
    evecs = np.load(RESULTS / "evecs_v2.npy")
    # ensure descending (05 saves sorted)
    if evals[0] < evals[-1]:
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
    labels = np.load(RESULTS / "labels_threshold.npy")
    boot = pd.read_csv(RESULTS / "bootstrap_stability.csv")["ari"].values
    rob = pd.read_csv(RESULTS / "robustness.csv")
    dyn = json.loads((RESULTS / "dynamic_summary.json").read_text(encoding="utf-8"))
    traj = pd.read_csv(DATA_PROC / "trajectories.csv", index_col=0)
    summ = json.loads((RESULTS / "final_no_income_summary.json").read_text(encoding="utf-8"))
    v1 = evecs[:, 0]
    U1 = X @ v1
    pos = evals[evals > 0]
    lam_frac = float(evals[0] / pos.sum()) if len(pos) else 0.0
    n_switch = int((traj["n_switches"] > 0).sum())
    n_traj = int(len(traj))
    aris = [a["ari"] for a in dyn.get("aris", [])]
    return {
        "df": df, "feat": feat, "X": X, "evals": evals, "evecs": evecs,
        "labels": labels, "boot": boot, "rob": rob, "dyn": dyn, "traj": traj,
        "summ": summ, "U1": U1, "v1": v1, "lam_frac": lam_frac,
        "n_switch": n_switch, "n_traj": n_traj, "aris": aris,
        "N": len(df), "p": len(feat),
    }


# ─── Slide visuals ───────────────────────────────────────────────────────────

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
    ax.text(0.5, 0.48, "Задача 1 · типология муниципалитетов СЗФО",
            ha="center", fontsize=14, color=C_INK)
    # metric chips
    chips = [
        (0.22, f"N = {d['N']}"),
        (0.42, f"p = {d['p']}"),
        (0.62, "2 режима"),
        (0.82, "без дохода Росстата"),
    ]
    for x, t in chips:
        ax.add_patch(FancyBboxPatch((x - 0.09, 0.28), 0.18, 0.1,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#e8eef5", edgecolor=C_LINE))
        ax.text(x, 0.33, t, ha="center", va="center", fontsize=10, color=C_ACCENT)
    ax.text(0.5, 0.18, "СберИндекс 2023–2024 · защита для жюри ML + экономика",
            ha="center", fontsize=10, color=C_MUTED)
    return _save(fig, "slide01_title_card.png")


def slide02_hook_scatter(d):
    rng = np.random.default_rng(42)
    # 2D PCA-ish via top-2 standardized feature projections for abstract cloud
    X = d["X"]
    # use first two PCs of X for layout (no cluster coloring)
    u, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    xy = u[:, :2] * s[:2]
    jitter = rng.normal(0, 0.02, xy.shape)
    xy = xy + jitter
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.scatter(xy[:, 0], xy[:, 1], s=28, c=C_ACCENT, alpha=0.55,
               edgecolors="white", linewidths=0.3)
    ax.set_title(f"{d['N']} муниципалитетов · {d['p']} признаков — сколько режимов?",
                 color=C_ACCENT, pad=12)
    ax.set_xlabel("проекция (абстрактное облако, без кластеров)")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.02,
            "Вопрос: есть ли скрытая структура поведения?",
            transform=ax.transAxes, ha="right", fontsize=10, color=C_MUTED,
            style="italic")
    return _save(fig, "slide02_hook_scatter.png")


def slide03_kmeans_vs_spectral(d):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Один датасет — два вопроса", color=C_ACCENT, fontsize=14, pad=8)

    def box(x, y, w, h, title, lines, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=C_CARD, edgecolor=edge, lw=2))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", fontsize=12,
                fontweight="bold", color=edge)
        for i, ln in enumerate(lines):
            ax.text(x + 0.2, y + h - 0.9 - i * 0.45, ln, fontsize=10, color=C_INK)

    box(0.4, 1.2, 4.2, 3.8, "KMeans",
        ["Вопрос: кто на кого похож?",
         "17 признаков → расстояния",
         "→ центроиды → кластеры",
         "",
         "Похожесть строк матрицы X"],
        C_CORAL)
    box(5.4, 1.2, 4.2, 3.8, "Спектральный PLM",
        ["Вопрос: что связывает систему?",
         "17 → зависимости → J",
         "→ спектр → главная мода",
         "",
         "Коллективная структура связей"],
        C_TEAL)
    ax.annotate("", xy=(5.3, 3.1), xytext=(4.7, 3.1),
                arrowprops=dict(arrowstyle="<->", color=C_MUTED, lw=1.5))
    ax.text(5.0, 0.55, "Не просто «кто с кем» — а «что объединяет систему»",
            ha="center", fontsize=11, color=C_INK, style="italic")
    return _save(fig, "slide03_kmeans_vs_spectral.png")


def slide04_pipeline(d):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Как из 17 показателей появляется одна ось",
                 color=C_ACCENT, fontsize=14, pad=6)
    steps = [
        (0.3, "Расходы\nМО"),
        (2.3, "17\nпризнаков"),
        (4.3, "PLM"),
        (6.0, "J"),
        (7.5, "Спектр"),
        (9.3, "v₁"),
        (11.0, "U₁"),
    ]
    for i, (x, lab) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 1.4), 1.6, 1.4,
                                    boxstyle="round,pad=0.04",
                                    facecolor=C_CARD, edgecolor=C_ACCENT, lw=1.8))
        ax.text(x + 0.8, 2.1, lab, ha="center", va="center", fontsize=11,
                fontweight="bold", color=C_ACCENT)
        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i + 1][0] - 0.05, 2.1),
                        xytext=(x + 1.65, 2.1),
                        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.6))
    ax.text(7.0, 0.7,
            f"λ₁ ≈ {d['evals'][0]:.3f}  ·  {100*d['lam_frac']:.1f}% положительной спектральной массы  ·  p={d['p']}",
            ha="center", fontsize=11, color=C_INK)
    return _save(fig, "slide04_pipeline_flowchart.png")


def slide05_mode1_loadings(d):
    v1 = d["v1"]
    feat = d["feat"]
    order = np.argsort(np.abs(v1))[::-1]
    # show all 17, highlight top-5
    names = [feat[i].replace("share_", "").replace("growth_", "g_")
             .replace("cv_", "cv_") for i in order]
    vals = v1[order]
    colors = [C_TEAL if v >= 0 else C_CORAL for v in vals]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=C_INK, lw=0.8)
    ax.set_xlabel("loading v₁")
    ax.set_title(f"Главная мода (λ₁={d['evals'][0]:.3f}) — loadings всех {d['p']} признаков",
                 color=C_ACCENT)
    top = order[:5]
    tip = " · ".join(
        f"{feat[i].replace('share_', '')} {v1[i]:+.3f}" for i in top
    )
    ax.text(0.01, -0.08, f"Топ-5: {tip}", transform=ax.transAxes,
            fontsize=9, color=C_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "slide05_mode1_loadings.png")


def slide06_u1_split(d):
    U1 = d["U1"]
    lab = d["labels"]
    med = float(np.median(U1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    # strip / swarm-ish
    ax = axes[0]
    rng = np.random.default_rng(0)
    for k, col in enumerate([C_CORAL, C_TEAL]):
        mask = lab == k
        yy = rng.normal(k, 0.08, mask.sum())
        ax.scatter(U1[mask], yy, s=22, c=col, alpha=0.65,
                   edgecolors="white", linewidths=0.25, label=f"класс {k} (n={mask.sum()})")
    ax.axvline(med, color=C_ACCENT, ls="--", lw=2, label=f"медиана U₁={med:.2f}")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ниже медианы", "выше медианы"])
    ax.set_xlabel("U₁ = X · v₁")
    ax.set_title("Разрез по главной оси", color=C_ACCENT)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.hist(U1[lab == 0], bins=22, color=C_CORAL, alpha=0.7, edgecolor="white",
            label="класс 0")
    ax.hist(U1[lab == 1], bins=22, color=C_TEAL, alpha=0.7, edgecolor="white",
            label="класс 1")
    ax.axvline(med, color=C_ACCENT, ls="--", lw=2)
    ax.set_xlabel("U₁")
    ax.set_ylabel("частота")
    ax.set_title("140 / 140 by construction", color=C_ACCENT)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("Median threshold на U₁ — основной метод", color=C_INK, fontsize=13)
    fig.tight_layout()
    return _save(fig, "slide06_U1_median_split.png")


def slide07_metric_cards(d):
    boot_m, boot_s = float(d["boot"].mean()), float(d["boot"].std())
    rob_min = float(d["rob"]["ARI"].min())
    ari_km = float(d["summ"]["ARI_vs_KMeans"])
    pct = 100.0 * d["n_switch"] / d["n_traj"]
    cards = [
        ("Bootstrap ARI", f"{boot_m:.3f} ± {boot_s:.3f}",
         "устойчивость к ресемплу"),
        ("Robustness C_reg", f"{rob_min:.3f}",
         "min ARI при смене регуляризации"),
        ("ARI vs baselines", f"{ari_km:.3f}",
         "KMeans / Louvain ≈ тот же масштаб"),
        ("Перебежчики", f"{pct:.1f}%",
         f"{d['n_switch']} из {d['n_traj']} МО во времени"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
    for ax, (title, val, note) in zip(axes, cards):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.9, 0.84,
                                    boxstyle="round,pad=0.04",
                                    facecolor=C_CARD, edgecolor=C_ACCENT, lw=2))
        ax.text(0.5, 0.78, title, ha="center", fontsize=10, color=C_MUTED)
        ax.text(0.5, 0.48, val, ha="center", fontsize=18, fontweight="bold",
                color=C_ACCENT)
        ax.text(0.5, 0.22, note, ha="center", fontsize=8.5, color=C_INK,
                wrap=True)
    fig.suptitle("Структура выдерживает изменения метода", color=C_ACCENT,
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return _save(fig, "slide07_stability_metric_cards.png")


def slide08_dynamics(d):
    aris = d["aris"]
    xs = [a["from"] for a in d["dyn"]["aris"]]
    ys = [a["ari"] for a in d["dyn"]["aris"]]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(xs, ys, color=C_TEAL, edgecolor="white", width=0.75)
    ax.axhline(float(np.mean(ys)), color=C_CORAL, ls="--", lw=1.5,
               label=f"mean ARI = {np.mean(ys):.3f}")
    ax.set_xlabel("окно t → t+1")
    ax.set_ylabel("ARI")
    ax.set_ylim(0.6, 1.02)
    ax.set_title(
        f"Динамика: стабильность между окнами · перебежчики "
        f"{d['n_switch']}/{d['n_traj']} ({100*d['n_switch']/d['n_traj']:.1f}%)",
        color=C_ACCENT,
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # also copy/refresh companion from fig_dynamic for README note
    return _save(fig, "slide08_dynamics_ari.png")


def slide09_value_blocks(d):
    blocks = [
        ("Сегментация", f"{d['N']} МО → 2 режима\n140 / 140"),
        ("Интерпретация", "режимы из связей\nпризнаков (J), не только\nпохожести строк"),
        ("Мониторинг", f"кто меняет режим:\n{d['n_switch']} перебежчиков"),
        ("Воспроизводимость", "данные → PLM → спектр\n→ U₁ → типология\n→ динамика"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Что получает аналитик", color=C_ACCENT, fontsize=14)
    positions = [(0.4, 2.2), (2.7, 2.2), (5.0, 2.2), (7.3, 2.2)]
    for (x, y), (title, body) in zip(positions, blocks):
        ax.add_patch(FancyBboxPatch((x, y - 1.4), 2.1, 2.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor=C_CARD, edgecolor=C_TEAL, lw=2))
        ax.text(x + 1.05, y + 1.0, title, ha="center", fontsize=12,
                fontweight="bold", color=C_ACCENT)
        ax.text(x + 1.05, y - 0.2, body, ha="center", va="center",
                fontsize=9.5, color=C_INK, linespacing=1.35)
    ax.text(5.0, 0.45,
            "Мы не просто нашли два кластера — мы нашли ось, вдоль которой организована система.",
            ha="center", fontsize=10, style="italic", color=C_MUTED)
    return _save(fig, "slide09_value_blocks.png")


def slide10_takeaways(d):
    boot_m = float(d["boot"].mean())
    rob_min = float(d["rob"]["ARI"].min())
    items = [
        f"2 режима · p={d['p']} · λ₁≈{d['evals'][0]:.3f} ({100*d['lam_frac']:.1f}% Σλ₊)",
        f"Канон: median threshold → 140/140 · SW≈{d['summ']['SW']:.3f}",
        f"Доверие: bootstrap≈{boot_m:.2f} · robustness≈{rob_min:.2f} · "
        f"~{100*d['n_switch']/d['n_traj']:.0f}% переходов",
        "Доход Росстата отклонён · q75/Variant C — только ablation",
        "Полный метод → docs/TASK1_METHOD_REPORT.md",
    ]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.06, 0.08), 0.88, 0.84,
                                boxstyle="round,pad=0.02",
                                facecolor=C_CARD, edgecolor=C_ACCENT, lw=2))
    ax.text(0.5, 0.88, "Спектральный портрет СЗФО — итог",
            ha="center", fontsize=15, fontweight="bold", color=C_ACCENT)
    y = 0.72
    for it in items:
        ax.text(0.12, y, "▸", fontsize=14, color=C_TEAL, va="center")
        ax.text(0.16, y, it, fontsize=12, color=C_INK, va="center")
        y -= 0.11
    ax.text(0.5, 0.16,
            "Следующий шаг — раннее обнаружение структурных сдвигов",
            ha="center", fontsize=10, style="italic", color=C_MUTED)
    return _save(fig, "slide10_takeaways_checklist.png")


def write_readme(paths: list[Path], d: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    boot_m = float(d["boot"].mean())
    rob_min = float(d["rob"]["ARI"].min())
    lines = [
        "presentation_assets — визуалы для pitch Task 1 (TASK1_slides.md)",
        f"Сгенерировано: {now}",
        f"Канон: no-income p={d['p']}, N={d['N']}, λ₁={d['evals'][0]:.4f}, "
        f"bootstrap={boot_m:.3f}, robustness_min={rob_min:.3f}, "
        f"switchers={d['n_switch']}/{d['n_traj']}",
        "",
        "Файл → слайд:",
        "  slide01_title_card.png              → Слайд 1 (титул)",
        "  slide02_hook_scatter.png            → Слайд 2 (hook / облако МО)",
        "  slide03_kmeans_vs_spectral.png      → Слайд 3 (KMeans vs спектр)",
        "  slide04_pipeline_flowchart.png      → Слайд 4 (пайплайн → U₁)",
        "  slide05_mode1_loadings.png          → Слайд 5 (loadings моды 1)",
        "  slide06_U1_median_split.png         → Слайд 6 (ось U₁, 140/140)",
        "  slide07_stability_metric_cards.png  → Слайд 7 (4 карточки устойчивости)",
        "  slide08_dynamics_ari.png            → Слайд 8 (динамика ARI / перебежчики)",
        "  slide09_value_blocks.png            → Слайд 9 (ценность для аналитика)",
        "  slide10_takeaways_checklist.png     → Слайд 10 (итог / takeaways)",
        "",
        "Также скопированы pipeline-figures (после regen 06/07/09/10):",
        "  01_spectrum_bootstrap_robustness.png  ← figures/fig_final.png",
        "  02_network_louvain.png                ← figures/fig_network.png",
        "  03_dynamics.png                       ← figures/fig_dynamic.png",
        "  04_bootstrap_hist.png                 ← figures/fig_bootstrap.png",
        "",
        "Пересборка:  python scripts/make_pitch_assets.py",
        "PDF слайдов: python scripts/build_task1_pdfs.py",
    ]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → README.txt")


def copy_pipeline_figs():
    import shutil
    mapping = {
        "fig_final.png": "01_spectrum_bootstrap_robustness.png",
        "fig_network.png": "02_network_louvain.png",
        "fig_dynamic.png": "03_dynamics.png",
        "fig_bootstrap.png": "04_bootstrap_hist.png",
    }
    for src, dst in mapping.items():
        s = FIGURES / src
        if s.exists():
            shutil.copy2(s, OUT / dst)
            print(f"  → {dst} (copy)")


def main():
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading canon data…")
    d = load_canon()
    print(f"  N={d['N']} p={d['p']} λ1={d['evals'][0]:.4f} "
          f"sizes={np.bincount(d['labels']).tolist()} "
          f"switch={d['n_switch']}/{d['n_traj']}")
    print("Rendering slides…")
    paths = [
        slide01_title(d),
        slide02_hook_scatter(d),
        slide03_kmeans_vs_spectral(d),
        slide04_pipeline(d),
        slide05_mode1_loadings(d),
        slide06_u1_split(d),
        slide07_metric_cards(d),
        slide08_dynamics(d),
        slide09_value_blocks(d),
        slide10_takeaways(d),
    ]
    print("Copying pipeline figures…")
    copy_pipeline_figs()
    write_readme(paths, d)
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
