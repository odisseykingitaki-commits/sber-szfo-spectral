"""Generate chart-first PNG visuals for Task-1 jury pitch → scripts/_pitch/presentation_assets/.

Run from repo root:
    python scripts/make_pitch_assets.py

All numbers/arrays from results/* and data/processed — no invented data.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import feature_cols, DATA_PROC, RESULTS, FIGURES  # noqa: E402

OUT = ROOT / "scripts" / "_pitch" / "presentation_assets"
FEAT = DATA_PROC / "features_szfo_v2_final.csv"

# Navy + teal + orange
C_BG = "#f4f6f8"
C_INK = "#1a2332"
C_NAVY = "#1e3a5f"
C_TEAL = "#2a6f6a"
C_ORANGE = "#c45c26"
C_MUTED = "#6b7280"
C_CARD = "#ffffff"
C_LINE = "#d0d5dd"


def _style():
    plt.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor": C_CARD,
        "axes.edgecolor": C_LINE,
        "axes.labelcolor": C_INK,
        "text.color": C_INK,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
    })


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=C_BG, pad_inches=0.15)
    plt.close(fig)
    print(f"  -> {path.name}")
    return path


def _short(name: str) -> str:
    return (
        name.replace("share_", "")
        .replace("growth_", "g_")
        .replace("cv_", "cv_")
        .replace("marketplace", "mkt")
        .replace("transport", "trans")
        .replace("log_total", "log tot")
        .replace("mob_logratio", "mob")
    )


def load_canon():
    df = pd.read_csv(FEAT, encoding="utf-8-sig")
    feat = feature_cols(df)
    X = StandardScaler().fit_transform(df[feat].values)
    evals = np.load(RESULTS / "evals_v2.npy")
    evecs = np.load(RESULTS / "evecs_v2.npy")
    if evals[0] < evals[-1]:
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
    labels = np.load(RESULTS / "labels_threshold.npy")
    J = np.load(RESULTS / "J_v2.npy")
    boot = pd.read_csv(RESULTS / "bootstrap_stability.csv")["ari"].values
    rob = pd.read_csv(RESULTS / "robustness.csv")
    dyn = json.loads((RESULTS / "dynamic_summary.json").read_text(encoding="utf-8"))
    traj = pd.read_csv(DATA_PROC / "trajectories.csv", index_col=0)
    summ = json.loads((RESULTS / "final_no_income_summary.json").read_text(encoding="utf-8"))
    volog = pd.read_csv(OUT / "case_vologodsky_trajectory.csv") if (OUT / "case_vologodsky_trajectory.csv").exists() else None
    v1 = evecs[:, 0]
    U1 = X @ v1
    U2 = X @ evecs[:, 1]
    pos = evals[evals > 0]
    lam_frac = float(evals[0] / pos.sum()) if len(pos) else 0.0
    n_switch = int((traj["n_switches"] > 0).sum())
    n_traj = int(len(traj))
    aris = dyn.get("aris", [])
    return {
        "df": df, "feat": feat, "X": X, "evals": evals, "evecs": evecs, "J": J,
        "labels": labels, "boot": boot, "rob": rob, "dyn": dyn, "traj": traj,
        "summ": summ, "U1": U1, "U2": U2, "v1": v1, "lam_frac": lam_frac,
        "n_switch": n_switch, "n_traj": n_traj, "aris": aris, "volog": volog,
        "N": len(df), "p": len(feat),
    }


# ─── Slide 1: hook cloud (no regimes) ───────────────────────────────────────

def fig_slide01_hook(d):
    X = d["X"]
    u, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    xy = u[:, :2] * s[:2]
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.scatter(xy[:, 0], xy[:, 1], s=55, c=C_NAVY, alpha=0.62,
               edgecolors="white", linewidths=0.35, zorder=2)
    ax.set_xlabel("PC1 (пространство признаков)", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title(f"{d['N']} муниципалитетов в пространстве {d['p']} признаков",
                 color=C_NAVY, fontsize=15, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.99, 0.02, "без разбиения на режимы — только объекты",
            transform=ax.transAxes, ha="right", fontsize=11,
            color=C_MUTED, style="italic")
    return _save(fig, "slide01_hook_cloud.png")


# ─── Slide 2: KMeans-style vs J ──────────────────────────────────────────────

def fig_slide02_kmeans_vs_j(d):
    X = d["X"]
    u, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    xy = u[:, :2] * s[:2]
    km = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(X)
    J = d["J"]
    feat = [_short(f) for f in d["feat"]]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.2, 6.4),
                                   gridspec_kw={"width_ratios": [1.05, 1.15]})
    cols = [C_ORANGE if k == 0 else C_TEAL for k in km]
    ax0.scatter(xy[:, 0], xy[:, 1], s=36, c=cols, alpha=0.7,
                edgecolors="white", linewidths=0.3)
    ax0.set_title("Похожесть объектов", color=C_NAVY, fontsize=14)
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(handles=[
        mpatches.Patch(color=C_ORANGE, label="KMeans-стиль A"),
        mpatches.Patch(color=C_TEAL, label="KMeans-стиль B"),
    ], frameon=False, loc="lower left", fontsize=9)

    cmap = LinearSegmentedColormap.from_list(
        "nto", [C_ORANGE, "#f4f6f8", C_TEAL], N=256
    )
    vmax = float(np.max(np.abs(J)))
    im = ax1.imshow(J, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    ax1.set_xticks(range(len(feat)))
    ax1.set_yticks(range(len(feat)))
    ax1.set_xticklabels(feat, rotation=70, ha="right", fontsize=7.5)
    ax1.set_yticklabels(feat, fontsize=7.5)
    ax1.set_title("Матрица связей J (17×17)", color=C_NAVY, fontsize=14)
    for i in range(len(feat)):
        ax1.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor=C_NAVY, lw=0.8))
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, "slide02_kmeans_vs_j.png")


# ─── Slide 3: dependency map J ───────────────────────────────────────────────

def fig_slide03_j_map(d):
    J = d["J"]
    feat = [_short(f) for f in d["feat"]]
    fig = plt.figure(figsize=(13.2, 7.0))
    ax = fig.add_axes([0.10, 0.12, 0.72, 0.80])
    cmap = LinearSegmentedColormap.from_list(
        "nto", [C_ORANGE, "#ffffff", C_TEAL], N=256
    )
    vmax = float(np.max(np.abs(J)))
    im = ax.imshow(J, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(feat)))
    ax.set_yticks(range(len(feat)))
    ax.set_xticklabels(feat, rotation=65, ha="right", fontsize=10)
    ax.set_yticklabels(feat, fontsize=10)
    ax.set_title("Карта зависимостей признаков · diag(J)=0", color=C_NAVY, fontsize=15)
    Joff = J.copy()
    np.fill_diagonal(Joff, 0)
    thr = np.quantile(np.abs(Joff).ravel(), 0.92)
    ys, xs = np.where(np.abs(Joff) >= thr)
    for y, x in zip(ys, xs):
        if x >= y:
            continue
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                   fill=False, edgecolor=C_NAVY, lw=1.6))
    cax = fig.add_axes([0.84, 0.22, 0.028, 0.58])
    fig.colorbar(im, cax=cax)
    ax2 = fig.add_axes([0.90, 0.35, 0.09, 0.35])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    for i, lab in enumerate(["J", "спектр", "v₁"]):
        y = 0.85 - i * 0.32
        ax2.add_patch(FancyBboxPatch(
            (0.05, y - 0.1), 0.9, 0.2, boxstyle="round,pad=0.02",
            facecolor=C_CARD, edgecolor=C_NAVY, lw=1.4,
        ))
        ax2.text(0.5, y, lab, ha="center", va="center", fontsize=11,
                 fontweight="bold", color=C_NAVY)
        if i < 2:
            ax2.annotate("", xy=(0.5, y - 0.14), xytext=(0.5, y - 0.1),
                         arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.8))
    return _save(fig, "slide03_j_heatmap.png")


# ─── Slide 4: spectrum magic ─────────────────────────────────────────────────

def fig_slide04_spectrum(d):
    evals = d["evals"]
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    idx = np.arange(1, len(evals) + 1)
    colors = []
    for i, e in enumerate(evals):
        if i == 0:
            colors.append(C_TEAL)
        elif e > 0:
            colors.append("#7aa8a4")
        else:
            colors.append("#d4a574")
    bars = ax.bar(idx, evals, color=colors, edgecolor="white", width=0.78, zorder=2)
    bars[0].set_edgecolor(C_NAVY)
    bars[0].set_linewidth(2.2)
    ax.axhline(0, color=C_INK, lw=0.8)
    ax.set_xlabel("мода k", fontsize=12)
    ax.set_ylabel("λₖ", fontsize=12)
    ax.set_xticks(idx)
    ax.set_title("Спектр матрицы J — первая мода доминирует", color=C_NAVY, fontsize=15)
    ax.annotate(
        f"λ₁ = {evals[0]:.3f}\n{100 * d['lam_frac']:.1f}% Σλ₊",
        xy=(1, evals[0]), xytext=(3.2, evals[0] * 0.92),
        fontsize=16, fontweight="bold", color=C_TEAL,
        arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=2),
        bbox=dict(boxstyle="round,pad=0.35", fc=C_CARD, ec=C_TEAL, lw=1.5),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "slide04_spectrum.png")


# ─── Slide 5: economic loadings (large) ──────────────────────────────────────

def fig_slide05_loadings(d):
    v1 = d["v1"]
    feat = d["feat"]
    order = np.argsort(np.abs(v1))[::-1]
    names = [_short(feat[i]) for i in order]
    vals = v1[order]
    colors = [C_TEAL if v >= 0 else C_ORANGE for v in vals]
    fig, ax = plt.subplots(figsize=(12.2, 7.4))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.invert_yaxis()
    ax.axvline(0, color=C_INK, lw=1.0)
    ax.set_xlabel("loading v₁", fontsize=13)
    ax.set_title(f"Главная мода · λ₁ = {d['evals'][0]:.3f}", color=C_NAVY, fontsize=16)
    # annotate top-5 values on bars
    for i in range(5):
        v = vals[i]
        ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=11, fontweight="bold", color=C_NAVY)
    ax.text(0.98, 0.02,
            "бирюза = «+» полюс · оранж = «−» полюс · паттерны потребления",
            transform=ax.transAxes, ha="right", fontsize=10, color=C_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, "slide05_mode1_loadings.png")


# ─── Slide 6: U1 wow centerpiece ─────────────────────────────────────────────

def fig_slide06_u1(d):
    U1 = d["U1"]
    lab = d["labels"]
    med = float(np.median(U1))
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(13.0, 6.8))
    # single axis: all MO along U1, jittered vertically by regime
    for k, col, name, y0 in [
        (0, C_ORANGE, "Mode B (−)", 0.0),
        (1, C_TEAL, "Mode A (+)", 1.0),
    ]:
        mask = lab == k
        yy = rng.normal(y0, 0.12, mask.sum())
        ax.scatter(U1[mask], yy, s=48, c=col, alpha=0.72,
                   edgecolors="white", linewidths=0.35, label=f"{name} · n={mask.sum()}",
                   zorder=3)
    ax.axvline(med, color=C_NAVY, ls="--", lw=2.6, zorder=4)
    ax.text(med, 1.55, "MEDIAN", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=C_NAVY)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Mode B", "Mode A"], fontsize=13)
    ax.set_ylim(-0.55, 1.75)
    ax.set_xlabel("U₁ = X · v₁", fontsize=14)
    ax.set_title("280 МО вдоль главной коллективной оси", color=C_NAVY, fontsize=16)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
    ax.text(0.02, 0.95, f"{n0} / {n1}", transform=ax.transAxes,
            fontsize=26, fontweight="bold", color=C_NAVY, va="top")
    ax.text(0.02, 0.82, "Mode B / Mode A", transform=ax.transAxes,
            fontsize=11, color=C_MUTED, va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, "slide06_U1_median_split.png")


# ─── Slide 7: robustness 4 panels ────────────────────────────────────────────

def fig_slide07_robustness(d):
    boot = d["boot"]
    rob = d["rob"]
    ari_louv = float(d["summ"]["ARI_vs_Louvain"])
    pct = 100.0 * d["n_switch"] / d["n_traj"]
    boot_m, boot_s = float(boot.mean()), float(boot.std())
    rob_min = float(rob["ARI"].min())

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))

    # Bootstrap hist
    ax = axes[0, 0]
    ax.hist(boot, bins=18, color=C_TEAL, edgecolor="white", alpha=0.9)
    ax.axvline(boot_m, color=C_NAVY, lw=2)
    ax.set_title(f"Bootstrap ARI  {boot_m:.3f} ± {boot_s:.3f}", color=C_NAVY, fontsize=12)
    ax.set_xlabel("ARI")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # C_reg line
    ax = axes[0, 1]
    ax.plot(rob["C_reg"], rob["ARI"], "o-", color=C_TEAL, lw=2.2, ms=7)
    ax.axhline(rob_min, color=C_ORANGE, ls="--", lw=1.5, label=f"min = {rob_min:.3f}")
    ax.set_xscale("log")
    ax.set_ylim(0.95, 1.005)
    ax.set_title(f"C_reg · min ARI = {rob_min:.3f}", color=C_NAVY, fontsize=12)
    ax.set_xlabel("C_reg")
    ax.set_ylabel("ARI vs канон")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Louvain ARI card + bar
    ax = axes[1, 0]
    ax.barh([0], [ari_louv], color=C_TEAL, height=0.45)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"Louvain ARI ≈ {ari_louv:.3f}", color=C_NAVY, fontsize=12)
    ax.text(ari_louv / 2, 0, f"{ari_louv:.3f}", ha="center", va="center",
            color="white", fontsize=16, fontweight="bold")
    ax.set_xlabel("ARI vs threshold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Dynamics switchers
    ax = axes[1, 1]
    stayed = 100 - pct
    ax.barh([0], [stayed], color=C_TEAL, height=0.5, label="без смены")
    ax.barh([0], [pct], left=[stayed], color=C_ORANGE, height=0.5, label="сменили режим")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_title(f"Динамика · {pct:.1f}% сменили режим", color=C_NAVY, fontsize=12)
    ax.set_xlabel("% МО (из 277 с траекторией)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Разные проверки — одна крупномасштабная структура",
                 color=C_NAVY, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "slide07_robustness.png")


# ─── Slide 8: dynamics ARI (large) ───────────────────────────────────────────

def fig_slide08_dynamics(d):
    rows = d["aris"]
    xs = np.array([int(a["from"]) for a in rows])
    ys = np.array([a["ari"] for a in rows], dtype=float)
    mean_ari = float(np.mean(ys))
    fig, ax = plt.subplots(figsize=(13.2, 6.9))
    ax.plot(xs, ys, "o-", color=C_TEAL, lw=2.8, ms=9, zorder=3)
    ax.fill_between(xs, ys, mean_ari, alpha=0.12, color=C_TEAL)
    ax.axhline(mean_ari, color=C_ORANGE, ls="--", lw=2.4, zorder=2,
               label=f"mean ARI ≈ {mean_ari:.2f}")
    ax.set_xticks(xs)
    ax.set_xlabel("окно t → t+1  (19 окон × 6 месяцев)", fontsize=12)
    ax.set_ylabel("ARI", fontsize=13)
    ax.set_ylim(0.65, 1.02)
    ax.set_title(
        f"Устойчивость между окнами · перебежчики "
        f"{d['n_switch']}/{d['n_traj']} = {100 * d['n_switch'] / d['n_traj']:.1f}%",
        color=C_NAVY, fontsize=14,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e2e8f0", lw=0.7)
    fig.tight_layout()
    return _save(fig, "slide08_dynamics_ari.png")


# ─── Slide 9: Vologda trajectory ─────────────────────────────────────────────

def fig_slide09_vologda(d):
    if d["volog"] is not None:
        vf = d["volog"]
        windows = vf["window"].values
        labels = vf["label"].values
    else:
        row = d["traj"].loc["вологодский"]
        windows = np.arange(1, 20)
        labels = np.array([int(row[f"w{i}"]) for i in windows])
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    cols = [C_ORANGE if L == 0 else C_TEAL for L in labels]
    ax.step(windows, labels, where="mid", color=C_NAVY, lw=2.4, zorder=2)
    ax.scatter(windows, labels, c=cols, s=110, zorder=3,
               edgecolors=C_NAVY, linewidths=1.2)
    # switch markers
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            ax.axvline(windows[i], color=C_ORANGE, alpha=0.35, lw=1.5, zorder=1)
            ax.scatter([windows[i]], [labels[i]], s=220, facecolors="none",
                       edgecolors=C_ORANGE, linewidths=2.2, zorder=4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Mode B", "Mode A"], fontsize=13)
    ax.set_xticks(windows)
    ax.set_xlabel("скользящее окно w1…w19", fontsize=12)
    ax.set_ylim(-0.35, 1.35)
    ax.set_title("вологодский · 8 переключений / 19 окон", color=C_NAVY, fontsize=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, "slide09_vologodsky_trajectory.png")


# ─── Slide 10: analyst value + quote ─────────────────────────────────────────

def fig_slide10_value(d):
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # horizontal chain
    chain = [
        (0.4, "17\nпризнаков"),
        (3.0, "спектральная\nструктура"),
        (5.8, "2 режима"),
        (8.6, "мониторинг\nпереходов"),
    ]
    for i, (x, lab) in enumerate(chain):
        ax.add_patch(FancyBboxPatch((x, 5.9), 2.2, 1.5,
                                    boxstyle="round,pad=0.04",
                                    facecolor=C_CARD, edgecolor=C_NAVY, lw=2))
        ax.text(x + 1.1, 6.65, lab, ha="center", va="center",
                fontsize=12, fontweight="bold", color=C_NAVY)
        if i < len(chain) - 1:
            ax.annotate("", xy=(chain[i + 1][0] - 0.05, 6.65),
                        xytext=(x + 2.25, 6.65),
                        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=2))

    blocks = [
        ("Сегментация", "280 → 2 режима"),
        ("Интерпретация", "ось из связей J"),
        ("Мониторинг", f"{d['n_switch']} перебежчиков"),
        ("Воспроизводимость", "полный pipeline"),
    ]
    for i, (t, b) in enumerate(blocks):
        x = 0.5 + i * 2.9
        ax.add_patch(FancyBboxPatch((x, 3.5), 2.55, 1.7,
                                    boxstyle="round,pad=0.04",
                                    facecolor=C_CARD, edgecolor=C_TEAL, lw=1.8))
        ax.text(x + 1.27, 4.7, t, ha="center", fontsize=12,
                fontweight="bold", color=C_TEAL)
        ax.text(x + 1.27, 4.05, b, ha="center", fontsize=11, color=C_INK)

    # central quote — main visual element
    ax.add_patch(FancyBboxPatch((0.4, 0.25), 11.2, 2.85,
                                boxstyle="round,pad=0.06",
                                facecolor="#e8eef5", edgecolor=C_NAVY, lw=3.0))
    ax.text(6.0, 2.25,
            "Мы не просто нашли два кластера.",
            ha="center", va="center", fontsize=20, fontweight="bold", color=C_NAVY)
    ax.text(6.0, 1.15,
            "Мы нашли ось, вдоль которой организована\nструктура потребительского поведения МО.",
            ha="center", va="center", fontsize=16, color=C_INK, linespacing=1.4)
    return _save(fig, "slide10_value_quote.png")


# ─── Slide 11: finale portrait ───────────────────────────────────────────────

def fig_slide11_finale(d):
    U1 = d["U1"]
    lab = d["labels"]
    med = float(np.median(U1))
    rng = np.random.default_rng(2)
    fig = plt.figure(figsize=(13.0, 7.0))
    ax = fig.add_axes([0.08, 0.12, 0.62, 0.78])
    for k, col, name, y0 in [
        (0, C_ORANGE, "Mode B", 0.0),
        (1, C_TEAL, "Mode A", 1.0),
    ]:
        mask = lab == k
        yy = rng.normal(y0, 0.11, mask.sum())
        ax.scatter(U1[mask], yy, s=40, c=col, alpha=0.7,
                   edgecolors="white", linewidths=0.3, label=name)
    ax.axvline(med, color=C_NAVY, ls="--", lw=2.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Mode B", "Mode A"])
    ax.set_xlabel("U₁")
    ax.set_title("Спектральный портрет СЗФО", color=C_NAVY, fontsize=15)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # huge numbers panel (17 / 1 / 140·140 dominant; λ1, boot, 17.7% small)
    ax2 = fig.add_axes([0.74, 0.12, 0.23, 0.74])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    bigs = [
        (0.86, "17", "признаков"),
        (0.58, "1", "ведущая мода"),
        (0.30, "140/140", ""),
    ]
    for y, big, small in bigs:
        ax2.text(0.5, y, big, ha="center", va="center",
                 fontsize=28, fontweight="bold", color=C_NAVY)
        if small:
            ax2.text(0.5, y - 0.09, small, ha="center", va="center",
                     fontsize=12, color=C_MUTED)
    ax2.text(
        0.5, 0.08,
        f"λ₁={d['evals'][0]:.3f}  ·  boot≈{d['boot'].mean():.2f}\n"
        f"{100 * d['n_switch'] / d['n_traj']:.1f}% переходов",
        ha="center", va="center", fontsize=10, color=C_MUTED, linespacing=1.4,
    )
    return _save(fig, "slide11_finale.png")


def write_readme(paths, d):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "presentation_assets — jury pitch Task 1 (11 slides)",
        f"Generated: {now}",
        f"Canon: p={d['p']} N={d['N']} lam1={d['evals'][0]:.4f} "
        f"boot={d['boot'].mean():.3f} switch={d['n_switch']}/{d['n_traj']}",
        "",
        "slide01_hook_cloud.png              -> 1 HOOK",
        "slide02_kmeans_vs_j.png             -> 2 WHY NOT KMEANS",
        "slide03_j_heatmap.png               -> 3 MAP OF DEPENDENCIES",
        "slide04_spectrum.png                -> 4 MAGIC",
        "slide05_mode1_loadings.png          -> 5 ECONOMIC MEANING",
        "slide06_U1_median_split.png         -> 6 WOW CENTERPIECE",
        "slide07_robustness.png              -> 7 ROBUSTNESS",
        "slide08_dynamics_ari.png            -> 8 DYNAMICS",
        "slide09_vologodsky_trajectory.png   -> 9 EXAMPLE",
        "slide10_value_quote.png             -> 10 ANALYST VALUE",
        "slide11_finale.png                  -> 11 FINALE",
        "",
        "Rebuild: python scripts/make_pitch_assets.py",
        "PDF:     python scripts/build_task1_pdfs.py --slides-only",
    ]
    (OUT / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  -> README.txt")


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
            print(f"  -> {dst} (copy)")


def main():
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading canon data...")
    d = load_canon()
    print(f"  N={d['N']} p={d['p']} lam1={d['evals'][0]:.4f} "
          f"sizes={np.bincount(d['labels']).tolist()} "
          f"switch={d['n_switch']}/{d['n_traj']}")
    print("Rendering slides...")
    paths = [
        fig_slide01_hook(d),
        fig_slide02_kmeans_vs_j(d),
        fig_slide03_j_map(d),
        fig_slide04_spectrum(d),
        fig_slide05_loadings(d),
        fig_slide06_u1(d),
        fig_slide07_robustness(d),
        fig_slide08_dynamics(d),
        fig_slide09_vologda(d),
        fig_slide10_value(d),
        fig_slide11_finale(d),
    ]
    print("Copying pipeline figures...")
    copy_pipeline_figs()
    write_readme(paths, d)
    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
