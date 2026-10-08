"""Build chart-first jury PDF: для_жюри/TASK1_presentation.pdf (11 landscape slides).

Layout: conclusion headline → 2–4 short phrases → dominant chart (50–70%) → tiny footer.
Does NOT dump the method report into slides.

    python scripts/build_jury_slides.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JURY = ROOT / "для_жюри"
ASSETS = ROOT / "scripts" / "_pitch" / "presentation_assets"
OUT = JURY / "TASK1_presentation.pdf"
REPO_URL = "https://github.com/odisseykingitaki-commits/sber-szfo-spectral"

C_BG = "#f4f6f8"
C_NAVY = "#1e3a5f"
C_INK = "#1a2332"
C_MUTED = "#6b7280"
C_TEAL = "#2a6f6a"


def _fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    windir = Path(r"C:\Windows\Fonts")
    for reg, bold in [
        (windir / "arial.ttf", windir / "arialbd.ttf"),
        (windir / "Arial.ttf", windir / "Arialbd.ttf"),
        (windir / "calibri.ttf", windir / "calibrib.ttf"),
    ]:
        if reg.exists():
            pdfmetrics.registerFont(TTFont("Body", str(reg)))
            pdfmetrics.registerFont(TTFont("BodyBold", str(bold if bold.exists() else reg)))
            return "Body", "BodyBold"
    raise FileNotFoundError("Need Arial/Calibri TTF with Cyrillic")


def _metrics():
    summ = json.loads((ROOT / "results" / "final_no_income_summary.json").read_text(encoding="utf-8"))
    evals = np.load(ROOT / "results" / "evals_v2.npy")
    if evals[0] < evals[-1]:
        evals = np.sort(evals)[::-1]
    pos = evals[evals > 0]
    lam_frac = float(evals[0] / pos.sum())
    boot = pd.read_csv(ROOT / "results" / "bootstrap_stability.csv")["ari"].values
    traj = pd.read_csv(ROOT / "data" / "processed" / "trajectories.csv", index_col=0)
    n_sw = int((traj["n_switches"] > 0).sum())
    n_tr = int(len(traj))
    dyn = json.loads((ROOT / "results" / "dynamic_summary.json").read_text(encoding="utf-8"))
    mean_ari = float(np.mean([a["ari"] for a in dyn["aris"]]))
    return {
        "lam1": float(evals[0]),
        "lam_frac": lam_frac,
        "boot_m": float(boot.mean()),
        "boot_s": float(boot.std()),
        "rob_min": float(summ["robustness_min_ARI"]),
        "ari_louv": float(summ["ARI_vs_Louvain"]),
        "n_sw": n_sw,
        "n_tr": n_tr,
        "pct_sw": 100.0 * n_sw / n_tr,
        "mean_ari": mean_ari,
        "n_win": len(dyn["windows"]),
    }


def slides_spec(m: dict):
    """title, phrases (max 4), asset, footer_extra"""
    return [
        {
            "title": "280 территорий. 17 признаков. Одна скрытая структура?",
            "phrases": [
                "Не только чем МО отличаются —",
                "какая общая структура связывает их потребление.",
            ],
            "fig": "slide01_hook_cloud.png",
            "footer": "СЗФО · 280 МО · СберИндекс 2023–2024",
            "fig_frac": 0.72,
        },
        {
            "title": "KMeans ищет похожих. Мы ищем то, что связывает систему.",
            "phrases": [
                "Не только «кто с кем похож».",
                "А «что связывает систему».",
            ],
            "fig": "slide02_kmeans_vs_j.png",
            "footer": "слева — похожесть объектов · справа — матрица связей J",
            "fig_frac": 0.70,
        },
        {
            "title": "Сначала строим карту зависимостей",
            "phrases": [
                "PLM оценивает связи признаков",
                "→ симметричная матрица J",
                "→ готова к спектральному разложению",
            ],
            "fig": "slide03_j_heatmap.png",
            "footer": "",
            "fig_frac": 0.68,
        },
        {
            "title": "Из этой структуры появляется главная ось поведения",
            "phrases": [
                f"λ₁ = {m['lam1']:.3f} · {100 * m['lam_frac']:.1f}% положительной спектральной массы",
                "Первая мода доминирует.",
            ],
            "fig": "slide04_spectrum.png",
            "footer": "",
            "fig_frac": 0.70,
        },
        {
            "title": "Главная мода говорит на языке экономики",
            "phrases": [
                "Food +0.391 · Grocery −0.374 · Transport +0.334",
                "Log total +0.327 · Health +0.279",
                "Два полюса — паттерны потребления, не юр. статус МО.",
            ],
            "fig": "slide05_mode1_loadings.png",
            "footer": "",
            "fig_frac": 0.68,
        },
        {
            "title": "Одна ось превращает 17 признаков в два режима",
            "phrases": [
                "U₁ = X · v₁ · порог = медиана → 140 / 140",
                "Типология строится по главной коллективной оси.",
            ],
            "fig": "slide06_U1_median_split.png",
            "footer": "не заранее выбранные 2 кластера",
            "fig_frac": 0.72,
        },
        {
            "title": "Структура не исчезает, если изменить настройки",
            "phrases": [
                f"Bootstrap {m['boot_m']:.3f}±{m['boot_s']:.3f} · C_reg min ARI {m['rob_min']:.3f}",
                f"Louvain ARI≈{m['ari_louv']:.3f} · динамика {m['pct_sw']:.1f}% сменили режим",
                "Разные проверки показывают одну крупномасштабную структуру.",
            ],
            "fig": "slide07_robustness.png",
            "footer": "",
            "fig_frac": 0.66,
        },
        {
            "title": "Структура меняется — но не хаотично",
            "phrases": [
                f"{m['n_win']} окон × 6 месяцев · {m['n_sw']}/{m['n_tr']} = {m['pct_sw']:.1f}% сменили режим",
                f"Система живая, но крупная структура сохраняется (mean ARI ≈ {m['mean_ari']:.2f}).",
            ],
            "fig": "slide08_dynamics_ari.png",
            "footer": "",
            "fig_frac": 0.72,
        },
        {
            "title": "Некоторые МО действительно переходят между режимами",
            "phrases": [
                "вологодский · 8 переключений / 19 окон",
                "Часть МО у границы двух режимов.",
            ],
            "fig": "slide09_vologodsky_trajectory.png",
            "footer": "метка режима = threshold-тип, не юр. статус",
            "fig_frac": 0.72,
        },
        {
            "title": "Из 17 показателей — к карте экономических режимов",
            "phrases": [
                "сегментация · интерпретация · мониторинг · воспроизводимость",
            ],
            "fig": "slide10_value_quote.png",
            "footer": "",
            "fig_frac": 0.78,
            "quote_mode": True,
        },
        {
            "title": "Спектральный портрет СЗФО",
            "phrases": [
                "17 признаков · 1 ведущая мода · 140 / 140",
                "Мы не просто кластеризовали территорию.",
                "Мы нашли коллективную ось их поведения.",
            ],
            "fig": "slide11_finale.png",
            "footer": "метод и pipeline в репозитории",
            "fig_frac": 0.68,
        },
    ]


def build():
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader

    font, font_b = _fonts()
    m = _metrics()
    specs = slides_spec(m)
    page = landscape(A4)
    c = canvas.Canvas(str(OUT), pagesize=page)
    W, H = page
    mx = 1.15 * cm
    header_h = 1.55 * cm
    footer_h = 0.75 * cm

    for i, s in enumerate(specs):
        # background
        c.setFillColor(colors.HexColor(C_BG))
        c.rect(0, 0, W, H, fill=1, stroke=0)

        # navy header band
        c.setFillColor(colors.HexColor(C_NAVY))
        c.rect(0, H - header_h, W, header_h, fill=1, stroke=0)

        # title (conclusion) — wrap if needed
        c.setFillColor(colors.white)
        title = s["title"]
        size = 16 if len(title) < 70 else 14
        c.setFont(font_b, size)
        max_w = W - 2 * mx
        if c.stringWidth(title, font_b, size) <= max_w:
            c.drawString(mx, H - 1.0 * cm, title)
        else:
            # two-line wrap at nearest space mid
            words = title.split()
            line1, line2 = "", ""
            for w in words:
                trial = (line1 + " " + w).strip()
                if c.stringWidth(trial, font_b, size - 1) < max_w:
                    line1 = trial
                else:
                    line2 = (line2 + " " + w).strip()
            c.setFont(font_b, size - 1)
            c.drawString(mx, H - 0.75 * cm, line1)
            if line2:
                c.drawString(mx, H - 1.25 * cm, line2)

        # phrases
        y = H - header_h - 0.45 * cm
        c.setFillColor(colors.HexColor(C_INK))
        for ph in s["phrases"][:4]:
            c.setFont(font, 12)
            c.drawString(mx, y, ph)
            y -= 0.48 * cm

        # figure band — fill nearly all remaining space (chart-first)
        fig_path = ASSETS / s["fig"]
        band_top = y - 0.12 * cm
        band_bottom = footer_h + 0.28 * cm
        band_h = max(band_top - band_bottom, 3.5 * cm)
        band_w = W - 2 * mx

        if fig_path.exists():
            img = ImageReader(str(fig_path))
            nw, nh = img.getSize()
            scale = min(band_w / nw, band_h / nh)
            iw, ih = nw * scale, nh * scale
            x_img = (W - iw) / 2
            y_img = band_bottom + (band_h - ih) / 2
            c.drawImage(img, x_img, y_img, width=iw, height=ih,
                        preserveAspectRatio=True, mask="auto")
        else:
            c.setFillColor(colors.HexColor("#aa0000"))
            c.setFont(font, 10)
            c.drawString(mx, band_bottom + band_h / 2, f"[missing {s['fig']}]")

        # footer + teal/orange accent
        c.setFillColor(colors.HexColor(C_MUTED))
        c.setFont(font, 7.5)
        left = s.get("footer") or ""
        right = f"Task 1 · {i + 1}/11 · {REPO_URL}"
        # keep left footer short; full URL always on the right
        if left and "http" not in left:
            c.drawString(mx, 0.32 * cm, left[:100])
        elif left and "http" in left:
            c.drawString(mx, 0.32 * cm, "метод и pipeline в репозитории")
        c.drawRightString(W - mx, 0.32 * cm, right)
        c.setFillColor(colors.HexColor(C_TEAL))
        c.rect(0, 0, W * 0.55, 0.1 * cm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#c45c26"))
        c.rect(W * 0.55, 0, W * 0.45, 0.1 * cm, fill=1, stroke=0)

        c.showPage()

    c.save()
    print(f"OK slides -> {OUT}")


if __name__ == "__main__":
    build()
