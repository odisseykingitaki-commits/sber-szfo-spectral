"""Сборка PDF Task 1: методологический отчёт + презентация (reportlab).

Запуск из корня репозитория:
    python scripts/build_task1_pdfs.py

Требует: reportlab, Pillow. Шрифты Windows (Arial) для кириллицы.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES = ROOT / "figures"
ASSETS = DOCS / "presentation_assets"
OUT_REPORT = DOCS / "TASK1_method_report.pdf"
OUT_SLIDES = DOCS / "TASK1_presentation.pdf"
MD_REPORT = DOCS / "TASK1_METHOD_REPORT.md"
MD_SLIDES = DOCS / "TASK1_slides.md"
REPO_URL = "https://github.com/odisseykingitaki-commits/sber-szfo-spectral"

FIG_NAMES = [
    "fig_final.png",
    "fig_network.png",
    "fig_dynamic.png",
    "fig_bootstrap.png",
]

# 0-based slide index → pitch visual (see scripts/make_pitch_assets.py)
SLIDE_ASSETS = [
    ASSETS / "slide01_title_card.png",
    ASSETS / "slide02_hook_scatter.png",
    ASSETS / "slide03_kmeans_vs_spectral.png",
    ASSETS / "slide04_pipeline_flowchart.png",
    ASSETS / "slide05_mode1_loadings.png",
    ASSETS / "slide06_U1_median_split.png",
    ASSETS / "slide07_stability_metric_cards.png",
    ASSETS / "slide08_dynamics_ari.png",
    ASSETS / "slide08b_vologodsky_trajectory.png",  # slide 9 — Vologda case
    ASSETS / "slide09_value_blocks.png",
    ASSETS / "slide10_takeaways_checklist.png",
]


def _find_font_paths():
    windir = Path(r"C:\Windows\Fonts")
    candidates = [
        (windir / "arial.ttf", windir / "arialbd.ttf"),
        (windir / "Arial.ttf", windir / "Arialbd.ttf"),
        (windir / "times.ttf", windir / "timesbd.ttf"),
        (windir / "calibri.ttf", windir / "calibrib.ttf"),
        (windir / "DejaVuSans.ttf", windir / "DejaVuSans-Bold.ttf"),
    ]
    for reg, bold in candidates:
        if reg.exists():
            return str(reg), str(bold if bold.exists() else reg)
    raise FileNotFoundError(
        "Не найден TTF с кириллицей (Arial/Times/Calibri/DejaVu). "
        "Установите шрифт или поправьте пути в scripts/build_task1_pdfs.py"
    )


def _register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    reg, bold = _find_font_paths()
    pdfmetrics.registerFont(TTFont("Body", reg))
    pdfmetrics.registerFont(TTFont("BodyBold", bold))
    return "Body", "BodyBold"


def _strip_md(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[рис.: \1]", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "")
    text = re.sub(r"^#+\s*", "", text)
    text = text.replace("---", "")
    # упростить LaTeX-подобные куски
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\mathrm{reg}", "reg")
    text = text.replace(r"\mathrm", "")
    text = text.replace(r"\approx", "≈")
    text = text.replace(r"\lambda", "λ")
    text = text.replace(r"\Sigma", "Σ")
    text = text.replace(r"\mathbf{1}", "1")
    text = text.replace(r"\bigl", "").replace(r"\bigr", "")
    text = text.replace(r"\,", " ")
    text = text.replace("$", "")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    return text


def build_report_pdf():
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    font, font_b = _register_fonts()
    md = MD_REPORT.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(OUT_REPORT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Task 1 — Методологический отчёт",
        author="sber-project",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="H1R", fontName=font_b, fontSize=14, leading=18,
        spaceAfter=8, spaceBefore=12, textColor=colors.HexColor("#1a1a1a"),
    ))
    styles.add(ParagraphStyle(
        name="H2R", fontName=font_b, fontSize=12, leading=15,
        spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#222222"),
    ))
    styles.add(ParagraphStyle(
        name="H3R", fontName=font_b, fontSize=10.5, leading=13,
        spaceAfter=4, spaceBefore=8, textColor=colors.HexColor("#333333"),
    ))
    styles.add(ParagraphStyle(
        name="BodyR", fontName=font, fontSize=9.5, leading=13,
        alignment=TA_JUSTIFY, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BulletR", fontName=font, fontSize=9.5, leading=13,
        leftIndent=12, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="MetaR", fontName=font, fontSize=8.5, leading=11,
        textColor=colors.HexColor("#444444"), spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="CodeR", fontName=font, fontSize=8, leading=10,
        backColor=colors.HexColor("#f4f4f4"), leftIndent=6, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CaptionR", fontName=font, fontSize=8, leading=10,
        alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
        spaceAfter=10, spaceBefore=2,
    ))
    styles.add(ParagraphStyle(
        name="TitleR", fontName=font_b, fontSize=16, leading=20,
        alignment=TA_CENTER, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="CellR", fontName=font, fontSize=8, leading=10,
    ))

    story = []
    in_code = False
    code_buf: list[str] = []
    table_buf: list[list[str]] = []

    def flush_table():
        nonlocal table_buf
        if not table_buf:
            return
        rows = table_buf
        table_buf = []
        # drop markdown separator rows
        data = []
        for row in rows:
            if all(re.match(r"^:?-+:?$", c.strip()) for c in row):
                continue
            data.append([Paragraph(_strip_md(c).replace("|", ""), styles["CellR"]) for c in row])
        if not data:
            return
        ncols = max(len(r) for r in data)
        for r in data:
            while len(r) < ncols:
                r.append(Paragraph("", styles["CellR"]))
        tw = doc.width
        col_w = [tw / ncols] * ncols
        t = Table(data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
            ("FONTNAME", (0, 0), (-1, 0), font_b),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#99a")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))

    def flush_code():
        nonlocal code_buf, in_code
        if not code_buf:
            in_code = False
            return
        text = "\n".join(code_buf)
        story.append(Preformatted(text, styles["CodeR"]))
        code_buf = []
        in_code = False

    lines = md.splitlines()
    first_h1 = True
    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
            else:
                flush_table()
                in_code = True
                code_buf = []
            continue
        if in_code:
            code_buf.append(line)
            continue

        # images
        m_img = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if m_img:
            flush_table()
            alt, rel = m_img.group(1), m_img.group(2)
            img_path = (DOCS / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            if not img_path.exists():
                # try figures/
                name = Path(rel).name
                img_path = FIGURES / name
            if img_path.exists():
                iw = doc.width
                img = Image(str(img_path), width=iw, height=iw * 0.55)
                img.hAlign = "CENTER"
                story.append(KeepTogether([img, Paragraph(alt or img_path.name, styles["CaptionR"])]))
            else:
                story.append(Paragraph(f"[рисунок не найден: {rel}]", styles["MetaR"]))
            continue

        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_buf.append(cells)
            continue
        else:
            flush_table()

        if not line.strip():
            story.append(Spacer(1, 3))
            continue

        if line.startswith("# "):
            flush_table()
            txt = _strip_md(line[2:])
            if first_h1:
                story.append(Paragraph(txt, styles["TitleR"]))
                first_h1 = False
            else:
                story.append(Paragraph(txt, styles["H1R"]))
            continue
        if line.startswith("## "):
            story.append(Paragraph(_strip_md(line[3:]), styles["H2R"]))
            continue
        if line.startswith("### "):
            story.append(Paragraph(_strip_md(line[4:]), styles["H3R"]))
            continue
        if line.strip() in ("---", "***"):
            continue
        if line.lstrip().startswith("- ") or line.lstrip().startswith("* "):
            story.append(Paragraph("• " + _strip_md(line.lstrip()[2:]), styles["BulletR"]))
            continue
        if re.match(r"^\d+\.\s", line.lstrip()):
            story.append(Paragraph(_strip_md(line.lstrip()), styles["BulletR"]))
            continue

        # meta / normal
        style = styles["MetaR"] if line.startswith("**Аудитория") or line.startswith("**Дата") or line.startswith("**Репозиторий") or line.startswith("**Канонические") else styles["BodyR"]
        story.append(Paragraph(_strip_md(line), style))

    flush_table()
    flush_code()

    # ensure figures section even if md paths failed: append any missing
    existing_caps = " ".join(
        getattr(x, "text", "") if hasattr(x, "text") else "" for x in story
    )
    for name in FIG_NAMES:
        p = FIGURES / name
        if p.exists() and name not in existing_caps:
            # already embedded via md; skip duplicate check via filename in flow
            pass

    doc.build(story)
    print(f"OK report -> {OUT_REPORT}")


def build_slides_pdf():
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader

    font, font_b = _register_fonts()
    md = MD_SLIDES.read_text(encoding="utf-8")

    # split by "## Слайд"
    parts = re.split(r"\n## Слайд\s+", md)
    slides = []
    for part in parts[1:]:
        lines = part.strip().splitlines()
        if not lines:
            continue
        title = _strip_md(lines[0])
        # drop leading "N — "
        title = re.sub(r"^\d+\s*[—\-]\s*", "", title)
        body_lines = []
        for ln in lines[1:]:
            if ln.strip() in ("---",):
                continue
            if ln.startswith("#"):
                continue
            body_lines.append(ln)
        slides.append((title, body_lines))

    page = landscape(A4)
    c = canvas.Canvas(str(OUT_SLIDES), pagesize=page)
    W, H = page
    margin_x = 1.2 * cm
    footer_h = 0.9 * cm
    header_h = 1.4 * cm

    for i, (title, body_lines) in enumerate(slides):
        # background
        c.setFillColor(colors.HexColor("#f7f5f1"))
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#1e3a5f"))
        c.rect(0, H - header_h, W, header_h, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont(font_b, 15)
        c.drawString(margin_x, H - 0.9 * cm, title[:95])

        c.setFillColor(colors.HexColor("#666666"))
        c.setFont(font, 7)
        c.drawRightString(
            W - 1.0 * cm, 0.4 * cm,
            f"Task 1 · {i + 1}/{len(slides)} · {REPO_URL}",
        )

        fig = SLIDE_ASSETS[i] if i < len(SLIDE_ASSETS) else None
        show_fig = bool(fig and fig.exists())
        # Stacked layout: short bullets on top, full-width image below (no side float)
        # Image gets ~58% of page height when present
        text_bottom_limit = (H * 0.55) if show_fig else footer_h + 0.4 * cm
        max_text_w = W - 2 * margin_x
        y = H - header_h - 0.55 * cm
        c.setFillColor(colors.HexColor("#222222"))
        bullet_count = 0
        max_bullets = 4 if show_fig else 12

        for ln in body_lines:
            if y < text_bottom_limit + 0.3 * cm:
                break
            if not ln.strip():
                y -= 0.18 * cm
                continue
            # skip markdown figure hints (embedded via SLIDE_ASSETS)
            if "presentation_assets/" in ln or "figures/" in ln or ln.strip().startswith("**Рисунок"):
                continue
            if ln.strip().startswith("|"):
                if show_fig:
                    continue  # tables live in PNG assets
                row = _strip_md(ln.strip())
                c.setFont(font, 9)
                c.drawString(margin_x, y, row[:120])
                y -= 0.42 * cm
                continue
            text = _strip_md(ln.strip())
            if text.startswith("- "):
                text = "• " + text[2:]
            # Prefer bullets; allow a few non-bullet lines (title slide headlines)
            if text.startswith("•"):
                bullet_count += 1
                if bullet_count > max_bullets:
                    continue
            elif show_fig and i != 0 and len(text) > 70:
                continue
            size = 12 if i == 0 else 10.5
            use_bold = i == 0 and not text.startswith("•")
            c.setFont(font_b if use_bold else font, size)
            words = text.split()
            line = ""
            for w in words:
                trial = (line + " " + w).strip()
                if c.stringWidth(trial, font_b if use_bold else font, size) < max_text_w:
                    line = trial
                else:
                    c.drawString(margin_x, y, line)
                    y -= 0.48 * cm
                    line = w
                    if y < text_bottom_limit + 0.3 * cm:
                        break
            if line and y >= text_bottom_limit + 0.3 * cm:
                c.drawString(margin_x, y, line)
                y -= 0.48 * cm

        if show_fig:
            try:
                img = ImageReader(str(fig))
                nat_w, nat_h = img.getSize()
                # Image band: from just below text floor down to above footer
                band_top = text_bottom_limit - 0.15 * cm
                band_bottom = footer_h + 0.15 * cm
                band_h = max(band_top - band_bottom, 4 * cm)
                band_w = W - 2 * margin_x
                # Contain: max size inside band, centered (wider than old side-float)
                scale = min(band_w / nat_w, band_h / nat_h)
                iw, ih = nat_w * scale, nat_h * scale
                x_img = (W - iw) / 2
                y_img = band_bottom + (band_h - ih) / 2
                c.drawImage(
                    img, x_img, y_img, width=iw, height=ih,
                    preserveAspectRatio=True, anchor="c", mask="auto",
                )
            except Exception as e:
                c.setFont(font, 8)
                c.drawString(margin_x, footer_h + 0.5 * cm, f"[fig err: {e}]")

        c.showPage()

    c.save()
    print(f"OK slides -> {OUT_SLIDES}")


def main():
    if not MD_REPORT.exists():
        print(f"Missing {MD_REPORT}", file=sys.stderr)
        sys.exit(1)
    build_report_pdf()
    build_slides_pdf()
    print("Done.")


if __name__ == "__main__":
    main()
