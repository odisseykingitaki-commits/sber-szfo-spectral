"""Сборка PDF Task 2: методологический отчёт + презентация (reportlab).

Запуск из корня репозитория:
    python scripts/build_task2_pdfs.py

Требует: reportlab, Pillow. Шрифты Windows (Arial) для кириллицы.
Перед сборкой желательно:
    python scripts/make_task2_pitch_assets.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES = ROOT / "figures"
ASSETS = DOCS / "presentation_assets_task2"
OUT_REPORT = DOCS / "TASK2_method_report.pdf"
OUT_SLIDES = DOCS / "TASK2_presentation.pdf"
MD_REPORT = DOCS / "TASK2_METHOD_REPORT.md"
MD_SLIDES = DOCS / "TASK2_slides.md"

SLIDE_ASSETS = [
    ASSETS / "slide01_title_card.png",
    ASSETS / "slide02_two_levels.png",
    ASSETS / "slide03_architecture.png",
    ASSETS / "slide04_leakage_note.png",
    ASSETS / "slide05_mae_heatmap.png",
    ASSETS / "slide06_winner_map.png",
    ASSETS / "slide07_rural_h3_case.png",
    ASSETS / "slide08_changepoint_consensus.png",
    ASSETS / "slide09_cp_example.png",
    ASSETS / "slide10_trends_not_news.png",
    ASSETS / "slide11_gaps.png",
    ASSETS / "slide12_takeaways.png",
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
        "Установите шрифт или поправьте пути в scripts/build_task2_pdfs.py"
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
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = text.replace(r"\approx", "≈")
    text = text.replace(r"\lambda", "λ")
    text = text.replace(r"\Sigma", "Σ")
    text = text.replace("$", "")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    return text


def build_report_pdf():
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        KeepTogether,
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
        title="Task 2 — Методологический отчёт",
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
        name="CellR", fontName=font, fontSize=7.5, leading=9.5,
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
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
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

        m_img = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if m_img:
            flush_table()
            alt, rel = m_img.group(1), m_img.group(2)
            img_path = (DOCS / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            if not img_path.exists():
                img_path = FIGURES / Path(rel).name
            if not img_path.exists():
                img_path = ASSETS / Path(rel).name
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

        style = styles["MetaR"] if line.startswith("**") and ":" in line[:40] else styles["BodyR"]
        story.append(Paragraph(_strip_md(line), style))

    flush_table()
    flush_code()

    # append key pitch figures at end if present
    for name in [
        "slide05_mae_heatmap.png",
        "slide06_winner_map.png",
        "slide08_changepoint_consensus.png",
        "slide03_architecture.png",
    ]:
        p = ASSETS / name
        if p.exists():
            iw = doc.width
            img = Image(str(p), width=iw, height=iw * 0.52)
            img.hAlign = "CENTER"
            story.append(Spacer(1, 8))
            story.append(KeepTogether([
                img,
                Paragraph(name, styles["CaptionR"]),
            ]))

    doc.build(story)
    print(f"OK report -> {OUT_REPORT}")


def build_slides_pdf():
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors

    font, font_b = _register_fonts()
    md = MD_SLIDES.read_text(encoding="utf-8")

    parts = re.split(r"\n## Слайд\s+", md)
    slides = []
    for part in parts[1:]:
        lines = part.strip().splitlines()
        if not lines:
            continue
        title = _strip_md(lines[0])
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

    for i, (title, body_lines) in enumerate(slides):
        c.setFillColor(colors.HexColor("#f7f5f1"))
        c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#1e3a5f"))
        c.rect(0, H - 1.4 * cm, W, 1.4 * cm, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont(font_b, 15)
        c.drawString(1.2 * cm, H - 0.9 * cm, title[:95])

        c.setFillColor(colors.HexColor("#666666"))
        c.setFont(font, 8)
        c.drawRightString(
            W - 1.0 * cm, 0.5 * cm,
            f"Task 2 · {i + 1}/{len(slides)} · репозиторий: локально / будет на GitHub",
        )

        y = H - 2.2 * cm
        c.setFillColor(colors.HexColor("#222222"))
        max_text_w = W - 2.4 * cm
        fig = SLIDE_ASSETS[i] if i < len(SLIDE_ASSETS) else None
        show_fig = bool(fig and fig.exists())
        last_i = len(slides) - 1
        if show_fig and i in (0, last_i):
            max_text_w = W * 0.42
        elif show_fig:
            max_text_w = W * 0.48

        for ln in body_lines:
            if not ln.strip():
                y -= 0.25 * cm
                continue
            if "presentation_assets" in ln or "figures/" in ln or ln.strip().startswith("**Рисунок"):
                continue
            if ln.strip().startswith("|"):
                row = _strip_md(ln.strip())
                c.setFont(font, 8.5)
                c.drawString(1.2 * cm, y, row[:115])
                y -= 0.42 * cm
                if y < 1.5 * cm:
                    break
                continue
            text = _strip_md(ln.strip())
            if text.startswith("- "):
                text = "• " + text[2:]
            size = 13 if i == 0 else 10.5
            use_bold = i == 0 and not text.startswith("•")
            c.setFont(font_b if use_bold else font, size)
            words = text.split()
            line = ""
            for w in words:
                trial = (line + " " + w).strip()
                if c.stringWidth(trial, font_b if use_bold else font, size) < max_text_w:
                    line = trial
                else:
                    c.drawString(1.2 * cm, y, line)
                    y -= 0.52 * cm
                    line = w
                    if y < 1.5 * cm:
                        break
            if line and y >= 1.5 * cm:
                c.drawString(1.2 * cm, y, line)
                y -= 0.52 * cm
            if y < 1.5 * cm:
                break

        if show_fig:
            try:
                from reportlab.lib.utils import ImageReader
                iw, ih = (11.5 * cm, 7.2 * cm) if i in (0, last_i) else (10.0 * cm, 7.0 * cm)
                c.drawImage(
                    ImageReader(str(fig)),
                    W - iw - 0.8 * cm,
                    1.3 * cm,
                    width=iw,
                    height=ih,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception as e:
                c.setFont(font, 8)
                c.drawString(W - 10 * cm, 2 * cm, f"[fig err: {e}]")

        c.showPage()

    c.save()
    print(f"OK slides -> {OUT_SLIDES}")


def main():
    if not MD_REPORT.exists():
        print(f"Missing {MD_REPORT}", file=sys.stderr)
        sys.exit(1)
    if not MD_SLIDES.exists():
        print(f"Missing {MD_SLIDES}", file=sys.stderr)
        sys.exit(1)
    build_report_pdf()
    build_slides_pdf()
    print("Done.")


if __name__ == "__main__":
    main()
