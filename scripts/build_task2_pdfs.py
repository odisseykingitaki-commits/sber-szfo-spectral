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

# Lean DeepSeek/user deck (9 slides). Architecture folded into idea;
# MAE grid skipped (winners table only); CP consensus + PR₊ on one slide.
SLIDE_ASSETS = [
    ASSETS / "slide01_title_hook.png",
    ASSETS / "slide02_idea.png",
    ASSETS / "slide03_leakage.png",
    ASSETS / "slide04_mae_winners.png",
    ASSETS / "slide05_rural_h3.png",
    ASSETS / "slide06_changepoints.png",
    ASSETS / "slide07_external.png",
    ASSETS / "slide08_chronos.png",
    ASSETS / "slide09_final.png",
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
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
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
    styles.add(ParagraphStyle(
        name="CellSmallR", fontName=font, fontSize=6.2, leading=7.8,
    ))
    styles.add(ParagraphStyle(
        name="ArchCodeR", fontName=font, fontSize=6.5, leading=8.0,
        backColor=colors.HexColor("#f4f4f4"), leftIndent=2, spaceAfter=6,
    ))

    story = []
    in_code = False
    code_buf: list[str] = []
    table_buf: list[list[str]] = []
    pending_h2: str | None = None  # hold H2 until we know if next block is architecture
    pending_subhead = None  # H3/H4 held to KeepTogether with following table

    def flush_table():
        nonlocal table_buf, pending_subhead
        if not table_buf:
            return
        rows = table_buf
        table_buf = []
        data = []
        for row in rows:
            if all(re.match(r"^:?-+:?$", c.strip()) for c in row):
                continue
            data.append(row)
        if not data:
            return
        ncols = max(len(r) for r in data)
        # Wide MAE tables (many model columns): smaller font, keep on one page
        wide = ncols >= 8
        cell_style = styles["CellSmallR"] if wide else styles["CellR"]
        rendered = []
        for row in data:
            cells = [Paragraph(_strip_md(c).replace("|", ""), cell_style) for c in row]
            while len(cells) < ncols:
                cells.append(Paragraph("", cell_style))
            rendered.append(cells)
        tw = doc.width
        if wide:
            # Narrow H / winner cols; share remaining width among numeric models
            col_w = []
            for i in range(ncols):
                header = _strip_md(data[0][i]).lower() if i < len(data[0]) else ""
                if i == 0 or "winner" in header or header in ("h",):
                    col_w.append(tw * 0.07)
                else:
                    col_w.append(tw * 0.86 / max(ncols - 2, 1))
            # normalize
            s = sum(col_w)
            col_w = [w * tw / s for w in col_w]
        else:
            col_w = [tw / ncols] * ncols
        t = Table(rendered, colWidths=col_w, repeatRows=0)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
            ("FONTNAME", (0, 0), (-1, 0), font_b),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#99a")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 1 if wide else 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1 if wide else 2),
            ("TOPPADDING", (0, 0), (-1, -1), 1 if wide else 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1 if wide else 2),
            ("NOSPLIT", (0, 0), (-1, -1)),
        ]))
        block = []
        if pending_subhead is not None:
            block.append(pending_subhead)
            pending_subhead = None
        block.extend([t, Spacer(1, 6)])
        story.append(KeepTogether(block))

    def flush_code():
        nonlocal code_buf, in_code, pending_h2
        if not code_buf:
            in_code = False
            return
        text = "\n".join(code_buf)
        is_arch = ("УРОВЕНЬ A" in text) or ("УРОВЕНЬ B" in text) or ("Подготовка (один раз)" in text)
        style = styles["ArchCodeR"] if is_arch else styles["CodeR"]
        block = [Preformatted(text, style)]
        if pending_h2 is not None:
            # Architecture section: keep H2 + diagram on one page
            story.append(PageBreak())
            block = [Paragraph(pending_h2, styles["H2R"])] + block
            pending_h2 = None
            story.append(KeepTogether(block))
        else:
            story.append(KeepTogether(block) if is_arch else Preformatted(text, style))
        code_buf = []
        in_code = False

    def flush_pending_h2():
        nonlocal pending_h2
        if pending_h2 is not None:
            story.append(Paragraph(pending_h2, styles["H2R"]))
            pending_h2 = None

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
            flush_pending_h2()
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
            # If we held an H2 waiting for code, flush it — tables are not architecture
            flush_pending_h2()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_buf.append(cells)
            continue
        else:
            flush_table()

        if not line.strip():
            if pending_h2 is None:
                story.append(Spacer(1, 3))
            continue

        if line.startswith("# "):
            flush_pending_h2()
            flush_table()
            txt = _strip_md(line[2:])
            if first_h1:
                story.append(Paragraph(txt, styles["TitleR"]))
                first_h1 = False
            else:
                story.append(Paragraph(txt, styles["H1R"]))
            continue
        if line.startswith("## "):
            flush_pending_h2()
            if pending_subhead is not None:
                story.append(pending_subhead)
                pending_subhead = None
            h2_txt = _strip_md(line[3:])
            # Defer architecture H2 so it can KeepTogether with the ASCII diagram
            if "Архитектура" in h2_txt:
                pending_h2 = h2_txt
            elif h2_txt.startswith("4."):
                # Fresh page before forecast/MAE section so first MAE table fits
                story.append(PageBreak())
                story.append(Paragraph(h2_txt, styles["H2R"]))
            else:
                story.append(Paragraph(h2_txt, styles["H2R"]))
            continue
        if line.startswith("### ") or line.startswith("#### "):
            flush_pending_h2()
            if pending_subhead is not None:
                story.append(pending_subhead)
            level = 4 if line.startswith("#### ") else 3
            txt = _strip_md(line[level + 1 :])
            pending_subhead = Paragraph(txt, styles["H3R"])
            continue
        if line.strip() in ("---", "***"):
            continue
        if line.lstrip().startswith("- ") or line.lstrip().startswith("* "):
            flush_pending_h2()
            if pending_subhead is not None:
                story.append(pending_subhead)
                pending_subhead = None
            story.append(Paragraph("• " + _strip_md(line.lstrip()[2:]), styles["BulletR"]))
            continue
        if re.match(r"^\d+\.\s", line.lstrip()):
            flush_pending_h2()
            if pending_subhead is not None:
                story.append(pending_subhead)
                pending_subhead = None
            story.append(Paragraph(_strip_md(line.lstrip()), styles["BulletR"]))
            continue

        flush_pending_h2()
        if pending_subhead is not None:
            story.append(pending_subhead)
            pending_subhead = None
        style = styles["MetaR"] if line.startswith("**") and ":" in line[:40] else styles["BodyR"]
        story.append(Paragraph(_strip_md(line), style))

    flush_pending_h2()
    if pending_subhead is not None:
        story.append(pending_subhead)
        pending_subhead = None
    flush_table()
    flush_code()

    # append key pitch figures at end if present
    for name in [
        "slide03_leakage.png",
        "slide04_mae_winners.png",
        "slide06_changepoints.png",
        "slide02_idea.png",
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
            f"Task 2 · {i + 1}/{len(slides)} · репозиторий: локально, публикация — после конкурса",
        )

        # Visual-first pitch: PNG carries the message; short bullets on the left.
        fig = SLIDE_ASSETS[i] if i < len(SLIDE_ASSETS) else None
        show_fig = bool(fig and fig.exists())
        last_i = len(slides) - 1
        # Figure-dominant: title, idea, leakage, winners, rural, CP, final
        fig_heavy = i in (0, 1, 2, 3, 4, 5, last_i)

        y = H - 2.2 * cm
        c.setFillColor(colors.HexColor("#222222"))
        if show_fig and fig_heavy:
            max_text_w = W * 0.34
        elif show_fig:
            max_text_w = W * 0.42
        else:
            max_text_w = W - 2.4 * cm

        for ln in body_lines:
            if not ln.strip():
                y -= 0.22 * cm
                continue
            if "presentation_assets" in ln or "figures/" in ln or ln.strip().startswith("**Рисунок"):
                continue
            if ln.strip().startswith("```") or ln.strip() == "```":
                continue
            if ln.strip().startswith("|"):
                # winners table lives in PNG — skip ascii table on figure-heavy slides
                if fig_heavy:
                    continue
                row = _strip_md(ln.strip())
                c.setFont(font, 8.0)
                c.drawString(1.2 * cm, y, row[:90])
                y -= 0.38 * cm
                if y < 1.5 * cm:
                    break
                continue
            text = _strip_md(ln.strip())
            if text.startswith("- "):
                text = "• " + text[2:]
            # Skip long headline duplicate already in PNG for figure-heavy slides
            if fig_heavy and not text.startswith("•") and len(text) > 40 and i != 0:
                continue
            size = 12 if i == 0 else 10
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
                    y -= 0.48 * cm
                    line = w
                    if y < 1.5 * cm:
                        break
            if line and y >= 1.5 * cm:
                c.drawString(1.2 * cm, y, line)
                y -= 0.48 * cm
            if y < 1.5 * cm:
                break

        if show_fig:
            try:
                from reportlab.lib.utils import ImageReader
                if fig_heavy:
                    iw, ih = 16.5 * cm, 9.2 * cm
                    x_img = W - iw - 0.5 * cm
                    y_img = 1.0 * cm
                elif i in (0, last_i):
                    iw, ih = 12.0 * cm, 7.4 * cm
                    x_img = W - iw - 0.7 * cm
                    y_img = 1.2 * cm
                else:
                    iw, ih = 11.2 * cm, 7.2 * cm
                    x_img = W - iw - 0.7 * cm
                    y_img = 1.2 * cm
                c.drawImage(
                    ImageReader(str(fig)),
                    x_img,
                    y_img,
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
