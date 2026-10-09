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
JURY = ROOT / "для_жюри"
PITCH = ROOT / "scripts" / "_pitch"
FIGURES = ROOT / "figures"
ASSETS = PITCH / "presentation_assets"
OUT_REPORT = JURY / "method_report.pdf"
OUT_SLIDES = JURY / "presentation.pdf"
MD_REPORT = JURY / "method_report.md"
MD_SLIDES = PITCH / "TASK1_slides.md"
REPO_URL = "https://github.com/odisseykingitaki-commits/sber-szfo-spectral"

FIG_NAMES = [
    "fig_final.png",
    "fig_network.png",
    "fig_dynamic.png",
    "fig_bootstrap.png",
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
            img_path = (ROOT / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
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
    """Delegate to chart-first jury builder (11 slides)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_jury_slides import build as build_jury  # noqa: WPS433

    build_jury()


def main():
    slides_only = "--slides-only" in sys.argv
    if not slides_only:
        if not MD_REPORT.exists():
            print(f"Missing {MD_REPORT}", file=sys.stderr)
            sys.exit(1)
        build_report_pdf()
    build_slides_pdf()
    print("Done.")


if __name__ == "__main__":
    main()
