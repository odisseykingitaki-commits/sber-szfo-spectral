"""Build docs/TASK1_method_report.pdf from markdown + figures (reportlab)."""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "TASK1_METHOD_REPORT.md"
OUT = ROOT / "docs" / "TASK1_method_report.pdf"
FIGURES = ROOT / "figures"

# Prefer DejaVu for Cyrillic
FONT_REG = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_cands = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for reg, bold in zip(candidates, bold_cands):
        if reg.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("DocFont", str(reg)))
            pdfmetrics.registerFont(TTFont("DocFont-Bold", str(bold)))
            FONT_REG = "DocFont"
            FONT_BOLD = "DocFont-Bold"
            break
except Exception:
    pass


def esc(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<font face='Courier' size='8'>\1</font>", s)
    return s


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if re.match(r"^\|?\s*-+", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def build():
    text = MD.read_text(encoding="utf-8")
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1ru",
            fontName=FONT_BOLD,
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor("#1a1a1a"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2ru",
            fontName=FONT_BOLD,
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3ru",
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=13,
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Bodyru",
            fontName=FONT_REG,
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Coderu",
            fontName="Courier",
            fontSize=7.5,
            leading=9.5,
            backColor=colors.HexColor("#f4f4f4"),
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Capt",
            fontName=FONT_REG,
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cell",
            fontName=FONT_REG,
            fontSize=7.5,
            leading=9.5,
        )
    )

    story = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # images
        mimg = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if mimg:
            caption, rel = mimg.group(1), mimg.group(2)
            img_path = (MD.parent / rel).resolve()
            if img_path.exists():
                im = Image(str(img_path), width=15 * cm, height=9 * cm, kind="proportional")
                story.append(KeepTogether([im, Paragraph(esc(caption or img_path.name), styles["Capt"])]))
            else:
                story.append(Paragraph(esc(f"[нет файла: {rel}]"), styles["Capt"]))
            i += 1
            continue

        if line.startswith("# "):
            story.append(Paragraph(esc(line[2:].strip()), styles["H1ru"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(esc(line[3:].strip()), styles["H2ru"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(esc(line[4:].strip()), styles["H3ru"]))
            i += 1
            continue
        if line.strip() == "---":
            story.append(Spacer(1, 4 * mm))
            i += 1
            continue

        if line.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            story.append(Preformatted("\n".join(buf), styles["Coderu"]))
            continue

        if line.strip().startswith("|"):
            tlines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tlines.append(lines[i])
                i += 1
            rows = parse_table(tlines)
            if not rows:
                continue
            data = [
                [Paragraph(esc(c), styles["Cell"]) for c in row] for row in rows
            ]
            col_w = (17 * cm) / max(len(rows[0]), 1)
            tbl = Table(data, colWidths=[col_w] * len(rows[0]), hAlign="LEFT")
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 3 * mm))
            continue

        if line.strip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(
                    ListItem(
                        Paragraph(esc(lines[i].strip()[2:]), styles["Bodyru"]),
                        leftIndent=8,
                    )
                )
                i += 1
            story.append(
                ListFlowable(items, bulletType="bullet", start="•", leftIndent=12)
            )
            continue

        # numbered
        if re.match(r"^\d+\.\s", line.strip()):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                body = re.sub(r"^\d+\.\s", "", lines[i].strip())
                items.append(
                    ListItem(Paragraph(esc(body), styles["Bodyru"]), leftIndent=8)
                )
                i += 1
            story.append(
                ListFlowable(items, bulletType="1", leftIndent=12)
            )
            continue

        story.append(Paragraph(esc(line.strip()), styles["Bodyru"]))
        i += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="TASK1 Method Report",
        author="sber-project",
    )
    doc.build(story)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
