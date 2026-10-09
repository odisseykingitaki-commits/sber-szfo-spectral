"""Build для_жюри/method_report.pdf from markdown + figures (reportlab).

CRITICAL: all text styles (body, headings, tables, code) must use a TTF with
Cyrillic glyphs. Courier/Helvetica produce black-square tofu for Russian.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "для_жюри" / "method_report.md"
OUT = ROOT / "для_жюри" / "method_report.pdf"
FIGURES = ROOT / "figures"


def _register_cyrillic_fonts() -> tuple[str, str, str]:
    """Return (regular, bold, mono) registered font names. Fail loud if missing."""
    windir = Path(r"C:\Windows\Fonts")
    pairs = [
        (windir / "arial.ttf", windir / "arialbd.ttf"),
        (windir / "Arial.ttf", windir / "Arialbd.ttf"),
        (windir / "calibri.ttf", windir / "calibrib.ttf"),
        (windir / "times.ttf", windir / "timesbd.ttf"),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    # Prefer DejaVu Sans Mono (full Cyrillic). Avoid Courier/Consolas pitfalls
    # with reportlab subsetting; fall back to the same body TTF as mono.
    mono_cands = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        windir / "DejaVuSansMono.ttf",
    ]

    reg_path = bold_path = None
    for reg, bold in pairs:
        if reg.exists():
            reg_path, bold_path = reg, (bold if bold.exists() else reg)
            break
    if reg_path is None:
        raise FileNotFoundError(
            "No Cyrillic TTF found (Arial/Calibri/Times/DejaVu). "
            "Cannot build method_report.pdf without embedding a Cyrillic font."
        )

    pdfmetrics.registerFont(TTFont("DocFont", str(reg_path)))
    pdfmetrics.registerFont(TTFont("DocFont-Bold", str(bold_path)))

    # Code fences in this report mix Russian prose (§3.4) with Latin symbols —
    # always use a proven Cyrillic face (body TTF), not legacy Courier.
    mono_path = next((p for p in mono_cands if p.exists()), reg_path)
    pdfmetrics.registerFont(TTFont("DocMono", str(mono_path)))
    print(f"Fonts: body={reg_path.name} bold={bold_path.name} mono={mono_path.name}")
    return "DocFont", "DocFont-Bold", "DocMono"


FONT_REG, FONT_BOLD, FONT_MONO = _register_cyrillic_fonts()

# Arial lacks these; without remap they render as empty boxes (not Cyrillic tofu).
_GLYPH_FIX = str.maketrans({
    "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5",
    "₆": "6", "₇": "7", "₈": "8", "₉": "9", "₀": "0",
    "₊": "+", "₋": "-",
    "∈": "in",
})


def _fix_glyphs(s: str) -> str:
    # MD sometimes stores HTML entities literally; decode before reportlab escape
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return s.translate(_GLYPH_FIX)


def esc(s: str) -> str:
    s = _fix_glyphs(s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # Inline code must use Cyrillic-capable mono — never Courier
    s = re.sub(
        r"`([^`]+)`",
        rf"<font face='{FONT_MONO}' size='8'>\1</font>",
        s,
    )
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
    if not MD.exists():
        print(f"Missing {MD}", file=sys.stderr)
        sys.exit(1)

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
            fontName=FONT_MONO,  # Cyrillic-capable (Consolas / Arial fallback)
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
            if not img_path.exists():
                img_path = FIGURES / Path(rel).name
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
            story.append(Preformatted(_fix_glyphs("\n".join(buf)), styles["Coderu"]))
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
                        bulletFontName=FONT_REG,
                    )
                )
                i += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="•",
                    leftIndent=12,
                    bulletFontName=FONT_REG,
                )
            )
            continue

        # numbered
        if re.match(r"^\d+\.\s", line.strip()):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
                body = re.sub(r"^\d+\.\s", "", lines[i].strip())
                items.append(
                    ListItem(
                        Paragraph(esc(body), styles["Bodyru"]),
                        leftIndent=8,
                        bulletFontName=FONT_REG,
                    )
                )
                i += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    leftIndent=12,
                    bulletFontName=FONT_REG,
                )
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
        title="Method Report",
        author="sber-project",
    )
    doc.build(story)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
