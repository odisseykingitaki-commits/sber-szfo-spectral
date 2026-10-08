"""Build docs/TASK1_presentation.pdf — 10 landscape slides (reportlab)."""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Frame,
    Image,
    PageTemplate,
    BaseDocTemplate,
    FrameBreak,
    NextPageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "TASK1_slides.md"
OUT = ROOT / "docs" / "TASK1_presentation.pdf"
FIGURES = ROOT / "figures"

FONT_REG = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pairs = [
        (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
        (Path(r"C:\Windows\Fonts\calibri.ttf"), Path(r"C:\Windows\Fonts\calibrib.ttf")),
    ]
    for reg, bold in pairs:
        if reg.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("SlideFont", str(reg)))
            pdfmetrics.registerFont(TTFont("SlideFont-Bold", str(bold)))
            FONT_REG = "SlideFont"
            FONT_BOLD = "SlideFont-Bold"
            break
except Exception:
    pass

PAGE = landscape(A4)
W, H = PAGE


def esc(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<font face='Courier' size='10'>\1</font>", s)
    return s


def parse_slides(md: str) -> list[dict]:
    chunks = re.split(r"\n## Слайд\s+", md)
    slides = []
    for ch in chunks[1:]:
        lines = ch.strip().splitlines()
        if not lines:
            continue
        title = re.sub(r"^\d+\s*[—\-]\s*", "", lines[0]).strip()
        body_lines = []
        for ln in lines[1:]:
            if ln.strip() == "---":
                continue
            body_lines.append(ln)
        slides.append({"title": title, "body": "\n".join(body_lines).strip()})
    return slides


def figure_for_slide(idx: int) -> Path | None:
    mapping = {
        5: FIGURES / "fig_final.png",
        6: FIGURES / "fig_network.png",
        7: FIGURES / "fig_bootstrap.png",
    }
    p = mapping.get(idx)
    return p if p and p.exists() else None


def build_slide_flowables(slide: dict, idx: int, styles: dict):
    flows = []
    flows.append(Paragraph(esc(slide["title"]), styles["title"]))
    flows.append(Spacer(1, 4 * mm))

    fig = figure_for_slide(idx)
    body = slide["body"]

    # simple table blocks
    lines = body.splitlines()
    i = 0
    bullets = []
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.strip().startswith("|"):
            if bullets:
                for b in bullets:
                    flows.append(Paragraph("• " + esc(b), styles["body"]))
                bullets = []
            tlines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tlines.append(lines[i])
                i += 1
            rows = []
            for tl in tlines:
                if re.match(r"^\|?\s*-+", tl):
                    continue
                rows.append([c.strip() for c in tl.strip().strip("|").split("|")])
            if rows:
                data = [[Paragraph(esc(c), styles["cell"]) for c in r] for r in rows]
                cw = (22 * cm) / len(rows[0])
                tbl = Table(data, colWidths=[cw] * len(rows[0]))
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf7")),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                flows.append(Spacer(1, 2 * mm))
                flows.append(tbl)
                flows.append(Spacer(1, 3 * mm))
            continue
        if ln.strip().startswith("- ") or re.match(r"^\d+\.\s", ln.strip()):
            bullets.append(re.sub(r"^(-\s+|\d+\.\s+)", "", ln.strip()))
            i += 1
            continue
        if bullets:
            for b in bullets:
                flows.append(Paragraph("• " + esc(b), styles["body"]))
            bullets = []
        flows.append(Paragraph(esc(ln.strip()), styles["body"]))
        i += 1
    if bullets:
        for b in bullets:
            flows.append(Paragraph("• " + esc(b), styles["body"]))

    if fig is not None:
        flows.append(Spacer(1, 3 * mm))
        im = Image(str(fig), width=12 * cm, height=6.5 * cm, kind="proportional")
        flows.append(im)

    # footer
    flows.append(Spacer(1, 4 * mm))
    flows.append(
        Paragraph(
            esc(f"Task 1 · слайд {idx}/10 · https://github.com/odisseykingitaki-commits/sber-szfo-spectral"),
            styles["foot"],
        )
    )
    return flows


def build():
    styles = {
        "title": ParagraphStyle(
            "t",
            fontName=FONT_BOLD,
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f2744"),
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "b",
            fontName=FONT_REG,
            fontSize=13,
            leading=17,
            spaceAfter=3,
            textColor=colors.HexColor("#222222"),
        ),
        "cell": ParagraphStyle(
            "c", fontName=FONT_REG, fontSize=10, leading=12
        ),
        "foot": ParagraphStyle(
            "f",
            fontName=FONT_REG,
            fontSize=8,
            textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER,
        ),
        "cover": ParagraphStyle(
            "cover",
            fontName=FONT_BOLD,
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f2744"),
            spaceAfter=10,
        ),
        "sub": ParagraphStyle(
            "sub",
            fontName=FONT_REG,
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
        ),
    }

    slides = parse_slides(MD.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(OUT),
        pagesize=PAGE,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.0 * cm,
        title="TASK1 Presentation",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        W - doc.leftMargin - doc.rightMargin,
        H - doc.topMargin - doc.bottomMargin,
        id="normal",
    )
    doc.addPageTemplates([PageTemplate(id="slide", frames=[frame])])

    story = []
    for idx, sl in enumerate(slides, start=1):
        if idx == 1:
            story.append(Spacer(1, 3.5 * cm))
            story.append(Paragraph(esc(sl["title"]), styles["cover"]))
            story.append(Spacer(1, 8 * mm))
            for ln in sl["body"].splitlines():
                if ln.strip():
                    story.append(Paragraph(esc(ln.strip()), styles["sub"]))
            story.append(Spacer(1, 2 * cm))
            story.append(
                Paragraph(
                    esc("Репозиторий: https://github.com/odisseykingitaki-commits/sber-szfo-spectral"),
                    styles["foot"],
                )
            )
        else:
            story.extend(build_slide_flowables(sl, idx, styles))
        if idx < len(slides):
            story.append(NextPageTemplate("slide"))
            from reportlab.platypus import PageBreak

            story.append(PageBreak())

    doc.build(story)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes), slides={len(slides)}")


if __name__ == "__main__":
    build()
