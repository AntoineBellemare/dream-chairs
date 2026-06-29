"""Minimal Markdown -> PDF for the project handout (headings, bullets, bold/italic/code,
horizontal rules). Uses a Unicode TrueType font so glyphs like ✓ ◦ → ↔ render correctly.

  python scripts/md_to_pdf.py docs/SUMMARY_FOR_STUDENTS.md
  python scripts/md_to_pdf.py in.md out.pdf
"""
from __future__ import annotations

import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

FONT_SETS = [
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf",
     "C:/Windows/Fonts/segoeuii.ttf", "C:/Windows/Fonts/segoeuiz.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
     "C:/Windows/Fonts/ariali.ttf", "C:/Windows/Fonts/arialbi.ttf"),
]


# glyphs the base UI font may lack — routed through a symbol font so they don't box.
SYM_CHARS = "✓✗◦↔"
SYM_FONTS = ["C:/Windows/Fonts/seguisym.ttf", "C:/Windows/Fonts/arial.ttf"]


def register_font():
    sym = "Body"
    for p in SYM_FONTS:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont("Sym", p))
            sym = "Sym"
            break
    for reg, bold, ital, bi in FONT_SETS:
        if all(os.path.exists(p) for p in (reg, bold, ital, bi)):
            pdfmetrics.registerFont(TTFont("Body", reg))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold))
            pdfmetrics.registerFont(TTFont("Body-Italic", ital))
            pdfmetrics.registerFont(TTFont("Body-BoldItalic", bi))
            registerFontFamily("Body", normal="Body", bold="Body-Bold",
                               italic="Body-Italic", boldItalic="Body-BoldItalic")
            return "Body", sym
    return "Helvetica", sym


def inline(s: str, sym: str = "Body") -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
    if sym != "Body":
        for ch in SYM_CHARS:
            s = s.replace(ch, f'<font face="{sym}">{ch}</font>')
    return s


def parse_blocks(md: str):
    blocks, cur = [], None
    for raw in md.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            cur = None
            continue
        if re.match(r"^---+$", line.strip()):
            blocks.append(["hr", ""]); cur = None; continue
        h = re.match(r"^(#{1,3})\s+(.*)$", line)
        if h:
            blocks.append(["h%d" % len(h.group(1)), h.group(2)]); cur = None; continue
        b = re.match(r"^[-*]\s+(.*)$", line)
        if b:
            blocks.append(["bullet", b.group(1)]); cur = blocks[-1]; continue
        if cur and cur[0] in ("bullet", "para"):
            cur[1] += " " + line.strip(); continue
        blocks.append(["para", line.strip()]); cur = blocks[-1]
    return blocks


def build(md_path: str, pdf_path: str):
    with open(md_path, encoding="utf-8") as f:
        md = f.read()
    base, sym = register_font()
    bold = "Body-Bold" if base == "Body" else "Helvetica-Bold"
    body = ParagraphStyle("body", fontName=base, fontSize=10.5, leading=15.5,
                          spaceAfter=6, textColor=colors.HexColor("#1a1a1a"))
    h1 = ParagraphStyle("h1", parent=body, fontName=bold, fontSize=20, leading=24,
                        spaceBefore=2, spaceAfter=10, textColor=colors.HexColor("#0F6E56"))
    h2 = ParagraphStyle("h2", parent=body, fontName=bold, fontSize=14.5, leading=18,
                        spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#0F6E56"))
    h3 = ParagraphStyle("h3", parent=body, fontName=bold, fontSize=12, leading=15,
                        spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#333333"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=16, bulletIndent=3, spaceAfter=4)

    story = []
    for kind, text in parse_blocks(md):
        if kind == "hr":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cccccc")))
            story.append(Spacer(1, 4))
        elif kind == "h1":
            story.append(Paragraph(inline(text, sym), h1))
        elif kind == "h2":
            story.append(Paragraph(inline(text, sym), h2))
        elif kind == "h3":
            story.append(Paragraph(inline(text, sym), h3))
        elif kind == "bullet":
            story.append(Paragraph(inline(text, sym), bullet, bulletText="•"))
        else:
            story.append(Paragraph(inline(text, sym), body))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(base, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 1.05 * cm, f"Dream → Chairs   ·   {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                            topMargin=2.0 * cm, bottomMargin=1.8 * cm,
                            title="Dream to Chairs — approaches & design paths")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {pdf_path}  (font: {base})")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "docs/SUMMARY_FOR_STUDENTS.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".pdf"
    build(src, dst)
