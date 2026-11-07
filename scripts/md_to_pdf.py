"""Minimal Markdown -> PDF (reportlab) for the technical note: headings, paragraphs, bullet / numbered lists,
pipe tables, bold, inline code, links (text only) and fenced code blocks.

  python scripts/md_to_pdf.py docs/TECHNICAL_NOTE.md docs/TECHNICAL_NOTE.pdf
"""
from __future__ import annotations

import html
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#13234b")
ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, alignment=TA_LEFT, spaceAfter=4)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceAfter=6, spaceBefore=4)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=BODY, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=3)
CELL = ParagraphStyle("c", parent=BODY, fontSize=7.8, leading=9.6, spaceAfter=0)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=7.5, leading=9)


def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    return t


def table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    ncol = max(len(r) for r in rows)
    data = [[Paragraph(inline(c), CELL) for c in r + [""] * (ncol - len(r))] for r in rows]
    width = 180 * mm
    t = Table(data, colWidths=[width / ncol] * ncol, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f8")), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def convert(src, dst):
    lines = open(src, encoding="utf-8").read().splitlines()
    story, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("<!--"):
            i += 1
            continue
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            story.append(Preformatted("\n".join(lines[i + 1:j]), CODE))
            i = j + 1
            continue
        if ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            story += [table(lines[i:j]), Spacer(1, 5)]
            i = j
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", ln)
        if m:
            items, numbered = [], m.group(2)[0].isdigit()
            while i < len(lines):
                m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", lines[i])
                if not m:
                    if lines[i].startswith("   ") and lines[i].strip() and items:
                        items[-1] += " " + lines[i].strip()
                        i += 1
                        continue
                    break
                items.append(("    " if len(m.group(1)) >= 2 else "") + m.group(3))
                i += 1
            # group runs of top-level and nested items; nested runs are plain bullets, top-level numbering continues
            n_top, k = 0, 0
            while k < len(items):
                nested = items[k].startswith("    ")
                run = []
                while k < len(items) and items[k].startswith("    ") == nested:
                    run.append(items[k].strip())
                    k += 1
                if nested or not numbered:
                    story.append(ListFlowable([ListItem(Paragraph(inline(x), BODY)) for x in run], bulletType="bullet",
                                              leftIndent=24 if nested else 12, bulletFontSize=7 if nested else 8))
                else:
                    story.append(ListFlowable([ListItem(Paragraph(inline(x), BODY)) for x in run], bulletType="1",
                                              start=str(n_top + 1), leftIndent=12, bulletFontSize=8))
                    n_top += len(run)
            continue
        if ln.startswith("### "):
            story.append(Paragraph(inline(ln[4:]), H3))
        elif ln.startswith("## "):
            story.append(Paragraph(inline(ln[3:]), H2))
        elif ln.startswith("# "):
            story.append(Paragraph(inline(ln[2:]), H1))
        elif ln.startswith("> "):
            story.append(Paragraph("<i>" + inline(ln[2:]) + "</i>", BODY))
        elif ln.strip():
            para = [ln.strip()]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||```|\s*([-*]|\d+\.)\s|> )", lines[i + 1]):
                i += 1
                para.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(para)), BODY))
        i += 1
    doc = SimpleDocTemplate(dst, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                            title="RETROFIT AI - technical note")
    doc.build(story)
    print("pdf ->", dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
