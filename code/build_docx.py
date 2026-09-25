"""build_docx.py: render draft.md into a submission-ready Word manuscript
(elsarticle-num numbered references, double-spaced body, continuous line
numbers) plus a separate Highlights .docx, by parsing the manuscript's own
markdown source and this pipeline's own vocabulary.py constants directly --
so the build can't silently drift from either.

    python code/build_docx.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vocabulary as V  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
DRAFT = _ROOT / "draft.md"
FIGURES = _ROOT / "figures"
OUT = _ROOT / "manuscript.docx"
HIGHLIGHTS_OUT = _ROOT / "highlights.docx"

TITLE = "LLM-Based Automated RFI Classification and Routing for Construction Projects"

FIGURE_CAPTIONS = {
    "ablation-accuracy.png": (
        "Reviewer-routing accuracy by prompt condition, scoring arm, and provider "
        "(Anthropic claude-sonnet-5 and OpenAI gpt-5; bare = not scored for every "
        "arm/provider because 100% of its outputs failed closed-vocabulary schema "
        "validation and were excluded)."
    ),
    "gate-tradeoff.png": (
        "Confidence/escalation gating separates low- from high-accuracy "
        "subpopulations (composed arm, Anthropic claude-sonnet-5)."
    ),
    "iaa-by-field.png": "Field-level inter-annotator agreement.",
}

INLINE_RE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`")
FIGCAP_RE = re.compile(r"^\*\(Figure (\d+): `(?:figures/)?(.+?)`\)\*$")


def add_inline_runs(paragraph, text):
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos:m.start()])
        if m.group(1) is not None:
            r = paragraph.add_run(m.group(1))
            r.bold = True
        elif m.group(2) is not None:
            r = paragraph.add_run(m.group(2))
            r.italic = True
        else:
            r = paragraph.add_run(m.group(3))
            r.font.name = "Consolas"
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def _render_vocab_block() -> dict:
    csi_lines = []
    for d in sorted(V.DISCIPLINE_TO_CSI):
        divisions = V.DISCIPLINE_TO_CSI[d]
        if divisions:
            csi_lines.append(f"  {d}: {', '.join(sorted(divisions))}")
        else:
            csi_lines.append(f'  {d}: (no CSI division; use "\u2014")')
    return {
        "rfi_types": ", ".join(sorted(V.RFI_TYPES)),
        "disciplines": ", ".join(sorted(V.DISCIPLINES)),
        "csi_block": "\n".join(csi_lines),
        "urgency": ", ".join(V.URGENCY_TIERS),
        "reviewers": ", ".join(V.REVIEWER_ROLES),
    }


def apply_vocab_substitutions(code_text: str) -> str:
    if "Use only these closed vocabularies" not in code_text:
        return code_text
    v = _render_vocab_block()
    code_text = code_text.replace(
        "{sorted RFI_TYPES, six values \u2014 see \u00a73.1}", v["rfi_types"]
    )
    code_text = code_text.replace(
        "{sorted DISCIPLINES, eight values \u2014 see \u00a73.1}", v["disciplines"]
    )
    code_text = code_text.replace(
        '{DISCIPLINE_TO_CSI, one line per discipline: its group-level CSI '
        'division(s), or "(no CSI division; use \\"\u2014\\")" for General}',
        v["csi_block"],
    )
    code_text = code_text.replace(
        "{URGENCY_TIERS, in ordinal order: routine, priority, urgent, critical}",
        v["urgency"],
    )
    code_text = code_text.replace(
        "{REVIEWER_ROLES, ten values \u2014 see \u00a73.1}", v["reviewers"]
    )
    return code_text


def add_heading_text(doc, text, level):
    p = doc.add_paragraph(style=f"Heading {level}")
    add_inline_runs(p, text)
    return p


def add_bullets(doc, group_lines):
    items = []
    current = None
    for line in group_lines:
        s = line.strip()
        if s.startswith("- "):
            if current is not None:
                items.append(current)
            current = [s[2:]]
        elif current is not None:
            current.append(s)
    if current is not None:
        items.append(current)
    for item in items:
        text = " ".join(item).strip()
        p = doc.add_paragraph(style="List Bullet")
        add_inline_runs(p, text)


def add_table(doc, table_lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in table_lines]
    data_rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    table = doc.add_table(rows=0, cols=len(data_rows[0]))
    table.style = "Table Grid"
    for idx, r in enumerate(data_rows):
        cells = table.add_row().cells
        for cell, val in zip(cells, r):
            _set_cell_shading(cell, "FFFFFF")
            run = cell.paragraphs[0].add_run(val)
            run.font.color.rgb = RGBColor(0, 0, 0)
            if idx == 0:
                run.bold = True


def _set_cell_shading(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "000000")
        tcBorders.append(border)
    tcPr.append(tcBorders)


def add_code_block(doc, text):
    p = doc.add_paragraph(style="Code")
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i > 0:
            p.add_run().add_break()
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(9)


def add_figure(doc, fignum, fname):
    path = FIGURES / fname
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(5.5))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure {fignum}. {FIGURE_CAPTIONS.get(fname, fname)}")
    r.bold = True
    r.font.size = Pt(10)


def render_block(doc, lines):
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped == "":
            i += 1
            continue
        if stripped == "---":
            i += 1
            continue
        if stripped.startswith("### "):
            add_heading_text(doc, stripped[4:], 2)
            i += 1
            continue
        if stripped.startswith("## "):
            add_heading_text(doc, stripped[3:], 1)
            i += 1
            continue
        if stripped == "```":
            i += 1
            code_lines = []
            while i < n and lines[i].strip() != "```":
                code_lines.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, apply_vocab_substitutions("\n".join(code_lines)))
            continue
        m = FIGCAP_RE.match(stripped)
        if m:
            add_figure(doc, m.group(1), m.group(2))
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(doc, table_lines)
            continue
        if stripped.startswith("- "):
            group = []
            while i < n and lines[i].strip() != "":
                group.append(lines[i])
                i += 1
            add_bullets(doc, group)
            continue
        group = []
        while i < n and lines[i].strip() != "":
            group.append(lines[i].strip())
            i += 1
        text = " ".join(group).strip()
        if text:
            p = doc.add_paragraph(style="Body Text")
            add_inline_runs(p, text)


def setup_page(doc):
    sectPr = doc.sections[0]._sectPr
    lnNumType = OxmlElement("w:lnNumType")
    lnNumType.set(qn("w:countBy"), "1")
    lnNumType.set(qn("w:restart"), "continuous")
    sectPr.append(lnNumType)


def setup_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    normal.paragraph_format.space_after = Pt(0)

    try:
        body = doc.styles["Body Text"]
    except KeyError:
        body = doc.styles.add_style("Body Text", WD_STYLE_TYPE.PARAGRAPH)
    body.base_style = normal
    body.font.name = "Times New Roman"
    body.font.size = Pt(12)
    body.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    body.paragraph_format.space_after = Pt(6)

    try:
        code = doc.styles["Code"]
    except KeyError:
        code = doc.styles.add_style("Code", WD_STYLE_TYPE.PARAGRAPH)
    code.font.name = "Consolas"
    code.font.size = Pt(9)
    code.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    code.paragraph_format.space_before = Pt(6)
    code.paragraph_format.space_after = Pt(6)
    code.paragraph_format.left_indent = Inches(0.3)

    bullet = doc.styles["List Bullet"]
    bullet.font.name = "Times New Roman"
    bullet.font.size = Pt(12)
    bullet.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE

    for heading_name in ("Heading 1", "Heading 2"):
        heading = doc.styles[heading_name]
        heading.font.name = "Times New Roman"
        heading.font.color.rgb = RGBColor(0, 0, 0)


def find_after(lines, marker, after=0):
    return next(i for i in range(after, len(lines)) if lines[i].strip() == marker)


def main():
    lines = DRAFT.read_text(encoding="utf-8").splitlines()

    hl_start = find_after(
        lines,
        "## Highlights (draft — final copy goes in a separate Highlights file, not this manuscript)",
    )
    hl_end = find_after(lines, "---", hl_start + 1)
    highlights_lines = lines[hl_start + 1:hl_end]

    ab_start = find_after(lines, "## Abstract (draft)")
    ab_end = find_after(lines, "---", ab_start + 1)
    abstract_lines = lines[ab_start + 1:ab_end]

    body_start = find_after(lines, "## 1. INTRODUCTION")
    tracking_start = find_after(lines, "## Internal Tracking (not for submission)")
    body_lines = lines[body_start:tracking_start]
    while body_lines and body_lines[-1].strip() in ("", "---"):
        body_lines.pop()

    doc = docx.Document()
    setup_styles(doc)
    setup_page(doc)

    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title_p.add_run(TITLE)
    r.bold = True
    r.font.size = Pt(16)
    r.font.color.rgb = RGBColor(0, 0, 0)

    auth_p = doc.add_paragraph()
    auth_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, line in enumerate([
        "Nooka Sai Shiva Kumar Vunduru, Director of Technology",
        "University of Dallas",
        "ORCID: https://orcid.org/0009-0000-9466-1317",
        "email: dbushiva@gmail.com, nvunduru@udallas.edu",
    ]):
        if i > 0:
            auth_p.add_run().add_break()
        auth_p.add_run(line)

    add_heading_text(doc, "Abstract", 1)
    render_block(doc, abstract_lines)

    render_block(doc, body_lines)

    doc.save(OUT)
    print(f"Saved: {OUT}")

    hl_doc = docx.Document()
    setup_styles(hl_doc)
    add_heading_text(hl_doc, "Highlights", 1)
    for line in highlights_lines:
        s = line.strip()
        if not s.startswith("- "):
            continue
        text = re.sub(r"\s*\(\d+\)\s*$", "", s[2:]).strip()
        p = hl_doc.add_paragraph(style="List Bullet")
        add_inline_runs(p, text)
    hl_doc.save(HIGHLIGHTS_OUT)
    print(f"Saved: {HIGHLIGHTS_OUT}")


if __name__ == "__main__":
    main()
