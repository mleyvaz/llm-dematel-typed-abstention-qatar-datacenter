"""
Generic, non-journal-branded manuscript builder. Replaces the NSS-template
based helpers in build_paper*.py with a plain, professional layout built
from scratch (no external template file, no journal logo/letterhead) using
python-docx's built-in styles, styled for a generic Q1-submission-ready
appearance (Times New Roman, standard margins, numbered headings).
"""
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from lxml import etree

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

def omml(xml_inner):
    xml = f'<m:oMath xmlns:m="{M_NS}">{xml_inner}</m:oMath>'
    return etree.fromstring(xml.encode("utf-8"))

def r_(text, italic=False):
    it = '<m:i m:val="0"/>' if not italic else ""
    return f'<m:r><m:rPr>{it}</m:rPr><m:t xml:space="preserve">{text}</m:t></m:r>'

def sub(base, subscript):
    return f'<m:sSub><m:e>{r_(base, True)}</m:e><m:sub>{r_(subscript)}</m:sub></m:sSub>'

def sup(base, supscript):
    return f'<m:sSup><m:e>{r_(base, True)}</m:e><m:sup>{r_(supscript)}</m:sup></m:sSup>'

def frac(num, den):
    return f'<m:f><m:fPr><m:type m:val="bar"/></m:fPr><m:num>{num}</m:num><m:den>{den}</m:den></m:f>'


def new_document():
    doc = docx.Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15

    title = doc.styles["Title"]
    title.font.name = "Times New Roman"
    title.font.size = Pt(16)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)

    for lvl, size in [(1, 13), (2, 12), (3, 11)]:
        h = doc.styles[f"Heading {lvl}"]
        h.font.name = "Times New Roman"
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.italic = (lvl == 3)
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(6)

    # page numbers in footer
    footer = section.footer
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    fld = etree.SubElement(run._r, qn('w:fldSimple'))
    fld.set(qn('w:instr'), 'PAGE')
    return doc


def add(doc, text, style=None, bold=False, italic=False, align=None, size=None):
    p = doc.add_paragraph(style=style)
    if align:
        p.alignment = align
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)
    return p

def add_runs(doc, style, parts, align=None):
    p = doc.add_paragraph(style=style)
    if align:
        p.alignment = align
    for text, opts in parts:
        run = p.add_run(text)
        run.bold = opts.get("bold", False)
        run.italic = opts.get("italic", False)
        run.superscript = opts.get("sup", False)
    return p

def add_equation(doc, inner_xml, label=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(omml(inner_xml))
    if label:
        run = p.add_run("\t" + label)
        run.italic = True
    return p

def _set_cell_font(cell, size=9, bold=False):
    for p in cell.paragraphs:
        p.paragraph_format.space_after = Pt(2)
        for rn in p.runs:
            rn.font.size = Pt(size)
            rn.font.name = "Times New Roman"
            rn.bold = bold

def add_figure(doc, path, caption, width_in=6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width_in))
    add(doc, caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)

def add_table(doc, headers, rows, caption, widths=None):
    add(doc, caption, bold=True, size=9)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    total_w = 6.0
    if widths is None:
        widths = [total_w / len(headers)] * len(headers)
    else:
        scale = total_w / sum(widths)
        widths = [w * scale for w in widths]
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = h
        t.rows[0].cells[i].width = Inches(widths[i])
        _set_cell_font(t.rows[0].cells[i], size=9, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            cells[i].width = Inches(widths[i])
            _set_cell_font(cells[i], size=9)
    for col_idx, w in enumerate(widths):
        for row in t.rows:
            row.cells[col_idx].width = Inches(w)
    add(doc, "", size=4)
