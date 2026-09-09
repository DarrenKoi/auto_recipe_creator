"""img/07/fig*.svg -> PNG(2400px, md/HTML 용) + 07_slides.pptx (편집 가능한 PPT 도형·표).

슬라이드 1~3 은 도표(SVG -> 도형), 그 뒤는 요약본(07_ax_innovation_challenge_short.md)의
markdown 표를 PPT 표로 옮긴 것이다(제목 = 그 표가 속한 절 제목, 굵게/[검증]/[목표] 서식 유지).

SVG 가 단일 원본이다. 1920x1080 슬라이드 캔버스로 그려져 있고, 여기서 rect/text/line/path 만 써서
python-pptx 의 사각형·텍스트 상자·연결선·자유형 도형으로 1:1 옮긴다(그림 삽입이 아니라 도형이라
PPT 에서 글자·색·위치를 바로 고칠 수 있다). 지원 요소는 rect(rx, fill, stroke, dasharray),
text(text-anchor, font-size/weight, fill; g 상속), line, path(M/L/H/V/Z 절대좌표: 2점=직선,
닫힌 3점=화살촉, 그 외=꺾은선). 새 SVG 요소를 쓰면 여기도 같이 늘려야 한다.

실행 (rsvg-convert 는 brew, python-pptx 는 1회성으로 얹는다):
    uv run --with python-pptx python docs/project_progress/build_07_figures.py
"""
import pathlib, re, subprocess
import xml.etree.ElementTree as ET
from lxml import etree
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt, Emu

HERE = pathlib.Path(__file__).resolve().parent / "img" / "07"
MD = pathlib.Path(__file__).resolve().parent / "07_ax_innovation_challenge_short.md"
PNG_WIDTH = 2400
PX = Emu(Inches(13.333) / 1920)  # 1 SVG px (1920 캔버스) = 이만큼 EMU
FONT = "Malgun Gothic"
INHERIT = ("font-size", "font-weight", "fill", "stroke", "stroke-width", "text-anchor", "stroke-dasharray")
SVG = "{http://www.w3.org/2000/svg}"


def rgb(hexstr):
    return RGBColor.from_string(hexstr.lstrip("#"))


def style_line(line, attrs, default_w=1.0):
    stroke = attrs.get("stroke")
    if not stroke or stroke == "none":
        line.fill.background()
        return
    line.color.rgb = rgb(stroke)
    line.width = Pt(float(attrs.get("stroke-width", default_w)) * 0.5)
    if attrs.get("stroke-dasharray"):
        line.dash_style = MSO_LINE.DASH


def add_rect(slide, a):
    x, y, w, h = (float(a[k]) for k in ("x", "y", "width", "height"))
    rx = float(a.get("rx", 0))
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rx else MSO_SHAPE.RECTANGLE,
                                   x * PX, y * PX, w * PX, h * PX)
    if rx:
        shape.adjustments[0] = min(0.5, rx / min(w, h))
    fill = a.get("fill", "#000000")
    if fill == "none":
        shape.fill.background()
    else:
        shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fill)
    style_line(shape.line, a)
    shape.shadow.inherit = False
    return shape


def text_width_px(s, size):
    return sum(size * (0.55 if ord(c) < 0x2E80 else 1.0) for c in s) * 1.3 + size


def add_text(slide, a, s):
    size = float(a.get("font-size", 16)); x, y = float(a["x"]), float(a["y"])
    anchor = a.get("text-anchor", "start"); w = text_width_px(s, size); h = size * 1.4
    left = {"middle": x - w / 2, "end": x - w}.get(anchor, x)
    box = slide.shapes.add_textbox(left * PX, (y - size * 1.05) * PX, w * PX, h * PX)
    tf = box.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = {"middle": PP_ALIGN.CENTER, "end": PP_ALIGN.RIGHT}.get(anchor, PP_ALIGN.LEFT)
    r = p.add_run(); r.text = s
    r.font.name = FONT; r.font.size = Pt(size * 0.5)
    r.font.bold = a.get("font-weight") in ("700", "bold")
    r.font.color.rgb = rgb(a.get("fill", "#000000"))


def add_polyline(slide, pts, a, closed):
    if len(pts) == 2 and not closed:
        (x1, y1), (x2, y2) = pts
        c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1 * PX, y1 * PX, x2 * PX, y2 * PX)
        style_line(c.line, a); return
    fb = slide.shapes.build_freeform(pts[0][0], pts[0][1], scale=PX)
    fb.add_line_segments(pts[1:], close=closed)
    shape = fb.convert_to_shape()
    fill = a.get("fill", "#000000") if closed else "none"
    if fill == "none":
        shape.fill.background()
    else:
        shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fill)
    if closed:
        shape.line.fill.background()  # 화살촉: 면만
    else:
        style_line(shape.line, a)
    shape.shadow.inherit = False


def parse_path(d):
    toks = re.findall(r"[MLHVZ]|-?\d+(?:\.\d+)?", d)
    pts, i, cmd, closed = [], 0, None, False
    while i < len(toks):
        t = toks[i]
        if t in "MLHVZ":
            cmd = t; i += 1
            if t == "Z": closed = True
            continue
        if cmd in ("M", "L"):
            pts.append((float(t), float(toks[i + 1]))); i += 2
        elif cmd == "H":
            pts.append((float(t), pts[-1][1])); i += 1
        elif cmd == "V":
            pts.append((pts[-1][0], float(t))); i += 1
    return pts, closed


def walk(slide, el, inherited):
    a = {k: v for k, v in inherited.items()}
    a.update({k: v for k, v in el.attrib.items()})
    tag = el.tag.replace(SVG, "")
    if tag == "rect":
        if not (a.get("x", "0") == "0" and a.get("y", "0") == "0" and a.get("width") == "1920"):  # 배경 제외
            add_rect(slide, a)
    elif tag == "text":
        add_text(slide, a, "".join(el.itertext()).strip())
    elif tag == "line":
        add_polyline(slide, [(float(a["x1"]), float(a["y1"])), (float(a["x2"]), float(a["y2"]))], a, False)
    elif tag == "path":
        pts, closed = parse_path(a["d"]); add_polyline(slide, pts, a, closed)
    for child in el:
        walk(slide, child, {k: a[k] for k in INHERIT if k in a})


svgs = sorted(HERE.glob("fig*.svg"))
for svg in svgs:
    png = svg.with_suffix(".png")
    subprocess.run(["rsvg-convert", "-w", str(PNG_WIDTH), "-o", str(png), str(svg)], check=True)
    print(f"[INFO] {png.name} ({png.stat().st_size / 1024:.0f} KB)")


# ---------- markdown 표 -> PPT 표 ----------
ACC, INK, LINE, BAND = "1E3A8A", "1F2328", "D0D7DE", "F6F8FA"
TAG_COLOR = {"[검증]": "166534", "[목표]": "92400E"}


def md_tables(md_path):
    """(절 제목, header cells, body rows) 목록. 표 바로 앞의 ## / ### 제목을 슬라이드 제목으로 쓴다."""
    title, rows, out = "", [], []
    for line in md_path.read_text(encoding="utf-8").splitlines() + [""]:
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")]); continue
        if rows:
            if len(rows) >= 3:
                out.append((title, rows[0], rows[2:]))
            rows = []
        m = re.match(r"^(##|###) (.+)$", line)
        if m:
            title = m.group(2).strip()
    return out


def set_borders(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag), w=str(int(Pt(0.75))), cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr"), val=LINE)
        etree.SubElement(ln, qn("a:prstDash"), val="solid")


def fill_cell(cell, md_text, size, color=INK, bold_all=False):
    """**굵게** 와 `[검증]`/`[목표]` 만 서식으로 옮기고 나머지 markdown 표시는 뗀다."""
    cell.fill.solid(); cell.fill.fore_color.rgb = rgb(cell_bg[0])
    cell.margin_left = cell.margin_right = Pt(7); cell.margin_top = cell.margin_bottom = Pt(5)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    for tok in re.split(r"(\*\*.+?\*\*|`[^`]+`)", md_text):
        if not tok:
            continue
        r = p.add_run()
        bold, col = bold_all, color
        if tok.startswith("**"):
            tok, bold = tok[2:-2], True
        elif tok.startswith("`"):
            tok = tok[1:-1]; col = TAG_COLOR.get(tok, col); bold = True
        r.text = tok; r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = rgb(col)


def add_table_slide(prs, title, header, body):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    W, H = prs.slide_width, prs.slide_height
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), W - Inches(1.0), Inches(0.6))
    tb.text_frame.margin_left = 0
    r = tb.text_frame.paragraphs[0].add_run(); r.text = title
    r.font.name = FONT; r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = rgb(ACC)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.0), W - Inches(1.0), Pt(2))
    bar.fill.solid(); bar.fill.fore_color.rgb = rgb(ACC); bar.line.fill.background(); bar.shadow.inherit = False

    ncol, nrow = len(header), len(body) + 1
    longest = max(len(re.sub(r"[*`]", "", c)) for row in [header] + body for c in row)
    size = 14 if nrow <= 6 and longest < 90 else 13 if longest < 140 else 12
    shape = slide.shapes.add_table(nrow, ncol, Inches(0.5), Inches(1.2), W - Inches(1.0), Inches(0.4) * nrow)
    tbl = shape.table
    tblPr = tbl._tbl.tblPr; tblPr.set("firstRow", "1"); tblPr.set("bandRow", "0")
    # 열 너비: 열별 최대 글자 수에 비례(최소 12%)
    weights = [max(12, min(60, max(len(re.sub(r"[*`]", "", row[i])) for row in [header] + body))) for i in range(ncol)]
    total = sum(weights); avail = W - Inches(1.0)
    for i, w in enumerate(weights):
        tbl.columns[i].width = int(avail * w / total)
    global cell_bg
    for i, h in enumerate(header):
        cell_bg = (ACC,); c = tbl.cell(0, i); set_borders(c); fill_cell(c, h, size + 0.5, color="FFFFFF", bold_all=True)
    for r_i, row in enumerate(body, start=1):
        cell_bg = ("FFFFFF" if r_i % 2 else BAND,)
        for i in range(ncol):
            c = tbl.cell(r_i, i); set_borders(c); fill_cell(c, row[i] if i < len(row) else "", size)
    return slide


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
for svg in svgs:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    root = ET.parse(svg).getroot()
    walk(slide, root, {})
    print(f"[INFO] {svg.name}: {len(slide.shapes)} shapes")
for title, header, body in md_tables(MD):
    add_table_slide(prs, title, header, body)
    print(f"[INFO] table slide: {title} ({len(body)} rows x {len(header)} cols)")
out = HERE / "07_slides.pptx"
prs.save(out)
print(f"[INFO] wrote {out} ({len(prs.slides)} slides: {len(svgs)} figures + tables, all editable)")
