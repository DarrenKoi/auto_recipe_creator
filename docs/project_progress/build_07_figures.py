"""img/07/fig*.svg -> PNG(2400px, md/HTML 용) + 07_figures.pptx (편집 가능한 PPT 도형).

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
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt, Emu

HERE = pathlib.Path(__file__).resolve().parent / "img" / "07"
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

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
for svg in svgs:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    root = ET.parse(svg).getroot()
    walk(slide, root, {})
    print(f"[INFO] {svg.name}: {len(slide.shapes)} shapes")
out = HERE / "07_figures.pptx"
prs.save(out)
print(f"[INFO] wrote {out} ({len(svgs)} slides, native shapes)")
