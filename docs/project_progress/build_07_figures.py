"""img/07/fig*.svg -> PNG(2400px) + 07_figures.pptx (16:9 슬라이드, 한 장에 도표 하나).

SVG 는 1920x1080 슬라이드 캔버스로 그려져 있어 PowerPoint 에 SVG 를 그대로 삽입해도 되고,
여기서 만든 PPTX 를 열어 슬라이드를 복사해 써도 된다. PNG 는 md/HTML 삽입용.

실행 (rsvg-convert 는 brew, python-pptx 는 1회성으로 얹는다):
    uv run --with python-pptx python docs/project_progress/build_07_figures.py
"""
import pathlib, subprocess
from pptx import Presentation
from pptx.util import Inches

HERE = pathlib.Path(__file__).resolve().parent / "img" / "07"
PNG_WIDTH = 2400

svgs = sorted(HERE.glob("fig*.svg"))
for svg in svgs:
    png = svg.with_suffix(".png")
    subprocess.run(["rsvg-convert", "-w", str(PNG_WIDTH), "-o", str(png), str(svg)], check=True)
    print(f"[INFO] {png.name} ({png.stat().st_size / 1024:.0f} KB)")

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
for svg in svgs:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    slide.shapes.add_picture(str(svg.with_suffix(".png")), 0, 0, prs.slide_width, prs.slide_height)
out = HERE / "07_figures.pptx"
prs.save(out)
print(f"[INFO] wrote {out} ({len(svgs)} slides)")
