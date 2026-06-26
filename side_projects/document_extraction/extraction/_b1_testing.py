"""B1 스모크 테스트용 합성 harvest 번들 생성기.

`harvest/harvest_pdf.py` 가 떠내는 번들 포맷을 그대로 흉내 내, 사외(Mac)에서도
loader/structure/chunkers 를 실데이터·PyMuPDF 없이 검증할 수 있게 한다.

번들 구성(3페이지):
- page 1: heading(큰 폰트) + 본문 + param 표 + figure 1개
- page 2: 절차(Step 1/2/3) 한 블록
- page 3: 에러코드 표(Code/Meaning/Action)
"""

import json
from pathlib import Path


def _text_dict(width, height, blocks):
    """get_text('dict') 모양의 dict 를 만든다. blocks = [(bbox, [(size, font, text)])]."""
    out_blocks = []
    for bnum, (bbox, spans) in enumerate(blocks):
        out_blocks.append({
            "number": bnum, "type": 0, "bbox": list(bbox),
            "lines": [{
                "bbox": list(bbox),
                "spans": [{"bbox": list(bbox), "size": float(sz), "flags": 0,
                           "font": font, "text": text} for (sz, font, text) in spans],
            }],
        })
    return {"width": width, "height": height, "blocks": out_blocks}


def write_synthetic_bundle(root: Path) -> Path:
    """합성 harvest 번들을 root 아래에 쓴다. 번들 루트 경로를 반환."""
    for sub in ("text", "tables", "figures", "figures/by_xref", "render", "links", "done"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # ── 문서 단위 ──
    (root / "metadata.json").write_text(json.dumps(
        {"source_pdf": "syn", "page_count": 3, "metadata": {}}), encoding="utf-8")
    # toc: simple=False 모양 [level, title, page, dest]
    (root / "toc.json").write_text(json.dumps([
        [1, "Chapter 1 Overview", 1, {}],
        [2, "1.1 Setup", 1, {}],
        [1, "Chapter 2 Errors", 3, {}],
    ]), encoding="utf-8")

    # ── page 1: heading + 본문 + 표 + figure ──
    p1 = _text_dict(612, 792, [
        ([72, 60, 300, 82], [(18.0, "Bold", "1.1 Setup")]),
        ([72, 100, 520, 140], [(10.0, "Reg", "The alignment system aligns the wafer before measurement.")]),
    ])
    (root / "text/page_0001.json").write_text(json.dumps(p1), encoding="utf-8")
    (root / "text/page_0001.txt").write_text(
        "1.1 Setup\nThe alignment system aligns the wafer before measurement.\n", encoding="utf-8")
    (root / "tables/page_0001.json").write_text(json.dumps([
        {"source": "pymupdf", "bbox": [72, 200, 400, 280],
         "rows": [["Parameter", "Range"], ["Focus", "0-100"], ["Gain", "1-5"]],
         "row_count": 3, "col_count": 2},
    ]), encoding="utf-8")
    (root / "figures/page_0001.json").write_text(json.dumps([
        {"file": "by_xref/xref_000016.png", "ext": "png", "width": 40, "height": 40,
         "colorspace": 3, "xref": 16, "bboxes_on_page": [[350, 60, 420, 130]]},
    ]), encoding="utf-8")
    (root / "figures/by_xref/xref_000016.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")
    (root / "render/page_0001.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")

    # ── page 2: 절차 한 블록(Step 1/2/3) ──
    p2 = _text_dict(612, 792, [
        ([72, 60, 300, 82], [(16.0, "Bold", "2.1 Power-on Procedure")]),
        ([72, 100, 520, 200], [(10.0, "Reg",
            "Step 1: Turn on the main power.\nStep 2: Load the wafer.\nStep 3: Start alignment.")]),
    ])
    (root / "text/page_0002.json").write_text(json.dumps(p2), encoding="utf-8")
    (root / "text/page_0002.txt").write_text(
        "2.1 Power-on Procedure\nStep 1: Turn on the main power.\n"
        "Step 2: Load the wafer.\nStep 3: Start alignment.\n", encoding="utf-8")
    (root / "render/page_0002.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")

    # ── page 3: 에러코드 표 ──
    p3 = _text_dict(612, 792, [
        ([72, 60, 300, 82], [(16.0, "Bold", "Error Codes")]),
    ])
    (root / "text/page_0003.json").write_text(json.dumps(p3), encoding="utf-8")
    (root / "text/page_0003.txt").write_text("Error Codes\n", encoding="utf-8")
    (root / "tables/page_0003.json").write_text(json.dumps([
        {"source": "pymupdf", "bbox": [72, 120, 540, 260],
         "rows": [["Code", "Meaning", "Action"],
                  ["E9006", "Align fail", "Re-center and retry"],
                  ["E1002", "Focus fail", "Adjust focus offset"]],
         "row_count": 3, "col_count": 3},
    ]), encoding="utf-8")
    (root / "render/page_0003.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")

    # ── manifest ──
    (root / "manifest.json").write_text(json.dumps({
        "per_page": [
            {"page": 1, "text": True, "tables": 1, "figures": 1, "rendered": True, "size": [612, 792]},
            {"page": 2, "text": True, "tables": 0, "figures": 0, "rendered": True, "size": [612, 792]},
            {"page": 3, "text": True, "tables": 1, "figures": 0, "rendered": True, "size": [612, 792]},
        ],
        "failures": [],
        "summary": {"pages_processed": 3, "pages_with_text": 3, "total_tables": 2},
    }), encoding="utf-8")
    return root
