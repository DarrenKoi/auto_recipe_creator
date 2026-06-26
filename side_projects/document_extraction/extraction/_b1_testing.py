"""B1 스모크 테스트용 합성 harvest 번들 생성기.

`harvest/harvest_pdf.py` 가 떠내는 번들 포맷을 그대로 흉내 내, 사외(Mac)에서도
loader/structure/chunkers 를 실데이터·PyMuPDF 없이 검증할 수 있게 한다.

실데이터 형태를 일부러 반영한다(코드리뷰 afe9cc0 반영):
- 절차 블록은 **줄(line) 단위로 분리**(각 Step 이 별도 line) - 단순 space-join 회귀 방지.
- 에러표 header 는 **'Error Code'** 같은 다단어(정확일치 회귀 방지).
- param 표는 **pymupdf + pdfplumber 중복 출처**(이중 chunk 회귀 방지).
- 같은 그림(xref)을 **여러 페이지에 반복 배치**(cross-page dedup 회귀 방지).
"""

import json
from pathlib import Path


def _line(bbox, size, font, text):
    return {"bbox": list(bbox), "spans": [
        {"bbox": list(bbox), "size": float(size), "flags": 0, "font": font, "text": text}]}


def _block(bbox, lines):
    return {"number": 0, "type": 0, "bbox": list(bbox), "lines": lines}


def _fig_ref():
    return {"file": "by_xref/xref_000016.png", "ext": "png", "width": 40, "height": 40,
            "colorspace": 3, "xref": 16, "bboxes_on_page": [[350, 60, 420, 130]]}


def write_synthetic_bundle(root: Path) -> Path:
    """합성 harvest 번들을 root 아래에 쓴다. 번들 루트 경로를 반환."""
    for sub in ("text", "tables", "figures", "figures/by_xref", "render", "links", "done"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    (root / "metadata.json").write_text(json.dumps(
        {"source_pdf": "syn", "page_count": 3, "metadata": {}}), encoding="utf-8")
    (root / "toc.json").write_text(json.dumps([
        [1, "Chapter 1 Overview", 1, {}],
        [2, "1.1 Setup", 1, {}],
        [1, "Chapter 2 Errors", 3, {}],
    ]), encoding="utf-8")
    (root / "figures/by_xref/xref_000016.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")

    # ── page 1: heading + 본문 + param 표(중복 출처) + figure ──
    p1 = {"width": 612, "height": 792, "blocks": [
        _block([72, 60, 300, 82], [_line([72, 60, 300, 82], 18.0, "Bold", "1.1 Setup")]),
        _block([72, 100, 520, 140], [_line([72, 100, 520, 140], 9.96, "Reg",
            "The alignment system aligns the wafer before measurement.")]),
    ]}
    (root / "text/page_0001.json").write_text(json.dumps(p1), encoding="utf-8")
    (root / "text/page_0001.txt").write_text(
        "1.1 Setup\nThe alignment system aligns the wafer before measurement.\n", encoding="utf-8")
    _param_rows = [["Parameter", "Range"], ["Focus", "0-100"], ["Gain", "1-5"]]
    (root / "tables/page_0001.json").write_text(json.dumps([
        {"source": "pymupdf", "bbox": [72, 200, 400, 280], "rows": _param_rows,
         "row_count": 3, "col_count": 2},
        {"source": "pdfplumber", "bbox": None, "rows": _param_rows,  # 동일 표 2차 추출
         "row_count": 3, "col_count": 2},
    ]), encoding="utf-8")
    (root / "figures/page_0001.json").write_text(json.dumps([_fig_ref()]), encoding="utf-8")
    (root / "render/page_0001.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")

    # ── page 2: 절차(각 Step 별도 line) + 같은 figure 반복 ──
    p2 = {"width": 612, "height": 792, "blocks": [
        _block([72, 60, 300, 82], [_line([72, 60, 300, 82], 16.0, "Bold", "2.1 Power-on Procedure")]),
        _block([72, 100, 520, 200], [
            _line([72, 100, 520, 118], 10.0, "Reg", "Step 1: Turn on the main power."),
            _line([72, 120, 520, 138], 10.0, "Reg", "Step 2: Load the wafer."),
            _line([72, 140, 520, 158], 10.0, "Reg", "Step 3: Start alignment."),
        ]),
    ]}
    (root / "text/page_0002.json").write_text(json.dumps(p2), encoding="utf-8")
    (root / "text/page_0002.txt").write_text(
        "2.1 Power-on Procedure\nStep 1: Turn on the main power.\n"
        "Step 2: Load the wafer.\nStep 3: Start alignment.\n", encoding="utf-8")
    (root / "figures/page_0002.json").write_text(json.dumps([_fig_ref()]), encoding="utf-8")
    (root / "render/page_0002.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")

    # ── page 3: 에러코드 표('Error Code' 다단어 header + 빈/None 행 섞임) ──
    p3 = {"width": 612, "height": 792, "blocks": [
        _block([72, 60, 300, 82], [_line([72, 60, 300, 82], 16.0, "Bold", "Error Codes")]),
    ]}
    (root / "text/page_0003.json").write_text(json.dumps(p3), encoding="utf-8")
    (root / "text/page_0003.txt").write_text("Error Codes\n", encoding="utf-8")
    (root / "tables/page_0003.json").write_text(json.dumps([
        {"source": "pymupdf", "bbox": [72, 120, 540, 260],
         "rows": [["Error Code", "Meaning", "Action"],
                  ["E9006", "Align fail", "Re-center and retry"],
                  None,                                   # ragged: None 행
                  ["E1002", "Focus fail", "Adjust focus offset"]],
         "row_count": 4, "col_count": 3},
    ]), encoding="utf-8")
    (root / "render/page_0003.png").write_bytes(b"\x89PNG\r\n\x1a\n placeholder")

    (root / "manifest.json").write_text(json.dumps({
        "per_page": [
            {"page": 1, "text": True, "tables": 2, "figures": 1, "rendered": True, "size": [612, 792]},
            {"page": 2, "text": True, "tables": 0, "figures": 1, "rendered": True, "size": [612, 792]},
            {"page": 3, "text": True, "tables": 1, "figures": 0, "rendered": True, "size": [612, 792]},
        ],
        "failures": [],
        "summary": {"pages_processed": 3, "pages_with_text": 3, "total_tables": 3},
    }), encoding="utf-8")
    return root
