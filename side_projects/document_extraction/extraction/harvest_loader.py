"""C1: harvest 번들 -> PageModel (순수 로더, VLM/PyMuPDF 불필요).

`harvest/harvest_pdf.py` 가 떠둔 번들(`<stem>/text|tables|figures|render|toc|manifest`)을
읽어 PageModel 목록으로 만든다. 손상/누락 파일은 조용히 비우지 않고 load_warnings 에
기록한다(rag_db_plan 의 "보이는 것만, 창작 금지" 원칙의 로더판).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Span:
    """텍스트 span 한 개 (get_text dict 의 span)."""

    text: str = ""
    bbox: list = field(default_factory=list)
    size: float = 0.0
    font: str = ""
    flags: int = 0


@dataclass
class Block:
    """텍스트 block 한 개 (type 0). 여러 span 의 텍스트를 합친다."""

    bbox: list = field(default_factory=list)
    spans: list = field(default_factory=list)
    text: str = ""
    is_heading: bool = False         # structure 가 채움
    parent_heading: str = ""         # structure 가 채움

    @property
    def max_size(self) -> float:
        """블록 내 최대 span 폰트 크기 (heading 판정용)."""
        return max((s.size for s in self.spans), default=0.0)


@dataclass
class PageModel:
    """harvest 한 페이지의 구조화 입력."""

    page_no: int
    size: list = field(default_factory=list)
    blocks: list = field(default_factory=list)      # 채움: 다음 단계
    plain_text: str = ""
    tables: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    links: list = field(default_factory=list)
    render_path: str = ""
    has_text: bool = False
    section_path: list = field(default_factory=list)  # structure 가 채움
    load_warnings: list = field(default_factory=list)


@dataclass
class Bundle:
    root: Path
    doc_id: str
    toc: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    pages: list = field(default_factory=list)


def load_bundle(root: Path) -> Bundle:
    """번들 루트 -> Bundle. manifest 의 per_page 를 권위 인덱스로 페이지를 로드."""
    root = Path(root)
    doc_id = root.name

    toc = _read_json(root / "toc.json") if (root / "toc.json").exists() else []
    metadata = _read_json(root / "metadata.json") if (root / "metadata.json").exists() else {}

    manifest = _read_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    per_page = manifest.get("per_page", [])

    pages = []
    for rec in per_page:
        if not isinstance(rec.get("page"), int):
            print(f"[WARNING] manifest per_page 레코드에 정수 page 없음, 건너뜀: {rec.get('page')!r}")
            continue
        pages.append(_load_page(root, rec))

    return Bundle(root=root, doc_id=doc_id, toc=toc, metadata=metadata, pages=pages)


def _load_page(root: Path, rec: dict) -> PageModel:
    page_no = rec.get("page")
    stem = f"page_{page_no:04d}"
    page = PageModel(page_no=page_no, size=rec.get("size") or [])

    txt = root / "text" / f"{stem}.txt"
    if txt.exists():
        page.plain_text = txt.read_text(encoding="utf-8")
    page.has_text = bool(page.plain_text.strip())

    tjson = root / "text" / f"{stem}.json"
    if tjson.exists():
        page.blocks = _parse_blocks(_read_json(tjson))

    # tables: harvest dict 그대로 + 누락/손상 경고
    if rec.get("tables"):
        tpath = root / "tables" / f"{stem}.json"
        if tpath.exists():
            page.tables = _read_json(tpath)
        else:
            page.load_warnings.append(f"tables 누락: {tpath.name}")

    # figures: file 상대경로를 번들 기준 절대경로(path)로 해석
    if rec.get("figures"):
        fpath = root / "figures" / f"{stem}.json"
        if fpath.exists():
            figs = _read_json(fpath)
            for f in figs:
                f["path"] = str(root / "figures" / f.get("file", ""))
            page.figures = figs
        else:
            page.load_warnings.append(f"figures 누락: {fpath.name}")

    # render PNG
    if rec.get("rendered"):
        rpath = root / "render" / f"{stem}.png"
        if rpath.exists():
            page.render_path = str(rpath)
        else:
            page.load_warnings.append(f"render 누락: {rpath.name}")

    # links (옵션, manifest 플래그 없이 존재 여부로 로드)
    lpath = root / "links" / f"{stem}.json"
    if lpath.exists():
        page.links = _read_json(lpath)

    return page


def _parse_blocks(text_dict: dict) -> list:
    """get_text('dict') -> Block 목록 (type 0 텍스트 block 만).

    줄바꿈을 보존한다: line 내 span 은 이어 붙이고(line 텍스트), block 텍스트는 line 을
    '\\n' 로 합친다. (실데이터에선 각 시각적 line 이 별도 'lines' 항목이라, 단순 space-join
    하면 'Step 1\\nStep 2...' 의 줄 구분이 사라져 절차 감지가 무력화된다.)
    """
    blocks = []
    for b in text_dict.get("blocks", []):
        if b.get("type") != 0:
            continue  # 이미지 block 은 figures/ 로 따로 수확됨
        spans = []
        line_texts = []
        for line in b.get("lines", []):
            line_spans = line.get("spans", [])
            for s in line_spans:
                spans.append(Span(text=s.get("text", ""), bbox=list(s.get("bbox") or []),
                                  size=float(s.get("size") or 0.0), font=s.get("font", ""),
                                  flags=int(s.get("flags") or 0)))
            line_texts.append("".join(s.get("text", "") for s in line_spans))
        text = "\n".join(line_texts).strip()
        if not text and not spans:
            continue
        blocks.append(Block(bbox=list(b.get("bbox") or []), spans=spans, text=text))
    return blocks


__all__ = ["Block", "Bundle", "PageModel", "Span", "load_bundle"]
