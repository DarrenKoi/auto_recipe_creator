"""보고서 Markdown 파서 (docx/pptx 빌더 공용).

`docs/project_progress/*.md` 를 블록 리스트로 파싱하여, .docx/.pptx 빌더가 동일한
source-of-truth(Markdown)로부터 산출물을 생성하게 한다. 즉 보고서 내용을 고칠 때는
.md 만 수정하면 두 산출물에 함께 반영된다(README 작성 원칙과 일치).

지원 블록: heading(#~###), paragraph, blockquote(>), unordered/ordered list(중첩 1단계),
table(| ... |), fenced code block(```).
인라인: **bold**, `code`, [text](url) -> text.

순수 표준 라이브러리만 사용한다(외부 의존 없음).
"""

import re

# 인라인: **bold** | `code` | [text](url)
_INLINE = re.compile(r"\*\*(.+?)\*\*|`([^`]+?)`|\[([^\]]+?)\]\([^)]*?\)")
_LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_inline(text):
    """문자열을 [{text, bold, mono}] 세그먼트로 분해한다.

    bold/inline-code/link 를 식별하고, 링크는 표시 텍스트만 남긴다.
    매칭이 없으면 전체를 단일 평문 세그먼트로 반환한다.
    """
    segs = []
    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            segs.append({"text": text[pos:m.start()], "bold": False, "mono": False})
        if m.group(1) is not None:
            segs.append({"text": m.group(1), "bold": True, "mono": False})
        elif m.group(2) is not None:
            segs.append({"text": m.group(2), "bold": False, "mono": True})
        else:  # link -> 표시 텍스트만
            segs.append({"text": m.group(3), "bold": False, "mono": False})
        pos = m.end()
    if pos < len(text):
        segs.append({"text": text[pos:], "bold": False, "mono": False})
    if not segs:
        segs.append({"text": text, "bold": False, "mono": False})
    return segs


def _is_block_start(line):
    s = line.strip()
    if s.startswith("#") or s.startswith(">") or s.startswith("```") or s.startswith("|"):
        return True
    if _LIST_RE.match(line):
        return True
    return False


def _is_table_sep(line):
    """표 구분선(|---|---|) 판정."""
    s = line.strip()
    return bool(re.match(r"^\|?[\s:|-]+\|?$", s)) and "-" in s


def parse_markdown(text):
    """Markdown 문자열을 블록 dict 리스트로 파싱한다."""
    lines = text.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # fenced code block
        if stripped.startswith("```"):
            code = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # 닫는 fence 건너뜀
            blocks.append({"type": "code", "lines": code})
            continue

        # heading
        m = _HEAD_RE.match(stripped)
        if m:
            blocks.append({"type": "h", "level": len(m.group(1)), "text": m.group(2).strip()})
            i += 1
            continue

        # blockquote (연속 > 라인을 한 콜아웃으로)
        if stripped.startswith(">"):
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].lstrip())
                i += 1
            blocks.append({"type": "quote", "text": " ".join(q for q in quote if q)})
            continue

        # table
        if stripped.startswith("|") and i + 1 < n and _is_table_sep(lines[i + 1]):
            headers = [c.strip() for c in stripped.strip().strip("|").split("|")]
            i += 2  # 헤더 + 구분선 건너뜀
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            blocks.append({"type": "table", "headers": headers, "rows": rows})
            continue

        # list (중첩 1단계 + 연속 들여쓰기 라인 병합)
        lm = _LIST_RE.match(line)
        if lm:
            ordered = bool(re.match(r"\d+\.", lm.group(2)))
            items = []
            while i < n:
                lm2 = _LIST_RE.match(lines[i])
                if lm2:
                    indent = len(lm2.group(1))
                    items.append({
                        "level": 1 if indent >= 2 else 0,
                        "text": lm2.group(3).strip(),
                        "marker": lm2.group(2),
                    })
                    i += 1
                elif lines[i].strip() == "":
                    break
                elif re.match(r"^\s+\S", lines[i]) and items:
                    items[-1]["text"] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            blocks.append({"type": "list", "ordered": ordered, "items": items})
            continue

        # paragraph
        para = [stripped]
        i += 1
        while i < n and lines[i].strip() and not _is_block_start(lines[i]):
            para.append(lines[i].strip())
            i += 1
        blocks.append({"type": "p", "text": " ".join(para)})
    return blocks


def load_report_docs(base_dir):
    """00~05 보고서 .md 를 순서대로 (slug, blocks) 리스트로 로드한다."""
    import pathlib

    base = pathlib.Path(base_dir)
    names = [
        "00_executive_summary.md",
        "01_vlm_deployment.md",
        "02_workflow_1.md",
        "03_workflow_2.md",
        "04_workflow_3.md",
        "05_workflow_4.md",
    ]
    docs = []
    for name in names:
        path = base / name
        text = path.read_text(encoding="utf-8")
        docs.append({"name": name, "blocks": parse_markdown(text)})
    return docs
