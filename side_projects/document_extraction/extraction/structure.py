"""C2: 섹션/heading 구조 복원 (순수, VLM 불필요).

- TOC(authoritative): 페이지 -> 섹션 경로(레벨별 제목)를 만든다.
- In-page heading: span 폰트 크기로 body 대비 큰 블록을 heading 으로 식별.
- assign_structure: 각 페이지에 section_path, 각 텍스트 block 에 parent_heading 을 채운다.

heading 연결은 rag_db_plan.md 의 context 보존 규칙(각 chunk 를 가장 가까운 heading 에
연결)을 만족시켜, 검색·인용 품질의 backbone 이 된다.
"""

# heading 판정: block 최대 폰트가 body 기준 폰트의 이 배율을 넘으면 heading.
HEADING_RATIO = 1.15


def build_section_index(toc: list, page_count: int) -> dict:
    """toc([level, title, page, ...]) -> {page_no: [상위..현재 섹션 제목]}.

    페이지 순서로 한 번만 훑으며 레벨 스택을 갱신한다(O(page + toc)).
    어떤 섹션도 시작되지 않은 앞부분 페이지는 빈 경로.
    """
    entries = []
    for e in toc or []:
        if len(e) >= 3 and isinstance(e[2], int):
            entries.append((int(e[0]), str(e[1]), int(e[2])))
    entries.sort(key=lambda x: x[2])  # page 기준 안정 정렬

    index = {}
    stack = []  # [(level, title)]
    ei = 0
    for page in range(1, page_count + 1):
        while ei < len(entries) and entries[ei][2] <= page:
            level, title, _ = entries[ei]
            # 같은/더 얕은 레벨이 오면 그 깊이까지 자르고 새로 쌓는다.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            ei += 1
        index[page] = [title for _, title in stack]
    return index


def detect_headings(page) -> list:
    """페이지 텍스트 block 중 heading 후보를 표시(block.is_heading)하고 그 목록을 반환.

    body 기준 폰트 = 페이지에서 글자수가 가장 많은 폰트 크기(본문이 지배적).
    그보다 HEADING_RATIO 배 이상 큰 block 을 heading 으로 본다.
    """
    if not page.blocks:
        return []
    # 크기별 누적 글자수 -> 지배 크기 = body baseline.
    # 실 PDF 는 kerning/subset 으로 9.96/10.0/10.04 처럼 흩어지므로 0.5 단위로 버킷팅해
    # histogram 파편화를 막는다.
    char_by_size = {}
    for b in page.blocks:
        for s in b.spans:
            bucket = round(s.size * 2) / 2
            char_by_size[bucket] = char_by_size.get(bucket, 0) + len(s.text)
    baseline = max(char_by_size, key=char_by_size.get) if char_by_size else 0.0

    headings = []
    for b in page.blocks:
        bucket = round(b.max_size * 2) / 2
        # 크기가 body 대비 크거나, body 와 같은 크기라도 굵게(bold flag 16)면 heading.
        is_big = baseline > 0 and bucket > baseline * HEADING_RATIO
        # bold 는 짧은 한 줄(<=80자)일 때만 heading 신호로 - bold 본문이 통째로
        # heading 으로 오인돼 region_text 가 비는 것을 막는다.
        is_bold = (bucket >= baseline and len(b.text.strip()) <= 80
                   and any(int(s.flags) & 16 for s in b.spans))
        b.is_heading = bool(is_big or is_bold)
        if b.is_heading:
            headings.append(b)
    return headings


def assign_structure(bundle) -> None:
    """bundle 의 각 페이지에 section_path + 각 block.parent_heading 을 채운다(in-place)."""
    page_count = len(bundle.pages)
    index = build_section_index(bundle.toc, page_count)
    for page in bundle.pages:
        page.section_path = list(index.get(page.page_no, []))
        headings = detect_headings(page)
        section_leaf = page.section_path[-1] if page.section_path else ""
        for b in page.blocks:
            if b.is_heading:
                b.parent_heading = section_leaf
                continue
            b.parent_heading = _nearest_heading_above(b, headings) or section_leaf


def _nearest_heading_above(block, headings) -> str:
    """block 위쪽에 가장 가까운 heading 텍스트. bbox=[x0,y0,x1,y1]."""
    if not block.bbox or not headings:
        return ""
    btop = block.bbox[1]
    above = []
    for h in headings:
        if h is block or not h.bbox:
            continue
        hbottom = h.bbox[3]
        if hbottom <= btop:
            above.append((btop - hbottom, h.text))
    if not above:
        return ""
    above.sort(key=lambda x: x[0])
    return above[0][1]


__all__ = ["HEADING_RATIO", "assign_structure", "build_section_index", "detect_headings"]
