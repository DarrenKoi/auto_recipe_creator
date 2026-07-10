"""evidence(ExtractionResult) -> Marp Markdown 생성 (Stage 5).

분기 규칙(marp_roundtrip_design.md Stage 3):
    텍스트류(Marp 네이티브): title, body/list, table, formula, footer
    래스터류(crop 재삽입):   chart, figure/image, shape, 그 외(other)

차트는 이중 트랙: (a) 추출된 데이터(라벨/값)를 화자 노트(HTML 주석)에 보존,
(b) 화면 충실도는 원본 crop 이미지로(crop_lookup 에 경로가 있을 때). crop 이 없으면
보이는 라벨/값을 작은 표로 대체하고 노트에 한계를 표기한다.

값 창작 금지(Stage 5 계약): 추출된 evidence 만 사용하고, 읽을 수 없으면 표기하지
않는다(빈 칸으로 둠).
"""

from side_projects.document_extraction.extraction.schemas import ExtractionResult


def frontmatter_for_theme(theme: str = "default") -> str:
    """Marp 프론트매터를 만든다.

    theme: marp 내장 테마('default'/'gaia'/'uncover') 또는 커스텀 테마 이름
    (예: 'doc-restore' — 렌더 시 marp-cli --theme <css> 로 함께 등록해야 한다).
    """
    name = (theme or "default").strip() or "default"
    return f"---\nmarp: true\ntheme: {name}\npaginate: true\n---\n"


def _cell(value) -> str:
    """GFM 표 셀 escape: 파이프는 \\| 로, 줄바꿈은 <br> 로(표 깨짐 방지)."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _md_table(header: list, rows: list) -> list:
    """header/rows -> Marp(GFM) 마크다운 표 라인들.

    열 수는 header 와 가장 긴 row 중 큰 값으로 잡아, row 가 header 보다 길어도
    데이터를 잘라내지 않는다(부족한 header 이름은 colN 으로 보강). 셀은 escape.
    """
    lines: list = []
    n_cols = len(header)
    for row in rows:
        n_cols = max(n_cols, len(row))
    if n_cols == 0:
        return lines

    cols = [str(h) for h in header] + [
        f"col{i + 1}" for i in range(len(header), n_cols)
    ]
    lines.append("| " + " | ".join(_cell(c) for c in cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        cells = [_cell(c) for c in row]
        if len(cells) < n_cols:  # 모자라면 빈 칸 패딩(값 창작 아님)
            cells += [""] * (n_cols - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _title_text(result: ExtractionResult) -> str:
    for region in result.regions:
        if region.type == "title" and region.text.strip():
            return region.text.strip()
    return ""


def evidence_to_marp(result: ExtractionResult, *, crop_lookup: dict | None = None) -> str:
    """ExtractionResult 1장을 Marp 슬라이드 1개(프론트매터 제외)로 변환한다.

    crop_lookup: {region_id -> 이미지 경로}. 차트/래스터 영역 재삽입에 사용.
    """
    crop_lookup = crop_lookup or {}
    lines: list = []
    notes: list = []  # 화자 노트(HTML 주석)로 모을 부가 데이터

    # 제목
    title = _title_text(result)
    if title:
        lines.append(f"# {title}")
        lines.append("")

    # 본문(텍스트 region, 제목 제외) -> 불릿
    for region in result.regions:
        if region.type == "title" or not region.text.strip():
            continue
        for raw_line in region.text.splitlines():
            text = raw_line.strip()
            if text:
                lines.append(f"- {text}")
    if lines and lines[-1] != "":
        lines.append("")

    # 표 -> Marp 네이티브 표
    for table in result.tables:
        if table.title:
            lines.append(f"**{table.title}**")
        table_lines = _md_table(table.header, table.cells)
        if table_lines:
            lines.extend(table_lines)
            lines.append("")

    # 수식 -> KaTeX
    for formula in result.formulas:
        if formula.latex.strip():
            lines.append(f"$$ {formula.latex.strip()} $$")
            lines.append("")

    # 차트 -> 래스터 재삽입(crop 있으면) 또는 데이터 표 대체
    for chart in result.charts:
        if chart.title:
            lines.append(f"**{chart.title}**")
        crop_path = crop_lookup.get(chart.region_id)
        if crop_path:
            lines.append(f"![w:600]({crop_path})")
        else:
            # crop 없음: 보이는 라벨/값을 작은 표로 대체
            if chart.legend_labels or chart.visible_values:
                rows = []
                for i, label in enumerate(chart.legend_labels):
                    value = chart.visible_values[i] if i < len(chart.visible_values) else ""
                    rows.append([label, value])
                lines.extend(_md_table(["series", "value"], rows))
            notes.append(f"chart '{chart.title or chart.region_id}': original crop not available")
        if chart.trend_summary:
            lines.append(f"_{chart.trend_summary}_")
        lines.append("")
        # 데이터 트랙 보존: 차트 데이터(라벨/값)를 노트에 동봉
        if chart.legend_labels or chart.visible_values:
            notes.append(
                f"chart data {chart.region_id}: legend={chart.legend_labels} "
                f"values={chart.visible_values}"
            )

    # 미해결/저신뢰는 노트로 표기(사람 review 용)
    for item in result.unresolved:
        notes.append(f"unresolved: {item}")

    slide = "\n".join(lines).rstrip()
    if notes:
        note_block = "\n".join(notes)
        slide = f"{slide}\n\n<!--\n{note_block}\n-->"
    return slide.strip()


def results_to_deck(
    results: list,
    *,
    crop_lookups: dict | None = None,
    with_frontmatter: bool = True,
    theme: str = "default",
) -> str:
    """여러 ExtractionResult 를 하나의 Marp deck(.md 본문)으로 합친다.

    crop_lookups: {screenshot_id -> {region_id -> 이미지경로}} (선택).
    theme: 프론트매터 theme 이름(커스텀이면 렌더 시 --theme CSS 필요).
    슬라이드는 Marp 의 `---` 로 구분한다.
    """
    crop_lookups = crop_lookups or {}
    slides = []
    for result in results:
        lookup = crop_lookups.get(result.screenshot_id, {})
        slide = evidence_to_marp(result, crop_lookup=lookup)
        if slide.strip():  # 빈 슬라이드는 deck 에서 제외(빈 페이지/중복 구분자 방지)
            slides.append(slide)

    body = "\n\n---\n\n".join(slides)
    if not with_frontmatter:
        return body
    return frontmatter_for_theme(theme) + "\n" + body + "\n"


__all__ = ["evidence_to_marp", "frontmatter_for_theme", "results_to_deck"]
