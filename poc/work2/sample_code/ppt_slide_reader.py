"""PPT 슬라이드 이미지 → 구조화된 내용 추출.

이 폴더는 독립 실행 가능 — 프로젝트 내 다른 모듈에 의존하지 않는다.
동료에게 이 폴더만 복사해서 공유하면 바로 사용 가능.

사용법:
    pip install requests
    python ppt_slide_reader.py

    # 이 폴더에 .jpg 또는 .png 이미지를 넣어두면 자동으로 읽는다.

파이프라인 (2단계):
    1단계: paddleocr-vl-1.5 로 이미지에서 텍스트를 정확히 추출 (OCR)
    2단계: ui-venus-1.5-8b 가 OCR 텍스트를 참조하여 슬라이드 구조를 분석 (VLM)

필요 패키지: requests
"""

import sys
import time
from pathlib import Path

from vlm_common import (
    OCR_SERVICE,
    OCR_URL,
    VLM_MODEL,
    VLM_SERVICE,
    VLM_URL,
    build_combined_ocr,
    clean_model_text,
    collect_images,
    get_output_dir,
    parse_json_response,
    print_token_usage,
    vlm_chat,
)

# ──────────────────────────────────────────────
# VLM 프롬프트
# ──────────────────────────────────────────────
SYSTEM_MESSAGE = """\
You are an expert document analyst specializing in technical presentation slides.
Your task is to extract and structure ALL visible content from a PPT slide screenshot.
Be thorough — capture every piece of text, number, label, and visual element you can see.
Respond ONLY with a JSON object (no markdown fences, no extra text).

IMPORTANT: An OCR engine has already extracted the text from this slide.
The OCR text is provided below as a reference. Use it to ensure accurate spelling
of all words, names, numbers, and technical terms. When the OCR text and your
visual reading conflict, prefer the OCR text for exact spelling."""

USER_PROMPT_TEMPLATE = """\
Below is the OCR-extracted text from this slide (use it as a spelling reference):
--- OCR TEXT ---
{ocr_text}
--- END OCR TEXT ---

Now analyze this PPT slide screenshot and extract its full content into the following JSON structure.
Fill in every field you can identify. If a field is not present in the slide, use null.

{{
  "slide_title": "Main title text of the slide",
  "subtitle": "Subtitle or secondary heading if present",
  "body_text": [
    "Each bullet point or paragraph as a separate string",
    "Preserve the original text as closely as possible"
  ],
  "charts": [
    {{
      "chart_type": "bar|line|pie|scatter|table|diagram|flowchart|other",
      "chart_title": "Title or caption of the chart",
      "description": "What the chart shows — axes, categories, trends",
      "data_points": ["Key numbers, labels, or values visible in the chart"],
      "key_takeaway": "The main insight or conclusion from this chart"
    }}
  ],
  "tables": [
    {{
      "table_title": "Title or caption if any",
      "headers": ["column1", "column2"],
      "rows": [["val1", "val2"]]
    }}
  ],
  "images_and_diagrams": [
    {{
      "type": "photo|icon|diagram|logo|screenshot|other",
      "description": "What the image or diagram depicts"
    }}
  ],
  "footer_or_notes": "Page number, copyright, footnotes, or source citations",
  "overall_topic": "One-sentence summary describing the slide's purpose and context (do NOT repeat the title)",
  "key_insights": [
    "2-3 analytical insights or implications NOT already stated in body_text — interpret, don't repeat"
  ]
}}

Important:
- Use the OCR text above as the authoritative source for exact spelling of all words and numbers.
- For charts, try to read actual data values (numbers, percentages) where visible.
- Preserve technical terms, abbreviations, and units exactly as shown.
- Preserve any Korean text from the slide exactly as shown.
- NO DUPLICATION: each piece of text must appear in exactly ONE field.
  slide_title, subtitle, body_text, chart descriptions, table data, and footer must not overlap.
  key_insights must provide NEW analysis, not restate body_text.
  overall_topic must describe context/purpose, not copy the title.
- Write ALL your descriptions, summaries, and key_insights in Korean.
  Exception: if the slide is entirely in English with no Korean at all, respond in English."""


# ──────────────────────────────────────────────
# 분석
# ──────────────────────────────────────────────
def analyze_slide(image_path: Path) -> dict | None:
    """단일 슬라이드 이미지를 분석하여 구조화된 결과를 반환한다."""
    print(f"\n{'='*60}")
    print(f"[INFO] 분석 시작: {image_path.name}")
    print(f"{'='*60}")

    if not image_path.exists():
        print(f"[WARNING] 파일이 존재하지 않습니다: {image_path}")
        return None

    file_size_kb = image_path.stat().st_size / 1024
    print(f"[INFO] 파일 크기: {file_size_kb:.1f} KB")

    # ── 1단계: OCR 텍스트 추출 ──
    combined_ocr = build_combined_ocr(image_path)

    # ── 2단계: VLM 구조 분석 (OCR 텍스트 참조) ──
    print(f"[INFO] [VLM] 구조 분석 중... ({VLM_SERVICE})")
    user_prompt = USER_PROMPT_TEMPLATE.format(ocr_text=combined_ocr)

    start = time.time()
    try:
        raw_text, usage = vlm_chat(
            VLM_URL, VLM_MODEL, image_path, SYSTEM_MESSAGE, user_prompt, temperature=0.1,
        )
    except Exception as exc:
        print(f"[ERROR] VLM 호출 실패: {exc}")
        return None

    elapsed = time.time() - start
    print(f"[INFO] VLM 응답 시간: {elapsed:.1f}초")
    print_token_usage(usage)

    result = parse_json_response(raw_text, "구조 분석")
    if result is None:
        return {"raw_response": clean_model_text(raw_text), "_ocr_text": combined_ocr}

    print(f"[INFO] JSON 파싱 성공")
    result["_ocr_text"] = combined_ocr
    return result


# ──────────────────────────────────────────────
# 출력 / 저장
# ──────────────────────────────────────────────
def print_result(image_name: str, result: dict) -> None:
    """분석 결과를 보기 좋게 출력한다."""
    print(f"\n{'─'*60}")
    print(f"  결과: {image_name}")
    print(f"{'─'*60}")

    if "raw_response" in result:
        print(result["raw_response"])
        return

    title = result.get("slide_title")
    if title:
        print(f"\n■ 제목: {title}")
    subtitle = result.get("subtitle")
    if subtitle:
        print(f"  부제: {subtitle}")

    topic = result.get("overall_topic")
    if topic:
        print(f"\n■ 주제: {topic}")

    body = result.get("body_text") or []
    if body:
        print(f"\n■ 본문 ({len(body)}항목):")
        for i, text in enumerate(body, 1):
            print(f"  {i}. {text}")

    insights = result.get("key_insights") or []
    if insights:
        print(f"\n■ 핵심 인사이트:")
        for point in insights:
            print(f"  • {point}")

    charts = result.get("charts") or []
    if charts:
        print(f"\n■ 차트 ({len(charts)}개):")
        for i, chart in enumerate(charts, 1):
            print(f"  [{i}] {chart.get('chart_type', '?')} — {chart.get('chart_title', '제목 없음')}")
            desc = chart.get("description")
            if desc:
                print(f"      설명: {desc}")
            data = chart.get("data_points") or []
            if data:
                print(f"      데이터: {', '.join(str(d) for d in data)}")
            takeaway = chart.get("key_takeaway")
            if takeaway:
                print(f"      핵심: {takeaway}")

    tables = result.get("tables") or []
    if tables:
        print(f"\n■ 표 ({len(tables)}개):")
        for i, table in enumerate(tables, 1):
            print(f"  [{i}] {table.get('table_title', '제목 없음')}")
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            if headers:
                print(f"      헤더: {' | '.join(headers)}")
            for row in rows[:5]:
                print(f"      → {' | '.join(str(v) for v in row)}")
            if len(rows) > 5:
                print(f"      ... 외 {len(rows) - 5}행")

    images = result.get("images_and_diagrams") or []
    if images:
        print(f"\n■ 이미지/다이어그램 ({len(images)}개):")
        for img in images:
            print(f"  • [{img.get('type', '?')}] {img.get('description', '')}")

    footer = result.get("footer_or_notes")
    if footer:
        print(f"\n■ 푸터/노트: {footer}")

    print()


def _format_result_text(result: dict) -> str:
    """분석 결과를 사람이 읽기 좋은 텍스트로 변환한다."""
    if "raw_response" in result:
        return result["raw_response"]

    lines: list[str] = []

    title = result.get("slide_title")
    if title:
        lines.append(f"제목: {title}")
    subtitle = result.get("subtitle")
    if subtitle:
        lines.append(f"부제: {subtitle}")

    topic = result.get("overall_topic")
    if topic:
        lines.append(f"\n주제: {topic}")

    body = result.get("body_text") or []
    if body:
        lines.append(f"\n본문 ({len(body)}항목):")
        for i, text in enumerate(body, 1):
            lines.append(f"  {i}. {text}")

    insights = result.get("key_insights") or []
    if insights:
        lines.append("\n핵심 인사이트:")
        for point in insights:
            lines.append(f"  - {point}")

    charts = result.get("charts") or []
    if charts:
        lines.append(f"\n차트 ({len(charts)}개):")
        for i, chart in enumerate(charts, 1):
            lines.append(f"  [{i}] {chart.get('chart_type', '?')} - {chart.get('chart_title', '제목 없음')}")
            desc = chart.get("description")
            if desc:
                lines.append(f"      설명: {desc}")
            data = chart.get("data_points") or []
            if data:
                for d in data:
                    lines.append(f"      - {d}")
            takeaway = chart.get("key_takeaway")
            if takeaway:
                lines.append(f"      핵심: {takeaway}")

    tables = result.get("tables") or []
    if tables:
        lines.append(f"\n표 ({len(tables)}개):")
        for i, table in enumerate(tables, 1):
            lines.append(f"  [{i}] {table.get('table_title', '제목 없음')}")
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            if headers:
                lines.append(f"      {' | '.join(headers)}")
                lines.append(f"      {'-+-'.join('-' * len(h) for h in headers)}")
            for row in rows:
                lines.append(f"      {' | '.join(str(v) for v in row)}")

    images = result.get("images_and_diagrams") or []
    if images:
        lines.append(f"\n이미지/다이어그램 ({len(images)}개):")
        for img in images:
            lines.append(f"  - [{img.get('type', '?')}] {img.get('description', '')}")

    footer = result.get("footer_or_notes")
    if footer:
        lines.append(f"\n푸터/노트: {footer}")

    return "\n".join(lines)


def save_result(image_path: Path, result: dict) -> None:
    """분석 결과와 OCR 원문을 각각 별도 파일로 저장한다."""
    out_dir = get_output_dir(image_path)

    # VLM 분석 결과
    txt_path = out_dir / "result.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_format_result_text(result) + "\n")
    print(f"[INFO] 결과 저장: {out_dir.name}/result.txt")

    # OCR 원문
    ocr_text = result.get("_ocr_text", "").strip()
    if ocr_text:
        ocr_path = out_dir / "ocr_raw.txt"
        with open(ocr_path, "w", encoding="utf-8") as f:
            f.write(ocr_text + "\n")
        print(f"[INFO] OCR 저장: {out_dir.name}/ocr_raw.txt")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    """메인 실행."""
    image_paths = collect_images()

    print("[INFO] PPT 슬라이드 내용 분석기 시작")
    print(f"[INFO] OCR: {OCR_SERVICE} ({OCR_URL})")
    print(f"[INFO] VLM: {VLM_SERVICE} ({VLM_URL})")
    print(f"[INFO] 대상 이미지: {len(image_paths)}개")

    if not image_paths:
        print(f"[ERROR] 분석할 이미지가 없습니다.")
        print(f"[INFO] 이 폴더에 .jpg 또는 .png 이미지를 넣어주세요.")
        sys.exit(1)

    results = {}
    for image_path in image_paths:
        result = analyze_slide(image_path)
        if result is not None:
            results[image_path.name] = result
            print_result(image_path.name, result)
            save_result(image_path, result)

    print(f"\n{'='*60}")
    print(f"[INFO] 분석 완료: {len(results)}/{len(image_paths)}개 성공")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
