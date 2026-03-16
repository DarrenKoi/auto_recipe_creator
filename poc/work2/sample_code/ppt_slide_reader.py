"""PPT 슬라이드 이미지 → 구조화된 내용 추출 샘플 코드.

사용법:
    # 프로젝트 루트에서 실행
    uv run python poc/work2/sample_code/ppt_slide_reader.py

    # sample_code/ 폴더에 .jpg 또는 .png 이미지를 넣어두면 자동으로 읽는다.

파이프라인 (2단계):
    1단계: paddleocr-vl-1.5 로 이미지에서 텍스트를 정확히 추출 (OCR)
    2단계: ui-venus-1.5-8b 가 OCR 텍스트를 참조하여 슬라이드 구조를 분석 (VLM)

    OCR 결과를 VLM 에게 참고 자료로 제공하므로 텍스트 오탈자가 크게 줄어든다.
"""

import json
import sys
import time
from pathlib import Path

from poc.work2.vlm_client import Work2VLMClient

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
VLM_SERVICE_SLUG = "ui-venus"
OCR_SERVICE_SLUG = "paddleocr-vl-1.5"
SAMPLE_DIR = Path(__file__).parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# sample_code/ 폴더 내 jpg/png 이미지를 자동 수집
IMAGE_PATHS = sorted(
    p for p in SAMPLE_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
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
  "overall_topic": "One-sentence summary of what this slide is about",
  "key_points": [
    "Top 3-5 key takeaways or important facts from the slide"
  ]
}}

Important:
- Use the OCR text above as the authoritative source for exact spelling of all words and numbers.
- For charts, try to read actual data values (numbers, percentages) where visible.
- Preserve technical terms, abbreviations, and units exactly as shown.
- Preserve any Korean text from the slide exactly as shown.
- Write ALL your descriptions, summaries, takeaways, and key_points in Korean.
  Exception: if the slide is entirely in English with no Korean at all, respond in English."""


def run_ocr(ocr_client: Work2VLMClient, image_path: Path) -> str:
    """PaddleOCR-VL 로 이미지에서 텍스트를 추출한다."""
    print(f"[INFO] [1단계] OCR 텍스트 추출 중... ({OCR_SERVICE_SLUG})")
    start = time.time()
    try:
        response = ocr_client.chat_with_image_path(
            image_path=image_path,
            system_message="",
            user_text="OCR:",
            temperature=0.0,
        )
    except Exception as exc:
        print(f"[WARNING] OCR 호출 실패 (VLM 단독으로 진행): {exc}")
        return ""

    elapsed = time.time() - start
    ocr_text = response.text.strip()
    print(f"[INFO] OCR 완료: {elapsed:.1f}초, {len(ocr_text)}자 추출")
    return ocr_text


def run_chart_ocr(ocr_client: Work2VLMClient, image_path: Path) -> str:
    """PaddleOCR-VL 의 Chart Recognition 으로 차트 데이터를 추출한다."""
    print(f"[INFO] [1단계] 차트 인식 중... ({OCR_SERVICE_SLUG})")
    start = time.time()
    try:
        response = ocr_client.chat_with_image_path(
            image_path=image_path,
            system_message="",
            user_text="Chart Recognition:",
            temperature=0.0,
        )
    except Exception as exc:
        print(f"[WARNING] Chart OCR 호출 실패 (무시): {exc}")
        return ""

    elapsed = time.time() - start
    chart_text = response.text.strip()
    print(f"[INFO] Chart OCR 완료: {elapsed:.1f}초, {len(chart_text)}자 추출")
    return chart_text


def analyze_slide(
    vlm_client: Work2VLMClient,
    ocr_client: Work2VLMClient,
    image_path: Path,
) -> dict | None:
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
    ocr_text = run_ocr(ocr_client, image_path)
    chart_text = run_chart_ocr(ocr_client, image_path)

    # OCR 결과 결합
    combined_ocr = ocr_text
    if chart_text:
        combined_ocr += f"\n\n[Chart Data]\n{chart_text}"
    if not combined_ocr.strip():
        combined_ocr = "(OCR 결과 없음 — 이미지만으로 분석)"

    # ── 2단계: VLM 구조 분석 (OCR 텍스트 참조) ──
    print(f"[INFO] [2단계] VLM 구조 분석 중... ({VLM_SERVICE_SLUG})")
    user_prompt = USER_PROMPT_TEMPLATE.format(ocr_text=combined_ocr)

    start = time.time()
    try:
        response = vlm_client.chat_with_image_path(
            image_path=image_path,
            system_message=SYSTEM_MESSAGE,
            user_text=user_prompt,
            temperature=0.1,
        )
    except Exception as exc:
        print(f"[ERROR] VLM 호출 실패: {exc}")
        return None

    elapsed = time.time() - start
    print(f"[INFO] VLM 응답 시간: {elapsed:.1f}초")

    # 토큰 사용량
    usage = response.token_usage
    if usage:
        print(
            f"[INFO] 토큰: prompt={usage.get('prompt_tokens', '?')}, "
            f"completion={usage.get('completion_tokens', '?')}, "
            f"total={usage.get('total_tokens', '?')}"
        )

    # JSON 파싱 시도
    raw_text = response.text.strip()
    # markdown 코드 펜스 제거
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    try:
        result = json.loads(raw_text)
        print(f"[INFO] JSON 파싱 성공")
        result["_ocr_text"] = combined_ocr
        return result
    except json.JSONDecodeError as exc:
        print(f"[WARNING] JSON 파싱 실패: {exc}")
        print(f"[INFO] Raw 응답:\n{raw_text}")
        return {"raw_response": raw_text, "_ocr_text": combined_ocr}


def print_result(image_name: str, result: dict) -> None:
    """분석 결과를 보기 좋게 출력한다."""
    print(f"\n{'─'*60}")
    print(f"  결과: {image_name}")
    print(f"{'─'*60}")

    if "raw_response" in result:
        print(result["raw_response"])
        return

    # 제목
    title = result.get("slide_title")
    if title:
        print(f"\n■ 제목: {title}")
    subtitle = result.get("subtitle")
    if subtitle:
        print(f"  부제: {subtitle}")

    # 전체 주제
    topic = result.get("overall_topic")
    if topic:
        print(f"\n■ 주제: {topic}")

    # 본문
    body = result.get("body_text") or []
    if body:
        print(f"\n■ 본문 ({len(body)}항목):")
        for i, text in enumerate(body, 1):
            print(f"  {i}. {text}")

    # 핵심 포인트
    key_points = result.get("key_points") or []
    if key_points:
        print(f"\n■ 핵심 포인트:")
        for point in key_points:
            print(f"  • {point}")

    # 차트
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

    # 표
    tables = result.get("tables") or []
    if tables:
        print(f"\n■ 표 ({len(tables)}개):")
        for i, table in enumerate(tables, 1):
            print(f"  [{i}] {table.get('table_title', '제목 없음')}")
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            if headers:
                print(f"      헤더: {' | '.join(headers)}")
            for row in rows[:5]:  # 최대 5행만 출력
                print(f"      → {' | '.join(str(v) for v in row)}")
            if len(rows) > 5:
                print(f"      ... 외 {len(rows) - 5}행")

    # 이미지/다이어그램
    images = result.get("images_and_diagrams") or []
    if images:
        print(f"\n■ 이미지/다이어그램 ({len(images)}개):")
        for img in images:
            print(f"  • [{img.get('type', '?')}] {img.get('description', '')}")

    # 푸터
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

    key_points = result.get("key_points") or []
    if key_points:
        lines.append("\n핵심 포인트:")
        for point in key_points:
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


def save_result(image_path: Path, result: dict) -> Path:
    """분석 결과를 읽기 좋은 텍스트 파일로 저장한다."""
    txt_path = image_path.with_suffix(".result.txt")
    text = _format_result_text(result)

    # OCR 원문도 함께 기록
    ocr_text = result.get("_ocr_text", "")
    if ocr_text:
        text += f"\n\n{'─'*40}\nOCR 원문 (참고용):\n{'─'*40}\n{ocr_text}\n"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[INFO] 결과 저장: {txt_path.name}")
    return txt_path


def main():
    """메인 실행."""
    print("[INFO] PPT 슬라이드 분석기 시작")
    print(f"[INFO] OCR 모델: {OCR_SERVICE_SLUG}")
    print(f"[INFO] VLM 모델: {VLM_SERVICE_SLUG}")
    print(f"[INFO] 대상 이미지: {len(IMAGE_PATHS)}개")

    # 존재하는 이미지만 필터
    valid_images = [p for p in IMAGE_PATHS if p.exists()]
    if not valid_images:
        print(f"[ERROR] 분석할 이미지가 없습니다.")
        print(f"[INFO] sample_code/ 폴더에 .jpg 또는 .png 이미지를 넣어주세요.")
        print(f"[INFO] 경로: {SAMPLE_DIR}")
        sys.exit(1)

    print(f"[INFO] 존재하는 이미지: {len(valid_images)}개")

    # 클라이언트 생성
    try:
        ocr_client = Work2VLMClient(service_slug=OCR_SERVICE_SLUG)
        vlm_client = Work2VLMClient(service_slug=VLM_SERVICE_SLUG)
    except Exception as exc:
        print(f"[ERROR] 클라이언트 생성 실패: {exc}")
        sys.exit(1)

    print(f"[INFO] OCR endpoint: {ocr_client.endpoint}")
    print(f"[INFO] VLM endpoint: {vlm_client.endpoint}")

    # 각 이미지 분석
    results = {}
    for image_path in valid_images:
        result = analyze_slide(vlm_client, ocr_client, image_path)
        if result is not None:
            results[image_path.name] = result
            print_result(image_path.name, result)
            save_result(image_path, result)

    # 요약
    print(f"\n{'='*60}")
    print(f"[INFO] 분석 완료: {len(results)}/{len(valid_images)}개 성공")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
