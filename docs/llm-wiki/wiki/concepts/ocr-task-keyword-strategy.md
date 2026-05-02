---
tags: [concept, ocr, paddleocr-vl, prompt]
level: intermediate
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: [
  raw/journals/260312/20260312-paddleocr-prompt-debug-dir.md,
  raw/journals/260323/260323_154335_simplify-select-tool-dedup.md,
  raw/journals/260323/260323_160339_simplify-select-tool-full-session.md,
  raw/journals/260415/260415_081739-ocr-tool-list-research.md
]
---

# OCR Task Keyword Strategy

> PaddleOCR-VL-1.5에는 긴 JSON prompt보다 학습된 task keyword와 crop 기반 재판독을 먼저 적용하는 OCR 전략이다. (source: raw/journals/260312/20260312-paddleocr-prompt-debug-dir.md)

## 왜 필요한가? (Why)

- PaddleOCR-VL-1.5 prompt는 복잡한 system message와 JSON 지시를 제거하고 `"OCR:"` task keyword 중심으로 단순화되었다. (source: raw/journals/260312/20260312-paddleocr-prompt-debug-dir.md)
- Tool List처럼 텍스트가 많은 화면에서는 출력 길이와 context 한도가 OCR 누락의 원인이 될 수 있다. (source: raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)
- 저널 기준 최신 OCR 결론은 prompt를 길게 늘리기보다 left-side focused crop, upscaling, `OCR:`, 필요 시 `Spotting:` 또는 `got-ocr` fallback을 검토하는 것이다. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)

## 핵심 개념 (What)

### 정의

- OCR task keyword strategy는 OCR 모델의 학습된 task prefix를 그대로 사용하고, 입력 이미지 영역과 출력 budget을 조정해 인식 품질을 확보하는 접근이다. (source: raw/journals/260312/20260312-paddleocr-prompt-debug-dir.md)

### 관련 용어

- `OCR:`: broad reading에 쓰는 기본 PaddleOCR-VL task keyword. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)
- `Spotting:`: tool list 좌표 힌트 같은 별도 실험 pass 후보. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)
- `MAX_MODEL_LEN`: 입력과 출력 합산 context 한도이며, Tool List OCR에서 4096 한도가 부족해 8192로 늘린 기록이 있다. (source: raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)
- `normalize_lines`: OCR 원문을 line 단위 후보로 정규화하는 공유 helper. (source: raw/journals/260323/260323_154335_simplify-select-tool-dedup.md)

### 시각화 / 모델

```text
full screenshot
  -> focused crop
  -> optional upscale
  -> "OCR:" request
  -> normalize_lines
  -> target visibility or matched lines
  -> fallback: Spotting or GOT-OCR
```

## 어떻게 사용하는가? (How)

### 최소 예제

```text
system_message = ""
user_text = "OCR:"
```

PaddleOCR-VL prompt builder는 이처럼 빈 system message와 `OCR:` user text를 반환하도록 단순화되었다. (source: raw/journals/260312/20260312-paddleocr-prompt-debug-dir.md)

### 실무 패턴

- 먼저 crop 위치를 줄이고, 필요하면 crop image를 upscaling한 뒤 `OCR:`로 재시도한다. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)
- 긴 Tool List는 `OCR_MAX_TOKENS`, line limit, `MAX_MODEL_LEN`을 함께 조정한다. (source: raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)
- OCR raw response, summary JSON, source JPEG, VLM 전송 WebP, overlay를 함께 저장해 실패 분석이 가능하게 한다. (source: raw/journals/260312/20260312-paddleocr-prompt-debug-dir.md)

### 주의사항 / 함정

- JSON 응답을 강제하면 실제 PaddleOCR-VL plain text 응답과 맞지 않아 parser가 불필요하게 복잡해질 수 있다. (source: raw/journals/260312/20260312-paddleocr-prompt-debug-dir.md)
- max tokens만 늘려도 model context가 그대로면 upstream이 context 초과 에러를 낼 수 있다. (source: raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)

## 참고 자료 (References)

- 원본 메모: [20260312-paddleocr-prompt-debug-dir.md](../../raw/journals/260312/20260312-paddleocr-prompt-debug-dir.md)
- 원본 메모: [260323_160339_simplify-select-tool-full-session.md](../../raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)
- 원본 메모: [260415_081739-ocr-tool-list-research.md](../../raw/journals/260415/260415_081739-ocr-tool-list-research.md)
- 관련 컴포넌트: [rcs-tool-selection.md](../components/rcs-tool-selection.md)
