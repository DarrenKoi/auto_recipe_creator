---
tags: [component, rcs, tool-selection, ocr]
level: intermediate
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: [
  raw/journals/260323/260323_154335_simplify-select-tool-dedup.md,
  raw/journals/260323/260323_154626-add-view-tool-id-scanner.md,
  raw/journals/260323/260323_160339_simplify-select-tool-full-session.md,
  raw/journals/260415/260415_081739-ocr-tool-list-research.md
]
---

# RCS Tool Selection

> RCS main window에서 Tool List를 열고 OCR/grounding 조합으로 target Tool ID를 찾은 뒤 선택하는 컴포넌트다. (source: raw/journals/260323/260323_154626-add-view-tool-id-scanner.md)

## 왜 존재하는가? (Why)

- Tool List 화면은 tool ID 외에도 IP, location, model name, status 같은 텍스트가 많아 OCR 출력이 길고 누락이 생길 수 있다. (source: raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)
- full-window OCR만으로 하단 tool ID가 누락되는 문제가 있어 focused crop, context/token 한도, fallback OCR 경로가 필요했다. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)
- selection flow는 OCR visibility check와 `UI-Venus + MAI-UI` grounding을 조합하는 방향으로 분석되었다. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)

## 무엇인가? (What)

### 책임 범위

- `select_tool.py`는 대상 Tool ID가 list view 안에 보이는지 OCR로 확인하고, grounding 결과로 target row를 선택하는 흐름을 담당한다. (source: raw/journals/260323/260323_154335_simplify-select-tool-dedup.md)
- `scan_tools_from_view.py`는 View 탭을 연 뒤 green box 후보를 색상 기반으로 탐지하고 각 box를 OCR해 visible Tool ID를 수집한다. (source: raw/journals/260323/260323_154626-add-view-tool-id-scanner.md)
- `normalize_lines()`, `image_point_to_screen()`, `crop_image()`는 여러 script에 중복되던 helper에서 `util/` 공용 함수로 추출되었다. (source: raw/journals/260323/260323_154335_simplify-select-tool-dedup.md)

### 핵심 진입점

- `poc/work2/select_tool.py` — target visible 확인과 target row 선택. (source: raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)
- `poc/work2/scan_tools_from_view.py` — View tab 기반 visible Tool ID scanning. (source: raw/journals/260323/260323_154626-add-view-tool-id-scanner.md)
- `poc/work2/util/json_utils.py`, `poc/work2/util/window_utils.py`, `poc/work2/util/image_utils.py` — OCR line normalize, image-to-screen coordinate, crop helper. (source: raw/journals/260323/260323_154335_simplify-select-tool-dedup.md)

### 의존성

- 내부: [ocr-task-keyword-strategy.md](../concepts/ocr-task-keyword-strategy.md), [gui-coordinate-and-window-focus.md](../concepts/gui-coordinate-and-window-focus.md)
- 외부: `PaddleOCR-VL-1.5`, `UI-Venus`, `MAI-UI`, `cv2`, `numpy`, Windows mouse control. (source: raw/journals/260323/260323_154626-add-view-tool-id-scanner.md)

### 데이터 모델 / 인터페이스

```text
main window screenshot -> list/content crop -> OCR lines
OCR lines -> target_visible + matched_lines
grounded image point -> screen point -> double-click target row
```

## 어떻게 쓰는가? (How)

### 호출 예시

```powershell
uv run python poc/work2/scan_tools_from_view.py
uv run python poc/work2/select_tool.py
```

실제 mouse scroll/click을 쓰는 경우에는 safe/action env 값을 명시적으로 확인해야 한다. (source: raw/journals/260323/260323_154626-add-view-tool-id-scanner.md)

### 자주 쓰는 패턴

- full-window OCR이 실패하면 left-side focused crop을 먼저 적용하고 crop image를 upscaling한 뒤 `OCR:`로 재시도한다. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)
- OCR output이 너무 긴 경우 `OCR_MAX_TOKENS`, `normalize_lines max_items`, model `MAX_MODEL_LEN`을 함께 본다. (source: raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)
- green box detection 결과는 overlay, raw OCR response, summary JSON으로 남겨 threshold 조정을 쉽게 한다. (source: raw/journals/260323/260323_154626-add-view-tool-id-scanner.md)

### 안티패턴

- target visibility를 전체 normalized text 문자열 포함 여부만으로 판단하면 cross-line false positive가 생길 수 있다. (source: raw/journals/260323/260323_154335_simplify-select-tool-dedup.md)
- PaddleOCR prompt를 길게 늘리는 것으로 Tool List 누락 문제를 먼저 해결하려고 하지 않는다. (source: raw/journals/260415/260415_081739-ocr-tool-list-research.md)

## 참고 자료 (References)

- 원본 메모: [260323_154335_simplify-select-tool-dedup.md](../../raw/journals/260323/260323_154335_simplify-select-tool-dedup.md)
- 원본 메모: [260323_154626-add-view-tool-id-scanner.md](../../raw/journals/260323/260323_154626-add-view-tool-id-scanner.md)
- 원본 메모: [260323_160339_simplify-select-tool-full-session.md](../../raw/journals/260323/260323_160339_simplify-select-tool-full-session.md)
- 원본 메모: [260415_081739-ocr-tool-list-research.md](../../raw/journals/260415/260415_081739-ocr-tool-list-research.md)
