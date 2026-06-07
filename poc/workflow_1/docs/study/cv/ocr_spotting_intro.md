# OCR Spotting 파싱과 "확인 전용" 규칙 (intro)

> 대상: `ocr_spotting.py`, `prompts/prompt_ocr_assist.py`, `util/json_utils.py`
> 상위 개요: `../algorithms/automation_methods_intro.md` §3 · 함께 읽기: `../paddleOCR/README.md`

---

## 1. `OCR:` vs `Spotting:` — 두 가지 OCR 태스크

PaddleOCR-VL 라우트는 OpenAI 호환 chat 에 **태스크 라벨** 을 프롬프트로 전달합니다
(`prompts/prompt_ocr_assist.py`):

| 태스크 | 프롬프트 | 결과 |
|---|---|---|
| 평문 OCR | `OCR:` | 텍스트만 |
| Spotting | `Spotting:` | 텍스트 + **bbox 좌표** |

- **`OCR:`** 는 "이 crop 에 'alignment mark' 라고 쓰여 있나?" 같은 **존재 확인** 에 씁니다.
- **`Spotting:`** 는 "이 행 strip 에서 Tool ID 가 어디 있나?" 처럼 **위치까지** 필요할 때 씁니다.

---

## 2. Spotting 응답은 형식이 제각각 → 견고한 파서

`Spotting:` 응답은 모델과 버전마다 bbox 형식이 제각각입니다. `parse_spotting_items(raw_text)` 가 이를
`[{"text": str, "bbox": {left,top,right,bottom}}, ...]` 로 정규화합니다.

견디는 형식들(`_coerce_bbox`):
- 4원소 배열 `[x1,y1,x2,y2]`
- dict `{left,top,right,bottom}` / `{x1,y1,x2,y2}`
- 폴리곤(점들의 리스트) → min/max 로 bbox 환산
- wrapper 로 감싼 중첩 구조 (`layoutParsingResults`, `prunedResult`, `result`, `data`, `items`,
  `blocks`, `detections`, `texts` …) → 재귀 탐색

텍스트 라벨도 여러 키를 순서대로 탐색합니다(`text`, `content`, `block_content`, `transcription`, `label`,
`word`, `rec_text`, `caption`, `value`). 마지막에는 `_dedupe_spotting_items()` 로 (text,bbox) 중복을 제거합니다.

파싱 자체도 `_parse_json_like()` 가 JSON → `ast.literal_eval` 순으로 시도하여 깨진 응답까지 견딥니다.

---

## 3. 핵심 규칙 — OCR 은 "확인"만, "결정"은 안 한다

프로젝트 전역 안전 규칙(메모리: feedback_click_pipeline_coarse_fine_confirm):

```
VLM(ui-venus→mai-ui) 이 좌표를 결정한다.
PaddleOCR 는 그 자리를 "확인"만 한다.
확인되지 않으면 클릭하지 않는다.
```

왜?
- **전체 RCS 화면에 OCR 을 직접 돌리면 환각** 이 발생합니다 (메모리:
  project_paddleocr_vl_screenshot_hallucination). PaddleOCR-VL 은 문서 파서이므로 layout→crop→recognize
  가 정상 경로이고, all-in-one GUI OCR 은 금지합니다.
- 짧은 라벨(`OK`)은 화면 여러 곳에 있어 증거가 약합니다. **긴 고유 문구**(`alignment mark`, `Wait Input`)로
  확인해야 신뢰할 수 있습니다.

권장 패턴:
```
1) 화면 관찰 → 2) VLM/CV 로 영역 제안 → 3) 그 영역만 crop
→ 4) crop 에만 OCR → 5) 고유 문구/박스 증거 확인 → 6) 확인되면 가드된 클릭 / 아니면 클릭 안 함
```

---

## 4. workflow_1 에서의 쓰임

- **로그인 confirm**: 클릭점 주변 텍스트를 읽어 기대 라벨과 근접 확인.
- **Tool 행 verify(fallback)**: VLM 이 제안한 행 strip 에 `Spotting:` → `tool_name_match` 정규화 매칭.
- **debug 증거**: 실패한 run 에 사람이 읽을 OCR 텍스트를 아티팩트로 남깁니다(자동화 명령이 아니라 증거용).

상세 운영 가이드(crop-first, 토큰 캡, 검증 체크리스트)는 `../paddleOCR/README.md` 를 참조하세요.

---

## 5. 핵심 함수 한눈에

| 함수 | 역할 |
|---|---|
| `build_ocr_assist_prompt()` | `OCR:` 프롬프트 |
| `build_spotting_prompt()` | `Spotting:` 프롬프트 |
| `parse_spotting_items()` | 다양한 형식 → `{text,bbox}` 리스트 |
| `_coerce_bbox()` | 배열/dict/폴리곤/중첩 → bbox |
| `_dedupe_spotting_items()` | (text,bbox) 중복 제거 |
