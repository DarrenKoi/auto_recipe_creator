# VLM align-point fallback 설계 + Kimi-K2.6 마이그레이션 + 빈 응답 디버깅

- 일시: 2026-06-18 10:24
- 브랜치: main
- 주제: align fail 시 VLM 을 fallback 으로 쓰는 전략 논의 → 모델을 Kimi-K2.6 으로 교체 →
  workflow_2 프로브들의 verbose-model 버그 연쇄 수정 → 현재 VLM 이 region 을 못 내놓는 상태(빈 content) 진단 중

---

## 1. 진행 사항

### (A) 설계 논의 — align point 탐색에 ML/VLM 을 쓸 수 있는가
- **YOLO 적용 검토 → 기각.** align point 탐색은 closed-set object detection 이 아니라
  one-shot localization(등록된 key 를 라이브 FOV 에서 찾기)이라 YOLO 와 task shape 불일치.
  학습데이터 부재(298 recipe, S 희박, 오피스 데이터 반출 불가), 픽셀 동일성 가정 불가,
  "CV=좌표 권위 / VLM=영역만" 규칙(2026-05-25) 위배가 근거. UI 같은 안정 요소의 region
  proposer 로는 가능하나 핵심 task 엔 부적합.
- **SiamFC 류(learned correlation) 소개.** 한다면 YOLO 가 아니라 one-shot 인 Siamese/correlation
  tracker 또는 SuperPoint+LightGlue 류가 적합 — proposer/reranker 자리에 들어가는 방향.
- **VLM fallback 채택 방향 확정.** CV 점수면이 평평할 때만(=fallback) VLM 이 *영역/판단*만 하고
  좌표는 CV 가 확정하는 3-stage 프로토콜:
  `CV 가 코너 후보에 번호 부여 → VLM 이 같은 지점의 index 선택(좌표 아님) → CV 가 정밀화`.
  엔지니어 도메인 단서 반영: align point = 두 edge 가 교차하는 box 코너, 보통 Q3/Q4(아래쪽).

### (B) 프롬프트 작성
- `poc/workflow_3/vlm/prompts/prompt_align_compare.py` **신규 작성**(commit 5fca221).
  `build_align_compare_prompt(n_candidates, box_width, box_height) -> (system, user)`.
  좌표가 아닌 **match_index** 반환(= 좌표 환각 구조적 차단), `-1` 거부 경로, Q3/Q4 soft tiebreaker,
  junction-shape 서술(reasoning scaffold). 아직 caller 없음(= prompts/__init__ 미등록).

### (C) 모델 교체 — Qwen3-VL → Kimi-K2.6
- Gemma-4 는 2-image 시 400(image=1 cap)이라 후보에서 제외, **Kimi 단독** 결정.
- 사내에서 Qwen3-VL, Kimi-K2.5 모두 deprecated → 레지스트리/프로브 전부 Kimi-K2.6 으로 통일.

### (D) verbose-model(Kimi) 버그 연쇄 진단 — "결과가 전부 fail/empty"
- 근본 원인 패턴: **프로브들이 Qwen(간결 JSON)에 맞춰 튜닝**되어, verbose reasoning 모델(Kimi)을
  넣자 (1) 생성 토큰 truncate, (2) 파싱 전 텍스트 슬라이스, (3) JSON 전용 모드 빈 응답 으로 깨짐.
- 동일 패턴이 **세 군데**에서 발견됨(아래 수정 내용 참조).

---

## 2. 수정 내용

### `poc/workflow_3/vlm/flask_vlm.py` (commit 0abaf7a)
- `KIMI_K2_5_* → KIMI_K2_6_*`, slug `kimi-k2.5 → kimi-k2.6`, 표시명 `Kimi-K2.6`.
- deprecated `qwen3-vl-30b-instruct` 엔트리 + 상수 **제거**.
- 검증: `get_service_by_slug('kimi-k2.6')` 해석 OK, 옛 slug 들은 None.

### `poc/workflow_2/vlm_align_key_region.py` (commits 8fb9132, 0abaf7a, 75daaa8, +미커밋)
- `MAX_TOKENS 512 → 4096` (Kimi prose preamble 뒤 JSON truncate 방지).
- **숨은 truncation 제거**: `raw_text[:1000]` 으로 자른 뒤 파싱하던 것을 → 전체 텍스트 파싱,
  슬라이스는 로깅용으로만. (이게 max_tokens 와 별개인 2차 절단점이었음.)
- `response_format=json_object` 를 `PROBE_JSON_MODE`(기본 on, =0 으로 끔) 토글 뒤에 추가.
- `PROBE_PERCEPTION_MODEL` env override 추가, 기본 모델 `Kimi-K2.6`.
- **빈 응답 크래시 방지**(75daaa8): `extract_json` 이 `requests.RequestException` 만 잡는 try 안에
  있어 빈 content → `JSONDecodeError(char 0)` 가 미포착 크래시였음. `json.JSONDecodeError` 포착 →
  기록 후 계속. `finish_reason` 캡처, content null 시 `reasoning_content` 폴백.
  `empty_content` vs `json_parse_failed` 구분.
- **(미커밋) 진단 강화**: 빈 content 일 때 게이트웨이 **원본 body**(`raw_body`) + `usage` 캡처,
  not-ok 시 콘솔에 `finish`/`usage`/`raw_body`/`raw_text` 출력. (원인 못 보던 사각지대 해소용.)

### `poc/workflow_2/probe_multi_image_vlm.py` (commit 6bbd388)
- `MODELS → ["Kimi-K2.6"]` (Qwen 계열 deprecated).
- `MAX_TOKENS 64 → 1024` (verbose 답 truncate 로 operand 만 파싱돼 PARTIAL 오판하던 것 수정).
- 파싱을 `response_text[:300]` 슬라이스 *전*의 전체 텍스트에서 수행.
- `LARGE_VLM_API_BASE → http://common.llm.skhynix.com/v1` (flask_vlm Kimi 게이트웨이와 일치).
- **결과: 이 프로브는 정상 동작 확인**(Kimi-K2.6 가 2-image 합산 PASS = 두 이미지를 실제로 봄).

### `poc/workflow_3/vlm/prompts/prompt_align_compare.py` (commit 0abaf7a)
- docstring 타깃 모델 `Qwen3-VL → Kimi-K2.6`.

---

## 3. 현재 상태 (중요)

- **capability gate 통과**: `probe_multi_image_vlm.py` 에서 Kimi-K2.6 가 두 이미지를 합산 → 둘 다 인지 확인.
- **그러나 실제 task 실패**: `vlm_align_key_region.py` 에서 Kimi 가 **빈 content(200 OK)** 반환 →
  `ok=False, found=None, region_px=None`, ROI 매칭 미실행. = **현재 VLM 이 쓸만한 결과를 못 줌.**
- 빈 응답 원인 미확정. 유력 가설: `response_format=json_object` 가 이 게이트웨이/Kimi 에서 빈 content
  유발(정상 프로브는 response_format 미사용). 그 외 reasoning 토큰 소진(finish=length) /
  reasoning_content 분리 / webp 거부 가능성.
- 사각지대 해소를 위해 **raw_body 덤프 계측을 추가했고 아직 미커밋**.

---

## 4. 다음 단계

1. **(미커밋 계측 커밋 후)** 오피스에서 진단 재실행:
   `PROBE_JSON_MODE=0 uv run python poc/workflow_2/vlm_align_key_region.py`
   → 콘솔의 `[DEBUG] raw_body=...` 한 줄을 회수. 이게 원인을 확정함:
   - `finish_reason=length` + `usage.completion_tokens≈4096` → 토큰 소진 → MAX_TOKENS 상향(8192+).
   - `content:""` + `reasoning_content` 채워짐 → 답이 reasoning 필드 → 폴백 동작 확인.
   - body 에 error/message → 게이트웨이가 거부한 항목(webp/response_format) 확인.
   - `PROBE_JSON_MODE=0` 으로도 빈 응답이면 → json-mode 아님, body 가 가리키는 원인 추적.
2. 원인 해소 후 **region-bbox A/B 본 평가**: low-confidence subset 에서 `roi_hint=VLM-region` 이
   full-frame 대비 CV 점수/판정을 끌어올리는지(delta) 확인 = VLM escalation 채택 게이트.
3. delta 양수면 → **corner-index 드라이버 신규 작성**(prompt_align_compare 소비:
   CV 코너 → Kimi index 선택 → 정밀화 → `golden_localization_eval_cond` 대비 rank-1).
   delta 평탄/음수면 → VLM escalation 가설 기각, template bank/ROI 축으로 전환.

### 사용자 확인 필요 (추측 금지)
- probe digest 에서 **WebP 가 PASS 였는지** JPEG 만이었는지? region 프로브는 `ENCODE_FMT="webp"` 라
  WebP 거부면 jpeg 로 바꿔야 함 — 이번 빈 응답의 또 다른 후보.

---

## 5. 메모리 업데이트

`MEMORY.md` 에 다음 1줄 추가 권장(아직 미반영 — 다음 세션/사용자 확인 후):
- **사내 VLM 게이트웨이: Qwen3-VL · Kimi-K2.5 deprecated, 현재 Kimi-K2.6.** 게이트웨이 base =
  `http://common.llm.skhynix.com/v1`. Kimi 는 verbose/reasoning 모델이라 프로브 튜닝(max_tokens 소형,
  파싱 전 텍스트 슬라이스, response_format=json_object)이 빈/잘린 응답을 유발 → verbose-model 대비
  필요. (관련: [[project_vlm_multi_image_capability]], [[project_vlm_service_slug_not_model_name]])

> 위 메모리는 빈-응답 근본원인 확정(raw_body 확인) 후 반영하는 것이 안전. 현재는 저널에만 기록.
