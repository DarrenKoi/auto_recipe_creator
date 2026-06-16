# VLM ROI Prior — 설계 spec

작성일: 2026-06-16 · 상태: 설계(design), 미구현 · 상위 연구: `../study/align_point_accuracy_ml_vlm_research_260616.md`

> **한 줄 목표**: paused 프레임에서 VLM 이 "align key 가 있을 법한 영역(박스)" 를 grounding 하면,
> CV proposer 가 그 영역 안에서만 peak 를 찾는다. 탐색 공간 축소 → far/veryfar 오탐 peak 제거
> (= 벽 A 일부 + 평평한 점수면 완화). **VLM 은 영역만, 좌표는 CV** (doc §8).

---

## 0. 핵심 발견 — 엔진은 이미 준비됨

검증 결과(2026-06-16, `engine.py` 직접 확인):

- `compute_align_key_score_ensemble(template, frame, *, frame_nm_per_pixel=None, roi_hint=None, scales, policy)`
  — **`roi_hint: tuple[int,int,int,int]` 이미 존재** (`engine.py:854`). `compute_align_key_score` 도 동일
  (`engine.py:945`).
- 내부 `_prepare_match_inputs`(`engine.py:666`)가 `roi_hint=(x,y,w,h)` 를 검증/적용:
  - 4-tuple 아니면 ValueError(`:706-710`)
  - w/h ≤ 0 이면 ValueError(`:714`)
  - frame 과 교집합 없으면 ValueError(`:723`)
  - crop 이 template 보다 작으면 ValueError(`:730`)
  - 통과 시 `gray_frame = gray_frame[y0:y1, x0:x1]`, `roi_origin=(x0,y0)` 추적 → best_xy 를 다시
    **frame 절대 좌표**로 환산.

**결론: 엔진 변경 0.** 필요한 건 (a) VLM align-key-box locator 모듈, (b) `correction.py` 한 줄 배선,
(c) 플래그 + 폴백. 끝.

---

## 1. 어느 벽을 치나

- 벽 A(proposer recall): VLM 이 올바른 영역을 주면, proposer 가 엉뚱한 곳(far/veryfar)에서 잡던
  false peak 가 애초에 후보에서 빠진다. miss 의 구조적 절반 일부를 직접 공략
  ([[project_ensemble_on_consensus_rejected]] 가 "어려운 절반은 template bank/ROI/VLM 축" 이라 명시).
- 평평한 점수면(벽 B) 완화: 후보 풀이 좁아지면 decoy 수가 줄어 1등 오선택 확률↓.

단, **VLM 박스가 틀리면 정답을 영역에서 배제해 recall 을 해칠 수 있다.** → §5 의 2-tier 폴백이 필수.

---

## 2. 설계 원칙 (불변)

1. **VLM = 영역만**: 박스(ROI)만 반환, 좌표·결정 금지(doc §8, CLAUDE.md:194). `ok_button.py` 와 같은
   "VLM 은 region 식별" 패턴.
2. **CV 거부권 / 폴백 우선**: ROI 가 비거나(거부), 신뢰도 낮거나, ROI-제한 매칭이 low 면 **full-FOV
   재시도**. VLM 은 recall 을 *깎지 않는* 방향으로만 작동.
3. **좌표공간 구분(중요)**: 이 ROI 는 **frame(SEM ROI) 좌표** — `controller.capture()` 가 주는 바로
   그 프레임 공간. `ok_button` 의 *screen 절대 좌표* 와 다르다([[project_align_cond_files_and_coords]]
   계열 좌표 혼동 주의). VLM 도 같은 frame 을 본다.
4. **office-only VLM**: Flask proxy 경유. Mac 은 호출 불가(로직만 dry-run). 미주입/실패 시 자동 off.

---

## 3. 새 모듈 — `align/vlm_align_key_box.py`

`ok_button.py` 패턴을 그대로 미러. workflow_2 의 `vlm_align_key_box.py`(throwaway probe)의 프롬프트
내용을 **포팅**(import 아님 — workflow_3 은 workflow_2 를 import 하지 않음).

### 3.1 시그니처

```python
def locate_align_key_roi(
    *,
    frame_bgr: np.ndarray,
    client: Workflow1VLMClient,
    source_hint: str = "sem",          # "om" | "sem" | "live_sem"
    min_confidence: float = 0.5,
    pad_frac: float = 0.15,            # 박스 패딩(아래 §4.2)
) -> tuple[int, int, int, int] | None:
    """paused 프레임에서 align key 영역을 VLM 으로 grounding.
    반환: (x, y, w, h) frame 좌표 roi_hint, 또는 None(미검출/거부/저신뢰).
    """
```

### 3.2 내부 흐름 (ok_button.py 미러)

```python
image_b64, w, h = _frame_to_webp_b64(frame_bgr)        # ok_button._frame_to_webp_b64 재사용/복제
sys_msg, user_text = build_align_key_roi_prompt(source_hint)   # 새 prompt builder
resp = client.chat_with_image_b64(
    image_b64=image_b64, system_message=sys_msg, user_text=user_text,
    image_mime="image/webp", temperature=0.0,
)
parsed = extract_json(resp.text)                       # util.json_utils
if parsed.get("key_visible") is not True: return None
if float(parsed.get("confidence", 0)) < min_confidence: return None
bbox_px = bbox_to_pixels(parsed.get("key_bbox"), w, h, parsed.get("coord_system"))
if bbox_px is None: return None
return _pad_to_roi(bbox_px, w, h, pad_frac)            # (x,y,w,h), 패딩+클램프
```

### 3.3 Prompt builder — `vlm/prompts/prompt_align_key_roi.py`

기존 패턴: `(system_message, user_text)` 튜플 반환. UI-Venus 공식 bbox grounding 형식(relative_1000,
단일 요소, [-1,-1]/false refusal — [[project_ui_venus_official_grounding]]).

```python
def build_align_key_roi_prompt(source_hint: str) -> tuple[str, str]:
    system_message = (
        "You analyse a grayscale CD-SEM / optical metrology FOV. Locate the ALIGNMENT KEY "
        "(fabricated fiducial: nested boxes / cross / L-corner / dot cluster). "
        "Identify the REGION only, never a single coordinate. "
        "If not clearly visible, refuse rather than guess."
    )
    user_text = (
        f"Source: {source_hint}.\n"
        "Return ONLY JSON:\n"
        '{ "key_visible": true, "coord_system": "relative_1000", '
        '"key_bbox": {"left":0,"top":0,"right":0,"bottom":0}, "confidence": 0.0 }\n'
        "key_bbox must enclose the whole alignment-key pattern. "
        "If no key is clearly visible, set key_visible=false and key_bbox=null."
    )
    return system_message, user_text
```
- **first-letter/region anchoring** 원칙 적용([[project recall: VLM prompt 원칙]]).
- service slug 은 `"ui-venus"`(route_slug, 모델명 아님 — [[project_vlm_service_slug_not_model_name]]).

---

## 4. 통합 — `correction.py`

### 4.1 위치

`correct_align_fail` 의 paused 매칭 직전(`correction.py:233-240`). 현재:
```python
frame = controller.capture()                 # :233
...
result = compute_align_key_score_ensemble(   # :240
    template, frame, scales=PAUSED_SCALES, policy=STRUCTURE_POLICY)
```
변경(플래그 on 시):
```python
roi_hint = None
if settings.vlm_roi_enabled and vlm_client is not None:
    try:
        from poc.workflow_3.align.vlm_align_key_box import locate_align_key_roi
        roi_hint = locate_align_key_roi(
            frame_bgr=frame, client=vlm_client, source_hint=mode.lower())
    except Exception as exc:
        print(f"[WARNING] VLM ROI locator 실패, full-FOV 폴백: {exc}")
        roi_hint = None

result = compute_align_key_score_ensemble(
    template, frame, roi_hint=roi_hint,
    scales=PAUSED_SCALES, policy=STRUCTURE_POLICY)
```

### 4.2 박스 패딩 (필수)

VLM 박스를 그대로 쓰면 정답 peak 가 경계에서 잘릴 수 있다. **pad_frac(예 0.15) 만큼 확장 후
frame 으로 클램프**. 너무 빡빡한 ROI 는 recall 을 해친다 — "약간 넉넉하게, 그러나 full-FOV 보다는
좁게" 가 목표. roi_hint crop 이 template 보다 작으면 엔진이 ValueError → §5 폴백이 받아낸다.

---

## 5. 2-tier 폴백 (안전장치, 가장 중요)

bench 에서 **proposer recall 이 천장**이므로 VLM 이 그걸 더 깎으면 손해다. 따라서:

```
tier 1: roi_hint 로 매칭
  └ decision == "match" (또는 adjust+distinctive) → 채택. 끝.
tier 2: tier1 이 "low" 거나 roi_hint 가 None/ValueError → roi_hint=None 으로 full-FOV 재매칭
  └ 기존 동작과 동일. VLM 은 recall 을 절대 깎지 않음(상한만 좁혔다가 실패 시 원복).
```

즉 VLM-ROI 는 **best-effort 가속/정제**이지 gate 가 아니다. 최악의 경우 full-FOV 와 동일.

---

## 6. 설정 / 플래그

- `ALIGN_FAIL_VLM_ROI`(기본 0) → `Workflow3Settings.vlm_roi_enabled`. config 는 env/`Workflow3Settings`
  (no CLI args). `load_workflow3_settings()` 에 env 매핑 추가.
- `ALIGN_FAIL_VLM_ROI_MIN_CONF`(기본 0.5), `ALIGN_FAIL_VLM_ROI_PAD`(기본 0.15).
- service: `"ui-venus"` (fallback `"mai-ui"` — `service_fallback_order` 재사용).
- latency: paused 1프레임이라 수백 ms 수용. live broad-scan 경로엔 **미적용**(real-time 위반).

---

## 7. A/B 평가

`golden_localization_eval_cond.py` 에 VLM-box 주입 모드 추가(office-only, VLM 호출). baseline(full-FOV)
vs +ROI:

| 지표 | 기대 방향 | 비고 |
|---|---|---|
| recall@8 | ↑ 또는 불변 | **절대 하락 금지**(2-tier 폴백이 보장) |
| rank1 | ↑ | decoy 감소 효과 |
| far/veryfar miss % | ↓ | 주 효과 영역 |
| false-ROI rate | 측정 | VLM 박스가 GT 를 배제한 비율(폴백 발동률) |
| VLM 거부율 | 측정 | key_visible=false 비율(정상) |

합격선: recall 무손실 + rank1/far-miss 유의 개선. 미달 시 보류(연구문서 우선순위 2 → 후순위 강등).

---

## 8. 리스크 & 가드

| 리스크 | 가드 |
|---|---|
| VLM 환각(엉뚱한 박스) | 2-tier 폴백, min_confidence, key_visible 게이트, pad_frac |
| relative_1000 좌표 오독 | `bbox_to_pixels(coord_system)` 명시 파싱, ok_button 검증된 헬퍼 재사용 |
| 주기성 패턴서 박스 모호 | ROI 는 넉넉하게(pad), 최종 peak 는 CV — periodicity 영향은 CV 단계서 흡수 |
| 좌표공간 혼동(frame vs screen) | 이 ROI 는 frame 좌표(=capture 공간), screen 아님. 주석/테스트로 고정 |
| office-only / Mac 미호출 | vlm_client None → 자동 off, dry-run 은 로직만 |
| latency | paused 한정, temperature=0, 단일 호출 |

---

## 9. 단계 체크리스트

- [ ] `vlm/prompts/prompt_align_key_roi.py` — `build_align_key_roi_prompt(source_hint)`
- [ ] `align/vlm_align_key_box.py` — `locate_align_key_roi(...)` (+ `_pad_to_roi`, ok_button 헬퍼 재사용)
- [ ] workflow_2 `vlm_align_key_box.py` 프롬프트 내용 포팅(검증된 문구 재사용)
- [ ] `config.py` — `vlm_roi_enabled`/min_conf/pad + env 매핑
- [ ] `correction.py` — roi_hint 배선 + 2-tier 폴백 + `[WARNING]` 예외 처리
- [ ] (Mac) dry-run: roi_hint=None 경로 불변 확인 + ValueError 폴백 단위 테스트
- [ ] (office) golden A/B: recall 무손실 + rank1/far-miss 개선 + false-ROI rate
- [ ] (ship) shadow(ROI 계산만, 매칭 미반영) → 검증 후 활성

---

## 10. 비범위

- VLM 이 align 좌표/클릭 결정 — 금지(doc §8).
- live broad-scan 에 ROI 적용 — real-time 위반.
- 엔진 매칭 알고리즘 변경 — 불필요(roi_hint 이미 지원).
- OK 버튼 locator 와 좌표공간 공유 — 다름(frame vs screen).

> 관련: [[project_ensemble_on_consensus_rejected]], [[project_ui_venus_official_grounding]],
> [[project_vlm_service_slug_not_model_name]], [[project_align_cond_files_and_coords]],
> [[feedback_no_office_data_to_mac]], [[feedback_click_pipeline_coarse_fine_confirm]].
