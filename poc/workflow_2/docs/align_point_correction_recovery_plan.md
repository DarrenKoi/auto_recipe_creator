# Align Point Correction 실패 대응 계획

> 작성일: 2026-05-29 (개정)
> 대상: `poc/workflow_2/align_point_correction.py`
> 목적: `align_img_from_rcp` 에서 지정한 align 위치가 `align_img_from_msr` 의 S/E 이미지에서
> 다른 위치로 잡히는 문제를, **(1) 무엇이 실패했는지 먼저 계량하고 → (2) 가장 싼 진단부터
> 순서대로** 적용 가능한 알고리즘으로 좁혀 해결한다.

---

## 1. 문제 정리

기대 동작:

1. `align_img_from_rcp/IMAP0001`(OM), `IMAP0002`(SEM) 에는 엔지니어가 그린 흰색 box 가 있고,
   그 안쪽이 **유일하게 식별 가능한 영역**이다. 진짜 align point 는 box 중심이 아니라 **이미지 중심**.
2. 그 위치가 충분히 유니크하므로 `from_msr` 의 S(success) 이미지에서도 같은 물리 위치를 찾아야 한다.
3. 그런데 현재 S 이미지조차 다른 위치가 선택된다.

가능한 원인(가설):

- **(C1) key 품질**: rcp box / fallback crop 이 가리키는 영역이 *애초에 그 이미지 안에서 유일하지 않다*.
  유일하지 않은 패턴을 template 으로 쓰면 msr 에서 못 찾는 게 당연하다.
- **(C2) 매칭기 강건성**: `from_rcp` 와 `from_msr` 의 contrast/brightness/focus/scale 이 달라서
  (= align fail 의 원인 그 자체) Chamfer/ORB 가 wrong feature 에 끌린다. edge·keypoint 기반은
  픽셀값 drift 에 약하다.
- **(C3) crosshair 오염**: msr 의 tool crosshair 가 긴 밝은 edge 로 들어와 matcher 를 속인다.
- **(C4) scale miss**: correction path 의 `COMPARE_SCALES=(0.6,0.75,0.85,1.0)` 는 **확대 방향(>1.0)을 안 본다**.
  반면 기본 matcher 는 `1.2, 1.4` 도 본다. msr 이 더 확대된 배율이면 best 가 엉뚱한 곳에 잡힐 수 있다.

핵심 판단:

```text
S label 도 정답으로 믿지 않는다.
S/E 는 tool self-report 이며, 정답 여부는 rcp reference 와 msr evidence 의 geometry 로 다시 판단한다.
유일한 위치가 두 이미지에 모두 존재하면 → 보정. 한쪽에 없으면 → "다른 위치/문제" → stage 이동 필요.
```

### 1.1 "유일성"은 두 개의 다른 질문이다 (이 플랜의 출발점)

| | 질문 | 현재 상태 |
|---|---|---|
| **(A) key self-uniqueness** | rcp box 영역이 *rcp 이미지 자신* 안에서 유일한가? | **미측정** — Phase 0 |
| **(B) match distinctiveness** | msr 에서 찾은 위치가 *msr 이미지* 안에서 유일한가? | 코드에 있음 (`_match_against` ratio test) |

(A) 를 먼저 재면 실패 원인을 **C1(key 품질)** 쪽으로 좁힐 단서가 된다. 단 self-match 진단은
C1/C2 를 *깔끔하게 가르지 못한다* (§Phase 0 함정 참고) — early gate 로만 쓰고, 확정은 paired
sanity(Phase 0b) + 계측(Phase 1) 결과를 합쳐 내린다. 알고리즘을 더 붙이기 전에 (A) 진단이 최우선이다.

---

## 2. 이미 구현된 방어 장치

`align_point_correction.py` 가 이미 가진 것 (중복 구현 금지):

- rcp 흰 box 검출 + 안쪽 crop template (`_detect_white_box`, `_inner_crop_for_box`, `_build_rcp_template`)
- align point = image center offset (`align_offset_xy`); box 미검출 시 중심 15% area crop fallback
- msr crosshair 검출 (`_detect_existing_crosshair`) + inpaint 제거 후 매칭 (`_inpaint_crosshair`)
- crosshair prior ROI 재매칭 (`_match_with_prior_roi`, `_match_against`)
- match distinctiveness ratio test (= 위 1.1 의 (B))
- blur frame gate, out-of-frame flag, S label suspect surfacing, scale-bar OCR tiebreak

따라서 대응은 "기능 추가"가 아니라 **실패 계량 → key 품질 진단 → 매칭기 강건화 → VLM 게이트** 순으로
decision policy 를 엄격하게 만드는 것이다.

---

## 3. 대응 원칙

**좌표 권한은 CV, VLM 은 region 식별·feasibility 판정만.** (`workflow_2` 전역 원칙, CLAUDE.md)

- VLM 좌표는 coarse 하고 픽셀 repeatability 가 없다 → 최종 click/reposition 좌표로 쓰지 않는다.
- VLM 은 ① search ROI 를 좁히는 coarse hint, ② CV 결과가 "정말 같은 구조인가" 판정하는 **feasibility 게이트**
  로만 쓴다. CV score 가 낮은데 VLM 답변으로 override 하지 않는다.
- 모든 판정은 재현 가능한 threshold + overlay audit 로 남긴다.

---

## 4. 실행 계획 (순서가 곧 우선순위)

각 Phase 는 **완료 기준(=다음으로 넘어갈 근거)** 을 가진다. 앞 Phase 의 결과가 뒤 Phase 의 필요 여부를
결정한다. (순서는 Codex 검토 반영 — top-N 후보화·계측이 먼저, MI 는 후보 검증기로 뒤로.)

> 설계 원칙: **top-N 후보화 → 후보별 검증(contour/AKAZE/MI)** 구조. MI·AKAZE 같은 비싼 연산은
> 절대 full-frame 으로 돌리지 않고 top-N candidate crop 에만 적용한다.

### Phase 0 — input sanity + rcp key 품질 진단 (가장 싸고, 방향의 단서)

rcp box 안쪽 template 을 **rcp 이미지 자신** 위로 슬라이드시켜 self-similarity / autocorrelation
surface 를 만든다. true 위치에 sharp peak 1개 + 나머지 낮음 → 유일에 가깝다.

```text
rcp template --(compute_align_key_score 를 rcp→rcp 로 재사용)--> response map
  -> true peak 를 *template 크기/IoU 기준 넉넉히* 마스킹 (overlap 으로 2nd peak 부풀림 방지)
  -> non-overlap 2nd peak 측정 -> uniqueness_score = 2nd / 1st (낮을수록 유일)
```

**함정 (Codex 지적) — 그래서 보조 지표를 함께 기록한다:**
- self-match 는 true peak 주변 overlap 때문에 2nd peak 가 과대평가됨 → 마스킹 반경을 크게.
- rcp 에서 유일해도 msr 에서 그 구조가 공정/focus/scale/charging 으로 사라지거나 변형될 수 있음.
- **Chamfer 기반 self-similarity 는 현재 매처의 bias 를 그대로 재측정** (edge density 낮거나 긴 직선
  위주면 점수 왜곡). 따라서 단일 점수로 C1/C2 를 확정하지 않는다.

함께 기록: `edge_density`, `entropy/texture`, **PSR(peak-to-sidelobe ratio)**, non-overlap 2nd peak,
**localization peak width**, box 검출 신뢰도/fallback 여부, **scale sweep**(0.6~1.4 전 구간, C4 점검).

**Phase 0b — paired sanity check:** rcp template ↔ *알려진 정상 S 이미지* 한 장을 직접 매칭해보고,
true 위치를 잡는지 manual 로 확인. Phase 0 self-match 만으로 "C2 확정" 하지 않기 위한 anchor.

- RCS 불필요, Mac batch. 산출: `align_correction/uniqueness/<recipe>__{om,sem}_uniqueness.jpg` + score 표.

**완료 기준:** 모든 recipe 의 rcp key 에 대해 uniqueness + 보조 지표가 산출되고, "유일하지 않음/box
신뢰 불가/scale 불안정" key 가 분리된다. (이것만으로 C1/C2 를 단정하지 않음 — Phase 1 과 합산.)

### Phase 1 — 실패 계측 강화

실패 이미지 하나를 열었을 때 "왜 이 좌표가 선택됐는지" 를 overlay + JSON 만으로 설명 가능하게 한다.

JSONL/summary 필드:

- rcp box 검출 여부 / box bbox / inner crop bbox / `align_offset_xy`
- **4-way ablation**: `crosshair prior {off,on} × inpaint {off,on}` 각 match 결과 (순환 검증 방지용)
- free match / prior ROI match / top-N candidate (좌표·score·2nd-best ratio) / 최종 채택 match
- **`best_scale` 분포 + scale-band sweep 결과** (C4 — 확대 방향 miss 점검)
- score map thumbnail, final `corrected_xy` ↔ crosshair 거리
- 집계: `not_distinctive`, `low_match_both`, `rcp_box_missing`, `crosshair_prior_used`, `scale_unstable`
- S label 이면서 correction distance 큰 row 만 모은 **suspect review index**

**완료 기준:** 위 설명 가능성 + suspect index 생성 + golden(manual) overlay set 준비.

#### Phase 1 review/triage 출력 레이아웃 (100+ recipe 디버깅용)

문제: test set 이 100+ 로 늘어 recipe 폴더를 하나씩 여는 게 불가능. 해결: 저장은 recipe 별(source of
truth)로 두되, **review 뷰를 status 축으로 한 겹 더** 만든다 — "어떤 실패 유형인가" 가 디버깅 축이므로.

```text
debug_images/align_correction/<batch_ts>/
├─ by_recipe/<eqp>__<class>__<recipe>/      # source of truth (풀해상도)
│    ├─ rcp_{om,sem}_box_overlay.jpg
│    ├─ overlay/<msr>_overlay.jpg
│    ├─ results.jsonl   summary.json
├─ by_status/                               # ← review 축 (status별 thumbnail 재배치)
│    ├─ suspect_success/<recipe>__<msr>.jpg
│    ├─ not_distinctive/  low_match_both/  scale_unstable/
│    ├─ box_untrusted/  msr_unrecognizable/  ok/
├─ index.html              # status별 그룹 + thumbnail + 지표, worst-first 정렬, 클릭→풀해상도
└─ batch_summary.json      # status별 카운트
```

규칙:
- **primary 그룹 축 = status** (recipe 아님). 한 폴더만 열면 전 recipe 의 같은 실패 유형이 모인다.
- thumbnail 파일명 prefix = `<recipe>` 로 출처 식별. **복사**로 둔다 (Windows symlink 불안정).
  풀해상도 원본은 `by_recipe/` 에만 — 중복 저장 최소화.
- **index.html**: status 섹션별로 thumbnail grid, 각 카드에 label·score·corrected↔crosshair dist·
  best_scale·uniqueness 표기. **worst-first 정렬** (correction distance 큰 / uniqueness 나쁜 순) — 위에서
  몇 개만 봐도 문제 케이스 파악. 카드 클릭 → `by_recipe/` 풀해상도 overlay 로 이동.
- CSV 는 만들지 않음 (사용자 결정). 수치 통계는 `batch_summary.json` + index.html 상단 카운트로 충분.
- **HTML 생성은 순수 Python 표준 라이브러리** (f-string 템플릿 → `Path.write_text`). 의존성 추가 없음.
  thumbnail 은 **상대경로 `<img>`** (base64 인라인 X — 100+ 면 파일이 수십 MB), `loading="lazy"` 필수
  (스크롤 시 로드, 브라우저 멈춤 방지), status 토글 정렬은 인라인 `<script>` 로 자체 포함 (서버 불필요).

**완료 기준:** index.html 한 장으로 100+ recipe 의 실패를 status별·worst-first 로 훑고, 클릭으로 개별
overlay 까지 도달 가능. by_status/ 폴더만으로도 브라우저 없이 유형별 검토 가능.

### Phase 2 — score-map 기반 top-N / NMS distinctiveness

현재의 retry-mask 방식(`_match_against`)을 **명시적 score map → NMS → top-N 후보 목록**으로 승격한다.
best 중심 판단을 후보 집합으로 바꿔 wrong-top-1 자동 채택을 막고, 이후 검증기(Phase 3)의 입력이 된다.

```text
chamfer score map -> NMS(후보 간 최소 거리) -> top-N candidate
accept 후보화:
  best.score >= threshold
  and best.score - second.score >= min_gap
  and second.score / best.score <= max_ratio
else: status = not_distinctive | low_match_both
```

**완료 기준:** 유사 구조가 여러 개인 S 이미지에서 wrong top-1 을 채택하지 않고 후보를 보류/분리한다.

### Phase 3 — 후보 verifier (contour/AKAZE 먼저, MI 는 reranker)

Phase 2 의 top-N **후보 crop 에만** 비싼 검증을 적용한다. full-frame 금지.

1. **contour geometry**: box/nested box/corner/cross layout 비교 (keypoint 가 약한 mark 의 직접 evidence).
   `CLAHE → blur → Canny/adaptive → findContours → approxPolyDP → box count/center/size/nesting`.
2. **AKAZE fallback**: ORB inlier 가 낮은 후보에만. inlier ratio 를 feature score 로.
3. **Mutual Information (reranker)**: 후보 crop 에서 template↔candidate 의 joint histogram MI 로 rerank /
   local refinement. **MI 는 1차 탐색기가 아니다** — full-frame dense MI 는 `윈도우수×template픽셀×bins`
   비용 병목 + 반복패턴 false positive. 밝기/대비 drift 강건성(=C2 대응)은 후보 검증 단계에서 활용한다.

```text
match if chamfer >= C_min and (feature >= F_min or contour >= G_min) and distinctive
  (+ MI rerank 로 동점 후보 간 순위 결정)
```

- MI 의존성: `scikit-image` (모델 가중치 불필요).
- 천장이 부족하면(저텍스처·반복패턴 심해 MI 로도 안 됨) 학습 매칭기(LoFTR / SuperGlue·LightGlue) 검토.
  단 모델 배포 비용이 크므로 **위 단계로 한계를 확인한 뒤**에만 착수.

**완료 기준:** box-style key 의 edge-clutter false positive 감소. 각 검증기 latency 측정 후 live path
가능 여부 판단 (불가 시 offline/audit 전용).

### Phase 4 — acceptance / reject-state policy

`ok` 를 남발하지 않고 **reject 상태를 우선** 정의한다 — wrong 좌표를 stage 이동에 쓰는 게 최악이므로
"모르겠으면 보류" 가 안전하다.

reject 상태: `not_distinctive`, `low_match_both`, `scale_unstable`, `box_untrusted`,
`msr_unrecognizable`, `no_crosshair_drawn`(도구 포기 신호).

**acceptance 는 crosshair 수렴이 아니라 golden set 기준** — crosshair prior 가 결과를 crosshair 로
끌어당긴 걸 "성공"으로 오인하는 순환 검증을 피한다. Phase 1 의 4-way ablation 으로 prior/inpaint 의
순효과를 분리해 본다.

**완료 기준:** golden set 에서 정/오답이 reject 상태로 정확히 분류되고, crosshair 가 속인 케이스가
inpaint off/on 비교로 드러난다.

### Phase 5 — VLM feasibility 게이트 + coarse ROI

CV 가 좌표를 낸 뒤, `from_rcp | from_msr` 를 나란히 붙여 VLM 에 "같은 구조인가" 를 묻는다.
"아니오" → 사용자가 말한 *"live SEM 이 완전히 다른 위치 → 마우스 이동 필요"* 분기 트리거.

composite 입력:

```text
+----------------------+----------------------+
| left: from_rcp       | right: from_msr      |
| box / center marker  | (원본 + crosshair-   |
| inner crop / offset  |  inpaint 둘 다 실험) |
+----------------------+----------------------+
```

strict JSON contract:

```json
{
  "right_candidate_visible": true,
  "right_bbox": {"x": 120, "y": 80, "w": 64, "h": 64},
  "same_structure": true,
  "reason": "similar nested box/corner pattern",
  "confidence": 0.0
}
```

- bbox 는 오른쪽 image 좌표계 기준. `right_candidate_visible=false` 면 bbox=null.
- `confidence` 는 기록용 — 최종 score 로 쓰지 않음. bbox bounds/size/aspect 검증 실패 시 discard.
- 사용 방식: VLM bbox → `roi_hint` 로만 → 그 ROI 안에서 CV(top-N/contour/MI) 재검증 → CV gate 통과 시에만 채택.

**완료 기준(VLM 성공 기준 = "정확한 click 좌표"가 아님):**
- true region 포함 bbox **recall** 이 높은가
- wrong region bbox 빈도가 낮은가
- bbox 를 roi_hint 로 넣었을 때 CV top-N 품질이 좋아지는가
- CV 가 reject 해야 할 케이스를 VLM 이 억지로 통과시키지 않는가

---

## 5. 테스트 / 검수

```bash
uv run python poc/workflow_2/align_point_correction.py        # batch (uniqueness map + 보정)
uv run python poc/workflow_2/test_align_key_match.py          # 합성 smoke (10/10)
uv run python poc/workflow_2/test_match_on_captured_frames.py # 캡처 프레임 (있을 때)
```

리뷰 산출물:

- `debug_images/align_correction/uniqueness/*` (Phase 0)
- `.../results.jsonl`, `summary.json`, `rcp_{om,sem}_box_overlay.jpg`, `overlay/*_overlay.jpg`
- S label suspect review index, top-N candidate overlay, VLM probe report

acceptance criteria:

- (Phase 0) rcp key 의 uniqueness + 보조 지표(PSR/edge_density/scale sweep)가 모든 recipe 에 산출되고,
  유일하지 않음/box 신뢰 불가/scale 불안정 key 가 분리된다. Phase 0b paired sanity 가 anchor 로 통과.
- (Phase 1) 4-way ablation(prior×inpaint) + best_scale 분포가 기록되어, crosshair prior 와 inpaint 의
  순효과를 분리해 볼 수 있다.
- (Phase 2) wrong top-1 은 second-best ratio 로 채택되지 않고 후보 보류된다.
- (Phase 3) box-style key 에서 contour/MI 검증으로 edge-clutter false positive 가 감소한다.
- (Phase 4) **acceptance 는 golden set 기준** — crosshair 수렴을 성공 기준으로 쓰지 않는다(순환 검증 금지).
  정/오답이 reject 상태로 정확히 분류된다.
- (Phase 5) VLM 은 ROI hint / feasibility 판정으로만 쓰이고 최종 좌표를 단독 결정하지 않는다.

---

## 6. 참고 자료

- scikit-image registration / metrics: https://scikit-image.org/docs/stable/api/skimage.registration.html
- Mutual information (multimodal registration) 개념: ITK/Insight, Mattes MI
- OpenCV AKAZE matching: https://docs.opencv.org/4.x/db/d70/tutorial_akaze_matching.html
- OpenCV contours / shape descriptors: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- LoFTR (detector-free dense matching): https://arxiv.org/abs/2104.00680
- SuperGlue / LightGlue (learned feature matching): https://arxiv.org/abs/1911.11763 , https://arxiv.org/abs/2306.13643
