# Align Point Correction 정적 배치 테스트 구축

**일자**: 2026-05-28
**범위**: `poc/workflow_2/align_point_correction.py` (신규), `poc/workflow_2/align_fail_assets.py` (수정)
**상태**: 5회 commit + push 완료 — `542bb05`, `a50ef53`, `368eb08`, `a27b2c8` 등

---

## 1. 진행 사항

### 1.1 신규 모듈 설계 (계획 → 구현)

이전 세션에서 `vlm_wait_input_ok_button.py` / `outline_live_sem_box.py` 가 완성된 후, 다음 단계로 **align fail 시 correct align point 를 정적 이미지로부터 산출하는 배치 테스트** 가 필요해졌음. plan 작성, 사용자 인터뷰, 구현, code review (3 단계 finder + verify), 5 가지 fix landing 까지 한 사이클 수행.

- Plan 파일 작성: `/Users/daeyoung/.claude/plans/we-have-successfully-tested-glowing-eclipse.md`
- 사용자 확인 받은 핵심 결정사항:
  - 크로스헤어는 white/grayscale 만 (color 채널 트릭 사용 불가)
  - Modality 결정: OM/SEM template 둘 다 돌려 점수 race (OCR 보조)
  - S 라벨도 의심해야 함 (도구가 실패를 success 로 잘못 분류하는 경우 존재)
  - blur 한 msr 은 정렬 위치 추정 불가 → 다음 행동 권장 status 분기

### 1.2 코드 작성

`poc/workflow_2/align_point_correction.py` 전체 작성 (~1450 LOC). 주요 컴포넌트:

- **`_RcpTemplateBundle`** dataclass: template + align_offset_xy + detected_box + inner_crop 묶음.
- **`_detect_white_box`**: 엔지니어가 rcp 에 그린 흰색 unique-area 박스를 top-hat → Otsu → contour 로 검출. 면적/aspect/hollowness/edge-margin gate 적용.
- **`_centered_area_crop_bbox`**: 박스 검출 실패시 fallback — 이미지 중심 기준 면적 20% 크롭.
- **`_inner_crop_for_box`**: 박스 outline stroke 를 피해 안쪽만 잘라 template 으로.
- **`_build_rcp_template`**: rcp 이미지 → `_RcpTemplateBundle` 변환. align_offset = image_center − template_center 계산.
- **`_detect_existing_crosshair`**: msr 의 도구 그린 grayscale crosshair 를 top-hat + row/column projection 으로 검출. span / prominence / 가장자리 zone 제외 gate.
- **`_frame_sharpness`**: Laplacian variance focus measure (blur gate).
- **`_inpaint_crosshair`**: 검출된 crosshair 라인을 ±2 px 마스크로 cv2.inpaint(TELEA) — CV 매칭 distractor 제거.
- **`_match_with_prior_roi`**: prior 위치 주변 좁은 ROI 로 `compute_align_key_score` 재실행.
- **`_match_against`**: free 검색 + (crosshair 가 있으면) inpaint 후 매칭, free vs prior 비교, align_offset 적용, 프레임 안 clipping + out_of_frame 플래그.
- **`_race_templates`**: OM/SEM bundle 각각 매칭, 점수 높은 쪽을 winner.
- **`_ocr_scale_bar`**: 프레임 하단 8% strip 을 PaddleOCR-VL 로 OCR → "X um/nm" regex 파싱 → modality 힌트.
- **`_process_msr_image`**: blur gate → crosshair 검출 → race (with prior) → OCR → status 결정 → audit/tiebreak.
- **`_process_recipe`**: rcp 두 modality bundle 준비, 모든 msr 이미지 순회, error isolation, summary.json 집계.
- **`run` / `_resolve_recipes_to_process` / `_has_full_override`**: batch-all 기본, 환경변수 풀 override 시 single recipe 모드.

### 1.3 Code review + 5 가지 fix landing

세션 중간에 `/code-review` 수행. 3 finder agent (line-by-line, removed-behavior, cross-file) 가 ~13 candidate 식별 → verify → 10 finding 산출. 그중 5 가지를 즉시 fix:

1. `_pick_current_sem` mtime 정렬 → visit-order 정렬 (`iter_msr_images` 와 일관성)
2. `_detect_white_box` 에 rectangularity gate (aspect + hollowness fill-ratio)
3. pad-undo 후 negative 좌표 → frame 안으로 clip + `out_of_frame=True` 플래그
4. blur frame 분기에서 crosshair 검출 skip (noise-driven spurious xy 방지)
5. blur frame overlay 에 fake green 마커 제거 → 빨간 "NO CORRECTION" 배너

이후 사용자 실데이터 피드백을 반영한 추가 개선:

6. rcp box gate 엄격화: max area 0.70 → 0.40, edge-margin 2 px (절반 차지 + 가장자리 닿는 박스 거부)
7. Align point = image_center (박스 center 아님) — `align_offset_xy` 도입, 매치 후 offset 적용
8. crosshair-anchored spatial prior — free 매치와 prior 거리 > 30 px 면 prior ROI 재매칭, prior 점수 합리적이면 채택
9. 박스 검출 실패시 전체 이미지 대신 면적 20% 중앙 crop 사용
10. msr 의 도구 crosshair 를 매칭 전에 inpaint 로 제거 (S 라벨에서 green marker 가 엉뚱한 곳 가리키던 root cause)

### 1.4 Codex rescue review

사용자 요청으로 `/codex:rescue` 도 별도 실행. 8가지 finding 중 4가지 (no_crosshair_drawn 우선순위, ambiguous vs low_match_both 순서, 예외 isolation, 힌트-vs-winner 의미 분리) 도 landing.

---

## 2. 수정 내용

### 2.1 신규 파일
- `poc/workflow_2/align_point_correction.py` (1,450+ lines): 위에 정리한 전체 파이프라인.

### 2.2 수정 파일
- `poc/workflow_2/align_fail_assets.py`:
  - `_VISIT_ORDER_RE`, `_visit_order(path)` 추가 — 파일명에서 A000X 정수 추출.
  - `iter_msr_images(assets)` 추가 — from_msr 이미지를 visit-order 오름차순 sort.
  - `_pick_current_sem` 의 mtime 정렬을 visit-order 정렬로 교체 (callers 간 일관성).

### 2.3 Git 커밋 시퀀스
1. `368eb08` Add static batch test for align-point correction (CV race + OCR tiebreak)
2. `a50ef53` Tighten rcp box gates and anchor CV match with msr crosshair prior
3. `542bb05` Fallback to centered 20%-area crop when rcp white box is not detected
4. `a27b2c8` Inpaint the tool's crosshair out of the msr frame before CV match

### 2.4 산출물 구조 (recipe 당)
```
poc/workflow_2/debug_images/align_correction/<eqp>__<class>__<recipe>__<ts>/
├─ rcp_om_box_overlay.jpg    노란 박스, 초록 inner crop, 시안 offset 화살표, 파란 align point crosshair
├─ rcp_sem_box_overlay.jpg
├─ overlay/<msr>_overlay.jpg  빨간 도구 crosshair, 초록 corrected_xy, 노란 보정 화살표
├─ results.jsonl              row 당 status / winner_modality / winner_source / corrected_xy /
│                             crosshair_xy / scale_bar_* / out_of_frame / used_crosshair_prior /
│                             free_score / prior_score / prior_match_distance_px
└─ summary.json               status_counts, modality_distribution, suspect_success_images,
                              no_crosshair_drawn_images, unrecognizable_images,
                              modality_disagreement_images, tiebreak_applied_images,
                              processing_error_images, crosshair_prior_applied_images, rcp_box

+ 최상위 batch_summary_<ts>.json
```

### 2.5 Status 결정 트리 (각 msr 이미지)
1. Laplacian variance < 30 → `msr_unrecognizable` (corrected_xy=null, overlay 에 NO CORRECTION 배너)
2. 양쪽 template 부재 → `no_templates`
3. winner 점수 < 0.40 OR winner out_of_frame → `low_match_both`
4. modality margin < 0.05 → `ambiguous_modality`
5. crosshair 미검출 → `no_crosshair_drawn` (corrected_xy 는 유효)
6. crosshair 있고 보정 거리 < 3 px → `already_aligned`
7. 그 외 → `ok`

OCR tiebreak: status 가 `ambiguous_modality` 또는 `low_match_both` 일 때만 scale-bar hint 로 modality 교체. winner_source 가 "cv" → "ocr_tiebreak" 로 표시.

---

## 3. 다음 단계

### 3.1 사용자가 사무실 Windows 환경에서 검증해야 할 항목

이 모듈은 macOS 에서 import smoke check 만 통과한 상태. 실데이터로 다음을 검증해야 함:

- [ ] `rcp_om_box_overlay.jpg` 와 `rcp_sem_box_overlay.jpg` 에서 노란 박스가 엔지니어가 그린 unique-area 박스 위에 정확히 얹히는지
- [ ] Align point (파란 crosshair) 가 이미지 정중앙에 있고 offset 화살표가 그 방향을 가리키는지
- [ ] S 라벨 이미지에서 green corrected_xy 가 도구의 white crosshair 와 일치하는지 (crosshair-prior + inpaint 효과 확인)
- [ ] `summary.json::ocr_client_initialized` 가 true 인지 (Flask proxy 도달 가능 여부)
- [ ] `crosshair_prior_applied_images` 에 들어간 row 가 실제로 free-search 가 틀린 케이스인지 (audit)
- [ ] `modality_disagreement_images` 에 들어간 row — CV race vs OCR scale-bar 가 다른 결론을 낸 케이스, 임계값 (100 µm) 재검토 자료

### 3.2 알려진 한계 (Codex 가 지적, 데이터 부족으로 defer)

- 매우 작은 박스 (< 14 px) 의 fallback 경로는 outline stroke 가 포함된 bbox 사용 — 실데이터에서 fire 안 할 가능성이 높지만 fire 하면 wrong template
- `_parse_scale_bar_um` 의 `max()` 픽 — OCR 이 환각된 "X µm" 토큰 여러 개 반환할 때 가장 큰 값 채택 → 잘못된 modality hint 가능성
- Windows 예약 문자 (`:`, `*`, `?` 등) 가 recipe 이름에 들어가면 `out_dir.mkdir` 실패 — 현재 office MES 에서는 알파뉴메릭만 보였다고 가정
- 초 단위 timestamp collision — 동시 launch 또는 0-msr 레시피 빠른 처리시
- `cv2.imwrite` non-ASCII 경로 Windows 무음 실패 — 비ASCII 폴더명 출현시 대비 필요

### 3.3 향후 통합

검증 통과 후 `live_align_search.py` 의 Phase B confirm 단계에 동일 primitive 를 옮길 예정. 살려야 할 것:
- `_detect_white_box` + `_centered_area_crop_bbox` (rcp template build)
- `_detect_existing_crosshair` + `_inpaint_crosshair` (msr 전처리)
- `_match_with_prior_roi` (crosshair-anchored 매칭)
- `_RcpTemplateBundle` 의 align_offset 약속

### 3.4 (필요시) 사용자 의사결정 항목

다음 두 가지는 실데이터 결과 보고 사용자에게 물어볼 사항:

- TRUST_SUSPECT_PX (15 px) 임계값 — 첫 실데이터셋에서 suspect_success 수치 확인 후 조정
- CROSSHAIR_PRIOR_DISAGREEMENT_PX (30 px) 임계값 — free vs prior 거리 분포 보고 조정

---

## 4. 메모리 업데이트

이번 세션에서 도출된 **재사용 가능한 도메인 지식** 을 MEMORY.md 에 반영할 만한 항목:

- **rcp 이미지의 흰 박스 + 이미지 중심 align point 관계** — 박스는 "unique area 식별 단서" 뿐, align point 는 이미지 정중앙. 매칭 후 align_offset (image_center − box_center) 적용 필수.
- **msr 의 도구 crosshair 는 매칭 distractor** — Chamfer matcher 가 긴 밝은 직선에 lock-on 되므로 매칭 전 inpaint 제거가 정답.
- **S 라벨도 의심해야 함** — 도구가 false-success 로 잘못 분류하는 경우 존재. 보정 거리 큰 S 는 suspect_success 로 surfacing.

memory 파일에 새 entry 가 필요한 항목은 사용자 확인 후 별도 진행 권장. 본 저널 파일 자체가 현재 작업의 권위적 기록 역할.

---

**관련 파일**
- 모듈: `poc/workflow_2/align_point_correction.py`
- 의존: `poc/workflow_2/align_fail_assets.py`, `poc/workflow_2/align_key_matcher.py`
- 설계 문서: `poc/workflow_2/docs/study/runbooks/workflow_2_procedure.md` (CV authority, VLM feasibility-only 규칙)
- Plan 파일: `~/.claude/plans/we-have-successfully-tested-glowing-eclipse.md`
