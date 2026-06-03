# HANDOFF — office 검증 계획 (photometric box + crosshair v2 + old/new A/B)

날짜: 2026-06-03 17:10
유형: handoff (다음 세션 — 오피스 실데이터 보유 시 — 이 문서로 이어받는다)
선행 저널: `docs/journals/260603/260603_165200_photometric-box-and-crosshair-v2-wire.md` (구현 상세·근거)
관련 커밋: `ab61d58` (photometric box + crosshair v2 연결 + inpaint 검증 강화), `9a0aa42`(앞 단계)
관련 메모리: [[photometric-box-home-validated-only]]

## 0. 한 줄 요약

새 검출 방법(photometric 흰 box, crosshair v2)은 **집/합성 데이터로만** 6/6 검증됨.
**오피스 실데이터(align_images/ 트리)에서 old vs new 를 A/B 로 측정**해 정말 더 나은지 확인하고,
나으면 정식 채택 + `align_similarity.py` 의 지표 접근을 교체한다. 이 문서는 그 **테스트 계획**을 보존한다.

## 1. 지금 상태 (이미 끝난 것 — 중복 설명 생략, 위 저널 참조)

- `align_point_correction.py`: 흰 box 검출이 photometric 1차(`_detect_white_box_photometric`) + adaptive 폴백
  (`_detect_white_box_adaptive`)로 분리됨. crosshair 는 production 경로가 `detect_crosshair`(v2)로 연결됨
  (구 `_detect_existing_crosshair` 는 probe 비교용으로만 잔존).
- `align_similarity.py`: 동일 `_detect_white_box` 를 import 하므로 photometric 이 자동 적용됨. crosshair 는
  이미 v2 사용. **단, align_similarity 는 crosshair 를 inpaint 하지 않고 ground-truth 위치 마커로 씀**(진단 도구).
- 강화 inpaint 검증(재검출 + edge-band 비)은 **throwaway checker(`crosshair_removal_check.py`)에만** 있고
  production 경로엔 아직 없음.
- 검증 범위: 전부 `templates/{whitebox,crosshair}_samples`(인터넷/합성)뿐. **오피스 미검증.**

## 2. 오피스 테스트 계획 (이 핸드오프의 핵심 — 반드시 보존)

### 2.1 가장 먼저 — overlay 값 1회 확인 (5분)

photometric 의 전제는 "overlay 가 고정 디지털 값으로 렌더돼 히스토그램상 분리된 섬을 이룬다"(합성에선 255).
**실제 RCS/IMAP overlay 값이 255 가 아닐 수 있다.** 첫 `align_img_from_rcp/IMAP0001·0002` 한 장에서
히스토그램 상단을 확인:
- 상단에 gap 으로 분리된 봉우리가 있나? 그 값(예: 248~253)이 무엇인가?
- 있으면 photometric 그대로 동작(`_detect_overlay_saturation` 가 클러스터를 잡음). 없으면 adaptive 폴백.
- 안 맞으면 `RCP_BOX_SAT_NOTCH_GAP`(기본 8, 범위 4~16) / `RCP_BOX_SAT_MIN_MASS`·`MAX_MASS` 만 조정.

### 2.2 제안 — standalone old/new A/B 하니스 (다음 세션이 구현)

> 이전 세션에서 작성 직전 사용자가 "나중에 테스트" 로 보류함. 아래 설계대로 만들면 된다.
> 파일(제안): `poc/workflow_2/office_method_ab_test.py` (throwaway, no argparse, 상수 설정, Korean docstring).

- **자산 해석**: 완전 override env(`ALIGN_EQP_ID`+`ALIGN_CLASS_NAME`+`ALIGN_RECIPE_NAME`) 있으면 단일,
  없으면 `iter_recipe_dirs()` 전체. `resolve_assets()` / `iter_msr_images()` / `load_gray()` 사용
  (API 는 `poc/workflow_2/align_fail_assets.py` 참조).
- **Box A/B** (rcp: `recipe_om`, `recipe_sem`): 이미지마다
  `_detect_overlay_saturation`, `_detect_white_box_photometric`, `_detect_white_box_adaptive` 를 각각 돌려
  기록: sat 값, photo bbox, adaptive bbox, production 선택(photo 우선), 둘 다 있으면 IoU.
  overlay 저장(photo=초록, adaptive=빨강). 분류: both / photo_only(=신규 회복) / adaptive_only(=신규 회귀, ⚠️) / neither.
- **Crosshair A/B** (msr 전체): 이미지마다 `detect_crosshair`(v2) 와 `_detect_existing_crosshair`(old) 둘 다 돌려
  기록: v2_xy/conf/reason, old_xy/conf, 둘 다 있으면 center 거리, 파일 라벨(`_tool_label` S/E).
  overlay 저장(v2=초록, old=빨강). 분류: both / v2_only / old_only / neither.
- **출력**: `DEBUG_IMAGE_DIR/office_method_ab/<ts>/` 에 per-recipe overlay + `report.json` + `summary.json`,
  stdout 에 verdict(검출 수 old vs new, 신규 회복/회귀 건수, S/E 별 분리).
- **판정 기준**: box 는 photo_only > adaptive_only 이고 photo 검출수 ≥ adaptive 면 신규 우세.
  crosshair 는 v2 검출수 ≥ old 이고 S 에서 검출, E 에서 과검출 없음이면 v2 우세.
  **오크롭/오검출보다 no-detect(폴백) 선호** — CV 좌표 권위 원칙(틀린 template 은 downstream 전체 오염).

### 2.3 Codex 검증 계획 (전체 파이프라인 캘리브레이션 — A/B 다음 단계)

타입별 라벨 배치 수집: recipe OM/SEM(+box), msr S(+crosshair), msr E(no crosshair), blur/focus-fail.
측정: box bbox IoU(수동 라벨 대비), inner-crop stroke 잔존, crosshair center error, S recall, E false-positive,
**inpaint 전후 매처 winner 변화**. 스윕 대상:
- box: `RCP_BOX_SAT_NOTCH_GAP`, `MIN_MASS`, `MAX_MASS`, `CLOSE_KERNEL`, `RCP_BOX_SIDE_MIN_COVERAGE`, `FRAME_MIN_RATIO`.
- crosshair: `SAT_THRESH_LADDER`(245→175), `SPAN_RATIO`(0.20~0.45), `GAP_BRIDGE_RATIO`(0.03~0.15),
  `MAX_THICKNESS_PX`, `BOTTOM_SCALEBAR_RATIO`, `LEFT_AXIS_RATIO`.
- 공통: `MIN_SHARPNESS_LAPVAR`(15~80).
목표: S recall↑ & E false-positive→0; 게이트별 reject 사유코드를 로깅해 *실제로 firing 하는* 게이트만 튜닝(과적합 방지).

## 3. align_similarity.py 교체 의향 (사용자)

- 사용자: align_similarity 의 지표(MI/NCC/chamfer 분리도, consensus staleness 등)에서 **높은 점수 확보에 계속 고전**.
  새 방법으로 **교체에 긍정적**. 단 "나중에 테스트 후" 결정.
- 함의: align_similarity 는 *진단/캘리브레이션* 도구다. 교체 시 (a) 무엇을 새 신뢰 신호로 쓸지(예: photometric
  box 검출 성공 + matcher score),(b) S/E 분리를 어떤 단일 지표로 대체할지 재설계 필요. **A/B 결과가 그 근거가 된다** —
  A/B 에서 신규 검출이 우세하면, align_similarity 의 무거운 metric 탐색을 단순한 검출-기반 신뢰도로 대체하는 안 검토.
- 폐기 이력 참고(반복 금지): reranker(MI·contour) A/B 실패 — `docs/study/reranker_ab_failure_analysis.md`.

## 4. 다음 세션이 쓰면 좋은 스킬

- **`tdd`** 또는 throwaway 우선: A/B 하니스는 throwaway 라 TDD 불필요하지만, production 교체 시점엔 회귀 테스트 추가.
- **`codex:rescue`**: 실데이터 결과 해석 / 게이트 튜닝 2차 의견 (이번 세션에서 유효; 토큰 만료 시 `!codex login`).
- 작업 마무리 시 **저널 작성**(이 폴더 규칙) + 메모리 갱신([[photometric-box-home-validated-only]] 를 office 결과로 업데이트).

## 5. 시작 체크리스트 (다음 세션)

1. `git pull` 최신화. `align_images/` 트리 존재 확인(오피스만).
2. §2.1 overlay 값 히스토그램 1회 확인 → photometric 전제 성립 여부 판단.
3. §2.2 `office_method_ab_test.py` 구현 → 실행 → overlay 육안 + summary 판정.
4. 우세하면: 정식 채택 + §3 align_similarity 교체 설계 + 메모리/저널 갱신.
   열세/혼조면: §2.3 게이트 스윕으로 캘리브레이션 후 재판정.
