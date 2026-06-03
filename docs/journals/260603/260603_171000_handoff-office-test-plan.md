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

### 2.2 BUILT — index ablation 하니스 `poc/workflow_2/align_index_ablation.py`

> 구현 완료(2026-06-03). 인자 없이 `uv run python poc/workflow_2/align_index_ablation.py`.
> (파일명에 `office_` 접두 금지 — .gitignore `**/office_*` 가 잡아 untracked 됨.)
> import/무데이터 graceful 종료까지 home 검증됨. **실데이터 path 는 오피스 첫 실행에서 확인.**

핵심 아이디어 = **2×2 factorial ablation** 으로 "흰 box 제거 + crosshair 제거" 두 변수를 분리해, `align_similarity`
와 **동일 지표**(S/E 분리 balanced accuracy + gt-in-topK recall)가 오르는지 같은 잣대로 잰다. 지표 함수
(`_separation`, `_gt_in_topk`, `_race`, `_window_roi`, `_build_templates`)를 align_similarity 에서 그대로 import
→ 정의 표류 없음.

- 축: template = center(구) | box(신, photometric 깨끗 crop); frame = raw(crosshair 있음) | inpaint(crosshair 제거).
  - `center__raw` = OLD baseline, `box__raw` = clean-tpl only, `center__inpaint` = crosshair-only, `box__inpaint` = NEW.
- 셀마다 free / at_center / at_crosshair align score 의 S/E 분리 + gt-in-topK. **at_crosshair bACC** 가 주 변별자.
- 자산: 완전 override env 면 단일 recipe, 아니면 `iter_recipe_dirs()` 전체.
- 출력: `DEBUG_IMAGE_DIR/office_ablation/<ts>/{rows.jsonl, summary.json}` + stdout 표 + **VERDICT**
  (OLD center+raw vs NEW box+inpaint 의 bACC delta).
- 판정: NEW bACC > OLD 면 교체 근거. box 셀 비면 photometric 미검출(폴백) → §2.1 overlay 값 확인.

> (선택) 검출-레벨 overlay A/B(photometric vs adaptive 박스, v2 vs old crosshair 그림 비교)는 보조 도구로
> 추후 필요시. 위 index ablation 이 "지표가 오르나?" 라는 본 질문에 직접 답하므로 1순위.

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

## 4. VLM+CV 백업 — escalation 설계 (계획 · 미구현, 사용자: "leave it as a plan")

순수 CV(신규)가 실패하는 케이스의 안전망. **always-on 병렬이 아니라 gated escalation** 으로 둔다.
구현은 **ablation 결과로 필요량을 가늠한 뒤** 결정(아래 "sizing"). 워크스트림 규칙 준수:
*"VLM 이 영역, CV 가 좌표"* — VLM 은 영역/실행가능성만, **최종 좌표·반복 stage 전이는 CV** (2026-05-25 확정).

- **왜 필요**: 순수 CV 가 못 지우는 실패 모드가 실재한다(align_similarity 진단이 입증) — photometric box 미검출
  (저대비/anti-alias overlay), flat chamfer / not-distinctive(reranker A/B 실패로 MI·contour 로는 못 메움),
  reference staleness, focus blur. 이때 *틀린 좌표를 내지 말고* escalate 해야 한다(좌표 권위 원칙).
- **트리거(이미 존재하는 신호 재사용)**: 흰 box 미검출(center-crop 폴백) / status ∈ {`not_distinctive`,
  `low_match_both`, `ambiguous_modality`} → escalate. score ≥ adjust_threshold & distinctive → CV 신뢰, VLM 미호출.
- **동작**: ① VLM 이 msr FOV 에서 align-key 영역(또는 not-present/infeasible) 식별 → ROI narrow.
  ② 기존 matcher 를 그 ROI 로 제한해 정밀 좌표 산출. VLM 답이 **낮은 CV 점수를 override 못 함**.
- **기반 스캐폴딩**: `vlm_align_key_box.py`, `vlm_sem_monitor_box.py`(부분 구현, CLAUDE.md). Flask VLM proxy 필요 →
  **office-only · 느림** → escalation 전용(default 아님).
- **Sizing(언제·얼마나)**: §2.2 ablation 의 *잔여 실패율*(box+inpaint 에서도 at_crosshair bACC / gt-topK 낮은
  recipe 비율)이 곧 필요량. 대부분 통과 → tail 안전망(lean). 다수 실패 → load-bearing(정식 구축).

## 5. 다음 세션이 쓰면 좋은 스킬

- **`tdd`** 또는 throwaway 우선: A/B 하니스는 throwaway 라 TDD 불필요하지만, production 교체 시점엔 회귀 테스트 추가.
- **`codex:rescue`**: 실데이터 결과 해석 / 게이트 튜닝 2차 의견 (이번 세션에서 유효; 토큰 만료 시 `!codex login`).
- 작업 마무리 시 **저널 작성**(이 폴더 규칙) + 메모리 갱신([[photometric-box-home-validated-only]] 를 office 결과로 업데이트).

## 6. 시작 체크리스트 (다음 세션)

1. `git pull` 최신화. `align_images/` 트리 존재 확인(오피스만).
2. §2.1 overlay 값 히스토그램 1회 확인 → photometric 전제 성립 여부 판단.
3. §2.2 `uv run python poc/workflow_2/align_index_ablation.py` 실행 → summary VERDICT(OLD vs NEW bACC) 확인.
4. 우세하면: 정식 채택 + §3 align_similarity 교체 설계 + 메모리/저널 갱신.
   열세/혼조면: §2.3 게이트 스윕으로 캘리브레이션 후 재판정.
5. 잔여 실패율로 §4 VLM 백업 필요량 판단(필요시에만 구축).
