# 재등록/consensus 평가 현황 정리 + workflow_3 GUI 자동화로 전환

- 날짜: 2026-06-26 08:25
- 브랜치: main
- 주제: golden_reregister_report_cond / golden_consensus_eval_cond 테스트 현황 점검, 속도 개선(REREGISTER_TOPN), 오피스 첫 worklist 실행 결과 해석, workflow_3 GUI 자동화로 복귀, **PM dropdown 위치탐지를 2-VLM(coarse→fine)으로 재구현**

---

## 1. 진행 사항

이번 세션은 **코드 작성보다 현황 정리/진단/운영 절차 확립**이 중심이었다. workflow_2 벤치(재등록 우선순위 + consensus 평가)의 테스트가 어디까지 왔는지 짚고, 오피스 실행을 막던 속도 문제와 데이터 커버리지 문제를 규명했다.

1. **`golden_reregister_report_cond.py` 실행 전제 확인**
   - `golden_eval_config.py`는 *실행에는* 선택사항(loader가 `try/except ImportError`로 기본값 폴백). 실제로 필요한 건 **데이터**: `ALIGN_GOLDEN_ROOT` 미설정 시 빈 루트 walk → `no_data` 조기 종료(`run()`, line 1228~1237).
   - 오피스 풀 출력에는 `GOLDEN_ROOT`(필수) + `HISTORY_ROOT`(consensus rank-1용) + `EFRAME_ROOT`(Phase 2용)를 `golden_eval_config.py`에 설정해야 함을 정리.

2. **`golden_consensus_eval_cond.py` 결론 재확인 (이전 실험들)**
   - consensus lift = +0.40 (실효), clean-vs-raw A/B → crosshair 가짜 lock 가설 기각, consensus에 ensemble proposer 얹는 안 기각(잔여 miss 48% 구조적), **SEM은 ranking/distinctiveness 문제 = matcher-fusion 소진**(template-bank rank-1 ~0.5 = ADR 0006).
   - 결론: 남은 SEM 실패는 matcher가 아니라 **key 재등록(distinctiveness)** 레버로 풀어야 함.

3. **report ↔ consensus-eval 계약(contract) 검증**
   - report의 `_load_consensus_rank1` → `_build_rank1_lookup`이 소비하는 키(`recipe`, `modality`, `rcp_rank1_rate`, `cons_rank1_rate`, `n_S_loo`, `cons_pool_n`)를 **생산 측(`align_similarity._consensus_template_ab`, lines 1082~1093)이 모두 emit**함을 확인. 즉 결론(rank-1 측정 → 재등록 worklist)은 코드 레벨에서 완전히 맞물려 있고 테스트도 양쪽에 존재.
   - `summary.json`(consensus eval line 1035~1036)이 join 대상이며, report는 최신 `debug_images/golden_consensus_eval_cond/<ts>/summary.json`을 자동 발견(`_resolve_consensus_summary`).

4. **오피스 실행 절차 확립 (consensus eval 먼저)**
   - Step1 `golden_eval_config.py`에 GOLDEN_ROOT+HISTORY_ROOT 설정 → Step2 `golden_consensus_eval_cond.py` 실행 → Step3 `summary.json`의 `n_recipes`/`per_recipe[].rcp_rank1_rate` 검증 → Step4 `golden_reregister_report_cond.py` 실행(자동 join) → Step5 `rank1-join: matched M/N` 확인, 필요 시 `REREGISTER_DISTINCT_FLOOR` 보정.
   - 기본값 유지 규칙: `CONSENSUS_COREGISTER=1`, `CONSENSUS_CLEAN_FRAME=1`(production-faithful, clean-vs-raw 결론 근거).

5. **속도 병목 규명 + 영구 설정 적용**
   - 병목 = `_suggest_for_row`의 박스 제안 sweep: (flagged recipe) × (`_iter_candidate_boxes` 3 scales × stride 슬라이드 = 수십~수백 박스) × (S 프레임) ensemble 매칭. 코드 주석도 `>10분` 명시(line 77).
   - worklist(rank-1 fix-type 분류)는 sweep과 **무관**(순수 consensus rank-1 join) — sweep은 NEW_REGION 교체 박스 좌표만 채움.
   - `REREGISTER_TOPN=10`을 영구 설정(상위 10개/modality만 sweep). 스크리닝/랭킹/worklist는 전체 유지.

6. **오피스 첫 worklist 실행 결과 해석**
   - 결과: `worklist FRESH_SNAPSHOT=1 NEW_REGION=4 NO_DATA=185 OK=3 join=8/193` / `om[screened 89, strong 42, confirmed 0, w_sugg 1]` / `sem[screened 104, strong 84, confirmed 0, w_sugg 1]` / `distinct_floor=0.7`.
   - **스크리닝(Phase 1)은 정상·완전**: 193개 전부 평가, SEM 84/104(81%)·OM 42/89(47%)가 STRONG(in_topk=False ≥ 50% = 비변별 key). 이게 실제 재등록 후보 명단(`reregister_report.txt`).
   - **worklist(Phase 3)는 데이터 부족**: `join=8/193`. 1+4+3=8 = matched와 정확히 일치. 185 NO_DATA = consensus summary.json에 rank-1 데이터가 없는 recipe(`rcp_rank1 is None → NO_DATA`). 근본 원인 = **앞 단계 consensus eval이 8개 recipe만 평가**(골든 S-희박 + 프로덕션이 msr 다운로드 끊음 + HISTORY_ROOT 미적재 추정). report 버그 아님.
   - `confirmed=0`은 알려진 이슈: `S_FLOOR=0.6` ≫ 실측 점수 분포(~0.2~0.3) → high-S premise 미충족(Phase 2 threshold miscalib) + 골든 E 희박.

---

## 2. 수정 내용

| 파일 | git 추적 | 변경 |
|---|---|---|
| `poc/workflow_2/golden_eval_config.py` | ✗ (gitignore) | `REREGISTER_BOX_SUGGEST = 1` + `REREGISTER_TOPN = 10` 추가 (실편집 파일) |
| `poc/workflow_2/golden_eval_config.example.py` | ✓ | `REREGISTER_TOPN` 기본값 `0 → 10` + 주석 보강 |

- **커밋/푸시**: `3e59089` `chore(reregister): default REREGISTER_TOPN=10 to cap heavy box-suggestion sweep` → `db7391e..3e59089 main -> main` (pathspec로 example.py 1파일만).
- **검증 중 버그 포착**: loader가 `REREGISTER_BOX_SUGGEST, REREGISTER_TOPN`을 **한 import 문으로 쌍 import**(`golden_eval_config_loader.py:34`). config에 TOPN만 넣으면 그 import가 통째로 ImportError → 둘 다 기본값(TOPN=0)으로 폴백 → 설정이 안 먹음. 짝 상수(`REREGISTER_BOX_SUGGEST`)를 같이 넣어 해결. (검증 안 했으면 조용히 미적용으로 넘어갈 뻔.)

---

## 3. 다음 단계

### A. workflow_2 재등록/consensus (현 상태에서 남은 것 — 보류 가능)
지금 당장 안 해도 되지만, worklist를 193개로 채우려면 필요:

1. **HISTORY_ROOT 적재** (가장 큰 레버): `office_success_downloader`로 class·recipe·modality별 최근 S 8~10장 rolling 적재 → consensus eval `n_recipes` ↑ → join ↑. **현재 join=8/193의 근본 해결책.**
2. **consensus summary.json 커버리지 확인**: `n_recipes` 점검 1줄 명령으로 8 근처인지 확정(데이터 게이트 검증).
3. **Phase 2 `REREGISTER_S_FLOOR` 보정**: 실측 분포(~0.2~0.3)에 맞춰 floor를 내려 `confirmed`가 살아나는지(또는 E 데이터 부족이 진짜 원인인지) 분리.
4. (이미 충분) **스크리닝 결과로 즉시 재등록 명단 활용 가능**: SEM STRONG 84 + OM STRONG 42 = 비변별 key. FRESH vs NEW_REGION 구분만 consensus 필요.

### B. workflow_3 GUI 자동화로 전환 (이번 세션 이후 주 작업) ← **방향 확정**
사용자 인터뷰로 다음과 같이 좁힘:

- **타깃**: **zoom ladder / PM dropdown 실액션 calibration** — `monitor/cycle.py`의 `_run_zoom_ladder` / `_run_pm_dropdown_arms` + `sem_monitor/pm_dropdown.py`. 라이브 SEM box에서 배율 변경(wheel 또는 PM 버튼→dropdown)을 실제로 구동.
- **실행 환경**: **Mac 작성 → push → office pull**. Mac에서는 RCS를 못 보므로 blind 작성, 오피스에서 실행하고 콘솔/디버그 스크린샷(`debug_images/`)으로 피드백 받는 워크플로우.
- **액션 게이트**: **실클릭 활성화**(`SAFE_MODE=0` + 해당 DRY_RUN=0) — DRY-RUN 검증을 넘어 실제 wheel/PM-dropdown 클릭까지. 장비에 실제 영향 → 오피스에서 단계적 검증 필요.

**착수 시 핵심 점검 포인트** (CLAUDE.md / 메모리 기반):
- `ALIGN_FAIL_ZOOM_METHOD` 기본 = `pm_dropdown`(이 tool은 wheel이 배율 변경 안 되고 recenter로 오역됨). `wheel`/`auto`는 다른 tool용.
- PM 버튼 위치 = 2-stage VLM locator(`vlm/ui_venus_mai_locator.analyze_window_target`, ui-venus coarse→mai-ui refined), 첫 open 후 캐시. geometric fallback은 `ALIGN_FAIL_PM_BTN_GAP_RATIO`.
- dropdown 좌표계 자동 감지(crop-pixel/0-1/0-1000) + 매 re-open마다 option row 재read(stale 좌표 misclick 방지).
- 검색 순서 = verdict 따름: `ambiguous`→IN first, `not_visible`→OUT first.
- 커서 모션: RCS는 teleport 아닌 motion 전달 → `mouse_utils`의 glide+jiggle 필수(`ALIGN_FAIL_CURSOR_*`).
- 검증 산출물: `<tag>_pm_dropdown_open{N}.jpg`(PM box/버튼/dropdown crop 오버레이), `zoom_ladder.json`(method + pm_dropdown 섹션).
- 실클릭 게이트: `SAFE_MODE=0` 아니면 `[DRY-RUN]`(단 캡처는 실행되어 경로 확인 가능).

---

## 4. 메모리 업데이트

- **추가**: `project_first_office_reregister_worklist_run.md` — 오피스 첫 Phase 3 worklist 실행 실측(join=8/193, SEM 84/104 STRONG, confirmed=0)과 그 원인(consensus eval 데이터 게이트). `MEMORY.md`에 인덱스 한 줄 추가.
- **참고용 운영 지식**: loader의 쌍-import 게이트(`REREGISTER_BOX_SUGGEST`+`REREGISTER_TOPN`) — config에 한쪽만 넣으면 무효. 이미 `feedback_gate_fix_can_be_cosmetic_downstream_filter`(downstream filter 추적) 교훈과 동일 계열이라 신규 메모리 대신 이 저널에만 기록.

---

## 5. 후속 작업 — PM dropdown 위치탐지 2-VLM 재구현 (커밋 400d6e8)

위 3.B 전환 결정에 이어, 사용자 인터뷰로 타깃을 **PM dropdown 위치탐지 개선**으로 좁혀 실제 구현까지 진행했다.

### 진단 (사용자 도메인 피드백으로 확정)
- **PM 버튼은 2-VLM coarse→fine(ui-venus→mai-ui)으로 잘 잡힘** — 문제 아님.
- **약한 고리 = 드롭다운(dropbox) 위치**. 기존 코드는 `dropdown_region_below`(pm_dropdown.py)로 버튼 점에 앵커된 **고정 비율 추정**(`left=bx−0.04·fw, right=bx+0.12·fw, down=0.45·fh`)으로 crop → 실제 드롭다운과 안 맞으면 PaddleOCR이 엉뚱한 걸 읽어 `옵션 2개 미만` 중단/오클릭.
- **근본 비대칭**: 버튼은 VLM 적응형, 드롭다운은 사람이 박은 고정값. 같은 화면에서 한쪽만 빗나감. 사용자 제안 = "2-VLM zoom-in이면 다 잡아낸다"를 드롭다운에도 적용.

### 설계 (approach A — 합의)
검증된 2-VLM을 드롭다운에도 적용 + 행 클릭은 mai-ui:
```
PM 버튼 클릭(기존) → ① 영역=coarse(ui-venus) bbox→crop (고정비율은 폴백, 캐시)
  → ② 값 열거=PaddleOCR Spotting 1회(값공간만) → ③ CV stepping(choose_step_targets, 기존)
  → ④ 행 클릭=mai-ui로 "값 <text> 행" 직접 그라운딩(실패시 PaddleOCR 중심 폴백)
```
규약 부합: VLM=영역/좌표, OCR=값 확인, CV=OUT/IN 전환(VLM이 전환 결정 안 함).

### 수정 내용 (4 파일, 커밋 `400d6e8`)
- `poc/workflow_3/sem_monitor/pm_dropdown.py` — 순수함수 3개 추가:
  - `crop_region_from_bbox(coarse_bbox, frame_wh, pad_x/y_ratio)` — VLM bbox→패딩·clamp crop
  - `nearest_option(options, value)` — 값 최근접 행 매칭(cycle.py의 `_locate` 추출)
  - `row_target_description(value, text)` — mai-ui 행 설명 문자열
  - `__all__` 갱신
- `poc/workflow_3/sem_monitor/test_pm_dropdown.py` — **신규 TDD 테스트 4개**(crop 패딩/clamp, bad-input None, 설명 임베드, 최근접 매칭). 4/4 통과.
- `poc/workflow_3/vlm/ui_venus_mai_locator.py` — `TargetResult.bbox` 필드 추가 + success 시 coarse bbox 채움(영역 재사용용). sem_box_detect 11/11 무회귀.
- `poc/workflow_3/monitor/cycle.py:_run_pm_dropdown_arms` — 파이프라인 교체:
  - `_locate_dropdown_region(image, btn)` — coarse VLM bbox→crop(캐시), 실패시 `dropdown_region_below` 폴백
  - `_locate_row_point(image, value, text)` — mai-ui 행 그라운딩, 실패시 None
  - `_open_dropdown`이 full shot도 반환(mai-ui 입력용), 루프는 **매 목표 mai-ui 재탐색**(stale 좌표 오클릭 구조적 제거) + PaddleOCR 폴백
  - 3중 폴백: 영역(VLM→기하), 행(mai-ui→PaddleOCR), 그래도 실패시 해당 목표만 skip

### 검증
- **Mac 검증됨**: pm_dropdown TDD 4/4, sem_box_detect 11/11(무회귀), 전 모듈 compile+import, `TargetResult.bbox` 동작.
- **오피스 게이트(blind)**: 실제 VLM 영역/행 그라운딩 정확도. 새 디버그 아티팩트로 확인:
  - `<tag>_pmdd_*`(영역 coarse→fine), `<tag>_pmrow_<값>_*`(행 그라운딩), `<tag>_pm_dropdown_open{N}.jpg`(노랑=VLM 영역), `zoom_ladder.json`의 `selections[].locator`(mai-ui/paddle 어느 경로로 클릭했는지).

### 다음 단계 (PM dropdown)
- [ ] 오피스 pull → `SAFE_MODE=0` + 불확실 알람(ambiguous/not_visible)에서 PM dropdown 경로 실행.
- [ ] 위 아티팩트로 영역/행 그라운딩 정확도 확인. mai-ui 실패가 잦으면 `row_target_description` 문구 보정, 영역이 빗나가면 `crop_region_from_bbox` 패딩(0.4/0.3) 조정.
- [ ] `locator=paddle` 비율이 높으면 mai-ui 그라운딩 신뢰도 점검(폴백 의존 = mai-ui 미작동 신호).
