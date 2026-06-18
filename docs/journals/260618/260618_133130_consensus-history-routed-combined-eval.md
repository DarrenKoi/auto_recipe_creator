# Routed combined eval + consensus history root + OM/SEM 층화

- 날짜: 2026-06-18
- 범위: `poc/workflow_2/` 오프라인 CV 벤치 (production `workflow_3` 무수정 원칙 유지)
- 관련 커밋: `11662cd` → `f3f9ea0` (main 직접 commit/push)

## 배경

workflow_3 check-only(`align_fail_monitor_only_check.py`)로 live SEM box에서 align point를 제대로 못 잡는 문제.
workflow_2 벤치로 돌아가 매칭 정확도 개선 방법을 연구·실험하기로 함. Codex rescue로 시작.

---

## 1. 진행 사항

### (1) Codex rescue — 매칭 개선 연구
- `/codex:rescue`로 Codex에 align-point 매칭 개선 연구 위임 → 두 실험 산출:
  - `edge_ncc` C4 proposer (edge map에 직접 `TM_CCOEFF_NORMED`)
  - `cond_box` ROI 모드
- Explore 에이전트로 신규성 검증: **`edge_ncc`는 진짜 새것**(NCC를 reranker가 아닌 proposer로),
  **`cond_box` ROI는 이미 검증·포팅된 box-crop과 중복**(2026-06-11 `golden_localization_eval_cond`로 검증 완료).

### (2) cond_box 중복 제거, edge_ncc 유지
- `golden_localization_eval_cond.py` + 테스트에서 cond_box ROI 코드 제거 (ROI 테스트 3개 삭제, 62 pass).
- `roi_hint`은 production 엔진의 generic 파라미터라 `golden_localization_eval.py` 통과 코드는 유지.
- ADR `0003` 갱신: edge_ncc만 유지, cond_box는 box-crop 중복이라 제외 명시.

### (3) routed combined eval 드라이버 신규
- 신규 `golden_combined_eval_cond.py`: production 라우팅(consensus 우선·rcp 폴백)을 end-to-end 측정.
  기존 두 드라이버 재사용(LOO consensus = `align_similarity._consensus_template_ab`, bit-drift 0).
- 3축 리포트: **(A)** consensus scaling(`cons_pool_n`별 층화), **(B)** rcp-only arm(=edge_ncc testbed),
  **(C)** routed overall. 양 arm 지표 통일(`in_topk` + `topk_rank==1`).

### (4) 설정 분리 (env 회피 → 별도 config 파일 → 3 드라이버 공용)
- 사용자 요청: env 입력 싫음 → 모듈 상수 토글 → 다시 **별도 파일**로 분리.
- `golden_eval_config.py`(gitignore, 실편집) + `golden_eval_config.example.py`(tracked 템플릿).
- `golden_eval_config_loader.seed_env()`로 상수→env 브리지, **3 드라이버 공용**(combined/consensus/localization).
  상수: `GOLDEN_ROOT`, `HISTORY_ROOT`, `LAB_MODE`, `MIN_S`.

### (5) 한 줄 [DIGEST]
- 사용자가 콘솔 전체 타이핑이 번거롭다 → run 끝에 `[DIGEST]` 한 줄 + `digest.txt` 저장.
  그 한 줄만 복사해 전달하면 3축 다 읽힘.

### (6) consensus 과거 S 풀 = 별도 root, class/recipe 키(eqp 무관)
- 사용자 지적: from_msr에 S를 쌓으면 fail 궤적과 consensus 이력이 구분 안 됨 → **분기 필요**.
- 추가 지적: eqp 무관, **class/recipe로만 매칭**(같은 recipe면 장비 달라도 공유).
- 포맷은 production consensus 캐시와 동일: `<HISTORY_ROOT>/<class>/<recipe>/events/<event_id>/S*.jpeg`
  (+ `.<img>/cond.txt` 숨김 sidecar) — `office_success_downloader` 출력 무변환 적재.
- production `assets.py`는 **건드리지 않음**(처음 per-recipe 하위폴더로 구현했다가 별도 root로 갈아엎으며 revert).
- **history-first + LOO 폴백**: history 풀 ≥ min_s면 disjoint 풀로 consensus(eval=from_msr, 누설0, LOO 불필요),
  없으면 기존 from_msr LOO(byte-identical 보존).

### (7) Step 1 — OM/SEM modality 층화
- 사용자 통찰: OM(저배율·반복패턴)과 SEM(box/직선·contrast 다름)에 동일 단일 CV 정책은 최적 아닐 수 있음.
- 코드 확인: 엔진은 OM/SEM에 **동일 정책**(전역 Canny, 단일 scale, 섞은 Youden), `route_template`은 템플릿만 선택.
- "가정 말고 측정" → combined 드라이버가 3 arm을 OM vs SEM로 쪼개 rank1/topk 출력
  (`by_modality` summary + digest `mod[OM..SEM..]` + report 표).

---

## 2. 수정 내용

### 신규 파일
- `poc/workflow_2/golden_combined_eval_cond.py` — routed pipeline eval (3축 + modality 층화 + [DIGEST]).
- `poc/workflow_2/test_golden_combined_eval_cond.py` — 순수 헬퍼 단위 테스트(17개).
- `poc/workflow_2/golden_eval_config.example.py` — config 템플릿(tracked).
- `poc/workflow_2/golden_eval_config_loader.py` — `seed_env()` 공용 로더.
- `poc/workflow_2/golden_eval_config.py` — 실편집 config(**gitignore**, 추적 안 됨).
- `poc/workflow_2/docs/study/adr/0003-...md`, `0004-routed-combined-eval-and-s-collection.md` — ADR.

### 변경 파일
- `poc/workflow_2/golden_localization_eval_cond.py` + 테스트 — cond_box ROI 제거, `seed_env()` 추가.
- `poc/workflow_2/golden_consensus_eval_cond.py` — `seed_env()`, `HISTORY_ROOT`, `_history_images`,
  `_crop_history_by_mod`(별도 root에서 modality별 crop).
- `poc/workflow_2/align_similarity.py` — `_consensus_template_ab`에 history-first 분기 추가(LOO 경로 byte-identical 보존),
  per_recipe row에 `cons_pool_n`+`mode` 추가.
- `poc/workflow_2/ensemble_lab.py`, `golden_localization_eval.py` — edge_ncc C4(Codex).
- `.gitignore` — `poc/workflow_2/golden_eval_config.py` 무시.
- `CLAUDE.md` — workflow_2 벤치 섹션에 combined 드라이버 + config + history root 기록.

### 검증 (Mac, golden 데이터 없음)
- `py_compile` OK, **17개 신규 테스트 포함 76+ pass**, align engine smoke OK.
- production `assets.py`/`align/__init__.py` clean revert(diff 없음) 확인.
- eqp 무관 resolution 확인(EQP_X/EQP_Y → 같은 `CLS_A/REC_1`).
- 3 드라이버 모두 no-data에서 깨끗이 종료.
- ※ 실제 정확도 수치는 office `ALIGN_GOLDEN_ROOT`/`HISTORY_ROOT` 데이터에서만.

### 커밋
`11662cd`(routed eval+edge_ncc) → `31efed9`(config 상수) → `deab75a`([DIGEST]) →
`65cf0c1`(config 별도 파일) → `4a99a19`(3 드라이버 공용 config) → `15b19b3`(history root) →
`555733e`(docs) → `f3f9ea0`(OM/SEM 층화).

---

## 3. 다음 단계

1. **office 데이터 수집**: `<HISTORY_ROOT>/<class>/<recipe>/events/`에 class·recipe·modality별 최근 S 8~10장
   rolling 적재(S only). `<GOLDEN_ROOT>`는 기존 align_images.
2. **office 실행**: `golden_eval_config.py`에 두 경로 채우고 `uv run python poc/workflow_2/golden_combined_eval_cond.py`
   → `[DIGEST]` 한 줄 회신.
3. **판정 1 (Step 1)**: digest의 `mod[OM.. SEM..]`에서 OM rank1 ≪ SEM이고 실패유형이 다르면
   → Step 2(modality별 Canny/Youden/proposer 레버) ensemble_lab A/B. 비슷하면 split 안 함.
4. **판정 2 (scaling)**: `cons r1/topk`이 n_S bin 따라 단조 증가하면 "consensus 많을수록 좋음" 확인.
5. **판정 3 (edge_ncc)**: `LAB_MODE="edge_ncc"`로 재실행해 rcp-only rank1 상승 + 회귀 없으면 production 포팅 후보.
6. **office_success_downloader**: history root 적재 자동화(현재 수동 가정) — 활성화 게이트.

---

## 4. 메모리 업데이트

- `MEMORY.md` 인덱스 갱신 + `project_routed_combined_eval_and_s_collection.md` 갱신(routed eval, 공용 config,
  history root eqp무관, history-first+LOO, OM/SEM 층화).
- `CLAUDE.md` workflow_2 벤치 섹션 갱신(이번 세션 핵심 아키텍처 변경 반영).
