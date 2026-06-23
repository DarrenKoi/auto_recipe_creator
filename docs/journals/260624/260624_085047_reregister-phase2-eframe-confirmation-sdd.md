# Phase 2 E-frame Confirmation — SDD 실행 세션

- 날짜: 2026-06-24 08:50
- 브랜치: `main` (`dda6693` → `61f5222`, push 완료)
- 작업 방식: Subagent-Driven Development (superpowers) — 태스크별 구현 서브에이전트 + 태스크 리뷰 + 최종 whole-branch 리뷰
- 관련 스레드: [[Phase 1 box-fidelity]] (`docs/journals/260623/260623_155946_reregister-phase1-box-fidelity-debug.md`)의 직접 후속

---

## 1. 진행 사항

세션 시작 시 "다음 할 일"을 핸드오프/플랜 기준으로 확인 → **Phase 2 (E-frame confirmation)** 가 코드 미착수(0/6) 상태임을 확인하고, 사용자가 **SDD 전체 6태스크 실행**을 선택하여 진행.

- **플랜 사전 점검(pre-flight conflict scan)** 수행: 테스트가 실제 값을 단언하는지, "아무것도 단언 안 하는 테스트"·"로직 블록 그대로 중복" 같은 결함-강제가 없는지, tier override 가 upgrade-only 이고 `_evidence_tier` 를 건드리지 않는지 확인 → clean.
- **SDD 진행 원장(ledger)** 갱신: `.superpowers/sdd/progress.md` 에 Phase 2 섹션 추가(base `dda6693`), 태스크별 완료 1줄씩 기록.
- **6개 태스크 순차 실행** (구현=haiku 전사 작업, 통합 태스크 6만 sonnet; 리뷰=sonnet, 최종 리뷰=opus):

| Task | 내용 | 커밋 | 리뷰 |
|------|------|------|------|
| 1 | 설정 노브 4종(E-confirm 토글 + collapse 임계) `golden_eval_config` 브리지 | `af18d7b` | Spec OK / Approved (39 pass) |
| 2 | `_e_confirm` 규칙 + `E_CONFIRMED` 최상위 `TIER_WEIGHT` | `dd56400` | Spec OK / Approved (41 pass) |
| 3 | `_median` + `_s_rep_score` 순수 집계 헬퍼 | `08cbedc` | Spec OK / Approved (43 pass) |
| 4 | `_free_search_best_score` + `_e_rep_score` (E 프레임 proposer, bit-parity) | `223b8ce` | Spec OK / Approved (45 pass) |
| 5 | digest `confirmed N` + report `s_rep->e_rep(n_e)` 컬럼 + note | `d304b4a` | Spec OK / Approved (47 pass) |
| 6 | `_load_e_frames` + `run()` E-confirm 포스트패스 통합 | `61f5222` | Spec OK(정확도 office-gated) / Approved (48 pass) |

- **최종 whole-branch 리뷰(opus): Ready to merge = YES**, Critical/Important 없음.
  - 가장 중요한 정합성 질문 = "Phase 1 이 미리 계산한 S 점수와 포스트패스가 재계산하는 E 점수가 비교 가능한가?" → ensemble-RRF 경로와 C1-chamfer 경로 **양쪽 모두** 추적하여 `cand_scores[0] == max(c.score)` 임을 확인. `_gt_in_topk`(`align_similarity.py:348,361`)와 **bit-parity** 성립 → collapse delta 가 실제 의미를 가짐.
  - upgrade-only 불변식(다운그레이드 없음, `_evidence_tier` 밖에서만 override), 다운스트림(ranking/box-suggest C2/digest count) 무손상, E 데이터 없음은 `n_e=0` 로 관측 가능 — 모두 확인.
- **push 완료**: `git push origin main` → `dda6693..61f5222` (origin/main 동기화 확인). push 범위가 정확히 내 6커밋인지 `origin/main..main` 으로 사전 검증.

---

## 2. 수정 내용

전 기능이 워크플로우_2 오프라인 벤치 드라이버 1개 + 그 테스트 + 설정 2파일에 한정.

**`poc/workflow_2/golden_reregister_report_cond.py`** (핵심):
- `TIER_WEIGHT` 에 `"E_CONFIRMED": 3.0` 최상위 추가.
- 모듈 상수 `E_CONFIRM_ON / S_FLOOR / E_FLOOR / COLLAPSE_MARGIN` (env 읽기).
- `_e_confirm(s_rep, e_rep)` — high-S 전제(`s_rep>=S_FLOOR`) 후 `(s_rep-e_rep>=COLLAPSE_MARGIN) or (e_rep<=E_FLOOR)`; None 인자는 False. 순수 함수.
- `_median(xs)` / `_s_rep_score(frame_results)` — 짝수 길이 중앙 두 값 평균; 프레임별 `cand_scores[0]` 의 median, 빈 프레임 skip.
- `_free_search_best_score(center_tpl, gray)` / `_e_rep_score(center_tpl, e_frames)` — E 프레임 free-search proposer. 예외/후보없음만 None, **낮은 점수는 절대 None/0 으로 버리지 않음**(collapse 증거). `COMPARE_SCALES`(center-crop) 사용, `_FIDELITY_SCALES` 아님. import 확장: `_propose_topk, USE_ENSEMBLE_PROPOSER, TOPK_CANDIDATES`(align_similarity) + `preprocess_for_matching`(engine).
- `_load_e_frames(assets, modality)` — `_tool_label=="E"` 프레임만, raw gray(inpaint 없음 — E 는 crosshair/GT 없음), `_route_modality_for_mod` 로 modality 필터.
- `_recipe_row` 반환 dict 에 `e_confirmed/s_rep/e_rep/n_e` 기본값 + `_frame_results` 캐리.
- `run()` 에 `if E_CONFIRM_ON:` 포스트패스 — rows-build 루프 직후·랭킹/C2 box-suggest 직전. 비-NONE row 마다 s_rep/e_rep 계산 후 `_e_confirm` 이면 tier→`E_CONFIRMED`, `risk_score=_risk_score("E_CONFIRMED", max(0, s_rep-e_rep))`.
- `_format_digest` → modality 별 `confirmed N` 추가(단일 `|` 라인 유지). `_format_report` → `s_rep->e_rep(n_e=K)` 컬럼 + E_CONFIRMED note 라인(모두 ASCII).

**`poc/workflow_2/test_reregister_report.py`**: 태스크별 TDD 테스트 8종 추가(설정 브리지, `_e_confirm` 전 분기, median/ s_rep, free-search/e_rep, digest/report 포맷, no-data 스모크). 최종 48 통과.

**`poc/workflow_2/golden_eval_config.example.py` + `golden_eval_config_loader.py`**: Phase 2 노브 4종 추가 + `seed_env()` 브리지(inner/outer ImportError 폴백 양쪽 포함). gitignored `golden_eval_config.py` 는 미수정(규약 준수).

**문서/원장**: `.superpowers/sdd/progress.md`(gitignored) Phase 2 섹션 — 6태스크 + 최종 리뷰 완료 기록.

---

## 3. 다음 단계

**유일한 남은 게이트 = 오피스 정확도 보정 (Mac 은 오피스 데이터 접근 불가).**

3개 collapse 임계(`S_FLOOR=0.60 / E_FLOOR=0.50 / COLLAPSE_MARGIN=0.15`)는 점수 ~0.6 압축 분포 기준 **출발 추정값(미보정)**. 오피스에서 `git pull` 후:

```
# golden_eval_config.py: GOLDEN_ROOT=<align_images_golden>, REREGISTER_E_CONFIRM=1
# (fast A/B) REREGISTER_MAX_RECIPES=20
uv run python poc/workflow_2/golden_reregister_report_cond.py
```

릴레이할 것: `[INFO] e_confirm on: S_FLOOR=.. E_FLOOR=.. COLLAPSE_MARGIN=..` 라인 + `[DIGEST] ... confirmed C` + `E_CONFIRMED` 샘플 1~2행(`s_rep->e_rep(n_e=K)`, `debug_images/golden_reregister_report_cond/reregister_report.txt`). → 관측된 S/E 점수 분포로 3개 임계를 **env 만으로**(코드 무수정) 튜닝한 뒤에 `E_CONFIRMED` 카운트를 신뢰.

**미차단 보류 항목 (Phase 1)**: box-fidelity 변경의 full uncapped 확인 — `w_sugg=1` 은 20-recipe 캡에서만 검증됨. 여유 시 `REREGISTER_MAX_RECIPES=0 uv run python poc/workflow_2/golden_reregister_report_cond.py` → `[DIGEST]` 릴레이.

**선택(post-merge, 비차단)**: 공용 `_sample_rows()` 픽스처에 Phase 2 필드 추가하여 구조 리포트 테스트가 `s_rep->e_rep` 컬럼도 커버하게(현재는 전용 테스트가 커버). 향후 정리로 `frame_dt + _propose_topk` 2줄을 `_gt_in_topk`/`_free_search_best_score` 공용 헬퍼로 추출하면 bit-parity drift 방지.

리뷰 Minor 4건은 전부 WONTFIX/스타일(특히 `_load_e_frames` 의 한국어 `[WARNING]` — cp949 가 한국어 인코딩하고 파일 내 기존 7개 한국어 경고와 동일 관례라 유지; 진짜 금지 문자는 em-dash. digest/report 출력은 전부 ASCII로 올바름).

---

## 4. 메모리 업데이트

- **`memory/project_reregister_report_phase1.md` 갱신**: description 에 Phase 2 코드 shipped(2026-06-24, 정확도 office-gated/미보정) 명시 + 본문에 Phase 2 전체 섹션 추가(포스트패스 설계, `_e_confirm` 규칙, bit-parity 근거, 4 env 노브, 오피스 보정 게이트).
- **`.remember/remember.md` 핸드오프 작성**: 다음 세션(오피스 추정)이 보정 절차·릴레이 항목·WONTFIX 트리아지·git 상태를 바로 잡도록 정리.
- 루트 `MEMORY.md` 인덱스: 기존 `project_reregister_report_phase1.md` 1줄 포인터가 그대로 유효(파일 내용만 확장) → 추가 변경 없음.
- 새 아키텍처/컨벤션 변화는 없음(기존 드라이버에 upgrade-only 포스트패스 1개 추가). CLAUDE.md 변경 불필요.
