# Template-Bank Matcher — 구현·코드리뷰·오피스 측정·기각 (2026-06-25)

## 1. 진행 사항

이전 세션에서 작성한 template-bank 매처 구현 계획(`poc/workflow_2/docs/plans/2026-06-24-template-bank-matching-plan.md`)을 **subagent-driven-development** 로 끝까지 실행하고, 코드리뷰로 정량 버그를 잡은 뒤, 오피스 골든셋에서 측정하고, 결과를 보고 **기각** 결론까지 냈다.

- **7개 태스크 SDD 실행 (Mac, 합성 테스트):** config knob → `bank_build`(개별 N 템플릿, median 안 함) → `bank_match_heatmap`(soft-voting dense SUM, primary) → `bank_match_rrf`(1표/멤버 공간 RRF + max-member NCC, extra) → `estimate_lattice_period`+`classify_winner`(kill-test 버킷) → 순수 eval 헬퍼(bootstrap CI/버킷/digest) → 드라이버 arm 통합. 태스크마다 implementer→task-review(spec+quality)→fix 루프, 마지막에 opus whole-branch 리뷰.
- **좌표 불변식·bit-parity 검증:** heatmap 의 half-template shift(`oy,ox=th//2,tw//2`), `_peaks_center` divmod 행/열 순서, RRF proposer 의 `frame_dt=None if USE_ENSEMBLE_PROPOSER` 가 `_gt_in_topk` 와 byte-identical 임을 리뷰에서 확인.
- **efficiency-focused 코드리뷰(/code-review):** 8개 finder angle 병렬 → ~50 후보 → 코드 직접 읽어 판정 → 10개 ranked.
- **오피스 측정 2회 릴레이(coregister ON, consensus cond 67 recipe):** kill-test + digest + summary.json(CI/min_s bin/rrf/rank-1).
- **기각 결론 + 문서화:** ADR 0006, spec/plan 배너, 본 저널, 메모리 2건.

## 2. 수정 내용

### 신규 파일
- `poc/workflow_2/template_bank_lab.py` — bench bit-parity fork(`ensemble_lab.py` 패턴). `bank_build`/`BankResult`/`_accumulate_heatmap`/`_peaks_center`/`bank_match_heatmap`/`_dedup_within_member`/`bank_match_rrf`/`estimate_lattice_period`/`classify_winner`. workflow_3 import 만, 역import·수정 없음.
- `poc/workflow_2/test_template_bank_lab.py` — 합성 단위 테스트 16개.
- `poc/workflow_2/docs/study/adr/0006-template-bank-matcher-rejected-fusion-exhausted.md` — 기각 ADR.

### 변경 파일
- `poc/workflow_2/golden_consensus_eval_cond.py` — TBANK arm 통합(+순수 헬퍼 `_bootstrap_ci`/`_aggregate_buckets`/`_format_bank_digest`).
- `poc/workflow_2/golden_eval_config.example.py` + `golden_eval_config_loader.py` — `TBANK_*` knob 5개 브리지.
- spec/plan 상단에 CONCLUDED/REJECTED 배너.

### 코드리뷰에서 잡은 정량 버그(수정 커밋 `b7bc6d4`)
- **#1 (가장 중요):** LOO `bank_build(loo_crops, min_s=CONSENSUS_MIN_S)` 가 2-crop LOO 에서 `[]` 반환 → dominant min_s=3 bin 을 bank arm 에서 조용히 누락(consensus 는 평가). 세션 중 먼저 시도한 게이트 수정(`<2`)은 **cosmetic** 이었음 — 진짜 필터는 downstream `bank_build` 의 min_s. → LOO 만 `min_s=2` 로(consensus `len(others)<2` 패리티). 2개 리뷰(태스크·opus)가 이 줄을 고립해 봐서 놓쳤고, efficiency 리뷰가 두 줄을 함께 읽어 잡음.
- **#2:** `_format_bank_digest` 가 `cons_in_topk=None` 에서 `None:.3f` TypeError → summary.json 쓰기 전에 run 전체 abort. `_f()` None-safe 헬퍼("n/a")로 수정 + Mac 테스트 추가.
- **#3:** cons baseline 이 phantom 0.0 default + bank 와 다른 population. `_cons_pr_by_mod` 를 `{mod:{recipe:rate}}` + `rate is not None` 가드 + bank 평가 레시피와 교집합.

### efficiency 재구조화(커밋 `73bb04e`, `ac69876`)
- **#8:** `frame_dt` 를 프레임당 1회 계산해 두 arm 에 공유(`frame_dt=` kwarg).
- **#9:** `bank_match_rrf` 의 `sorted()` key 전에 `_ncc_max` 캐시.
- **#6/#7 + double-coregister:** `full_bank` 를 레시피/modality 당 1회 빌드(`coregister=False`, 이미 상류 line 421/482 에서 coregister됨 → consensus 패리티 + 이중 coregister 제거) + LOO 는 `full_bank[:i]+full_bank[i+1:]` 슬라이싱(O(N^2)→O(N)).

전 커밋 main 직접 push(`2d8c35f..ac69876`), 병렬 세션 커밋(measurement-abort docs)이 중간에 섞였으나 pathspec 으로 내 파일만 커밋.

## 3. 다음 단계

template-bank 스레드는 **종료**. 결론이 가리키는 다음 방향(사용자 확인됨 — re-registration 권장):

- **레버 = align key distinctiveness, matcher 아님.** SEM 은 recall 문제가 아니라 랭킹 문제(참 점이 후보엔 들어오나 1등 랭킹 ~0.5). median-consensus·heatmap·RRF 세 fusion 이 같은 벽 → matcher-fusion 축 소진.
- **권장: re-registration 우선순위 리포트(Phase 1, `golden_reregister_report_cond.py`) 를 실행 단계로** — chronic-ambiguous SEM key 를 distinctive 영역으로 재등록하도록 엔지니어에 구동.
- 대안(더 큰 스윙): non-template SEM 신호(VLM-ROI / learned feature).

## 4. 메모리 업데이트

2건 신규 + MEMORY.md 인덱스 반영:
- `project_template_bank_concluded_fusion_exhausted.md` — 기각 결론, rank-1 ~0.5 벽, fusion 축 소진, 레버=distinctiveness, "bench A/B 는 rank-1 로 비교" 교훈.
- `feedback_gate_fix_can_be_cosmetic_downstream_filter.md` — 게이트 수정이 downstream 동등 필터에 가려져 cosmetic 일 수 있음; 경계 입력을 consumer 까지 추적할 것(LOO min_s 버그가 2개 리뷰를 통과한 사례).
