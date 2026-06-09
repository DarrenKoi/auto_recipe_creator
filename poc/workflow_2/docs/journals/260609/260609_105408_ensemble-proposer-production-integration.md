# Ensemble Proposer A/B 검증 + 프로덕션 통합

> 2026-06-09 10:54. single rcp box 경로의 proposer recall 을 끌어올린 ensemble proposer 를
> 오피스 A/B 로 검증(채택)하고, 그 이득을 실제 매칭 파이프라인에 연결했다.

## 1. 진행 사항

### (A) Ensemble proposer 오피스 A/B 검증 — 채택
- `poc/workflow_2/proposer_recall_ab.py` 실행 결과(`docs/console_results/propeser_recall_ab_260609_test1.txt`, S=756) 수치 확인.
- **recall@8: baseline(C1 Canny-DT) 0.557 → ensemble 0.698 (+0.141)**. @16 +0.133, @24 0.692→0.772 (+0.080).
- sanity: `solo:canny == baseline`(0.557) 정확 일치 → 하네스 정상(회귀 가드 통과).
- `ensemble(0.698) > 최강 solo orient(0.64)` → 3채널 decorrelated(RRF 가 상보적 발견을 합산).
- 채널 귀속: @8 랭킹은 **orient**(C3 directional, 0.64) 최강, @24 풀 멤버십은 **scharr**(C2, 0.729) 최강.
- 판정: **PROPOSER_WALL 깸 → 채택**. 진실이 후보에 없던 비율 44.3% → 22.8%.

### (B) 프로덕션 통합 설계 → 구현 (Subagent-Driven, TDD)
- brainstorming 으로 통합 방식 확정: **별도 함수 분리**(`compute_align_key_score_ensemble` 신규,
  기존 `compute_align_key_score` 무변경) + **공유 `_finalize_match` 추출** + **Subagent-Driven** 실행.
- spec: `docs/specs/2026-06-09-ensemble-proposer-production-integration-design.md`.
- plan: `docs/plans/2026-06-09-ensemble-proposer-production-integration.md` (4 tasks).
- 핵심 설계 인사이트: **recall@N ≠ 최종 정확도**. A/B 가 잰 건 "진실이 후보 집합에 듦"이지
  "RRF-top 이 진실"이 아니다(fused rank-1 미측정). 그래서 proposer 만 바꾸면 최종 픽 개선이 보장되지
  않음 → **pool 전체를 chamfer+ORB verifier 로 rerank** 하는 단계가 0.698 을 최종 픽으로 전환하는
  전환 장치. 이 rerank 가 빠지면 통합이 무의미.
- 파이프라인: `_prepare_match_inputs`(전처리 공유) → ensemble proposer → `_rescore_positions_to_candidates`
  (chamfer rescore) → ORB pool-rerank(top_n, combined=chamfer_w·chamfer + orb_w·orb 의 argmax) →
  공유 `_finalize_match`(distinctiveness·decision·overlay·result).
- 리뷰: Task별 spec 리뷰 + Task3·4 code-quality 리뷰 + 통합 final 리뷰. 발견 이슈 3건 모두 반영(아래).

## 2. 수정 내용

변경 파일: `poc/workflow_2/align_key_matcher.py`, `poc/workflow_2/test_align_key_score_ensemble.py`(신규).

커밋(main 직접):
- `8bfb8f4` refactor: `_prepare_match_inputs` 추출(전처리, behavior-preserving).
- `1cfbedc` refactor: `_finalize_match` + `_no_candidate_result` 추출(best-선택-이후, behavior-preserving).
- `79a7280` feat: `_rescore_positions_to_candidates`(ensemble center 위치 → per-scale chamfer
  score-map 룩업으로 `AlignKeyCandidate` 환원; 맵 밖=0.0; RRF 순서 보존).
- `3d8ed75` feat: `compute_align_key_score_ensemble`(proposer→rescore→ORB pool-rerank→finalize).
  순환 import(`ensemble_proposer`↔`align_key_matcher`)는 파일 하단 import 로 해소(모듈 전역 이름이라
  monkeypatch 도 동작).
- `e1455c9` fix(code-review I4): distinctiveness 를 선택 풀(top_n)로 trim — shadow(~24) 전체가 아니라
  ORB-rerank 가 best 를 고른 `candidates[:top_n]` 과 동일 풀로 마감(거짓 not_distinctive 방지).
- `f14dac1` fix(final review): (Issue 2) 모든 후보 chamfer=0 이면 `no_candidates` 로 통일 —
  `compute_align_key_score` reject_reason 계약과 drop-in 호환. (Issue 1) distinctive 가 chamfer-top
  기준(ORB-flip 시 best_xy 와 불일치 가능)임을 docstring 명시(soft advisory).

검증: 신규 6 테스트(rescore 2 + ensemble 4) 통과, `poc/workflow_2/` 전체 **124 passed**,
`test_align_key_match.py` **10/10**(기존 함수 비트 동일 회귀 가드).

설계 규칙 준수: OpenCV 가 좌표·점수·결정, ensemble 은 proposer(후보 위치)만. VLM 미관여.

## 3. 다음 단계

우선순위 순(미착수):
1. **호출자 전환(opt-in)** — `compare_align_images`·`align_fail_correct`·`align_point_correction`
   (free-localize)를 `compute_align_key_score_ensemble` 로 한 줄 교체. live broad-scan
   (`live_align_search`)은 프레임당 비용(~1s) 때문에 기존 함수 유지. drop-in 호환은 final 리뷰에서
   확인됨(AlignKeyMatchResult 필드 동일).
2. **오피스 e2e 검증** — 신규 함수 적용 시 localization 정확도가 실제로 오르는지 blind write +
   오피스 run + digest. (Mac 합성 검증은 완료, 실데이터 검증 필요.)
3. **reranker** — @8→@24 갭(0.074) 회복(이제 후보에 진실이 더 자주 있어 rerank 레버 재생).
4. **하드 플로어 0.228** — 채널 추가/풀 확대/template bank(3채널 top-24 로도 못 찾는 잔여).
5. **consensus 정리(보류)** — `golden_consensus_eval_cond` 출력을 localization 형식으로 +
   `align_similarity._gt_in_topk` 의 modality race 를 `cond_file.msr_modality` 로 라우팅
   (현재 consensus 수치는 clean localization 과 apples-to-apples 아님).

## 4. 메모리 업데이트

`project_ensemble_proposer_and_consensus_race.md` 를 "프로덕션 통합 완료"로 갱신(별도 작업).
MEMORY.md 포인터는 기존 항목 유지(같은 메모리 파일). 새 컨벤션/아키텍처 변경은 없음 — 기존
align_key_matcher 패턴(proposer/verifier 분리, AlignKeyMatchResult 계약) 내에서의 확장이라
MEMORY.md 본문 추가 변경 없음.
