# 260602 — test2: contour reranker 실패, reranker 레버 사망 → VLM-region 전환 + 코드 정리(대기)

> 세션: 2026-06-02 (오전, 이어서)
> 주제: test2 결과(contour rerank 실패) 판독 → reranker 레버 사망 확정 → 분석 리포트 작성 →
>        escalation 방향 VLM-region+CV 채택 → 죽은 reranker 코드 전체 정리 결정(실행 전).
> 선행 저널: [contour/compare 구현](260602_091614_contour-reranker-and-success-compare-built.md),
>           [test1·MI 폐기](260602_075313_mi-reranker-ruled-out-contour-next.md)
> 핵심 산출물: **분석 리포트** `docs/study/reranker_ab_failure_analysis.md` (근거·실패원인 종합)

---

## 1. 진행 사항

### (A) test2 결과 판독 — contour reranker 실패 (`console_results/260602_test2.txt`)
- contour rerank: consensus `rerank_lift = −0.167`, gt-topk `chamfer rank1 0.341 → contour 0.214`.
  MI(−0.013)보다 **훨씬 나쁨** — baseline 아래로 떨어뜨림(정답과 역상관).
- 나머지 수치는 test1 과 동일(test2 = test1 + contour 컬럼): 재등록 recall 0.436→0.718,
  rank1 0.269→0.538, 선명도 vs S edge 0.979/lap 0.959, staleness stale=17/ok=10/판단불가=44(scored=71).
- **판정: reranker 레버 사망.** MI·contour 두 계열 독립 실패 → chamfer 후보가 패치 단위로 본질 모호.

### (B) 분석 리포트 작성 (사용자 요청 "근거+실패원인 리포트")
- `docs/study/reranker_ab_failure_analysis.md` 신설(한국어). 배경(정밀도 갭)·가설(MI/contour)·
  실험설계(A/B 게이트 ≥+0.10)·결과·**근본원인**(MI=프레임축만/위치축 무력, contour=Otsu contrast
  불안정+Hu 전역·불변)·결론·재현 부록 종합. 음성 결과의 단일 기록처.

### (C) escalation 방향 결정 — VLM-region + CV (사용자 택)
- CV 단독 reranking 죽음 → VLM 이 msr 에서 align key 후보 영역(coarse box)을 좁히고 CV 가 그 안에서
  최종 좌표. 모호 후보를 *순위* 아닌 *공간 축소*로 해결. 도메인 경계 유지(VLM=영역, CV=좌표).
  대안 proposer 교체(contour-후보/AKAZE/multi-scale)는 후순위.

### (D) 죽은 reranker 코드 전체 정리 — 결정됨(아직 실행 안 함)
- 사용자 승인: **전체 정리**. 제거 대상(test2 로 전부 dead 확정):
  - `align_key_matcher.py`: `MatchPolicy.mi_rerank` 플래그, `rerank_candidates_by_mi`/
    `_mutual_information`/`_match_crop_for_mi`, `compute_align_key_score` 의 `if policy.mi_rerank` 블록,
    `AlignKeyCandidate.mi_score/contour_score/chamfer_rank` 필드(전부 rerank 전용 확인).
  - `align_similarity.py`: `_contour_sim`/`_hu_log`, `_gt_in_topk`/gt_topk/_consensus_template_ab 의
    MI·contour rerank 컬럼, 콘솔 줄, self-test rerank assertion. **`_mi` 는 유지**(mi_free/staleness 핵심).
  - `test_align_key_mi_rerank.py` 삭제. `_TEMPLATE_align_similarity.txt` rerank 줄 삭제.
  - 가드: `test_align_key_match.py`(10/10) + align_similarity self-test.
- 사용성 확인: rerank 키 외부 reader 없음(grep). 음성 결과는 (B) 리포트로 보존.

## 2. 수정 내용 (커밋·push 완료)
- `fdbdab6` — `docs/study/reranker_ab_failure_analysis.md` 분석 리포트.
- (앞 세션 구간) `9d40bdb` 저널, `ba939d3` compare 템플릿, `f99adbb` success_vs_fail_compare.py,
  `a0c98e2`/`c1eb570` contour rerank, `e4f8555` test1 판정.
- `console_results/260602_test2.txt` — 사용자 typing(템플릿 채움).
- ⏳ **코드 정리 커밋은 아직** — (D) 실행 시 별도 커밋 예정.

## 3. 다음 단계

**즉시(내 작업, Mac):**
1. **죽은 reranker 코드 전체 정리** 실행(§1-D) → 가드 통과 후 커밋·push.
2. 정리 반영해 메모리(`project_matcher_flat_chamfer_distinctiveness.md` test2 섹션) +
   procedure §6.1 갱신(contour 폐기, reranker 레버 사망, 코드 제거).

**그다음(VLM-region+CV 워크스트림 — office VLM 필요, blind 작성):**
3. 기존 VLM 스캐폴딩 탐색: `vlm_align_key_box.py`, `vlm_sem_monitor_box.py`, procedure §4.3-6
   "VLM 보조 broad spotting". VLM=coarse region → CV=좌표 경계 유지.
4. align-fail 런타임(`align_fail_correct.py`)과 연결 지점 설계(region hint → matcher roi_hint).

**병행(사용자, office):**
5. golden 데이터 적재 → `success_vs_fail_compare.py` 실행(재등록 calibration, 별도 워크스트림).

> ❓ 확인 대기: (D) 코드 정리를 지금 실행할지 사용자 컨펌 중. 승인되면 1→2 진행.

## 4. 메모리 업데이트
- `project_matcher_flat_chamfer_distinctiveness.md` — test2(contour 폐기) 섹션 추가 **예정**(코드 정리와
  함께 반영). 현재까지는 test1 섹션만 반영됨.
- `.remember/now.md` — 후속 갱신 예정.
- MEMORY.md 인덱스 — 변경 없음.
