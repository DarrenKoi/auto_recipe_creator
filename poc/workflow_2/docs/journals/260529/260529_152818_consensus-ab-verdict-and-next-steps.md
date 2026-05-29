# 260529 — consensus A/B 결과 확정 + 다음 단계 (재개용 핸드오프)

> 세션: 2026-05-29 (오후 늦게, 퇴근 전 핸드오프)
> 주제: consensus A/B(test4) 최종 수치로 "재등록 + reranker" 확정, co-registration 배제. 다음 재개 시 할 일 정리.
> 관련: [오후 진단/리뷰 저널](260529_145625_office-results-diagnosis-and-review-fixes.md), [지표 학습 문서](../../study/cv/align_evaluation_metrics_intro.md), [matcher 신뢰도 저널](260529_130919_matcher-reliability-and-diagnostics.md), [golden dataset plan](../../align_success_dataset_plan.md)

---

## 1. 진행 사항 (이번 구간)

코드 리뷰 수정 반영본으로 오피스에서 `align_similarity.py` 를 돌려 **consensus A/B(재등록 검증)** 의 신뢰 가능한 수치를 받았고, rank-1·선명도 컬럼까지 받아 **다음 방향을 데이터로 확정**했다.

### test4 (CONSENSUS A/B, recipes=15, S(LOO)=78) — `console_results/260529_test4.txt`
- **recall (in_topk):** rcp=0.436 → consensus=0.718, **lift +0.282**
- **precision (rank1):** rcp=0.269 → consensus=0.538, **rank1_lift +0.269**, **topk_not_rank1=0.179**
- **선명도(blur) 비율:** vs S개별 edge=0.979 lap=0.959 | vs rcp edge=0.826 lap=1.141
- (참고) cons free_best chamfer S=0.9348 E=0.934 — 이 도메인선 변별 신호 아님(E 에도 key 존재 가능)

### 확정된 판정 (Codex 합의 규칙 적용)
- **재등록 = 확정.** recall +0.282 에 더해 rank-1 도 0.27→0.54(2배). rcp staleness 가 실재했고 S-consensus 가 새 rcp 로 명확히 우수.
- **co-registration = 배제.** 선명도 비율 모두 임계 통과(edge≥0.70, lap≥0.50; vs S 0.979/0.959). median blur 없음.
- **MI/contour reranker = 필요.** topk_not_rank1=0.179(≥0.15) & rank1(0.538)<0.8×in_topk(0.574). 정답이 후보엔 있는데 1등이 아닌 18% 를 리랭커가 회복.

### green mark 정확도 로드맵 (rank-1 기준)
| 단계 | rank-1 |
|---|---|
| 현재(rcp) | 27% |
| + 재등록(consensus→새 rcp) | 54% (검증됨) |
| + MI/contour reranker | ~72% (topk_not_rank1 회복 → in_topk 천장) |
| 잔여 28% | proposer recall 한계 → 2차 proposer / live-search / golden 확장 |

---

## 2. 수정 내용 (이번 구간 커밋, main 직접 — 모두 push 완료)

- `5fb1876` — `_consensus_template_ab` 에 **rank-1 정밀도 + 선명도 비율** 컬럼 추가, chamfer-magnitude 가드 강등(도메인 오해 소지). `_lap_var` 헬퍼. 콘솔에 recall/precision/선명도 + 판정 3규칙 출력. `console_results/260529_test4.txt`.
- `f20d776` — 지표 학습 문서 HTML 렌더본.
- (앞선 구간) `1991071` 코드 리뷰 14/15 수정, `8ed78c8` 저널.

검증(Mac): self-test 통과 + 신규 컬럼 정상. working tree clean.

---

## 3. 다음 단계 (재개 시 — 우선순위 순)

### (A) MI reranker — "진단 검증 → 생산 승격" (다음 1순위)
재등록 때와 똑같이 **A/B 하네스에서 먼저 검증**한 뒤 production 으로 옮긴다.
1. **`align_similarity.py` A/B 에 MI-rerank 측정 컬럼 추가** (코드 작성):
   - held-out S 의 consensus top-K chamfer 후보 각각의 crop 을 `_matched_crop(frame, cand.xy, tw, th, cand.scale)` 로 추출 → `_mi(tpl.raw_image, crop)` 로 재정렬(**chamfer 는 멤버십 게이트, MI 는 순서만**; 하드 reorder, 필요시 alpha 블렌딩은 후순위).
   - 새 컬럼: `cons_rank1_rate_reranked`, `rerank_rank1_lift(=reranked − chamfer rank1)`.
   - **기대:** rank1 0.538 → in_topk 0.718 쪽으로 상승하면 reranker 검증 완료.
   - ⚠️ **Codex 설계 verdict 가 아직 안 들어옴**(reranker 설계 consult 가 백그라운드 중단). 재개 시 (a) Codex 응답 확인 후 반영하거나 (b) 위 설계로 진행 — 핵심 질문: MI 단독 vs MI+contour 두 번째 컬럼, 후보 crop 의 scale 불일치 시 ±1 scale 국소 refine 여부, circularity 점검(consensus 로 만든 후보를 같은 consensus 로 MI 재랭크 — 위치 비교라 비순환으로 판단했으나 재확인).
2. 검증되면 **production 승격**: `align_point_correction.py` 가 쓰는 `compute_align_key_score(...).candidates`(Slice-1 에서 이미 반환) 를 MI 로 reorder. 함수 내부 vs 호출 후 reorder 위치는 Codex E 질문 참고.

### (B) rcp 재등록 도구 — golden 수집과 함께 (다음 주, 2026-06-01 주)
- consensus 는 이미 `<out_dir>/consensus/<recipe>__<mod>.png` 에 저장됨 → **그걸 그대로 새 rcp 로 사용**(별도 재계산 금지, divergence 방지).
- 현재는 S≥4 인 15 recipe 만 consensus 가능. **golden(항상 성공) 데이터셋**으로 판단불가 45개(대부분 S<3) 를 채워 커버리지 확대 + staleness 임계 실측 calibration.
- 재등록 = `align_images/.../align_img_from_rcp/` 의 reference 를 consensus 로 교체하는 절차(설계 필요). align_point_correction 의 rcp 로딩 경로와 연결.

### (C) 잔여 / 후속
- **2차 proposer**(잔여 28% miss): contour/AKAZE/MI-coarse 후보 생성기 — reranker 검증 후, recall 천장(0.718)을 더 올릴 때.
- **crosshair v2 production swap**: 오피스에서 `crosshair_detect.probe()` montage 로 E false-positive(71/198) 가 진짜인지 눈으로 확인 후.
- **코드 리뷰 #10 보류분**: 공유 `frame_dt` threading(`_gt_in_topk`/`_truth_forced`/race) — `align_key_matcher` API 변경 필요, 별도 리팩터.

### 재개 시 오피스 실행 (현재 코드 기준)
```bash
uv run python poc/workflow_2/align_similarity.py   # (A) MI-rerank 컬럼 추가 후 재실행 → rerank_rank1_lift 확인
```

---

## 4. 메모리 업데이트
- `project_matcher_flat_chamfer_distinctiveness.md` 갱신 완료(test4: recall lift +0.282, rank1 0.27→0.54, 가드 재해석, 재등록+reranker 확정). MEMORY.md 인덱스 기존 줄 유지. 추가 변경 없음.
