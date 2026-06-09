# Ensemble Proposer 프로덕션 통합 설계

> 2026-06-09. ensemble proposer(3채널 RRF, recall@8 0.557→0.698 검증)를 생산 매칭 경로에
> 통합한다. 선행 spec: `2026-06-09-ensemble-proposer-design.md`(빌드+A/B). 이 문서는 그 결과를
> 실제 파이프라인에 연결하는 **통합** 설계.

## 목표 (한 문장)

fallback/static-compare 경로가 `compute_align_key_score_ensemble`(신규)를 호출하면, ensemble
proposer의 향상된 후보 recall(0.698)이 chamfer+ORB verifier-rerank를 거쳐 **더 정확한 최종
align point**로 전환된다. 기존 `compute_align_key_score`는 출력 비트 동일하게 보존한다.

## 배경 / 문제

- single rcp box 경로의 proposer recall(gt_in_topk)은 `compute_chamfer_candidates`(C1 Canny-DT
  단일 채널) 기준 **0.557** — 진실이 top-8 후보에 44% 없는 PROPOSER_WALL.
- ensemble proposer(C1 Canny-DT + C2 Scharr + C3 orientation directional chamfer → RRF 융합)는
  오피스 A/B에서 **recall@8 0.698**(+0.141)로 벽을 깸. 하지만 현재 ensemble은 *A/B 평가 스크립트*
  (`proposer_recall_ab.py`)에만 존재하고 생산 경로는 여전히 C1 단일 proposer를 쓴다.
- 생산 관문은 `align_key_matcher.compute_align_key_score` — 6개 호출자(`live_align_search`,
  `align_fail_correct`, `compare_align_images`, `align_point_correction`, `match_recipe_key_on_crop`,
  `vlm_align_key_region`)가 모두 경유. 그 내부 line 632가 `compute_chamfer_candidates`를 호출.

### 왜 단순 함수 교체가 아닌가

1. **점수 의미 불일치.** 다운스트림(distinctiveness 판정, ORB 게이팅, decision)은 후보의
   `chamfer_score`에 의존. ensemble `_Cand`는 RRF 융합 점수(순위 기반)·`template_size` 없음 →
   자료형/의미가 다름. 그대로 꽂으면 distinctiveness·decision이 깨진다.
2. **recall ≠ 최종 정확도.** A/B가 잰 것은 recall@N(=진실이 후보 *집합*에 듦)이지 "RRF-top이
   진실"이 아니다. fused **rank-1** recall은 미측정. proposer만 바꾸고 ORB를 RRF-top 1개에만
   돌리면 최종 픽 개선이 **보장되지 않는다**(rank-1 미측정이라 회귀 위험). 향상된 풀을 verifier로
   **rerank**해야 0.698이 최종 픽으로 전환된다.
3. **지연.** ensemble은 프레임당 ~1s(8-bin×4-scale matchTemplate). `compute_align_key_score`는
   live 루프(`live_align_search.py:230`)에서 매 프레임 호출 → 무조건 켜면 broad-scan이 느려진다.
   측정된 0.698 이득은 **static localization(=fallback/compare 경로)**에서 나온 것이므로, 그
   경로에만 적용하는 것이 정확하고 안전하다.

## 설계 결정 (확정)

- **별도 함수 분리.** `compute_align_key_score`는 무변경(출력 비트 동일), 신규
  `compute_align_key_score_ensemble`를 추가. 호출자가 명시적으로 선택.
- **통합 형태: proposer→rescore→verify-rerank→finalize.** ensemble은 후보 *위치*만 생성, CV가
  점수·좌표·결정 — 설계 규칙("OpenCV가 좌표/점수 결정, VLM은 영역/타당성만") 준수.
- **공유 finalize 헬퍼 추출.** best-선택 이후 로직(distinctiveness/decision/overlay/result)을
  `_finalize_match()`로 추출, 두 함수가 공유. `compute_align_key_score`는 내부 리팩터되지만 동작
  불변(기존 `test_align_key_match` 10/10이 회귀 가드).

## 아키텍처

### 신규: `compute_align_key_score_ensemble`

```
compute_align_key_score_ensemble(template, frame, *,
    frame_nm_per_pixel=None, roi_hint=None, scales=None, policy=DEFAULT_POLICY)
    -> AlignKeyMatchResult
```

기존 `compute_align_key_score`와 **동일 시그니처**(드롭인 교체용). 파이프라인:

1. **전처리** — `_to_grayscale`, scale 해석(override/nm_per_pixel/DEFAULT_SCALES), ROI crop,
   `preprocess_for_matching`로 `frame_dt`. 기존 함수와 동일 로직(전처리도 공유 헬퍼로 추출 가능,
   단 동작 불변 전제).
2. **Proposer(ensemble)** — `compute_ensemble_candidates(template.raw_image, gray_frame,
   scales=scales, top_n=policy.top_n)` → fused 위치 리스트(`_Cand`: center xy, scale).
   후보 없으면 기존 no-candidate 경로(center, score 0, `reject_reason="no_candidates"`)를 공유.
3. **Rescore → MatchCandidate** — fused 위치별로 chamfer_score 부여:
   - scale별로 `_mean_dt_map_at_scale(template.edge_map, frame_dt, scale)`를 1회 계산(unique
     scale 캐시).
   - 각 위치(center xy)를 top-left(`x0=cx-tw//2, y0=cy-th//2`, `tw,th=round(tpl*scale)`)로
     변환해 mean_dt_map에서 룩업 → `chamfer_score = exp(-mean_dt / DT_TAU_PX)`.
   - 경계 밖/맵 무효 영역이면 `chamfer_score=0.0`.
   - 결과: 기존 후보 자료형(`AlignKeyCandidate` 또는 동등; `xy`=center, `chamfer_score`,
     `scale`, `template_size`)으로 환원.
4. **Verifier-rerank (핵심 전환 단계)** — top-k(=`policy.top_n`) 후보 각각에 ORB 실행:
   - `chamfer_score > 0`인 후보만 `_crop_with_padding` + `compute_orb_inlier_ratio`.
   - `combined = policy.chamfer_weight*chamfer + policy.orb_weight*orb`.
   - `best = argmax(combined)`. 이 단계가 향상된 풀에서 진실을 최종 픽으로 끌어올린다.
5. **Finalize(공유)** — `_finalize_match(best, candidates, gray_frame, frame, template, policy,
   roi_origin)` → distinctiveness(chamfer 정렬 top-2 gap), decision, overlay, roi 절대좌표 환산,
   `AlignKeyMatchResult`.

### 리팩터: 공유 `_finalize_match`

`compute_align_key_score`의 line 651~715(distinctiveness/decision/overlay/result)를 추출:

```
_finalize_match(best_cand, all_cands, gray_frame, frame, template, policy, roi_origin,
                *, chamfer_score, orb_ratio) -> AlignKeyMatchResult
```

- `compute_align_key_score`: chamfer 후보 → `best=candidates[0]` → ORB(best) → `_finalize_match`.
- `compute_align_key_score_ensemble`: ensemble → rescore → ORB(pool) → `best=argmax` →
  `_finalize_match`.
- 동작 불변 전제: 추출 후 `compute_align_key_score` 출력은 기존과 비트 동일.

## 데이터 흐름

```
template(AlignKeyTemplate) + frame(ndarray)
  └─ ensemble proposer ─→ fused 위치[center xy, scale]   (recall 0.698)
       └─ rescore ─→ AlignKeyCandidate[xy, chamfer_score, scale, template_size]
            └─ ORB over pool ─→ combined score per cand
                 └─ best=argmax ─→ _finalize_match ─→ AlignKeyMatchResult
                      (best_xy, score, chamfer, orb, decision, distinctive, candidates, overlay)
```

## 호출자 통합 (opt-in, 별도 작업)

이번 spec 범위는 **함수 구현 + 단위테스트**까지. 실제 호출자 전환(`compare_align_images`,
`align_fail_correct`, `align_point_correction` free-localize를 신규 함수로)은 함수 검증 후
별도 단계에서 한 줄 교체. live broad-scan(`live_align_search`)은 기존 함수 유지(지연).

## 에러 처리

- ensemble 후보 0개 → 기존 no-candidate 경로(center, 0점, reject) 공유.
- rescore에서 위치가 맵 밖 → 해당 후보 chamfer_score=0.0(드롭 아님, ORB가 구제 가능하나
  `chamfer_score>0` 게이트에 막혀 ORB 생략 → combined=0).
- scales override 검증/ROI 검증은 기존과 동일 규칙 재사용.

## 테스트 (TDD, Mac 합성 데이터)

1. **rescore 정확성** — 합성 template/frame에서 ensemble 위치의 rescored chamfer_score가
   `_mean_dt_map_at_scale` 직접 룩업과 일치(`np.isclose`).
2. **pool-rerank 선택** — ORB가 RRF-top이 아닌 후보를 우세하게 만들도록 구성 → `best`가
   combined argmax(=ORB 우세 후보)인지 확인.
3. **finalize 추출 회귀** — `compute_align_key_score` 출력(best_xy/score/decision/distinctive)이
   리팩터 전후 동일. 기존 `test_align_key_match.py` 10/10 유지.
4. **no-candidate 경로** — ensemble 0후보 시 center/0/`reject_reason="no_candidates"` 반환.
5. **결과 형태** — `compute_align_key_score_ensemble`가 유효한 `AlignKeyMatchResult`(필드 채움,
   candidates roi 절대좌표) 반환.
6. **scales/ROI override** — 신규 함수도 `scales=()`·잘못된 `roi_hint`에 기존과 동일 ValueError.

오피스 실데이터 end-to-end(신규 함수 적용 시 localization 정확도가 실제로 오르는지)는 함수
검증 후 별도 실행(blind write + 오피스 run + digest).

## 비범위 (YAGNI)

- live broad-scan 통합(지연 — 기존 함수 유지).
- ensemble 결과 캐싱/병렬화(성능 최적화는 별도 레버).
- consensus_eval 출력 정리(직교 작업, 보류).
- 호출자 일괄 전환(함수 검증 후 별도 단계).
