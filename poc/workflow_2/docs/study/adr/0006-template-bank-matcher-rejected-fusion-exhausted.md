---
status: rejected
---

# template-bank matcher (heatmap+RRF) 기각 — SEM 은 fusion 으로 랭킹 불가, matcher-fusion 축 소진

## 맥락

consensus arm 의 잔존 SEM recall_miss(ensemble+C4 후에도 ~15% 생존, ADR 0005 에서 box-crop 으로도
못 풀림)와, Phase 2 가 도달한 결론("rcp align key 가 under-distinctive 해 fail 시 옳은 align point 를
못 집는다")을 배경으로, **단일 median-consensus 템플릿 대신 최근 성공 S crop 을 *개별*(median 으로
안 뭉친) 여러 reference 로 두고 cross-member 합의로 align point 를 집으면** 자기유사(periodic) 구조의
phase tie 를 깰 수 있는지 본다.

median-consensus 의 약점 가설: N 장을 median 으로 합치면 sharp 한 개별 신호가 blur 되고, per-member
top-K 가 참 peak 을 떨어뜨리면 영영 복구 못 한다. 이를 두 arm 으로 공략:
- **heatmap (primary)**: 멤버×scale 의 dense chamfer score-map 을 frame-중심 좌표계에 **SUM** 누적 후
  전역 peak. per-member top-K 가 떨어뜨린 약하지만 일관된 peak 을 합산으로 살린다(soft-voting).
- **RRF (extra)**: 멤버별 proposer top-K → member 내 dedup(1표/멤버) → cross-member 공간클러스터 RRF
  → max-member NCC rerank. heatmap 을 못 이기면 discrete 기교가 값을 못 한다는 뜻.

Codex rescue 리뷰의 핵심 반론 H0(= 멤버들이 *같은* distractor 에 합의해 SUM 이 distractor 를 강화):
kill-test(near_periodic 버킷 + GT-bucket 분류 correct/near_periodic/far_wrong/one_member_only)로 게이트.
bench 전용 bit-parity fork(`template_bank_lab.py`, ensemble_lab 패턴), workflow_3 무수정.

## 실험

`golden_consensus_eval_cond.py` 에 TBANK arm 추가(`TBANK_HEATMAP`/`TBANK_RRF` knob, 기본 off-safe).
crop/GT/history-first·LOO 선택은 `_consensus_template_ab` 와 동일 소스(공정 비교·no-leakage 동일 by
construction). heatmap/rrf winner 를 consensus 와 같은 GT·tolerance 로 in_topk + rank-1 측정, OM/SEM
층화 + min_s bin + bootstrap CI. 좌표 불변식(템플릿-중심 frame px, half-template shift)·proposer
bit-parity(`frame_dt=None if USE_ENSEMBLE_PROPOSER`) 검증.

구현 중·후 코드리뷰(efficiency-focused)에서 정량 왜곡 버그 다수 수정: LOO min_s 가 dominant min_s=3
bin 을 조용히 떨어뜨림(게이트가 cosmetic, downstream `bank_build(min_s)` 가 진짜 필터였음 — 2개 리뷰가
놓침), digest None crash, cons baseline phantom-zero·population mismatch, double co-registration,
LOO O(N^2) 재빌드. 수정 후 재측정.

## 결과 (office golden, coregister ON, consensus cond 67 recipe)

kill-test **PASS** (H0 아님): heatmap near_periodic om=0.014 sem=0.052 (<< 0.30). SUM 이 distractor 를
강화하지 않음 — Codex 가 우려한 실패모드는 안 일어남.

```
            in_topk (recall ceiling)        rank-1 (production 산출물 = 1점 reposition)
            heatmap   rrf     cons           heatmap   rrf
  OM        0.831    0.965   0.927           0.531    0.902
  SEM       0.943    0.916   0.849           0.492    0.524
```
heatmap_in_topk_ci: OM [0.742,0.906], SEM [0.901,0.974] (둘 다 cons point 가 CI 밖 → in_topk 차이 유의).
SEM min_s bin: 4-6=0.933(n75) / 7+=0.948(n116) (3-bin 이 set 에 없음). cons_rank1 은 consensus 드라이버가
미기록.

**in_topk 만 보면 두 bank arm 이 각 modality 에서 consensus 를 이긴다**(SEM heatmap +9.4pp,
OM rrf +3.8pp, CI 분리). 그러나 **rank-1 이 결론을 뒤집는다**: heatmap top-peak 가 참 align point 인
비율이 OM 0.53 / SEM 0.49 — **동전던지기**. in_topk→rank-1 붕괴 SEM −45pp. RRF 도 SEM rank-1 0.524.
참 점이 *후보군엔* 들어오지만(in_topk 0.92~0.94) *1등으로 안 꼽힌다*(rank-1 ~0.5) — periodic distractor
가 참 key 와 **구조적으로 동등한** 템플릿 매치라 합산/투표가 tie 를 못 깬다.

## 결정

**§14 "neither" — 어느 arm 도 workflow_3 에 포팅하지 않는다.** production 은 1점 reposition 이라 rank-1
이 산출물인데 primary(heatmap) rank-1 이 ~0.5 로 배포 불가(consensus 대비가 아니라 intrinsic). RRF 는
OM rank-1 0.902 로 좋지만 OM 은 bottleneck 이 아니고(consensus 가 이미 처리), SEM rank-1 0.524 로 무용.

**핵심 발견: SEM 은 recall 문제가 아니라 *랭킹/distinctiveness* 문제이며 member-fusion 으로 못 푼다.**
median-consensus, soft-voting heatmap, one-vote RRF — 세 fusion 이 모두 참 점을 후보로 복구(in_topk
~0.9)하지만 1등 랭킹은 ~0.5 로 같은 벽. fusion 은 동등하게-좋은 매치를 평균낼 뿐 key 자기유사성에 박힌
tie 를 못 깬다. **matcher-fusion 축은 경험적으로 소진**(flat-chamfer ADR·Phase 2 rcp-under-distinctive
와 같은 벽). 레버는 matcher 가 아니라 **align key 의 distinctiveness** → re-registration 우선순위
리포트(Phase 1)가 가리키는 방향이거나 non-template SEM 신호(VLM-ROI 등).

방법론 교훈: bench A/B 는 **rank-1 로 비교**할 것. in_topk 는 천장이고 rank-1 이 배포 산출물 — high
in_topk / low rank-1 은 recall 착시다.

코드는 정상(16/16 test, main 2d8c35f..ac69876), `TBANK_HEATMAP=0` kill switch 로 bench arm 으로 보존.
관련: ADR 0003/0004(ensemble/routed), 0005(box-crop 기각).
