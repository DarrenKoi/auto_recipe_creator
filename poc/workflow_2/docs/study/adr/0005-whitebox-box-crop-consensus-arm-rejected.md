---
status: rejected
---

# whitebox box-crop in consensus arm — 기각 (SEM structural-recall 레버 아님)

## 맥락

consensus arm 의 잔존 recall_miss(특히 SEM ~15%, 전체 ensemble+C4 후에도 생존)를 줄일 레버를
찾던 중, "엔지니어가 zoom-in 된 SEM align-key 안의 특정 위치를 **whitebox**(distinctive anchor,
`cond.box_ltrb`)로 지정한다 → 자기유사(periodic) 구조에서 phase tie 를 깨는 사람-제공 신호인데
consensus arm 은 crosshair-중심 center crop 만 써 그 신호를 버린다"는 가설을 세웠다. 가설대로면
center crop 대신 whitebox box-crop 으로 consensus 템플릿을 만들면 SEM recall 이 오를 것.

box-crop 자체는 rcp localization arm 에서 이미 검증·포팅됐으나(ADR 0003/0004), 그 검증은 OM/SEM
혼합·localization 기준이라 SEM 효과가 평균에 가려졌을 수 있어 **consensus arm 에서 OM/SEM 층화로
재측정**하기로 했다. VLM 무사용(whitebox 는 recipe 이미지 cond 에서 옴, 라이브 프레임엔 안 보임).

## 실험

`CONSENSUS_BOX_CROP` env knob(기본 off, bench 전용). `_consensus_template_ab(box_crop=)` 가
center vs box 두 arm 을 **고정 분모**(no-candidate=miss, `center.n_eval==box.n_eval`)로 A/B.
align_offset(box center→align point) 적용. box pool 도 center 와 동일 co-register, history pool
box 재료(`history_crops_box`)도 빌드(final review I1/I2 수정). hit tolerance 는 양 arm 모두 center
템플릿 short side 로 정규화(`_gt_in_topk(tol_short=center_short)`) — 초기 측정은 arm 별 자기 short 로
정규화해 box(작은 템플릿)에 불리한 confound 가 있었고, 이를 제거한 뒤 재측정.

## 결과 (office golden, tolerance-matched)

```
box-crop A/B (consensus arm, per modality):
om:  center 0.930 -> box 0.888  (delta -0.042)  [n_eval 143/143, hit 133/127, box_no_cand 0]
sem: center 0.848 -> box 0.738  (delta -0.110)  [n_eval 191/191, hit 162/141, box_no_cand 0]
```

검증 재료 유효: 양 arm n_eval 동일, box_no_cand=0(box 템플릿 항상 후보 생성). confound 제거 전
초기 수치는 om -0.126 / sem -0.115 였고, tolerance 수정 후 **OM 이 ~8pp 상승**(box 가 불리한 tolerance
로 채점되던 게 사실이었음을 확인)했으나 **SEM 은 -0.115 → -0.110 으로 거의 불변**.

## 결정 — 기각, production 미포팅

box-crop 은 **양 modality 모두 center 보다 나쁘고, 가설이 도우리라던 SEM 에서 가장 나쁘다**.
신뢰할 수 있는 음성. workflow_3 로 아무것도 포팅하지 않는다.

**기각이 mechanistically consistent**(버그 의심이 아닌 진짜 음성인 근거): whitebox crop 은 *더 작은*
템플릿이다. periodic/self-similar SEM 구조에서 작은 템플릿은 프레임 안에 더 많은 wrong-phase 위치에
맞아(주기가 더 많이 타일링됨) false peak 가 *늘고* recall 이 떨어진다. 인접 wrong-phase 를 억제하려면
주기 한 칸보다 작은 ROI 라야 하는데 그건 매칭 가능한 템플릿보다 작아 — 매칭도 되고 변별도 되는 radius 가
없다. "사람 눈에 distinctive" != "chamfer/NCC matcher 에 distinctive". 이는
[[project_om_sem_mag_key_fills_frame]] 의 구조적 논거와 일치.

## 함의 — 레버 방향 전환

결과는 오히려 **더 많은 주변 context 가 도움**임을 시사한다(작은 crop=나쁨). 즉 SEM recall 레버는
crop-down 이 아니라:

- **lower-mag context / zoom-out** — workflow_3 의 zoom ladder OUT arm 방향과 일치(검증된 음성이
  zoom-out 방향을 간접 지지).
- **re-registration 정량화** — chronic-ambiguous key(`reregister_recommended`, feasibility verdict
  ambiguous) 를 엔지니어에게 "더 변별적인 영역에 align key 재등록" 으로 surface. workflow_3 에 이미 일부
  구현(`feasibility_check` second_xy/reregister_recommended audit line).

`CONSENSUS_BOX_CROP` knob 은 bench A/B 도구로 유지(기본 off). production consensus 경로는 center crop 유지.

## 검증

Mac dev: `uv run pytest poc/workflow_2/test_consensus_box_crop.py -q` → 207 passed.
accuracy 숫자는 office golden 에서만(위). bench main 커밋: box-crop 구현 9c70eae, tolerance 수정 30991e9.
