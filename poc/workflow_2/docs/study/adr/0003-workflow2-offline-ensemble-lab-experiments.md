---
status: accepted
---

# workflow_2 offline bench 에서만 align-point matching 개선안을 실험한다

## 결정

`workflow_3/align/matching/engine.py` 는 건드리지 않고, `poc/workflow_2/ensemble_lab.py` 에
opt-in 실험 경로를 둔다. 검증된 변화만 나중에 production engine 으로 포팅한다.

이번에 남기는 실험 스위치는 `edge_ncc` C4 proposer 하나다.

- `ALIGN_ENSEMBLE_LAB_MODE=edge_ncc`: `golden_localization_eval_cond.py` 가 production ensemble 대신
  workflow_2 lab matcher 를 호출한다. 기본 production 경로(`ALIGN_USE_ENSEMBLE`)는 그대로다.
- `ALIGN_LAB_ENSEMBLE_CHANNELS=canny,scharr,orient,edge_ncc`: lab proposer 채널을 명시한다.
  `edge_ncc` 는 Canny edge map 간 `TM_CCOEFF_NORMED` peak 를 뽑는 opt-in C4 채널이다.

기존 metric 정의(`rank1_hit_rate`, `gt_in_topk_rate`, `topk_not_rank1_rate`, binned report 등)는
바꾸지 않았다. 대신 row/cell payload 에 confidence 보조 필드만 추가했다:
`score_gap`, `second_ratio`, `lab_selection_gap`, `lab_selection_second_ratio`, `lab_channels`.

## 제외: cond-box ROI (이미 production 에서 검증·포팅됨)

초기 실험에는 `ALIGN_LAB_ROI_MODE=cond_box` (cond box 주변으로 search ROI 축소)도 있었으나,
이는 신규 가설이 아니라 **이미 production 에 들어간 box-crop 의 재실험**이라 제외했다.

- `align/cond_template.py` 의 `cond_template_crop()` + `cond_align_offset()` 가 rcp 템플릿을 cond box
  안쪽으로 crop 하고 align offset 을 분리해 `ALIGN_FAIL_COND_BOX_CROP` 로 이미 동작한다.
- 이 가설은 2026-06-11 에 같은 `golden_localization_eval_cond.py` 로 검증되어
  (box template 이 center-area crop 대비 모든 displacement bin 에서 localization 상승) production 포팅까지 끝났다.

따라서 ROI 재실험은 결론이 이미 난 항목의 중복이라 코드에서 정리했다. cond box 정보는 frame ROI 축소가
아니라 기존 template crop/offset 경로로 계속 활용한다.

## 연구 요약

현재 C1/C2/C3 는 모두 edge/chamfer 계열이다. Canny, Scharr, orientation-binned 로 edge 추출은
다르지만, 결국 "template edge 가 frame edge 근처에 있는가"라는 같은 score surface 를 본다.
따라서 flat chamfer surface 와 repeating SEM pattern 에서 구조적 diversity 한계가 있다.

기존 문서/코드 기준 주요 숫자는 다음과 같다.

- 단일 C1 proposer recall@8: 0.557.
- C1/C2/C3 RRF ensemble recall@8: 0.698.
- ensemble NCC selection 의 Youden threshold: match 0.6053, adjust 0.4727.
- 잔여 miss 중 structural far/veryfar 비중이 커서 absolute score threshold 만으로는 해결되지 않는다.

이번 C4 는 "새로운 edge 추출"이 아니라 edge-only NCC 이므로 완전히 독립적인 신호는 아니다. 다만
Chamfer 의 distance-transform 완화 대신 normalized edge-layout correlation 을 보므로, C1/C2/C3 와
다른 peak ordering 을 줄 수 있는 낮은 리스크 채널이다. deep feature, SSIM, MI, learned reranker 는
이번 범위에서 제외했다. 기존 MI/contour reranker 실패 기록상, 후보에 truth 가 없는 문제를 reranker 로
고치는 것은 우선순위가 낮다.

## 검증 결과

Mac dev checkout 에는 golden 데이터가 없었다. 따라서 신규 C4 의 accuracy 숫자는 아직 없다.
실제 수치는 office `ALIGN_GOLDEN_ROOT` 또는 `poc/workflow_3/align_images_golden` 데이터에서 채워야 한다.

로컬에서 확인한 것은 import/구문/합성 회귀다.

```text
UV_CACHE_DIR=.uv-cache uv run python -m py_compile \
  poc/workflow_2/ensemble_lab.py \
  poc/workflow_2/golden_localization_eval.py \
  poc/workflow_2/golden_localization_eval_cond.py

UV_CACHE_DIR=.uv-cache uv run pytest \
  poc/workflow_2/test_ensemble_lab.py \
  poc/workflow_2/test_golden_localization_eval_cond.py -q
# 62 passed

ALIGN_ENSEMBLE_LAB_MODE=edge_ncc \
  UV_CACHE_DIR=.uv-cache uv run python poc/workflow_2/golden_localization_eval_cond.py
# [WARNING] golden 데이터를 찾지 못했습니다 ... cond 판은 self-test 없음.
```

## Office 실행 권장

먼저 기존 production ensemble 과 lab parity 경로를 같은 데이터에서 비교한다.

```text
uv run python poc/workflow_2/golden_localization_eval_cond.py

ALIGN_ENSEMBLE_LAB_MODE=1 \
ALIGN_LAB_ENSEMBLE_CHANNELS=canny,scharr,orient \
uv run python poc/workflow_2/golden_localization_eval_cond.py
```

그 다음 C4 를 추가해서 본다.

```text
ALIGN_ENSEMBLE_LAB_MODE=edge_ncc \
uv run python poc/workflow_2/golden_localization_eval_cond.py
```

판정은 `box__inpaint` 기준으로 한다. 최소 확인 항목은 `gt_in_topk_rate`, `rank1_hit_rate`,
`topk_not_rank1_rate`, displacement bin 별 near/mid/far/veryfar 변화, 그리고
`lab_selection_gap` 이 miss 를 잘 분리하는지다.

## 포팅 권고

아직 production 포팅하지 않는다.

포팅 조건은 다음 중 하나를 만족할 때다.

- `edge_ncc` 포함 lab 경로가 `gt_in_topk_rate` 를 올리고, rank1 회귀가 없다.
- `lab_selection_gap` 또는 `lab_selection_second_ratio` 가 absolute score 보다 더 좋은 ambiguity
  routing signal 로 확인된다.

숫자가 확인되면 `workflow_3` 로 옮길 최소 단위는 C4 전체가 아니라 먼저 confidence routing 이다.
자동 reposition 을 늘리는 방향보다, flat surface 에서 engineer review / VLM-region / fallback 으로
분기하는 쪽이 안전하다.
