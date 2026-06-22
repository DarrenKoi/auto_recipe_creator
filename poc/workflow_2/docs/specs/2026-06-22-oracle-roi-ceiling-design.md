# Oracle-ROI ceiling sweep — 설계

작성: 2026-06-22 · 대상 코드: `poc/workflow_2/align_similarity.py`
(`_gt_in_topk`, `_consensus_template_ab`), `poc/workflow_2/golden_consensus_eval_cond.py`,
`poc/workflow_2/golden_eval_config.py`(+example, loader)
선행: per-modality consensus eval(`20efe49`), consensus proposer = 프로덕션 3채널
ensemble 거울(`c8d5571`, [[project_edge_ncc_consensus_ab_3arm]])

---

## 1. 배경 / 문제

consensus arm 의 SEM `recall_miss ~15%`(rank1 0.71 vs OM 0.91)는 full ensemble+C4 에도
살아남는다. 분해 결과 이건 **mis-rank 가 아니라 proposer recall** 문제다 — 정답(crosshair)
위치가 top-N 후보 pool 에 *아예 없다*. 채널 튜닝(C1/C2/C3/C4)은 천장에 도달했다
([[project_edge_ncc_consensus_ab_3arm]], [[project_matcher_flat_chamfer_distinctiveness]]).

가설: SEM 은 반복 주기 구조(line/space, contact array)가 지배해 점수면이 평평하고,
프레임 *다른 위치*의 distractor peak 들이 진짜 peak 를 top-N 밖으로 밀어낸다. 그렇다면
**탐색 영역을 진짜 위치 주변으로 좁히면**(ROI) distractor 가 pool 에서 빠지고 truth 가
surface 할 것이다. 이게 VLM-ROI prior 의 이론적 근거다(VLM 이 live SEM frame 에서
align-key 영역을 grounding → CV 탐색 좁힘, 2026-05-25 규칙: VLM 은 영역만, CV 가 좌표/점수).

**그러나** 이 가설은 미검증이다. SEM 의 distractor 가 프레임 *다른 곳*에 있으면 ROI 가
듣지만, 주기 구조가 ROI *안에도* 있으면(키가 반복 필드 위에 있어 국소적으로도 모호) ROI 는
무용하고 template-bank 가 필요하다. VLM grounding 구현(오피스 VLM 호출 + 프롬프트 튜닝)은
비용이 큰데, 그 비용을 쓰기 전에 "ROI 가 애초에 옳은 레버인가"를 먼저 알아야 한다.

## 2. 목표 / 비목표

**목표**
- **오피스 1회 실행**으로 답한다(theoretical ceiling / go-no-go): 진짜 crosshair 주변으로
  탐색을 좁히면 SEM `recall_miss` 가 무너지나? = **ROI 가 애초에 옳은 레버인가**.
- 결과가 레버를 판정: tight recall ≫ full → VLM-ROI 가 *지을 가치 있는* 방향 → 다음 단계
  (degraded-oracle precision bar) 착수; flat → template-bank 로 선회.

**비목표 (YAGNI / 명시적 한계)**
- VLM 호출 없음 — 순수 oracle(ground-truth crosshair 사용, eval-only, 배포 불가).
- jitter/offset oracle 없음(degraded-oracle 는 본 작업 범위 밖 — 사용자가 ceiling-first 선택).
  → **본 sweep 은 truth-centered·zero-offset 라 *이상적* ceiling 일 뿐, "VLM grounding 이
  얼마나 정밀해야 하나(precision bar)"는 답하지 못한다.** 실제 VLM 은 center error 가 있어
  radius margin 을 먹고 truth 를 window 밖으로 밀거나 distractor 를 다시 들인다. precision bar
  는 ceiling 이 headroom 을 보일 때만 진행하는 **후속 작업**(center offset / box-size error
  sweep)이다. 본 결과를 그 후속 없이 VLM build 의 precision 근거로 쓰지 말 것.
- 프로덕션(`workflow_3`) 변경 없음 — 본 실행은 *VLM 버전을 지을지 말지*만 결정한다.

## 3. Oracle 의 정의(중요)

"Oracle" = eval 시점에만 가진 ground truth(golden set 라벨 `f["xy"]`)를 써서 *ceiling* 을
재는 측정 장치다. 런타임엔 절대 없다(런타임엔 VLM 이 이 영역을 추정). Oracle-ROI =
**search frame** 좌표계에서 진짜 crosshair 주변에 그린 window. (구분: rcp 흰 box 는
*template* 좌표계의 "무엇을 찾나" 단서로, 이것과 다른 이미지·다른 공간이다
[[project_rcp_white_box_unique_area]].) window 는 항상 truth-centered(offset 없음).

## 4. 설계

### 4.1 Component 1 — 핵심 단위: `_gt_in_topk(..., oracle_roi_radius=None)`

매칭 로직을 건드리는 유일한 곳. template 짧은 변(short) 단위의 optional param 추가:
- `None` → 현재와 **byte-identical**(full-frame). regression test 로 고정.
- 설정 시 → per-modality 루프 *안에서*: `gray` 를 진짜 crosshair 중심 ±`radius*short`
  window 로 crop, crosshair 를 local 좌표로 shift, `frame_dt` 를 local 로 재계산
  (ensemble 경로는 `None`), 그 window 안에서 propose. truth 와 candidate 가 같이
  shift 되므로 거리/recall 의미는 보존 — distractor field 만 줄어든다.

per-modality 루프 안에서 crop 하는 이유: radius 가 template-short 단위라 modality 별로
다르고, oracle 은 항상 단일 modality dict(`{mod: cons_tpl}`)로 호출되기 때문(consensus arm).
multi-modality 호출자(`align_similarity.py:523` 등)는 `oracle_roi_radius=None` 이라 무영향.

helper `_crop_oracle_window(gray, center, half) -> (crop, local_center)`:
프레임 경계로 clamp(edge 근처 window 는 그쪽만 작아지고 truth 는 알려진 local 좌표 유지).

### 4.2 Component 2 — sweep: `_consensus_template_ab(..., oracle_roi_radii=None)`

radius 리스트가 주어지면, 기존 consensus 루프의 `gc = _gt_in_topk(...)` 바로 옆에서 각
`r` 에 대해 `_gt_in_topk(..., oracle_roi_radius=r)` 을 추가 호출, `(modality, radius)` 별로
집계. 기존 cons-vs-rcp 측정은 그대로 — 같은 frame/template 에 측정만 덧붙인다. `None`(기본) = off.

**측정 계약 — 고정 denominator (survivorship 금지).** `_gt_in_topk` 은 후보 0개일 때
`None` 을 반환하고 기존 consensus 루프는 그 점을 `continue` 로 *건너뛴다*(n 미증가). 이걸
sweep 에 그대로 답습하면 tight/edge-clamped ROI 가 후보를 못 내는 점을 *miss 로 세지 않고
빼버려* recall 이 거짓으로 올라간다(분모 축소 = survivorship). → **각 radius 의 분모는
full-frame 에서 유효(=`gc is not None`)했던 *동일* eval set 으로 고정**한다. 그 분모 안에서:
- candidate 있고 truth in_topk → **hit**
- candidate 있고 truth not in_topk → **miss**
- 후보 0개 / crop 이 template 보다 작음 / propose 예외 → **miss**(skip 아님)

분모에서 빠지는 유일한 경우는 그 점이 *full-frame 에서도* 무효(gc None)였을 때뿐(애초에
oracle 분석 대상이 아님 = `n_skipped`). 반환에 카운트를 분리해 곡선이 정직하게 한다:

```
res["oracle_roi"] = {
  "radii": [...],
  "per_modality": {mod: {r: {"recall": h/e, "rank1": r1/e,
                             "n_eval": e, "n_hit": h,
                             "n_no_candidate": nc, "n_skipped": sk}}}
}
```
(`n_eval` = full-frame-valid 고정 분모, `recall = n_hit/n_eval`, `n_no_candidate` ⊆ miss.)

### 4.3 Component 3 — config + digest

`golden_eval_config.py` 에 `ORACLE_ROI_RADII`(기본 `[]` = off) 추가(+example, loader.seed_env).
`golden_consensus_eval_cond` 가 읽어 passthrough + oracle digest table 출력(modality × radius
→ recall_miss, full→tightest delta). `golden_combined` 는 consensus driver 재사용이라 자동 노출.

## 5. 실행 / 산출

오피스 1회 실행 `ORACLE_ROI_RADII=[1.5,2,3,5]`(full-frame baseline 은 같은 고정 분모로 digest
에 함께 출력). SEM 행에 집중 — **결정 규칙은 ROI 레버 유무까지만**(precision bar 아님):
- **tightest recall ≫ full** → ROI 가 옳은 레버 → 다음 단계 **degraded-oracle**(center
  offset / box-size error sweep)로 precision bar 측정 후 VLM-ROI grounding build. (centered
  ceiling 만으로 VLM build 의 정밀도를 정당화하지 않는다 — §2 한계.)
- **radius 전반 flat** → SEM distractor 가 국소/주기적 → ROI 무용 → template-bank 로 선회.
  Job 2 를 싸게 redirect.

digest 는 `recall` 옆에 `n_eval/n_hit/n_no_candidate` 도 같이 찍어, recall 변화가 진짜 회복인지
분모 축소인지 한눈에 검증 가능하게 한다(survivorship 자가-점검).

## 6. TDD 계획 (vertical slices)

1. oracle 가 distractor 를 차단: 합성 프레임에 진짜 key + 멀리 떨어진 near-identical
   distractor; tight oracle ROI 에서 모든 candidate 가 window 안 + truth `in_topk`
   (메커니즘 + 효과 동시 검증).
2. `oracle_roi_radius=None` 이 현재와 byte-identical(regression guard).
3. `_crop_oracle_window` 가 경계에서 clamp.
4. `_consensus_template_ab` sweep 가 per-(modality,radius) **고정 분모** 카운트
   (`n_eval/n_hit/n_no_candidate/n_skipped`) 반환; 기본 off.
5. **survivorship guard**: full-frame 에서 유효한 점이 tight ROI 에서 후보 0개가 되면
   그 점은 `n_no_candidate`(⊆ miss)로 세지고 `n_eval` 은 그대로 — recall 분모가 줄지
   않음을 단언(곡선이 진짜 회복만 반영).

## 7. 위험 / 완화

- **합성 프레임에서 ensemble 의 finicky 한 동작** → test1 의 1차 단언은 "candidate 가 모두
  ROI 안"(scoring 무관, 결정적). truth in_topk 는 2차.
- **oracle 누수 오해** → oracle 은 *eval-only ceiling*, 배포 코드 아님을 spec·docstring·digest
  라벨에 명시("[ORACLE]"). `_gt_in_topk`/`_consensus_template_ab` 는 이미 ground truth 사용.
- **edge-clamp 로 작아진 window** → truth 는 항상 local 좌표 보존; recall 측정 유효(현실적).
