# Search-around 재설계(절대 배율 zoom-out + FOV 격자) — opencode oc-discuss 기록

- Run date: 2026-08-28
- Skill: oc-discuss
- Model: opencode-go/glm-5.3 variant=high (tier=heavy)
- Target: `live_search.py` fallback 탐색을 (c) PM 드롭다운 절대 배율 zoom-out + (d) 1-FOV
  step 격자 pan 으로 재설계하는 결론(`poc/workflow_3/docs/study/search_around_fov_reach_260828.md` §5)
- Elapsed: round1 117s · round2 35s · Session: ses_fb7b91f43ffeADUzDRppVaO4dj

## Claude 의 처음 입장

SEM 모달리티만: ① cond.txt 등록 배율에서 scale ≥ 0.15 인 가장 낮은 CG 단으로 PM 드롭다운
zoom-out(30K→5K, 50K→8K) ② 알게 된 한 scale 로만 매칭 ③ 안 보이면 step = 1 FOV(가장자리
더블클릭 2회) 사각 spiral 격자로 2R=60 µm 박스 커버(R=30 µm 시험값) ④ 후보 → recenter →
드롭다운으로 등록 배율 복귀 → confirm ⑤ 실패 시 delta 역행으로 원점 복귀 → exhausted → cube
⑥ OM 은 pan 만 ⑦ 클릭 횟수는 제약 아님(사용자 결정), 시간은 여전히 제약.

반증 조건: ~75 px SEM key 가 구조 매칭 안 됨 / 드롭다운 배율 변경의 부작용 / recenter 가
선형·반복 가능하지 않아 격자에 구멍 / OM align 이 먼저라 30 µm SEM 오차의 전제가 틀림.

## What the model said — round 1

> Inspected the diagnosis, physics doc, `live_search.py`, `controller.py`, `pm_dropdown.py`, and the ladder table. The position's direction (absolute-mag zoom-out + FOV-scale grid) is sound, but the spec as written has three load-bearing errors.
>
> **1. Registered mag is not a PM rung — the round-trip is impossible for the flagship case.** The CG ladder (your own `hitachi_mag_fov_pixel_260828.md` §3) is `…5000, 8000, 10000, 20000, 50000…` — there is **no 30000** (GT only adds higher rungs). The one calibrated office sample is registered at 30K. Step 4 "zoom back IN to registered mag via the dropdown (absolute)" cannot terminate: nearest rungs are 20K (scale 0.67, scraping the `MIN_CONFIRM_SCALE=0.6` floor, exactly the shrunken-key regime where aperture-problem precision is worst) or 50K. Confirm "at scale ~1.0" is unreachable. Either the spec needs a "nearest rung + wheel trim" hybrid (uncalibrated) or must argue the tool doesn't need the monitor mag restored — currently it does neither.
>
> **2. The single-scale formula is wrong as stated.** Matching scale = `(new_mag/reg_mag) × (template_px/frame_px)`. The frame is a screen-ROI crop (`controller.py:122-131`); `fw` is unmeasured (demo 320/768, real value OFFICE-VERIFY). Every feasibility number (key ≈77 px, "0.15 floor OK") assumes fw=512. At fw=320 the 5K key is ~48 px, effective scale 0.107 — below the floor the design itself cites. Today's 0.7–1.4 band partially absorbs this; single-scale doesn't. Cheap fix, but the spec's core arithmetic rests on it.
>
> **3. False candidates corrupt the grid.** Rank-1 precision is the SEM bottleneck; at 5K the FOV holds ~36× the area, i.e. ~36× more junction-like periodic structure, with key internals at ~53 nm/px. A false candidate triggers a mag round-trip, moves the stage, and invalidates the remaining recorded deltas — the spec has no grid-resume bookkeeping. Cost: minutes per false positive and a silently gapped sweep.
>
> **4. Dead-reckoning with zero feedback.** 16–32 chained edge double-clicks + mag changes, `nm_per_pixel=None`, no stage readback; the 2026-07-07 calibration covered single clicks, not chains. Origin-return and gap-free 1-FOV tiling (which also ignores the 0.12 margin clamp → 0.38 FOV/click, so 3 clicks not 2) are untested assumptions.
>
> Strongest objection I rejected: "time doesn't matter anymore" — the user's 2026-08-28 decision explicitly keeps time in scope, and this design multiplies per-event time ~5-10×; but the abort latch plus exhausted→notification bounds the damage.

## Claude 의 round 2 답변 (요지)

1. 부분 양보/부분 반박 — `choose_step_targets` 는 드롭다운 옵션을 **런타임에 읽어** 가장
   가까운 항목을 고르고(`pm_dropdown.py:247-269`), PM box OCR 판독(`_capture_rung`)이 배율
   피드백을 준다. 정확 복귀는 가정하지 않고 "가장 가까운 단 + OCR 확인" 으로 바꾼다.
   confirm 은 scale 1.0 이 필요 없다(20K/30K = 0.67 ≥ 0.6).
2. 양보 — `scale = (new/reg) × (fw/template_w)`. zoom-out 단은 고정 0.15 가 아니라 런타임
   fw 로 계산한 **최소 key 픽셀**로 고른다.
3. 양보 — **collect-then-chase**: 격자를 먼저 다 돌아 셀별 점수를 모으고 그다음 best-first
   추격. broad 점수는 경량 score 가 아니라 3채널 ensemble + rerank.
4. 산술 양보(0.12 margin → 0.38 FOV/click → FOV 당 3 클릭), "무피드백" 은 반박 — 0.38 FOV
   step 이면 연속 프레임이 62% 겹치므로 phase correlation 으로 **실제 이동량을 측정**(odometry).

## What the model said — round 2

> Verified the rebuttals against source: `_capture_rung` + `parse_pm_magnification` readback exists (cycle.py:1869-2013), ECC machinery exists (`mind_rerank.py`).
>
> **Objection 1 — dropping the termination claim.** OCR-verified nearest-rung restore is real feedback; that failure mode is dead. **Pressing one residual:** "20K/30K = 0.67 ≥ 0.6 by construction" is a *mag-ratio*, but `MIN_CONFIRM_SCALE` gates on the matcher's *pixel-ratio* scale — the exact fw-dependence you conceded in point 2. If runtime `fw` < template width, the effective confirm scale at the restored rung drops below 0.6 and the terminal guard (`live_search.py:251-255`) blocks the "match" verdict: the loop finds the key and cannot legally finish. Cost: a found-key event still ends `exhausted` → spurious engineer notification. Fix is the same as your #2 fix extended to step 4: resample the template (or recompute the floor) against runtime `fw` before declaring the confirm scale "by construction."
>
> **Objection 4 — accepting odometry, pressing one concrete failure case.** Phase/ECC correlation on aperture-problem frames (periodic lines, flat surrounds) doesn't just get noisy — it converges *confidently wrong*: a grating shifted by one period is indistinguishable, and ECC's cc stays high. Your odometry would accumulate a plausible-looking but period-aliased delta, so origin-return lands one period off with no error signal. Same physics that makes rank-1 precision the SEM bottleneck. The spec must gate each step: reject the measured shift when |measured − commanded| exceeds a bound, fall back to commanded delta, and flag drift for the office run. Also note "capture ~1 ms" is screen-capture cost only — the SEM monitor repaints at scan rate, so odometry frames must respect the existing `settle_sec`, or you'll correlate against the pre-move frame and measure zero.
>
> No new objections — the remaining risks (dropdown side-effects, recenter repeatability, real R) are already in your falsification list and are office-test questions, not spec defects.

## Judgement

### Agreed
- 방향(절대 배율 zoom-out + FOV 단위 격자)은 맞다. 모델도 "sound" 로 인정.
- 배율 복귀는 "가장 가까운 드롭다운 단 + PM OCR 확인" 이며 정확 복귀를 가정하지 않는다.
- 모든 scale 은 런타임 `fw/template_w` 기준으로 정규화한다 — 템플릿을 프레임 기준 base
  scale 로 한 번 리샘플하면 zoom-out 단 선택도 confirm 게이트도 순수 배율비가 된다.
- zoom-out 단은 고정 scale 0.15 가 아니라 **최소 key 픽셀**로 고른다(오피스 실측 상수).
- collect-then-chase. 추격 중 stage 이동으로 격자 delta 가 깨지지 않게 sweep 을 먼저 끝낸다.
- FOV 당 3 클릭(0.38 FOV/click). 각 클릭 뒤 `settle_sec` 을 지킨 프레임으로 phase
  correlation odometry 를 하되 `|측정 − 명령| > 한도` 면 명령값으로 폴백하고 drift 를 기록한다.
- 시간은 제약으로 남는다(3×3 ≈ 1 분, 4×4 ≈ 1.5-2 분 추정, abort 래치 + exhausted→cube 로 상한).

### Disputed
- 없음. 남은 불확실성(드롭다운 부작용, recenter 반복성, 실제 R, 최소 key px)은 양쪽 다
  오피스 실장비 테스트 항목으로 합의.

### I was wrong
- "1-scale 매칭" 을 배율비만으로 적었다 — 프레임 폭/템플릿 폭 비율을 빠뜨려 5K/8K 수치 전부가
  fw=512 가정이었다. 결과: zoom-out 단 선택 기준이 scale 상수에서 **런타임 key 픽셀**로 바뀐다.
- 후보 추격을 sweep 도중에 하려 했다 — 오탐 하나가 격자 bookkeeping 을 깨뜨린다. collect-then-chase 로 변경.
- FOV 당 recenter 2회로 계산했다 — margin clamp 때문에 3회.
- "등록 배율로 복귀" 를 당연시했다 — 드롭다운에 그 값이 없을 수 있고, confirm 게이트는 픽셀
  비율이라 복귀 단에서 0.6 아래로 떨어질 수 있다. base scale 정규화로 흡수.
- odometry 를 무조건 신뢰하려 했다 — 주기 구조에서 한 주기 어긋난 값이 높은 cc 로 나온다. 게이트 추가.

## Follow-up

- 스펙: `poc/workflow_3/docs/superpowers/specs/2026-08-28-search-around-zoomout-grid-design.md`
  (이 기록의 Agreed 를 그대로 반영).
