---
status: accepted
---

# Align Fail 보정의 기본 경로는 즉시 reposition+OK, pan/zoom 탐색은 fallback

## 결정

Align Fail(ALID=9006) 보정의 **PRIMARY 경로**는 paused 화면에서 곧바로 crosshair 를
recipe-matched 점으로 옮기고(더블클릭 recenter) OK 를 누르는 것이다(`align_fail_correct.py`).
`live_align_search.py` 의 pan/zoom two-phase 탐색은 **FALLBACK** 으로 강등하며, paused 프레임에서
key 가 인식되지 않을 때(`key_visibility_gate` 가 low)만 진입한다.

- paused 프레임 매칭은 **near-native scale**(`PAUSED_SCALES = DEFAULT_SCALES`)로 한다. 장비는
  레시피 등록 배율에서 멈췄으므로 key 가 보인다면 ~1.0 크기다.
- 가시성 게이트는 `best_scale >= MIN_CONFIRM_SCALE` 를 요구하고, 약한 `adjust` 는 `orb>0`
  (feature 보강)일 때만 가시로 인정한다(강한 `match` 는 edge 구조만으로 인정).
- OK 버튼은 SEM ROI 밖 dialog 컨트롤이라 VLM 으로 찾고(`vlm_ok_button_box.py`) **screen 절대
  좌표**로 single-click 한다. reposition(FOV-local `move_to_point`)과 좌표공간이 다르다.

## 맥락 / 이유

기존 설계는 "wafer 에서 key 를 *찾아야* 한다"는 가정 위에 pan/zoom 탐색 루프를 main flow 로 두었다.
그러나 사용자가 확정한 바(2026-05-27), 엔지니어가 박스로 표시한 key 는 **대개 잘못된 crosshair 근처에 이미
보인다.** 따라서 흔한 경우에는 hunt-first 가 불필요할 뿐 아니라, 과도한 pan 때문에 오히려 위험하다.

또한 paused 프레임을 broad(miniature) scale band 로 매칭하면, tiny-scale chamfer 가 feature 없는
배경에서도 높게 나와 거짓 가시(false visible)로 잡힌다 — `live_align_search` 의 terminal 가드가
막던 바로 그 함정. 그래서 가시성 판정에 near-native scale + scale/feature 가드를 둔다.

## 결과 (Consequences)

- 신규 진입점 `align_fail_correct.py`(PRIMARY) + `vlm_ok_button_box.py`(OK locator).
- `SEMMonitorController` Protocol 에 `capture_screen()`(전체 화면)과 `click_screen()`(screen 단일
  클릭) 추가. `move_to_point`(FOV-local 더블클릭 recenter)와 좌표공간이 분리된다.
- `live_align_search.py` 의 `_clamp_click`→`clamp_to_fov`, `_route_template`→`route_template` 로
  승격해 두 경로가 공유.
- 절차 문서 §3 step 표 재구성: Step 4=PRIMARY, Step 5~8=FALLBACK.
- 게이트 임계(scale/feature 조건, `STRUCTURE_POLICY`)는 cold-start 이며 실데이터 calibration 대상.
- 미해결: 등록 이미지의 엔지니어 박스 내부로 template 을 crop 할지(`crop_template_to_box`, 기본
  off)는 오피스 실파일 확인 후 결정(align key 자체가 box-in-box 일 수 있어 주석과 혼동 가능).
