# white box photometric 전환 + crosshair v2 production 연결 + inpaint 검증 강화 (Codex 협업)

날짜: 2026-06-03 16:52
대상 파일: `poc/workflow_2/align_point_correction.py`(흰 box 검출 photometric 전환 + crosshair v2 연결),
`poc/workflow_2/crosshair_removal_check.py`(inpaint 검증 강화), `poc/workflow_2/box_crop_real_check.py`(앞 세션: templates 입력/flat 결과)

## 0. 한 줄 요약

흰 box 검출이 1/6 → **6/6** 으로 뛴 건 알고리즘을 더 정교하게 만들어서가 아니라 **더 단순하게** 만들어서다 —
overlay 는 소프트웨어가 그린 **고정 디지털 값**(샘플에선 정확히 255)이라 히스토그램상 SEM 내용과 **분리된 섬**
을 이룬다. adaptive(top-hat+Otsu)가 이 신호를 상대화로 버린 게 실패 원인이었다. 더불어 Codex 협업으로 (a)
production 이 검증 안 된 *구* crosshair 검출기를 쓰던 divergence 를 잡아 v2 로 연결, (b) inpaint 검증을 강화했다.

## 1. 진행 사항

- **흰 box 검출 photometric 전환** (`_detect_white_box`): front-end 를 교체.
  - 신규 `_detect_overlay_saturation` — 히스토그램 상단에서 *고립된 밝은 섬*(notch 로 bulk 와 분리된 클러스터)
    의 하한을 찾는다. 255 하드코딩이 아니라, anti-alias/JPEG 로 252~255 클러스터가 돼도 통째로 잡는다.
  - 신규 `_detect_white_box_photometric` — `inRange(sat_low,255)` → `MORPH_CLOSE(5x5)` → contour → 공유 게이트.
  - 기존 top-hat+Otsu 는 `_detect_white_box_adaptive` 로 이름만 바꿔 **폴백**으로 유지(저대비/연속분포 대비).
  - 게이트 루프는 `_select_box_from_mask` 로 추출해 두 경로가 공유.
  - 검증: `box_crop_real_check.py`(templates/whitebox_samples 6장) **6/6 검출**(이전 1/6). stroke(255)는
    6/6 모두 inner crop 에서 0 픽셀. (checker 의 `crop_max<thr` 지표는 1.png 에서 박스 *내부의 실제* 밝은
    소자(235)를 stroke 잔존으로 오인해 WARN — 지표 한계지 검출/크롭 문제 아님. crop 내 255 픽셀 = 0 확인.)
  - 합성/매처 무회귀: `synth_white_box_demo.py` OK, `test_align_key_match.py` 10/10.

- **crosshair v2 production 연결** (`align_point_correction.py:1487` 부근): 검증된 `crosshair_detect.detect_crosshair`
  (절대 saturation ladder + 방향성 morphology, 합성 6/6)를 main 처리 경로에 연결. 기존엔 *구* `_detect_existing_crosshair`
  (top-hat+projection)를 쓰고 있었다 — Codex 가 호출부 cross-ref 로 잡아낸 divergence. 구 함수는 crosshair_detect.py
  probe 비교용으로 남겨 둠. `align_similarity` 와 동일하게 로컬 import 로 순환참조 회피.

- **inpaint 검증 강화** (`crosshair_removal_check.py`, Codex 제안): 밝은-잔존-픽셀 count 외에 두 지표 추가 —
  (a) inpaint 결과를 `detect_crosshair` 로 **재검출**(검출되면 실패), (b) 제거 band 의 **Canny edge 밀도 / 인접
  평행 band** 비(`_edge_band_ratio`, 임계 `EDGE_BAND_RATIO_MAX=1.8`)로 *어두운* 이음새/고스트 탐지.
  검증: 6/6 모두 redetect=N, edge_ratio ≤1.73(<1.8). 잔존 count 만 보던 약한 지표를 보완.

## 2. 핵심 발견 — overlay 는 고정 채도값(photometric), 모양 아님

- 실측: 6장 모두 box stroke 가 **정확히 255**, 그 아래 가장 밝은 SEM 픽셀은 239/215/204/195/207/200 으로 **gap 분리**.
- `(gray>=254)` → `MORPH_CLOSE` → 최대 contour 가 6/6 박스 bbox 복원, fill≈0.01(hollow frame), connected component **1개**
  (소자 오염 0). 즉 device feature 는 255 에 *닿지 않는다*.
- 왜 adaptive 가 실패했나: top-hat 은 얇고 밝은 구조(=소자 라인 포함)를 살리고, Otsu 는 *상대* 임계라 그 위에서
  소자 픽셀을 대거 통과 → 박스 edge 가 묻힘 → 직사각형 게이트가 깨끗한 frame 을 못 찾고 폴백.
- 일반 원리: 소프트웨어가 *그려 넣은* 합성 콘텐츠는 보통 **히스토그램 문제**(부자연스런 spike/notch)지 morphology/
  shape 문제가 아니다. crosshair 가 6/6 인 이유(전역 span 서명)와 짝을 이루는, box 의 *광도* 서명.

## 3. Codex 협업 — 강화 목록(요약, 우선순위)

- **High**: ① crosshair v2 연결(완료) ② crosshair 렌더값/소프트닝 튜너블 노출(`SAT_THRESH_LADDER`, `SPAN_RATIO`,
  `GAP_BRIDGE_RATIO`) ③ wafer edge/scale bar 오검출 게이트(`MAX_THICKNESS_PX`, h/v 강도 일치 ±15) ④ box photometric
  노브(`RCP_BOX_SAT_*`, 완료) ⑤ sub-pixel 보존 ⑥ clipped crosshair: 최장 연결성분 게이트 + 낮은 SPAN_RATIO
  ⑦ inpaint 검증 강화(완료) — 재검출 + edge-band 비.
- **Medium**: 다중 box top-N 랭킹, broken-corner 시 Hough 재구성 폴백, stroke 폭 추정으로 close-kernel/trim 적응,
  focus blur `MIN_SHARPNESS_LAPVAR` 15~80 스윕.
- **Shared**: 히스토그램 island 유틸은 box/crosshair 가 *공유*하되, mass 게이트와 caller geometry 는 **분리** 유지
  (crosshair 는 scale bar/text 가 있어 monolithic 통합 시 오발화).
- **Office 검증 계획**: 타입별 라벨 배치(recipe OM/SEM+box, msr S+crosshair, msr E, blur) → box IoU·crop 잔존·crosshair
  center error·S recall·E false-positive·inpaint 전후 매처 winner 변화 측정. 목표: S recall↑ & E FP→0; box 는 **오크롭보다
  no-detect/폴백을 선호**(CV 좌표 권위 원칙). 게이트별 reject 사유코드 로깅으로 과적합 없이 튜닝.

## 4. 한계 — office 미검증, VLM+CV 백업 유지 (사용자 결정)

- 위 전부 **집(home)의 인터넷/합성 샘플**로만 검증됐다. office 실데이터에서 overlay 값/anti-alias/콘트라스트가
  다르면 photometric 섬을 못 잡고 adaptive 폴백으로 떨어질 수 있다 — 첫 office IMAP 에서 **히스토그램 1회 확인**이
  최우선. notch_gap/mass 만 튜닝하면 될 가능성이 높다.
- **VLM+CV 하이브리드는 백업 플랜으로 유지**한다(사용자 명시). photometric 이 주 경로지만, busy/모호 FOV 에서
  실패 시 워크스트림 철학("VLM 이 영역, CV 가 좌표")대로 VLM-region → CV 정밀화로 폴백한다. [[photometric-box-home-validated-only]]

## 5. 다음 단계

- office 첫 배치에서 §3 검증 계획 실행 → `RCP_BOX_SAT_*` / crosshair ladder 캘리브레이션.
- 미구현 High 잔여: crosshair clipped-arm 게이트, sub-pixel 보존, wafer-edge h/v 강도일치 게이트.
- box top-N 랭킹 + broken-corner Hough 폴백(Medium)은 실데이터에서 필요성 확인 후.

## 6. 메모리 업데이트

- 신규 [[photometric-box-home-validated-only]] — photometric box 는 home 검증만, office 미검증, VLM+CV 백업 유지.
- 관련: [[align-fail-correction-model]](흰 box=align key 위치 표시, crosshair=틀린 현재 위치).
