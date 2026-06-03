# white box(crop) vs crosshair(inpaint) 제거 — 알고리즘 비교 및 실데이터 검증

날짜: 2026-06-03 09:52
대상 파일: `poc/workflow_2/align_point_correction.py` (수정), `poc/workflow_2/crosshair_detect.py` (재사용),
신규 throwaway: `synth_white_box_demo.py`, `box_crop_real_check.py`, `box_hough_experiment.py`, `crosshair_removal_check.py`

## 0. 한 줄 요약

reference 의 **white box** 와 scene 의 **crosshair** 는 둘 다 "흰 오버레이 제거"처럼 보이지만,
**구조가 전역적이냐(crosshair) 국소적이냐(box)** 때문에 알고리즘이 정반대다 — crosshair 는 검출이
명확해 inpaint 로 6/6 깔끔히 지워지고, box 는 dense 배경에서 검출 자체가 불완전하다.

## 1. 진행 사항

- **white box inner-crop 수정**: 기존 `_inner_crop_for_box` 의 고정 inset(3px)이, 검출 bbox 가
  Otsu+dilation 으로 stroke 바깥까지 부풀어 **반대편 흰 선을 못 피하는** 케이스를 합성으로 발견
  (`max_white_in_crop=255`). inner-hole contour + bright-border trim 으로 교체해 **흰 선 0 보장**.
  검증: 합성 `255→189`, 엣지 3종(두꺼운 stroke·열린 ring 포함) `ALL OK`, 실데이터 6.png `196 OK`.
- **white box 직사각형 게이트 추가**: busy SEM 에서 소자 패턴(구불구불한 밝은 선)을 박스로 오인하는
  문제에 대해, 후보 contour 가 *축정렬 사각형 frame* 인지 검사(`_rect_frame_ok`: frame_ratio +
  4변 coverage + approxPolyDP 코너 수). 효과는 **fail-safe** — busy 배경에서 오검출 대신 *거부*(None)
  → centered-crop 폴백. 깨끗한 배경/합성은 정상 검출.
- **white box 검출 정직한 한계 확인**: 인터넷 SEM 6장(인위 삽입 박스)으로 검증.
  - contour+게이트: **1/6**(6.png, 매끈 배경)만 검출, 나머지는 거부.
  - Hough 조립 실험: 6/6 무언가 검출하나 정확한 건 ~2.5/6 (dense 에선 소자 직선을 변으로 오조립).
  - → 순수 CV 로는 dense 배경의 흐릿한 박스를 신뢰성 있게 못 짚음. Hough 실험은 포팅하지 않고 throwaway 유지.
- **crosshair 검출+제거**: 동일 base SEM 6장(십자선 삽입)으로 검증. 기존 `detect_crosshair` 재사용
  → **6/6 검출**(conf~0.93), full-span band mask + `cv2.inpaint`(Telea) → **6/6 제거**
  (선 자리 밝은 잔존 픽셀 `~2300→0`, 육안상 고스트·이음새 없음, 소자 라인 교차부도 자연 복원).

## 2. 두 방식의 알고리즘 비교 (이 저널의 핵심)

| 축 | **crosshair 제거** (`crosshair_detect.py`) | **white box 처리** (`align_point_correction.py`) |
|---|---|---|
| 신호 모델 | 독립적인 **1D 선 2개**(전폭 H + 전높이 V) | 연결된 **2D 닫힌 사각형 1개**(4변) |
| 임계 | **절대** saturation `gray>=235` (ladder 235/215/195) | **국소** top-hat(9×9) + **적응** Otsu |
| 핵심 판별자 | **전역 span**(프레임을 가로지름) | **모양**(닫힘·hollow·종횡비·코너·frame coverage) |
| 위치 추정 | 방향성 opening → row/col **projection argmax** | contour **boundingRect** 중 best 선택 |
| "제거" 연산 | **inpaint**(덮인 구조를 메움) | **crop**(둘러싼 주석을 잘라냄) |

### 2.1 무엇을 찾나 — 1D 선 vs 2D 사각형

- **crosshair**: 가로선·세로선을 *따로* 찾는 두 개의 1D peak 문제.
  방향성 morphology 로 "선 방향으로만 긴 커널" opening → 그 방향 긴 얇은 런만 생존:
  ```
  h_mask = open(close(bright,(gap,1)), (lh=0.30·W, 1))   # 가로선
  v_mask = open(close(bright,(1,gap)), (1, lv=0.30·H))   # 세로선
  cy = argmax(row_coverage(h_mask)); cx = argmax(col_coverage(v_mask))
  ```
- **box**: 4변이 연결된 닫힌 윤곽 하나를 찾는 2D 형상 매칭. 여러 후보 중 "진짜 사각형"을 골라야 함.
  ```
  contours = findContours(tophat→Otsu→dilate, RETR_EXTERNAL)
  filters: 면적/짧은변/edge margin/종횡비/hollow + 직사각형 게이트 → 면적 최대 통과 후보
  ```

### 2.2 임계 — 절대 vs 국소·적응

- crosshair 는 **절대 임계**(`gray>=235`). 십자선은 거의 순백 + 전역적이라 배경 밝기와 무관하게 자른다.
  top-hat 을 안 쓰는 이유는 흰 배경에서 top-hat 응답이 0 이 되는 문제 회피(v2 설계 근거).
- box 는 **top-hat(국소 대비) + Otsu(적응)**. 전역적으로 흐릿해도 국소적으론 밝은 걸 잡으려는 의도지만,
  **소자 라인도 국소적으로 밝고 가늘어 동일하게 통과** → busy 배경 오검출의 1차 원인.

### 2.3 distractor 분리 — 가장 결정적 차이

- **crosshair 의 무기 = 전역 span.** opening 커널 길이가 `0.30×프레임` 이라, 프레임의 30%+ 를
  가로지르는 구조만 생존한다. 소자 패턴은 아무리 밝아도 전폭을 안 가로지르니 opening 에서 소멸 → 6/6.
- **box 엔 전역 서명이 없다.** "닫힘·hollow·종횡비·코너·frame coverage" 같은 *국소 형상* 단서만 있고,
  이는 소자 덩어리가 흉내낼 수 있어 판별력이 약하다 → 1/6.
- 결론: 검출 난이도를 가른 건 대비·두께가 아니라 **구조의 전역성(global span) 여부**.

### 2.4 "제거" 연산이 정반대인 이유 — 메움 vs 잘라냄

- **crosshair 는 원하는 구조 *위에 덮여*** 있다 → 지우면 아래 SEM 을 복원해야 함 → `cv2.inpaint`(주변 채움).
- **box 는 원하는 구조(align key)를 *둘러싸고*** 있다 → 안쪽만 두고 테두리는 버리면 됨 → **crop**(복원 불필요).
- 즉 **덮인 방해물 → inpaint(crosshair), 둘러싼 주석 → crop(box).** 같은 "제거"라도 방향이 반대다.

## 3. 다음 단계

- **box 검출을 busy 배경까지 끌어올리기** = 남은 진짜 과제. 워크스트림 철학("VLM 이 영역, CV 가 좌표")대로
  **VLM-region → CV 정밀화** 하이브리드 권장(오피스 네트워크 필요). 인터넷 dense 이미지는 worst-case 라,
  실제 align-key recipe(보통 더 깨끗)에선 순수 CV 가 더 잘 될 가능성도 있으니 **실 office IMAP 로 먼저 측정** 후 판단.
- **crosshair 제거의 매칭 파이프라인 통합**: inpaint 된 scene 을 matcher 에 넘겨 false anchor 제거.
  재사용 함수로 `remove_crosshair(gray)->inpainted` 를 `crosshair_detect.py` 에 추가하는 안.
- throwaway 4종(synth/real check/hough/crosshair check)은 검증 하니스로 유지 — 실데이터로 재검증 시 재사용.

## 4. 메모리 업데이트

- 신규/갱신 없음 (코드 주석에 근거가 모두 들어가 있어, 이 저널을 단일 기록으로 남김).
- 관련 기존 메모리: [[align-fail-correction-model]] (white box=align key 위치 표시, crosshair=틀린 현재 위치).
