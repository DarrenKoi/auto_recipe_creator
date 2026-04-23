# SEM 모니터에서 Align Key 탐색 — 타당성 분석

**목표.** Align-fail 알람이 발생하면 회복 플로우를 자동화한다:
RCS 로 해당 Tool 에 접속 (이미 `poc/workflow_1/` 에서 POC 완료) → RCS 안의
SEM monitoring 화면을 조작해 **레시피에 저장된 align-key 이미지를 기준으로
웨이퍼 위에서 같은 패턴을 찾을 때까지 탐색**한다. 입력은 레시피에 저장된
align-key 이미지, 출력은 그 패턴이 보이는 stage / FOV 위치.

이 문서는 다음 질문에 순서대로 답한다.

1. 이 접근이 현실적으로 가능한가?
2. 현재 배포 중인 VLM 들이 이미지 매칭을 할 수 있는가?
3. 전통적인 image processing 이 여전히 필요한가?
4. VLM 이 "레시피 이미지 ≈ 현재 SEM 이미지" 인지 판단할 수 있는가?
5. 실제로 추진할 구체적인 아키텍처.

---

## 1. 전체 아이디어는 타당한가?

**가능하다. 단, 순수 VLM 기반은 아니다.** 워크플로우 전체 — align fail 감지
→ Tool 접속 → SEM monitor 조작 → Key 탐색 — 는 범위가 명확하고 이미
작동 중인 `poc/workflow_1/` 위에 깔끔하게 얹을 수 있다. 문제는 핵심 단계,
즉 "레시피의 align-key 패치를 기준으로 라이브 SEM FOV 에서 같은 패턴을
찾아라" 부분이 **현재 우리가 가진 VLM 들에게 맞지 않는다**는 점이다.

이 핵심 단계는 전형적인 **template matching / registration** 문제이며,
이미지 도메인도 매우 좁다 (grayscale, 고배율, 나노 스케일 기하 패턴 —
cross mark, box-in-box, checkerboard, L/T 형태 등). 실제 반도체 계측
장비 (KLA, Hitachi, AMAT) 도 내부적으로 generative VLM 이 아니라
normalized cross-correlation 과 feature-based registration 으로 이
문제를 푼다. 외부에서 만드는 자동화도 같은 패턴을 따라야 한다.

그래서 현실적인 재정의는 이렇게 된다.

- 가능한 형태: **classical CV 매칭 엔진 + 공간 추론용 보조 VLM +
  RCS 기반 마우스/스테이지 제어.**
- 무리한 형태: **"VLM 에게 두 이미지를 던져주고 매칭 위치를 가리키게 한다."**

---

## 2. 현재 배포한 VLM 들로 가능한가?

Flask VLM 프록시에 등록된 모델은 다음과 같다 (`flask_api/vlm_serve/config.py`).

| Slug               | Model              | 주 용도                            |
|--------------------|--------------------|------------------------------------|
| `ui-venus`         | UI-Venus-1.5-8B    | GUI element grounding (click-point)|
| `mai-ui`           | MAI-UI-8B          | GUI element grounding              |
| `ui-tars`          | UI-TARS-1.5-7B     | GUI agent (현재 비활성)            |
| `paddleocr-vl-1.5` | PaddleOCR-VL-1.5   | OCR 보조 VLM                       |
| `got-ocr`          | GOT-OCR-2.0-hf     | OCR                                |

전부 **앱 스크린샷 / 문서 / 자연 이미지** 로 학습된 모델이다. SEM prior 가
있는 모델은 하나도 없다. 이 사실은 생각보다 훨씬 치명적이다.

- **Grounding 계열** (UI-Venus, MAI-UI, UI-TARS) 은 *텍스트로 기술된 UI
  요소* ("Login 버튼", "Server 콤보박스") 의 `[x, y]` 를 출력하도록
  최적화되어 있다. "두 번째 이미지와 같은 시각 패턴을 찾아라" 라는 태스크
  자체가 학습 타깃이 아니며, 대부분은 single-image 입력만 받기 때문에
  reference-vs-live 비교 프롬프트 자체가 어색하다.
- **OCR 계열** (PaddleOCR-VL, GOT-OCR) 은 문자와 레이아웃을 인식한다.
  SEM 의 기하 패턴 매칭은 수행하지 않는다.

따라서 "`ui-venus-1.5-8b` 에 align-key 이미지를 주고 SEM 캡처에서 위치를
찾게 할 수 있는가" 에 대한 답은 이렇다.

- **기술적으로는**: 백엔드가 multi-image prompt 를 받고, 타깃을 텍스트로
  설명해준다면 ("네 개의 팔을 가진 십자 표식") 좌표는 돌려준다.
- **실용적으로는**: **주 검출기로는 쓸 수 없다.** Recall 낮고, 특징 없는
  웨이퍼 영역에서 false positive 많고, confidence 가 calibrated 되지
  않는다. Stage 이동에 필요한 sub-pixel 정확도는 grounding 모델이 UI
  도메인에서도 제공하지 않는 특성이다.

즉, 현재 VLM 들은 SEM 도메인의 *픽셀 수준* 매칭에는 맞지 않는 도구다.
상위 레벨에서의 보조 역할은 여전히 가능하다 (§5 참조).

---

## 3. Classical image processing 은 여전히 필요한가?

**필요하다. 그리고 주 매칭기로 써야 한다.** 이 문제는 classical CV 가
이기는 모든 조건을 가지고 있다.

- Reference 와 target 이 같은 modality (둘 다 grayscale SEM 캡처).
- 회전은 대체로 bounded (스테이지가 웨이퍼 notch 에 정렬되어 있어 yaw 가
  거의 0 에 가깝다).
- 스케일은 레시피 배율로 알려져 있고, 차이가 있어도 한두 단계의 이산
  배율만 시도하면 된다.
- 조명은 단일 Tool 안에서는 비교적 안정적이다.
- 성공 판정 기준이 단단한 숫자 — NCC correlation score.

구체적 툴박스 (내가 시도할 순서대로).

1. **Normalized Cross-Correlation (NCC) template matching.**
   `cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)`. 싸고 빠르고,
   업계 default. Multi-scale (±1 magnification step) 과 작은 rotation
   sweep (예: ±3°) 을 얹으면 거의 대부분 잡힌다. Peak score 와 위치를
   보고, threshold 이하면 기각 — threshold 는 실제 데이터 몇 장으로 튜닝.

2. **Feature-based matching (ORB, AKAZE, SIFT).** Key 에 corner/junction
   이 분명하지만 배경이 복잡할 때 유리. RANSAC homography 로 sub-pixel
   좌표까지 얻는다. NCC 보다 느리지만 부분 가림이나 대비 변화에 훨씬
   robust 하다.

3. **Phase correlation** — 회전이 진짜로 0 인 pure translation 상황이면
   가장 빠르고 조명 변화에 매우 강하다.

4. **가벼운 preprocessing** 을 두 이미지 모두에 공통으로 먼저 적용:
   CLAHE (local contrast), 작은 Gaussian blur (픽셀 노이즈 억제),
   그리고 선택적으로 Sobel / gradient 이미지로 변환해서 "밝기" 가 아닌
   "구조" 로 매칭하게 한다.

이 스택은 deterministic 이고, CPU 에서 FOV 당 수 ms 로 돌고, frame 간에
비교 가능한 물리적 confidence 를 돌려준다. "웨이퍼를 돌아다니며 Key 를
찾을 때까지 탐색한다" 는 문제가 실제로 필요로 하는 것은 바로 이것 —
stage 가 움직일 때마다 단조 증가해야 할 *정량적 유사도 신호* 다.

---

## 4. VLM 이 "레시피 이미지 ≈ 현재 SEM 이미지" 판단을 할 수 있는가?

이 질문은 "Key 위치를 찾아라" 보다 훨씬 좁은 질문이고, 답도 더 미묘하다.

- **Soft 한 sanity-check 이진 분류로?** VLM 은 "이 두 패치가 같은 패턴을
  보이는가?" 를 *의미 수준* 에서 종종 맞춘다 ("양쪽 다 네 팔 십자
  fiducial 이 보임"). 완전히 무쓸모는 아니다 — NCC 가 grain noise peak
  에 잘못 걸렸을 때 veto 하는 용도로 쓸 수 있다.
- **Stage 이동의 최종 결정자로?** 불가. Calibrated similarity score 가
  없고, 특징 없는 웨이퍼 영역에서 유사하다고 환각하며, 같은 두 프레임에
  대해서도 답변이 프레임 간에 반복되지 않는다.
- **SEM 이미지에서의 정확도?** 미검증. 우리가 배포한 모델들은 SEM 데이터로
  학습된 적이 없다. 실제 Tool 캡처로 벤치마크하기 전까지는 VLM 의 판정을
  "측정" 이 아니라 "의견" 으로 취급해야 한다.

이 프로젝트의 경험칙: **픽셀 점수는 OpenCV 에게, 분위기 파악은 VLM 에게.**

---

## 5. 제안 아키텍처

각 도구의 강점을 살린 최소한의 정직한 설계.

```
┌──────────────────────────────────────────────────────────────┐
│ Align-fail 알람 (poc/workflow_1 에서 이미 작동)              │
│   → Tool 접속 → SEM monitor 창 forefront                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Loop: search_align_key()                                     │
│                                                              │
│  (a) 현재 SEM FOV 캡처 (전체 화면이 아니라 monitor 영역만)   │
│                                                              │
│  (b) classical matcher (OpenCV):                             │
│        score, (x,y), scale, rot = match(template, fov)       │
│      if score ≥ high_threshold: → 성공, (x,y) 반환           │
│      if score ≥ low_threshold:  → 후보, VLM 에게 확인 요청   │
│      else:                      → 매칭 없음                  │
│                                                              │
│  (c) VLM 보조 (선택적, classical 이 애매할 때만):            │
│      • "이 FOV 는 대체로 feature 없는 웨이퍼인가?"           │
│      • "어느 방향에 구조가 더 많은가 — 좌/우/상/하?"         │
│         (정성적 이동 힌트)                                   │
│      • "이 두 패치는 같은 fiducial 을 보이는가?"             │
│                                                              │
│  (d) RCS 를 통한 stage / FOV 이동:                           │
│      • 기존 poc/workflow_1 의 pywinauto + pynput 경로로      │
│        SEM view 를 translate                                 │
│      • 기본은 coarse scan (spiral / raster), VLM 힌트로      │
│        다음 스텝을 살짝 bias                                 │
└──────────────────────────────────────────────────────────────┘
```

이 분리가 유효한 이유.

- **Classical matcher** 가 ground truth 역할을 한다. Threshold 로
  성공/실패를 결정하고 로그에 남길 수 있는 숫자를 준다.
- **VLM 보조** 는 애매한 중간 구간과 상위 "다음에 어디로 움직일까" 추론
  에만 호출된다. VLM 호출이 드물게 유지되므로 latency/비용이 작고, happy
  path 에서는 시스템이 deterministic 으로 굴러간다.
- **RCS 자동화** 는 `poc/workflow_1/` 이 이미 알고 있는 창 검색/활성화/
  SEM monitor 내 클릭·드래그 경로를 그대로 재사용한다.

---

## 6. 실제 다음 단계 (리스크 낮고 비용 작은 순서)

1. **먼저 작은 데이터셋을 수집한다.** (레시피 align-key 이미지, 그것이
   포함된 live SEM FOV) 쌍 10~30 개 + Key 가 없는 FOV 10~30 개. 이게
   없으면 threshold 튜닝도 평가도 불가능하다. 사무실 실장비에서만 찍을
   수 있으므로, 매칭 코드 짜기 전에 **이 데이터 확보가 가장 우선**이다.

2. **Classical matcher 를 오프라인에서 먼저 프로토타입**한다 (Mac 에서도
   OK). 위 쌍들에 대해: matches 와 non-matches 의 NCC peak score 분포,
   단일 threshold 로 분리 가능한지. 가능하면 문제의 8할이 이미 풀린
   것이고, 나머지는 배관일 뿐이다.

3. **Classical score 가 애매한 경우에만** VLM confirmation 단계를 붙인다.
   `ui-venus-1.5-8b` 에 "same pattern? yes/no" 태스크를 주고, ground
   truth 와의 불일치율을 먼저 측정한 뒤에만 루프에 넣는다.

4. **그 다음에** `poc/workflow_1/` 의 RCS 제어 루프에 통합한다
   (`workflow_runner.py` + window helper 재사용). 처음에는 `SAFE_MODE`
   로 클릭 로그만 찍고 dispatch 는 막아둔 상태에서 의사결정을 종이로
   검증한 뒤, 실이동을 켠다.

5. **디버그 흔적을 남긴다.** 모든 탐색 시도마다 FOV 캡처, 템플릿,
   matcher 의 score map, 선택된 (x,y), 그리고 VLM 대화 (있다면) 를
   `poc/workflow_1/debug_images/align_search/<ts>/` 아래에 저장한다.
   원격 장비에서 miss 를 진단할 수 있는 유일한 방법이다.

---

## 7. 짧은 답변 요약

- **타당한가?** 그렇다. 단, classical CV 가 중심이고 VLM·RCS 자동화는
  바깥을 감싸는 형태다.
- **현재 VLM 들이 매칭 자체를 할 수 있는가?** 신뢰도 있게는 불가. 그들은
  GUI/OCR 모델이지 SEM 패턴 매처가 아니며, calibrated score 를 주지 않는다.
- **Image processing 이 필요한가?** 필요하다. 이 문제에 가장 적합한
  도구이며, 주 매칭기로 써야 한다 (NCC template matching 기본, 필요하면
  feature matching 과 전처리 추가).
- **VLM 이 "유사/비유사" 판정을 할 수 있는가?** 거친 의미 수준에서는
  가능하고, veto / tiebreaker 로는 유용하다. Stage 이동의 핵심 결정자로는
  부적합하다.
- **무엇을 먼저 만들어야 하는가?** 실장비에서 촬영한 레시피/FOV 쌍
  데이터셋. 그다음 오프라인 NCC 프로토타입. 그 score 분포가 이후 모든
  설계의 토대가 된다.
