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

**필요하다. 단, 순수 픽셀 NCC 가 아니라 "구조(structure) 수준" 의 매칭이
중심이어야 한다.** 이 구분이 매우 중요하다.

### 3.0 "같다" 를 어느 레벨에서 판단할 것인가

레시피에 저장된 align-key 이미지와 SEM monitor 에서 본 이미지는 **픽셀로는
같지 않다.** 같은 Tool, 같은 웨이퍼 위 같은 fiducial 이라도 다음이 전부
다르게 찍힌다.

- beam current / detector gain / dwell time 차이 → 밝기·대비·노이즈 분포
- 디포커스 미묘한 변화 → 엣지 sharpness
- 스테이지 yaw 의 아주 작은 회전 (수 도 이내)
- 배율이 정확히 같아도 scan rotation 이나 rescan 으로 인한 1~2% 스케일
- 웨이퍼 표면 오염, charging, 노이즈 프레임

그래서 사용자의 직관이 정확하다 — "픽셀이 똑같다" 가 아니라 **"큰 박스
3~4개가 특정 배치로 있다"** 가 우리가 실제로 의지할 signature 다. 이 수준은
다음과 같이 계층적으로 정리된다.

| 레벨 | 무엇을 비교하는가 | 대표 기법 | 대비/노이즈 변화에 |
|------|------------------|-----------|-------------------|
| L0 픽셀 | 밝기 값 그대로 | `cv2.matchTemplate(TM_CCOEFF_NORMED)` | **약함** |
| L1 엣지/그래디언트 | Sobel / Canny 후 매칭 | Edge-NCC, **Chamfer matching** | **강함** |
| L2 Keypoint | corner/blob descriptor | **ORB / AKAZE / SIFT** + RANSAC | **매우 강함** |
| L3 형상(contour) | 박스 개수·크기·배치 | Contour detect → geometric matching | **매우 강함** |
| L4 의미 | "박스 3개 삼각 배치" | VLM / CNN embedding | 강하지만 비정량 |

사용자가 묘사한 "큰 박스 3~4개와 마크" 는 **L1 ~ L3 에 정확히 들어맞는
패턴**이다. 따라서 본 시스템의 주 매칭기는 L1/L3 조합이고, L0 raw NCC 는
보조/샘플링용이다.

### 3.1 왜 raw 픽셀 NCC 만으로는 부족한가 (그리고 그래도 쓸모는 있다)

`cv2.matchTemplate(TM_CCOEFF_NORMED)` 은 평균을 뺀 cross-correlation 이기
때문에 **전역 밝기 offset 과 contrast scale 에는 이미 정규화되어 있다.**
즉 "전체가 조금 어둡다" 정도는 견딘다. 하지만 다음에는 빠르게 무너진다.

- **국소적 contrast 변화** (CLAHE 가 필요한 이유)
- **노이즈 분포가 크게 다른 경우** — 노이즈도 밝기 통계에 섞여 들어간다
- **엣지의 부호가 반전된 경우** (밝은 박스 ↔ 어두운 박스) — 이건 raw NCC
  에서는 치명적이다
- **텍스처가 다른 경우** — 템플릿은 깨끗, live FOV 는 charging 으로 인해
  같은 박스가 texture 로 덮여 있을 수 있다

그래서 raw 픽셀 NCC 는 **전처리 없이 주 매칭기로 쓰면 안 된다.** 다만
다음 두 용도로는 유용하다.

- **초기 coarse search 의 속도 캐리**: 다른 기법보다 10~100배 빠르니까,
  의심 영역을 top-N 개로 줄이는 1차 필터로 돌리고, 상세 검증은 L1/L2 로
  한다.
- **동일 Tool 내에서 같은 스캔 조건이 보장되는 경우** 의 확인용.

### 3.2 권장 주 매칭기 1 — Edge-NCC / Chamfer matching (L1)

아이디어는 단순하다 — "박스는 **엣지**다." 두 이미지를 모두 엣지 영상으로
바꿔 놓고 매칭하면 절대 밝기·노이즈·텍스처는 거의 날아간다.

**Edge-NCC 파이프라인:**

```python
def preprocess(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    return mag

def match_edge_ncc(template, fov):
    t = preprocess(template)
    f = preprocess(fov)
    res = cv2.matchTemplate(f, t, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc  # max_val 이 신뢰도, max_loc 이 좌상단 좌표
```

**Chamfer matching (한 단계 더 강함):**

1. 두 이미지에 `cv2.Canny` 로 이진 엣지 맵을 만든다.
2. FOV 엣지 맵에 `cv2.distanceTransform` 을 적용해서 "각 픽셀에서 가장
   가까운 엣지까지의 거리" 를 구한다.
3. 템플릿 엣지를 슬라이딩시키며 **거리 맵 값의 합** 을 계산 — 합이 작을
   수록 엣지 배치가 잘 포개진 위치다.
4. OpenCV 에 `cv2.distanceTransform` 은 있고, chamfer matcher 자체는
   `cv2.createChamferMatcher` (버전에 따라 contrib) 또는 직접 convolve
   로 몇 줄로 구현 가능.

Chamfer 는 **엣지 부분 누락, 끊김, 살짝의 노이즈** 에도 매우 강건해서
"박스 몇 개의 배치" 같은 sparse 구조 매칭에 사실상 표준이다. 반도체
업계의 여러 "address mark" 검출 알고리즘이 이 계열이다.

### 3.3 권장 주 매칭기 2 — Feature matching (L2)

박스의 **코너** 를 keypoint 로 잡고, descriptor 로 비교한 뒤, RANSAC 으로
기하 정합(homography/affine) 을 푸는 방식. 박스 4개만 있어도 코너 16개라
충분하다.

```python
def match_features(template, fov):
    detector = cv2.AKAZE_create()             # ORB 도 OK, 훨씬 빠름
    kp1, des1 = detector.detectAndCompute(template, None)
    kp2, des2 = detector.detectAndCompute(fov,      None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    # inliers / len(good) 이 높으면 강한 매칭 — 0.6 이상을 threshold 로.
    return H, inliers, len(good)
```

장점:

- **회전·스케일·부분 가림 전부 내성**. FOV 가 템플릿의 일부만 보여줘도
  inlier 가 살아 있으면 매칭된다.
- Homography 덕분에 **sub-pixel 위치 + 회전/스케일까지 동시에 나온다.**
  그대로 stage 좌표로 환산할 수 있다.
- `inliers` 수가 자연스러운 confidence.

단점:

- SEM 의 **매끈한 박스 내부** 에는 feature 가 적어 코너 탐지기가 주로
  박스 모서리만 잡는다 → keypoint 수 자체가 적을 수 있다. AKAZE 가 ORB
  보다 잘 잡는 편.
- 템플릿이 **repetitive pattern** (바둑판 등) 이면 descriptor 가 혼동된다.
  이때는 L3 geometric matching 으로 넘어가야 한다.

### 3.4 권장 주 매칭기 3 — Contour / Geometric matching (L3)

"박스 3~4개 배치" 를 *직접* 비교하는 가장 honest 한 접근.

1. 두 이미지에 CLAHE + Gaussian blur + Otsu 이진화 (`cv2.threshold
   ..., cv2.THRESH_BINARY | cv2.THRESH_OTSU`).
2. `cv2.findContours` 로 외곽선 추출.
3. 면적 / 종횡비 / `cv2.approxPolyDP` 꼭짓점 4개 조건으로 "큰 박스" 만
   필터링.
4. 템플릿에서 박스 N개, FOV 에서 박스 M개가 나오면, **두 박스 집합 간
   기하 정합** 을 푼다:
   - 각 박스의 중심을 점군으로 보고, N 점과 M 점 사이의 최적 매칭을
     RANSAC + affine fit 로 찾는다 (박스 수가 4개 내외면 조합 탐색도
     충분히 빠르다).
   - 박스 크기 비율과 상대 거리 비율까지 비교하면 false positive 가
     크게 줄어든다.

이 방식의 장점은 **스케일·회전·밝기 전부 무관** 하고, 결과가 "박스
배치가 동일한가" 에 대한 **해석 가능한 증거** (N개 박스 중 K개가
matched) 로 나온다는 것이다. 단점은 이진화가 실패하면 (FOV 콘트라스트가
너무 낮거나 charging 이 심하면) contour 자체가 무너진다는 것. 그래서
preprocessing (CLAHE, 필요하면 adaptive threshold) 이 중요하다.

### 3.5 권장 실전 조합

하나만 고르지 말고 **짧은 파이프라인** 으로 쓴다.

```
FOV 캡처
   │
   ├── (fast) raw NCC coarse  → top-3 후보 위치
   │
   ├── 각 후보 근방 crop
   │     ├── Edge-NCC / Chamfer  → score_struct
   │     └── AKAZE + RANSAC      → inliers
   │
   └── 최고 후보에서:
         • Contour-based 박스 개수/배치 교차확인
         • (선택) VLM 에 "두 패치가 같은 fiducial 인가?" 최종 veto

결정:
  score_struct ≥ HIGH  AND  inliers ≥ K    → 성공, 위치 반환
  그 외                                      → 실패 → stage 이동
```

각 단계가 서로 다른 가정을 검증하므로 하나가 무너져도 나머지가 잡아낸다.
각 단계의 수치는 실제 데이터 10~30 쌍으로 반드시 튜닝해야 한다 (§6).

### 3.6 Preprocessing 체크리스트

세 매칭기 모두 아래 전처리를 공유하는 게 권장된다.

- Grayscale 변환
- **CLAHE** (`clipLimit=2.0, tileGridSize=(8,8)`) — 국소 대비 균등화
- 작은 **Gaussian blur** (3×3) — 샷 노이즈 억제
- (엣지 계열 한정) Sobel magnitude 또는 Canny (`low=50, high=150` 정도
  기본, 이미지에 맞춰 튜닝)
- (이진화 계열 한정) Otsu 또는 adaptive threshold

### 3.7 Sub-pixel / sub-FOV 정확도가 필요한 경우

최종 좌표가 필요한 stage 이동에는 sub-pixel 정밀도가 중요하다.

- NCC 계열은 peak 주변에 **2차 parabola fit** 을 해서 sub-pixel refinement.
- Feature matching 은 homography 자체가 이미 sub-pixel.
- Contour 중심은 `cv2.moments` 로 centroid 까지 sub-pixel.

### 3.8 이 스택의 성질

- **Deterministic** — 같은 입력이면 같은 출력.
- **CPU-only, FOV 당 수~수십 ms.** GPU 불필요.
- **Calibrated confidence** — Edge-NCC score, inlier ratio, matched-box
  count 전부 숫자. Threshold 로 성공/실패 로깅 가능.
- **해석 가능** — 어느 단계가 왜 떨어졌는지 디버그 이미지로 남길 수 있다
  (엣지 맵, matched keypoint drawing, contour 오버레이).

"웨이퍼를 돌아다니며 Key 를 찾을 때까지 탐색한다" 는 문제가 실제로
필요로 하는 신호 — **stage 가 움직일 때마다 단조 증가해야 할 정량적
유사도** — 를 이 스택이 준다. VLM 은 이 숫자를 만들 수 없다.

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
- **Image processing 이 필요한가?** 필요하다. 단, **raw 픽셀 NCC 가 아니라
  "구조 수준" 매칭이 중심**이어야 한다. 레시피 이미지와 SEM monitor
  이미지는 픽셀이 다르지만 "박스 3~4개 배치" 라는 구조는 같다. 권장
  스택은 **Edge-NCC / Chamfer matching (L1) + AKAZE feature matching
  (L2) + Contour 기반 박스 배치 매칭 (L3)** 의 교차 검증 (§3).
- **VLM 이 "유사/비유사" 판정을 할 수 있는가?** 거친 의미 수준에서는
  가능하고, veto / tiebreaker 로는 유용하다. Stage 이동의 핵심 결정자로는
  부적합하다.
- **무엇을 먼저 만들어야 하는가?** 실장비에서 촬영한 (레시피 key 이미지,
  그것이 포함된 FOV) 쌍 10~30 장. 그다음 오프라인에서 Edge-NCC +
  AKAZE + Contour 3종 프로토타입을 돌려 score/inlier/matched-box 분포를
  본다. 그 분포가 threshold 튜닝과 이후 모든 설계의 토대가 된다.
