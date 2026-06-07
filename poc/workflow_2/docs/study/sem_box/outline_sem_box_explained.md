# Live SEM Box 외곽선 그리기 — CV 입문자용 해설

> `poc/workflow_2/outline_live_sem_box.py` 동반 문서입니다.
> CV(computer vision)를 처음 접하는 분을 위해 작성했습니다. 모든 단계가 그 파일의 코드 한 줄에 대응됩니다.

---

## 0. 무슨 문제를 푸는가?

RCS 스크린샷(`captured_img_from_rcs/<tag>/<tag>_rcs.jpg`) 안에는 wafer 의 **live SEM 영상**을
보여주는 큰 사각형 하나가 있습니다. 이후의 모든 동작 — align mark 클릭, zoom, "이 프레임은 너무
흐려서 클릭하면 안 됨" 판단 — 은 전부 *그 사각형의 정확한 픽셀 좌표*를 필요로 합니다. box 가 20px
어긋나면 뒤따르는 모든 클릭이 20px 어긋납니다.

그래서 이 스크립트의 단 하나의 임무는 **live SEM box 의 외곽선을 정밀하게 그리는 것**입니다.

방식은 **2-pass(2단계)** 입니다:

```
RCS 스크린샷
   │
   ├─ [Pass 1] VLM   → 대략적인 사각형 (어느 영역이 SEM 영상인지 "압니다")
   │
   └─ [Pass 2] CV    → 네 변을 실제 경계선 픽셀에 snap (정밀하게)
                       + sharpness 측정 (클릭하기엔 너무 흐린가?)
```

한 줄 슬로건은 **"VLM 이 제안하고, CV 가 확정한다(VLM proposes, CV disposes)"** 입니다. AI 모델은
화면을 *이해*하는 데 강하고("'Optics' 라벨 아래의 노이즈 낀 회색 사각형이 live 영상이다"), classical
CV 는 edge 가 정확히 어디 있는지 *측정*하는 데 강합니다. 즉 각자 잘하는 일을 맡기는 것입니다.

---

## 1. 반드시 체화해야 할 단 하나의 개념: 이미지는 숫자 격자입니다

grayscale 이미지는 밝기 값의 2D 표(matrix)일 뿐입니다. `0` = 검정, `255` = 흰색입니다. 600×800
이미지는 600행 × 800열의 숫자입니다.

```
        col 0   col 1   col 2  ...
row 0 [  43      41      40   ... ]
row 1 [  42      45      210  ... ]   ← 210 은 밝은 픽셀입니다
row 2 [  44      40      215  ... ]
```

아래의 모든 것(edge, gradient, blur)은 **이 격자에 대한 산수**입니다. 마법은 없습니다.

코드에서는 이렇게 읽습니다:

```python
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)   # → numpy array, shape (height, width)
```

`gray[y, x]` 는 y행 x열의 밝기입니다.

---

## 2. "edge(경계)" 란 무엇인가?

**edge** 는 밝기가 *갑자기* 변하는 곳입니다. SEM box 의 테두리가 곧 edge 입니다 — 한쪽은 어두운
wafer 영상, 다른 쪽은 밝은 UI 프레임이죠. 그 경계를 가로지르면 숫자가 점프합니다.

```
box 의 왼쪽 테두리를 가로지르는 한 가로 스캔라인의 밝기:

  220 218 219 | 40 42 41 39 ...
              ^
              경계: ~219 에서 ~40 으로 큰 점프
```

평평한 영역은 거의 변하지 않습니다(`40 41 40 42 39`). edge 는 크게 변합니다(`219 → 40`). **CV 는
이웃 픽셀 간의 큰 변화를 찾아 edge 를 검출합니다.**

---

## 3. Gradient 와 Sobel operator (핵심 도구)

한 픽셀에서의 "변화량"을 **gradient** 라고 합니다. 이미지판 *기울기(slope, 미분)*인 셈으로, 어떤
방향으로 한 걸음 움직일 때 밝기가 얼마나 빠르게 오르내리는지를 뜻합니다.

두 방향이 중요합니다:

| Gradient | 무엇을 재는가 | 어떤 edge 를 밝히는가 |
|----------|--------------|----------------------|
| `grad_x` = **x**(좌↔우) 방향 변화 | 옆으로 갈 때 밝기가 얼마나 빨리 변하는지 | **세로(vertical)** edge (box 의 좌/우 벽) |
| `grad_y` = **y**(상↕하) 방향 변화 | 아래로 갈 때 밝기가 얼마나 빨리 변하는지 | **가로(horizontal)** edge (box 의 상/하 벽) |

> **왜 교차될까요?** *세로* 벽은 왼쪽이 밝고 오른쪽이 어두운 픽셀 열이라, 변화는 **가로로**
> 가로지를 때 일어납니다. 그래서 세로 edge 는 가로 변화(`grad_x`)로 검출하고, 가로 edge 는
> 그 반대입니다. 입문자가 꼭 헷갈리는 부분이니 두 번 읽으시길 권합니다.

**Sobel operator** 는 이 gradient 를 계산하는 표준 레시피입니다. 작은 3×3 가중치 stencil 을 모든
픽셀 위로 미끄러뜨리며 이웃을 조합해 기울기를 추정합니다. 정확한 가중치를 외울 필요는 없습니다 —
`cv2.Sobel(...)` 이 픽셀마다 "여기 변화가 얼마나 세고 어느 방향인지"를 준다는 것만 알면 됩니다.

코드는 다음과 같습니다:

```python
grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))   # (1,0)=x 미분 → 세로 edge
grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))   # (0,1)=y 미분 → 가로 edge
```

- `(1, 0)` 은 "x 방향 미분", `(0, 1)` 은 그 반대를 뜻합니다.
- `cv2.CV_32F` 는 부동소수점으로 계산하라는 의미입니다(기울기는 음수일 수 있습니다 — 어두움→밝음
  vs 밝음→어두움).
- `np.abs(...)` 는 변화의 *크기*만 신경 쓰고 부호는 보지 않겠다는 뜻입니다. 벽은 밝음→어두움이든
  그 반대든 벽입니다. `abs` 후에는 큰 값 = 강한 edge, ~0 = 평평입니다.

이 연산 후, `grad_y` 는 box 의 상/하 테두리가 환하게 빛나는 새 격자가 되고(긴 가로 edge),
`grad_x` 는 좌/우 테두리가 빛나는 격자가 됩니다.

---

## 4. Pass 1 — VLM 이 대략적인 사각형을 줍니다

CV 가 경계에 snap 하려면 *대략 어디를 봐야 하는지* 먼저 알아야 합니다. 그것이 VLM 의 일입니다
(`_run_sem_box_detection`, 프롬프트는 `vlm_sem_monitor_box.py` 의 것을 재사용합니다).

VLM 은 사람처럼 화면을 읽습니다: 상단의 `Optics`/`OM` 텍스트 라벨을 찾고, 노이즈 낀 회색
live-video 질감을 알아보고, 그 위에 겹쳐 그려진 floating control panel 들을 인지한 뒤 사각형을
반환합니다.

**좌표에 대한 중요한 디테일이 있습니다:** 모델은 box 를 픽셀이 아니라 **0–1000 격자**로 반환합니다.
이미지를 가로·세로 1000 단위로 상상하고 예를 들어 `{left: 210, top: 150, right: 800, bottom: 760}`
이라고 말합니다. 우리는 이것을 실제 픽셀로 변환합니다:

```python
bbox_1000 = normalize_bbox_1000(parsed["panel_bbox"])      # 0–1000 box 검증
bbox_px   = bbox_1000_to_pixels(bbox_1000, width, height)  # 실제 픽셀 좌표로 스케일
```

이것이 *VLM box 가 "대략적"인 이유*입니다. 1920px 폭 스크린샷에서 1 격자 단위 ≈ 2px 인데, 모델은
눈대중한 edge 로 반올림하므로 보통 **5–15px 어긋납니다**. "어디를 볼지"에는 충분하지만 클릭
기준으로는 부족하며, 바로 그 간극을 Pass 2 가 메웁니다.

---

## 5. Pass 2 — 각 변을 실제 edge 로 snap (`_snap_box_to_edges`)

이제 대략적인 box 와 두 개의 gradient map 이 있습니다. **네 변을 각각 독립적으로** 다듬습니다.
**top(위) 변**을 예로 들겠습니다.

### 5a. search band — 짧은 목줄

top 경계를 찾으려고 이미지 전체를 뒤지지 않습니다(엉뚱한 다른 선에 걸릴 위험이 있습니다). VLM 의
추정값 *주변의 얇은 띠* 안에서만 봅니다:

```python
band_y = max(EDGE_SNAP_BAND_MIN_PX, int(round(box_h * EDGE_SNAP_BAND_RATIO)))
# EDGE_SNAP_BAND_RATIO = 0.06  → box 높이의 ±6% 안에서 탐색, 최소 ±6px
```

VLM 이 top 을 150행이라 했고 box 높이가 600px 라면, `150 ± 36` 행만 살핍니다. 이 band 가 안전
목줄입니다: CV 는 VLM 을 수십 px 보정할 수 있지만 엉뚱한 edge 로 도망갈 수는 없습니다.

### 5b. projection 기법 — edge 를 따라 합산하기

band 안에서 어느 행이 *진짜* top 경계일까요? 경계는 **긴 가로 선**이므로 바로 그 행에서는 box
*폭 전체*에 걸쳐 `grad_y` 가 강한 반면, box 내부의 노이즈/질감은 흩어진 단일 픽셀에서만 강합니다.

그래서 각 후보 행마다 **box 폭에 걸쳐 `grad_y` 를 더합니다**:

```python
strength = grad_y[lo:hi, left:right].sum(axis=1)   # 후보 행마다 합계 하나
return lo + int(np.argmax(strength))               # 합계가 가장 큰 행 = 경계
```

각 행에 빛을 통과시켜 "edge 가 얼마나 쌓이는지" 재는 것으로 생각하면 됩니다:

```
band 안의 후보 행들:        box 폭에 걸친 grad_y 합
   row 146  ──────────►    1,240   (약간의 노이즈)
   row 147  ──────────►    1,310
   row 148  ──────────►    9,870   ◄── 진짜 경계! 연속된 선은 거대한 합을 쌓습니다
   row 149  ──────────►    1,510
   row 150  ──────────►    1,090
```

`np.argmax` 는 합이 가장 큰 행을 고를 뿐입니다 → 148행입니다. top 변을 거기로 *snap* 합니다.

> **합산이 통하는 이유(핵심 통찰):** 진짜 경계는 *coherent(일관)*합니다 — 길이 전체에 걸쳐
> 강합니다. 무작위 질감은 *incoherent(비일관)*합니다 — 여기저기 강하지만 합하면 평균으로 묻힙니다.
> 후보 선을 따라 합산하면 진짜 경계는 증폭되고 노이즈는 억제됩니다. 이 스크립트에서 가장 중요한
> 한 가지 아이디어입니다.

### 5c. 나머지 세 변 — 같은 아이디어를 회전

- **bottom 변:** 동일하며, VLM 의 bottom 행을 중심으로 합니다.
- **left & right 변:** 이제 *세로* 선을 원하므로 `grad_x` 를 쓰고, box *높이 방향으로* 합산
  (`axis=0`)해 각 후보 **열(column)**을 채점합니다:

```python
strength = grad_x[top:bottom, lo:hi].sum(axis=0)   # 후보 열마다 합계 하나
return lo + int(np.argmax(strength))
```

네 번의 독립 snap → 다듬어진 네 변 → 딱 맞는 사각형이 됩니다.

### 5d. crossing guard — 불가능한 box 를 만들지 않기

band 에 진짜 edge 가 없으면(예: 완전히 흐린 프레임) `argmax` 는 그래도 *무언가*를 반환하고, 드물게
snap 된 bottom 이 snap 된 top 보다 위로 갈 수 있습니다. 이는 말이 안 되는(뒤집힌) 사각형이며,
이를 다음과 같이 방지합니다:

```python
if new_bottom <= new_top:      # 뒤집힘 → snap 을 불신
    new_top, new_bottom = top, bottom   # VLM 값으로 폴백
if new_right <= new_left:
    new_left, new_right = left, right
```

방어 코드입니다: CV 가 명백히 혼란스러우면 쓰레기를 내놓느니 VLM 의 대략 box 를 믿습니다.

---

## 6. sharpness / blur 측정 (`_sharpness_in_box`)

일부 캡처는 *완전히 흐려서* 클릭하면 **안 됩니다**(대신 zoom-out 하거나 이동합니다). "초점이
맞았다" vs "뭉개졌다"를 말해줄 숫자가 필요합니다.

표준적인 싼 기법이 **Laplacian 의 분산(variance of the Laplacian)**입니다.

- **Laplacian** 은 2차 미분 필터로, 모든 방향의 *미세 디테일과 edge*에 반응합니다. 선명한
  이미지는 또렷한 edge 로 가득 차서 Laplacian 에 큰 값이 많은 반면, 흐린 이미지는 뭉개진 완만한
  변화라 Laplacian 이 어디서나 작습니다.
- **분산(variance)** 은 "이 값들이 얼마나 퍼져 있는가"를 잽니다. 선명하면 넓게 퍼지고(높은 분산),
  흐리면 전부 0 근처라 작은 분산이 됩니다.

```python
roi = gray[top:bottom, left:right]                 # SEM box 내부만
return float(cv2.Laplacian(roi, cv2.CV_64F).var()) # focus 숫자 하나
```

테두리 주변의 또렷한 UI 텍스트에 속지 않도록 **snap 된 box 내부**에서 계산합니다. 그 다음:

```python
blurry = sharpness < SHARPNESS_BLUR_THRESHOLD   # 60.0, 콜드스타트 추정값
```

임계값 미만이면 overlay 에 "BLURRY (do NOT click)" 로 표시됩니다. **60 은 추정값이고 실데이터로
보정해야 합니다** — JSON 이 모든 값을 기록하므로 실제 분기점을 직접 고를 수 있습니다.

---

## 7. overlay 그리기

마지막으로 사람이 판단할 수 있게 찾은 것을 그립니다:

- **magenta(자홍)** 사각형 = VLM 의 대략 box (`VLM coarse`).
- **cyan(청록)** 사각형 = CV-snap 된 정밀 box (`CV snapped`) + 중심 십자.
- 상단 배너에 sharpness 숫자와 `sharp (clickable)` / `BLURRY (do NOT click)`.

cyan box 가 실제 경계를 딱 감싸고 magenta 가 그 주변에 헐겁게 걸쳐 있으면, 전체 파이프라인이
설계대로 동작하고 있다는 뜻입니다.

---

## 8. 출력 읽기 — `summary.json` 의 숫자가 각각 무슨 뜻인가

매 실행은 `debug_images/outline_live_sem_box/<tag>/summary.json` 을 씁니다. **최상위**(실행 전체
합계)와 **`reports` 리스트**(이미지당 항목 하나)로 나뉩니다. 실제 형태 예시는 다음과 같습니다:

```json
{
  "tag": "260528_091500",
  "capture_count": 37,
  "processed": 37,
  "vlm_detected": 34,
  "blurry": 5,
  "sharpness_threshold": 60.0,
  "reports": [
    {
      "image_path": ".../captured_img_from_rcs/260528_084210/260528_084210_rcs.jpg",
      "width": 1920,
      "height": 1080,
      "vlm_detected": true,
      "vlm_bbox": {"left": 405, "top": 168, "right": 1530, "bottom": 905},
      "cv_bbox":  {"left": 412, "top": 173, "right": 1521, "bottom": 898},
      "mode_label": "Optics",
      "vlm_confidence": 0.86,
      "sharpness": 214.7,
      "blurry": false,
      "overlay_path": ".../260528_084210_rcs_outline.jpg"
    }
  ]
}
```

### 최상위 필드 (실행 성적표)

| 필드 | 의미 | 읽는 법 |
|------|------|---------|
| `tag` | 이 실행의 타임스탬프 id (`YYMMDD_HHMMSS`) | 출력 폴더 이름이며, 실행들을 구분합니다 |
| `capture_count` | 찾아서 투입한 `*_rcs.jpg` 장수 | 캡처 수와 같아야 합니다(예: 37) |
| `processed` | 크래시 없이 실제 처리된 장수 | `processed < capture_count` 면 일부 디코드 실패이니 콘솔 `[ERROR]` 를 확인합니다 |
| `vlm_detected` | VLM 이 box 를 반환한 이미지 수 (`panel_visible=true`) | **검출률입니다.** `34/37` 은 34장에서 box 를 찾은 것입니다. 낮으면 프롬프트/모델이 box 를 놓치는 것입니다 |
| `blurry` | sharpness 임계값 미만인 이미지 수 | **클릭하면 안 되는 프레임 수입니다**(zoom-out/이동) |
| `sharpness_threshold` | 이번 실행에 쓴 컷오프 (`SHARPNESS_BLUR_THRESHOLD`, 기본 60.0) | 어떤 임계값이 `blurry` 수를 냈는지 기록합니다 — 바꾸면 수가 달라집니다 |

> 빠른 건강 체크입니다: `processed == capture_count` 이고, `vlm_detected` 가 `processed` 에
> 근접하며, `blurry` 수가 "흐린 캡처" 체감과 일치하면 정상입니다.

### 이미지별 필드 (`reports[]` 안)

| 필드 | 의미 | 읽는 법 |
|------|------|---------|
| `image_path` | 원본 캡처의 절대 경로 | 이 행이 설명하는 파일입니다 |
| `width`, `height` | 그 스크린샷의 픽셀 크기 | bbox 가 사는 좌표 공간입니다(`0..width`, `0..height`) |
| `vlm_detected` | 이 이미지에서 VLM 이 box 를 반환했는가? (`true`/`false`) | `false` 면 아래가 전부 `null` 입니다 — VLM 이 box 를 못 본 것입니다 |
| `vlm_bbox` | VLM 의 **대략(coarse)** box, **픽셀** `{left,top,right,bottom}` | overlay 의 magenta box 입니다. 설계상 헐겁습니다 |
| `cv_bbox` | **edge-snap 된 정밀** box, 픽셀 | cyan box 입니다. **이후 모든 단계가 믿는 좌표 프레임입니다.** `vlm_bbox` 와 약간 다르면 CV 가 다듬은 것이고, 숫자가 동일하면 crossing-guard 가 폴백한 것입니다(CV 가 자기 snap 을 불신) |
| `mode_label` | VLM 이 box 상단에서 읽은 텍스트 (`Optics`, `OM`, 또는 `null`) | 모니터링 모드입니다. `null` 은 라벨을 못 읽은 것입니다 |
| `vlm_confidence` | VLM 이 자체 보고한 confidence, `0.0–1.0` | **소프트 신호일 뿐입니다.** 낮은 confidence + 이상한 box 면 이 프레임을 불신합니다. 절대 CV 결과를 덮어쓰지 않습니다 |
| `sharpness` | `cv_bbox` 내부의 variance-of-Laplacian | **focus 숫자입니다.** 높을수록 선명합니다. 절대 척도는 없으며 *당신의* 이미지에 상대적이라 임계값을 보정합니다 |
| `blurry` | `sharpness < sharpness_threshold` | 이 프레임의 클릭/금지 판정입니다 |
| `overlay_path` | 그려진 JPEG 저장 위치 | 열어서 box 를 눈으로 확인합니다 |

### 이 숫자들로 시스템을 개선하는 법

- **blur 임계값 보정:** 37개 report 의 `sharpness` 값을 모아, 진짜로 클릭 못 할 만큼 흐린 이미지를
  눈으로 가린 뒤, `SHARPNESS_BLUR_THRESHOLD` 를 선명 군집과 흐림 군집 사이에 둡니다. 지금 값은
  *추정*입니다.
- **VLM 미검출 찾기:** `vlm_detected: false` 인 report 는 프롬프트가 실패한 프레임입니다 — 그
  이미지를 열어 무엇이 다른지 봅니다(overlay panel 이 너무 많이 가렸는지, 특이한 레이아웃인지).
- **snap 실패 찾기:** `cv_bbox == vlm_bbox` 로 정확히 같은 행은 crossing-guard 가 snap 을 거부한
  것입니다 — edge-snap 이 깨끗한 경계를 못 찾은 프레임(대개 흐린 것들)이니 `sharpness` 와 교차
  확인합니다.
- **다듬은 양 측정:** `vlm_bbox` 와 `cv_bbox` 의 픽셀 차이는 CV 단계가 한 일의 양입니다. 항상 ~0
  이면 VLM 이 이미 타이트한 것(또는 band 가 너무 작은 것)이고, 크면 VLM 이 헐겁고 CV 가 정밀도를
  책임지는 것입니다.

---

## 9. 아직 틀릴 수 있는 곳 (개선 여지)

이제 원리를 알았으니, 현재 로직이 취약한 지점들을 정리해 봅니다:

1. **projection 은 직선·축정렬 경계를 가정합니다.** floating control panel 이 box edge 바로 위에
   앉으면, band 안에서 *그것의* edge 가 box edge 를 이겨버릴 수 있습니다.
2. **각 변을 독립적으로 snap 합니다** — 네 변이 합리적 종횡비의 사각형을 이루도록 강제하는 제약이
   없고, snap 은 정수 행/열(sub-pixel 정밀도 없음)입니다.
3. **band 가 고정 6% 입니다.** 어떤 프레임에서 VLM 이 6% 넘게 어긋나면 진짜 edge 가 band 밖이라
   snap 이 닿지 못합니다.
4. **흐린 / VLM 미검출 프레임**은 나쁜 시작 box 를 주고 CV 가 그것을 충실히 노이즈에 snap 합니다.

다음 후보 단계입니다(나중에 가능): 변마다 단일 argmax 행 대신 Hough/RANSAC 직선 *fitting*, 네 변에
걸친 사각형 일관성 제약, sub-pixel edge localization, 강한 edge 가 없으면 넓어지는 적응형 band.

---

## 10. 용어집 (Glossary)

| 용어 | 쉬운 뜻 |
|------|---------|
| **Grayscale** | 밝기 숫자의 격자로 본 이미지(0=검정, 255=흰색) |
| **Edge** | 밝기가 급격히 변하는 곳 |
| **Gradient** | 한 픽셀에서 밝기가 얼마나 빨리/어느 방향으로 변하는지("기울기") |
| **Sobel** | gradient 를 계산하는 표준 3×3 레시피 |
| **`grad_x` / `grad_y`** | x / y 방향 gradient → 세로 / 가로 edge 검출 |
| **Projection(선 따라 합산)** | 행/열을 따라 픽셀 값을 더해 일관된 선을 검출 |
| **`argmax`** | "가장 큰 값의 인덱스를 줘" |
| **Laplacian** | 미세 디테일을 밝히는 2차 미분 필터; focus 측정에 사용 |
| **Variance(분산)** | 숫자 집합이 얼마나 퍼져 있는지 |
| **bbox** | bounding box = `{left, top, right, bottom}` |
| **0–1000 격자** | VLM 의 정규화 좌표 공간(이미지를 1000×1000 으로 취급) |

---

## 11. 한 문단 요약

VLM 이 스크린샷을 보고 live SEM box 의 *대략적인* 사각형을 반환합니다(coarse 한 0–1000 격자라 몇
px 헐겁습니다). 그 다음 classical CV 가 Sobel 로 밝기-gradient map 을 계산하고, 네 변 각각에 대해
VLM 추정 주변의 얇은 band 를 훑으며 **각 후보 행/열을 따라 gradient 를 합산하고, 그 합이 peak 인
곳으로 변을 snap 합니다** — 진짜 경계는 큰 합을 쌓는 긴 일관된 선이고 노이즈는 평균으로 묻히기
때문입니다. box 내부의 variance-of-Laplacian focus 점수가 클릭하기엔 너무 흐린 프레임을 표시합니다.
결과는 이후 모든 단계의 좌표 프레임으로 믿을 수 있는, 픽셀 단위로 딱 맞는 cyan 외곽선입니다.
