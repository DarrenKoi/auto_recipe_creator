# Live SEM Box — 검출 과정과 결과 데이터 다루기

> 동반 코드: `poc/workflow_2/sem_box_detect.py`(검출 코어), `poc/workflow_2/outline_live_sem_box.py`(offline 드라이버).
> 픽셀 단위 CV 알고리즘의 *입문용* 설명은 같은 폴더의 [`outline_sem_box_explained.md`](./outline_sem_box_explained.md) 를 보세요.
> 이 문서는 그 위 단계 — **"어디서 검출하고, 나온 좌표를 어떻게 저장하고 production 에서 어떻게 쓰는가"** — 를 다룹니다.

---

## 0. 한 줄 요약

RCS tool monitor 화면 안의 **live SEM box** 위치를 매번 새로 검출하고, 그 좌표를 **장비(eqp_id)별 기준값(reference)** 으로 저장한다. production 에서는 tool monitor 에 들어갈 때마다 다시 검출해 기준값과 비교함으로써 **box 가 옮겨졌거나(moved) 닫혔는지(closed)** 를 감지해, 잘못된 위치에 클릭하는 사고를 막는다.

---

## 1. 왜 "매번 새로 검출" 인가

박스 위치를 한 번 저장해두고 그대로 믿으면, 사용자가 SEM box 를 옮기거나 닫은 순간부터 이후 모든 클릭이 어긋난다. 그래서 **신뢰의 출처를 "저장된 스냅샷"이 아니라 "지금 화면"** 으로 둔다.

- **저장된 reference** = "정상일 때 박스는 여기 있어야 한다"는 기대값.
- **매 방문 시 검출** = "지금 실제로 박스가 어디 있나".
- 둘을 **비교**해서 일치하면 진행, 어긋나거나 없으면 정지/경고.

---

## 2. 검출 코어 — `sem_box_detect.detect_sem_box()`

검출 로직은 **단일 소스**(`sem_box_detect.py`)에 모았다. offline 캡처 분석(`outline_live_sem_box.py`)과 online RCS 방문 경로가 **같은 함수**를 호출하므로, 임계값을 한 곳에서 튜닝하면 양쪽에 동시에 반영된다.

```python
from poc.workflow_2.sem_box_detect import detect_sem_box

detection = detect_sem_box(pil_image, client)   # pil_image: capture_window() 반환값 또는 Image.open()
```

내부 3단계 (역할 분담: **VLM 은 영역만, CV 가 좌표를 확정**):

| 단계 | 담당 | 하는 일 |
|------|------|---------|
| 1. coarse | VLM(ui-venus) | SEM Monitor Box 를 대략적 bbox 로 제안 (`vlm_sem_monitor_box._run_sem_box_detection`) |
| 2. snap | CV | 네 변을 band 안에서 **프레임 회색((170~190) 무채색) 의 끊김 없는 긴 직선 run** 으로 정렬. 프레임 색이 약하면 Sobel edge peak 로 폴백 (`snap_box_to_edges`) |
| 3. sharpness | CV | box 내부 Laplacian 분산 측정 → `total blur` 면 "클릭 금지" 후보 (`sharpness_in_box`) |

> **핵심 설계**: 좌표의 최종 권한은 항상 CV 에 있다. VLM 의 confidence 는 좌표를 확정하는 데 쓰지 않는다(흔들리는 신호다). 자세한 이유는 입문 문서 참고.

### 반환값 — `SemBoxDetection`

| 필드 | 의미 |
|------|------|
| `detected` | VLM coarse 검출 성공 여부 (실패면 나머지 좌표는 `None`) |
| `bbox_px` | CV 로 확정한 박스, **픽셀** 좌표 |
| `bbox_1000` | 캡처 **창 크기 기준 0–1000 정규화** 좌표 (해상도/창 크기 달라도 비교 가능) |
| `sharpness` / `blurry` | box 내부 Laplacian 분산 / 임계값 미만 여부 |
| `mode_label` | 박스 상단 모드 라벨(Optics / OM 등) |
| `confidence` | VLM coarse confidence (기록용, 좌표 신뢰엔 미사용) |
| `vlm_bbox_px` | snap 전 VLM coarse 박스 (디버그/overlay 용) |

`bbox_1000` 이 **비교의 공용 단위**다. 픽셀 좌표는 창 크기에 묶이지만, 0–1000 정규화는 해상도·DPI 차이를 흡수하므로 저장된 reference 와 현재 검출을 맞대볼 수 있다.

---

## 3. offline 드라이버 — `outline_live_sem_box.py`

`detect_sem_box()` 를 캡처 파일들에 돌려서 **눈으로 검증 + reference 생성**을 하는 도구다. 검출 코어 위에 다음 부가물만 얹는다.

- **overlay JPEG** (`<stem>_outline.jpg`): VLM coarse(magenta) / CV snapped(cyan) / sharpness 배너.
- **greymask 디버그** (`<stem>_greymask.jpg`): 프레임 회색으로 잡힌 픽셀을 초록으로 강조 — 색 band 튜닝 확인용. `RCS_OUTLINE_SAVE_GREY_MASK=0` 으로 끔.
- **`summary.json`**: 캡처별 `OutlineReport` 전체 + 생성된 reference 파일 목록.
- **per-eqp reference**: 아래 4장.

실행:

```bash
uv run python poc/workflow_2/outline_live_sem_box.py
# 입력: ALIGN_IMAGES_ROOT/*/*/*/captured_img_from_rcs/<tag>/<tag>_rcs.jpg
#       (RCS_CAPTURE_DIR 로 임의 폴더 지정 가능)
# 출력: debug_images/outline_live_sem_box/<tag>/  +  sem_box_references/<eqp_id>.json
```

---

## 4. 결과 데이터 저장 — 장비별 reference

### 저장 위치와 키

- 경로: `poc/workflow_2/sem_box_references/<eqp_id>.json`
- 키: **eqp_id** (캡처 경로 `<eqp_id>/<class>/<recipe>/captured_img_from_rcs/...` 에서 추출, 없으면 `ALIGN_EQP_ID` 환경변수).
- `align_images/` 트리는 **읽기 전용 입력**(MES 소유)이라 거기 쓰지 않고, 우리가 만든 파생물은 workflow_2 안에 따로 둔다.

### robust 한 기준값 만들기

한 장만 쓰면 몇 px 흔들릴 수 있으므로, **detected 된 여러 캡처의 좌표별 중앙값(median)** 으로 기준 박스를 만든다. 표본 간 **범위(spread)** 도 함께 적어 신뢰도를 사람이 가늠하게 한다.

> 위치는 sharpness 와 무관하다(프레임은 흐릿해도 그 자리에 있다). 그래서 `blurry` 여부와 상관없이 `detected + bbox` 면 표본으로 쓴다. 중앙값이라 한두 장의 오검출은 자연히 눌린다.

### reference JSON 스키마

```json
{
  "eqp_id": "MCD719",
  "created_tag": "20260528_xxxxxx",
  "coord_system": "relative_1000",
  "coord_note": "캡처된 tool 창 크기 기준 0-1000 정규화 (해상도/창 크기 달라도 비교 가능).",
  "bbox_1000":   {"left": 120, "top": 90, "right": 880, "bottom": 760},   // 표본 중앙값
  "spread_1000": {"left": 6,   "top": 4,  "right": 8,   "bottom": 5},     // 표본 간 범위(작을수록 신뢰)
  "sample_count": 31,
  "ref_window_size": {"width": 1920, "height": 1040},
  "mode_label": "Optics",
  "sharpness_median": 142.0,
  "source_images": ["...", "..."],
  "note": "..."
}
```

- `bbox_1000` = 비교 기준 박스. `spread_1000` 이 크면 표본들이 서로 안 맞는다는 뜻 → 그 reference 는 덜 믿고 tolerance 를 넓혀야 한다.
- 읽기 진입점: `load_sem_box_reference(eqp_id) -> dict | None` (없으면 `None`).

### 운영 메모 — reference 는 "의도적 보정 데이터"

같은 장비라면 누가 캡처해도 정규화 좌표는 비슷하게 나와야 한다(정규화가 해상도 차이를 지우므로). 따라서 reference 는 **공유 1개/장비** 가 맞고, **알려진 정상 화면에서 의도적으로 갱신**한다. 박스가 이미 옮겨진 화면으로 무심코 재생성하면 "비정상"을 "정상"으로 굳혀버려 감지가 깨진다.

---

## 5. production 에서 데이터 다루기 (모니터링) — 설계

> 아직 구현 전. online 검출 진입점은 이미 준비됨(`detect_sem_box` 가 `capture_window()` 의 PIL 이미지를 그대로 받음).

```
RCS 로 tool monitor 방문
        │
        ▼
capture_window(tool_window) → PIL 이미지
        │
        ▼
detect_sem_box(image, client) → SemBoxDetection
        │
        ├── detected == False ───────────────► box 닫힘 추정 → 정지/경고
        │
        ▼
load_sem_box_reference(eqp_id) → reference
        │
        ▼
detection.bbox_1000  vs  reference["bbox_1000"]  비교
        │
        ├── tolerance 내 (예: IoU ≥ 임계, 또는 모서리 거리 작음) ──► 정상 → 진행
        └── tolerance 밖 ───────────────────────────────────────► box 이동 → 정지/경고
```

판정 기준 비교는 **정규화 좌표(`bbox_1000`)** 끼리 한다(해상도 무관). tolerance 는 `spread_1000` 을 참고해 정한다 — 표본이 잘 모인 장비는 빡빡하게, 흔들리는 장비는 느슨하게.

| detection 상태 | 해석 | 조치 |
|----------------|------|------|
| `detected=False` | 박스 안 보임 → 닫힘 가능성 | 정지/경고 |
| `detected=True`, reference 와 멀다 | 박스 이동 | 정지/경고 |
| `detected=True`, reference 와 가깝다 | 정상 | 진행 |
| (참고) `blurry=True` | 위치는 맞지만 신호 없음 | 클릭 금지 — zoom-out/대기 판단 |

---

## 6. 데이터 흐름 한눈에

```
[캡처 이미지 / RCS 창]
        │  detect_sem_box()  (sem_box_detect.py — 공용 코어)
        ▼
   SemBoxDetection  ──(offline)──►  outline_live_sem_box.py
        │                              ├─ overlay/greymask JPEG  (눈으로 검증)
        │                              ├─ summary.json           (캡처별 기록)
        │                              └─ sem_box_references/<eqp_id>.json  (장비 기준값)
        │
        └──(online, 예정)──►  production 모니터
                                load_sem_box_reference(eqp_id) 와 비교
                                → moved / closed / ok 판정
```

---

## 7. 튜닝 포인트 (모두 `sem_box_detect.py`)

| 상수 / 환경변수 | 역할 |
|------|------|
| `GREY_FRAME_LO/HI` · `SEM_BOX_GREY_LO/HI` | 프레임 회색 밝기 band |
| `GREY_FRAME_CHROMA_TOL` · `SEM_BOX_GREY_CHROMA_TOL` | 무채색 판정 채널 편차 허용치 |
| `GREY_FRAME_MIN_FRAC` | 색 단서를 신뢰할 최소 run 길이(변 길이 대비) |
| `EDGE_SNAP_BAND_RATIO` / `_MIN_PX` | snap 탐색 band 폭 |
| `SHARPNESS_BLUR_THRESHOLD` | `total blur` 판정 임계값 |

모두 **콜드스타트값** 이다. 실데이터로 보정할 때는 offline 도구의 `_outline.jpg`/`_greymask.jpg` 와 per-side `edge-snap 방법: T=… B=… L=… R=…` 로그(grey/grad)를 보면서 조정한다.
