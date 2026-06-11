# recording_filter — 녹화 프레임 필터링 + 상호작용 타임라인 설계

- **날짜**: 2026-06-11
- **상태**: 승인됨 (구현 대기)
- **대상 패키지**: `poc/workflow_3/recording_filter/` (신규)
- **선행 코드**: `poc/workflow_2/filter_frames_by_change.py`, `poc/workflow_2/vlm_cursor_click_filter.py`

## 1. 배경 / 목적

align fail 사이클은 `monitor/recording.py` 의 `RecordingSession` 으로 tool 창을
상시 녹화한다(성공/실패 무관). 자동 보정이 실패해 엔지니어가 직접 장비를 조작하는
구간까지 같은 세션이 캡처하므로, 이 프레임들이 **다음 개선(절차 분석 / 모방 학습)의
원천 데이터**다.

녹화는 변화 감지 적응 캡처라 idle 중복은 이미 줄지만, 한 세션은 여전히 수백~수천 장
규모다. 이를 (1) **더 적은 수의 프레임으로 필터**하고, (2) 그 위에서 **엔지니어의
마우스/키 입력을 추출**해, 특정 상황(현재는 align fail, 이후 더 많은 상황)에 대한
**워크플로우를 수립할 구조화 데이터**로 만든다.

지금까지 이 기능은 두 개의 오프라인 도구로 흩어져 있었다:

1. `workflow_2/filter_frames_by_change.py` — cv2 absdiff 변화 이벤트 필터 (프레임 축소).
2. `workflow_2/vlm_cursor_click_filter.py` — VLM 커서 탐지 + 변화-근접 검사 (클릭 추출).

이 둘을 workflow_3 의 신규 패키지 `recording_filter/` 로 이전·정리하고, 향후 타이핑
추출까지 확장 가능한 골격을 만든다.

## 2. 범위 (이번 반복)

- **포함 (Stage 1 + Stage 2a)**:
  - Stage 1 프레임 축소(cv2) 를 workflow_3 로 이식 + `change_bbox` 추가.
  - Stage 2a 커서-기하 클릭 탐지(VLM + cv2) 를 이식.
  - 두 산출물(축소 프레임 + 상호작용 타임라인 JSON)을 만드는 단일 실행 엔트리.
  - 합성 데이터 스모크 테스트.
  - align fail 녹화에 대해서만 튜닝/검증 (그러나 아키텍처는 확장 가능하게).
- **연기 (다음 반복, Stage 2b)**: OCR-diff 기반 **타이핑 추출**.
  `timeline.build_timeline()` 이 `typing_events` 인자를 미리 받도록 설계해 추가 시
  재설계가 없도록 한다.
- **불포함**: 워크플로우 자동 수립/합성 (타임라인은 그 입력일 뿐, 이번 범위 아님).
  실시간 사이클(`monitor/cycle.py`) 변경 없음 — 본 도구는 **온디맨드 오프라인**이다.

## 3. 핵심 설계 결정 (브레인스토밍 합의)

| 항목 | 결정 | 이유 |
|---|---|---|
| 실행 시점 | **오프라인 온디맨드** | 실시간 루프 무변경; 수집된 녹화를 나중에 분석 |
| 주 산출물 | **축소 프레임 + 타임라인 JSON (동격)** | 사람이 프레임으로 검토 + 기계가 타임라인 소비 |
| 입력 추출 방식 | **기하 커서(클릭) + OCR-diff(타이핑, 연기)** | 작고 명시적인 두 전용 검출기; VLM 비용↓ |
| 1차 범위 | **Stage 1 + 2a 우선**, 타이핑은 fast-follow | 골격 검증 후 타이핑 추가 |
| 커서 프롬프트 출처 | **workflow_2 파일에 자족적으로 존재** | frozen workflow_1 을 건드릴 필요 없음 |

## 4. 패키지 구조

```
poc/workflow_3/recording_filter/
├─ __init__.py            # 공개 export + 출력 경로 상수 + env override
├─ settings.py            # RecordingFilterSettings dataclass + load_recording_filter_settings()
├─ frame_reduce.py        # STAGE 1 (순수 cv2): 프레임 → ChangeEvent[]
├─ click_detect.py        # STAGE 2a (VLM+cv2): ChangeEvent[] → ClickEvent[]
├─ cursor_prompt.py       # 커서 coarse 탐지 system/user 프롬프트 (workflow_2 에서 이식)
├─ timeline.py            # ClickEvent[] (+미래 TypingEvent[]) → InteractionEvent[] + 오버레이 프레임
├─ filter_recording.py    # 엔트리포인트: 입력 해석 → 단계 실행 → 산출물 기록
├─ text_diff.py           # (연기) STAGE 2b OCR-diff 타이핑 검출
├─ test_frame_reduce.py
├─ test_click_detect.py
└─ test_timeline.py
```

**단위 경계 (각 모듈의 한 가지 책임):**

- `frame_reduce` — 순수 CV. 네트워크/VLM 없음. 입력: 프레임 디렉터리. 출력: `ChangeEvent[]`.
  bench 대비 **변경점**: 가장 큰 변화 blob 의 위치 bbox(`change_bbox`)를 함께 산출한다.
  (현재 bench 는 면적만 계산하고 위치를 버린다. 이 bbox 는 지금은 클릭 ROI 시드로,
  나중엔 OCR crop 영역으로 두 번 쓰인다 — Stage 1 에서 한 번 잡아 둔다.)
- `click_detect` — 생존 프레임에 대해서만 VLM 커서 탐지 → 커서 중심 정사각 ROI 안의
  변화 픽셀 카운트 → 임계 이상이면 클릭. 입력: `ChangeEvent[]` + settings + VLM client.
  출력: `ClickEvent[]`.
- `timeline` — 이벤트들을 시간순 `InteractionEvent[]` 로 병합·직렬화하고 오버레이
  프레임(커서 bbox + ROI 박스)을 기록. 타이핑 이벤트 인자를 미리 받는다.
- `filter_recording` — 오케스트레이터(엔트리). 입력 디렉터리 해석 → Stage1 → Stage2a
  → timeline → 산출물 + summary 기록. 상태 문자열 반환. CLI 인자 없음.
- `settings` — env 주도 파라미터 dataclass (bench 의 튜닝 상수 이식).

## 5. 데이터 흐름 / 자료구조

```
recording/<tag>_rcs_<seq>_<elapsed_ms>ms.jpg        (원본 적응 캡처 프레임)
   │
   ▼  frame_reduce.reduce_frames()                  STAGE 1 (cv2, 인접쌍 전체)
ChangeEvent[]
   │
   ▼  click_detect.detect_clicks()                  STAGE 2a (VLM 커서 + cv2 ROI, 생존만)
ClickEvent[]
   │
   ▼  timeline.build_timeline()
InteractionEvent[]
```

**ChangeEvent** (Stage 1 산출):
```
rank: int                  # 0부터, 생존 순서
frame_path: str            # 현재(curr) 생존 프레임 절대경로
prev_frame_path: str       # 직전 프레임
timestamp_sec: float       # 파일명 <elapsed_ms> 에서 복원
frame_index: int           # 파일명 seq (없으면 -1)
change_bbox: {left,top,right,bottom}   # 가장 큰 변화 blob 의 native px bbox  ← 신규
largest_blob_area_px: int
changed_pixels: int
```

**ClickEvent** (Stage 2a 산출, ChangeEvent 확장):
```
... (ChangeEvent 필드) ...
is_click: bool
cursor_visible: bool
cursor_kind: str | null    # dvr_x | rcs_black_arrow | rcs_white_arrow | unknown
cursor_bbox: {left,top,right,bottom} | null   # native px
cursor_xy: [x,y] | null    # 커서 중심
click_window: {left,top,right,bottom}         # 커서 중심 정사각 ROI
changed_in_window_px: int
confidence: float
evidence: str
status: "click" | "no_click" | "cursor_unavailable"
```

**InteractionEvent** (타임라인, 통합 — 미래 타이핑과 공용 스키마):
```
t_sec: float
seq: int
action: "click"           # 미래: "type"
coords: {x,y} | null      # click 좌표 (커서 중심)
element: null             # 미래: VLM/OCR 로 식별한 요소 라벨
text: null                # 미래: 타이핑 텍스트
confidence: float
frame: str                # 오버레이 프레임 상대경로
source_frames: {prev, curr}
```

## 6. 입력 / 출력 계약

**입력 해석 (CLI 인자 없음 — 프로젝트 규칙):**
1. `RECORDING_FILTER_INPUT_DIR` env → 해당 `recording/` 디렉터리.
2. 없으면 모듈 상수 `INPUT_DIR_OVERRIDE = ""` (bench 의 `*_OVERRIDE` 선례).
3. 없으면 자동 선택: `ALIGN_IMAGES_DIR` 아래
   `*/*/*/captured_img_from_rcs/*/recording/` 중 mtime 최신.
- `recording/` 폴더는 `<tag>_rcs_*ms.jpg` 를 직접 담는다. bench 호환을 위해 `frames/`
  하위 폴더가 있으면 그쪽도 허용한다.

**출력 위치:**
- 기본: 입력의 형제 `recording_filter/` → `<tag>/recording_filter/`.
- `RECORDING_FILTER_OUTPUT_DIR` env override.
- dev 폴백: 입력이 `align_images` 밖이면 `debug_images/recording_filter/<tag>/`.

**산출 아티팩트 (`<tag>/recording_filter/`):**

| 아티팩트 | 내용 | 역할 |
|---|---|---|
| `change_events/` | Stage 1 생존 프레임의 rank 접두 사본 | "더 적은 수의 이미지" |
| `click_events/` | 클릭으로 분류된 프레임의 오버레이 JPEG (커서 bbox + ROI) | 주석 증거 |
| `interaction_timeline.json` | 정렬된 `InteractionEvent[]` + 실행 메타데이터 | 구조화 타임라인 |
| `change_events.json` | Stage 1 전체 상세 (bench 와 동등) | 감사 |
| `summary.json` | 카운트/파라미터/elapsed/vlm_calls/입출력 경로/truncated 플래그 | 실행 기록 |

## 7. 설정 — `RecordingFilterSettings`

env 주도 dataclass. bench 의 튜닝 상수를 이식한다.

```
# Stage 1
diff_threshold: int = 25            # FILTER_DIFF_THRESHOLD
resize_width: int = 1280            # FILTER_DIFF_RESIZE_WIDTH
min_change_area_px: int = 5000      # FILTER_MIN_CHANGE_AREA_PX
# Stage 2a
cursor_click_window_px: int = 200   # CURSOR_CLICK_WINDOW_PX (ROI 한 변)
click_min_changed_px: int = 1500    # CLICK_MIN_CHANGED_PX
click_diff_threshold: int = 25      # CURSOR_FILTER_DIFF_THRESHOLD
# VLM
vlm_service: str = "ui-venus"
vlm_model: str = UI_VENUS_MODEL_NAME
vlm_request_delay_sec: float = 1.0  # 프록시 과부하 방지 간격
max_vlm_calls: int = 0              # 0 = 생존 전체 처리 (샘플링 없음)
```

**bench 대비 의도된 행위 변경:** `vlm_cursor_click_filter.py` 는 5장을 random
sample 하는 *프로브*다. 본 production 도구는 기본적으로 **생존 전체**를 처리한다
(`max_vlm_calls=0`). 캡에 걸리면 `summary.json` 에 `truncated: true` + 스킵 수를
기록한다 — 부분 실행을 전체 커버리지로 오인하지 않도록 **무음 절단 금지**.

## 8. 에러 처리

- 프레임 `< 2` → 상태 `not_enough_frames`, 정상 종료(크래시 없음).
- 프레임 로드 실패 → 해당 쌍 스킵, `[WARNING]`, 계속.
- VLM 호출 실패/타임아웃 → catch, 해당 생존을 `cursor_unavailable` 로 표시.
  `change_events/` 에는 남지만 클릭으로는 방출하지 않음. 실행은 계속(한 프레임이
  전체 패스를 죽이지 않음). `vlm_request_delay_sec` 간격 유지.
- `max_vlm_calls` 캡 도달 → 이후 분류 중단(절단을 명시 로그), 남은 생존은 클릭
  라벨 없이 `change_events/` 에만 둔다.
- **읽기 전용**: 마우스/키보드 출력이 전혀 없으므로 `SAFE_MODE` 게이트 없음.
  Mac/dev 에서 복사한 녹화에 자유롭게 실행 가능(순수 분석 + VLM 호출).
- 모든 산출은 기존 헬퍼(`save_debug_json`, `save_marked_bboxes`)를 통해 기록.

## 9. 테스트 (합성 데이터, 네트워크 없음 — repo 스모크 패턴)

- `test_frame_reduce`: numpy 로 변화 blob 을 주입한 N 프레임 → 생존/`change_bbox`
  정확성, 정적 쌍 탈락, `<2` → `not_enough_frames`.
- `test_click_detect`: **가짜 VLM client 주입**(고정 커서 bbox 반환) + 합성 diff
  마스크(ROI 안/밖 변화) → `is_click` True/False, `cursor_visible=false` → 비클릭,
  VLM 예외 → `cursor_unavailable` + 실행 생존.
- `test_timeline`: 순서 뒤섞인 rank → 정렬 출력, JSON 스키마 키 존재, 오버레이
  프레임 경로 유효.

## 10. 선행 코드 처리 (이전 = move)

- 엔진을 `workflow_3/recording_filter/` 로 이식.
- **삭제**: `workflow_2/filter_frames_by_change.py`, `workflow_2/vlm_cursor_click_filter.py`
  (대체됨). 벤치가 프레임 축소가 필요하면
  `from poc.workflow_3.recording_filter import reduce_frames` 로 import
  (bench→workflow_3 의존 규칙 준수).
- **frozen `workflow_1` 은 손대지 않음** — 커서 프롬프트가 workflow_2 파일에
  자족적으로 존재하므로 frozen 아카이브를 건드릴 필요가 없다.
- 이식 전 `grep` 으로 두 파일을 import 하는 다른 사용처가 없는지 확인한다.

## 11. 향후 확장 (이번 범위 밖, 설계만)

- **Stage 2b — 타이핑(OCR-diff)**: `change_bbox` crop 에 PaddleOCR-VL 을 적용해
  필드 텍스트 변화를 검출 → `TypingEvent[]`. **전체 스크린샷 OCR 금지**(환각),
  반드시 변화 영역 crop 후 인식. `build_timeline(typing_events=...)` 으로 합류.
- **요소 식별**: 클릭 좌표 위의 UI 요소 라벨을 VLM/OCR 로 채워 `element` 필드 완성.
- **워크플로우 합성**: 여러 세션 타임라인을 묶어 상황별 절차를 일반화 (별도 설계).

## 12. 미해결 / 가정

- VLM client 는 선행 코드가 쓰는 `poc.workflow_3.vlm.vlm_client.Workflow1VLMClient`
  를 그대로 사용한다고 가정(이식 시 workflow_3 컨벤션과 정합 확인).
- 커서 프롬프트의 3-변형(DVR-X / RCS-black / RCS-white) 정의를 그대로 유지.
- align fail 외 상황의 녹화 레이아웃이 동일 파일명 규약(`*_rcs_*ms.jpg`)을 따른다고 가정.
