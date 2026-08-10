# 엔지니어 수동 조작 녹화 세션 (Manual Recording Session)

- 작성일: 2026-08-10
- 대상: `poc/workflow_3/monitor/manual_record.py` (신규), `poc/workflow_3/recording_filter/` (확장)
- 상태: 설계 승인됨 (구현 계획 대기)

## 1. 배경과 동기

지금 녹화(`monitor/recording.py` 의 `RecordingSession`)는 **align fail 알람이 떠야만**
돈다. `monitor/cycle.py` 의 `start_recording` 스텝에서만 생성되기 때문이다.

하지만 엔지니어의 수동 작업을 관찰하려는 목적에는 알람이 필요 없다. 엔지니어와
"지금부터 녹화하겠다"고 약속하고, 이미 열려 있는 Remote Monitoring 창을 그 자리에서
녹화할 수 있어야 한다. 목표는 **모방 학습/절차 분석에 쓸 수 있는 데이터가 실제로
나오는지 확인**하는 것이다 — 즉 이번 작업의 산출물은 자동화가 아니라 **판단 근거**다.

분석 쪽에도 빈 곳이 있다. `recording_filter` 는 "언제 어디를 클릭했나"까지만 뽑고,
`timeline.py:31` 의 `element` 필드는 예약만 된 채 항상 `null` 이다. workflow 로
변환하려면 **무엇을 눌렀는지**가 있어야 한다.

## 2. 기존 파이프라인이 수동 세션에서 깨지는 지점

| ID | 지점 | 내용 |
|----|------|------|
| G1 | `recording_filter` Stage 2a 전제 | 알람 시점은 장비가 멈춰 화면이 정적이라 "변화 = 사람의 조작"이 성립했다. 수동 작업 중에는 라이브 SEM 영상·스테이지 이동·진행 표시가 계속 갱신되어 **전제가 깨진다** |
| G2 | `util/image_utils.py:17` `capture_window` | 창 핸들이 아니라 **창 rect 의 mss 스크린 그랩**이다. 다른 앱이 위에 뜨면 그 앱이 찍힌다. 알람 사이클은 강제 전면화 후 짧게 캡처해 문제가 없었으나, 수십 분 방치되는 수동 세션에서는 실제로 발생한다 |
| G3 | 영역 지도의 재사용 | 장비 A/B/C 마다 창 위치·크기·패널 배치가 다르고, 세션 도중에도 창이 이동/리사이즈된다. 세션당 한 번 만든 영역 지도를 끝까지 쓰면 어긋난다 |
| G4 | `debug_artifacts.py:35` `save_debug_jpeg` | quality=95 고정. 단발 캡처에는 문제없지만 연속 녹화에서는 시간당 수 GB가 된다 |
| G5 | `config.py:74` `recording_max_sec=900` | 알람 사이클 기준값이라 수동 세션의 상한으로 쓰기엔 의미가 다르다 |

## 3. 결정 사항 (브레인스토밍 합의)

| 항목 | 결정 | 근거 |
|------|------|------|
| 접속 | **하지 않는다.** 이미 열린 창에 붙는다 | 엔지니어가 직접 툴을 열고 작업하는 시나리오. EQP 지정 접속은 불필요 |
| 장비 식별 | 창 제목에서 추출해 폴더명에 반영 | `"Remote Monitoring System - <EQP>"` (`login_rcs_common.py:25`) |
| 입력 신호 | **픽셀 추론만.** pynput 입력 후킹 없음 | 엔지니어 입력을 후킹하지 않는다. 커서는 RCS 가 화면 콘텐츠로 그려주므로 프레임에서 관측 가능 |
| 요소 라벨링 | OCR 먼저, 실패 시 VLM 폴백 | 텍스트 버튼에서 VLM 비용 회피. repo 규약(VLM 식별 / OCR 확인)과 일치 |
| 분석 실행 | 런처는 **녹화만**. 분석은 `filter_recording.py` 별도 실행 | 엔지니어 옆에서 무거운 VLM 배치를 돌리지 않는다. 책임 분리 |
| 시간 상한 | **600초(10분)로 시작**, 확인 후 점차 확대 | 실패해도 엔지니어 시간을 10분만 쓴다 |

### 3.1 사이드카 메타데이터 (승인됨)

프레임 저장 시마다 창 rect · 전면 창 제목 · 가림 여부 · 로컬 커서 좌표를
`frame_meta.jsonl` 에 남긴다. **입력 후킹이 아니라 좌표 폴링**이며 키 입력은 일절
기록하지 않는다. `MANUAL_RECORD_META=0` 으로 끌 수 있다.

## 4. 구조

두 부분으로 나눈다. 녹화는 엔지니어 옆에서 도는 가벼운 프로세스, 분석은 나중에
수십 분 돌아도 되는 배치다.

### 4.1 런처 — `poc/workflow_3/monitor/manual_record.py` (신규)

```
열린 Remote Monitoring 창 탐색 (find_window_by_title_prefix)
  ├ 0개  → 안내 후 종료 ("툴 창을 먼저 열어주세요")
  ├ 1개  → 채택
  └ 2개+ → 목록 출력 + MANUAL_RECORD_EQP_ID 로 지정 요구 (임의 선택 금지)
        ↓
제목에서 EQP 추출 → 폴더명 정규화
        ↓
RecordingSession 시작 (out: align_images/<EQP>/_manual/<tag>/recording/)
  + capture_fn 래퍼가 프레임마다 frame_meta.jsonl 1줄 append
        ↓
감시 루프 (5초 주기): 프레임 수 / 디스크 상한 확인 → 초과 시 session.stop()
        ↓
Ctrl+C / 창 닫힘 / max_sec / 예산 초과 → stop() → 경로 안내 후 종료
```

`RecordingSession` 은 **수정하지 않는다.** 사이드카는 생성자의 `capture_fn` 주입점
(`recording.py:88`)을 이용해 감싼다. 예산 감시도 런처 쪽 루프에 둔다. 따라서 알람
사이클의 녹화 동작에는 영향이 없다.

### 4.2 분석 — `recording_filter/` 확장

```
Stage 1    frame_reduce     (기존) cv2 변화감지로 프레임 축소
Stage 1.5  region_gate      (신규) 영역 지도 + 세대 관리 + 가림 필터 — CV only, VLM 0콜
Stage 2a   click_detect     (기존) 커서 탐지 + ROI 변화로 클릭 판정
Stage 2c   element_label    (신규) 클릭 지점 crop → OCR, 실패 시 VLM 폴백
           timeline         (기존, 스키마 확장)
```

`filter_recording.py` 의 오케스트레이션에 두 스테이지를 끼우고, 자동 탐색 glob 에
`*/_manual/*/recording` 을 추가한다. 기존 알람 녹화 분석 경로는 그대로 동작한다
(신규 스테이지는 사이드카가 없으면 degrade).

**Stage 1.5 를 VLM 0콜로 두는 것이 비용 설계의 핵심이다.** 영역 지도는 세대당 한 번만
`detect_sem_box` 를 쓰고, 프레임 단위 게이팅은 순수 기하 비교다. 세션 길이가 아니라
세대 수에만 비용이 비례하므로, 비싼 Stage 2a/2c 는 이미 걸러진 소수 프레임만 본다.

## 5. G1~G3 대응 상세

### 5.1 영역 게이트 (G1)

세대별 영역 지도로 변화 이벤트를 분류한다.

영역 지도는 프레임을 두 영역으로만 나눈다: `live_image` (라이브 SEM 박스) 와 `ui`
(나머지 전부). 패널 단위 세분화는 하지 않는다 — 게이팅에 필요하지 않고, 장비마다
패널 배치가 달라 신뢰할 수 없다.

| 판정 | 조건 |
|------|------|
| `ambient` | 변화가 **`live_image` 안에만** 있고 커서가 박스 밖 → 장비 자율 갱신, 이벤트 아님 |
| `candidate` | 변화가 `ui` 에 걸침 → Stage 2a 로 승격 |
| `candidate` | 커서가 `live_image` 안 → 직접 조작 가능성이 있으므로 승격 (예외) |

커서 위치는 사이드카의 `cursor_screen_xy` 를 창 rect 기준으로 환산해 쓴다.
**사이드카가 없으면**(`MANUAL_RECORD_META=0`, 또는 알람 사이클 녹화를 분석하는 경우)
커서 예외 규칙을 적용할 수 없으므로, 안전한 쪽으로 기울여 `live_image` 안의 변화도
`candidate` 로 승격시킨다. 즉 게이트가 사실상 무효화되고 기존 동작으로 돌아간다 —
조용히 이벤트를 잃는 것보다 오탐이 낫다.

`detect_sem_box` 가 실패하면 그 세대는 게이트 없이 전부 `candidate` 로 통과시킨다.
오탐이 늘 뿐 데이터는 잃지 않는다.

### 5.2 레이아웃 세대 (G3)

영역 지도를 "세션당 1장"이 아니라 **세대(generation)** 로 관리한다. 프레임마다 창
rect 를 기록해두고, rect(또는 프레임 크기)가 바뀌면 그 시점부터 새 세대를 열어
재검출한다. 이벤트는 자기가 속한 세대의 지도로 게이팅되고, 좌표는 항상 그 프레임
기준이다. 엔지니어가 창을 옮겨도 앞뒤 구간이 모두 살아난다.

장비별 레이아웃 차이는 자동으로 흡수된다 — `detect_sem_box` 는 좌표를 하드코딩하지
않고 그 장비의 실제 프레임을 보고 찾기 때문이다. 다만 검출이 틀렸을 때 조용히 전체가
어긋나지 않도록, 세대마다 오버레이 `region_map_gen{N}.jpg` 를 남겨 오피스에서 눈으로
확인한다.

### 5.3 가림 검사 (G2)

전면 창 제목만으로는 부족하다 — 다른 창이 포커스를 뺏지 않은 채 일부만 겹칠 수 있다.
프레임 저장 시 창 영역의 5개 지점(중앙 + 사분면)에서 `WindowFromPoint` 를 찍어 그
지점의 최상위 창이 우리 창인지 확인한다. 픽셀 비용 없이 부분 가림까지 잡힌다.

결과는 `occlusion: none | partial | full` 로 프레임 메타에 남기고, 분석 단계에서
가려진 구간은 이벤트 후보에서 제외한다. **프레임 자체는 버리지 않는다** — 나중에
판단이 바뀔 수 있다.

## 6. 이식성 (A 장비 → B 장비)

추출한 workflow 를 다른 장비에서 재생할 수 있는지는 **표현 형식이 결정한다.**
A 와 B 는 같은 RCS 클라이언트 exe 이고 다른 것은 창 위치·크기·일부 패널 값뿐이다.

| 표현 | 이식성 | 이유 |
|------|--------|------|
| `click(x, y)` | 불가 | 창 위치·크기가 다름 |
| `click(label="Start Measurement")` | 가능 | B 화면에서 라벨을 다시 찾으면 됨 |
| 라이브 영상 위 드래그/더블클릭 | 조건부 | 좌표가 아니라 영상 **내용**에 의존 → CV 재해석 필요 |

따라서 타임라인에 `target_kind` (`ui_control` | `live_image` | `unknown`) 를 둔다.
필드 하나를 지금 추가하지 않으면, 나중에 이식성을 따질 때 전 프레임을 다시 돌려야 한다.

**재생기는 이번 범위가 아니다.** 다만 만들 때 새로 만들 것이 거의 없다 —
`vlm/ui_venus_mai_locator.analyze_window_target()` 이 이름을 주면 화면에서 찾아
좌표를 돌려주고(`sem_monitor/pm_dropdown.py` 가 'PM' 버튼에 이미 사용),
클릭 전 OCR 확인은 `rcs/tool_row_verify.py` 의 row confirm gate 와 같은 패턴이다.

## 7. 설정과 정지 조건

CLI 인자 없음 규칙에 따라 모듈 상수 + env 오버라이드로 둔다.

| 설정 | 기본값 | 역할 |
|------|--------|------|
| `MANUAL_RECORD_MAX_SEC` | `600` | **실질 상한.** `0` 이면 무제한 (`RecordingSession` 이 `max_sec > 0` 일 때만 검사) |
| `MANUAL_RECORD_MAX_FRAMES` | `1500` | 백스톱 (정상이면 안 걸림) |
| `MANUAL_RECORD_MAX_DISK_MB` | `800` | 백스톱 (정상이면 안 걸림) |
| `MANUAL_RECORD_POLL_SEC` | `0.5` | 샘플링 간격 (알람용 0.3 보다 완화) |
| `MANUAL_RECORD_JPEG_QUALITY` | `85` | q95 대비 용량 약 절반 |
| `MANUAL_RECORD_EQP_ID` | `""` | 모니터링 창이 여럿일 때만 필요 |
| `MANUAL_RECORD_META` | `1` | 사이드카 기록 on/off |

백스톱을 실질 상한보다 넉넉히 잡은 이유는, 이것들이 걸린다는 것이 **예상 못 한 일이
벌어졌다는 신호**여야 하기 때문이다. 시간 상한과 비슷하게 잡으면 manifest 만 보고
원인을 구분할 수 없다.

정지 사유는 manifest 에 기록된다:

| 사유 | 발생 |
|------|------|
| `user_interrupt` | Ctrl+C (정상 경로) |
| `max_sec` | 시간 상한 (`RecordingSession` 제공) |
| `window_gone` | 엔지니어가 툴 창을 닫음 (`RecordingSession` 제공) |
| `frame_budget` / `disk_budget` | 런처 감시 루프가 `stop()` 호출 |

상한 도달 시 콘솔에 크게 알리고 종료한다. **자동 연장하지 않는다** — 무한 디스크
사용을 막는 것이 상한의 목적이므로 자동 연장은 목적을 무효화한다.

`save_debug_jpeg` 에는 `quality: int = 95` 인자를 추가한다. 기본값 유지이므로 기존
호출부는 전부 그대로 동작한다.

## 8. 데이터 스키마

### 8.1 녹화 산출물 — `align_images/<EQP>/_manual/<tag>/recording/`

```
<tag>_rcs_0000_00000000ms.jpg  …     프레임 (기존 명명 규칙 그대로)
frame_meta.jsonl                     프레임당 1줄 사이드카 (신규)
recording_manifest.json              기존 + eqp_id / window_title / 정지 사유
```

`frame_meta.jsonl` 한 줄:

```json
{"frame": "<파일명>", "t_sec": 12.4,
 "window_rect": {"left": 100, "top": 50, "right": 1700, "bottom": 1050},
 "foreground_title": "Remote Monitoring System - MCD916",
 "occlusion": "none", "cursor_screen_xy": [1420, 880], "cursor_in_window": true}
```

`frame` 키로 프레임과 조인한다. 파일이 없으면 분석은 이 신호 없이 동작한다.

### 8.2 분석 산출물 — `<tag>/recording_filter/`

```
region_map.json + region_map_gen{N}.jpg    세대별 영역 지도 + 확인용 오버레이
change_events.json                          기존 + generation / occlusion / gate 판정
interaction_timeline.json                   기존 스키마 + 신규 필드
element_crops/                              클릭 지점 crop (라벨 근거 보존)
summary.json                                기존 + 세대 수 / 게이트 통과율 / VLM 콜 수
```

타임라인 이벤트:

```json
{"t_sec": 12.4, "seq": 3, "action": "click",
 "coords": {"x": 1420, "y": 880},
 "element": "Start Measurement",
 "element_source": "ocr",
 "target_kind": "ui_control",
 "region": "ui",
 "generation": 0,
 "occlusion": "none",
 "confidence": 0.82}
```

`region` 은 §5.1 의 영역 지도 값 (`live_image` | `ui` | `unknown`) 이고,
`target_kind` 는 거기에 라벨링 결과를 합친 **파생값**이다: `region == "ui"` 이고 라벨을
얻었으면 `ui_control`, `region == "live_image"` 이면 `live_image`, 그 외는 `unknown`.
둘을 모두 남기는 이유는 파생 규칙이 나중에 바뀌어도 원본 판정을 다시 계산할 수 있게
하기 위해서다.

`element_source` (`ocr` | `vlm` | `none`) 를 따로 두는 이유는 OCR 로 읽은 라벨과 VLM 이
서술한 라벨의 **신뢰 수준이 다르기** 때문이다. 이식성을 따질 때 `ocr` 만 신뢰하는
식으로 걸러낼 수 있어야 한다.

## 9. 에러 처리 원칙

녹화는 죽지 않고, 분석은 부분 실패를 허용한다.

| 상황 | 처리 |
|------|------|
| 사이드카 기록 실패 (권한/디스크) | 경고 후 계속. 프레임 손실보다 나쁜 것은 없다 |
| `detect_sem_box` 실패 | 그 세대는 게이트 없이 통과 (전부 `candidate`) |
| OCR·VLM 실패 | `element=null`, `element_source="none"`. 이벤트는 남는다 |
| `stop()` 중 재차 Ctrl+C | manifest 는 반드시 기록된다 |
| 모니터링 창 0개 | 안내 메시지 후 종료 (예외 아님) |
| 모니터링 창 2개 이상 | 목록 출력 후 종료. **임의 선택하지 않는다** |

## 10. 테스트

기존 `recording_filter` pytest 18개 옆에 추가한다. 창 탐색·실제 캡처를 제외한
전 로직을 Mac 에서 검증할 수 있다.

| 대상 | 방식 |
|------|------|
| EQP 제목 파싱 | 정상 / 접두어만 / 공백 / 특수문자 (Windows 불필요) |
| 정지 조건 | 가짜 `capture_fn` 주입 — 프레임·디스크·시간 상한이 각각 올바른 사유로 멈추는지 |
| 세대 관리 | 합성 rect 시퀀스로 세대 분할 지점 검증 |
| 영역 게이트 | 합성 프레임 (라이브 박스 안만 변화 vs UI 변화) 으로 판정 검증 |
| 요소 라벨링 | OCR/VLM 클라이언트 주입 — OCR 성공 시 VLM 미호출, 실패 시 폴백 |
| 사이드카 조인 | `frame_meta.jsonl` 있음/없음 두 경우 모두 분석이 동작하는지 |

오피스 Windows 확인 항목: 창 탐색, EQP 추출 실제 제목, 캡처 동작, `region_map_gen0.jpg`
의 영역 지도 정확성.

## 11. 범위 밖 (명시적 제외)

- **재생기(replay)** — 추출 데이터를 보고 결정할 일 (§6 참조)
- **키보드 입력 검출** — 픽셀 추론만으로는 신뢰할 수 없다. 필요해지면 별도 스펙
- **여러 장비 창을 동시에 녹화** — 이번은 창 1개 세션
- **툴 상태(status) 요약** — 라벨링이 쓸만한지 먼저 확인한 뒤 판단
- **`RecordingSession` 수정** — 감싸기만 한다
