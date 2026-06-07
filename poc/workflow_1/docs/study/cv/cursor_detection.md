# 프레임에서 커서 끝점 찾기 — 2단계 검출 (intro)

> 대상: `locate_cursor_in_captured_frames.py`, `capture_window_frames_ch4.py`,
> `extract_recorded_ch4_frames.py`, `monitor_align_fail.py`
> 상위 개요: `../algorithms/two_stage_vlm_locator.md` (같은 coarse→fine 패턴)

---

## 1. 왜 커서를 찾나

DVR/CCTV(CH4) 녹화나 캡처 프레임에서 **마우스 커서가 어디를 가리켰는지** 알면, 엔지니어가 수동으로
무엇을 클릭했는지 재구성할 수 있습니다. 이는 align fail 분석과 동작 학습의 단서가 됩니다.

핵심 난점: 커서는 **작고**, 진짜 클릭 지점은 **bbox 중심이 아니라 화살표 끝(hotspot)** 입니다.

---

## 2. CH4 프레임 캡처

`capture_window_frames_ch4.py` 의 `capture_frames()`:

```python
DEFAULT_FRAME_INTERVAL_MS = 100     # 100ms 간격
DEFAULT_MAX_DURATION_SEC  = 480.0   # 최대 8분
DEFAULT_JPEG_QUALITY      = 95
```

루프는 캡처에 걸린 시간을 빼고 sleep 하여 **간격을 일정하게** 유지합니다(`sleep = interval - elapsed_loop`).
산출물:
```
recordings/capture_window_frames_ch4/<ts>_<window_title>/
├─ frames/ frame_0000_00000000ms.jpg ...
├─ summary.json   (프레임 메타)
└─ timeline.txt   (사람이 읽는 타임라인)
```
DVR 창 열기는 `monitor_align_fail.open_cctv_for_tool()` → `select_tool_cctv_from_main_window()`.

---

## 3. 2단계 커서 검출 (coarse → fine)

로그인 로케이터와 **같은 coarse→fine 철학** 을 따릅니다.

### Stage 1 — coarse (ui-venus)
- 입력: 전체 프레임.
- 출력(JSON): `cursor_visible`, `cursor_bbox`(0–1000), `confidence`.

### Stage 2 — refine (mai-ui)
- coarse bbox 둘레를 `DEFAULT_CROP_PADDING_PX=48` 만큼 넓혀 crop, `DEFAULT_ZOOM_SCALE=3x` 확대.
- 출력: `cursor_bbox` + **`cursor_tip`(클릭 hotspot)** + confidence + evidence.
- **`cursor_tip` 은 bbox 중심이 아니라 화살표 끝점** — 실제 클릭이 일어난 픽셀.

### 좌표 역변환
`_translate_local_point_to_full()` 로 crop-local 좌표 → 전체 프레임 픽셀:
```python
full_x = crop_box["left"] + local_x   # y 동일
```

프레임별 결과: `coarse_bbox` / `refined_bbox` / `cursor_tip` (모두 전체 프레임 픽셀) + confidence.
오버레이 저장: 전체 프레임(cyan=coarse, red=refine) + 확대 crop.

---

## 4. 정적 모니터 가정 — 중복 제거 불필요

Align fail 시 SEM Monitor 가 멈추면서 화면이 고정됩니다. 그래서 프레임 간 전환 감지나 중복 제거 같은
무거운 필터가 **필요 없습니다**(메모리: project_sem_monitor_static_at_align_fail). 캡처는 단순하게 하고,
분석은 필요한 프레임에만 합니다.

---

## 5. 상사 예상 질문

**Q. 커서 bbox 중심을 클릭 지점으로 쓰면 안 되나?**
A. 화살표 커서는 중심과 hotspot(끝점)이 서로 다릅니다. 중심을 쓰면 실제 클릭 지점에서 몇 px 빗나갑니다.
그래서 fine 단계가 `cursor_tip` 을 따로 돌려줍니다.

**Q. 왜 또 2단계인가?**
A. 커서는 작아서 전체 프레임에서 끝점을 픽셀 단위로 잡기 어렵습니다. 확대한 뒤 fine 으로 잡는 편이 안정적입니다.

---

## 6. 핵심 상수 한눈에

| 상수 | 값 | 의미 |
|---|---|---|
| `DEFAULT_FRAME_INTERVAL_MS` | 100 | 프레임 캡처 간격 |
| `DEFAULT_MAX_DURATION_SEC` | 480 | 최대 캡처 시간(8분) |
| `DEFAULT_JPEG_QUALITY` | 95 | 프레임 JPEG 품질 |
| `DEFAULT_ZOOM_SCALE` | 3x | refine 확대 배율 |
| `DEFAULT_CROP_PADDING_PX` | 48 | coarse bbox 둘레 패딩 |
