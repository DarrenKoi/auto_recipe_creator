# DPI 좌표 변환 — 이미지 픽셀을 실제 화면 픽셀로 (deep dive)

> 대상: `util/window_utils.py` (`image_point_to_screen`), `util/mouse_utils.py`, `__init__.py` (`_enable_dpi_awareness`)
> 상위 개요: `automation_methods_intro.md` §2

---

## 1. 문제 — "사진 속 좌표"와 "마우스 좌표"는 다르다

VLM 은 **스크린샷 이미지 위의 좌표** 를 주는데, pynput 마우스는 **화면(screen) 좌표** 를 필요로
합니다. 두 좌표계가 항상 같다면 문제가 없겠지만, **Windows DPI 배율** 때문에 서로 어긋납니다.

- 1920×1080 모니터를 **150% 배율** 로 쓰면, 논리 해상도는 1280×720 처럼 보이지만 실제 픽셀은
  여전히 1920×1080(또는 캡처 설정에 따라 더 큼).
- mss 는 **물리 픽셀** 로 캡처합니다 (예: 큰 이미지).
- pywinauto 의 `window.rectangle()` 은 **논리 픽셀** 로 창 위치를 보고합니다 (작은 숫자).

→ 이미지 좌표를 그대로 화면 좌표로 쓰면 배율만큼 **일관되게 빗나갑니다.** (오피스 PC 가 125/150%
라서, "내 Mac 에선 되는데 오피스에선 빗나간다"의 단골 원인입니다.)

---

## 2. 해결 1 — import 시 DPI awareness 선언

`poc/workflow_1/__init__.py` 의 `_enable_dpi_awareness()` 가 **pywinauto import 보다 먼저** 호출됩니다.
가장 정확한 API 부터 사다리식으로 시도합니다:

```python
1) user32.SetProcessDpiAwarenessContext(-4)   # PER_MONITOR_AWARE_V2 (Win10 1703+)
2) shcore.SetProcessDpiAwareness(2)            # PER_MONITOR_DPI_AWARE (Win8.1+)
3) user32.SetProcessDPIAware()                 # system-level fallback
```

이렇게 선언하면 OS 가 좌표를 멋대로 가상화(virtualize)하지 않으므로, **캡처와 rect 가 같은 픽셀 기준**
위에 놓입니다. 비-Windows 이거나 선언에 실패하면 조용히 넘어갑니다(Mac 개발 환경 호환).

> 왜 "import 시점"인가? pywinauto 가 로드되며 DPI 컨텍스트를 굳히기 전에 선언해야 효과가 있기 때문입니다.

---

## 3. 해결 2 — rect/screenshot 비율로 좌표 보정

`util/window_utils.py` 의 `image_point_to_screen(window, image_point, image_size)`:

```python
rect = window.rectangle()
rect_w = rect.right - rect.left
rect_h = rect.bottom - rect.top

scale_x = rect_w / img_w     # image_size 가 주어졌을 때
scale_y = rect_h / img_h

screen_x = round(rect.left + image_point["x"] * scale_x)
screen_y = round(rect.top  + image_point["y"] * scale_y)
```

핵심: **하드코딩한 DPI 배수를 쓰지 않습니다.** "실제 캡처 이미지 크기"와 "실제 창 rect 크기"의 비율을
직접 계산하므로, **어떤 배율 설정에서도 자동으로 들어맞습니다.** `image_size=None` 이면 scale=1.0(100%
가정)이 됩니다.

### 예시
- 캡처 이미지 1920×1080, 창 rect 1280×720 → scale=(1.5,1.5) → 이미지 (960,540) → 화면 (1280×.. 보정).
- DPI awareness 까지 켜져 있으면 보통 캡처=rect 라 scale≈1.0 이 되어 그대로 통과.

---

## 4. 클릭 실행 — pynput, 그리고 safe mode 게이팅

`util/mouse_utils.py`:

```python
def click_at_screen(screen_point, target_key, click_count=1, *, action_enabled=True) -> bool:
    if not action_enabled or not PYNPUT_MOUSE_AVAILABLE:
        # [DRY-RUN] 로그만 찍고 True 반환 — 실제 마우스 출력 없음
        return True
    mouse.position = (sx, sy); time.sleep(0.01)
    mouse.click(Button.left, click_count)
```

- **safe mode**: `action_enabled=False` 이면 실제 마우스가 움직이지 않고 `[DRY-RUN]` 만 출력.
  `WorkflowSettings` 에서 `SAFE_MODE` 의 역으로 기본값이 정해집니다 → Mac 에서 안전하게 dry-run.
- **import guard**: pynput 이 없으면(`PYNPUT_MOUSE_AVAILABLE=False`) 역시 dry-run. headless/서버 호환.
- `scroll_at_screen()` 도 같은 게이팅 패턴(List 탭 스크롤 등에 사용).

---

## 5. 보조 — 창 찾기/상태 제어 (window_utils 전체 그림)

같은 모듈은 좌표 변환 외에 창 라이프사이클도 다룹니다(클릭이 올바른 창으로 가도록):

- `collect_window_rows()` — Win32 `EnumWindows` 로 빠르게 전체 top-level 창 열거(제목 버퍼 512자,
  null/zero-width/BOM 정규화, PID·visible 필터).
- `find_window_by_title_prefix()` — 제목 접두사 정규식 매칭 후 pywinauto wrapper 생성 (UIA→win32 순).
- `activate_window()` / `foreground_window()` — restore→SetForegroundWindow→set_focus→click_input 사다리,
  `GetForegroundWindow()` 로 검증.
- `close_window()` — close()→WM_CLOSE→WM_SYSCOMMAND(SC_CLOSE)→빨간 X 클릭(offset `(-18,18)`) 사다리.

요점: **창은 pywinauto/Win32 로 다루고, 창 내부 클릭은 VLM 좌표 + DPI 보정 + pynput 으로** 다룹니다.

---

## 6. 상사 예상 질문

**Q. 그냥 배율을 1.5 로 곱하면 안 되나?**
A. PC마다 배율이 다르고(125/150/175%), 멀티모니터에서는 모니터마다 다릅니다. 하드코딩한 배수는
깨질 수밖에 없습니다. "실측한 rect/이미지 비율"을 쓰면 설정과 무관하게 맞습니다.

**Q. DPI awareness 만 켜면 보정은 필요 없지 않나?**
A. awareness 를 켜면 보통 캡처=rect 가 되어 scale≈1.0 이 되지만, 캡처 경로나 환경에 따라 차이가 남을
수 있습니다. 그래서 **선언과 비율 보정을 모두** 둡니다. 안전벨트를 두 개 매는 셈입니다.

**Q. 클릭이 여전히 빗나가면?**
A. 로그에 rect 경계·이미지 크기·계산된 scale·원좌표/최종좌표가 모두 찍힙니다. 거기서 scale 이
1.0 인지, rect 가 엉뚱한 창을 가리키는지부터 확인합니다.

---

## 7. 핵심 상수 한눈에

| 항목 | 값/위치 | 의미 |
|---|---|---|
| DPI 컨텍스트 | `-4` (PER_MONITOR_AWARE_V2) | 1순위 awareness |
| 제목 버퍼 | `_TITLE_BUF_SIZE=512` | Win32 GetWindowTextW |
| 빨간 X offset | `(-18, 18)` | close fallback 클릭 위치 |
| activate settle | 0.15s | 상태 변경 후 대기 |
| click 전 sleep | 0.01s | 마우스 이동 후 안정화 |
