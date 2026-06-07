---
status: accepted
---

# RCS 내부 컨트롤은 pywinauto UIA 가 아니라 스크린샷→VLM 좌표→pynput 으로 조작한다

## 결정

RCS 창 **내부의 버튼/입력칸/콤보 클릭** 은 pywinauto 의 UIA/win32 backend 로 컨트롤을 직접 찾아서
조작하지 않는다. 대신 다음 경로를 쓴다:

```
스크린샷(mss) → VLM 으로 좌표 찾기(ui-venus coarse → mai-ui fine) → OCR 확인 → DPI 보정 → pynput 클릭/입력
```

pywinauto 는 **창 라이프사이클** 에만 쓴다: exe 실행 보조, 창 제목 탐색(`find_window_by_title_prefix`),
foreground/restore/close.

## 맥락 / 이유

- **RCS(RcsMainHD.exe)는 legacy GUI** 다. UIA/win32 접근성 트리에 ComboBox·Button 같은 컨트롤이
  제대로 노출되지 않아, `window.child_window(...)` 류로는 잡히지 않거나 빈 트리만 돌아온다.
- 화면에는 분명히 보이는 요소인데도 프로그램적으로는 "물어볼 수 없는" 상태다. 그래서 **사람처럼 화면을
  보고 누르는** 경로만이 유일하게 안정적이다.
- VLM 은 "보이는 것"을 좌표로 돌려줄 수 있으므로 이 공백을 메운다.

## 결과 (Consequences)

- 모든 클릭은 VLM 좌표 의존 → **정확도·안정성 장치가 필요**: 2단계(coarse→fine), OCR 확인,
  "미확인 시 클릭 금지", Tool ID 정규화 + 모호 시 포기.
- **DPI 보정이 필수** 가 된다(이미지 좌표 ≠ 화면 좌표). → `../algorithms/dpi_coordinate_mapping.md`.
- VLM 호출 비용·지연이 생긴다 → coarse 는 싸게 범위만 좁히고, 정밀은 확대 후 fine 1회.
- pywinauto 의존을 **창 단위로 최소화** → 버전/backend 변동에 견고하다(`_extract_window_handle` 이
  handle/hwnd/element_info 를 차례로 시도).

## 대안 (기각)

- **순수 pywinauto UIA 조작**: legacy 컨트롤 미노출로 실패. (메모리: poc/work 의 automate_rcs_login
  에서 이미 확인됨.)
- **하드코딩 좌표**: 창 위치·DPI·레이아웃 변동에 깨짐.
