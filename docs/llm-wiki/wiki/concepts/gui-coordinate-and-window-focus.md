---
tags: [concept, gui-automation, coordinates, windows]
level: intermediate
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: [
  raw/journals/260316/20260316-login-rcs-coordinate-contract.md,
  raw/journals/260316/20260316-window-titles-성능개선.md,
  raw/journals/260316/20260316-work2-window-search-correction.md,
  raw/journals/260316/20260316-work2-rcs-window-focus.md,
  raw/journals/260318/260318_143312_ui-tars-1token-fix.md,
  raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md,
  raw/journals/260323/260323_154335_simplify-select-tool-dedup.md
]
---

# GUI Coordinate and Window Focus

> VLM이 낸 이미지 좌표를 실제 Windows screen action으로 안전하게 연결하기 위한 좌표 계약과 foreground 검증 원칙이다. (source: raw/journals/260316/20260316-login-rcs-coordinate-contract.md)

## 왜 필요한가? (Why)

- GUI automation에서는 잘못된 창을 foreground로 잡거나 이미지 좌표를 screen 좌표로 잘못 변환하면 엉뚱한 위치를 클릭할 수 있다. (source: raw/journals/260316/20260316-work2-rcs-window-focus.md)
- `GetWindowTextLengthW` 같은 호출이 hung window에서 느려질 수 있어 창 목록 수집 성능도 automation 안정성의 일부로 다뤄졌다. (source: raw/journals/260316/20260316-window-titles-성능개선.md)

## 핵심 개념 (What)

### 정의

- 좌표 계약은 VLM 응답 좌표의 기준계를 명시하고, 최종 클릭 전 image coordinate를 window/screen coordinate로 변환하는 약속이다. (source: raw/journals/260316/20260316-login-rcs-coordinate-contract.md)
- window focus contract는 capture 직전 대상 창 활성화와 실제 foreground handle 일치 여부를 확인하는 약속이다. (source: raw/journals/260316/20260316-work2-window-search-correction.md)

### 관련 용어

- `relative_1000`: VLM이 0~1000 상대 좌표로 클릭 지점을 반환하는 좌표 체계. (source: raw/journals/260316/20260316-login-rcs-coordinate-contract.md)
- `UI-Venus [0,1000]`: 공식 UI-Venus grounding prompt가 `[x,y]` 중심점 좌표를 0~1000 정규화 좌표로 반환하는 규칙. (source: raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md)
- `UI-TARS smart-resize`: UI-TARS action 좌표는 smart-resize 공간 기준이므로 실제 클릭 전 역변환이 필요하다. (source: raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md)
- `image_point_to_screen`: 이미지 좌표를 실제 화면 좌표로 변환하는 helper. (source: raw/journals/260323/260323_154335_simplify-select-tool-dedup.md)
- `collect_window_rows`: raw Win32 top-level window enumeration 결과를 공용화한 helper. (source: raw/journals/260316/20260316-work2-window-search-correction.md)
- `foreground_window`: `SetForegroundWindow()` 반환값뿐 아니라 실제 foreground handle 일치 여부를 확인하는 흐름. (source: raw/journals/260316/20260316-work2-window-search-correction.md)

### 시각화 / 모델

```text
target window
  -> find by PID/title prefix
  -> activate + verify foreground
  -> capture image
  -> VLM returns relative_1000 point
  -> convert image point to screen point
  -> click/type action
```

## 어떻게 사용하는가? (How)

### 최소 예제

```json
{
  "coord_system": "relative_1000",
  "point": {"x": 512, "y": 344}
}
```

로그인과 main tab prompt는 클릭용 좌표 응답을 `relative_1000` 계약으로 통일했다. (source: raw/journals/260316/20260316-login-rcs-coordinate-contract.md)

### 실무 패턴

- 창 탐색은 PID 우선 탐색 후 필요할 때만 title prefix scan으로 fallback한다. (source: raw/journals/260316/20260316-work2-rcs-window-focus.md)
- top-level window scan은 raw Win32 enumeration으로 먼저 후보를 좁히고, 매칭된 handle만 backend wrapper로 변환한다. (source: raw/journals/260316/20260316-work2-window-search-correction.md)
- overlay 좌표는 이미지 범위 안으로 clamp해 debug image가 좌표 오류로 깨지지 않게 한다. (source: raw/journals/260316/20260316-login-rcs-coordinate-contract.md)
- UI-Venus refusal `[-1, -1]`은 요소가 보이지 않는 신호로 처리해 hallucinated click을 줄인다. (source: raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md)

### 주의사항 / 함정

- `SetForegroundWindow()` 호출이 성공처럼 보여도 실제 foreground가 다른 창일 수 있으므로 capture 직전 재확인이 필요하다. (source: raw/journals/260316/20260316-work2-window-search-correction.md)
- 창 제목 스캔에서 모든 창마다 blocking text length query를 호출하면 hung window 때문에 전체 scan이 느려질 수 있다. (source: raw/journals/260316/20260316-window-titles-성능개선.md)

## 참고 자료 (References)

- 원본 메모: [20260316-login-rcs-coordinate-contract.md](../../raw/journals/260316/20260316-login-rcs-coordinate-contract.md)
- 원본 메모: [20260316-window-titles-성능개선.md](../../raw/journals/260316/20260316-window-titles-성능개선.md)
- 원본 메모: [20260316-work2-window-search-correction.md](../../raw/journals/260316/20260316-work2-window-search-correction.md)
- 원본 메모: [20260316-work2-rcs-window-focus.md](../../raw/journals/260316/20260316-work2-rcs-window-focus.md)
- 원본 메모: [260318_143312_ui-tars-1token-fix.md](../../raw/journals/260318/260318_143312_ui-tars-1token-fix.md)
- 원본 메모: [260318_163432_ui-venus-official-grounding-overhaul.md](../../raw/journals/260318/260318_163432_ui-venus-official-grounding-overhaul.md)
- 관련 컴포넌트: [rcs-login-automation.md](../components/rcs-login-automation.md)
