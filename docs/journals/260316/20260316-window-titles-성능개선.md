# 2026-03-16 — window_titles.py 성능 개선 및 CubeMain 포커스 기능 추가

## 1. 진행 사항

- `window_titles.py`의 창 목록 스캔 속도가 느린 원인 분석
- 성능 병목 제거 후 속도 개선 커밋 및 푸시 (`a6f433b`)
- "CubeMain" 창을 자동으로 foreground로 활성화하는 기능 추가

## 2. 수정 내용

### 파일: `window_titles.py`

#### 성능 개선 (커밋 `a6f433b`)

**원인 분석 — 왜 느렸나:**

- `_read_window_text()`가 창마다 `GetWindowTextLengthW` + `GetWindowTextW` 두 번의 Win32 API 호출을 수행
- `GetWindowTextLengthW`는 대상 창에 `WM_GETTEXTLENGTH` 메시지를 전송하는데, 응답이 없는(hung) 창이 있으면 타임아웃까지 블로킹
- 수백 개의 top-level 창 각각에 대해 이 블로킹 호출이 누적되어 전체 스캔이 느려짐
- 매 창마다 `ctypes.create_unicode_buffer()` 새로 할당하는 오버헤드도 존재

**해결 방법 — 어떻게 빨라졌나:**

| 변경 전 | 변경 후 |
|---|---|
| `GetWindowTextLengthW` + `GetWindowTextW` (2회 API 호출/창) | `GetWindowTextW`만 호출 (1회/창) |
| 창마다 `create_unicode_buffer()` 새로 할당 | 고정 512자 버퍼 1개를 사전 할당 후 재사용 |
| `_read_window_text()` 함수 호출 (콜백 내) | 콜백 내 인라인 처리로 Python 호출 오버헤드 제거 |

- `_TITLE_BUF_SIZE = 512` 상수 추가
- `_read_window_text()`: `buffer` 파라미터 추가 (외부에서 전달 가능)
- `_collect_window_rows()`: 버퍼 사전 할당, 콜백 내 `GetWindowTextW` 직접 호출
- `main()`: 공유 버퍼 생성 후 `_read_window_text()`에 전달

#### CubeMain 포커스 기능 추가

- `_focus_window(user32, hwnd)`: 최소화 상태면 `ShowWindow(SW_RESTORE)` 후 `SetForegroundWindow` 호출
- `_find_and_focus_cubemain(user32, rows)`: 수집된 창 목록에서 "CubeMain" 키워드 매칭 후 foreground 활성화
- `main()`에서 창 수집 직후 `_find_and_focus_cubemain()` 호출

## 3. 다음 단계

- Windows 환경에서 `CubeMain` 포커스 기능 동작 확인 (SetForegroundWindow 권한 이슈 가능)
- 필요 시 `SetForegroundWindow` 실패 대응 (Alt 키 시뮬레이션 등 workaround)

## 4. 메모리 업데이트

변경 없음
