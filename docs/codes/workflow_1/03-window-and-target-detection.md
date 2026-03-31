# 창 탐색과 타겟 좌표 탐지

이 문서는 아래 네 파일을 함께 설명한다.

- `open_rcs.py`
- `login_rcs_common.py`
- `login_rcs_ui_venus_mai.py`
- `ui_venus_mai_locator.py`

이 네 파일은 "RCS가 떠 있는지 확인하고, 로그인 창을 찾고, 창 내부 타겟을 VLM으로 찾는 과정"을 담당한다.

## 1. 큰 흐름

```text
open_rcs.py
  -> RCS 실행 또는 기존 PID 재사용
  -> open_rcs_state.json 저장

login_rcs_common.py
  -> 상태 파일 읽기
  -> PID 생존 확인
  -> 로그인 창 또는 updater 창 찾기

login_rcs_ui_venus_mai.py
  -> 어떤 타겟을 찾을지 정의
  -> analyze_window_target() 호출

ui_venus_mai_locator.py
  -> 전체 창 캡처
  -> ui-venus로 coarse bbox 찾기
  -> bbox 주변 crop 생성
  -> crop을 확대
  -> mai-ui로 refined point 찾기
  -> full image 좌표로 복원
  -> overlay/JSON 저장
```

## 2. `open_rcs.py`: RCS bootstrap 전용 파일

이 파일은 워크플로 엔진이 아니라 "프로세스 시작기"다.

### 상수 선언 블록

| 상수 | 의미 |
| --- | --- |
| `RCS_EXE` | 실행할 `RcsMainHD.exe` 경로 |
| `OPEN_RCS_STATE_PATH` | PID 상태 파일 경로 |
| `OPEN_ANOTHER_RCS_PROCESS` | 기존 프로세스가 있어도 새로 띄울지 여부 |
| `EARLY_CRASH_WAIT_SEC` | 실행 직후 즉시 죽는지 짧게 확인하는 시간 |

### `write_open_rcs_state()`

이 함수는 PID, 상태, exe 경로를 JSON으로 저장한다.

의미:

- 다음 스크립트가 프로세스를 다시 스캔하지 않고
- "방금 띄운 RCS가 무엇인지"를 안정적으로 재사용할 수 있게 한다.

### `find_existing_rcs_processes()`

이 블록은 `psutil`로 현재 실행 중인 프로세스를 훑어
이름이나 exe 경로가 `RcsMainHD.exe`와 맞는 프로세스를 찾는다.

의미:

- 중복 실행 방지
- 기존 세션 재활용

### `launch_rcs()`

이 함수는 실제 `subprocess.Popen()`으로 RCS를 띄운다.

코드 블록 의미:

1. 작업 디렉터리를 exe가 있는 폴더로 잡는다.
2. Windows라면 새 process group으로 실행한다.
3. 실행 성공 시 PID를 반환한다.

### `main()`

메인 흐름은 아래처럼 읽으면 된다.

```text
exe 존재 확인
-> 기존 RCS 프로세스 탐색
-> 이미 있으면 state 파일만 갱신하고 종료
-> 없으면 새로 실행
-> 짧게 생존 확인
-> state 파일 기록
```

즉, 이 파일의 목적은 "RCS 실행 상태를 파일로 외부화"하는 것이다.

## 3. `login_rcs_common.py`: 창 탐색 공용 계층

이 파일은 `workflow_1`에서 가장 Windows 의존성이 강한 공용 헬퍼다.

### 상수/함수 포인터 준비 블록

이 초반부 블록은 `poc.work2.util`에서 창 관련 함수를 가져온다.

핵심 의미:

- `workflow_1`이 직접 pywinauto 구현 세부를 들고 있지 않다.
- 대신 `activate_window`, `find_window_by_title_prefix` 같은 공용 유틸에 의존한다.

### `_login_window_filter()`

로그인 창 후보를 크기로 거르는 함수다.

판단 기준:

- 폭/높이가 0보다 커야 한다
- `RCS_LOGIN_WINDOW_MAX_WIDTH`, `RCS_LOGIN_WINDOW_MAX_HEIGHT` 이하인 작은 대화상자만 통과시킨다

이 블록이 필요한 이유:

- 제목 prefix만 보면 메인 창이나 다른 하위 창도 잡힐 수 있으므로
- "작은 로그인 다이얼로그"라는 물리적 특징을 함께 사용한다.

### 상태 파일 읽기 블록

- `_load_open_rcs_pid()`
- `_load_open_rcs_exe_path()`

의미:

- `open_rcs.py`가 남긴 state file을 읽어
- 이번 로그인 시도가 어느 프로세스에 붙어야 하는지 파악한다.

### PID 검증 블록

- `_normalize_path_text()`
- `_is_pid_alive()`

의미:

- PID가 살아 있는지
- 살아 있다면 정말 기대한 RCS exe인지

를 검증한다.

이 블록은 "PID가 있으니 맞겠지"를 방지한다.

### fallback 실행 블록

- `_run_open_rcs_fallback()`
- `_ensure_rcs_running()`

의미:

- 상태 파일이 없거나 PID가 죽었으면
- `open_rcs.py`를 다시 실행해서 bootstrap을 복구한다.

즉, `login_rcs_common.py`는 창 탐색 전에 항상 "RCS가 정말 살아 있는가?"를 먼저 보장하려고 한다.

### 창 탐색 블록

#### `find_login_window()`

탐색 순서:

1. 상태 파일 기반 PID를 확보한다.
2. 해당 PID 안에서 제목 prefix가 맞는 창을 찾는다.
3. 못 찾으면 desktop 전체를 스캔한다.
4. 찾으면 foreground/activate를 시도한다.

이 블록의 의미는 "PID 우선, 전체 스캔은 fallback"이다.

#### `find_rcs_main_window()` / `find_rcs_updater_window()`

로그인 이후 나타나는 새 창을 제목 prefix로 찾는다.

이 함수들은 verify step에서 사용된다.

#### `wait_for_rcs_main_window()` / `wait_for_rcs_updater_window()`

timeout까지 폴링하며 새 창 등장 여부를 기다린다.

의미:

- 로그인 직후 UI 전환이 느릴 수 있으므로
- 단발성 체크 대신 polling 기반 검증을 한다.

## 4. `login_rcs_ui_venus_mai.py`: 로그인 타겟 정의 레이어

이 파일은 탐지 엔진이라기보다 "로그인 화면 전용 target registry"다.

### `PREDEFINED_TARGETS`

여기에는 아래 타겟들이 들어 있다.

- `userid_input`
- `password_input`
- `server_input`
- `login_button`
- `cancel_button`

각 항목의 의미:

- `key`
  후속 step과 결과 JSON에서 쓰는 식별자
- `description`
  VLM prompt에 들어갈 자연어 설명
- pad ratio들
  `ui-venus` coarse bbox 주변을 얼마나 넓게 crop할지 결정하는 튜닝값

즉, 이 블록은 "무엇을 찾고 싶은가"를 선언한다.

### `analyze_login_target()`

이 함수는 로그인 창이라는 구체적 시나리오 정보를 채워서,
일반화된 `analyze_window_target()`에 넘긴다.

의미:

- 디버그 이미지 경로
- artifact prefix
- result mode
- log 이름

같은 시나리오 전용 메타데이터를 한 곳에서 묶는다.

### `main()`

이 파일을 단독 실행할 때는 `ACTIVE_TARGET_KEY` 하나만 골라 탐지한다.

즉, 이 스크립트는 "로그인 워크플로 전체"가 아니라 "타겟 탐지 디버거"다.

## 5. `ui_venus_mai_locator.py`: 2단계 VLM 위치 탐지 엔진

이 파일이 실제 좌표 탐지의 핵심이다.

## 5.1 구조 요약

```text
창 준비
-> 스크린샷 캡처
-> full image WebP 인코딩
-> ui-venus 호출
-> coarse bbox 획득
-> bbox 주변 crop box 계산
-> crop 이미지 생성
-> 필요 시 확대
-> mai-ui 호출
-> refined point 획득
-> 원본 full image 좌표계로 역변환
-> overlay / response / result JSON 저장
```

## 5.2 작은 기하학 헬퍼 블록들

### `_ensure_min_span()`

crop 영역이 너무 작으면 최소 폭/높이를 만족하도록 키운다.

의미:

- `ui-venus`가 아주 작은 bbox를 줘도
- `mai-ui`가 읽을 만큼의 주변 문맥을 확보하려는 장치다.

### `_build_crop_box()`

coarse bbox 주위에 좌우/상하 padding을 적용해 crop 영역을 만든다.

왜 필요한가:

- coarse bbox가 입력 필드 전체가 아니라 일부만 잡을 수 있다
- refined 단계에는 주변 label, 테두리, 빈 공간이 함께 보여야 클릭 포인트를 더 잘 찾는다

### `_resize_crop_for_mai()`

작은 crop은 최소 크기 이상으로 확대한다.

의미:

- 작은 GUI 요소는 바로 넣으면 `mai-ui`가 놓칠 수 있으므로
- 일정 해상도 이상으로 키워 정밀 탐지를 돕는다.

### 좌표 변환 블록

- `_map_resized_point_to_full_image()`
- `_scale_bbox_to_resized_crop()`
- `_point_to_tiny_bbox()`

이 함수들의 역할:

- 확대된 crop 좌표를 원본 화면 좌표로 되돌린다
- overlay 이미지를 그릴 때 coarse/refined 위치를 같은 좌표계로 맞춘다

즉, 탐지 정확도 개선을 위해 중간 좌표계를 쓰되, 최종 결과는 다시 전체 화면 좌표로 복구한다.

## 5.3 입력/응답 저장 블록

### `_save_pipeline_inputs()`

원본 전체 캡처와 zoom crop을 JPEG로 저장한다.

의미:

- 나중에 사람이 디버깅할 때
- "모델이 실제로 어떤 그림을 보고 판단했는지"를 재현할 수 있게 한다.

### `_print_vlm_understanding()`

응답 텍스트와 토큰 사용량을 콘솔에 바로 읽기 좋게 출력한다.

의미:

- 로그 파일만 보지 않고도 프롬프트 품질을 빠르게 확인할 수 있다.

## 5.4 VLM 호출 블록

### `_run_ui_venus_coarse_bbox()`

이 함수는 전체 이미지에서 대략적인 bbox를 찾는다.

내부 의미:

1. `build_ui_venus_single_element_bbox_prompt()`로 prompt 생성
2. `Work2VLMClient.chat_with_image_b64()`로 호출
3. 응답 JSON 추출
4. 1000 기준 bbox를 pixel bbox로 변환
5. 중심점 계산

산출물:

- `bbox_1000`
- `bbox_pixels`
- `center`
- 원문 응답 텍스트
- 토큰 사용량

### `_run_mai_ui_refinement()`

이 함수는 zoom crop 안에서 세밀한 클릭 point를 찾는다.

내부 의미:

1. `build_mai_ui_zoom_prompt()`로 refined prompt 생성
2. `mai-ui` 호출
3. 응답 JSON 파싱
4. 좌표를 crop 크기에 맞게 정규화
5. 타겟 key에 해당하는 최종 point 추출

산출물:

- refined point
- 원문 응답
- 토큰 사용량

## 5.5 `analyze_window_target()`: 전체 파이프라인 오케스트레이터

이 함수는 앞선 모든 블록을 묶는 메인 함수다.

### 블록 1: 창 준비

`image`가 없으면:

1. 창 activate
2. foreground 전환
3. window capture

를 수행한다.

의미:

- 탐지 실패 원인을 줄이기 위해
- 항상 최신 foreground 상태의 캡처를 쓰려는 것이다.

### 블록 2: coarse detection

전체 이미지를 WebP로 인코딩한 뒤 `ui-venus`에 보낸다.

여기서 coarse bbox가 없으면 즉시 실패한다.

의미:

- refine 단계는 coarse 단계 성공을 전제로 한다.

### 블록 3: crop 생성 및 확대

coarse bbox 주변을 잘라내고, 작으면 키운다.

의미:

- 전체 화면에서 직접 point를 찾는 대신
- 관련 영역만 크게 보여 정확도를 높인다.

### 블록 4: refine detection

확대 crop을 `mai-ui`에 보내 refined point를 얻는다.

point가 없으면 실패한다.

### 블록 5: 결과 좌표 복원

refined point는 crop/zoom 좌표계에 있으므로,
이를 full image 좌표계로 되돌린다.

이 블록의 결과가 실제 클릭 후보 좌표다.

### 블록 6: artifact 저장

저장되는 것:

- 원본 캡처 JPEG
- zoom crop JPEG
- `ui-venus` 응답 텍스트
- `mai-ui` 응답 텍스트
- full image overlay
- zoom overlay
- 최종 result JSON

의미:

- 디버깅 시 "입력", "모델 응답", "좌표 시각화", "최종 결과"를 모두 남긴다.

### 블록 7: 이벤트 로그와 반환

최종 point와 사용 서비스/모델명을 로그에 남기고,
`TargetResult(EXIT_SUCCESS, target.key, point=...)`를 반환한다.

즉, 이 함수는 단순 helper가 아니라
"탐지 전체 과정을 통째로 실행하고 증거를 남기는 파이프라인 함수"다.

## 6. 요약

이 네 파일의 역할 분담은 아래처럼 보면 된다.

| 파일 | 질문 |
| --- | --- |
| `open_rcs.py` | RCS가 떠 있는가 |
| `login_rcs_common.py` | 어느 창에 붙어야 하는가 |
| `login_rcs_ui_venus_mai.py` | 로그인 화면에서 무엇을 찾을 것인가 |
| `ui_venus_mai_locator.py` | 그 타겟의 클릭 좌표를 어떻게 안정적으로 찾을 것인가 |
