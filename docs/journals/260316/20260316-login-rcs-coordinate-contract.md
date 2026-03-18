## 1. 진행 사항
- `poc/work2/login_rcs.py` 기준으로 로그인 창 재포커스 흐름을 점검하고, 캡처 직전에도 창을 다시 활성화하도록 정리했다.
- `poc/work2/open_rcs.py`가 남기는 상태 파일(`poc/work2/logs/open_rcs_state.json`)을 `login_rcs.py`가 읽어 PID 기준으로 로그인 창을 다시 찾는 흐름을 유지했다.
- `poc/work2/util/json_utils.py`의 좌표 해석 방식을 검토하고, VLM 응답 좌표를 최종적으로 이미지 픽셀 좌표로 변환하는 공통 경로를 정리했다.
- `poc/work2/prompts/rcs_login.py`, `poc/work2/prompts/rcs_main_tabs.py`의 좌표 계약을 검토해 `coord_system="relative_1000"` 기반의 클릭용 좌표 응답 형식으로 통일했다.
- 오래된 `poc/work2/check_tool_screen.py` 제거 요청과 함께 현재 `work2` 구조를 반영하도록 관련 문서/패키지 노출 항목을 정리했다.
- 전체 변경을 포함해 `git commit -m "Reorganize work2 RCS automation flow"` 커밋을 생성했다.

## 2. 수정 내용
- 수정 파일: `poc/work2/login_rcs.py`
  로그인 창 재활성화 후 캡처하는 흐름을 유지하도록 정리했다.
- 수정 파일: `poc/work2/util/json_utils.py`
  `coord_system` 필드 해석과 `relative_1000` -> 픽셀 변환 로직을 추가했다.
- 수정 파일: `poc/work2/rcs_utils.py`
  공통 `parse_coords()`가 `json_utils.py` 구현을 사용하도록 맞추고, 오버레이 좌표를 이미지 범위 안으로 clamp 하도록 변경했다.
- 수정 파일: `poc/work2/util/debug_image_utils.py`
  오버레이 마킹 좌표를 이미지 범위 안으로 clamp 하도록 변경했다.
- 수정 파일: `poc/work2/prompts/rcs_login.py`
  로그인 요소 응답 형식을 `coord_system="relative_1000"` + 클릭 안전 좌표로 명시했다.
- 수정 파일: `poc/work2/prompts/rcs_main_tabs.py`
  탭 좌표 응답 형식을 `coord_system="relative_1000"` + 클릭 안전 좌표로 명시했다.
- 수정 파일: `poc/work2/util/window_utils.py`
  기본 pywinauto backend 우선순위를 `uia` 먼저로 유지했다.
- 수정 파일: `poc/work2/prompts/ocr_assist.py`
  삭제된 파일명을 직접 언급하던 설명 문구를 정리했다.
- 수정 파일: `poc/work2/__init__.py`, `AGENTS.md`, `CLAUDE.md`
  제거된 `work2` 스크립트 기준으로 현재 구조 설명을 정리했다.
- 삭제 파일: `poc/work2/check_tool_screen.py`
- 삭제 파일: `poc/work2/click_rcs_view_mode.py`
- 삭제 파일: `poc/work2/pipeline_ocr.py`
- 삭제 파일: `poc/work2/vlm_screen_analysis.py`

## 3. 다음 단계
- 사무실 Windows 환경에서 `uv run python poc/work2/login_rcs.py`를 실행해 `relative_1000` 좌표가 오버레이와 실제 클릭 위치에 맞는지 확인한다.
- 로그인 화면 기준으로 `coord_system="relative_1000"` 계약이 안정적이면 이후 후속 GUI 자동화 스크립트도 같은 좌표 체계로 통일할지 결정한다.
- 현재 생성된 커밋을 `origin/main`으로 push 한다.

## 4. 메모리 업데이트
- 변경 없음
