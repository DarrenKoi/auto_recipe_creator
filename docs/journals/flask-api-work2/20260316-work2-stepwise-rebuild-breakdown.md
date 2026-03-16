## 1. 진행 사항
- `poc/work2/vlm_screen_analysis.py`, `poc/work2/pipeline_ocr.py`, `poc/work2/click_rcs_view_mode.py`, `poc/work2/check_tool_screen.py` 삭제 요청을 반영해 파일을 제거했다.
- `poc/work2/__init__.py`의 `__all__`에서 삭제된 `work2` 모듈 export를 정리했다.
- `AGENTS.md`와 `CLAUDE.md`에서 삭제된 스크립트 실행 예시와 설명을 제거하고, 현재 남아 있는 `poc/work2/` 구조를 tree 형태로 다시 정리했다.
- `AGENTS.md`의 `poc/work2` 상세 맵을 현재 파일 기준으로 갱신하고, `connection_check.py` -> `open_rcs.py` -> `login_rcs.py` 중심 rebuild baseline으로 설명을 축소했다.
- `CLAUDE.md`에 현재 `poc/work2/` 디렉터리 구조를 명시하고, 더 이상 존재하지 않는 OCR pipeline / tool-screen workflow 설명을 정리했다.

## 2. 수정 내용
- 삭제 파일: `poc/work2/vlm_screen_analysis.py`
- 삭제 파일: `poc/work2/pipeline_ocr.py`
- 삭제 파일: `poc/work2/click_rcs_view_mode.py`
- 삭제 파일: `poc/work2/check_tool_screen.py`
- 수정 파일: `poc/work2/__init__.py`
  삭제된 모듈 이름을 `__all__`에서 제거해 패키지 export와 실제 파일 구조를 맞췄다.
- 수정 파일: `AGENTS.md`
  현재 `poc/work2/` 트리, 남아 있는 핵심 스크립트, `util/` 및 `prompts/` 하위 구조를 반영했다.
- 수정 파일: `CLAUDE.md`
  `poc/work2/` 설명을 현재 구조 기준으로 바꾸고 삭제된 entrypoint 참조를 제거했다.

## 3. 다음 단계
- `poc/work2` rebuild를 더 작은 단계로 다시 나눈다.
- 1단계: `connection_check.py`로 Flask proxy / service 상태 확인만 담당한다.
- 2단계: `open_rcs.py`로 RCS 실행 여부 확인과 실행만 담당한다.
- 3단계: `login_rcs.py`로 로그인 화면 캡처와 좌표 마킹만 담당한다.
- 4단계: 이후 필요한 기능은 한 파일에 여러 동작을 다시 합치지 말고, 탭 탐지, 툴 화면 감지, 후속 클릭/검증을 각각 별도 스크립트로 재생성한다.
- 새 스크립트를 다시 만들 때는 각 단계가 `observe -> decide -> act -> verify` 중 어느 범위를 담당하는지 파일 단위로 명확히 분리한다.

## 4. 메모리 업데이트
- 변경 없음
