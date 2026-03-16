## 1. 진행 사항
- `poc/work2/automate_rcs_login.py` 삭제 요청을 반영해 파일을 제거했다.
- `poc/work2/__init__.py`에서 `automate_rcs_login` export를 제거하고 `open_rcs` export를 추가해 현재 `work2` 기준 경계를 정리했다.
- `AGENTS.md`에서 `poc/work2` 상세 맵과 실행 명령 목록을 현재 재구성 방향에 맞게 수정했다.
- `CLAUDE.md`에서 `poc/work2/automate_rcs_login.py` 실행 예시와 설명을 제거하고 `poc/work2/open_rcs.py`, `poc/work2/login_rcs.py` 중심 설명으로 바꿨다.
- `rg -n "automate_rcs_login" AGENTS.md CLAUDE.md poc/work2`로 활성 경로 기준 잔여 참조를 점검했다.

## 2. 수정 내용
- 삭제 파일: `poc/work2/automate_rcs_login.py`
- 수정 파일: `poc/work2/__init__.py`
  `__all__`에서 `automate_rcs_login`을 제거하고 `open_rcs`를 추가했다.
- 수정 파일: `AGENTS.md`
  `poc/work2` 현재 기준 스크립트 설명을 `open_rcs.py`, `login_rcs.py`, `connection_check.py` 중심으로 정리했다.
- 수정 파일: `CLAUDE.md`
  실행 예시와 RCS 자동화 workflow 설명에서 삭제된 `automate_rcs_login.py`를 제거하고 rebuild baseline을 반영했다.
- 확인 결과:
  `AGENTS.md`, `CLAUDE.md`, `poc/work2/` 범위에서는 `automate_rcs_login` 활성 참조가 남아 있지 않았다.
- 미정리 범위:
  `docs/` 하위 journal/research 문서에는 과거 기록으로 남아 있는 `automate_rcs_login.py` 언급이 아직 존재한다.

## 3. 다음 단계
- `poc/work2/rcs_utils.py` 삭제 여부를 확정하고, 남아 있는 `login_rcs.py`, `click_rcs_view_mode.py` 의존성을 더 작은 모듈로 재분리한다.
- `open_rcs.py` -> `login_rcs.py` -> `click_rcs_view_mode.py` 순서로 rebuild baseline을 다시 문서화한다.
- 필요하면 `AGENTS.md`, `CLAUDE.md`뿐 아니라 `docs/` 하위 최신 운영 문서에서도 삭제된 스크립트 참조를 추가 정리한다.

## 4. 메모리 업데이트
- 변경 없음
