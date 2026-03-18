# 2026-03-16 deploy_vlms 기준 VLM 백그라운드 실행 정리

## 1. 진행 사항

- `deploy_vlms/` 기준으로 터미널에서 VLM을 백그라운드로 유지하는 실행 흐름을 정리했다.
- `docs/setup_vlms/README.md`, `docs/setup_vlms/03-operations-and-repo-integration.md`를 기준으로 실제 운영 명령을 확인했다.
- `deploy_vlms/scripts/start_model.py` 구현을 확인해 기본 동작이 포그라운드가 아니라 백그라운드 실행임을 재확인했다.
- `deploy_vlms/scripts/start_model.py`에서 `RUN_IN_BACKGROUND = 1`, `subprocess.Popen(..., start_new_session=True)` 조합으로 터미널 세션과 분리해 프로세스를 띄우는 점을 확인했다.
- 로그와 PID 기록 위치가 `deploy_vlms/runtime/logs/<instance>.log`, `deploy_vlms/runtime/pids/<instance>.pid`로 남는 점을 함께 정리했다.
- 운영 명령 기준으로 아래 흐름을 저널에 남겼다.
  - 시작: `python scripts/start_ui_venus.py`, `python scripts/start_mai_ui.py`, `python scripts/start_model.py ui-venus 30b`
  - 확인: `python scripts/check_vlm.py http://127.0.0.1:8001 ui-venus-1.5-8b`, `curl http://127.0.0.1:8001/v1/models`, `tail -f runtime/logs/ui-venus.log`
  - 중지: `python scripts/stop_model.py ui-venus`

## 2. 수정 내용

- 새 저널 파일 추가: `docs/journals/deploy-vlms/20260316-vlm-background-terminal-runbook.md`
- 코드 수정은 없고, `deploy_vlms/` 운영 방식만 문서화했다.
- 참고한 핵심 경로:
  - `docs/setup_vlms/README.md`
  - `docs/setup_vlms/03-operations-and-repo-integration.md`
  - `deploy_vlms/scripts/start_model.py`
  - `deploy_vlms/scripts/stop_model.py`

## 3. 다음 단계

- 실제 배포 서버에서 `python scripts/start_ui_venus.py` 실행 후 셸을 분리한 뒤에도 `runtime/pids/`와 `runtime/logs/`가 정상 유지되는지 다시 확인한다.
- 운영 표준을 더 명확히 하려면 `deploy_vlms/` 아래에 `BACKGROUND_RUN.md` 같은 짧은 실무용 문서를 추가해 저널 내용을 상시 문서로 승격한다.
- 포그라운드 디버깅이 자주 필요하면 `start_model.py`의 `RUN_IN_BACKGROUND` 전환 방식을 `.env` 기반으로 바꾸는지 검토한다.

## 4. 메모리 업데이트

- 변경 없음
