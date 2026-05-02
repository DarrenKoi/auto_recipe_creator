# 2026-03-15 VLM 프로세스 터미널 종료 시 유지 방안 조사

## 1. 진행 사항

- **VLM 배포 스크립트 프로세스 관리 방식 분석**
  - `deploy_vlms/scripts/` 디렉토리 전체 스크립트 조사
  - `serve_vlm.py`가 `os.execvpe()` (line 643)로 vLLM 프로세스를 실행하며, 포그라운드 프로세스로 동작함을 확인
  - 데몬화, nohup, signal 핸들링 등 백그라운드 실행 메커니즘이 전혀 없음을 확인
  - `stop_model.py`의 종료 방식 확인: SIGTERM → 10초 대기 → SIGKILL 순서

- **터미널 종료 시 VLM 프로세스 종료 문제 확인**
  - 터미널 세션 종료 시 SIGHUP 시그널이 전달되어 VLM 프로세스가 함께 죽는 문제 식별
  - `start_*.py` → `start_model.py` → `serve_vlm.py` → `os.execvpe()` 체인 전체가 포그라운드 실행임을 파악

- **해결 방안 3가지 제시**
  1. `tmux` / `screen` 세션 활용 (코드 변경 불필요)
  2. `nohup` 명령어 활용 (코드 변경 불필요)
  3. `systemd` 서비스 파일 생성 (가장 안정적, auto-restart/부팅 시 자동 시작)

- **사내 프라이빗 클라우드 환경 제약 확인 방법 안내**
  - 사용자 권한 확인 명령어 (`whoami`, `id`, `sudo -l`)
  - 도구 설치 여부 확인 (`which tmux`, `which screen`, `which systemctl`)
  - 컨테이너 환경 여부 확인 (`/proc/1/cgroup`, `hostname`)
  - 패키지 설치 가능 여부 확인 (`apt`, `yum`, `conda`)

## 2. 수정 내용

- 코드 변경 없음 (조사 및 분석 세션)

## 3. 다음 단계

- **사내 클라우드 서버에서 아래 항목 직접 테스트:**

  ### 환경 확인 체크리스트
  ```bash
  # 1. 사용자 권한 확인
  whoami
  id
  sudo -l

  # 2. tmux/screen 설치 여부
  which tmux
  which screen

  # 3. systemd 사용 가능 여부
  which systemctl
  systemctl --user status

  # 4. nohup 사용 가능 여부 (거의 항상 가능)
  which nohup

  # 5. 패키지 설치 가능 여부
  which apt && apt list --installed 2>/dev/null | head
  which conda

  # 6. 컨테이너 환경 여부
  cat /proc/1/cgroup 2>/dev/null
  hostname
  ```

  ### 테스트 결과에 따른 구현 방향
  | 결과 | 추천 솔루션 |
  |------|------------|
  | `tmux` 설치됨 | tmux 세션으로 VLM 실행 래퍼 스크립트 작성 |
  | `tmux` 없음 + `sudo` 가능 | tmux 설치 후 래퍼 스크립트 작성 |
  | `tmux` 없음 + `sudo` 불가 | `nohup` 기반 실행 스크립트 작성 |
  | `systemctl` 사용 가능 | systemd 서비스 파일 생성 (가장 안정적) |

- 테스트 결과를 가지고 돌아오면 선택한 방안에 맞는 구현 진행 예정

## 4. 메모리 업데이트

- 변경 없음
