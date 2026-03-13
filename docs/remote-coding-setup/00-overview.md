# 원격 코딩 환경 가이드

이 디렉터리는 이제 `Tailscale + Termius + code-server` 조합만 다룹니다. 실제로 계속 사용할 두 가지 접근 방식에 집중하고, 그 외 대안 경로는 문서 범위에서 제외합니다.

## 목표 구성

집의 Mac Mini를 항상 켜진 개발 서버로 두고, 회사에서는 Galaxy Tab에서 아래 두 경로로 접속합니다.

```text
Galaxy Tab
 ├─ Termius
 │   └─ SSH / SFTP
 └─ Browser
     └─ code-server (VS Code in browser)
            │
            ▼
       Tailscale VPN
            │
            ▼
         Mac Mini
         ├─ SSH Server
         ├─ code-server
         ├─ tmux / git / uv
         └─ Claude Code
```

## 언제 무엇을 쓰는가

| 작업 | 도구 | 권장 이유 |
|------|------|-----------|
| 빠른 접속, 서버 점검, 명령 실행 | `Termius` | 가장 빠르고 안정적이며 네트워크 상태가 나빠도 대응하기 좋음 |
| 코드 편집, 검색, 다중 파일 수정 | `code-server` | 브라우저에서 VS Code 수준의 편집 경험 제공 |
| 장시간 작업 세션 유지 | `Termius + tmux` | 접속이 끊겨도 세션이 유지됨 |
| 간단한 파일 업/다운로드 | `Termius SFTP` | 추가 앱 없이 바로 사용 가능 |

## 운영 원칙

1. 네트워크는 항상 `Tailscale`로만 연결합니다.
2. 셸 작업의 기본 진입점은 `Termius`입니다.
3. 에디터가 필요할 때만 브라우저에서 `code-server`를 엽니다.
4. 장시간 작업은 `tmux` 세션 안에서 진행합니다.
5. 외부 공개 URL 대신 `mac-mini:22`, `mac-mini:8080` 같은 Tailscale 주소를 우선 사용합니다.

## 빠른 시작 순서

1. Galaxy Tab에서 Tailscale 연결 확인
2. Termius로 `mac-mini` SSH 접속
3. `tmux new -As work` 실행
4. 필요한 경우 브라우저에서 `http://mac-mini:8080` 접속
5. 작업 후에는 `exit` 대신 `tmux detach`로 세션 유지

## 자주 쓰는 명령어

### Mac Mini에서 확인

```bash
tailscale status
tailscale ip -4
sudo systemsetup -getremotelogin
brew services list | grep code-server
lsof -nP -iTCP:22 -sTCP:LISTEN
lsof -nP -iTCP:8080 -sTCP:LISTEN
curl -fsS http://127.0.0.1:8080/healthz
```

### Termius로 접속한 뒤

```bash
whoami
hostname
tmux new -As work
cd ~/Codes/auto_recipe_creator
git status
uv run pytest
```

### code-server 관리

```bash
brew services start code-server
brew services restart code-server
brew services stop code-server
code-server --list-extensions
code-server --install-extension ms-python.python
```

## 권장 디렉터리 구조

```text
~/Codes/
└─ auto_recipe_creator

~/.config/code-server/
└─ config.yaml

~/.ssh/
├─ authorized_keys
└─ config
```

## 보안 기준

- `Tailscale`은 동일 계정으로만 연결
- SSH는 가능하면 공개키 인증만 사용
- `code-server`는 강한 비밀번호 사용
- Mac Mini는 잠자기 해제 또는 최소화
- 작업 세션은 `tmux`로 유지하고, 셸에서 민감한 토큰을 평문 파일에 남기지 않음

## 문서 구성

| 문서 | 설명 |
|------|------|
| [01-mac-mini-setup.md](./01-mac-mini-setup.md) | Mac Mini 서버 준비, SSH, code-server, 원격 작업용 기본 도구 설정 |
| [02-galaxy-tab-setup.md](./02-galaxy-tab-setup.md) | Galaxy Tab에서 Termius와 브라우저 기반 작업 환경 구성 |
| [03-connection-guide.md](./03-connection-guide.md) | 연결 점검 순서, 운영용 명령어, 자주 생기는 문제와 해결책 |

## 범위 밖

이 폴더는 원격 셸과 브라우저 기반 편집 경로만 다룹니다. 실제 사용 경로를 줄여야 점검 포인트와 장애 대응도 단순해지기 때문입니다.
