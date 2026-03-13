# Mac Mini 서버 설정 가이드

이 문서는 Mac Mini를 `Termius SSH + code-server` 전용 원격 개발 서버로 정리하는 방법을 설명합니다.

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [Tailscale 설정](#1-tailscale-설정)
3. [SSH 서버 설정](#2-ssh-서버-설정)
4. [원격 작업용 셸 준비](#3-원격-작업용-셸-준비)
5. [code-server 설치 및 설정](#4-code-server-설치-및-설정)
6. [Claude Code 및 개발 도구](#5-claude-code-및-개발-도구)
7. [절전 및 자동 시작](#6-절전-및-자동-시작)
8. [운영 점검 명령어](#7-운영-점검-명령어)

---

## 사전 요구사항

- macOS Sonoma 이상
- 관리자 권한 계정
- 안정적인 인터넷
- Homebrew 설치 가능 상태
- Galaxy Tab에서 사용할 Tailscale 계정 준비

---

## 1. Tailscale 설정

### 1.1 설치

```bash
brew install --cask tailscale
```

설치 후 앱을 한 번 실행합니다.

```bash
open -a Tailscale
```

### 1.2 로그인 및 기본 확인

1. 메뉴 막대의 Tailscale 아이콘을 열고 로그인합니다.
2. Galaxy Tab과 동일한 계정으로 로그인합니다.
3. 연결 후 아래 명령으로 상태를 확인합니다.

```bash
tailscale status
tailscale ip -4
```

### 1.3 호스트명 고정

브라우저 접속과 SSH 접속에서 같은 이름을 쓰기 위해 호스트명을 정리합니다.

```bash
scutil --get ComputerName
sudo tailscale set --hostname=mac-mini
```

Tailscale 관리 콘솔에서 `MagicDNS`를 켜 두면 `mac-mini` 이름으로 바로 접근할 수 있습니다.

### 1.4 연결 점검

```bash
tailscale netcheck
```

`direct` 경로가 잡히면 가장 좋고, `relay`만 잡히면 속도가 느릴 수 있습니다.

---

## 2. SSH 서버 설정

### 2.1 원격 로그인 활성화

```bash
sudo systemsetup -setremotelogin on
sudo systemsetup -getremotelogin
```

22번 포트가 실제로 열렸는지 확인합니다.

```bash
lsof -nP -iTCP:22 -sTCP:LISTEN
```

### 2.2 SSH 키 인증 준비

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Galaxy Tab의 Termius에서 생성한 공개키를 `~/.ssh/authorized_keys`에 추가합니다.

```bash
echo "ssh-ed25519 AAAA... galaxy-tab-termius" >> ~/.ssh/authorized_keys
```

### 2.3 선택: SSH 서버 안정화 옵션

`/etc/ssh/sshd_config`에 아래 항목을 검토합니다.

```text
PubkeyAuthentication yes
PasswordAuthentication no
PermitRootLogin no
ClientAliveInterval 60
ClientAliveCountMax 3
```

수정 후 재시작:

```bash
sudo launchctl stop com.openssh.sshd
sudo launchctl start com.openssh.sshd
```

### 2.4 로컬 테스트

```bash
ssh localhost
```

---

## 3. 원격 작업용 셸 준비

### 3.1 기본 패키지 설치

```bash
brew install tmux git
```

### 3.2 자주 쓰는 alias 추가

`~/.zshrc`에 아래 정도는 넣어두는 편이 좋습니다.

```bash
alias ll='ls -lah'
alias croot='cd ~/Codes/auto_recipe_creator'
alias gs='git status'
alias tma='tmux attach -t work || tmux new -s work'
alias cs='brew services restart code-server'
```

한 번에 추가하려면:

```bash
cat <<'EOF' >> ~/.zshrc
alias ll='ls -lah'
alias croot='cd ~/Codes/auto_recipe_creator'
alias gs='git status'
alias tma='tmux attach -t work || tmux new -s work'
alias cs='brew services restart code-server'
EOF
```

적용:

```bash
source ~/.zshrc
```

### 3.3 tmux 기본 사용

```bash
tmux new -s work
tmux attach -t work
tmux ls
```

원격 연결이 자주 바뀌는 환경에서는 모든 작업을 `tmux` 안에서 진행하는 것이 안전합니다.

### 3.4 작업 디렉터리 준비

```bash
mkdir -p ~/Codes
cd ~/Codes
git clone <your-repo-url>
```

이미 저장소가 있다면 현재 위치만 확인하면 됩니다.

```bash
cd ~/Codes/auto_recipe_creator
pwd
git status
```

---

## 4. code-server 설치 및 설정

### 4.1 설치

```bash
brew install code-server
```

### 4.2 설정 파일 생성

설정 파일 경로:

```bash
mkdir -p ~/.config/code-server
```

예시 `~/.config/code-server/config.yaml`:

```yaml
bind-addr: 0.0.0.0:8080
auth: password
password: change-this-to-a-strong-password
cert: false
```

파일을 바로 만들려면:

```bash
cat <<'EOF' > ~/.config/code-server/config.yaml
bind-addr: 0.0.0.0:8080
auth: password
password: change-this-to-a-strong-password
cert: false
EOF
```

설정 포인트:

- `bind-addr: 0.0.0.0:8080`
  Tailscale 네트워크에서 Galaxy Tab이 접속할 수 있게 합니다.
- `auth: password`
  브라우저에서 비밀번호를 요구하게 합니다.
- `cert: false`
  Tailscale 내부망에서 간단히 운영할 때 흔히 쓰는 설정입니다.

### 4.3 서비스 시작

```bash
brew services start code-server
brew services list | grep code-server
```

직접 디버깅할 때는 포그라운드 실행이 편합니다.

```bash
code-server
```

### 4.4 상태 확인

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
curl -fsS http://127.0.0.1:8080/healthz
```

브라우저에서 열 주소:

```text
http://mac-mini:8080
```

또는 Tailscale IP 기준:

```text
http://100.x.y.z:8080
```

### 4.5 기본 확장 설치

```bash
code-server --install-extension ms-python.python
code-server --install-extension ms-python.vscode-pylance
code-server --install-extension ms-toolsai.jupyter
code-server --list-extensions
```

### 4.6 워크스페이스 실행 팁

```bash
cd ~/Codes/auto_recipe_creator
code-server .
```

이 명령은 포그라운드 디버깅용으로 좋고, 평소에는 `brew services` 방식이 더 안정적입니다.

---

## 5. Claude Code 및 개발 도구

### 5.1 Node.js 설치

```bash
brew install node@20
echo 'export PATH="/opt/homebrew/opt/node@20/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
node --version
npm --version
```

### 5.2 Claude Code 설치

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

### 5.3 인증

```bash
claude login
```

또는 API 키를 쓸 경우:

```bash
echo 'export ANTHROPIC_API_KEY="your-api-key"' >> ~/.zshrc
source ~/.zshrc
```

### 5.4 Python/uv 환경 확인

```bash
brew install uv
uv --version
cd ~/Codes/auto_recipe_creator
uv sync --extra dev
```

---

## 6. 절전 및 자동 시작

### 6.1 잠자기 최소화

```bash
sudo pmset -a sleep 0
sudo pmset -a displaysleep 0
sudo pmset -a womp 1
pmset -g
```

### 6.2 로그인 항목

아래 앱은 로그인 시 자동 시작되도록 등록합니다.

- Tailscale
- code-server를 `brew services`로 운영 중이면 별도 로그인 항목은 불필요

### 6.3 재부팅 후 상태 확인

```bash
tailscale status
brew services list | grep code-server
sudo systemsetup -getremotelogin
```

---

## 7. 운영 점검 명령어

### 빠른 상태 점검

```bash
date
hostname
whoami
tailscale status
sudo systemsetup -getremotelogin
brew services list | grep code-server
lsof -nP -iTCP:22 -sTCP:LISTEN
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

### 서비스 재시작

```bash
brew services restart code-server
open -a Tailscale
```

### 디버깅

```bash
ps aux | grep -v grep | grep code-server
ps aux | grep -v grep | grep tailscale
cat ~/.config/code-server/config.yaml
curl -v http://127.0.0.1:8080/healthz
```

---

## 설치 완료 체크리스트

- [ ] Tailscale 설치 및 로그인 완료
- [ ] `mac-mini` 이름 또는 Tailscale IP 확인
- [ ] SSH 원격 로그인 활성화
- [ ] Termius 공개키를 `authorized_keys`에 등록
- [ ] `tmux` 설치 완료
- [ ] `code-server` 설치 및 `config.yaml` 설정 완료
- [ ] `brew services start code-server` 성공
- [ ] `curl http://127.0.0.1:8080/healthz` 성공
- [ ] `claude --version` 확인
- [ ] `uv sync --extra dev` 가능 상태 확인

---

## 다음 단계

[02-galaxy-tab-setup.md](./02-galaxy-tab-setup.md)에서 Galaxy Tab 측 설정을 진행합니다.
