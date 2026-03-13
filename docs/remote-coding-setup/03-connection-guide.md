# 연결 테스트 및 트러블슈팅 가이드

이 문서는 `Tailscale + Termius + code-server` 환경을 빠르게 점검하고, 자주 생기는 문제를 바로 복구하기 위한 운영용 문서입니다.

## 목차

1. [권장 점검 순서](#권장-점검-순서)
2. [빠른 진단 명령어](#빠른-진단-명령어)
3. [문제별 해결 방법](#문제별-해결-방법)
4. [일상 운영 팁](#일상-운영-팁)

---

## 권장 점검 순서

### Step 1. Tailscale 연결 확인

Mac Mini에서:

```bash
tailscale status
tailscale ip -4
tailscale netcheck
```

Galaxy Tab에서:

- Tailscale 앱이 연결 상태인지 확인
- `mac-mini`가 기기 목록에 보이는지 확인
- 배터리 절약으로 앱이 중지되지 않았는지 확인

### Step 2. SSH 연결 확인

Termius에서 `mac-mini` 호스트로 접속한 뒤 아래 명령을 실행합니다.

```bash
whoami
hostname
tmux new -As work
```

Mac Mini에서 함께 볼 항목:

```bash
sudo systemsetup -getremotelogin
lsof -nP -iTCP:22 -sTCP:LISTEN
```

### Step 3. 저장소 접근 확인

```bash
cd ~/Codes/auto_recipe_creator
pwd
git status
```

### Step 4. code-server 확인

Mac Mini에서:

```bash
brew services list | grep code-server
lsof -nP -iTCP:8080 -sTCP:LISTEN
curl -fsS http://127.0.0.1:8080/healthz
```

Galaxy Tab 브라우저에서:

```text
http://mac-mini:8080
```

또는:

```text
http://100.x.y.z:8080
```

### Step 5. 실제 작업 명령 테스트

SSH 또는 code-server 통합 터미널에서:

```bash
cd ~/Codes/auto_recipe_creator
uv --version
git status
```

필요하면:

```bash
uv sync --extra dev
uv run pytest
```

---

## 빠른 진단 명령어

### 네트워크

```bash
tailscale status
tailscale ip -4
tailscale netcheck
```

### SSH

```bash
sudo systemsetup -getremotelogin
sudo launchctl list | grep com.openssh.sshd
lsof -nP -iTCP:22 -sTCP:LISTEN
```

### code-server

```bash
brew services list | grep code-server
ps aux | grep -v grep | grep code-server
lsof -nP -iTCP:8080 -sTCP:LISTEN
curl -v http://127.0.0.1:8080/healthz
cat ~/.config/code-server/config.yaml
```

### 절전 상태

```bash
pmset -g
```

### 작업 세션

```bash
tmux ls
```

---

## 문제별 해결 방법

### 문제: Tailscale에서 `mac-mini`가 안 보임

확인:

```bash
tailscale status
open -a Tailscale
```

해결:

- Mac Mini와 Galaxy Tab이 같은 Tailscale 계정인지 확인
- Galaxy Tab의 Tailscale 앱 배터리 제한 해제
- Mac Mini에서 Tailscale 앱을 다시 열고 로그인 상태 확인

### 문제: Termius에서 `Connection refused`

확인:

```bash
sudo systemsetup -getremotelogin
lsof -nP -iTCP:22 -sTCP:LISTEN
```

해결:

```bash
sudo systemsetup -setremotelogin on
sudo launchctl stop com.openssh.sshd
sudo launchctl start com.openssh.sshd
```

### 문제: `Permission denied (publickey)`

확인:

```bash
ls -ld ~/.ssh
ls -l ~/.ssh/authorized_keys
cat ~/.ssh/authorized_keys
```

해결:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

필요하면 Galaxy Tab의 Termius 공개키를 다시 붙여 넣습니다.

### 문제: Termius에서 호스트 키 오류가 남

증상:

- `Host key verification failed`
- 이전에 되다가 갑자기 호스트 키가 바뀐 것으로 보임

해결:

- Termius의 Known Hosts에서 해당 항목 삭제
- Mac Mini에 재접속해서 새 호스트 키를 다시 신뢰

### 문제: SSH는 되는데 자꾸 끊김

해결 순서:

1. Termius `Keep Alive` 활성화
2. Mac Mini `sshd_config`에 아래 값 확인

```text
ClientAliveInterval 60
ClientAliveCountMax 3
```

3. 실제 작업은 `tmux` 안에서만 수행

```bash
tmux new -As work
```

### 문제: 브라우저에서 code-server 페이지가 안 열림

Mac Mini에서:

```bash
brew services list | grep code-server
lsof -nP -iTCP:8080 -sTCP:LISTEN
curl -fsS http://127.0.0.1:8080/healthz
```

해결:

```bash
brew services restart code-server
```

그래도 안 되면 포그라운드로 띄워서 오류를 봅니다.

```bash
brew services stop code-server
code-server
```

### 문제: code-server 로그인 비밀번호가 맞지 않음

확인:

```bash
cat ~/.config/code-server/config.yaml
```

비밀번호를 새로 정한 뒤 다시 시작합니다.

```bash
brew services restart code-server
```

브라우저에 저장된 오래된 세션이 꼬였으면 해당 사이트의 쿠키를 지우고 재로그인합니다.

### 문제: code-server는 열리는데 편집기 화면이 비거나 멈춤

가능한 원인:

- 브라우저 탭 절전
- Android 메모리 회수
- 네트워크가 `relay` 경로만 사용 중

해결:

- 브라우저에서 데스크톱 사이트 사용
- Galaxy Tab 배터리 절약 모드 해제
- Tailscale `netcheck` 결과 확인
- code-server 탭을 너무 많이 띄우지 않기

### 문제: 8080 포트가 이미 사용 중임

확인:

```bash
lsof -nP -iTCP:8080
```

해결:

- 기존 프로세스를 정리하거나
- `config.yaml`의 포트를 다른 값으로 바꾸고 재시작

예시:

```yaml
bind-addr: 0.0.0.0:18080
```

변경 후:

```bash
brew services restart code-server
```

브라우저 주소도 함께 변경합니다.

### 문제: Mac Mini가 잠자기로 들어가서 접속이 안 됨

확인:

```bash
pmset -g
```

해결:

```bash
sudo pmset -a sleep 0
sudo pmset -a displaysleep 0
sudo pmset -a womp 1
```

재부팅 후에도 유지되는지 다시 확인합니다.

### 문제: Termius SFTP에서 업로드가 안 됨

확인:

```bash
pwd
ls -ld ~/Codes ~/Codes/auto_recipe_creator
```

해결:

- 쓰기 권한이 있는 경로로 업로드
- 저장소 내부 파일이면 git 권한과 사용자 계정이 맞는지 확인
- 원격 사용자명이 다른 계정으로 설정되지 않았는지 확인

### 문제: 한글 입력이 이상함

확인:

```bash
echo $LANG
echo $LC_ALL
```

필요하면 `~/.zshrc`에 추가:

```bash
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8
```

적용:

```bash
source ~/.zshrc
```

---

## 일상 운영 팁

### 출근 후 1분 점검

```bash
tailscale status
brew services list | grep code-server
tmux ls
```

### 작업 시작 루틴

```bash
tmux new -As work
cd ~/Codes/auto_recipe_creator
git status
```

### 테스트 전 확인

```bash
uv --version
uv run python --version
pwd
```

### 서비스 재시작 루틴

```bash
brew services restart code-server
open -a Tailscale
```

### 브라우저 주소를 잊었을 때

아래 두 가지 중 하나만 기억하면 됩니다.

```text
http://mac-mini:8080
http://100.x.y.z:8080
```

---

## 체크리스트

### 매일

- [ ] Tailscale 연결 상태 확인
- [ ] Termius SSH 접속 확인
- [ ] `tmux` 세션 진입 확인
- [ ] code-server 접속 확인

### 주간

- [ ] `brew upgrade` 여부 검토
- [ ] Tailscale 앱 업데이트 확인
- [ ] code-server 설정 파일 백업 여부 확인
- [ ] 디스크 공간 확인

---

## 도움말 링크

- Tailscale: https://tailscale.com/kb/
- Termius: https://support.termius.com/
- code-server: https://coder.com/docs/code-server/latest
