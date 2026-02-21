# 연결 테스트 및 트러블슈팅 가이드

이 문서는 Mac Mini와 Galaxy Tab 간의 연결을 테스트하고 발생할 수 있는 문제를 해결하는 방법을 설명합니다.

## 목차

1. [연결 테스트 순서](#연결-테스트-순서)
2. [Tailscale 트러블슈팅](#tailscale-트러블슈팅)
3. [SSH 트러블슈팅](#ssh-트러블슈팅)
4. [화면 공유 (Moonlight + Sunshine) 트러블슈팅](#화면-공유-moonlight--sunshine-트러블슈팅)
5. [Code-Server 실행 점검](#code-server-실행-점검)
6. [성능 최적화](#성능-최적화)
7. [일반적인 문제 해결](#일반적인-문제-해결)

---

## 연결 테스트 순서

### Step 1: Tailscale 연결 확인

**Galaxy Tab에서:**
```
1. Tailscale 앱 열기
2. VPN 토글 ON 확인
3. 기기 목록에서 "mac-mini" 확인
4. 상태: "Connected" 또는 녹색 점
```

**Mac Mini IP 확인:**
- Tailscale 앱에서 mac-mini의 IP 확인 (예: 100.64.x.x)

### Step 2: 네트워크 연결 테스트

**Termius 또는 터미널 앱에서:**
```bash
# Ping 테스트 (Android 터미널 필요)
ping -c 4 100.64.x.x

# 또는 Termius의 Ping 기능 사용
```

**예상 결과:**
```
PING 100.64.x.x: 56 data bytes
64 bytes from 100.64.x.x: icmp_seq=0 ttl=64 time=25.3 ms
64 bytes from 100.64.x.x: icmp_seq=1 ttl=64 time=23.1 ms
```

### Step 3: SSH 연결 테스트

**Termius에서:**
1. 설정한 "Mac Mini" 호스트 탭
2. 연결 시도

**성공시 화면:**
```
Last login: Mon Jan 20 10:30:00 2026
username@mac-mini ~ %
```

**테스트 명령어:**
```bash
# 시스템 정보 확인
uname -a

# 현재 사용자 확인
whoami

# Claude Code 확인
claude --version
```

### Step 4: Moonlight 연결 테스트

**테스트 항목:**
- [ ] Moonlight 앱에서 기기 목록에 `mac-mini` 또는 Tailscale IP가 표시됨
- [ ] PIN 입력 후 스트리밍 시작됨
- [ ] 마우스/터치 입력이 정상 동작
- [ ] 키보드 입력이 즉시 반응
- [ ] 가로 모드에서 UI가 찢기지 않고 전체 표시됨
- [ ] 문서/터미널 폰트가 과도하게 작지 않음

**Mac Mini에서 실행/포트 점검**
```bash
# Sunshine 실행 확인
ps aux | grep -v grep | grep -i sunshine

# 일반적인 Sunshine 포트 확인
lsof -nP -iTCP:47984 -sTCP:LISTEN
lsof -nP -iTCP:47989 -sTCP:LISTEN
lsof -nP -iUDP:47998
```

**Galaxy Tab에서 확인**
1. Moonlight 실행
2. `mac-mini` 장치 탭 선택
3. 화면/입력 동작 정상인지 확인

**해상도 튜닝 가이드(태블릿 기준)**
- 1차: `1920x1200` + `60fps`
- 불안정: `1920x1080` + `60fps`
- 지속적 지연: `1600x900` + `30fps`
- 저대역폭: `1280x720` + `30fps` + 품질 `Medium`

### Step 5: Claude Code 테스트

**SSH 접속 후:**
```bash
# Claude Code 실행 테스트
claude "안녕하세요, 연결 테스트입니다."

# 프로젝트 디렉토리에서 테스트
cd ~/projects
claude "현재 디렉토리 구조를 설명해주세요."
```

### Step 6: code-server 실행 상태 점검

**Mac Mini에서 code-server 기동/확인**
```bash
# code-server 상태 확인
brew services list | grep code-server

# 프로세스 확인
ps aux | grep -v grep | grep "code-server"

# 8080 포트 리스너 확인
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

**code-server 실행 (미실행 시)**
```bash
# 백그라운드 실행
brew services start code-server

# 수동 실행(포트/인증 지정)
code-server --bind-addr 0.0.0.0:8080 --auth password
```

**클라이언트에서 접속 테스트**
```bash
# Tailscale IP 기준 예시
curl -I http://100.64.x.x:8080
```

---

## Tailscale 트러블슈팅

### 문제: 기기가 목록에 나타나지 않음

**원인 1: 다른 계정으로 로그인**
```
해결: 양쪽 기기 모두 동일한 Tailscale 계정으로 로그인 확인
```

**원인 2: Tailscale 서비스 미실행 (Mac)**
```bash
# Mac Mini에서 확인
tailscale status

# 재시작
sudo killall tailscaled
open -a Tailscale
```

**원인 3: VPN 권한 미승인 (Android)**
```
해결: 설정 → 앱 → Tailscale → 권한 → VPN 권한 확인
```

### 문제: 연결이 자주 끊김

**해결책 1: Android 배터리 최적화 제외**
```
설정 → 앱 → Tailscale → 배터리 → "제한 없음"
```

**해결책 2: 상시 VPN 설정**
```
설정 → 연결 → 기타 연결 설정 → VPN
→ Tailscale 옆 톱니바퀴 → "상시 VPN" 활성화
```

**해결책 3: Mac Mini 잠자기 방지**
```bash
# Mac Mini에서
sudo pmset -a sleep 0
sudo pmset -a displaysleep 0
```

### 문제: 연결 속도가 느림

**확인사항:**
```bash
# Mac Mini에서 Tailscale 상태 확인
tailscale status
tailscale netcheck
```

**해결책: DERP 서버 확인**
- 직접 연결(Direct)이 아닌 릴레이 연결인 경우 속도 저하
- 방화벽 설정 확인 (UDP 41641 포트)

---

## SSH 트러블슈팅

### 문제: Connection refused

**원인 1: SSH 서버 비활성화**
```bash
# Mac Mini에서 (로컬 또는 SSH로)
sudo systemsetup -getremotelogin

# SSH 데몬 상태 확인
sudo launchctl list | grep com.openssh.sshd

# 22 포트 리스너 확인
lsof -nP -iTCP:22 -sTCP:LISTEN

# Off인 경우:
sudo systemsetup -setremotelogin on
```

**원인 2: 잘못된 포트**
```
해결: Termius에서 포트 22 확인
```

### 문제: Permission denied (publickey)

**원인 1: 공개키 미등록**
```bash
# Mac Mini에서 authorized_keys 확인
cat ~/.ssh/authorized_keys

# 키가 없으면 Galaxy Tab에서 복사한 공개키 추가
```

**원인 2: 파일 권한 문제**
```bash
# Mac Mini에서 권한 수정
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

**원인 3: 잘못된 사용자명**
```
해결: Mac Mini의 정확한 사용자명 확인
whoami 명령어로 확인
```

### 문제: Host key verification failed

**해결: 알려진 호스트 키 초기화**
```
Termius → 설정 → Known Hosts → 해당 호스트 삭제
```

### 문제: 연결이 자주 끊김 (Broken pipe)

**해결책 1: SSH Keep-Alive 설정 (Mac Mini)**
```bash
# /etc/ssh/sshd_config 편집
sudo nano /etc/ssh/sshd_config

# 추가:
ClientAliveInterval 60
ClientAliveCountMax 3
```

**해결책 2: Termius 설정**
```
Termius → 호스트 설정 → Keep Alive: 활성화
```

**해결책 3: mosh 사용 (더 안정적인 연결)**
```bash
# Mac Mini에 mosh 설치
brew install mosh

# Galaxy Tab에서 mosh 지원 앱 사용 (Blink 등)
```

### 문제: SSH가 실제로 실행 중인지 확인

```bash
# 서비스 목록
sudo launchctl list | grep com.openssh.sshd

# 실행 중 프로세스
pgrep -af sshd

# 로컬에서 포트 점검
lsof -nP -iTCP:22 -sTCP:LISTEN
```

---

## 화면 공유 (Moonlight + Sunshine) 트러블슈팅

### 문제: Moonlight에서 mac-mini가 보이지 않음

**원인 1: Tailscale 연동 불일치**
- Galaxy Tab과 Mac mini가 동일 계정/네트워크인지 확인
- `tailscale status`로 연결 상태 점검

**원인 2: Sunshine 미실행**
```bash
# 실행 여부 확인
ps aux | grep -v grep | grep -i sunshine
```

**원인 3: 서비스 포트 차단**
- Mac mini 방화벽 또는 네트워크 정책에서 47984/47989/47998 접근 제한

### 문제: Pairing PIN 실패

**해결:**
- Mac에서 Sunshine PIN 코드를 새로고침
- Galaxy Tab에서 기존 페어링 삭제 후 재등록
- 기기명/IP 변경 시 주소 재등록

### 문제: 화면은 열리는데 입력이 지연되거나 끊김

**Mac Mini 최적화**
```bash
# 해상도/애니메이션 최적화
defaults write com.apple.universalaccess reduceTransparency -bool true
defaults write NSGlobalDomain NSAutomaticWindowAnimationsEnabled -bool false
```

**Moonlight 최적화**
- Wi-Fi 환경이면 60fps, 유선이면 90fps 또는 1080p 고정
- 모바일 데이터면 30fps/중간 화질로 조정

### 문제: 화면이 잘려 보이거나 모서리가 찢어짐

**원인**
- 해상도/비율이 Moonlight에서 지정한 값과 Galaxy Tab 화면비가 다름
- 기기 전환/회전 후 세션이 이전 스케일을 유지한 상태

**해결**
- Moonlight에서 `1920x1200` 또는 `1600x900`로 재설정 후 재접속
- Sunshine 앱에서 새 스트리밍 프로필을 저장하고 고정
- Android에서 가로 모드 고정 후 `기기별 DPI/글꼴` 재조정

### 문제: 화면이 아예 안 들어올 때 (fallback)

- VNC는 백업 수단으로 유지합니다.
- Mac에서 `화면 공유`를 켜고, Galaxy Tab의 VNC Viewer에서 `100.64.x.x:5900`으로 연결 테스트

### 문제: VNC 백업 연결이 안 될 때

**Mac 점검**
```bash
# 화면 공유 서비스 확인
sudo launchctl list | grep screensharing

# 화면 공유 포트 확인
lsof -nP -iTCP:5900 -sTCP:LISTEN

# 화면 공유 설정 요약(설정 파일이 있으면 표시됨)
defaults read /Library/Preferences/com.apple.ScreenSharing 2>/dev/null
```

**Galaxy Tab 점검**
- VNC 앱이 `VNC` 모드인지 확인(HTML5 모드가 아닌 네이티브 접속)
- 주소 형식: `100.64.x.x:5900`
- 앱의 품질 레벨을 `Medium`부터 시작

**권장 순서**
1. Moonlight 연결을 완전 종료 후 앱 재시작
2. Mac 화면 공유를 재시작
3. VNC 앱에서 새 연결 삭제 후 재등록
4. VNC 비밀번호 재설정 후 재시도

---

## Code-Server 실행 점검

### 문제: code-server 미실행

```bash
# 설치 확인
code-server --version

# 서비스 실행 상태
brew services list | grep code-server

# 실행
brew services start code-server
```

### 문제: code-server 시작이 안 됨

```bash
# 즉시 재시작
brew services restart code-server

# 로그 확인 (기본 경로)
tail -n 50 ~/Library/Logs/code-server/*.log

# 설정 파일 점검
cat ~/.config/code-server/config.yaml

# 포트 바인딩 점검
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

### 문제: 외부에서 접속 안 됨

```bash
# 바인딩 주소/포트 점검
grep -n "^bind-addr" ~/.config/code-server/config.yaml

# health 체크
curl -sSf http://127.0.0.1:8080/healthz | cat
```

---

## 성능 최적화

### 터미널 (SSH) 최적화

**압축 활성화:**
```bash
# Termius 호스트 설정
Compression: 활성화
```

**tmux/screen 사용 (세션 유지):**
```bash
# Mac Mini에 tmux 설치
brew install tmux

# 세션 시작
tmux new -s coding

# 재접속시
tmux attach -t coding
```

### Moonlight 최적화

**macOS 시각 효과 줄이기:**
```bash
# Mac Mini에서
# 투명도 비활성화
defaults write com.apple.universalaccess reduceTransparency -bool true

# 애니메이션 비활성화
defaults write NSGlobalDomain NSAutomaticWindowAnimationsEnabled -bool NO
```

### 네트워크 최적화

**Tailscale 직접 연결 확인:**
```bash
# Mac Mini에서
tailscale status --peers

# "direct" 연결 확인
# "relay" 표시시 방화벽/NAT 설정 확인
```

---

## 일반적인 문제 해결

### 문제: Mac Mini가 잠자기 모드로 전환됨

**해결:**
```bash
# Mac Mini에서 영구 설정
sudo pmset -a sleep 0
sudo pmset -a hibernatemode 0
sudo pmset -a disablesleep 1

# 네트워크 접근시 깨우기
sudo pmset -a womp 1
```

### 문제: 집 인터넷이 끊기면 접속 불가

**해결책 1: 모바일 핫스팟 백업**
- Mac Mini에 USB 테더링 가능한 Android 폰 연결

**해결책 2: Tailscale Funnel (고급)**
```bash
# 외부에서 직접 접속 가능하게 설정
tailscale funnel 22
```

### 문제: Galaxy Tab 배터리 빠른 소모

**해결:**
1. Tailscale: 배터리 최적화 제외
2. Moonlight: 사용 안 할 때 스트리밍 종료
3. 밝기 자동 조절 활성화

### 문제: 한글 입력이 안됨

**SSH (터미널):**
```bash
# Mac Mini .zshrc에 추가
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8
```

**Moonlight/키보드:**
- Mac Mini에서 한글 입력기 설정 확인
- 입력 소스 전환: Caps Lock 또는 Control+Space

---

## 연결 상태 모니터링

### 간단한 상태 확인 스크립트

**Mac Mini에 저장 (~/.local/bin/status.sh):**
```bash
#!/bin/bash

echo "=== 시스템 상태 ==="
echo "시간: $(date)"
echo ""

echo "=== Tailscale ==="
tailscale status | head -5
echo ""

echo "=== SSH 서비스 ==="
sudo systemsetup -getremotelogin
echo ""

echo "=== 화면 공유 (Moonlight) ==="
ps aux | grep -v grep | grep -i sunshine
echo ""

echo "=== 시스템 부하 ==="
uptime
```

**사용:**
```bash
chmod +x ~/.local/bin/status.sh
~/.local/bin/status.sh
```

---

## 긴급 복구 방법

### 원격 접속이 모두 불가능한 경우

1. **가족/친구에게 도움 요청**
   - Mac Mini에서 Tailscale 재시작
   - SSH 서비스 재활성화

2. **iCloud 원격 관리**
   - iCloud.com → 나의 Mac 찾기
   - (제한적 기능만 가능)

3. **예방책: 자동 복구 스크립트**

**crontab 설정 (Mac Mini):**
```bash
crontab -e

# 매 시간 Tailscale 상태 확인 및 재시작
0 * * * * /usr/local/bin/tailscale up 2>/dev/null || open -a Tailscale
```

---

## 체크리스트: 매일 확인

- [ ] Tailscale 연결 상태 (녹색)
- [ ] SSH 연결 테스트
- [ ] Mac Mini 잠자기 상태 아님

## 체크리스트: 주간 확인

- [ ] Tailscale 앱 업데이트
- [ ] macOS 보안 업데이트
- [ ] SSH 로그 확인 (`/var/log/system.log`)
- [ ] 디스크 공간 확인

---

## 도움이 필요한 경우

### 공식 문서
- Tailscale: https://tailscale.com/kb/
- Sunshine: https://github.com/LizardByte/Sunshine
- Moonlight Android: https://play.google.com/store/apps/details?id=com.limelight
- Termius: https://support.termius.com/

### 커뮤니티
- Tailscale Discord: https://tailscale.com/discord
- Reddit r/tailscale: https://reddit.com/r/tailscale
