# 연결 테스트 및 트러블슈팅 가이드

이 문서는 Mac Mini와 Galaxy Tab 간의 연결을 테스트하고 발생할 수 있는 문제를 해결하는 방법을 설명합니다.

## 목차

1. [연결 테스트 순서](#연결-테스트-순서)
2. [Tailscale 트러블슈팅](#tailscale-트러블슈팅)
3. [SSH 트러블슈팅](#ssh-트러블슈팅)
4. [VNC 트러블슈팅](#vnc-트러블슈팅)
5. [성능 최적화](#성능-최적화)
6. [일반적인 문제 해결](#일반적인-문제-해결)

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

### Step 4: VNC 연결 테스트

**VNC Viewer에서:**
1. "Mac Mini" 연결 탭
2. VNC 암호 입력
3. macOS 데스크톱 화면 확인

**테스트 항목:**
- [ ] 화면이 정상적으로 표시됨
- [ ] 마우스 커서 움직임
- [ ] 키보드 입력 동작
- [ ] 스크롤 동작

### Step 5: Claude Code 테스트

**SSH 접속 후:**
```bash
# Claude Code 실행 테스트
claude "안녕하세요, 연결 테스트입니다."

# 프로젝트 디렉토리에서 테스트
cd ~/projects
claude "현재 디렉토리 구조를 설명해주세요."
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
# Mac Mini에서 (로컬 또는 VNC로)
sudo systemsetup -getremotelogin
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

---

## VNC 트러블슈팅

### 문제: Unable to connect

**원인 1: 화면 공유 비활성화**
```
Mac Mini: 시스템 설정 → 일반 → 공유 → 화면 공유 확인
```

**원인 2: 잘못된 포트**
```
VNC Viewer: 주소에 :5900 포트 명시
예: 100.64.x.x:5900 또는 mac-mini:5900
```

**원인 3: 방화벽 차단**
```bash
# Mac Mini에서 방화벽 예외 추가
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /System/Library/CoreServices/Screen\ Sharing.app
```

### 문제: Authentication failed

**해결: VNC 암호 재설정**
```
Mac Mini: 시스템 설정 → 일반 → 공유
→ 화면 공유 (i) → 컴퓨터 설정 → 암호 재설정
```

### 문제: 화면이 검게 표시됨

**원인: macOS 화면 보호기/잠금**
```bash
# Mac Mini에서 화면 보호기 비활성화
defaults write com.apple.screensaver idleTime 0

# 또는 VNC 연결 후 암호 입력하여 잠금 해제
```

### 문제: 화면이 느리거나 깨짐

**해결책 1: 화질 설정 낮춤**
```
VNC Viewer → 연결 설정 → Picture quality: Medium 또는 Low
```

**해결책 2: 해상도 조정**
```
Mac Mini: 시스템 설정 → 디스플레이 → 해상도 낮춤
```

**해결책 3: 애니메이션 비활성화**
```bash
# Mac Mini에서
defaults write NSGlobalDomain NSAutomaticWindowAnimationsEnabled -bool false
defaults write -g QLPanelAnimationDuration -float 0
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

### VNC 최적화

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
2. VNC: 사용 안할 때 앱 종료
3. 밝기 자동 조절 활성화

### 문제: 한글 입력이 안됨

**SSH (터미널):**
```bash
# Mac Mini .zshrc에 추가
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8
```

**VNC:**
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

echo "=== 화면 공유 ==="
sudo launchctl list | grep screensharing
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
- RealVNC: https://help.realvnc.com/
- Termius: https://support.termius.com/

### 커뮤니티
- Tailscale Discord: https://tailscale.com/discord
- Reddit r/tailscale: https://reddit.com/r/tailscale
