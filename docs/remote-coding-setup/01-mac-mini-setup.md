# Mac Mini 서버 설정 가이드

이 문서는 집에 있는 Mac Mini를 원격 개발 서버로 설정하는 방법을 설명합니다.

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [Tailscale VPN 설정](#1-tailscale-vpn-설정)
3. [SSH 서버 설정](#2-ssh-서버-설정)
4. [화면 공유 (VNC) 설정](#3-화면-공유-vnc-설정)
5. [Claude Code 설치](#4-claude-code-설치)
6. [추가 보안 설정](#5-추가-보안-설정)
7. [자동 시작 설정](#6-자동-시작-설정)

---

## 사전 요구사항

- macOS Sonoma (14.0) 이상 또는 Sequoia (15.0)
- 관리자 계정 접근 권한
- 안정적인 인터넷 연결
- Apple ID (Tailscale 로그인용)

---

## 1. Tailscale VPN 설정

### 1.1 Tailscale 설치

**방법 1: Mac App Store (권장)**
```
Mac App Store에서 "Tailscale" 검색 → 설치
```

**방법 2: Homebrew**
```bash
brew install --cask tailscale
```

**방법 3: 공식 웹사이트**
```
https://tailscale.com/download/mac
```

### 1.2 Tailscale 로그인

1. 메뉴바의 Tailscale 아이콘 클릭
2. "Log in" 클릭
3. 브라우저에서 인증 방법 선택:
   - Google 계정
   - Microsoft 계정
   - GitHub 계정
   - Apple ID (권장 - Apple 생태계 사용시)

### 1.3 Tailscale 설정 최적화

**MagicDNS 활성화 (Tailscale 관리 콘솔)**
```
https://login.tailscale.com/admin/dns
```
- "Enable MagicDNS" 활성화
- 이후 IP 대신 `mac-mini` 같은 호스트명 사용 가능

**기기명 설정**
```bash
# 터미널에서 실행
sudo tailscale set --hostname=mac-mini
```

### 1.4 Tailscale IP 확인

```bash
tailscale ip -4
# 예: 100.64.x.x
```

### 1.5 연결 상태 확인

```bash
tailscale status
```

---

## 2. SSH 서버 설정

### 2.1 원격 로그인 활성화

**GUI 방법:**
1. 시스템 설정 → 일반 → 공유
2. "원격 로그인" 활성화
3. "허용된 사용자" → "모든 사용자" 또는 특정 사용자 선택

**터미널 방법:**
```bash
# 원격 로그인 활성화
sudo systemsetup -setremotelogin on

# 상태 확인
sudo systemsetup -getremotelogin
```

### 2.2 SSH 키 인증 설정 (강력 권장)

**2.2.1 SSH 키 저장 디렉토리 준비**
```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

**2.2.2 authorized_keys 파일 생성**
```bash
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**2.2.3 공개키 등록**

Galaxy Tab에서 생성한 공개키를 이 파일에 추가합니다.
(Galaxy Tab 설정 가이드 참조)

```bash
# 공개키 추가 (예시)
echo "ssh-ed25519 AAAAC3Nza... your-key-comment" >> ~/.ssh/authorized_keys
```

### 2.3 SSH 보안 강화 (선택사항)

**/etc/ssh/sshd_config 수정:**

```bash
# 설정 파일 백업
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# 설정 파일 편집
sudo nano /etc/ssh/sshd_config
```

**권장 설정:**
```bash
# 패스워드 인증 비활성화 (키 인증 설정 후)
PasswordAuthentication no

# 루트 로그인 비활성화
PermitRootLogin no

# 빈 패스워드 비활성화
PermitEmptyPasswords no

# 공개키 인증 활성화
PubkeyAuthentication yes
```

**변경사항 적용:**
```bash
# SSH 서비스 재시작
sudo launchctl stop com.openssh.sshd
sudo launchctl start com.openssh.sshd
```

### 2.4 SSH 연결 테스트 (로컬)

```bash
ssh localhost
```

---

## 3. 화면 공유 (VNC) 설정

### 3.1 화면 공유 활성화

**GUI 방법:**
1. 시스템 설정 → 일반 → 공유
2. "화면 공유" 활성화
3. "허용된 사용자" 설정

**터미널 방법:**
```bash
# 화면 공유 활성화
sudo defaults write /var/db/launchd.db/com.apple.launchd/overrides.plist com.apple.screensharing -dict Disabled -bool false
sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.screensharing.plist
```

### 3.2 VNC 암호 설정

1. 시스템 설정 → 일반 → 공유
2. 화면 공유 옆의 (i) 버튼 클릭
3. "컴퓨터 설정..." 클릭
4. "VNC 뷰어에서 암호로 화면을 제어할 수 있음" 체크
5. 강력한 암호 설정

### 3.3 화면 공유 옵션 설정

**권장 설정:**
- "누군가 화면 제어를 요청하면" → "허용 요청"
- 또는 특정 사용자만 허용하여 보안 강화

### 3.4 VNC 포트 확인

```bash
# 기본 포트: 5900
netstat -an | grep 5900
```

---

## 4. Claude Code 설치

### 4.1 Node.js 설치 (필수)

**Homebrew 사용:**
```bash
# Homebrew 설치 (없는 경우)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Node.js 설치 (LTS 버전)
brew install node@20

# PATH 설정 (zsh 사용시)
echo 'export PATH="/opt/homebrew/opt/node@20/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 버전 확인
node --version  # v20.x.x
npm --version
```

### 4.2 Claude Code 설치

```bash
# Claude Code CLI 설치
npm install -g @anthropic-ai/claude-code

# 설치 확인
claude --version
```

### 4.3 Claude Code 인증

```bash
# Claude Code 로그인
claude login

# 또는 API 키 직접 설정
export ANTHROPIC_API_KEY="your-api-key-here"

# 영구 설정 (zsh)
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.zshrc
```

### 4.4 Claude Code 테스트

```bash
# 테스트 실행
claude "Hello, can you see this?"
```

---

## 5. 추가 보안 설정

### 5.1 macOS 방화벽 활성화

```bash
# 방화벽 활성화
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on

# 스텔스 모드 활성화 (ping 응답 차단)
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setstealthmode on

# 상태 확인
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

### 5.2 Tailscale ACL 설정 (고급)

Tailscale 관리 콘솔에서 ACL 설정:
```
https://login.tailscale.com/admin/acls
```

**예시 ACL (본인 기기만 접근 허용):**
```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["*"],
      "dst": ["*:*"]
    }
  ],
  "ssh": [
    {
      "action": "accept",
      "src": ["autogroup:members"],
      "dst": ["autogroup:self"],
      "users": ["autogroup:nonroot"]
    }
  ]
}
```

### 5.3 2단계 인증 설정

**Tailscale 2FA:**
1. https://login.tailscale.com/admin/settings/security
2. "Two-factor authentication" 활성화

**Apple ID 2FA:**
- 이미 활성화되어 있는 것이 좋음
- 시스템 설정 → Apple ID → 암호 및 보안

---

## 6. 자동 시작 설정

### 6.1 Mac Mini 자동 부팅 설정

**전원 복구 시 자동 시작:**
```bash
# Intel Mac
sudo systemsetup -setrestartfreeze on
sudo systemsetup -setrestartpowerfailure on

# Apple Silicon Mac (M1/M2/M3/M4)
# 시스템 설정 → 에너지 절약 → "정전 후 자동으로 시작"
```

### 6.2 로그인 시 자동 실행 앱

**Tailscale 자동 시작:**
1. 시스템 설정 → 일반 → 로그인 항목
2. "+" 버튼 → Tailscale 앱 추가

### 6.3 잠자기 방지 설정

```bash
# 디스플레이 잠자기 비활성화
sudo pmset -a displaysleep 0

# 시스템 잠자기 비활성화
sudo pmset -a sleep 0

# 네트워크 접근 시 깨우기
sudo pmset -a womp 1

# 현재 설정 확인
pmset -g
```

### 6.4 자동 로그인 설정 (선택사항)

⚠️ **보안 주의:** 물리적 접근이 불가능한 환경에서만 사용

1. 시스템 설정 → 사용자 및 그룹
2. 자동 로그인 사용자 설정

---

## 설치 완료 체크리스트

- [ ] Tailscale 설치 및 로그인 완료
- [ ] Tailscale IP 확인 (100.x.x.x)
- [ ] SSH 서버 활성화
- [ ] SSH 키 인증 설정 (선택)
- [ ] 화면 공유 활성화
- [ ] VNC 암호 설정
- [ ] Claude Code 설치 및 인증
- [ ] 방화벽 활성화
- [ ] 자동 시작 설정

---

## 다음 단계

[02-galaxy-tab-setup.md](./02-galaxy-tab-setup.md)로 이동하여 Galaxy Tab 설정을 진행하세요.
