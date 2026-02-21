# Galaxy Tab 클라이언트 설정 가이드

이 문서는 회사에서 사용할 Galaxy Tab을 원격 개발 클라이언트로 설정하는 방법을 설명합니다.

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [Tailscale VPN 설정](#1-tailscale-vpn-설정)
3. [Termius (SSH 클라이언트) 설정](#2-termius-ssh-클라이언트-설정)
4. [Moonlight 설정](#3-moonlight-설정)
5. [VS Code 설정 (DeX 모드)](#4-vs-code-설정-dex-모드)
6. [추가 권장 앱](#5-추가-권장-앱)
7. [키보드 단축키 설정](#6-키보드-단축키-설정)

---

## 사전 요구사항

- Galaxy Tab S7/S8/S9 시리즈 (DeX 지원 모델 권장)
- Android 13 이상
- 외장 키보드 (Bluetooth 또는 USB-C)
- Samsung DeX (선택사항이지만 강력 권장)

---

## 1. Tailscale VPN 설정

### 1.1 Tailscale 설치

**Google Play Store:**
```
Play Store → "Tailscale" 검색 → 설치
```

또는 직접 링크:
```
https://play.google.com/store/apps/details?id=com.tailscale.ipn
```

### 1.2 Tailscale 로그인

1. Tailscale 앱 실행
2. "Get Started" 탭
3. **Mac Mini와 동일한 계정**으로 로그인
   - Google / Microsoft / GitHub / Apple ID

### 1.3 VPN 연결 활성화

1. 앱에서 토글 스위치 ON
2. Android VPN 권한 요청 → "확인"
3. 상태바에 VPN 아이콘 확인

### 1.4 Mac Mini 연결 확인

```
Tailscale 앱 → 기기 목록에서 "mac-mini" 확인
IP 주소 확인 (예: 100.64.x.x)
```

### 1.5 Tailscale 설정 최적화

**앱 설정 → 권장 옵션:**
- "Always-on VPN" → 활성화 (연결 유지)
- "Allow LAN access" → 필요시 활성화

**Android 시스템 설정:**
1. 설정 → 연결 → 기타 연결 설정 → VPN
2. Tailscale 옆 톱니바퀴 → "상시 VPN" 활성화

---

## 2. Termius (SSH 클라이언트) 설정

### 2.1 Termius 설치

**Google Play Store:**
```
Play Store → "Termius" 검색 → 설치
```

또는:
```
https://play.google.com/store/apps/details?id=com.server.auditor.ssh.client
```

### 2.2 SSH 키 생성

**2.2.1 앱 내 키 생성:**
1. Termius 실행
2. 설정 (⚙️) → Keychain
3. "+ Generate Key" 탭
4. 설정:
   - **Type:** Ed25519 (권장) 또는 RSA 4096
   - **Name:** mac-mini-key
   - **Passphrase:** 선택사항 (추가 보안)
5. "Generate" 탭

**2.2.2 공개키 복사:**
1. Keychain → 생성된 키 선택
2. "Export to Clipboard" → Public Key
3. 이 키를 Mac Mini의 `~/.ssh/authorized_keys`에 추가

### 2.3 호스트 추가

1. Hosts 탭 → "+ New Host"
2. 설정 입력:

| 항목 | 값 | 설명 |
|-----|-----|------|
| **Label** | Mac Mini | 표시 이름 |
| **Address** | mac-mini | Tailscale MagicDNS 이름 |
| | 또는 100.64.x.x | Tailscale IP |
| **Port** | 22 | SSH 기본 포트 |
| **Username** | your-mac-username | Mac 사용자 이름 |
| **Password** | (비워둠) | 키 인증 사용시 |
| **Key** | mac-mini-key | 생성한 SSH 키 선택 |

3. "Save" 탭

### 2.4 연결 테스트

1. Hosts → "Mac Mini" 탭
2. 첫 연결시 호스트 키 확인 → "Continue"
3. 터미널 접속 확인

### 2.5 Termius 고급 설정

**터미널 설정 (Settings → Terminal):**
- **Font:** 원하는 폰트 선택
- **Font size:** 14-16 (태블릿 화면 크기에 맞게)
- **Color scheme:** Dracula / One Dark (가독성 좋음)

**키보드 설정 (Settings → Keyboard):**
- **Hardware keyboard:** 활성화
- **Key mapping:** 필요시 커스터마이징

### 2.6 SFTP 사용법 (파일 전송)

1. 호스트 연결 상태에서 하단 "SFTP" 탭
2. 좌측: 로컬 (Galaxy Tab)
3. 우측: 원격 (Mac Mini)
4. 드래그앤드롭으로 파일 전송

---

## 3. Moonlight 설정

### 3.1 Moonlight 설치

**Google Play Store:**
```
Play Store → "Moonlight" 검색 → 설치
```

또는:
```
https://play.google.com/store/apps/details?id=com.limelight
```

### 3.2 새 연결 추가

1. Moonlight 실행
2. `+` 버튼 탭
3. Mac mini Tailscale IP 또는 호스트명(`mac-mini`, `100.64.x.x`) 입력
4. Mac에서 Sunshine PIN 코드 확인 후 입력
5. 페어링 완료

### 3.3 연결 및 스트리밍

1. 기기 선택 후 `Stream` 버튼 탭
2. 해상도/프레임레이트 설정
3. 연결 완료 후 마우스/터치 입력 확인

### 3.4 성능/연결 최적화
**권장 설정:**
- LAN/안정 네트워크: 60fps + 1080p
- 데이터 절약 모드: `30fps`, `Medium`
- 하드웨어 키보드 사용

### 3.5 Galaxy Tab 화면 크기 최적화

### 화면비 우선 설정(권장)
- **세로(세로 모드)**: 개발 작업 시 비추천
- **가로(landscape) 고정**: 텍스트 가독성, 터치 정확도 우수
- 기본적으로 `16:10` 계열 해상도로 고정

### 기기별 권장 시작 해상도(예시)

| 태블릿 | 권장 시작 해상도 | 대안(절전) |
|--------|-----------------|-----------|
| Galaxy Tab S8/S9 계열 | `1920x1200` | `1600x900`, `1280x720` |
| Galaxy Tab S7/유사 모델 | `1280x800` | `1600x900`(Wi-Fi), `1280x720` |

### Moonlight 클라이언트에서 빠른 설정 가이드

1. 연결 가능한 기기 탭에서 `스트리밍 설정` 열기
2. `해상도`를 먼저 고정값으로 지정
3. `FPS`를 `60`에서 시작
4. `대역폭`/품질을 `Medium` 또는 `High` 중 선택
5. 지연이 보이면 `FPS 30`로 즉시 하향

### 가독성 개선
- Moonlight 확대/축소는 크게 쓰지 말고, 대신:
  - Android `설정 > 디스플레이 > 글꼴 크기`를 상향
  - 터미널/에디터 기본 폰트 크기를 2~4 포인트 올림
  - 필요 시 Sunshine 해상도를 `1920x1080`로 내려 가독성 우선

### 3.6 대안(백업): VNC Viewer

Moonlight 연결이 안 되면 VNC Viewer로 임시 접속합니다.
```
https://play.google.com/store/apps/details?id=com.realvnc.viewer.android
```

기본 주소: `mac-mini:5900` 또는 `100.64.x.x:5900`

### 3.7 VNC 추천 앱 (백업)

#### 1순위: RealVNC Viewer (무료/유료)
- 주소: `https://play.google.com/store/apps/details?id=com.realvnc.viewer.android`
- 장점: 초기 설정이 빠르고 안정적

#### 2순위: bVNC (오픈소스)
- 주소: `https://play.google.com/store/apps/details?id=net.christianbeier.bVNC`
- 장점: 경량, 커스텀 옵션 많음

#### 3순위: Jump Desktop
- 주소: `https://play.google.com/store/apps/details?id=com.p5sys.android.jump.free`
- 장점: VNC/RDP 모두 다룰 수 있는 통합형 클라이언트

### 3.8 VNC 설정 체크리스트(백업)

1. `서버 주소`: `mac-mini` 또는 `100.64.x.x:5900`
2. 암호: Mac 화면 공유에서 설정한 VNC 비밀번호 입력
3. 스케일: 화면이 잘릴 때 100% 또는 맞춤 스케일로 맞춤
4. 성능: 이미지 품질을 `중간`으로 시작 후 필요 시 상향
5. 실패 시: Mac의 `화면 공유` 켜짐 + Tailscale 연결 상태 재확인

---

## 4. VS Code 설정 (DeX 모드)

### 4.1 Samsung DeX 활성화

1. Galaxy Tab을 모니터에 연결 또는
2. 빠른 설정 → DeX 탭

### 4.2 VS Code 설치

**방법 1: VS Code for Web (권장)**
```
브라우저 → https://vscode.dev
```

**방법 2: code-server 접속**
Mac Mini에 code-server 설치 후 브라우저로 접속
```
http://mac-mini:8080
```

**방법 3: 네이티브 VS Code (Linux on DeX - 제한적)**
- Samsung Linux on DeX 지원 종료됨
- code-server 사용 권장

### 4.3 VS Code Remote SSH (웹 버전 제한)

⚠️ **참고:** vscode.dev는 Remote SSH를 직접 지원하지 않음

**대안 솔루션:**

**1. code-server 설치 (Mac Mini에서):**
```bash
# Mac Mini에서 실행
brew install code-server

# 설정 파일 수정
code-server --config ~/.config/code-server/config.yaml

# 시작
code-server
```

**config.yaml:**
```yaml
bind-addr: 0.0.0.0:8080
auth: password
password: your-secure-password
cert: false
```

**2. Galaxy Tab에서 접속:**
```
브라우저 → http://mac-mini:8080
```

### 4.4 code-server 자동 시작 (Mac Mini)

**LaunchAgent 생성:**
```bash
# Mac Mini에서
cat << 'EOF' > ~/Library/LaunchAgents/com.code-server.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.code-server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/code-server</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

# 서비스 로드
launchctl load ~/Library/LaunchAgents/com.code-server.plist
```

---

## 5. 추가 권장 앱

### 5.1 터미널 앱 대안

| 앱 | 특징 | 가격 |
|----|------|------|
| **JuiceSSH** | 가볍고 빠름 | 무료/Pro |
| **ConnectBot** | 오픈소스 | 무료 |
| **Blink Shell** | mosh 지원 | 유료 (iOS) |

### 5.2 코드 편집기 (로컬)

| 앱 | 특징 |
|----|------|
| **Acode** | 가벼운 코드 에디터 |
| **QuickEdit** | 빠른 편집 |
| **Dcoder** | 다중 언어 IDE |

### 5.3 Git 클라이언트

| 앱 | 특징 |
|----|------|
| **MGit** | 무료 Git 클라이언트 |
| **Pocket Git** | 기능 풍부 |

### 5.4 파일 관리

| 앱 | 특징 |
|----|------|
| **Solid Explorer** | SFTP 지원 |
| **FX File Explorer** | 네트워크 지원 |

---

## 6. 키보드 단축키 설정

### 6.1 Samsung 키보드 단축키

**DeX 모드 기본 단축키:**
| 단축키 | 동작 |
|--------|------|
| `Ctrl + Esc` | 앱 목록 |
| `Alt + Tab` | 앱 전환 |
| `Cmd + L` | 화면 잠금 |

### 6.2 Termius 단축키

| 단축키 | 동작 |
|--------|------|
| `Ctrl + C` | 인터럽트 |
| `Ctrl + D` | 로그아웃 |
| `Ctrl + L` | 화면 지우기 |
| `Tab` | 자동 완성 |

### 6.3 외장 키보드 설정

**물리 키보드 연결:**
1. Bluetooth 키보드 페어링 또는
2. USB-C 허브 + USB 키보드

**키보드 레이아웃 설정:**
1. 설정 → 일반 → 물리 키보드
2. 키보드 언어 추가 (한/영)
3. 단축키로 언어 전환 설정

---

## 설치 완료 체크리스트

- [ ] Tailscale 설치 및 동일 계정 로그인
- [ ] Tailscale VPN 연결 확인
- [ ] Mac Mini가 기기 목록에 표시됨
- [ ] Termius 설치
- [ ] SSH 키 생성 및 Mac Mini에 등록
- [ ] SSH 연결 성공
- [ ] Moonlight 설치
- [ ] Mac mini 페어링(PIN) 완료
- [ ] Moonlight 스트리밍 성공
- [ ] (백업) VNC Viewer 설치/연결 테스트
- [ ] (선택) code-server 접속 테스트

---

## Claude Code 사용 준비

Galaxy Tab에서 Mac Mini의 Claude Code를 사용하는 방법:

```bash
# Termius로 Mac Mini 접속 후
ssh mac-mini

# Claude Code 실행
claude

# 또는 특정 디렉토리에서
cd ~/projects/my-project
claude
```

---

## 다음 단계

[03-connection-guide.md](./03-connection-guide.md)로 이동하여 연결 테스트 및 트러블슈팅을 진행하세요.
