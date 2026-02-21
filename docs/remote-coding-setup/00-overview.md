# 원격 코딩 환경 구축 가이드

## 개요

집에 있는 Mac Mini를 서버로, 회사에서 Galaxy Tab을 클라이언트로 사용하여 원격 코딩 환경을 구축하는 가이드입니다.

## 시스템 구성도

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              인터넷                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │         Tailscale VPN         │
                    │    (암호화된 P2P 터널링)        │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   SSH (22)    │           │ Moonlight(47984)│         │ VS Code SSH   │
│   터미널 접속  │           │   GUI 접속    │           │   원격 개발    │
└───────────────┘           └───────────────┘           └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Mac Mini (서버)          │
                    │  - macOS Sonoma/Sequoia      │
                    │  - Claude Code CLI           │
                    │  - 개발 환경                  │
                    └───────────────────────────────┘
```

## 사용 시나리오

| 작업 유형 | 권장 도구 | 특징 |
|----------|----------|------|
| Claude Code 사용 | Termius / SSH | 터미널 기반, 빠른 응답 |
| 코드 편집 | VS Code Remote SSH | 완전한 IDE 경험 |
| 시스템 관리 | Moonlight + Sunshine | 저지연 GUI 접속 |
| 파일 전송 | Termius SFTP | 간편한 드래그앤드롭 |

## 핵심 소프트웨어 스택

### 네트워크 레이어 (VPN)

| 소프트웨어 | 설명 | 보안 |
|-----------|------|------|
| **Tailscale** | WireGuard 기반 메시 VPN | AES-256, 제로 트러스트 |
| ZeroTier | P2P 가상 네트워크 | AES-256-GCM |
| Cloudflare Tunnel | 터널링 서비스 | TLS 1.3 |

**권장: Tailscale**
- 설정이 가장 간단함
- WireGuard 프로토콜 사용 (최신 암호화)
- 무료 플랜으로 개인 사용 충분
- MagicDNS로 IP 대신 호스트명 사용 가능

### 터미널 접속 (SSH)

| 소프트웨어 | 플랫폼 | 특징 |
|-----------|--------|------|
| **Termius** | Android/iOS/Mac/Windows | 크로스플랫폼, SFTP 지원, 동기화 |
| JuiceSSH | Android | 무료, 가벼움 |
| Prompt 3 | iOS | Apple 생태계 최적화 |

**권장: Termius**
- 태블릿 UI 최적화
- 키보드 단축키 지원
- SFTP 파일 관리 내장
- 클라우드 동기화 (유료)

### GUI 접속 (원격 데스크톱)

| 소프트웨어 | 설명 | 특징 |
|-----------|------|------|
| **Moonlight + Sunshine** | 게임 스트리밍 방식 원격 데스크톱 | 저지연 스트리밍, 입력 지연이 적음 |
| Jump Desktop | 고성능 원격 데스크톱 | Fluid 프로토콜 지원 |
| RealVNC (Apple Screen Sharing) | macOS 기본 화면 공유 기반 | 설정이 빠르고 호환성 좋음 |
| Screens 5 | macOS 전용 솔루션 | 고품질 화면 공유 |
| VNC Viewer (Android) | VNC 전용 클라이언트 | 백업 연결에 안정적 |

**권장: Moonlight + Sunshine**
- Android에서 Moonlight 앱으로 Mac Mini 스트리밍
- VNC는 백업 경로로 운영. Android는 RealVNC + bVNC/Jump Desktop으로 실패 시 대응

### VNC 백업 권장 조합

- **Mac 서버:** `시스템 설정 → 공유 → 화면 공유`(기본) + Tailscale 터널링
- **Android 클라이언트:** `VNC Viewer` (1순위), `bVNC`, `Jump Desktop`
- **선택:** RealVNC Server 패키지(상용)로 접속 방식 통합 관리

### 코드 편집 (IDE)

| 소프트웨어 | 설명 | 특징 |
|-----------|------|------|
| **VS Code + Remote SSH** | 원격 개발 | 완전한 IDE 경험 |
| Code-Server | 웹 기반 VS Code | 브라우저에서 접속 |
| JetBrains Gateway | JetBrains IDE 원격 | 고급 리팩토링 |

**권장: VS Code Remote SSH**
- 로컬과 동일한 개발 경험
- 확장 프로그램 지원
- 터미널 통합

## 설치 순서 (권장)

### Phase 1: 네트워크 구축 (필수)
1. Mac Mini에 Tailscale 설치 및 로그인
2. Galaxy Tab에 Tailscale 설치 및 동일 계정 로그인
3. 연결 테스트 (ping)

### Phase 2: 터미널 환경 (Claude Code용)
4. Mac Mini SSH 서버 활성화
5. Galaxy Tab에 Termius 설치
6. SSH 연결 설정 및 테스트
7. Claude Code 설치 및 테스트

### Phase 3: GUI 환경 (선택)
8. Mac Mini에서 Sunshine 설치 및 실행 설정
9. Galaxy Tab에 Moonlight 앱 설치
10. Moonlight 스트리밍 연결 테스트

### Phase 4: IDE 환경 (선택)
11. Galaxy Tab에 VS Code 설치 (DeX 모드용)
12. Remote SSH 확장 설치
13. 원격 개발 환경 테스트

## 보안 체크리스트

- [ ] Tailscale 2FA 활성화
- [ ] SSH 키 인증 설정 (패스워드 비활성화)
- [ ] macOS 방화벽 활성화
- [ ] Sunshine 인증 패스워드 설정
- [ ] Tailscale ACL 설정 (필요시)

## 문서 구성

| 문서 | 설명 |
|------|------|
| [01-mac-mini-setup.md](./01-mac-mini-setup.md) | Mac Mini 서버 설정 |
| [02-galaxy-tab-setup.md](./02-galaxy-tab-setup.md) | Galaxy Tab 클라이언트 설정 |
| [03-connection-guide.md](./03-connection-guide.md) | 연결 테스트 및 트러블슈팅 |

## 예상 소요 시간

- Phase 1 (네트워크): 15-20분
- Phase 2 (터미널): 20-30분
- Phase 3 (GUI): 10-15분
- Phase 4 (IDE): 15-20분

**총 예상 시간: 1-1.5시간**
