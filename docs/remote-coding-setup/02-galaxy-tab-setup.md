# Galaxy Tab 클라이언트 설정 가이드

이 문서는 Galaxy Tab에서 `Termius`와 브라우저 기반 `code-server`를 사용하는 전제로 작성되었습니다.

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [Tailscale 설정](#1-tailscale-설정)
3. [Termius 설정](#2-termius-설정)
4. [브라우저에서 code-server 사용](#3-브라우저에서-code-server-사용)
5. [권장 작업 흐름](#4-권장-작업-흐름)
6. [키보드 및 입력 팁](#5-키보드-및-입력-팁)
7. [자주 쓰는 명령어](#6-자주-쓰는-명령어)

---

## 사전 요구사항

- Galaxy Tab S7 이상 권장
- Android 13 이상
- 하드웨어 키보드 권장
- 가능하면 Samsung DeX 사용
- Mac Mini가 [01-mac-mini-setup.md](./01-mac-mini-setup.md) 기준으로 준비된 상태

---

## 1. Tailscale 설정

### 1.1 설치

Play Store에서 `Tailscale` 앱을 설치합니다.

### 1.2 로그인

Mac Mini와 동일한 Tailscale 계정으로 로그인합니다.

### 1.3 권장 설정

Android 설정과 Tailscale 앱에서 아래 항목을 확인합니다.

- 상시 VPN 활성화
- 배터리 사용 제한 해제
- 백그라운드 실행 허용

시스템 설정 경로 예시:

```text
설정 > 연결 > VPN > Tailscale > 상시 VPN
설정 > 앱 > Tailscale > 배터리 > 제한 없음
```

### 1.4 연결 확인

Tailscale 앱에서 아래를 확인합니다.

- `mac-mini`가 기기 목록에 보임
- 상태가 연결됨으로 표시됨
- IP가 `100.x.x.x` 형태로 표시됨

---

## 2. Termius 설정

### 2.1 설치

Play Store에서 `Termius`를 설치합니다.

### 2.2 SSH 키 생성

1. `Settings`
2. `Keychain`
3. `Generate Key`
4. `Type: Ed25519`
5. 이름 예시: `mac-mini-key`

생성 후 `Public Key`를 복사해서 Mac Mini의 `~/.ssh/authorized_keys`에 등록합니다.

### 2.3 호스트 등록

권장 입력값:

| 항목 | 값 |
|------|----|
| Label | `Mac Mini` |
| Address | `mac-mini` |
| Port | `22` |
| Username | Mac 사용자명 |
| Key | `mac-mini-key` |

MagicDNS가 불안정하면 Address에 Tailscale IP를 직접 입력합니다.

### 2.4 첫 연결 테스트

호스트를 눌러 연결한 뒤 아래 명령을 실행합니다.

```bash
whoami
hostname
pwd
```

정상이면 바로 `tmux` 세션을 엽니다.

```bash
tmux new -As work
```

### 2.5 Keep Alive 권장

Termius 호스트 설정에서 `Keep Alive`를 켭니다. 네트워크가 잠깐 흔들려도 세션이 덜 끊깁니다.

### 2.6 SFTP 사용

코드 파일 몇 개를 내려받거나 설정 파일을 확인할 때는 Termius의 SFTP 탭을 사용합니다.

자주 보는 경로:

```text
~/Codes/auto_recipe_creator
~/.ssh
~/.config/code-server
```

---

## 3. 브라우저에서 code-server 사용

### 3.1 어떤 브라우저를 쓸지

`Samsung Internet` 또는 `Chrome` 중 하나로 고정해서 쓰는 편이 좋습니다. 여러 브라우저를 섞어 쓰면 로그인 세션과 다운로드 위치가 자주 꼬입니다.

### 3.2 접속 주소

```text
http://mac-mini:8080
```

또는:

```text
http://100.x.y.z:8080
```

첫 접속 시 Mac Mini에 설정한 `code-server` 비밀번호를 입력합니다.

### 3.3 브라우저 권장 설정

- 데스크톱 사이트 사용
- 탭 자동 절전 기능이 있다면 예외 처리
- 홈 화면 바로가기 또는 북마크 추가

DeX를 쓸 경우에는 Termius와 브라우저를 나란히 띄워 두면 가장 편합니다.

### 3.4 로그인 후 기본 작업

code-server 안에서 아래를 확인합니다.

1. 저장소 폴더 열기
2. 통합 터미널 열기
3. Python 확장 동작 여부 확인
4. Git 상태 표시 확인

통합 터미널에서 바로 쓸 명령:

```bash
cd ~/Codes/auto_recipe_creator
git status
uv run pytest
```

### 3.5 PWA처럼 고정해서 사용

브라우저에 따라 지원 방식은 다르지만, 가능하면 홈 화면 바로가기를 만들어 앱처럼 여는 것이 편합니다.

이유:

- 주소를 다시 입력할 필요가 없음
- 로그인 세션이 안정적임
- 전체 화면 작업이 쉬움

---

## 4. 권장 작업 흐름

### 흐름 1: 빠른 점검

1. Tailscale 연결 확인
2. Termius 접속
3. 아래 명령 실행

```bash
tmux new -As work
cd ~/Codes/auto_recipe_creator
git status
```

### 흐름 2: 코드 수정

1. Termius로 먼저 접속
2. 필요한 서비스 상태 확인
3. 브라우저에서 `code-server` 열기
4. 수정 후 통합 터미널 또는 Termius에서 테스트

### 흐름 3: 연결이 불안정한 날

1. Termius만 먼저 안정화
2. `tmux` 세션에서 작업 시작
3. 편집이 꼭 필요할 때만 code-server 접속

---

## 5. 키보드 및 입력 팁

### 5.1 하드웨어 키보드 권장

장시간 작업은 소프트 키보드보다 외장 키보드가 훨씬 안정적입니다.

### 5.2 Termius에서 유용한 키

| 키 | 동작 |
|----|------|
| `Ctrl + C` | 현재 명령 중단 |
| `Ctrl + D` | 셸 종료 |
| `Ctrl + L` | 화면 정리 |
| `Tab` | 자동 완성 |

### 5.3 한영 전환

한영 전환이 꼬이면 Galaxy Tab과 Mac Mini 양쪽 입력 설정을 같이 봐야 합니다.

Mac Mini에서 확인:

```bash
defaults read ~/Library/Preferences/com.apple.HIToolbox.plist AppleEnabledInputSources 2>/dev/null
```

필요하면 셸 로케일도 같이 확인합니다.

```bash
echo $LANG
echo $LC_ALL
```

### 5.4 DeX 사용 팁

- Termius는 왼쪽, 브라우저는 오른쪽에 두는 구성이 편함
- code-server는 브라우저 확대 비율 `90%~110%` 범위에서 조정
- 화면 회전은 가로 고정 권장

---

## 6. 자주 쓰는 명령어

### 접속 직후

```bash
whoami
hostname
tmux new -As work
cd ~/Codes/auto_recipe_creator
```

### 개발 작업

```bash
git status
git pull
uv sync --extra dev
uv run pytest
```

### code-server 점검

```bash
brew services list | grep code-server
curl -fsS http://127.0.0.1:8080/healthz
```

### SSH / 네트워크 점검

```bash
tailscale status
tailscale ip -4
sudo systemsetup -getremotelogin
```

---

## 설치 완료 체크리스트

- [ ] Galaxy Tab에 Tailscale 설치 및 연결 완료
- [ ] Galaxy Tab에 Termius 설치 완료
- [ ] Ed25519 키 생성 완료
- [ ] Mac Mini에 공개키 등록 완료
- [ ] `mac-mini` 호스트로 SSH 접속 성공
- [ ] `tmux new -As work` 실행 성공
- [ ] 브라우저에서 `http://mac-mini:8080` 접속 성공
- [ ] code-server 로그인 및 저장소 열기 성공
- [ ] DeX 또는 브라우저 확대 비율이 작업 가능한 수준으로 조정됨

---

## 다음 단계

[03-connection-guide.md](./03-connection-guide.md)에서 실제 연결 테스트 순서와 문제 해결 방법을 확인합니다.
