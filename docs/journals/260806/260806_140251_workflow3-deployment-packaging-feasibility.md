# workflow_3 배포/패키징 방식 타당성 검토 (브레인스토밍)

- 일시: 2026-08-06 14:02 KST
- 상태: **설계 진행 중 — 사용자 결정 대기로 중단** (코드 변경 없음)
- 스킬: `superpowers:brainstorming` (설계 승인 전 구현 금지 게이트 적용 중)

## 1. 진행 사항

사용자 질문: "이 패키지를 실사용 엔지니어에게 배포하려면 installer 나 standalone exe 로 묶을 수 있나? 배포 방식 타당성을 확인하고 싶다."

### 1-1. 리포 사실 조사 (배포 관점)

- **workflow_3 실제 런타임 서드파티 의존성 추출** — `poc/workflow_3/` 전체 import 스캔 결과:
  `numpy`, `cv2`(opencv-contrib), `PIL`, `mss`, `pynput`, `pywinauto`, `requests`, `dotenv`, `psutil`, `pandas`.
  → **torch/transformers/easyocr/ultralytics 없음.** 이들은 `test/video_frame_parser` 쪽이며
  루트 `requirements.txt` 는 모노레포 superset(제품 의존성 아님).
  → 배포 대상 표면이 생각보다 작음. onedir freeze 시 150~250MB 수준(opencv+numpy+pandas 지배).
- `pyproject.toml` 확인 — hatchling 백엔드, `packages = ["poc","test","flask_api","gpu_dashboard"]`,
  `requires-python >=3.10`. `[project.dependencies]` 는 이미 lean 한 목록.
- `.gitignore` 확인 — `**/office_*`(3행), `poc/workflow_3/workflow_3_config.py`(93행).
- `uv.lock` 존재. `.env.example` 만 있고 `.env` 없음.
- 기존 패키징 시도 흔적(PyInstaller/Nuitka/cx_freeze/Inno Setup) **없음** — 신규 작업.

### 1-2. 발견한 배포 blocker (설계에 반영해야 할 사실)

1. **데이터 루트가 패키지 경로 파생** — `ALIGN_IMAGES_DIR` 이 `poc/workflow_3/__init__.py` 의
   `WORKFLOW_3_DIR` 기준. PyInstaller freeze 시 `sys._MEIPASS`(임시 추출 디렉토리)로 해석되어
   `align_images/`, `debug_images/`, `logs/` 가 종료 시 삭제되는 폴더에 조용히 기록됨.
   → **빌드 플래그가 아니라 코드 변경이 필요한 사안.** wheel 배포(site-packages)에도 동일하게 적용.
2. **office_* 모듈은 오피스 PC 전용(gitignore)** — 리포만으로 빌드하면 MES/cube 연동이
   자동 비활성인 바이너리가 나옴. freeze 하려면 **빌드를 오피스에서** 하거나 별도 plug-in slot 필요.
3. **`workflow_3_config.py` 는 edit-often scratch** — freeze 하면 안 되고 디스크에 남아야 함.
4. check-only 모니터는 **관리자 권한 필요**(UIPI/BlockInput, `print_elevation_status` 경고).

### 1-3. 요구사항 인터뷰 (3문 1답씩)

| 질문 | 사용자 답변 |
|---|---|
| 왜 `git pull` + `uv sync` 로는 안 되나? | **"Python 은 되지만 git/uv 커멘드 자체가 마찰"** — 비개발자 엔지니어에게 리포 pull·.env 관리·uv 명령은 과함. 더블클릭 런처를 원함 |
| 사용자가 실제로 실행하는 건? | **check-only 모니터 + 원샷 진단 도구.** 장기적으로는 엔지니어와 함께 일하는 agent 지향. **현재 PoC 단계** |
| 대상 PC 상태 / IT 승인? | **"아무것도 안 깔려 있음. 단 uv 설치는 강제 가능. PyPI 는 프록시 미러 있음"** |

### 1-4. 옵션 3안 비교 및 권고

- **A. uv-native 런처 (권고)** — self-elevate `.bat` → `uv run --frozen`. 비용 ~1일,
  패키징 리스크 0, gitignore 경계와 자연히 호환, 코드 변경 반영 = `git pull`.
- **B. wheel + `uv tool install`** — `[project.scripts]` 추가하면 단일 `.whl` 배포 가능.
  단 데이터 루트 이전(blocker #1)이 선결 조건. 업데이트 = 재설치.
- **C. PyInstaller onedir + Inno Setup** — 진짜 "Python 불필요" 해답이나 PoC 단계엔 과잉.
  오피스에서만 빌드 가능, blocker #1 코드 수정 필수, onefile 금지(매 실행 200MB 임시 추출 +
  fab 백신 오탐), threshold 튜닝마다 rebuild-redeploy.
- **권고: 지금 A → check-only 안정화 후 B → IT 가 Python 을 막을 때만 C.**
- 근거 보강: 장기 목표인 "엔지니어와 함께 일하는 agent" 는 freeze 와 상극 —
  프롬프트/threshold/VLM 라우팅이 빌드 타임에 고정되면 진화 루프가 길어짐.

### 1-5. "왜 pip 이 아니라 uv 냐" 재검토

사용자 반문에 따라 앵커링 여부를 재검토하고 정직하게 양측 정리:

- uv 쪽 실제 근거: ① `uv.lock` 이 이미 있고 pip 은 읽지 못함(`uv export` 로 별도 생성·동기화 필요)
  ② **uv 는 인터프리터를 설치할 수 있고 pip 은 못 함** ③ venv 생성 암묵 처리
  ④ CLAUDE.md 가 리포 전역에 `uv run python` 을 규정.
- pip 쪽 정당한 근거: IT 화이트리스트 이미 통과, 사내 미러 지정 간단, `requirements.txt` 가 감사에 유리.
- **pip 채택 시 주의**: 루트 `requirements.txt` 설치 금지(torch 계열 1GB+ 무관 의존성).
  `requirements-workflow3.txt` 를 별도 작성·유지해야 함.
- **최종 결론: 대상 PC 에 Python 이 아예 없으므로 pip 은 탈락**(pip 을 돌릴 Python 자체가 없음).
  uv 확정 — 앵커링이 아니라 제약에서 도출.

### 1-6. 오피스 환경 검증 (`uv 0.9.26` 로컬 확인)

**핵심 발견: PyPI 미러는 인터프리터를 커버하지 않는다.**

- `uv help python install` 확인 — `--mirror` 는
  `https://github.com/astral-sh/python-build-standalone/releases/download` 를 치환.
  즉 CPython 은 **GitHub 릴리스**에서 받으며 패키지 인덱스와 무관.
  fab 에서 GitHub 이 프록시 차단이면 `uv sync` 가 의존성 해결까지 성공한 뒤 인터프리터에서 실패.
- 완화책 확인 — `--mirror` 는 **`file://` URL scheme 을 명시 지원** → 공유 드라이브의 CPython
  tarball 로 인터넷 의존 제거 가능. env: `UV_PYTHON_INSTALL_MIRROR`.
- `uv help sync` 확인 — `--default-index`(구 `--index-url` 은 deprecated), `--frozen`(`UV_FROZEN`),
  `--offline`(`UV_OFFLINE`) 존재 확인.
- uv 자체도 Python 없이 `pip install uv` 불가 → **단일 static `.exe`** 를 공유 드라이브에서 복사.
- **부트스트랩 소스가 3개이고 그중 하나만 사내 미러**:
  `uv.exe`(공유 드라이브) → CPython(공유 드라이브, `UV_PYTHON_INSTALL_MIRROR`) →
  패키지(사내 PyPI 프록시, `UV_DEFAULT_INDEX`). 하나만 틀려도 "내 PC 는 되는데 2번 PC 는 실패".

### 1-7. 제시한 설계 (섹션 1~2까지, 승인 대기)

**섹션 1 — 산출물 형태: installer 도 exe 도 아닌 "portable 폴더"**

```
align-tool\
├─ uv.exe                        ← 단일 static 바이너리, 설치 불필요
├─ python\cpython-3.12...tar.gz  ← UV_PYTHON_INSTALL_MIRROR 용 인터프리터
├─ repo\                         ← 체크아웃 또는 export 한 zip
├─ env.cmd                       ← 위 3개 소스를 한 곳에서 설정
├─ 1_check_monitor.cmd           ← self-elevating
├─ 2_diagnostics.cmd             ← 원샷 도구 메뉴
└─ README_office.txt
```

시스템 무변경, 설치 시 관리자 불필요(실행 시에만 필요), IT 검토용으로 넘기기 쉬운 드래그앤드롭 산출물.

**섹션 2 — 부트스트랩 체인**

`env.cmd` 가 `UV_DEFAULT_INDEX` / `UV_PYTHON_INSTALL_MIRROR` / `UV_FROZEN=1` 설정 →
각 런처가 이를 로드하고 `uv run python poc\workflow_3\...` 호출.
첫 실행만 sync 비용, 이후 즉시. `UV_FROZEN=1` 로 조용한 재해석(drift) 차단 — `uv.lock` 그대로 쓰거나 명시적 실패.

## 2. 수정 내용

**코드 변경 없음.** 조사·설계 단계이며 brainstorming 스킬의 "설계 승인 전 구현 금지" 게이트를 지키는 중.
생성 파일은 이 저널 1개:

- `docs/journals/260806/260806_140251_workflow3-deployment-packaging-feasibility.md` (신규)

## 3. 다음 단계

### 3-1. 사용자 답변 대기 중인 미결 질문 (재개 지점)

> **repo 를 체크아웃으로 둘 것인가, export 한 zip 으로 둘 것인가?**
> - 체크아웃: 업데이트가 `git pull` 로 끝남. 단 대상 PC 에 git 필요 + `office_*` 와
>   `workflow_3_config.py` 가 엔지니어가 편집할 수 있는 위치에 노출됨.
> - zip: 노출 없음. 단 업데이트 경로를 따로 설계해야 함.

이 답을 받으면 브레인스토밍 섹션 3 이후를 이어서 제시:

1. **설계 섹션 3** — config 처리 (`workflow_3_config.py` 를 번들 밖 설치 경로에서 읽기,
   `.env` 초회 부트스트랩)
2. **설계 섹션 4** — 권한 상승 (PowerShell `Start-Process -Verb RunAs` self-elevate 또는
   "관리자 권한으로 실행" 체크된 바로가기)
3. **설계 섹션 5** — 업데이트 경로 (3-1 답변에 종속)
4. **설계 섹션 6** — 검증 방법 (깨끗한 2번째 PC 에서 부트스트랩 3소스 전부 확인)
5. 승인 후 → `docs/superpowers/specs/2026-08-06-workflow3-deployment-design.md` 작성 + 커밋
6. → spec self-review → 사용자 spec 리뷰 → `superpowers:writing-plans` 로 구현 계획

### 3-2. 오피스에서 사전 확인이 필요한 사항 (Mac 에서 불가)

- [ ] fab 네트워크에서 `github.com/astral-sh/python-build-standalone` 접근 가능 여부
      → 차단이면 `UV_PYTHON_INSTALL_MIRROR` + 공유 드라이브 tarball 이 **필수**
- [ ] 사내 PyPI 프록시의 정확한 index URL (그리고 인증 필요 여부)
- [ ] 엔지니어 PC 에서 UAC 승격이 정책상 허용되는지 (check-only 모니터 필수 조건)
- [ ] 공유 드라이브 경로 및 엔지니어 읽기 권한

### 3-3. 옵션 B/C 로 갈 경우의 선결 과제 (지금은 착수 안 함)

- `ALIGN_IMAGES_DIR` / `debug_images` / `logs` 의 데이터 루트를 패키지 경로에서
  사용자 데이터 경로(예: `%LOCALAPPDATA%` 또는 명시적 설치 경로)로 이전.
  **B·C 공통 선결 조건이며 그 자체로도 가치 있는 정리.**

## 4. 메모리 업데이트

이번 세션에서 확정된 것은 "조사 결과 + 방향 권고"이며 아키텍처/컨벤션 변경은 아직 없음.
다만 아래 2건은 재발견 비용이 큰 사실이라 **설계 승인 시점에 메모리 반영 후보**:

1. **workflow_3 의 실제 런타임 의존성은 10개이며 루트 `requirements.txt` 는 모노레포 superset** —
   배포/설치 논의 때마다 다시 스캔하게 되는 사실.
2. **uv 의 CPython 은 PyPI 가 아니라 GitHub 릴리스에서 온다** — 사내 PyPI 미러가 있어도
   인터프리터는 별도 미러(`UV_PYTHON_INSTALL_MIRROR`, `file://` 지원)가 필요.

현 시점 `MEMORY.md` 변경: **없음** (설계 미승인 상태이므로 보류).
