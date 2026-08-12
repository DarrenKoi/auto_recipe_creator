# poc/workflow_3/workflow_3_config.example.py
"""workflow_3 모니터 설정 템플릿 — 이 파일을 workflow_3_config.py 로 복사해서 쓴다.

왜 분리 파일인가
----------------
workflow_3 루프는 동작을 전부 env(`ALIGN_FAIL_*`/`SAFE_MODE`)로 제어한다(코드 No-CLI 규약).
오피스에서 .env 를 매번 손대거나 긴 `KEY=VAL ... uv run` 한 줄을 외우는 대신, 자주 바꾸는
토글만 상수로 떼어내 한 파일에서 편집한다. 실편집 파일 workflow_3_config.py 는 gitignore 라
"지금 어떤 설정으로 돌리는 중인가" 스크래치가 git 에 안 남는다(템플릿만 추적).

이 파일은 workflow_2 의 golden_eval_config 패턴과 동일하다(상수 → env 브리지, 실 env 우선).

쓰는 법
-------
    1) 이 파일을 같은 폴더에 workflow_3_config.py 로 복사.
       (Windows)  copy poc\\workflow_3\\workflow_3_config.example.py poc\\workflow_3\\workflow_3_config.py
       (bash)     cp   poc/workflow_3/workflow_3_config.example.py poc/workflow_3/workflow_3_config.py
    2) 아래 값을 채운다(바꾸기 싫은 건 None 으로 두면 코드 기본값 유지).
    3) uv run python poc/workflow_3/monitor/align_fail_monitor_only_check.py
       (또는 production: align_fail_monitor.py)

규약
----
  * 값 = None         → env 를 건드리지 않음(코드 기본값 그대로).
  * 값 = 0/1/숫자/문자 → 해당 env 를 setdefault 로 주입.
  * 실제 OS env 가 이미 있으면 그게 우선(setdefault). 즉 워크플로 한 줄
    `ALIGN_FAIL_BLOCK_INPUT=0 uv run ...` 로 이 파일을 일시 오버라이드할 수 있다.
  * workflow_3_config.py 가 없으면 로더가 조용히 폴백(전부 코드 기본값).

주의: ALIGN_IMAGES_DIR 은 패키지 import 시점에 읽혀 이 파일로는 못 바꾼다 — 데이터 루트는
실제 env 또는 기본 경로(poc/workflow_3/align_images)를 쓴다.
"""

# ============================================================================
# [1] FOREGROUND TAKEOVER  — 알람 시 다른 앱을 쓰는 중이어도 RCS 를 강제로 띄우기
# ----------------------------------------------------------------------------
# "잠깐 점유하고 끝나면 돌려주기" 모델의 핵심 3종. 효과를 보려면 **반드시 관리자
# 권한(Run as administrator)으로 터미널을 띄우고** SAFE_MODE=0 으로 둘 것.
# (비관리자면 UIPI 로 강제 전면화/BlockInput 이 조용히 실패 — 시작 로그에 경고 출력.)

# SAFE_MODE: 모든 실제 마우스/키보드 출력 게이트. None=건드리지 않음(코드 기본).
#   takeover 를 실제로 켜려면 0 으로 설정(자동 클릭/wheel/BlockInput 활성).
#   0 으로 두는 게 부담되면 None 으로 두고 워크플로 한 줄에서 SAFE_MODE=0 을 준다.
SAFE_MODE = None

# 자동 GUI 구간(접속~캡처~닫기) 동안 사용자 물리 입력 차단 → 커서가 RCS SEM box 로
# glide 하는 걸 손이 방해하지 못하게 한다. SAFE_MODE=1 이면 자동 no-op 이라 1 로 둬도 안전.
# Ctrl+Alt+Del 로 항상 탈출 가능. 1=on, 0=off.
BLOCK_INPUT = 1

# 모니터링 중 화면/시스템 절전 방지(SetThreadExecutionState). 1=on, 0=off.
KEEP_AWAKE = 1

# tool 창 탐색 최대 시도(점유 'select' 팝업 조기 감지로 보통 3 이면 충분). None=기본(3).
RCS_WINDOW_MAX_TRIALS = None

# ============================================================================
# [2] 알람 폴링 / 소스
# ----------------------------------------------------------------------------
POLL_SEC = None              # 폴링 주기(초). None=기본(10).
WINDOW_SEC = None            # 알람 감지 윈도우(초). None=기본(60).
# "office"=실제 MES, "replay"=CSV 재생(개발 PC dry-run). None=기본(office).
ALARM_SOURCE = None
# replay 소스일 때 재생할 CSV 경로(ALARM_SOURCE="replay" 와 함께). None=미설정.
REPLAY_CSV = None

# ============================================================================
# [3] 알림
# ----------------------------------------------------------------------------
POPUP = None                 # 감지 시 로컬 팝업. 1/0/None(기본 on).
RICH_NOTIFY = None           # 큐브 rich notification. 1/0/None(기본 on).

# ============================================================================
# [4] 점검 모니터(check-only) — 보정 가능성 마킹 / zoom 탐색
# ----------------------------------------------------------------------------
FEASIBILITY_MARK = None      # 캡처 후 보정 가능/불가 마킹(_marked.jpg). 1/0/None(기본 on).
REPOSITION_PREVIEW = None    # align point 로 커서 이동 미리보기(SAFE_MODE=0 필요). 1/0/None(기본 off).
ZOOM_PROBE = None            # 모호/부재 verdict 에서 zoom in/out ladder 탐색. 1/0/None(기본 on).
# zoom 방식: "auto"(wheel→무효 시 PM 드롭다운), "pm_dropdown"(곧장 드롭다운), "wheel"(휠만). None=기본(auto).
ZOOM_METHOD = None
SEM_BOX_DETECT = None        # live SEM box VLM 검출 + PM 모드(OM/SEM) 판정. 1/0/None(기본 on).
PM_DROPDOWN = None           # wheel 무효 tool 용 PM 버튼 드롭다운 fallback. 1/0/None(기본 on).

# ============================================================================
# [5] CV 보정 (production align_fail_monitor 전용)
# ----------------------------------------------------------------------------
CORRECTION = None            # CV 보정 마스터 토글. 1/0/None(기본 on).
# 보정의 실제 reposition/OK 클릭 차단(이중 게이트의 두 번째). 1=dry-run(클릭 안 함),
# 0=실제 클릭(SAFE_MODE=0 도 동시 충족해야 함). None=기본(1=dry-run, 안전).
CORRECTION_DRY_RUN = None

# ============================================================================
# [6] 과거 데이터 수집 / consensus
# ----------------------------------------------------------------------------
GATHER_RCP_MSR = None        # 알람 시 rcp 동기 다운로드(downloader 있을 때). 1/0/None(기본 on).
RCP_GATHER_TIMEOUT_SEC = None  # rcp 동기 다운로드 대기 상한(초). None=기본 60.
GATHER_SUCCESS = None        # 최근 성공 S 이미지 비차단 수집. 1/0/None(기본 on).
CONSENSUS = None             # consensus 라우팅 마스터 토글(off=순수 rcp). 1/0/None(기본 on).
CONSENSUS_MIN_S = None       # consensus build 최소 S(바닥 3). None=기본(4).

# ============================================================================
# [7] 재시도 정책
# ----------------------------------------------------------------------------
FAILURE_COOLDOWN_SEC = None  # 실패 tool 재시도 유예(초). None=기본 300.

# ============================================================================
# [7.5] 상시 녹화 / 엔지니어 watch (production align_fail_monitor 전용)
# ----------------------------------------------------------------------------
# align fail 마다 성공/실패 무관하게 녹화한다. 보정이 성공하지 않은 경우에만 이어서
# 엔지니어 수동 조작을 watch 하며 녹화한다(= 보정을 끄면 항상 watch 한다).
RECORDING_MAX_SEC = None     # 녹화 하드 상한(초). None=기본 900.
ENGINEER_WATCH_SEC = None    # 미보정 watch 상한(초). None=기본 300. 엔지니어 조작을
                             # 끝까지 담고 싶으면 RECORDING_MAX_SEC 와 함께 올린다.
ENGINEER_DONE_DETECT = None  # 우선순위 완료 신호(창 닫힘/Assist/분자 fallback)로 watch 조기 종료. 1/0/None(기본 off).
                            # fallback 임계는 환경변수로만 조정: ALIGN_FAIL_ENGINEER_DONE_ASSIST_UNUSABLE_AFTER=3,
                            # ALIGN_FAIL_ENGINEER_DONE_NUMERATOR_READS=3. 모듈 상수는 추가하지 않는다.

# ============================================================================
# [8] VLM 로케이터 조합 (로그인 / List 탭 / tool 선택 / PM 버튼 공통)
# ----------------------------------------------------------------------------
# 2단계 로케이터의 coarse>fine 서비스 조합(route_slug, 모델명 아님).
# None = 코드 기본값 사용(현재 "mai-ui>mai-ui", vlm/ui_venus_mai_locator.py 의 DEFAULT_*).
# 옛 조합으로 임시 복귀: "ui-venus>mai-ui". 상시 기본을 바꾸려면 코드 상수를 고친다.
#
# env 이름은 VLM_LOCATOR_COMBO (ALIGN_FAIL_* 아님 - rcs/ 단독 스크립트도 같은 스위치를
# 쓴다). config.py 의 Workflow3Settings.locator_combo 로 미러링되어 모니터 시작 로그에
# 찍힌다. 다만 이 파일은 모니터 2종에만 적용되므로, rcs/ 단독 스크립트를 돌릴 때는
# shell 에서 직접 줘야 한다:  $env:VLM_LOCATOR_COMBO = "mai-ui>mai-ui"
LOCATOR_COMBO = None
