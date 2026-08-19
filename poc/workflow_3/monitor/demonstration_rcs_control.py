"""RCS 자동 조작 시연 스크립트 - 알람 없이 정해진 시나리오를 순서대로 보여준다.

알람을 기다리지 않고, 사람이 보는 앞에서 "이 시스템이 RCS 를 이렇게 몹니다" 를
재생하는 것이 목적이다. 시나리오(사용자 지정, 2026-08-19):

    RCS 실행 -> 로그인 -> View 탭 + 휠로 위아래 훑기 -> List 탭
      -> MCD019 접속 -> [Optics... -> Memory 탭 -> Close] -> tool 창 닫기
      -> MCDC22 접속 -> [Optics... -> Memory 탭 -> Close] -> tool 창 닫기

`align_fail_monitor_only_check.py` 로도 replay CSV 를 물려 비슷한 것을 할 수 있지만,
그쪽은 알람 사이클(manifest/cube 알림/보정 가능성 판정)을 통째로 끌고 오고 장비를
**하나씩만** 처리한다. 시연에 필요한 것은 여러 장비 순회와 화면 체류 시간이라
별도 진입점으로 둔다.

**이 스크립트는 보정을 하지 않는다.** 하는 일은 탭 클릭 / 휠 / tool 더블클릭 /
Optics 대화상자 열고 닫기 / 창 닫기뿐이다. reposition·OK 클릭처럼 레시피나 측정에
영향을 주는 조작은 전혀 없다.

사용법 (오피스 Windows):

    uv run python poc/workflow_3/monitor/demonstration_rcs_control.py

장비 목록만 바꿔서:

    DEMO_RCS_TOOL_IDS="MCD019,MCDC22,MCD916" \
      uv run python poc/workflow_3/monitor/demonstration_rcs_control.py

env (`DEMO_RCS_*` 네임스페이스 - 루프의 `ALIGN_FAIL_*` 과 섞지 않는다):

    DEMO_RCS_TOOL_IDS       접속할 장비 (콤마/공백 구분, 기본 "MCD019,MCDC22")
    DEMO_RCS_DWELL_SEC      접속 화면 체류 시간 (기본 3.0) - 관객이 볼 시간
    DEMO_RCS_GAP_SEC        창을 닫고 다음 장비까지 간격 (기본 3.0)
    DEMO_RCS_SCROLL_NOTCHES View 탭에서 아래/위로 굴릴 휠 눈금 수 (기본 3)
    DEMO_RCS_SCROLL_PAUSE_SEC  휠 한 눈금 사이 간격 (기본 0.6)
    DEMO_RCS_REPEAT         장비 순회 반복 횟수 (기본 1)
    DEMO_RCS_VIEW_TAB       View 탭 훑기 on/off (기본 1)
    DEMO_RCS_OPTICS         tool 창 안 Optics 시퀀스 on/off (기본 1)
    DEMO_RCS_OPTICS_SETTLE_SEC    대화상자가 그려질 대기 (기본 1.5)
    DEMO_RCS_OPTICS_ATTEMPTS      Optics 클릭 재시도 횟수 (기본 2)
    DEMO_RCS_CONFIRM        라벨 확인 정책 strict|lenient|off (기본 strict)
    DEMO_RCS_PRE_CLICK_SETTLE_SEC  커서 도착 후 클릭까지 대기 (기본 0.6)
                            원격 뷰가 커서를 따라올 시간 - 짧으면 클릭이 삼켜진다
    SAFE_MODE=1             모든 클릭 차단 (리허설 - 화면은 안 움직인다)

판정 로직은 전부 협력자 주입식이라 Mac 에서 시험된다:

    uv run pytest poc/workflow_3/monitor/test_demonstration_rcs_control.py
"""

import os
import time
from dataclasses import dataclass, field

from poc.workflow_3.config import Workflow3Settings, load_workflow3_settings
from poc.workflow_3.util.time_utils import make_timestamp_tag

LOG_COMPONENT = "demonstration_rcs_control"

# 시연 기본 장비. env 로 덮을 수 있지만, 아무것도 안 줘도 바로 돌아야 시연 직전에
# 셸 따옴표와 씨름하지 않는다.
DEFAULT_TOOL_IDS = ["MCD019", "MCDC22"]

# 장비 1대 방문 결과 status.
STATUS_CONNECTED = "connected"
STATUS_CONNECT_FAILED = "connect_failed"
STATUS_WINDOW_NOT_FOUND = "window_not_found"
STATUS_ERROR = "error"

# View 탭 훑기 status.
STATUS_VIEW_OK = "view_ok"
STATUS_VIEW_TAB_FAILED = "view_tab_failed"
STATUS_VIEW_SKIPPED = "view_skipped"

# tool 창 안 Optics 시퀀스 status.
STATUS_OPTICS_OK = "optics_ok"
STATUS_OPTICS_BUTTON_FAILED = "optics_button_failed"
STATUS_OPTICS_WINDOW_NOT_FOUND = "optics_window_not_found"
STATUS_OPTICS_MEMORY_FAILED = "optics_memory_tab_failed"
STATUS_OPTICS_CLOSE_FAILED = "optics_close_failed"
STATUS_OPTICS_SKIPPED = "optics_skipped"


@dataclass
class ToolVisit:
    """장비 1대의 [접속 -> Optics 조작 -> 닫기] 결과."""

    tool_id: str
    status: str = STATUS_CONNECTED
    closed: bool = False
    close_error: str = ""
    error: str = ""
    optics_status: str = ""
    elapsed_sec: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_CONNECTED


@dataclass
class DemoRunResult:
    """시연 1회 실행 요약."""

    preflight_status: str = ""
    view_status: str = ""
    list_tab_status: str = ""
    visits: list = field(default_factory=list)
    aborted: str = ""  # 비어 있으면 끝까지 수행. no_tools / interrupted / preflight status.

    @property
    def ok_count(self) -> int:
        return sum(1 for v in self.visits if v.ok)


# ------------------------------------------------------------------
# 장비 목록.
# ------------------------------------------------------------------


def parse_tool_ids(raw, default: list) -> list:
    """"MCD019, MCDC22" 같은 문자열을 장비 목록으로 만든다. 비면 default.

    중복은 **대소문자 무시**로 제거하되 첫 표기를 남긴다. 같은 장비를 두 번 열면
    두 번째 접속이 '이미 열려 있는 창' 을 만나 시연 흐름이 깨지기 때문이다.
    """
    tokens = []
    for chunk in (raw or "").replace(",", " ").split():
        token = chunk.strip()
        if token:
            tokens.append(token)
    if not tokens:
        return list(default)

    seen = set()
    unique = []
    for token in tokens:
        key = token.lower()
        if key in seen:
            print(f"[INFO] 장비 목록 중복 제거: {token}")
            continue
        seen.add(key)
        unique.append(token)
    return unique


# ------------------------------------------------------------------
# 장비 1대 방문.
# ------------------------------------------------------------------


def visit_tool(
    tool_id: str,
    *,
    connect_fn,
    wait_window_fn,
    close_fn,
    dwell_fn,
    dwell_sec: float,
    optics_fn=None,
) -> ToolVisit:
    """장비 1대를 접속 -> 체류 -> Optics 조작 -> 닫기 한다. **닫기는 어떤 경로로든 시도한다.**

    협력자:
      connect_fn(tool_id)     -> 결과 객체 또는 None (List 탭에서 더블클릭)
      wait_window_fn(tool_id) -> (window, title, backend)
      close_fn(tool_id)       -> exit_code 문자열 ("success" 면 닫힘)
      dwell_fn(sec)           -> None (관객이 화면을 볼 시간)
      optics_fn(window, title, backend) -> status (없으면 접속만 보여주고 나온다)

    닫기를 무조건 거는 이유: `connect_fn` 이 실패로 보고해도 더블클릭 자체는 먹었을
    수 있다(접속은 open-loop 다). 창을 남긴 채 다음 장비로 넘어가면 화면에 창이
    쌓여 시연이 그 자리에서 무너진다. 닫기 실패는 기록만 하고 삼킨다 - 한 대 때문에
    나머지 시연을 못 보는 편이 더 나쁘다.
    """
    started_at = time.time()
    visit = ToolVisit(tool_id=tool_id)

    try:
        print(f"[INFO] === {tool_id} 접속 시도 ===")
        result = connect_fn(tool_id)
        if result is None:
            visit.status = STATUS_CONNECT_FAILED
            print(f"[WARNING] {tool_id} 접속 실패(List 탭에서 행을 찾지 못함)")
        else:
            window, title, backend = wait_window_fn(tool_id)
            if window is None:
                visit.status = STATUS_WINDOW_NOT_FOUND
                print(f"[WARNING] {tool_id} tool 창이 뜨지 않음(점유 중이거나 오클릭)")
            else:
                visit.status = STATUS_CONNECTED
                print(f"[INFO] {tool_id} 접속 완료: title={title!r} - {dwell_sec:.0f}s 체류")
                dwell_fn(dwell_sec)
                if optics_fn is not None:
                    # Optics 조작이 깨져도 접속 자체는 성공한 것이고, 무엇보다 tool
                    # 창은 닫고 나가야 한다. 그래서 여기서 따로 삼킨다.
                    try:
                        visit.optics_status = optics_fn(window, title, backend)
                    except Exception as exc:
                        visit.optics_status = f"{type(exc).__name__}: {exc}"
                        print(f"[WARNING] {tool_id} Optics 조작 예외(창은 닫고 계속): "
                              f"{visit.optics_status}")
    except Exception as exc:
        visit.status = STATUS_ERROR
        visit.error = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {tool_id} 방문 예외: {visit.error}")

    _close_tool_window(visit, close_fn)
    visit.elapsed_sec = time.time() - started_at
    return visit


def _close_tool_window(visit: ToolVisit, close_fn) -> None:
    """tool 창을 닫고 결과를 visit 에 기록한다. 예외는 올리지 않는다."""
    try:
        exit_code = close_fn(visit.tool_id)
    except Exception as exc:
        visit.close_error = f"{type(exc).__name__}: {exc}"
        print(f"[WARNING] {visit.tool_id} 창 닫기 예외(시연은 계속): {visit.close_error}")
        return

    if exit_code == "success":
        visit.closed = True
        print(f"[INFO] {visit.tool_id} 창 닫기 완료")
    else:
        visit.close_error = str(exit_code)
        print(f"[WARNING] {visit.tool_id} 창 닫기 실패: exit_code={exit_code}")


# ------------------------------------------------------------------
# tool 창 안 Optics 시퀀스 - Optics... -> Memory 탭 -> Close.
#
# **대화상자는 로컬 창이 아니다.** Remote Monitoring 창은 장비 화면의 원격 뷰라,
# Optics 를 누르면 대화상자가 그 뷰 **안에** 그려진다 - 로컬 top-level 창 열거로는
# 절대 찾을 수 없다. 첫 오피스 실행에서 창 제목으로 찾다 실패하고도 "그래도 계속"
# 폴백이 걸려 화면 어딘가의 **다른 Close** 를 눌렀다. 그래서 확인은 창 열거가 아니라
# **라벨 판독(OCR)** 으로 하고, 확인되지 않으면 누르지 않는다(share_request 와 같은
# fail-closed actuator 규약).
# ------------------------------------------------------------------

# 확인 게이트 토큰. required 는 언어별 묶음이며 한 묶음을 전부 만족해야 확인이다.
# 오피스 실제 문구는 첫 실행의 debug_images crop/OCR 원문으로 확인한 뒤 조정한다.
OPTICS_BUTTON_KEY = "optics_button"
OPTICS_MEMORY_KEY = "optics_memory_tab"
OPTICS_CLOSE_KEY = "optics_close_button"

OPTICS_BUTTON_REQUIRED = (("optics",),)
OPTICS_BUTTON_FORBIDDEN = ("cancel", "stop", "terminat", "취소")
OPTICS_MEMORY_REQUIRED = (("memory",), ("메모리",))
OPTICS_MEMORY_FORBIDDEN = ("cancel", "취소")
OPTICS_CLOSE_REQUIRED = (("close",), ("닫기",))
OPTICS_CLOSE_FORBIDDEN = ("cancel", "terminat", "logout", "abort", "취소", "종료")


def _optics_targets():
    """Optics 시퀀스 3개 요소의 VLM 타겟 정의.

    설명문은 이 저장소의 규약대로 **첫 글자를 anchor** 로 잡게 쓴다(전역 프롬프트 원칙).
    'Optics...' 는 PM 버튼 바로 위라는 위치 단서를 함께 준다 - tool 창에는 버튼이 많아
    라벨만으로는 coarse 단계가 흔들린다.
    """
    from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

    optics_button = TargetConfig(
        key="optics_button",
        description=(
            "the 'Optics...' button in the Remote Monitoring window's button area, "
            "located directly above the 'PM' button. Use the first letter 'O' as the "
            "anchor, then click safely inside the Optics button area."
        ),
    )
    memory_tab = TargetConfig(
        key="optics_memory_tab",
        description=(
            "the 'Memory' tab in the tab strip of the Optics dialog window. "
            "Use the first letter 'M' as the anchor, then click safely inside the "
            "Memory tab area."
        ),
    )
    close_button = TargetConfig(
        key="optics_close_button",
        description=(
            "the 'Close' button of the Optics dialog window. Use the first letter 'C' "
            "as the anchor, then click safely inside the Close button area."
        ),
    )
    return optics_button, memory_tab, close_button


def _confirm_point(
    image, target, required, forbidden,
    *, locate_fn, read_tokens_fn, policy,
):
    """좌표를 찍고 그 자리 라벨을 읽어 **확인된 경우에만** 그 점을 돌려준다.

    좌표는 VLM 이 정하고 OCR 은 확인만 한다(이 저장소의 클릭 규약). 확인이 안 되면
    None - 원격 뷰에서 한 칸 어긋난 클릭은 무엇을 눌렀는지 알 수 없다.
    클릭을 여기서 하지 않는 이유는 호출부가 "라벨은 확인됐는데 클릭이 실패했다" 와
    "라벨부터 확인이 안 됐다" 를 구분해야 하기 때문이다 - 전자는 대화상자가 떠 있다는
    증거라 뒤 단계를 계속해야 하고, 후자는 멈춰야 한다.
    """
    from poc.workflow_3.monitor.share_request import accepts_label, classify_label

    point = locate_fn(image, target)
    if point is None:
        print(f"[WARNING] 좌표 미검출 - 클릭 안 함: {target.key}")
        return None

    tokens = read_tokens_fn(image, point, target.key)
    verdict = classify_label(tokens, required, forbidden)
    if not accepts_label(verdict, policy):
        print(
            f"[WARNING] 라벨 확인 실패 - 클릭 안 함: {target.key} "
            f"verdict={verdict} policy={policy} tokens={tokens!r}"
        )
        return None
    return point


def run_optics_sequence(
    tool_window,
    tool_title: str,
    tool_backend: str,
    *,
    capture_fn,
    locate_fn,
    read_tokens_fn,
    click_fn,
    sleep_fn,
    settle_sec: float = 1.5,
    confirm_policy: str = "strict",
    attempts: int = 2,
) -> str:
    """tool 창에서 Optics... -> Memory 탭 -> Close 를 확인하며 차례로 누른다.

    협력자(share_request 와 같은 모양이라 배선을 그대로 재사용한다):
      capture_fn(window)                  -> image
      locate_fn(image, target)            -> point dict | None (**이미지 픽셀 좌표**)
      read_tokens_fn(image, point, key)   -> list[str]
      click_fn(window, image, point, key) -> None

    세 가지가 계약이다.

    ① **확인되지 않으면 누르지 않는다.** 좌표만 믿고 누르면 원격 뷰의 엉뚱한 버튼을
       누르게 된다. 실제로 첫 오피스 실행이 그렇게 깨졌다.
    ② **Memory 라벨이 대화상자의 유일한 증거다.** 그것이 확인되기 전에는 Close 를 찾아
       나서지 않는다 - 대화상자가 없는 화면에서 'Close' 를 찾으면 tool 창 자체의 닫기
       같은 다른 것을 누른다. 확인 안 된 채 남은 대화상자는 다음 단계의 tool 창 닫기가
       정리하므로, 여기서 무리하게 닫는 것보다 안 누르는 편이 낫다.
    ③ **Memory 가 확인된 뒤에는 Close 를 반드시 시도한다.** 대화상자가 떠 있는 것이
       확인된 상태이므로, Memory 클릭이 실패해도 닫기까지는 가 본다.

    `attempts` 는 Optics 클릭 재시도 횟수다. 원격 뷰는 커서 이동을 따라오지 못해 첫
    클릭이 삼켜지는 일이 있어(오피스 1회차 증상: "마우스만 이동") 확인이 안 되면 한 번
    더 누른다.
    """
    optics_target, memory_target, close_target = _optics_targets()
    max_attempts = max(1, attempts)

    def _confirm(image, target, required, forbidden):
        return _confirm_point(
            image, target, required, forbidden,
            locate_fn=locate_fn, read_tokens_fn=read_tokens_fn, policy=confirm_policy,
        )

    # --- Optics... 버튼 -> 대화상자가 떴는지 확인(= Memory 라벨 판독) ---
    memory_point = None
    try:
        for attempt in range(1, max_attempts + 1):
            print(f"[INFO] Optics... 버튼 확인 후 클릭 (시도 {attempt}/{max_attempts})")
            image = capture_fn(tool_window)
            point = _confirm(
                image, optics_target, OPTICS_BUTTON_REQUIRED, OPTICS_BUTTON_FORBIDDEN,
            )
            if point is None:
                # 버튼 자체를 확인 못 했다면 다시 눌러도 같은 화면이다 - 즉시 포기.
                return STATUS_OPTICS_BUTTON_FAILED
            click_fn(tool_window, image, point, optics_target.key)

            sleep_fn(settle_sec)  # 대화상자가 그려질 시간(원격이라 로컬보다 느리다).

            print("[INFO] Optics 창 확인(Memory 탭 판독)")
            image = capture_fn(tool_window)
            memory_point = _confirm(
                image, memory_target, OPTICS_MEMORY_REQUIRED, OPTICS_MEMORY_FORBIDDEN,
            )
            if memory_point is not None:
                memory_image = image
                break
            if attempt >= max_attempts:
                print("[WARNING] Optics 창을 확인하지 못함 - Close 를 찾지 않습니다"
                      "(tool 창 닫기가 정리합니다).")
                return STATUS_OPTICS_WINDOW_NOT_FOUND
            print("[INFO] Optics 창 미확인 - 클릭이 삼켜졌을 수 있어 다시 누릅니다")
    except Exception as exc:
        print(f"[WARNING] Optics 버튼/창 확인 중 예외: {type(exc).__name__}: {exc}")
        return STATUS_OPTICS_BUTTON_FAILED

    # --- 여기부터 대화상자가 떠 있는 것이 확인된 상태다 ---
    status = STATUS_OPTICS_OK
    try:
        print("[INFO] Memory 탭 클릭")
        click_fn(tool_window, memory_image, memory_point, memory_target.key)
        sleep_fn(settle_sec)
    except Exception as exc:
        # 대화상자는 떠 있다 - Memory 를 못 눌렀어도 Close 까지는 간다(계약 ③).
        print(f"[WARNING] Memory 탭 클릭 예외(Close 는 계속): {type(exc).__name__}: {exc}")
        status = STATUS_OPTICS_MEMORY_FAILED

    try:
        print("[INFO] Optics 창 Close 확인 후 클릭")
        image = capture_fn(tool_window)
        close_point = _confirm(
            image, close_target, OPTICS_CLOSE_REQUIRED, OPTICS_CLOSE_FORBIDDEN,
        )
        if close_point is None:
            print("[WARNING] Optics Close 를 확인하지 못함(대화상자가 남았을 수 있음)")
            return STATUS_OPTICS_CLOSE_FAILED
        click_fn(tool_window, image, close_point, close_target.key)
    except Exception as exc:
        print(f"[WARNING] Optics Close 예외(대화상자가 남았을 수 있음): "
              f"{type(exc).__name__}: {exc}")
        return STATUS_OPTICS_CLOSE_FAILED

    sleep_fn(settle_sec)
    return status


# ------------------------------------------------------------------
# View 탭 + 휠 훑기.
# ------------------------------------------------------------------


def browse_view_tab(
    window,
    title: str,
    backend: str,
    *,
    click_tab_fn,
    scroll_fn,
    sleep_fn,
    notches: int,
    pause_sec: float = 0.6,
) -> str:
    """View 탭으로 옮긴 뒤 휠로 아래/위를 훑는다.

    협력자:
      click_tab_fn(window, title, backend) -> exit_code ("success" 면 전환됨)
      scroll_fn(dy, step_index)            -> bool (dy<0 아래로, dy>0 위로)

    **탭 클릭이 실패하면 휠을 굴리지 않는다.** 지금 화면이 무엇인지 모르는 상태라
    (보통 직전의 List 탭) 엉뚱한 목록을 스크롤해 다음 장면의 전제를 망친다.
    내려간 만큼 되올라와 원래 위치로 복귀시킨다 - 반복 시연에서 재현성이 유지된다.
    """
    try:
        exit_code = click_tab_fn(window, title, backend)
    except Exception as exc:
        print(f"[WARNING] View 탭 클릭 예외(휠 훑기 생략): {type(exc).__name__}: {exc}")
        return STATUS_VIEW_TAB_FAILED

    if exit_code != "success":
        print(f"[WARNING] View 탭 클릭 실패: exit_code={exit_code} (휠 훑기 생략)")
        return STATUS_VIEW_TAB_FAILED

    print(f"[INFO] View 탭 전환 완료 - 휠로 아래 {notches}칸 / 위 {notches}칸 훑기")
    for step_index in range(notches):
        scroll_fn(-1, step_index)
        sleep_fn(pause_sec)
    for step_index in range(notches):
        scroll_fn(1, notches + step_index)
        sleep_fn(pause_sec)
    return STATUS_VIEW_OK


# ------------------------------------------------------------------
# 시나리오 전체.
# ------------------------------------------------------------------


def run_demonstration(
    tool_ids: list,
    *,
    preflight_fn,
    view_fn,
    list_tab_fn,
    visit_fn,
    sleep_fn,
    gap_sec: float,
    repeat: int = 1,
) -> DemoRunResult:
    """시나리오를 순서대로 재생한다. 개별 실패는 삼키고 끝까지 간다.

    협력자:
      preflight_fn()                 -> PreflightOutcome (실행+로그인+List 탭)
      view_fn(window, title, backend) -> view status
      list_tab_fn(window, title, backend) -> exit_code
      visit_fn(tool_id)              -> ToolVisit

    중단하는 경우는 둘뿐이다: 장비 목록이 비었거나(RCS 를 띄울 이유가 없다),
    preflight 가 메인 창을 못 잡았거나(접속이 전부 실패할 것이 확실하다).
    """
    result = DemoRunResult()

    if not tool_ids:
        # RCS 를 띄우기 **전에** 판정한다 - 목적 없이 로그인만 하고 끝나면 안 된다.
        print("[ERROR] 접속할 장비가 없습니다. DEMO_RCS_TOOL_IDS 를 확인하세요.")
        result.aborted = "no_tools"
        return result

    print(f"[INFO] 시연 장비 {len(tool_ids)}대: {', '.join(tool_ids)}")
    print("[INFO] --- 1단계: RCS 실행 -> 로그인 -> List 탭 ---")
    preflight = preflight_fn()
    result.preflight_status = getattr(preflight, "status", "")
    window = getattr(preflight, "window", None)
    if window is None:
        # List 탭 클릭만 실패한 경우는 창이 살아 있으므로 여기 안 걸린다 - 그때는
        # 이미 List 였을 수도 있고, 아래 list_tab_fn 이 다시 시도한다.
        print(
            f"[ERROR] RCS 메인 창을 확보하지 못했습니다(status={result.preflight_status}). "
            "시연을 중단합니다 - RCS 를 직접 로그인한 뒤 다시 실행하세요."
        )
        result.aborted = result.preflight_status or "rcs_unavailable"
        return result

    title = getattr(preflight, "title", "")
    backend = getattr(preflight, "backend", "")

    print("[INFO] --- 2단계: View 탭 + 휠 훑기 ---")
    try:
        result.view_status = view_fn(window, title, backend)
    except Exception as exc:
        result.view_status = STATUS_VIEW_TAB_FAILED
        print(f"[WARNING] View 훑기 예외(시연은 계속): {type(exc).__name__}: {exc}")

    print("[INFO] --- 3단계: List 탭 복귀 ---")
    try:
        result.list_tab_status = list_tab_fn(window, title, backend)
    except Exception as exc:
        result.list_tab_status = f"{type(exc).__name__}: {exc}"
        print(f"[WARNING] List 탭 복귀 예외(접속 단계에서 재판정): {result.list_tab_status}")
    if result.list_tab_status != "success":
        # 막지는 않는다 - preflight 가 이미 List 를 열어 뒀을 수 있고, 실제 판정은
        # connect 가 한다. 여기서 시연 전체를 버리는 편이 더 비싸다.
        print(f"[WARNING] List 탭 복귀 실패: {result.list_tab_status} (접속을 그대로 시도)")

    print("[INFO] --- 4단계: 장비 순회 (접속 -> 닫기) ---")
    try:
        for round_index in range(max(1, repeat)):
            if repeat > 1:
                print(f"[INFO] === 순회 {round_index + 1}/{repeat} ===")
            for tool_id in tool_ids:
                try:
                    result.visits.append(visit_fn(tool_id))
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    # 한 대가 깨져도 나머지를 보여줘야 시연이 성립한다.
                    detail = f"{type(exc).__name__}: {exc}"
                    result.visits.append(
                        ToolVisit(tool_id=tool_id, status=STATUS_ERROR, error=detail)
                    )
                    print(f"[ERROR] {tool_id} 처리 예외(다음 장비 계속): {detail}")
                sleep_fn(gap_sec)
    except KeyboardInterrupt:
        print("\n[INFO] 시연 중단 (Ctrl+C)")
        result.aborted = "interrupted"

    _print_summary(result)
    return result


def _print_summary(result: DemoRunResult) -> None:
    """콘솔에 한눈에 보이는 결과 표를 찍는다."""
    print("=" * 70)
    print(
        f"[INFO] 시연 요약: 접속 성공 {result.ok_count}/{len(result.visits)}대, "
        f"View={result.view_status or '-'}, preflight={result.preflight_status or '-'}"
        + (f", 중단={result.aborted}" if result.aborted else "")
    )
    for visit in result.visits:
        closed = "닫힘" if visit.closed else f"닫기실패({visit.close_error or '-'})"
        optics = f" / Optics={visit.optics_status}" if visit.optics_status else ""
        detail = f" {visit.error}" if visit.error else ""
        print(
            f"[INFO]   {visit.tool_id}: {visit.status} / {closed}{optics} / "
            f"{visit.elapsed_sec:.1f}s{detail}"
        )
    print("=" * 70)


# ------------------------------------------------------------------
# Windows 실배선 - 여기부터는 오피스에서만 돈다.
# ------------------------------------------------------------------


def _build_preflight_fn(settings: Workflow3Settings):
    """RCS 실행 + 로그인 + List 탭까지 올리는 협력자를 조립한다.

    `align_fail_monitor._run_rcs_preflight` 와 같은 배선이다 - 판정은 공용
    `rcs_preflight.ensure_rcs_session_ready` 가 하고 여기서는 협력자만 묶는다.
    RCS 가 이미 로그인돼 있으면 재실행/재로그인은 생략된다(중복 프로세스 방지).
    """
    from poc.workflow_3.monitor.cycle import _scan_rcs_processes
    from poc.workflow_3.monitor.rcs_preflight import ensure_rcs_session_ready
    from poc.workflow_3.monitor.rcs_recovery import recover_rcs_session
    from poc.workflow_3.rcs.login_rcs_common import wait_for_rcs_main_window
    from poc.workflow_3.rcs.open_rcs import launch_rcs
    from poc.workflow_3.rcs.view_list_tab_rcs import click_list_tab_in_main_window
    from poc.workflow_3.rcs.workflow_login import run_login_workflow

    def _recover():
        return recover_rcs_session(
            settings,
            find_processes_fn=_scan_rcs_processes,
            launch_fn=launch_rcs,
            login_fn=run_login_workflow,
            wait_window_fn=wait_for_rcs_main_window,
        )

    def _open_list(window, title, backend):
        return click_list_tab_in_main_window(window, title, backend).exit_code

    def _preflight():
        return ensure_rcs_session_ready(
            settings,
            find_window_fn=wait_for_rcs_main_window,
            recover_fn=_recover,
            open_list_fn=_open_list,
        )

    return _preflight


def _window_center_screen_point(window):
    """창 중심의 스크린 절대 좌표. 실패 시 None.

    rect 크기를 그대로 image point 로 넘긴다 - image_size 를 생략하면 scale=1.0 이라
    변환이 rect 좌표계 안에서 이뤄져 DPI 배율(오피스 125/150%)과 무관하게 중심이 맞는다.
    """
    from poc.workflow_3.util.window_utils import image_point_to_screen, window_rect_size

    size = window_rect_size(window)
    if size is None:
        print("[WARNING] 창 rect 조회 실패 - 휠 좌표를 만들 수 없습니다.")
        return None
    width, height = size
    return image_point_to_screen(window, {"x": width // 2, "y": height // 2})


def _build_view_fn(settings: Workflow3Settings, notches: int, pause_sec: float):
    """View 탭 클릭 + 창 중심 휠 훑기 협력자."""
    from poc.workflow_3.rcs.view_list_tab_rcs import VIEW_TAB_TARGET, click_main_tab
    from poc.workflow_3.util.mouse_utils import scroll_at_screen

    def _click_view(window, title, backend):
        return click_main_tab(
            window, title, backend, VIEW_TAB_TARGET,
            action_enabled=settings.action_enabled,
        ).exit_code

    def _view(window, title, backend):
        point = _window_center_screen_point(window)
        if point is None:
            return STATUS_VIEW_TAB_FAILED

        def _scroll(dy, step_index):
            return scroll_at_screen(
                point, dy, "demo_view_tab", step_index,
                action_enabled=settings.action_enabled,
            )

        return browse_view_tab(
            window, title, backend,
            click_tab_fn=_click_view,
            scroll_fn=_scroll,
            sleep_fn=time.sleep,
            notches=notches,
            pause_sec=pause_sec,
        )

    return _view


def _build_list_tab_fn(settings: Workflow3Settings):
    """List 탭 복귀 협력자 - connect 는 '현재 List 탭' 을 전제한다."""
    from poc.workflow_3.rcs.view_list_tab_rcs import click_list_tab_in_main_window

    def _list_tab(window, title, backend):
        return click_list_tab_in_main_window(
            window, title, backend, action_enabled=settings.action_enabled,
        ).exit_code

    return _list_tab


def _build_optics_fn(
    settings: Workflow3Settings,
    settle_sec: float,
    *,
    confirm_policy: str,
    attempts: int,
    pre_click_settle_sec: float,
    tag: str,
):
    """tool 창 안 Optics -> Memory -> Close 협력자 (VLM 좌표 + OCR 확인 + 클릭).

    `share_request` 의 주입점과 같은 모양이라 그 배선을 그대로 옮겨 쓴다. 확인 실패 시
    crop 과 OCR 원문이 `debug_images/demo_rcs_optics/<tag>/` 에 남는다 - Mac 에서는 이
    화면을 볼 수 없어, 오피스 실행이 실제 문구(required 토큰)를 아는 유일한 경로다.
    """
    from poc.workflow_3 import DEBUG_IMAGE_DIR
    from poc.workflow_3.util.mouse_utils import click_at_screen, move_cursor_to_screen
    from poc.workflow_3.util.window_utils import capture_window, image_point_to_screen
    from poc.workflow_3.vlm.label_verify import (
        crop_box_around_point,
        read_text_near_point,
        tokens_from_text,
    )
    from poc.workflow_3.vlm.ui_venus_mai_locator import analyze_window_target

    debug_dir = DEBUG_IMAGE_DIR / "demo_rcs_optics" / tag

    def _locate(image, target):
        result = analyze_window_target(
            None, "Remote Monitoring System", "uia", target,
            debug_image_dir=debug_dir,
            log_name=LOG_COMPONENT,
            component_name=LOG_COMPONENT,
            artifact_prefix=target.key,
            image=image,
        )
        return result.point

    def _read_tokens(image, point, key):
        box = crop_box_around_point(
            point, image.width, image.height,
            left_ratio=0.30, right_ratio=0.30, half_height_ratio=0.05,
        )
        read = read_text_near_point(
            image, box,
            debug_image_dir=debug_dir,
            timestamp_tag=make_timestamp_tag(time.time()),
            artifact_label=key,
            log_name=LOG_COMPONENT,
        )
        return tokens_from_text(read.raw_text) if read.ok else []

    def _click(window, image, point, key):
        """이미지 픽셀 좌표를 스크린 좌표로 변환해 클릭한다.

        **도착과 클릭 사이에 체류를 둔다.** tool 창은 장비 화면의 원격 뷰라 커서 이동이
        원격에 반영되기까지 지연이 있고, 곧바로 누르면 원격 쪽 커서가 아직 이전 위치에
        있어 클릭이 삼켜진다(오피스 1회차 증상: "마우스만 이동하고 클릭이 안 됨").
        먼저 커서만 옮겨 원격이 따라오게 한 뒤 잠깐 쉬고 그 자리에서 누른다.
        """
        screen = image_point_to_screen(window, point, image_size=image.size)
        if screen is None:
            raise RuntimeError(f"Optics 좌표 변환 실패: {key} point={point}")
        print(
            f"[INFO] Optics 클릭: {key} px={point} -> screen={screen}"
            f"{'' if settings.action_enabled else ' [dry-run]'}"
        )
        move_cursor_to_screen(screen, f"demo_{key}", action_enabled=settings.action_enabled)
        time.sleep(max(0.0, pre_click_settle_sec))
        click_at_screen(screen, f"demo_{key}", action_enabled=settings.action_enabled)

    def _optics(tool_window, tool_title, tool_backend):
        return run_optics_sequence(
            tool_window, tool_title, tool_backend,
            capture_fn=capture_window,
            locate_fn=_locate,
            read_tokens_fn=_read_tokens,
            click_fn=_click,
            sleep_fn=time.sleep,
            settle_sec=settle_sec,
            confirm_policy=confirm_policy,
            attempts=attempts,
        )

    return _optics


def _build_visit_fn(settings: Workflow3Settings, dwell_sec: float, optics_fn=None):
    """장비 1대 [접속 -> 체류 -> Optics -> 닫기] 협력자."""
    from poc.workflow_3.rcs.login_rcs_common import wait_for_remote_monitoring_window
    from poc.workflow_3.rcs.workflow_close_tool import close_tool
    from poc.workflow_3.rcs.workflow_select_tool import connect_to_tool

    action_enabled = settings.action_enabled and settings.connect_action_enabled

    def _connect(tool_id):
        return connect_to_tool(
            tool_id,
            action_enabled=action_enabled,
            main_window_timeout_sec=settings.connect_window_timeout_sec,
        )

    def _wait_window(tool_id):
        return wait_for_remote_monitoring_window(
            tool_id, max_attempts=settings.rcs_window_max_trials,
        )

    def _close(tool_id):
        return close_tool(tool_id, action_enabled=settings.action_enabled).exit_code

    def _visit(tool_id):
        return visit_tool(
            tool_id,
            connect_fn=_connect,
            wait_window_fn=_wait_window,
            close_fn=_close,
            dwell_fn=time.sleep,
            dwell_sec=dwell_sec,
            optics_fn=optics_fn,
        )

    return _visit


def _env_float(name: str, default: float) -> float:
    """숫자 env 를 읽는다. 잘못된 값이면 시연 직전에 죽지 않고 기본값으로 간다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARNING] {name}={raw!r} 를 숫자로 못 읽음 - 기본값 {default} 사용")
        return default


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _apply_demo_mode_defaults() -> None:
    """시연은 실제로 화면이 움직여야 의미가 있으므로 SAFE_MODE=0 을 기본으로 둔다.

    setdefault 라 **실제 셸 env 가 항상 이긴다** - 리허설만 하려면
    `SAFE_MODE=1 uv run python ...` 로 실행하면 클릭이 전부 막힌다(화면은 안 움직이고
    콘솔에 [DRY-RUN] 만 찍힌다). 이 스크립트가 내는 클릭은 탭/더블클릭/창 닫기뿐이며
    보정(reposition/OK)은 아예 없다.
    """
    os.environ.setdefault("SAFE_MODE", "0")
    live = os.environ.get("SAFE_MODE", "0") == "0"
    print("=" * 70)
    if live:
        print("[WARNING] 시연 모드: 실제 마우스 조작이 발생합니다 "
              "(탭 클릭 / 휠 / tool 더블클릭 / 창 닫기).")
        print("[WARNING] 리허설만 하려면 중단 후 'SAFE_MODE=1' 을 붙여 다시 실행하세요.")
    else:
        print("[INFO] SAFE_MODE=1 - 모든 클릭이 차단된 리허설입니다(화면은 움직이지 않음).")
    print("=" * 70)


def main(settings: Workflow3Settings | None = None) -> DemoRunResult:
    """시연 시나리오를 1회 재생한다."""
    settings = settings or load_workflow3_settings()

    tool_ids = parse_tool_ids(os.environ.get("DEMO_RCS_TOOL_IDS"), DEFAULT_TOOL_IDS)
    dwell_sec = _env_float("DEMO_RCS_DWELL_SEC", 3.0)
    gap_sec = _env_float("DEMO_RCS_GAP_SEC", 3.0)
    notches = _env_int("DEMO_RCS_SCROLL_NOTCHES", 3)
    pause_sec = _env_float("DEMO_RCS_SCROLL_PAUSE_SEC", 0.6)
    repeat = max(1, _env_int("DEMO_RCS_REPEAT", 1))
    view_enabled = _env_flag("DEMO_RCS_VIEW_TAB", True)
    optics_enabled = _env_flag("DEMO_RCS_OPTICS", True)
    optics_settle_sec = _env_float("DEMO_RCS_OPTICS_SETTLE_SEC", 1.5)
    optics_attempts = max(1, _env_int("DEMO_RCS_OPTICS_ATTEMPTS", 2))
    confirm_policy = os.environ.get("DEMO_RCS_CONFIRM", "strict").strip().lower() or "strict"
    pre_click_settle = _env_float("DEMO_RCS_PRE_CLICK_SETTLE_SEC", 0.6)
    tag = make_timestamp_tag(time.time())

    print(
        f"[INFO] 시연 설정: 체류={dwell_sec:.0f}s, 간격={gap_sec:.0f}s, "
        f"View훑기={'on' if view_enabled else 'off'}(휠 {notches}칸), "
        f"Optics조작={'on' if optics_enabled else 'off'}"
        f"(확인={confirm_policy}, 재시도={optics_attempts}, 클릭전대기={pre_click_settle:.1f}s), "
        f"반복={repeat}회"
    )

    try:
        preflight_fn = _build_preflight_fn(settings)
        view_fn = (
            _build_view_fn(settings, notches, pause_sec)
            if view_enabled
            else (lambda w, t, b: STATUS_VIEW_SKIPPED)
        )
        list_tab_fn = _build_list_tab_fn(settings)
        optics_fn = (
            _build_optics_fn(
                settings, optics_settle_sec,
                confirm_policy=confirm_policy,
                attempts=optics_attempts,
                pre_click_settle_sec=pre_click_settle,
                tag=tag,
            )
            if optics_enabled
            else None
        )
        visit_fn = _build_visit_fn(settings, dwell_sec, optics_fn)
    except Exception as exc:
        # Mac/개발 PC 에서는 pywinauto 등이 없어 여기서 걸린다 - 무엇이 없는지 이름을 남긴다.
        print(f"[ERROR] RCS 모듈을 불러오지 못했습니다(오피스 Windows 전용): {exc}")
        return DemoRunResult(aborted="rcs_modules_unavailable")

    return run_demonstration(
        tool_ids,
        preflight_fn=preflight_fn,
        view_fn=view_fn,
        list_tab_fn=list_tab_fn,
        visit_fn=visit_fn,
        sleep_fn=time.sleep,
        gap_sec=gap_sec,
        repeat=repeat,
    )


__all__ = [
    "DEFAULT_TOOL_IDS",
    "STATUS_CONNECTED",
    "STATUS_CONNECT_FAILED",
    "STATUS_ERROR",
    "STATUS_OPTICS_BUTTON_FAILED",
    "STATUS_OPTICS_CLOSE_FAILED",
    "STATUS_OPTICS_MEMORY_FAILED",
    "STATUS_OPTICS_OK",
    "STATUS_OPTICS_SKIPPED",
    "STATUS_OPTICS_WINDOW_NOT_FOUND",
    "STATUS_VIEW_OK",
    "STATUS_VIEW_SKIPPED",
    "STATUS_VIEW_TAB_FAILED",
    "STATUS_WINDOW_NOT_FOUND",
    "DemoRunResult",
    "ToolVisit",
    "browse_view_tab",
    "main",
    "parse_tool_ids",
    "run_demonstration",
    "run_optics_sequence",
    "visit_tool",
]


if __name__ == "__main__":
    # 실편집 workflow_3_config.py 의 토글을 env 로 브리지(있으면).
    # 시연 기본값(SAFE_MODE=0)을 **먼저** 못박아, 오피스 사본에 남은 SAFE_MODE=1 이
    # 조용히 덮지 않게 한다(양쪽 다 setdefault 라 먼저 잡은 쪽이 이긴다).
    from poc.workflow_3.workflow_3_config_loader import seed_env

    _apply_demo_mode_defaults()
    seed_env()
    main()
