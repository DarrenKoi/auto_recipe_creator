"""RCS 자동 조작 시연 스크립트 - 알람 없이 정해진 시나리오를 순서대로 보여준다.

알람을 기다리지 않고, 사람이 보는 앞에서 "이 시스템이 RCS 를 이렇게 몹니다" 를
재생하는 것이 목적이다. 시나리오(사용자 지정, 2026-08-19 / MCD019 흐름 2026-08-24 교체):

    RCS 실행 -> 로그인 -> View 탭 + 휠로 위아래 훑기 -> List 탭
      -> MCD019 접속 -> [Utility -> Memo Print -> 세 줄 메모 입력 -> Close] -> tool 창 닫기
                        (Utility 가 다른 창에 가려 안 보이면 Alt+click 으로 밀어낸다)
      -> MCDC10 접속 -> [Work Sheet 아래 버튼 -> File -> Exit]    -> tool 창 닫기

장비마다 **다른 조작**을 보여주는 것이 요점이다 - 같은 동작을 반복하면 스크립트로
보이고, 창을 열어 메뉴를 타고 들어갔다 빠져나오면 자동화로 보인다.

`align_fail_monitor_only_check.py` 로도 replay CSV 를 물려 비슷한 것을 할 수 있지만,
그쪽은 알람 사이클(manifest/cube 알림/보정 가능성 판정)을 통째로 끌고 오고 장비를
**하나씩만** 처리한다. 시연에 필요한 것은 여러 장비 순회와 화면 체류 시간이라
별도 진입점으로 둔다.

**이 스크립트는 보정을 하지 않는다.** 하는 일은 탭 클릭 / 휠 / tool 더블클릭 /
Utility 메뉴와 MemoPrint 창 열기 / 메모 입력 / 창 닫기뿐이다. reposition·OK 클릭처럼
레시피나 측정에 영향을 주는 조작은 전혀 없다.

사용법 (오피스 Windows):

    uv run python poc/workflow_3/monitor/demonstration_rcs_control.py

장비 목록만 바꿔서:

    DEMO_RCS_TOOL_IDS="MCD019,MCDC10,MCD916" \
      uv run python poc/workflow_3/monitor/demonstration_rcs_control.py

env (`DEMO_RCS_*` 네임스페이스 - 루프의 `ALIGN_FAIL_*` 과 섞지 않는다).
기본값은 파일 상단 타이밍 상수 블록에 있고, 그중 **원격 입력 성사 조건 4개는
줄이지 않는다**(PRE_CLICK_SETTLE / CLICK_HOLD / ALT_SETTLE / SHIFT_SETTLE):

    DEMO_RCS_TOOL_IDS       접속할 장비 (콤마/공백 구분, 기본 "MCD019,MCDC10")
    DEMO_RCS_DWELL_SEC      접속 화면 체류 시간 (기본 2.1) - 관객이 볼 시간
    DEMO_RCS_GAP_SEC        창을 닫고 다음 장비까지 간격 (기본 2.1)
    DEMO_RCS_SCROLL_NOTCHES View 탭에서 아래/위로 굴릴 휠 눈금 수 (기본 3)
    DEMO_RCS_SCROLL_PAUSE_SEC  휠 한 눈금 사이 간격 (기본 0.42)
    DEMO_RCS_REPEAT         장비 순회 반복 횟수 (기본 1)
    DEMO_RCS_VIEW_TAB       View 탭 훑기 on/off (기본 1)
    DEMO_RCS_FLOW           tool 창 안 조작 on/off (기본 1)
    DEMO_RCS_FLOWS          장비별 흐름 배정 (기본 "MCD019=memo_print,MCDC10=worksheet")
                            고를 수 있는 흐름: memo_print / optics / worksheet
    DEMO_RCS_DEFAULT_FLOW   목록에 없는 장비의 흐름 (기본 memo_print)
    DEMO_RCS_FLOW_SETTLE_SEC      창/드롭다운이 그려질 대기 (기본 1.05)
    DEMO_RCS_FLOW_ATTEMPTS        여는 버튼 재시도 횟수 (기본 2)
    DEMO_RCS_CONFIRM        라벨 확인 정책 strict|lenient|off (기본 lenient)
                            lenient 도 금지 토큰(cancel/exit 등)은 그대로 막는다
    DEMO_RCS_PRE_CLICK_SETTLE_SEC  커서 도착 후 클릭까지 대기 (기본 0.6)
                            원격 뷰가 커서를 따라올 시간 - 짧으면 클릭이 삼켜진다
    DEMO_RCS_CLICK_HOLD_SEC 버튼을 누르고 있는 시간 (기본 0.15)
                            즉시 press/release 는 원격 샘플링 사이로 빠져나간다
    DEMO_RCS_CHAR_TYPE_DELAY_SEC  메모 글자 사이 입력 간격 (기본 0.056)
                            원격 화면이 입력을 샘플링하므로 한 번에 보내지 않는다
    DEMO_RCS_SHIFT_MODE     대문자 입력 방식 caps_all|caps|shift|type (기본 caps_all)
                            caps_all = 문구 전체를 Caps 토글 한 쌍으로(전부 대문자)
                            caps = 대문자마다 토글(4회차: memo 가 깨졌다)
                            shift = Shift 쥐기(2회차: 도착하지만 소문자로 들어옴)
                            type = pynput 에 그대로 맡김(1회차: 글자가 사라짐)
    DEMO_RCS_CAPS_SETTLE_SEC  Caps 토글 전후 대기 (기본 0.4)
    DEMO_RCS_SHIFT_SETTLE_SEC  수정자를 잡고/놓기 전 대기 (기본 0.12)
                            대문자가 여전히 어긋나면 이 값을 올린다
    DEMO_RCS_POST_TYPE_WAIT_SEC  입력을 끝내고 Close 를 누르기 전 대기 (기본 1.4)
    DEMO_RCS_MEMO_TEXT      메모 문구 교체 ('\\n' 이 줄바꿈)
    ACTION_LOGIN_TYPING_ENABLED=0  클릭은 두고 **메모 입력만** 끈다(롤백 스위치)
    DEMO_RCS_REVEAL         여는 버튼이 가려졌을 때 Alt+click 으로 밀어내기 (기본 1)
    DEMO_RCS_REVEAL_ATTEMPTS  밀어낼 창 수 = Alt+click 반복 상한 (기본 2)
    DEMO_RCS_ALT_SETTLE_SEC   Alt 를 누른 뒤 클릭까지 대기 (기본 0.3)
                            커서를 오른쪽 아래로 옮긴 **뒤에** Alt 를 잡고, 원격이 그
                            수정자를 등록할 틱을 준 다음 누른다
    DEMO_RCS_REVEAL_X_RATIO / _Y_RATIO  누를 지점(창 크기 대비, 기본 0.88 / 0.92)
                            Utility 가 오른쪽 아래에 있어 그 자리를 누른다 - 빗나가면
                            오피스 콘솔의 px/screen 값을 보고 이 비율만 옮긴다
    ALIGN_FAIL_RCS_KILL_STALE=1  창 없는 좀비 RCS 프로세스를 종료하고 재실행 (기본 off)
    SAFE_MODE=1             모든 클릭 차단 (리허설 - 화면은 안 움직인다)

판정 로직은 전부 협력자 주입식이라 Mac 에서 시험된다:

    uv run pytest poc/workflow_3/monitor/test_demonstration_rcs_control.py
"""

import os
import time
from dataclasses import dataclass, field

from poc.workflow_3.config import Workflow3Settings, load_workflow3_settings
from poc.workflow_3.util.abort_switch import abort_reason, is_aborted
from poc.workflow_3.util.time_utils import make_timestamp_tag

LOG_COMPONENT = "demonstration_rcs_control"

# 시연 기본 장비. env 로 덮을 수 있지만, 아무것도 안 줘도 바로 돌아야 시연 직전에
# 셸 따옴표와 씨름하지 않는다.
DEFAULT_TOOL_IDS = ["MCD019", "MCDC10"]

# ------------------------------------------------------------------
# 시연 속도(초). 두 종류를 **반드시 구분한다**.
#
# ① 동작 사이의 간격 - 관객이 보는 속도다. 2026-08-24 사용자 요청으로 종전 대비 30%
#    줄였다(체류 3.0->2.1, 간격 3.0->2.1, step 1.5->1.05, 글자 0.08->0.056,
#    입력후 2.0->1.4, 휠 0.6->0.42). 더 줄이려면 여기를 고친다.
# ② 원격 입력이 **성사되는 조건** - 오피스 실측 3회로 얻은 값이라 줄이지 않는다.
#    이 값을 깎으면 커서는 가는데 클릭이 안 먹거나(1·2회차) 대문자가 사라진다(3회차).
#    시연이 30% 빠른 것과 시연이 안 되는 것은 비교 대상이 아니다.
#
# env(`DEMO_RCS_*`)가 항상 이긴다 - 오피스에서 값을 찾을 때는 env, 찾은 뒤에는 여기.
# ------------------------------------------------------------------

# ① 간격 - 줄여도 되는 것.
DWELL_SEC = 2.1              # 접속 화면 체류
GAP_SEC = 2.1                # 창을 닫고 다음 장비까지
SCROLL_PAUSE_SEC = 0.42      # View 탭 휠 한 눈금 사이
FLOW_SETTLE_SEC = 1.05       # 창/드롭다운이 그려질 대기
CHAR_TYPE_DELAY_SEC = 0.056  # 메모 글자 사이
POST_TYPE_WAIT_SEC = 1.4     # 입력을 끝내고 Close 를 누르기 전

# ② 입력 성사 조건 - 줄이지 않는 것.
PRE_CLICK_SETTLE_SEC = 0.6   # 커서 도착 -> 클릭. 원격이 커서를 따라올 시간.
CLICK_HOLD_SEC = 0.15        # 누름 유지. 즉시 press/release 는 샘플링 사이로 빠진다.
ALT_SETTLE_SEC = 0.3         # Alt 를 잡고 -> 클릭. 수정자가 등록될 틱.
SHIFT_SETTLE_SEC = 0.12      # Shift 를 잡고/놓기 전. 같은 이유.
CAPS_SETTLE_SEC = 0.4        # Caps 토글 전후. 토글은 링크를 건너 장비가 적용해야 한다.

# MCD019 MemoPrint 에 입력할 시연 문구. 줄바꿈마다 Enter 를 누른다.
# `!!` 와 큰따옴표는 뺐다(사용자 결정 2026-08-24). Shift 기호는 이 원격을 못 건너
# `!`->`1`, `"`->`'` 로 찍혔다 - 화면에 틀리게 나올 글자를 문구에 남길 이유가 없다.
# 이제 이 문구에 필요한 수정자는 전체를 감싸는 Caps 토글 한 쌍뿐이다.
DEFAULT_MEMO_TEXT = (
    "Infra. Tech Center\n"
    "One Stop Solution\n"
    "This is the PoC of Auto Recipe Creation."
)


def parse_memo_text(raw, default: str) -> str:
    """env 문구를 읽는다. env 에는 실제 줄바꿈을 담기 어려우니 `\\n` 을 줄바꿈으로 본다.

    문구를 env 로 뺀 이유: `!` 는 Shift+1(US 기호 배열) 로 입력하는데 배열이 다르면
    `1` 이 들어간다. 그때 코드를 고치지 않고 문구만 바꿔 시연을 살릴 수 있어야 한다.
    """
    text = (raw or "").strip()
    if not text:
        return default
    return text.replace("\\n", "\n")

# 라벨 확인 정책 기본값. 기존 시연 버튼은 **돌고 있는 장비에 영향을 주지 않는다** 고
# 오피스에서 확인됐다(2026-08-19). 새 Utility/Memo Print 흐름은 2026-08-24 추가됐고
# 아직 오피스 실측 전이므로 첫 실행은 DEMO_RCS_CONFIRM=strict 로 캘리브레이션한다.
# 기존 기본값 lenient 는 "못 읽음" 을 통과시키되 **금지 토큰(cancel/exit/terminate
# 등)은 어떤 정책에서도 막는다**
# (share_request.accepts_label). 좌표 자체를 못 찾으면 정책과 무관하게 안 누른다.
# 라벨 문구를 확정하는 진단 실행에서는 DEMO_RCS_CONFIRM=strict 로 되돌린다.
DEFAULT_CONFIRM_POLICY = "lenient"

# 장비 1대 방문 결과 status.
STATUS_CONNECTED = "connected"
STATUS_CONNECT_FAILED = "connect_failed"
STATUS_WINDOW_NOT_FOUND = "window_not_found"
STATUS_ERROR = "error"

# View 탭 훑기 status.
STATUS_VIEW_OK = "view_ok"
STATUS_VIEW_TAB_FAILED = "view_tab_failed"
STATUS_VIEW_SKIPPED = "view_skipped"



@dataclass
class ToolVisit:
    """장비 1대의 [접속 -> 장비별 창 안 조작 -> 닫기] 결과."""

    tool_id: str
    status: str = STATUS_CONNECTED
    closed: bool = False
    close_error: str = ""
    error: str = ""
    action_status: str = ""
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
    """"MCD019, MCDC10" 같은 문자열을 장비 목록으로 만든다. 비면 default.

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
    action_fn=None,
) -> ToolVisit:
    """장비 1대를 접속 -> 체류 -> 창 안 조작 -> 닫기 한다. **닫기는 어떤 경로로든 시도한다.**

    협력자:
      connect_fn(tool_id)     -> 결과 객체 또는 None (List 탭에서 더블클릭)
      wait_window_fn(tool_id) -> (window, title, backend)
      close_fn(tool_id)       -> exit_code 문자열 ("success" 면 닫힘)
      dwell_fn(sec)           -> None (관객이 화면을 볼 시간)
      action_fn(tool_id, window, title, backend) -> status
        창 안에서 보여줄 조작(장비마다 다르다). 없으면 접속만 보여주고 나온다.

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
                if action_fn is not None:
                    # 창 안 조작이 깨져도 접속 자체는 성공한 것이고, 무엇보다 tool
                    # 창은 닫고 나가야 한다. 그래서 여기서 따로 삼킨다.
                    try:
                        visit.action_status = action_fn(tool_id, window, title, backend)
                    except Exception as exc:
                        visit.action_status = f"{type(exc).__name__}: {exc}"
                        print(f"[WARNING] {tool_id} 창 안 조작 예외(창은 닫고 계속): "
                              f"{visit.action_status}")
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
# tool 창 안 조작 흐름 - 여는 버튼 -> 창 확인 -> 안에서 차례로 클릭.
#
# **대화상자는 로컬 창이 아니다.** Remote Monitoring 창은 장비 화면의 원격 뷰라,
# MemoPrint 나 Work Sheet 를 열면 그 창이 뷰 **안에** 그려진다 - 로컬 top-level 창
# 열거로는 절대 찾을 수 없다. 첫 오피스 실행에서 창 제목으로 찾다 실패하고도 "그래도
# 계속" 폴백이 걸려 화면 어딘가의 **다른 Close** 를 눌렀다. 그래서 확인은 창 열거가
# 아니라 **라벨 판독(OCR)** 으로 하고, 확인되지 않으면 누르지 않는다(share_request 와
# 같은 fail-closed actuator 규약).
#
# 세 흐름 모두 모양이 같아 한 엔진으로 돈다:
#   MemoPrint : [Utility] -> Memo Print -> 편집 영역 -> 두 줄 입력
#   Optics    : [Optics...] -> Memory 탭 -> Close
#   Work Sheet: [Work Sheet 아래 버튼] -> File -> Exit
# 차이는 **step 사이의 의존성** 하나다. Optics 의 Close 는 Memory 와 무관하게 누를 수
# 있지만, 드롭다운/popup 항목(File -> Exit, Memo Print -> 편집 영역)은 앞 step 이
# 성공해야만 존재한다. 그래서 `requires_previous` 로 그 차이를 데이터에 적는다.
# ------------------------------------------------------------------

# 흐름 status(장비별 요약에 그대로 실린다). "<흐름이름>:<status>" 로 붙여서 보고한다.
FLOW_OK = "ok"
FLOW_OPENER_FAILED = "opener_failed"       # 여는 버튼의 라벨이 기대와 달랐다.
FLOW_OPENER_NOT_VISIBLE = "opener_not_visible"  # 가림 해제 후에도 좌표가 안 나왔다.
FLOW_WINDOW_NOT_FOUND = "window_not_found"  # 열었는데 창이 확인되지 않았다.
FLOW_SKIPPED = "skipped"

# `_confirm_point` 의 실패 이유. 오피스에서 할 일이 갈리므로 하나로 뭉치지 않는다.
#   NOT_LOCATED    : VLM 이 좌표를 못 찍었다 - 화면에 없거나 **다른 창에 가려졌다**.
#                    Alt+click 으로 가린 창을 밀어내면 되살아날 수 있다.
#   LABEL_REJECTED : 좌표는 나왔는데 그 자리 라벨이 기대와 다르다 - 이미 보이는
#                    화면이므로 창을 밀어내도 달라지지 않는다(엉뚱한 창만 뒤로 간다).
CONFIRM_OK = "ok"
CONFIRM_NOT_LOCATED = "not_located"
CONFIRM_LABEL_REJECTED = "label_rejected"


@dataclass
class FlowStep:
    """확인 후 클릭할 요소 하나.

    `required` 가 비면 **라벨 요구 없이 forbidden 만** 본다. 오피스에서 실제 문구를
    모르는 요소(예: 'Work Sheet' 아래 버튼)를 위한 것이다 - 읽힌 토큰은 콘솔에 찍히므로
    첫 실행 뒤 그 값을 required 에 박으면 게이트가 온전해진다.

    `requires_previous` 는 "앞 step 의 클릭이 성공해야 이 요소가 존재한다" 는 뜻이다.
    드롭다운 항목(File -> Exit)이 그렇다. 앞이 실패했는데 찾아 나서면 열리지도 않은
    메뉴 자리를 클릭한다.
    """

    target: object
    required: tuple = ()
    forbidden: tuple = ()
    requires_previous: bool = False
    input_text: str = ""


@dataclass
class InToolFlow:
    """tool 창 안에서 보여줄 조작 한 벌."""

    name: str
    opener: FlowStep
    steps: list


def _confirm_point(
    image, step: FlowStep,
    *, locate_fn, read_tokens_fn, policy,
):
    """좌표를 찍고 그 자리 라벨을 읽어 **확인된 경우에만** 그 점을 돌려준다.

    `(point, reason)` 을 돌려준다. 좌표는 VLM 이 정하고 OCR 은 확인만 한다(이 저장소의
    클릭 규약). 확인이 안 되면 point 는 None - 원격 뷰에서 한 칸 어긋난 클릭은 무엇을
    눌렀는지 알 수 없다. 클릭을 여기서 하지 않는 이유는 호출부가 "라벨은 확인됐는데
    클릭이 실패했다" 와 "라벨부터 확인이 안 됐다" 를 구분해야 하기 때문이다.

    reason 을 함께 주는 이유: **좌표 미검출과 라벨 불일치는 대응이 다르다.** 미검출은
    다른 창이 덮은 것일 수 있어 Alt+click 으로 되살릴 여지가 있지만, 라벨 불일치는
    이미 보이는 화면을 잘못 짚은 것이라 창을 밀어내도 달라지지 않는다.
    """
    from poc.workflow_3.monitor.share_request import accepts_label, classify_label

    key = step.target.key
    point = locate_fn(image, step.target)
    if point is None:
        print(f"[WARNING] 좌표 미검출 - 클릭 안 함: {key}")
        return None, CONFIRM_NOT_LOCATED

    tokens = read_tokens_fn(image, point, key)
    if not step.required:
        # 기대 문구를 모르는 요소 - forbidden 만 거른다. 읽힌 토큰을 반드시 남긴다
        # (다음 실행에서 이 값을 required 로 승격하는 것이 목적이다).
        verdict = classify_label(tokens, ((),), step.forbidden)
        blocked = verdict == "forbidden"
        print(f"[INFO] 라벨 요구 없음({key}) - 읽힌 토큰={tokens!r}"
              f"{' -> 금지 토큰이라 클릭 안 함' if blocked else ''}")
        if blocked:
            return None, CONFIRM_LABEL_REJECTED
        return point, CONFIRM_OK

    verdict = classify_label(tokens, step.required, step.forbidden)
    if not accepts_label(verdict, policy):
        print(
            f"[WARNING] 라벨 확인 실패 - 클릭 안 함: {key} "
            f"verdict={verdict} policy={policy} tokens={tokens!r}"
        )
        return None, CONFIRM_LABEL_REJECTED
    return point, CONFIRM_OK


def run_in_tool_flow(
    tool_window,
    tool_title: str,
    tool_backend: str,
    flow: InToolFlow,
    *,
    capture_fn,
    locate_fn,
    read_tokens_fn,
    click_fn,
    type_fn=None,
    reveal_fn=None,
    sleep_fn,
    settle_sec: float = 1.5,
    confirm_policy: str = "strict",
    attempts: int = 2,
    reveal_attempts: int = 2,
) -> str:
    """tool 창 안에서 흐름 한 벌을 확인하며 차례로 누른다. "<흐름>:<status>" 반환.

    협력자(share_request 와 같은 모양이라 배선을 그대로 재사용한다):
      capture_fn(window)                  -> image
      locate_fn(image, target)            -> point dict | None (**이미지 픽셀 좌표**)
      read_tokens_fn(image, point, key)   -> list[str]
      click_fn(window, image, point, key) -> None
      type_fn(text, key)                  -> None (input_text 가 있는 step 만)
      reveal_fn(window, image, round_index) -> bool
        여는 버튼이 **가려져서** 안 보일 때 가린 창을 밀어낸다(Alt+click). 없으면
        종전처럼 즉시 포기한다.

    다섯 가지가 계약이다.

    ① **확인되지 않으면 누르지 않는다.** 좌표만 믿고 누르면 원격 뷰의 엉뚱한 버튼을
       누르게 된다. 실제로 첫 오피스 실행이 그렇게 깨졌다.
    ② **첫 step 의 라벨이 창이 떴다는 유일한 증거다.** 그것이 확인되기 전에는 뒤
       요소를 찾아 나서지 않는다 - 창이 없는 화면에서 'Close'/'Exit' 를 찾으면 tool 창
       자체의 닫기 같은 다른 것을 누른다. 확인 안 된 채 남은 창은 다음 단계의 tool 창
       닫기가 정리하므로, 여기서 무리하게 닫는 것보다 안 누르는 편이 낫다.
    ③ **창이 확인된 뒤에는 독립 step 을 끝까지 시도한다.** 중간 step 이 실패해도
       `requires_previous=False` 인 뒤 step 은 눌러 본다.

    ④ **여는 버튼이 안 보이는 것과 라벨이 다른 것은 다르다.** 좌표가 아예 안 나오면
       다른 창이 덮었을 수 있으므로 `reveal_fn` 으로 밀어내고 다시 찾는다. 라벨이
       다르게 읽혔다면 화면은 이미 보이는 것이라 밀어내지 않는다.

    `attempts` 는 여는 버튼의 재시도 횟수다. 원격 뷰는 커서 이동을 따라오지 못해 첫
    클릭이 삼켜지는 일이 있어(오피스 1회차 증상: "마우스만 이동") 창이 확인되지 않으면
    한 번 더 누른다. `reveal_attempts` 는 그와 **별개 예산**이다 - 창이 여러 장 겹쳐
    있으면 Alt+click 을 여러 번 해야 드러나는데, 그 때문에 클릭 재시도가 줄어들 이유는
    없다(둘은 서로 다른 실패를 고친다).
    """
    max_attempts = max(1, attempts)

    def _tag(status: str) -> str:
        return f"{flow.name}:{status}"

    if not flow.steps:
        return _tag(FLOW_SKIPPED)

    first_step = flow.steps[0]
    first_point = None
    first_image = None

    # --- 여는 버튼 -> 창이 떴는지 확인(= 첫 step 의 라벨 판독) ---
    max_reveals = max(0, reveal_attempts) if reveal_fn is not None else 0
    try:
        attempt = 0
        reveals = 0
        while attempt < max_attempts:
            attempt += 1
            print(f"[INFO] [{flow.name}] {flow.opener.target.key} 확인 후 클릭 "
                  f"(시도 {attempt}/{max_attempts})")
            image = capture_fn(tool_window)
            point, reason = _confirm_point(
                image, flow.opener,
                locate_fn=locate_fn, read_tokens_fn=read_tokens_fn, policy=confirm_policy,
            )
            if point is None:
                if reason != CONFIRM_NOT_LOCATED or max_reveals == 0:
                    # 라벨이 다르게 읽혔다면 다시 눌러도 같은 화면이다 - 즉시 포기.
                    return _tag(FLOW_OPENER_FAILED)
                if reveals >= max_reveals:
                    print(f"[WARNING] [{flow.name}] 가림 해제 {reveals}회 후에도 "
                          f"{flow.opener.target.key} 를 찾지 못했습니다.")
                    return _tag(FLOW_OPENER_NOT_VISIBLE)
                reveals += 1
                print(f"[INFO] [{flow.name}] {flow.opener.target.key} 가 안 보입니다 - "
                      f"가린 창을 밀어냅니다({reveals}/{max_reveals})")
                if not reveal_fn(tool_window, image, reveals):
                    return _tag(FLOW_OPENER_NOT_VISIBLE)
                # 가림 해제는 '클릭이 삼켜졌다' 재시도 예산을 쓰지 않는다 - 둘은 서로
                # 다른 실패를 고친다. `reveals` 가 상한을 가지므로 무한 루프는 없다.
                attempt -= 1
                continue
            click_fn(tool_window, image, point, flow.opener.target.key)

            sleep_fn(settle_sec)  # 창이 그려질 시간(원격이라 로컬보다 느리다).

            print(f"[INFO] [{flow.name}] 창 확인({first_step.target.key} 판독)")
            image = capture_fn(tool_window)
            first_point, _ = _confirm_point(
                image, first_step,
                locate_fn=locate_fn, read_tokens_fn=read_tokens_fn, policy=confirm_policy,
            )
            if first_point is not None:
                first_image = image
                break
            if attempt >= max_attempts:
                print(f"[WARNING] [{flow.name}] 창을 확인하지 못함 - 이후 요소를 찾지 "
                      "않습니다(tool 창 닫기가 정리합니다).")
                return _tag(FLOW_WINDOW_NOT_FOUND)
            print(f"[INFO] [{flow.name}] 창 미확인 - 클릭이 삼켜졌을 수 있어 다시 누릅니다")
        else:
            # while 이 예산 소진으로 끝난 경우(위 분기에서 이미 반환되는 것이 정상).
            return _tag(FLOW_WINDOW_NOT_FOUND)
    except Exception as exc:
        print(f"[WARNING] [{flow.name}] 여는 버튼/창 확인 중 예외: {type(exc).__name__}: {exc}")
        return _tag(FLOW_OPENER_FAILED)

    # --- 여기부터 창이 떠 있는 것이 확인된 상태다 ---
    failed_key = ""
    previous_ok = True
    for index, step in enumerate(flow.steps):
        key = step.target.key
        if step.requires_previous and not previous_ok:
            print(f"[INFO] [{flow.name}] 앞 단계 실패로 {key} 는 건너뜁니다"
                  "(열리지 않은 메뉴 자리를 누르지 않기 위해).")
            break

        try:
            if index == 0:
                image, point = first_image, first_point  # 방금 확인한 것을 다시 안 찾는다.
            else:
                sleep_fn(settle_sec)
                image = capture_fn(tool_window)
                point, _ = _confirm_point(
                    image, step,
                    locate_fn=locate_fn, read_tokens_fn=read_tokens_fn,
                    policy=confirm_policy,
                )
            if point is None:
                previous_ok = False
                failed_key = failed_key or key
                continue

            print(f"[INFO] [{flow.name}] {key} 클릭")
            click_fn(tool_window, image, point, key)
            if step.input_text:
                if type_fn is None:
                    raise RuntimeError(f"텍스트 입력 협력자 없음: {key}")
                print(f"[INFO] [{flow.name}] {key} 텍스트 입력(chars={len(step.input_text)})")
                type_fn(step.input_text, key)
            previous_ok = True
        except Exception as exc:
            print(f"[WARNING] [{flow.name}] {key} 클릭 예외: {type(exc).__name__}: {exc}")
            previous_ok = False
            failed_key = failed_key or key

    if failed_key:
        return _tag(f"step_failed({failed_key})")
    sleep_fn(settle_sec)
    return _tag(FLOW_OK)


# US 기호 배열에서 Shift 를 함께 눌러야 나오는 문자 -> 그 자리의 기본 문자.
# 오피스 PC 는 한글 Windows 지만 기호 배열은 US 표준이다. 이 표가 틀린 배열에서는
# '!' 가 '1' 로 들어가므로, 그때는 `DEMO_RCS_MEMO_TEXT` 로 문구를 바꾼다.
SHIFTED_CHARS = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7",
    "*": "8", "(": "9", ")": "0", "_": "-", "+": "=", "{": "[", "}": "]",
    "|": "\\", ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`",
}


# 대문자를 만드는 방법. 오피스 실측 2회가 이 선택지를 만들었다(2026-08-24).
#   caps_all : 문구 전체를 Caps Lock **한 쌍**으로 감싸고 모든 글자를 소문자 기본 키로
#              보낸다 -> 전부 대문자로 찍힌다(기본값). 토글이 2번뿐이라 유실/경합
#              위험이 최소이고, 실패해도 '전부 소문자' 라는 읽을 수 있는 형태로 어긋난다.
#   caps     : 대문자마다 Caps 를 켜고 끈다. 4회차에서 이것이 memo 를 깨뜨렸다 -
#              토글은 상태라서 (a) 장비가 적용하기 전에 글자가 도착하거나 (b) 24번 중
#              하나만 유실되면 그 뒤 글자가 **전부** 반대 case 가 된다.
#   shift    : Shift 를 쥔 채 기본 키를 누른다. 2회차에서 글자는 도착했지만 소문자로
#              들어왔다 - 이 원격은 쥐고 있는 수정자를 실어 보내지 않는다. pynput 의
#              `Key.shift` 는 Windows 에서 VK.LSHIFT + scancode 0x2A, 즉 실제 왼쪽
#              Shift 와 같은 신호이므로 '다른 Shift 를 쓰면 된다' 는 선택지는 없다.
#   type     : pynput 의 `type()` 에 그대로 맡긴다. 1회차 경로이며 **글자가 사라진다**
#              (vk=0 유니코드 이벤트라 중계할 것이 없다). 원인 재확인용.
SHIFT_MODE_CAPS_ALL = "caps_all"
SHIFT_MODE_CAPS = "caps"
SHIFT_MODE_SHIFT = "shift"
SHIFT_MODE_TYPE = "type"
KNOWN_SHIFT_MODES = (
    SHIFT_MODE_CAPS_ALL, SHIFT_MODE_CAPS, SHIFT_MODE_SHIFT, SHIFT_MODE_TYPE,
)
DEFAULT_SHIFT_MODE = SHIFT_MODE_CAPS_ALL


def resolve_shift_mode(raw) -> str:
    """대문자 입력 방식. 오타는 조용히 '입력 안 함' 이 되지 않게 기본값으로 되돌린다."""
    mode = (raw or "").strip().lower()
    if not mode:
        return DEFAULT_SHIFT_MODE
    if mode not in KNOWN_SHIFT_MODES:
        print(f"[WARNING] 알 수 없는 대문자 입력 방식: {mode!r} - "
              f"{DEFAULT_SHIFT_MODE} 로 대체. 가능한 값: {', '.join(KNOWN_SHIFT_MODES)}")
        return DEFAULT_SHIFT_MODE
    return mode


def shift_plan(char: str) -> tuple:
    """`(눌러야 할 기본 문자, Shift 필요 여부)`.

    오피스 1회차(2026-08-24)에서 **Shift 글자만** 통째로 사라졌다("Infra. Tech
    Center!!" -> "nfra. ech enter"). 빠진 것이 정확히 I/T/C/!/O/S/S 였고, 전부
    Shift 조합이다. 그래서 Shift 는 pynput 의 `type()` 에 맡기지 않고 우리가 직접
    잡는다 - 그래야 수정자에 체류 시간을 줄 수 있다.
    """
    if char.isalpha() and char.isupper():
        return char.lower(), True
    if char in SHIFTED_CHARS:
        return SHIFTED_CHARS[char], True
    return char, False


def shift_symbols(text: str) -> list:
    """Shift 기호와 **그 자리에 실제로 들어올 문자** 목록. 없으면 빈 리스트.

    Caps Lock 은 글자만 바꾸므로 대문자는 여기 대상이 아니다. 문제는 기호다 - 이
    원격은 쥐고 있는 수정자를 실어 보내지 않으므로 `!` 는 `1`, `"` 는 `'` 로 들어온다.
    조용히 틀린 글자를 넣지 않으려고, 입력 **전에** 무엇이 어떻게 들어올지 콘솔에
    적는다. 오피스에서 화면과 대조할 유일한 근거다(Mac 에서는 그 화면을 볼 수 없다).
    """
    found = []
    for char in text:
        base, needs_shift = shift_plan(char)
        if needs_shift and not char.isalpha():
            found.append((char, base))
    return found


def _local_caps_on():
    """로컬 PC 의 Caps Lock 상태. 알 수 없으면 None(Windows 전용 조회).

    None 과 False 를 **구분**하는 것이 요점이다 - 모르는 상태에서 토글하면 꺼져
    있던 Caps 를 켜는 쪽이 될 수 있다.
    """
    try:
        import ctypes

        return bool(ctypes.windll.user32.GetKeyState(0x14) & 1)  # VK_CAPITAL
    except Exception:
        return None


def type_multiline_text(
    text: str,
    key: str,
    *,
    action_enabled: bool,
    keyboard=None,
    enter_key=None,
    shift_key=None,
    caps_key=None,
    shift_mode: str = "",
    sleep_fn=time.sleep,
    char_delay_sec: float = 0.08,
    shift_settle_sec: float = 0.12,
    caps_settle_sec: float = 0.4,
    post_dwell_sec: float = 0.0,
    is_aborted_fn=None,
    caps_state_fn=None,
) -> bool:
    """포커스된 입력창에 줄바꿈을 Enter 로 바꿔 천천히 입력한다.

    원격 tool 화면은 입력을 샘플링하므로 문자열 전체를 한 번에 보내지 않고 글자마다
    간격을 둔다. `SAFE_MODE=1` 에서는 pynput 을 만들기 전 반환해 키 입력을 완전히
    차단한다. keyboard/enter_key/shift_key/is_aborted_fn 은 Mac 단위 테스트용
    주입점이다.

    **대문자는 Caps Lock 으로 만든다**(`shift_mode`, 기본 caps). 오피스 실측 2회가
    이 선택을 강제했다:

      * 1회차 - 대문자와 '!' 만 **정확히 사라졌다**(소문자는 하나도 안 빠졌다). 원인은
        pynput win32 구현이다(`pynput/keyboard/_win32.py:83-92`): `VkKeyScan(char)` 이
        "Shift 필요" 라고 답하면 `vk=0 / scan=유니코드 코드포인트 / flags=UNICODE` 로
        보낸다 - **vk 도 scan code 도 없는 이벤트**라, vk/scancode 를 중계하는 RCS
        원격에는 중계할 것이 없다.
      * 2회차 - Shift 를 직접 쥐고 기본 키를 누르니 **글자는 도착했지만 소문자였다**.
        기본 키는 진짜 vk 이벤트라 중계되지만, 이 원격은 키를 개별 타건으로 넘기고
        **쥐고 있는 수정자를 함께 실어 보내지 않는다**.

    그래서 필요한 것은 '쥐는 수정자' 가 아니라 **상태를 남기는 키**다. Caps Lock 은
    평범한 vk 타건이라 중계되고, 그 상태는 장비 쪽 OS 가 기억한다. 켠 뒤 반드시
    끈다(`finally`) - 켠 채 끝나면 그 뒤 입력이 전부 대문자가 되고, `SendInput` 은
    전역이라 **로컬 PC 의 Caps Lock 도 같이 켜진 채로 남는다**.

    Caps Lock 은 글자만 바꾸므로 `!` 같은 기호는 여전히 Shift 쥐기로 간다(이 원격에서
    안 먹을 수 있다 - 그때는 `DEMO_RCS_MEMO_TEXT` 로 문구를 바꾸는 것이 답이다).

    `post_dwell_sec` 은 입력을 끝낸 뒤 머무는 시간이다("글자를 다 넣고 2초 기다린 뒤
    Close" - 사용자 지시). 화면에 반영될 시간이면서 관객이 읽을 시간이다.

    **긴급 해제(전역 단축키)를 글자마다 확인한다.** 이 저장소의 마우스 출력은 전부
    `abort_switch` 를 지나는데(`mouse_utils.click_at_screen`), 메모 입력은 기본값에서
    수십 글자 x 0.08s = 수 초간 키를 흘려보내므로 그 사이 단축키가 안 먹으면
    "해제됐다" 고 느껴지지 않는다. 도중에 끊긴 경우 False 를 돌려준다.
    """
    if not action_enabled:
        print(
            f"[INFO] [DRY-RUN] 텍스트 입력 생략: target={key}, "
            f"text={text!r}, action_enabled={action_enabled}"
        )
        return True

    aborted = is_aborted_fn if is_aborted_fn is not None else is_aborted
    if aborted():
        print(f"[WARNING] 긴급 해제 상태 - 텍스트 입력 생략: target={key}, "
              f"reason={abort_reason()}")
        return False

    mode = resolve_shift_mode(shift_mode)
    if keyboard is None or enter_key is None or shift_key is None or caps_key is None:
        key_enum = None
        try:
            from pynput.keyboard import Key, Controller as KeyboardController

            key_enum = Key
        except ImportError as exc:
            # 키보드 대역이 주입돼 있으면 계속한다 - 안 쓰는 특수 키 때문에 죽을 이유가
            # 없다(Mac 단위 테스트가 이 경로로 돈다). 대역조차 없으면 진짜 실패다.
            if keyboard is None:
                raise RuntimeError(
                    "pynput.keyboard 미설치 - 텍스트를 입력할 수 없음"
                ) from exc
        else:
            keyboard = keyboard if keyboard is not None else KeyboardController()
        if key_enum is not None:
            enter_key = enter_key if enter_key is not None else key_enum.enter
            shift_key = shift_key if shift_key is not None else key_enum.shift
            caps_key = caps_key if caps_key is not None else key_enum.caps_lock

    delay = max(0.0, char_delay_sec)
    shift_settle = max(0.0, shift_settle_sec)
    caps_settle = max(0.0, caps_settle_sec)

    def _tap(tap_key):
        keyboard.press(tap_key)
        keyboard.release(tap_key)

    def _type_via_caps(base):
        """Caps Lock 을 켜고 기본 키를 눌렀다 끈다 - 상태를 장비가 기억한다."""
        _tap(caps_key)
        try:
            sleep_fn(caps_settle)   # 장비가 Caps 상태를 반영할 틱(토글은 링크를 건넌다).
            keyboard.press(base)
            keyboard.release(base)
            sleep_fn(caps_settle)
        finally:
            # 켠 채 남으면 이후 입력이 전부 대문자가 되고 로컬 PC 도 켜진 채 남는다.
            _tap(caps_key)

    def _restore_local_caps():
        """엔지니어 PC 의 Caps 가 켜진 채 남았으면 끈다.

        `SendInput` 은 전역이라 우리가 보낸 토글이 **로컬 PC 에도** 걸린다. 장비 쪽
        상태는 읽을 수 없지만 로컬은 읽을 수 있으므로, 최소한 사람이 쓰는 키보드는
        원래대로 돌려놓는다. **모르면(None) 건드리지 않는다** - 추측으로 토글하면
        꺼져 있던 것을 켜는 쪽이 될 수 있다.
        """
        reader = caps_state_fn if caps_state_fn is not None else _local_caps_on
        try:
            state = reader()
        except Exception:
            return
        if state is True:
            print("[INFO] 로컬 Caps Lock 이 켜진 채라 되돌립니다.")
            _tap(caps_key)

    def _type_via_shift(base):
        """Shift 를 쥔 채 기본 키를 누른다(기호는 이 방법밖에 없다)."""
        keyboard.press(shift_key)
        try:
            sleep_fn(shift_settle)   # 원격이 수정자 상태를 등록할 틱.
            keyboard.press(base)
            keyboard.release(base)
            sleep_fn(shift_settle)   # 키가 Shift 눌린 상태로 넘어갈 틱.
        finally:
            # 눌린 채 남으면 그 뒤 입력과 클릭이 전부 Shift 조합으로 변질된다.
            keyboard.release(shift_key)

    print(f"[INFO] 텍스트 입력 시작: target={key}, 대문자방식={mode}, "
          f"글자수={len(text)}")
    risky = shift_symbols(text)
    if risky and mode != SHIFT_MODE_TYPE:
        pairs = ", ".join(f"{c!r}->{b!r}" for c, b in risky)
        print(
            f"[WARNING] Shift 기호는 이 원격을 못 건넙니다 - 다음이 어긋날 수 있습니다: "
            f"{pairs}. 화면이 이상하면 DEMO_RCS_MEMO_TEXT 로 문구에서 그 기호를 빼세요."
        )
    def _type_body():
        """문구를 글자마다 보낸다. 긴급 해제로 중단되면 False."""
        caps_on = mode == SHIFT_MODE_CAPS_ALL
        for index, char in enumerate(text):
            if aborted():
                print(f"[WARNING] 긴급 해제 - 텍스트 입력 중단: target={key}, "
                      f"입력={index}/{len(text)}글자")
                return False
            if char == "\n":
                keyboard.press(enter_key)
                keyboard.release(enter_key)
                sleep_fn(delay)
                continue

            if caps_on and char.isalpha():
                # Caps 가 켜져 있으므로 **소문자 기본 키**를 보내면 대문자로 찍힌다.
                # 글자마다 수정자가 없다 = 경합도 유실 위험도 없다.
                keyboard.type(char.lower())
                sleep_fn(delay)
                continue

            base, needs_shift = shift_plan(char)
            if not needs_shift or mode == SHIFT_MODE_TYPE:
                # 수정자가 필요 없는 글자는 1회차에서도 정상 입력됐다 - 건드리지 않는다.
                keyboard.type(char)
                sleep_fn(delay)
                continue

            if mode == SHIFT_MODE_CAPS and char.isalpha():
                _type_via_caps(base)
            else:
                # 기호(Caps Lock 이 못 바꾼다) 또는 shift 모드.
                _type_via_shift(base)
            sleep_fn(delay)
        return True

    if mode == SHIFT_MODE_CAPS_ALL and any(c.isupper() for c in text):
        # 토글을 **한 쌍**만 쓴다. 4회차에서 글자마다 토글한 것이 memo 를 깨뜨렸다:
        # 24번 중 하나만 유실되면 장비의 caps 상태가 뒤집혀 그 뒤가 전부 틀린다.
        _tap(caps_key)
        try:
            sleep_fn(caps_settle)
            if not _type_body():
                return False
        finally:
            _tap(caps_key)
            _restore_local_caps()
    elif not _type_body():
        return False

    print(f"[INFO] 텍스트 입력 완료: target={key}, text={text!r}")
    if post_dwell_sec > 0:
        print(f"[INFO] 입력 후 {post_dwell_sec:.1f}s 체류(화면 반영 + 관객이 읽을 시간)")
        sleep_fn(post_dwell_sec)
    return True


def perform_remote_click(
    window, screen_point: dict, key: str,
    *, foreground_fn, move_fn, click_fn, sleep_fn, settle_sec: float,
    press_modifier_fn=None, release_modifier_fn=None,
    modifier_settle_sec: float = 0.0,
) -> None:
    """원격 뷰의 한 지점을 실제로 눌리게 클릭한다.

    순서가 계약이다: **전면화 -> 커서 이동 -> 체류 -> (수정자 누름 -> 체류) -> 누름**.

      * 전면화를 빼면 포커스 없는 창의 첫 클릭이 창 활성화에 쓰이고 버튼에는 닿지
        않는다. 2026-08-19 오피스에서 "커서는 버튼 위로 가는데 클릭이 안 먹는" 증상의
        원인이며, 이 저장소의 다른 원격 뷰 조작(`sem_monitor.controller`)은 제스처마다
        같은 일을 한다. 전면화에 **실패하면 누르지 않는다**(fail-closed) - 포커스가
        어디 있는지 모르는 상태의 클릭은 어디로 갈지 모른다.
      * 이동과 누름 사이의 체류는 원격이 커서 위치를 따라올 시간이다.

    `foreground_fn` 이 None 이면(그 수단이 없는 환경) 막지 않는다 - 그건 게이트가
    아니라 부재이고, 여기서 시연을 통째로 멈출 이유가 없다.

    `press_modifier_fn` 은 Alt+click(가린 창 밀어내기)용이고, **커서가 도착한 뒤에**
    잡는 것이 계약이다("오른쪽 아래로 마우스를 옮긴 뒤 Alt+click" - 사용자 설명,
    2026-08-24). 세 가지가 이 자리를 정한다:

      * **전면화보다 뒤**여야 한다. `window_utils.foreground_window` 는 Windows 의
        foreground-lock 을 우회하려고 더미 **Alt down/up** 을 합성 주입하므로, 먼저
        잡은 Alt 를 그 up 이 놓아버려 수정자 없는 평범한 클릭이 된다(커서는 맞는데
        창이 안 밀리는, 원인 찾기 어려운 실패).
      * **커서 이동보다 뒤**여야 한다. Alt 를 쥔 채 커서를 끌고 가면 그 이동 전체가
        Alt 눌린 상태가 된다 - 원격이 그것을 창 조작 제스처로 읽을 여지를 만들 이유가
        없다.
      * **누름보다 한 틱 앞**이어야 한다(`modifier_settle_sec`). 원격은 입력을
        샘플링하므로 Alt down 과 버튼 down 이 같은 틱에 들어가면 수정자 없는 클릭으로
        넘어갈 수 있다 - `click_at_screen(hold_sec=)` 이 생긴 것과 같은 이유다.

    해제는 `finally` 로 보장한다 - 눌린 채 남으면 이후 **모든** 클릭이 Alt+click 으로
    변질된다(window_utils 가 같은 이유로 경고하는 stuck-modifier). 전면화에 실패하면
    애초에 누르지 않으므로 Alt 도 잡지 않는다.
    """
    if foreground_fn is not None and not foreground_fn(window):
        raise RuntimeError(f"tool 창 foreground 확보 실패 - 클릭하지 않음: {key}")
    try:
        move_fn(screen_point, key)
        sleep_fn(settle_sec)
        if press_modifier_fn is not None:
            press_modifier_fn()
            sleep_fn(max(0.0, modifier_settle_sec))
        click_fn(screen_point, key)
    finally:
        if release_modifier_fn is not None:
            release_modifier_fn()


# ------------------------------------------------------------------
# 가려진 여는 버튼 되살리기 - Alt+click 으로 덮은 창을 뒤로 밀어낸다.
#
# 사용자 보고(2026-08-24): Utility 버튼은 tool 모니터 **오른쪽 아래**에 있는데 다른
# 창이 그 위에 떠서 VLM 이 아예 찾지 못하는 일이 있다. 엔지니어는 그 자리를
# **Alt+click** 해서 창을 뒤로 밀고 Utility 를 되살린다. 이 저장소의 "여는 버튼을 못
# 찾으면 즉시 포기" 규칙은 '다시 눌러도 같은 화면' 이라는 전제에서 나온 것이므로,
# 화면을 바꿀 수단이 있는 이 경우에만 예외가 된다.
#
# 누를 지점은 **가린 창이 아니라 Utility 가 있어야 할 자리**다. 그 위를 덮고 있는
# 것이 밀어낼 창이므로 같은 좌표를 누르면 된다. Mac 에서 이 화면을 볼 수 없어 비율은
# env 로 옮길 수 있게 둔다(DEMO_RCS_REVEAL_X_RATIO / _Y_RATIO).
# ------------------------------------------------------------------

# Utility 가 있는 자리(창 크기 대비 비율). 오른쪽 아래이되 창 테두리/스크롤바를 피해
# 약간 안쪽으로 둔다.
DEFAULT_REVEAL_X_RATIO = 0.88
DEFAULT_REVEAL_Y_RATIO = 0.92


def covering_window_point(
    width: int, height: int,
    *, x_ratio: float = DEFAULT_REVEAL_X_RATIO, y_ratio: float = DEFAULT_REVEAL_Y_RATIO,
) -> dict:
    """가린 창을 밀어낼 지점(**이미지 픽셀 좌표**). 항상 프레임 안으로 자른다.

    비율을 잘못 줘도 창 밖을 누르지 않게 하는 것이 요점이다 - 창 밖 클릭은 그 자리에
    있는 다른 앱으로 가고, 그건 시연 중에 가장 하면 안 되는 일이다.
    """
    max_x = max(0, int(width) - 1)
    max_y = max(0, int(height) - 1)
    x = min(max_x, max(0, int(round(int(width) * float(x_ratio)))))
    y = min(max_y, max(0, int(round(int(height) * float(y_ratio)))))
    return {"x": x, "y": y}


def alt_hold_hooks(
    *, action_enabled: bool, keyboard=None, alt_key=None, is_aborted_fn=None,
):
    """Alt 를 쥐고/놓는 `(press, release)` 한 쌍을 만든다.

    `SAFE_MODE=1` 이나 긴급 해제 상태에서는 **아무 키도 내보내지 않는다**. release 는
    누른 적이 없어도 안전하게 호출될 수 있어야 한다(`perform_remote_click` 이
    `finally` 에서 무조건 부르기 때문). keyboard/alt_key/is_aborted_fn 은 Mac 단위
    테스트용 주입점이다.
    """
    aborted = is_aborted_fn if is_aborted_fn is not None else is_aborted
    state = {"held": False}

    def _press():
        if not action_enabled:
            print("[INFO] [DRY-RUN] Alt 누름 생략(SAFE_MODE)")
            return
        if aborted():
            print(f"[WARNING] 긴급 해제 상태 - Alt 누름 생략: reason={abort_reason()}")
            return
        board, key = _resolve_keyboard(keyboard, alt_key)
        board.press(key)
        state["held"] = True

    def _release():
        if not state["held"]:
            return
        board, key = _resolve_keyboard(keyboard, alt_key)
        board.release(key)
        state["held"] = False

    return _press, _release


def _resolve_keyboard(keyboard, alt_key):
    """주입된 대역이 있으면 그것을, 없으면 pynput 을 쓴다(캐시해 같은 객체를 유지)."""
    if keyboard is not None and alt_key is not None:
        return keyboard, alt_key
    from pynput.keyboard import Key, Controller as KeyboardController

    board = keyboard if keyboard is not None else _shared_keyboard()
    return board, (alt_key if alt_key is not None else Key.alt)


_KEYBOARD_CACHE = {}


def _shared_keyboard():
    """pynput KeyboardController 를 하나만 만든다 - press 와 release 가 같은 객체여야
    Alt 상태가 이어진다(새로 만들면 놓지 못한 Alt 가 남을 수 있다)."""
    if "board" not in _KEYBOARD_CACHE:
        from pynput.keyboard import Controller as KeyboardController

        _KEYBOARD_CACHE["board"] = KeyboardController()
    return _KEYBOARD_CACHE["board"]


# ------------------------------------------------------------------
# 장비별 조작 흐름 배정.
# ------------------------------------------------------------------

FLOW_MEMO_PRINT = "memo_print"
FLOW_OPTICS = "optics"
FLOW_WORKSHEET = "worksheet"
KNOWN_FLOW_NAMES = (FLOW_MEMO_PRINT, FLOW_OPTICS, FLOW_WORKSHEET)

# 시연 기본 배정 - 장비마다 다른 조작을 보여줘야 "자동화" 로 보인다.
# MCD019 에서 memo_print 는 오피스 확인됨(2026-08-24). 두 번째 장비는 MCDC10 으로
# 바꾸고 **같은 것을 반복하지 않는다** - 'File' 쪽(Work Sheet -> File -> Exit)만 본다.
DEFAULT_TOOL_FLOWS = {"mcd019": FLOW_MEMO_PRINT, "mcdc10": FLOW_WORKSHEET}


def parse_flow_map(raw, default: dict) -> dict:
    """"MCD019=memo_print,MCDC10=worksheet" 를 {소문자 장비: 흐름} 으로 만든다. 비면 default.

    형식이 깨진 항목은 버리고 계속한다 - 시연 직전 오타로 스크립트가 죽는 것보다,
    그 항목만 기본 흐름으로 도는 편이 낫다.
    """
    mapping = {}
    for chunk in (raw or "").replace(";", ",").split(","):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" not in entry:
            print(f"[WARNING] 흐름 배정 형식 오류(무시): {entry!r} - 'TOOL=flow' 로 쓰세요")
            continue
        tool, _, flow_name = entry.partition("=")
        tool, flow_name = tool.strip().lower(), flow_name.strip().lower()
        if not tool or not flow_name:
            print(f"[WARNING] 흐름 배정 형식 오류(무시): {entry!r}")
            continue
        mapping[tool] = flow_name
    return mapping or dict(default)


def resolve_flow_name(tool_id: str, mapping: dict, default_flow: str) -> str:
    """장비에 배정된 흐름 이름. 미등록/오타는 default_flow.

    오타난 이름을 조용히 '아무것도 안 함' 으로 만들지 않는다 - 시연에서 왜 아무 일도
    안 일어나는지 찾을 수 없게 된다.
    """
    name = (mapping or {}).get((tool_id or "").strip().lower(), default_flow)
    if name not in KNOWN_FLOW_NAMES:
        print(f"[WARNING] 알 수 없는 흐름 이름: {name!r} (장비={tool_id}) - "
              f"{default_flow} 로 대체. 가능한 값: {', '.join(KNOWN_FLOW_NAMES)}")
        return default_flow
    return name


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
        action = f" / {visit.action_status}" if visit.action_status else ""
        detail = f" {visit.error}" if visit.error else ""
        print(
            f"[INFO]   {visit.tool_id}: {visit.status} / {closed}{action} / "
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
    from poc.workflow_3.monitor.cycle import (
        _list_process_windows,
        _scan_rcs_processes,
        _terminate_process,
    )
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
            list_windows_fn=_list_process_windows,
            terminate_fn=_terminate_process,
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


def build_flows(memo_text: str = ""):
    """시연 흐름 정의 - 이름 -> InToolFlow. `memo_text` 가 비면 기본 문구를 쓴다.

    설명문은 이 저장소의 규약대로 **첫 글자를 anchor** 로 잡게 쓰고, 화면 안 위치
    단서를 함께 준다(tool 창에는 버튼이 많아 라벨만으로는 coarse 단계가 흔들린다).
    """
    from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

    memo_text = memo_text or DEFAULT_MEMO_TEXT

    memo_print_flow = InToolFlow(
        name=FLOW_MEMO_PRINT,
        opener=FlowStep(
            TargetConfig(
                key="utility_button",
                description=(
                    "the 'Utility' button in the Remote Monitoring window's button "
                    "area. Use the first letter 'U' as the anchor, then click safely "
                    "inside the Utility button area. This button opens its dropdown "
                    "upward, above the Utility button."
                ),
            ),
            required=(("utility",),),
            forbidden=("cancel", "stop", "terminat", "exit", "취소", "종료"),
        ),
        steps=[
            FlowStep(
                TargetConfig(
                    key="memo_print_menu_item",
                    description=(
                        "the 'Memo Print' item in the Utility dropdown that opened "
                        "upward above the Utility button. Use the first letter 'M' as "
                        "the anchor, then click safely inside the Memo Print menu item."
                    ),
                ),
                required=(("memo", "print"), ("메모", "프린트")),
                forbidden=("cancel", "exit", "close", "취소", "종료", "닫기"),
            ),
            FlowStep(
                TargetConfig(
                    key="memo_print_editor",
                    description=(
                        "the large editable memo text area inside the popup titled "
                        "'MemoPrint'. Click safely near the upper-left inside the empty "
                        "white editor body, not the MemoPrint title bar and not any "
                        "button around the editor."
                    ),
                ),
                # 편집 영역 자체에는 읽을 글자가 없으므로 확인 근거는 **popup 제목**이다.
                # `required=()` 로 비워 두면 `_confirm_point` 가 정책을 건너뛰고 조기
                # 반환해 strict 에서도 무검증 통과가 된다 - 클릭이라면 그게 설계지만
                # (문구를 모르는 요소를 알아내기 위한 장치) 타이핑은 상태를 남기므로
                # 같은 구멍을 쓸 수 없다. 'memo' 한 needle 이면 'MemoPrint'/'Memo Print'
                # 가 모두 부분 일치하고, 제목이 crop 밖이면 unreadable 이라 기본 정책
                # (lenient)에서는 그대로 진행한다 - strict 만 거부한다.
                required=(("memo",), ("메모",)),
                forbidden=(
                    "cancel", "close", "exit", "ok",
                    "취소", "닫기", "종료", "확인",
                ),
                # Memo Print 항목이 눌리지 않았다면 popup/편집 영역은 존재하지 않는다.
                requires_previous=True,
                input_text=memo_text,
            ),
            FlowStep(
                TargetConfig(
                    key="memo_print_close_button",
                    description=(
                        "the 'Close' button of the popup titled 'MemoPrint'. Use the "
                        "first letter 'C' as the anchor, then click safely inside the "
                        "Close button area. Choose the button that belongs to the "
                        "MemoPrint popup, not the close control of the surrounding "
                        "Remote Monitoring window."
                    ),
                ),
                required=(("close",), ("닫기",)),
                forbidden=("cancel", "terminat", "logout", "abort", "취소", "종료"),
                # **편집 영역 클릭이 popup 존재의 유일한 증거다.** 그것이 실패했는데
                # 'Close' 를 찾아 나서면 화면 어딘가의 다른 Close 를 누른다(엔진 계약
                # ②와 같은 이유). 그때는 popup 이 열린 채 남지만, 엔지니어가 손으로
                # 닫는 편이 정체불명의 Close 를 누르는 것보다 훨씬 낫다.
                requires_previous=True,
            ),
        ],
    )

    worksheet_flow = InToolFlow(
        name=FLOW_WORKSHEET,
        opener=FlowStep(
            TargetConfig(
                key="worksheet_button",
                description=(
                    "the 'Work Sheet' button in the Remote Monitoring window's button "
                    "area. Use the first letter 'W' as the anchor, then click safely "
                    "inside the Work Sheet button area. If the text 'Work Sheet' "
                    "appears more than once, choose the clickable button, which is the "
                    "lower one, not the section label above it."
                ),
            ),
            # 버튼 문구 확인(오피스 확인: 버튼에 'Work Sheet' 라고 쓰여 있다). OCR 이
            # 'WorkSheet' 로 붙여 읽어도 두 needle 이 모두 부분 일치해 통과한다.
            required=(("work", "sheet"), ("워크시트",)),
            forbidden=("cancel", "stop", "terminat", "exit", "취소", "종료"),
        ),
        steps=[
            FlowStep(
                TargetConfig(
                    key="worksheet_file_menu",
                    description=(
                        "the 'File' menu of the window titled 'Work Sheet'. It is a "
                        "SMALL text label in the menu bar just under that window's "
                        "title bar, near the top-left corner of the Work Sheet window, "
                        "close to the 'Work Sheet' title text itself. Use the first "
                        "letter 'F' as the anchor, then click safely inside the small "
                        "File label. Clicking it opens a dropdown menu."
                    ),
                ),
                required=(("file",), ("파일",)),
                # **형제 메뉴 이름을 금지어로 두면 안 된다.** OCR crop 은 클릭 지점
                # 좌우 30% 를 담으므로 메뉴 바에서는 Edit/View/Help 가 반드시 함께
                # 읽힌다. `classify_label` 은 forbidden 을 required 보다 먼저 보고
                # forbidden 은 lenient 에서도 막으므로, 그 목록이 File 클릭을 스스로
                # 막았다(오피스 1회차 실패). File 자체는 드롭다운만 여는 무해한
                # 클릭이라 금지어가 필요 없다 - 확인은 required 가 한다.
                forbidden=(),
            ),
            FlowStep(
                TargetConfig(
                    key="worksheet_file_exit",
                    description=(
                        "the 'Exit' item in the opened File dropdown menu of the Work "
                        "Sheet window. Use the first letter 'E' as the anchor, then "
                        "click safely inside the Exit menu item."
                    ),
                ),
                required=(("exit",), ("종료",)),
                # 같은 이유로 비운다 - File 드롭다운에는 Save/Print/Export 가 당연히
                # 함께 있어서, 금지어로 두면 Exit 를 읽어 놓고도 막힌다. 대신 문구를
                # 확정하는 진단 실행에서는 `DEMO_RCS_CONFIRM=strict` 가 'exit' 를
                # 실제로 읽었을 때만 누르게 한다.
                forbidden=(),
                # Exit 는 File 이 드롭다운을 열어야만 존재한다 - File 이 실패하면
                # 열리지도 않은 메뉴 자리를 누르게 되므로 건너뛴다.
                requires_previous=True,
            ),
        ],
    )

    # Optics 흐름은 MCD019 기본 배정에서 memo_print 로 교체됐지만 **등록은 유지한다**.
    # 이 설명문의 좌표는 오피스에서 실측 검증된 것이고(2026-08-19, 커서가 Optics 버튼에
    # 정확히 도달), Mac 에서는 그 화면을 볼 수 없어 지우면 되살릴 방법이 없다.
    # `DEMO_RCS_FLOWS="MCD019=optics"` 로 그대로 다시 고를 수 있다.
    optics_flow = InToolFlow(
        name=FLOW_OPTICS,
        opener=FlowStep(
            TargetConfig(
                key="optics_button",
                description=(
                    "the 'Optics...' button in the Remote Monitoring window's button "
                    "area, located directly above the 'PM' button. Use the first letter "
                    "'O' as the anchor, then click safely inside the Optics button area."
                ),
            ),
            required=(("optics",),),
            forbidden=("cancel", "stop", "terminat", "취소"),
        ),
        steps=[
            FlowStep(
                TargetConfig(
                    key="optics_memory_tab",
                    description=(
                        "the 'Memory' tab in the tab strip of the Optics window. Use the "
                        "first letter 'M' as the anchor, then click safely inside the "
                        "Memory tab area."
                    ),
                ),
                required=(("memory",), ("메모리",)),
                forbidden=("cancel", "취소"),
            ),
            FlowStep(
                TargetConfig(
                    key="optics_close_button",
                    description=(
                        "the 'Close' button of the Optics window. Use the first letter "
                        "'C' as the anchor, then click safely inside the Close button area."
                    ),
                ),
                required=(("close",), ("닫기",)),
                forbidden=("cancel", "terminat", "logout", "abort", "취소", "종료"),
                # Optics 의 Close 는 대화상자의 상시 버튼이라 Memory 와 무관하게 누른다.
                requires_previous=False,
            ),
        ],
    )

    return {
        FLOW_MEMO_PRINT: memo_print_flow,
        FLOW_OPTICS: optics_flow,
        FLOW_WORKSHEET: worksheet_flow,
    }


def _build_action_fn(
    settings: Workflow3Settings,
    settle_sec: float,
    *,
    flow_map: dict,
    default_flow: str,
    confirm_policy: str,
    attempts: int,
    pre_click_settle_sec: float,
    click_hold_sec: float,
    char_type_delay_sec: float,
    reveal_enabled: bool,
    reveal_attempts: int,
    reveal_x_ratio: float,
    reveal_y_ratio: float,
    alt_settle_sec: float,
    shift_settle_sec: float,
    caps_settle_sec: float,
    shift_mode: str,
    post_type_wait_sec: float,
    memo_text: str,
    tag: str,
):
    """장비별 창 안 조작 협력자 (VLM 좌표 + OCR 확인 + 클릭).

    `share_request` 의 주입점과 같은 모양이라 그 배선을 그대로 옮겨 쓴다. 확인 실패 시
    crop 과 OCR 원문이 `debug_images/demo_rcs_flow/<tag>/` 에 남는다 - Mac 에서는 이
    화면을 볼 수 없어, 오피스 실행이 실제 문구(required 토큰)를 아는 유일한 경로다.
    """
    from poc.workflow_3 import DEBUG_IMAGE_DIR
    from poc.workflow_3.util.image_utils import capture_window
    from poc.workflow_3.util.mouse_utils import click_at_screen, move_cursor_to_screen
    from poc.workflow_3.util.window_utils import (
        foreground_window,
        image_point_to_screen,
    )
    from poc.workflow_3.vlm.label_verify import (
        crop_box_around_point,
        read_text_near_point,
        tokens_from_text,
    )
    from poc.workflow_3.vlm.ui_venus_mai_locator import analyze_window_target

    debug_dir = DEBUG_IMAGE_DIR / "demo_rcs_flow" / tag
    flows = build_flows(memo_text)

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

    def _remote_click(window, screen, key, *, press_fn=None, release_fn=None):
        """스크린 좌표를 실제로 누른다 - 순서는 `perform_remote_click` 이 고정한다."""

        def _foreground(target_window):
            return foreground_window(target_window, debug_label=f"demo_{key}")

        perform_remote_click(
            window, screen, key,
            foreground_fn=_foreground if callable(foreground_window) else None,
            move_fn=lambda pt, k: move_cursor_to_screen(
                pt, f"demo_{k}", action_enabled=settings.action_enabled,
            ),
            click_fn=lambda pt, k: click_at_screen(
                pt, f"demo_{k}", action_enabled=settings.action_enabled,
                hold_sec=click_hold_sec,
            ),
            sleep_fn=time.sleep,
            settle_sec=max(0.0, pre_click_settle_sec),
            press_modifier_fn=press_fn,
            release_modifier_fn=release_fn,
            modifier_settle_sec=alt_settle_sec,
        )

    def _click(window, image, point, key):
        """이미지 픽셀 좌표를 스크린 좌표로 변환해 클릭한다."""
        screen = image_point_to_screen(window, point, image_size=image.size)
        if screen is None:
            raise RuntimeError(f"좌표 변환 실패: {key} point={point}")
        print(
            f"[INFO] 클릭: {key} px={point} -> screen={screen}"
            f"{'' if settings.action_enabled else ' [dry-run]'}"
        )
        _remote_click(window, screen, key)

    def _reveal(window, image, round_index):
        """여는 버튼을 덮은 창을 Alt+click 으로 뒤로 밀어낸다.

        커서를 그 자리로 옮긴 뒤 Alt 를 잡는다(순서는 `perform_remote_click`).
        `click_at_screen` 이 누르기 전에 커서를 ±`ALIGN_FAIL_CURSOR_JIGGLE_PX`(3) 만큼
        흔드는데(원격이 커서 위치를 등록하게 하는 이 저장소의 규약) 그 흔들림만은 Alt
        를 쥔 상태로 일어난다 - 버튼은 안 눌린 상태이므로 창 끌기(Alt+drag)가 되지는
        않는다. 원격이 그마저 제스처로 읽는다면 `ALIGN_FAIL_CURSOR_JIGGLE_PX=0`.

        예외를 올리지 않고 True/False 로 답한다 - 실패는 "가려서 못 찾음" 이라는
        진단으로 남아야 하고, 그 때문에 시연의 나머지가 죽으면 안 된다.
        """
        point = covering_window_point(
            image.width, image.height,
            x_ratio=reveal_x_ratio, y_ratio=reveal_y_ratio,
        )
        screen = image_point_to_screen(window, point, image_size=image.size)
        if screen is None:
            print(f"[WARNING] 가림 해제 좌표 변환 실패 - Alt+click 생략: px={point}")
            return False

        press, release = alt_hold_hooks(action_enabled=settings.action_enabled)
        print(
            f"[INFO] Alt+click 으로 가린 창 밀어내기({round_index}회): "
            f"px={point} -> screen={screen} "
            f"(비율 x={reveal_x_ratio}, y={reveal_y_ratio})"
            f"{'' if settings.action_enabled else ' [dry-run]'}"
        )
        try:
            _remote_click(
                window, screen, "reveal_opener", press_fn=press, release_fn=release,
            )
        except Exception as exc:
            print(f"[WARNING] 가림 해제 실패: {type(exc).__name__}: {exc}")
            return False
        time.sleep(max(0.0, settle_sec))  # 창이 내려가고 다시 그려질 시간.
        return True

    def _action(tool_id, tool_window, tool_title, tool_backend):
        flow_name = resolve_flow_name(tool_id, flow_map, default_flow)
        print(f"[INFO] {tool_id} 창 안 조작 흐름: {flow_name}")
        return run_in_tool_flow(
            tool_window, tool_title, tool_backend, flows[flow_name],
            capture_fn=capture_window,
            locate_fn=_locate,
            read_tokens_fn=_read_tokens,
            click_fn=_click,
            type_fn=lambda text, key: type_multiline_text(
                text,
                key,
                # 클릭과 별개로 타이핑만 끄는 스위치를 남긴다
                # (`ACTION_LOGIN_TYPING_ENABLED=0` - 로그인 타이핑과 같은 게이트).
                action_enabled=settings.action_enabled and settings.typing_enabled,
                sleep_fn=time.sleep,
                char_delay_sec=char_type_delay_sec,
                shift_settle_sec=shift_settle_sec,
                caps_settle_sec=caps_settle_sec,
                shift_mode=shift_mode,
                post_dwell_sec=post_type_wait_sec,
            ),
            reveal_fn=_reveal if reveal_enabled else None,
            sleep_fn=time.sleep,
            settle_sec=settle_sec,
            confirm_policy=confirm_policy,
            attempts=attempts,
            reveal_attempts=reveal_attempts,
        )

    return _action


def _build_visit_fn(settings: Workflow3Settings, dwell_sec: float, action_fn=None):
    """장비 1대 [접속 -> 체류 -> 장비별 창 안 조작 -> 닫기] 협력자."""
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
            action_fn=action_fn,
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
    `SAFE_MODE=1 uv run python ...` 로 실행하면 클릭과 키 입력이 전부 막힌다(화면은 안
    움직이고 콘솔에 [DRY-RUN] 만 찍힌다). 이 스크립트는 보정(reposition/OK)을 하지
    않는다.
    """
    os.environ.setdefault("SAFE_MODE", "0")
    live = os.environ.get("SAFE_MODE", "0") == "0"
    print("=" * 70)
    if live:
        print("[WARNING] 시연 모드: 실제 마우스/키보드 조작이 발생합니다 "
              "(탭 클릭 / 휠 / tool 더블클릭 / 메모 입력 / 창 닫기).")
        print("[WARNING] 리허설만 하려면 중단 후 'SAFE_MODE=1' 을 붙여 다시 실행하세요.")
    else:
        print("[INFO] SAFE_MODE=1 - 모든 클릭/키 입력이 차단된 리허설입니다"
              "(화면은 움직이지 않음).")
    print("=" * 70)


def main(settings: Workflow3Settings | None = None) -> DemoRunResult:
    """시연 시나리오를 1회 재생한다."""
    settings = settings or load_workflow3_settings()

    tool_ids = parse_tool_ids(os.environ.get("DEMO_RCS_TOOL_IDS"), DEFAULT_TOOL_IDS)
    dwell_sec = _env_float("DEMO_RCS_DWELL_SEC", DWELL_SEC)
    gap_sec = _env_float("DEMO_RCS_GAP_SEC", GAP_SEC)
    notches = _env_int("DEMO_RCS_SCROLL_NOTCHES", 3)
    pause_sec = _env_float("DEMO_RCS_SCROLL_PAUSE_SEC", SCROLL_PAUSE_SEC)
    repeat = max(1, _env_int("DEMO_RCS_REPEAT", 1))
    view_enabled = _env_flag("DEMO_RCS_VIEW_TAB", True)
    flow_enabled = _env_flag("DEMO_RCS_FLOW", True)
    flow_settle_sec = _env_float("DEMO_RCS_FLOW_SETTLE_SEC", FLOW_SETTLE_SEC)
    flow_attempts = max(1, _env_int("DEMO_RCS_FLOW_ATTEMPTS", 2))
    flow_map = parse_flow_map(os.environ.get("DEMO_RCS_FLOWS"), DEFAULT_TOOL_FLOWS)
    default_flow = os.environ.get("DEMO_RCS_DEFAULT_FLOW", FLOW_MEMO_PRINT).strip().lower()
    confirm_policy = (
        os.environ.get("DEMO_RCS_CONFIRM", DEFAULT_CONFIRM_POLICY).strip().lower()
        or DEFAULT_CONFIRM_POLICY
    )
    pre_click_settle = _env_float("DEMO_RCS_PRE_CLICK_SETTLE_SEC", PRE_CLICK_SETTLE_SEC)
    click_hold_sec = _env_float("DEMO_RCS_CLICK_HOLD_SEC", CLICK_HOLD_SEC)
    char_type_delay_sec = _env_float("DEMO_RCS_CHAR_TYPE_DELAY_SEC", CHAR_TYPE_DELAY_SEC)
    reveal_enabled = _env_flag("DEMO_RCS_REVEAL", True)
    reveal_attempts = max(0, _env_int("DEMO_RCS_REVEAL_ATTEMPTS", 2))
    reveal_x_ratio = _env_float("DEMO_RCS_REVEAL_X_RATIO", DEFAULT_REVEAL_X_RATIO)
    reveal_y_ratio = _env_float("DEMO_RCS_REVEAL_Y_RATIO", DEFAULT_REVEAL_Y_RATIO)
    alt_settle_sec = _env_float("DEMO_RCS_ALT_SETTLE_SEC", ALT_SETTLE_SEC)
    shift_settle_sec = _env_float("DEMO_RCS_SHIFT_SETTLE_SEC", SHIFT_SETTLE_SEC)
    shift_mode = resolve_shift_mode(os.environ.get("DEMO_RCS_SHIFT_MODE"))
    caps_settle_sec = _env_float("DEMO_RCS_CAPS_SETTLE_SEC", CAPS_SETTLE_SEC)
    post_type_wait_sec = _env_float("DEMO_RCS_POST_TYPE_WAIT_SEC", POST_TYPE_WAIT_SEC)
    memo_text = parse_memo_text(os.environ.get("DEMO_RCS_MEMO_TEXT"), DEFAULT_MEMO_TEXT)
    tag = make_timestamp_tag(time.time())

    assigned = ", ".join(
        f"{tool}={resolve_flow_name(tool, flow_map, default_flow)}" for tool in tool_ids
    )
    print(
        f"[INFO] 시연 설정: 체류={dwell_sec:.0f}s, 간격={gap_sec:.0f}s, "
        f"View훑기={'on' if view_enabled else 'off'}(휠 {notches}칸), "
        f"창안조작={'on' if flow_enabled else 'off'}"
        f"(확인={confirm_policy}, 재시도={flow_attempts}, "
        f"클릭전대기={pre_click_settle:.1f}s, 누름유지={click_hold_sec:.2f}s), "
        f"글자간격={char_type_delay_sec:.2f}s(대문자={shift_mode}, "
        f"수정자대기={shift_settle_sec:.2f}s/caps={caps_settle_sec:.2f}s, "
        f"입력후체류={post_type_wait_sec:.1f}s), "
        f"가림해제={'on' if reveal_enabled else 'off'}"
        f"(Alt+click {reveal_attempts}회, 지점 x={reveal_x_ratio:.2f}/y={reveal_y_ratio:.2f}, "
        f"Alt대기={alt_settle_sec:.2f}s), "
        f"반복={repeat}회"
    )
    print(f"[INFO] 장비별 조작 흐름: {assigned or '-'}")

    try:
        preflight_fn = _build_preflight_fn(settings)
        view_fn = (
            _build_view_fn(settings, notches, pause_sec)
            if view_enabled
            else (lambda w, t, b: STATUS_VIEW_SKIPPED)
        )
        list_tab_fn = _build_list_tab_fn(settings)
        action_fn = (
            _build_action_fn(
                settings, flow_settle_sec,
                flow_map=flow_map,
                default_flow=default_flow,
                confirm_policy=confirm_policy,
                attempts=flow_attempts,
                pre_click_settle_sec=pre_click_settle,
                click_hold_sec=click_hold_sec,
                char_type_delay_sec=char_type_delay_sec,
                reveal_enabled=reveal_enabled,
                reveal_attempts=reveal_attempts,
                reveal_x_ratio=reveal_x_ratio,
                reveal_y_ratio=reveal_y_ratio,
                alt_settle_sec=alt_settle_sec,
                shift_settle_sec=shift_settle_sec,
                caps_settle_sec=caps_settle_sec,
                shift_mode=shift_mode,
                post_type_wait_sec=post_type_wait_sec,
                memo_text=memo_text,
                tag=tag,
            )
            if flow_enabled
            else None
        )
        visit_fn = _build_visit_fn(settings, dwell_sec, action_fn)
    except Exception as exc:
        # 두 원인이 섞이는 자리다. Mac 은 pywinauto 부재로 걸리지만(정상), 오피스는
        # 의존성이 있으므로 여기서 걸렸다면 **우리 코드의 결함**이다 - 예전에 이 자리가
        # 무조건 "Windows 전용 의존성 없음" 이라고 찍어, 엉뚱한 모듈에서 이름을 가져온
        # 버그를 환경 탓으로 읽게 만들었다. 그래서 둘을 갈라 찍고 traceback 을 남긴다.
        import traceback

        missing_dep = isinstance(exc, ModuleNotFoundError)
        if missing_dep:
            print(f"[ERROR] Windows 전용 모듈 없음(개발 PC 에서는 정상): {exc}")
        else:
            print(
                f"[ERROR] 시연 배선 조립 실패 - 코드 결함일 가능성이 높습니다: "
                f"{type(exc).__name__}: {exc}"
            )
        traceback.print_exc()
        return DemoRunResult(
            aborted="rcs_modules_unavailable" if missing_dep else "wiring_error"
        )

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
    "DEFAULT_MEMO_TEXT",
    "DEFAULT_TOOL_FLOWS",
    "DEFAULT_TOOL_IDS",
    "FLOW_OK",
    "FLOW_OPENER_FAILED",
    "FLOW_MEMO_PRINT",
    "FLOW_OPTICS",
    "FLOW_OPENER_NOT_VISIBLE",
    "FLOW_SKIPPED",
    "FLOW_WINDOW_NOT_FOUND",
    "FLOW_WORKSHEET",
    "KNOWN_FLOW_NAMES",
    "STATUS_CONNECTED",
    "STATUS_CONNECT_FAILED",
    "STATUS_ERROR",
    "STATUS_VIEW_OK",
    "STATUS_VIEW_SKIPPED",
    "STATUS_VIEW_TAB_FAILED",
    "STATUS_WINDOW_NOT_FOUND",
    "DemoRunResult",
    "FlowStep",
    "InToolFlow",
    "ToolVisit",
    "alt_hold_hooks",
    "browse_view_tab",
    "build_flows",
    "covering_window_point",
    "main",
    "parse_flow_map",
    "parse_memo_text",
    "parse_tool_ids",
    "perform_remote_click",
    "resolve_flow_name",
    "run_demonstration",
    "run_in_tool_flow",
    "resolve_shift_mode",
    "shift_plan",
    "shift_symbols",
    "type_multiline_text",
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
