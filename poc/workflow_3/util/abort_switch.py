"""긴급 해제 스위치 - 자동화가 쥔 마우스를 사용자에게 즉시 돌려준다.

자동 보정 구간에는 커서가 수백 ms 마다 다시 움직이므로, 사용자가 손으로 마우스를
움직여도 곧바로 덮어써진다. 물리 입력이 막힌 것은 아니지만 **실질적으로 잠긴다**.
문제가 생겨도 사이클이 끝날 때까지 손을 쓸 수 없다는 뜻이라, 전역 단축키로
actuation 을 즉시 끄는 경로가 필요하다.

Ctrl+C 로는 안 된다. Ctrl+C 는 터미널이 **포그라운드 콘솔 프로세스**에 보내는
신호인데, 자동화 중 포그라운드는 RCS 창이라 파이썬에 도달하지 않는다. 그래서
OS 레벨 키보드 훅(pynput 전역 단축키)이어야 한다.

**BlockInput 은 여기서 풀지 않는다.** Win32 BlockInput 은 그것을 건 스레드만
해제할 수 있는데(window_utils.block_input 참조) 단축키 리스너는 별도 스레드다.
대신 래치만 걸고, 사이클의 기존 teardown(input_unblock, 메인 스레드)이 해제한다.
래치 즉시 actuation 이 멈추므로 커서 다툼은 바로 끝난다.
"""

import threading


DEFAULT_HOTKEY = "<ctrl>+<alt>+q"


class AbortSwitch:
    """한 번 걸리면 풀리지 않는 중단 래치 (스레드 안전)."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""

    def request(self, reason: str = "") -> bool:
        """중단을 래치한다. 이번 호출이 실제로 래치했으면 True.

        사유는 **처음 것만** 남긴다. 나중 호출이 덮으면 "무엇이 중단을 일으켰나"가
        사라진다.
        """
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._event.set()
            return True

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def reset(self) -> None:
        """래치를 푼다. 프로세스 재시작과 같은 의미이며 테스트/재무장에 쓴다."""
        with self._lock:
            self._event.clear()
            self._reason = ""


SWITCH = AbortSwitch()


def request_abort(reason: str = "") -> bool:
    """전역 스위치를 래치한다. 이번 호출이 래치했으면 True."""
    return SWITCH.request(reason)


def is_aborted() -> bool:
    """actuation 을 해도 되는지 판정하는 유일한 질문."""
    return SWITCH.is_set()


def abort_reason() -> str:
    return SWITCH.reason


def _default_listener_factory(mapping):
    """pynput 전역 단축키 리스너. import 는 함수 안에서 한다(개발 PC 부재 허용)."""
    from pynput import keyboard

    return keyboard.GlobalHotKeys(mapping)


def start_abort_hotkey(
    hotkey: str = DEFAULT_HOTKEY,
    *,
    listener_factory=None,
    on_abort=None,
) -> bool:
    """전역 단축키 리스너를 띄운다. 시작했으면 True.

    OS 레벨 키보드 훅이라 RCS 창이 포커스를 쥐고 있어도 눌린다. pynput 이 없거나
    훅 설치가 실패하면 경고만 남기고 False - 단축키가 없다고 루프를 못 띄우면
    본말이 전도된다.

    ``on_abort`` 는 래치 **후** 부른다. 여기서 BlockInput 을 풀지 말 것 - 그것을 건
    스레드만 해제할 수 있는데 이 콜백은 리스너 스레드에서 돈다.
    """
    factory = listener_factory or _default_listener_factory

    def _fire() -> None:
        if request_abort(f"긴급 해제 단축키 {hotkey}"):
            print("=" * 70)
            print(f"[WARNING] 긴급 해제: {hotkey} - 자동 마우스/키보드 출력을 중단합니다.")
            print("[WARNING] 진행 중인 사이클은 teardown(창 닫기/녹화 저장)만 마치고 끝납니다.")
            print("=" * 70)
            if on_abort is not None:
                try:
                    on_abort()
                except Exception as exc:
                    print(f"[WARNING] 긴급 해제 콜백 실패(무시): {exc}")

    try:
        listener = factory({hotkey: _fire})
        listener.daemon = True
        listener.start()
    except Exception as exc:
        print(f"[WARNING] 긴급 해제 단축키 등록 실패({hotkey}) - 단축키 없이 진행: {exc}")
        return False
    print(f"[INFO] 긴급 해제 단축키 준비됨: {hotkey} (누르면 마우스를 즉시 돌려받습니다)")
    return True


__all__ = [
    "AbortSwitch",
    "start_abort_hotkey",
    "DEFAULT_HOTKEY",
    "SWITCH",
    "abort_reason",
    "is_aborted",
    "request_abort",
]
