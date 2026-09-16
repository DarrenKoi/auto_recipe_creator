"""RCS GUI 자동화 진행 로그의 볼륨 스위치.

RCS 경로(실행/로그인/List 탭/tool 선택/창 닫기)는 단계마다 창 탐색·좌표·타이핑·
VLM 응답 전문을 찍는다. 그 경로가 안정된 뒤에는 이 출력이 보정/알람 로그를
스크롤 밖으로 밀어내 **실제로 봐야 할 줄을 가린다**.

계약 둘:
  * `[ERROR]`/`[WARNING]`/`[EXIT]` 은 플래그와 무관하게 항상 통과한다 - 볼륨
    스위치가 실패를 숨기면 안 된다. 진행 로그(`[INFO]`)만 끈다.
  * 기본값은 **출력**(env 미설정). `rcs/` 단독 스크립트는 종전 그대로 보이고,
    모니터 진입점만 상수 블록에서 `RCS_VERBOSE=0` 으로 내린다.
"""

import os

# 볼륨 스위치가 삼키면 안 되는 태그. 실패/종료 사유는 조용해지면 안 된다.
_ALWAYS_TAGS = ("[ERROR]", "[WARNING]", "[EXIT]")


def rcs_verbose() -> bool:
    """RCS 진행 로그를 출력할지. env 미설정이면 True(종전 동작)."""
    raw = os.environ.get("ALIGN_FAIL_RCS_VERBOSE", "1").strip().lower()
    return raw not in ("0", "", "false", "off", "no")


def rcs_print(*args, **kwargs) -> None:
    """RCS 진행 로그 한 줄. quiet 모드면 [INFO] 만 삼킨다."""
    first = args[0] if args else ""
    if isinstance(first, str) and any(tag in first for tag in _ALWAYS_TAGS):
        print(*args, **kwargs)
        return
    if rcs_verbose():
        print(*args, **kwargs)


def demo() -> None:
    """자체 점검 - 태그별 통과/차단 계약."""
    import io
    import contextlib

    def _capture(env_value, text):
        if env_value is None:
            os.environ.pop("ALIGN_FAIL_RCS_VERBOSE", None)
        else:
            os.environ["ALIGN_FAIL_RCS_VERBOSE"] = env_value
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rcs_print(text)
        return buf.getvalue()

    assert _capture(None, "[INFO] a") == "[INFO] a\n", "env 미설정이면 종전대로 출력"
    assert _capture("1", "[INFO] a") == "[INFO] a\n"
    assert _capture("0", "[INFO] a") == "", "quiet 이면 INFO 는 삼킨다"
    assert _capture("0", "[WARNING] a") == "[WARNING] a\n", "경고는 항상 통과"
    assert _capture("0", "[ERROR] a") == "[ERROR] a\n", "오류는 항상 통과"
    assert _capture("0", "[EXIT] 0") == "[EXIT] 0\n", "종료 코드는 항상 통과"
    assert _capture("off", "[INFO] a") == "", "off/false/no 도 quiet"
    os.environ.pop("ALIGN_FAIL_RCS_VERBOSE", None)
    print("[INFO] console.demo OK")


if __name__ == "__main__":
    demo()
