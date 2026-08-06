# poc/workflow_3/vlm/test_locator_combo.py
"""VLM_LOCATOR_COMBO 해석 스모크 테스트 - RCS/VLM 없이 Mac 에서 돈다.

왜 이 테스트가 필요한가: 조합 전환은 로그인/List 탭/tool 선택/PM 버튼을 한 번에
바꾸는 광역 스위치라, 파싱이 조용히 실패해 기본값으로 되돌아가면 "바꿨는데 왜
그대로지?" 를 오피스에서 디버깅하게 된다. 깨진 입력이 기본값으로 안전하게
떨어지는 것과, 정상 입력이 실제로 반영되는 것 둘 다 못 박는다.

실행: uv run python poc/workflow_3/vlm/test_locator_combo.py
"""

import os

from poc.workflow_3.vlm import ui_venus_mai_locator as loc


def _resolve(raw):
    """raw(None=미설정)를 env 에 넣고 조합을 해석한다."""
    if raw is None:
        os.environ.pop(loc.LOCATOR_COMBO_ENV, None)
    else:
        os.environ[loc.LOCATOR_COMBO_ENV] = raw
    loc._announced_combo = None   # 1회 출력 가드 초기화(테스트 간 독립성).
    return loc.resolve_locator_services()


def _check(label, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def main():
    ok = True
    default = (loc.DEFAULT_COARSE_SERVICE, loc.DEFAULT_REFINE_SERVICE)

    ok &= _check("미설정 -> production 기본(ui-venus>mai-ui)",
                 _resolve(None) == ("ui-venus", "mai-ui"))
    ok &= _check("빈 문자열 -> 기본", _resolve("") == default)
    ok &= _check("공백만 -> 기본", _resolve("   ") == default)

    ok &= _check("mai-ui>mai-ui -> 양쪽 mai-ui",
                 _resolve("mai-ui>mai-ui") == ("mai-ui", "mai-ui"))
    ok &= _check("주변 공백 허용", _resolve(" mai-ui > mai-ui ") == ("mai-ui", "mai-ui"))
    ok &= _check("비대칭 조합 그대로 반영",
                 _resolve("mai-ui>ui-venus") == ("mai-ui", "ui-venus"))

    # 깨진 입력은 죽지 않고 기본값으로 - 로케이터가 못 뜨면 로그인부터 막힌다.
    ok &= _check("'>' 없음 -> 기본", _resolve("mai-ui") == default)
    ok &= _check("좌측 비어있음 -> 기본", _resolve(">mai-ui") == default)
    ok &= _check("우측 비어있음 -> 기본", _resolve("mai-ui>") == default)

    # 호출 시점 read: import 후에 주입해도 반영되어야 한다(rcs 단독 스크립트 경로).
    _resolve(None)
    os.environ[loc.LOCATOR_COMBO_ENV] = "mai-ui>mai-ui"
    ok &= _check("import 이후 주입도 반영(호출 시점 read)",
                 loc.resolve_locator_services() == ("mai-ui", "mai-ui"))
    os.environ.pop(loc.LOCATOR_COMBO_ENV, None)

    # --- 시작 로그 문구: 실제 적용값을 보여줘야 한다 ---
    ok &= _check("describe: 빈 값 -> 기본 표기",
                 loc.describe_locator_combo("") == "ui-venus>mai-ui (기본)")
    ok &= _check("describe: 설정값 그대로",
                 loc.describe_locator_combo("mai-ui>mai-ui") == "mai-ui>mai-ui")
    # 깨진 값을 그대로 찍으면 "설정대로 돌고 있다" 고 오독한다 - 적용값을 보여줘야 한다.
    ok &= _check("describe: 깨진 값 -> 실제 적용될 기본 조합",
                 loc.describe_locator_combo("mai-ui") == "ui-venus>mai-ui")

    # --- config.py 미러링: settings 필드가 같은 env 를 읽는가 ---
    from poc.workflow_3.config import load_workflow3_settings
    os.environ.pop(loc.LOCATOR_COMBO_ENV, None)
    ok &= _check("settings.locator_combo 기본은 빈 문자열",
                 load_workflow3_settings().locator_combo == "")
    os.environ[loc.LOCATOR_COMBO_ENV] = "mai-ui>mai-ui"
    ok &= _check("settings.locator_combo 가 env 를 반영",
                 load_workflow3_settings().locator_combo == "mai-ui>mai-ui")
    os.environ.pop(loc.LOCATOR_COMBO_ENV, None)

    # --- workflow_3_config.py -> env 브리지 ---
    import types
    from poc.workflow_3 import workflow_3_config_loader as cfg_loader
    saved_cfg = cfg_loader._cfg
    try:
        cfg_loader._cfg = types.SimpleNamespace(LOCATOR_COMBO="mai-ui>mai-ui")
        cfg_loader.seed_env()
        ok &= _check("workflow_3_config.LOCATOR_COMBO -> VLM_LOCATOR_COMBO 브리지",
                     os.environ.get(loc.LOCATOR_COMBO_ENV) == "mai-ui>mai-ui")
    finally:
        cfg_loader._cfg = saved_cfg
        os.environ.pop(loc.LOCATOR_COMBO_ENV, None)

    print(f"\n[{'PASS' if ok else 'FAIL'}] test_locator_combo")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
