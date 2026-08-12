# poc/workflow_3/vlm/test_locator_combo.py
"""VLM_LOCATOR_COMBO 해석 스모크 테스트 - RCS/VLM 없이 Mac 에서 돈다.

왜 이 테스트가 필요한가: 조합 전환은 로그인/List 탭/tool 선택/PM 버튼을 한 번에
바꾸는 광역 스위치라, 파싱이 조용히 실패해 기본값으로 되돌아가면 "바꿨는데 왜
그대로지?" 를 오피스에서 디버깅하게 된다. 깨진 입력이 기본값으로 안전하게
떨어지는 것과, 정상 입력이 실제로 반영되는 것 둘 다 못 박는다.

실행: uv run python poc/workflow_3/vlm/test_locator_combo.py
"""

import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from poc.workflow_3.vlm import ui_venus_mai_locator as loc


def test_service_artifact_names_use_actual_route_slugs():
    names = loc._artifact_names("assist_panel", "mai-ui", "mai-ui", mode="service")
    assert names["coarse_response"] == "assist_panel_coarse_mai_ui_response.txt"
    assert names["refine_response"] == "assist_panel_refine_mai_ui_response.txt"
    assert names["result_json"] == "assist_panel_locator_result.json"


def test_legacy_artifact_names_remain_default():
    names = loc._artifact_names("assist_panel", "mai-ui", "mai-ui", mode="legacy")
    assert names["coarse_response"] == "assist_panel_ui_venus_response.txt"


class _FakeLocatorClient:
    def __init__(self, service_slug, **_kwargs):
        self.service_slug = service_slug
        self.model_name = service_slug


def _failure_payload(*, artifact_naming, failure_stage):
    """실제 locator failure JSON을 만들되 네트워크 호출만 고립한다."""
    saved = {
        "Workflow1VLMClient": loc.Workflow1VLMClient,
        "_run_ui_venus_coarse_bbox": loc._run_ui_venus_coarse_bbox,
        "_run_mai_ui_refinement": loc._run_mai_ui_refinement,
        "log_work2_event": loc.log_work2_event,
    }
    coarse_result = {
        "response_text": "coarse",
        "token_usage": {},
        "bbox_1000": {"left": 200, "top": 200, "right": 800, "bottom": 800},
        "bbox_pixels": {"left": 20, "top": 16, "right": 80, "bottom": 64},
        "center": {"x": 50, "y": 40},
    }
    try:
        loc.Workflow1VLMClient = _FakeLocatorClient
        loc._run_ui_venus_coarse_bbox = (
            (lambda *_args, **_kwargs: None)
            if failure_stage == "coarse"
            else (lambda *_args, **_kwargs: coarse_result)
        )
        loc._run_mai_ui_refinement = lambda *_args, **_kwargs: {
            "response_text": "refine",
            "token_usage": {},
            "point": None,
        }
        loc.log_work2_event = lambda **_kwargs: None
        with tempfile.TemporaryDirectory() as tmp:
            result = loc.analyze_window_target(
                None,
                "",
                "",
                loc.TargetConfig(key="assist_panel", description="Assist panel"),
                debug_image_dir=Path(tmp),
                log_name="test",
                component_name="test",
                artifact_prefix="assist_panel",
                coarse_service_slug="mai-ui",
                refine_service_slug="mai-ui",
                artifact_naming=artifact_naming,
                image=Image.new("RGB", (100, 80), (240, 240, 240)),
            )
            return json.loads(
                Path(result.artifacts["result_json"]).read_text(encoding="utf-8")
            )
    finally:
        for name, value in saved.items():
            setattr(loc, name, value)


def test_service_failure_payloads_use_neutral_stage_names():
    assert _failure_payload(
        artifact_naming="service", failure_stage="coarse"
    )["failure_stage"] == "coarse"
    assert _failure_payload(
        artifact_naming="service", failure_stage="refine"
    )["failure_stage"] == "refine"


def test_legacy_failure_payloads_keep_stage_names():
    assert _failure_payload(
        artifact_naming="legacy", failure_stage="coarse"
    )["failure_stage"] == "ui_venus"
    assert _failure_payload(
        artifact_naming="legacy", failure_stage="refine"
    )["failure_stage"] == "mai_ui"


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

    test_service_artifact_names_use_actual_route_slugs()
    test_legacy_artifact_names_remain_default()
    test_service_failure_payloads_use_neutral_stage_names()
    test_legacy_failure_payloads_keep_stage_names()

    # 기본값은 코드 상수(DEFAULT_*)가 정한다 - 상수를 바꿔도 이 테스트는 살아있어야
    # 하므로 리터럴이 아니라 상수와 비교한다.
    ok &= _check("미설정 -> 코드 기본 상수", _resolve(None) == default)
    ok &= _check("빈 문자열 -> 기본", _resolve("") == default)
    ok &= _check("공백만 -> 기본", _resolve("   ") == default)

    ok &= _check("mai-ui>mai-ui -> 양쪽 mai-ui",
                 _resolve("mai-ui>mai-ui") == ("mai-ui", "mai-ui"))
    ok &= _check("주변 공백 허용", _resolve(" mai-ui > mai-ui ") == ("mai-ui", "mai-ui"))
    ok &= _check("비대칭 조합 그대로 반영",
                 _resolve("mai-ui>ui-venus") == ("mai-ui", "ui-venus"))
    ok &= _check("옛 production 조합으로 되돌리기",
                 _resolve("ui-venus>mai-ui") == ("ui-venus", "mai-ui"))

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
                 loc.describe_locator_combo("") == f"{default[0]}>{default[1]} (기본)")
    ok &= _check("describe: 설정값 그대로",
                 loc.describe_locator_combo("ui-venus>mai-ui") == "ui-venus>mai-ui")
    # 깨진 값을 그대로 찍으면 "설정대로 돌고 있다" 고 오독한다 - 적용값을 보여줘야 한다.
    ok &= _check("describe: 깨진 값 -> 실제 적용될 기본 조합",
                 loc.describe_locator_combo("mai-ui") == f"{default[0]}>{default[1]}")

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
