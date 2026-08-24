"""rcp 등록 align key(IMAP0001/IMAP0002) 부재 진단 - 오피스 전용, read-only.

증상: 보정/판정이 `[WARNING] recipe_om 이미지를 찾지 못했습니다` +
`[ERROR] 등록 OM/SEM 이미지가 없습니다: <recipe_dir>` 로 끝난다(= no_assets).

그 메시지는 "align_img_from_rcp 안에 IMAP0001/IMAP0002 가 없다"는 사실만 말하고
**왜** 없는지는 말하지 않는다. 원인 후보가 서로 완전히 다른 조치를 요구하는데
콘솔 한 줄로는 구분되지 않는 것이 문제다:

  (1) 코드가 읽는 루트와 MES/다운로더가 쓰는 루트가 다르다 (ALIGN_IMAGES_DIR 불일치)
  (2) office_rcp_msr_downloader 어댑터가 없어 다운로드 자체가 안 일어난다
  (3) 어댑터는 있는데 서명이 어긋나 호출이 통째로 실패한다 (gitignore 라 git 으로
      갱신되지 않는 사본 - 과거 include_msr 로 실제 발생)
  (4) 게이트가 꺼져 있다 (ALIGN_FAIL_GATHER_RCP_MSR=0) - 이 경로는 **아무 메시지도
      찍지 않고** 조용히 skip 되므로 콘솔만 봐서는 영영 모른다
  (5) 다운로드는 됐는데 파일명이 IMAP0001/IMAP0002 stem 이 아니다
  (6) class/recipe 폴더명이 알람의 RECIPE_ID 와 다르다 (경로는 맞는데 leaf 가 없음)
  (7) 그 recipe 에 애초에 등록 align key 가 없다 (코드 문제 아님)

이 모듈은 위 7가지를 한 번에 갈라 찍는다. **부작용 0** - RCS 를 만지지 않고,
다운로드하지 않고, 아무 파일도 쓰지 않는다. 마우스도 움직이지 않으므로 장비가
돌고 있어도 안전하다.

실행 (venv 활성화 후, 저장소 루트에서):
  python poc/workflow_3/align/diagnostics/verify_rcp_assets.py

대상 지정: 아래 EQP_ID/RECIPE_ID 상수를 고치거나, env 로 덮는다
(ALIGN_EQP_ID / ALIGN_RECIPE_NAME - env 가 상수를 이긴다).

끝에 [DIGEST] 한 줄이 나온다. 오피스 화면을 Mac 으로 못 가져오므로 그 한 줄이
피드백 채널이다 - 그대로 복사해 주면 된다.
"""

import inspect
import os
import sys
from pathlib import Path

# venv 에서 파일 경로로 직접 실행할 때 저장소 루트를 sys.path 에 얹는다.
# (monitor/manual_align_correction.py, monitor/test_cycle_report.py 와 같은 규약)
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3 import ALIGN_IMAGES_DIR, WORKFLOW_3_DIR  # noqa: E402
from poc.workflow_3.align import (  # noqa: E402
    FROM_RCP_DIRNAME,
    RCP_OM_STEM,
    RCP_SEM_STEM,
)
from poc.workflow_3.align.assets import SUPPORTED_EXTS, _find_by_stem  # noqa: E402

# ===========================================================================
# 진단 대상 - 여기만 고쳐서 쓴다 (env ALIGN_EQP_ID / ALIGN_RECIPE_NAME 가 이긴다).
# ===========================================================================

EQP_ID = "MCD513"
RECIPE_ID = "RJ1BXXX/RJ1B_ISOLINERPOLY_R1"   # '<class>/<recipe>'

# 진단 출력 상한 - 폴더가 클 때 콘솔이 넘치지 않게 한다.
MAX_LIST = 12

# MES 가 과거에 쓰던 루트. 코드 기본값이 workflow_3 로 옮겨간 뒤에도 MES 출력이
# 따라오지 않았으면 여기에 이미지가 남아 있다(가장 흔한 원인).
LEGACY_ROOT = _REPO_ROOT / "poc" / "workflow_1" / "align_images"


def _resolve_target() -> tuple[str, str]:
    """진단 대상 eqp/recipe 를 env > 파일 상수 순으로 고른다."""
    eqp = (os.environ.get("ALIGN_EQP_ID", "").strip() or EQP_ID).strip()
    recipe = (os.environ.get("ALIGN_RECIPE_NAME", "").strip() or RECIPE_ID).strip()
    return eqp, recipe.replace("\\", "/").strip("/")


def _list_dir(directory: Path) -> list[str]:
    """폴더 내용을 이름만 나열한다(이미지 내용은 읽지 않는다)."""
    if not directory.is_dir():
        return []
    try:
        return sorted(p.name for p in directory.iterdir())
    except OSError as exc:
        print(f"[WARNING] 폴더를 읽지 못했습니다: {directory} ({exc})")
        return []


def _show(names: list[str], label: str) -> None:
    """이름 목록을 상한을 걸어 찍는다."""
    if not names:
        print(f"    {label}: (없음)")
        return
    head = names[:MAX_LIST]
    more = f" ... 외 {len(names) - len(head)}개" if len(names) > len(head) else ""
    print(f"    {label} ({len(names)}개): {', '.join(head)}{more}")


def _check_root() -> None:
    """코드가 rcp 를 읽는 루트와 env override 여부를 찍는다."""
    env_value = os.environ.get("ALIGN_IMAGES_DIR", "").strip()
    print("[1] align_images 루트 (코드가 rcp 를 '읽는' 곳)")
    print(f"    ALIGN_IMAGES_DIR = {ALIGN_IMAGES_DIR}")
    print(f"    env 지정? {'예 (' + env_value + ')' if env_value else '아니오 (기본값 사용)'}")
    print(f"    기본값       = {WORKFLOW_3_DIR / 'align_images'}")
    print(f"    존재?        = {'예' if ALIGN_IMAGES_DIR.is_dir() else '아니오'}")
    if not ALIGN_IMAGES_DIR.is_dir():
        print("[ERROR] 루트 자체가 없습니다. MES 출력 위치를 ALIGN_IMAGES_DIR 로 "
              "가리키거나 MES 를 이 경로에 쓰도록 바꿔야 합니다.")


def _check_leaf(eqp_id: str, recipe_id: str) -> dict:
    """대상 recipe leaf 와 align_img_from_rcp 내용을 갈라 본다."""
    leaf = ALIGN_IMAGES_DIR.joinpath(eqp_id, *recipe_id.split("/"))
    from_rcp = leaf / FROM_RCP_DIRNAME

    print(f"[2] 대상 recipe leaf: {leaf}")
    print(f"    leaf 존재?           = {'예' if leaf.is_dir() else '아니오'}")
    print(f"    {FROM_RCP_DIRNAME} 존재? = {'예' if from_rcp.is_dir() else '아니오'}")

    if leaf.is_dir():
        _show(_list_dir(leaf), "leaf 내용")

    rcp_names = _list_dir(from_rcp)
    _show(rcp_names, f"{FROM_RCP_DIRNAME} 내용")

    om = _find_by_stem(from_rcp, RCP_OM_STEM)
    sem = _find_by_stem(from_rcp, RCP_SEM_STEM)
    print(f"    recipe_om ({RCP_OM_STEM})  = {om.name if om else '못 찾음'}")
    print(f"    recipe_sem ({RCP_SEM_STEM}) = {sem.name if sem else '못 찾음'}")

    # 이미지는 있는데 stem 이 안 맞는 경우(원인 5) 를 갈라낸다.
    images = [n for n in rcp_names if Path(n).suffix.lower() in SUPPORTED_EXTS]
    unmatched = [
        n for n in images
        if not n.upper().startswith(RCP_OM_STEM) and not n.upper().startswith(RCP_SEM_STEM)
    ]
    if unmatched and not (om and sem):
        print(f"[WARNING] stem 불일치 의심: 이미지는 {len(images)}장 있는데 "
              f"IMAP0001/IMAP0002 로 시작하지 않습니다 -> {', '.join(unmatched[:MAX_LIST])}")

    return {
        "leaf": leaf,
        "leaf_exists": leaf.is_dir(),
        "from_rcp_exists": from_rcp.is_dir(),
        "n_images": len(images),
        "om": om is not None,
        "sem": sem is not None,
        "unmatched": unmatched,
    }


def _check_downloader() -> dict:
    """office 다운로더 어댑터 존재/서명을 확인한다 (호출하지는 않는다)."""
    print("[3] office_rcp_msr_downloader 어댑터")
    try:
        from poc.workflow_3.monitor.rcp_msr_gather import (
            RCP_MSR_DOWNLOADER_AVAILABLE,
            _accepts_include_msr,
            _DOWNLOADER,
        )
    except Exception as exc:
        print(f"[ERROR] rcp_msr_gather 를 import 하지 못했습니다: {exc}")
        return {"available": False, "accepts_include_msr": None}

    print(f"    사용 가능? = {'예' if RCP_MSR_DOWNLOADER_AVAILABLE else '아니오'}")
    if not RCP_MSR_DOWNLOADER_AVAILABLE:
        print("[WARNING] 어댑터가 없습니다. 이 경우 rcp 는 office MES 가 align_images "
              "트리에 '직접' 적재해야 하며, 코드는 다운로드를 시도하지 않습니다.")
        return {"available": False, "accepts_include_msr": None}

    accepts = _accepts_include_msr(_DOWNLOADER)
    try:
        sig = str(inspect.signature(_DOWNLOADER.download_rcp_msr))
    except (TypeError, ValueError):
        sig = "(서명을 읽을 수 없음)"
    print(f"    download_rcp_msr{sig}")
    print(f"    include_msr 수용? = {'예' if accepts else '아니오 (구버전 사본)'}")
    if not accepts:
        print("[WARNING] 구버전 어댑터입니다. 호출은 되지만 msr 까지 받아 느려집니다.")
    return {"available": True, "accepts_include_msr": accepts}


def _check_gate() -> dict:
    """gather 게이트/타임아웃을 찍는다 - 꺼져 있으면 조용히 skip 되는 경로다."""
    print("[4] gather 게이트 (꺼져 있으면 아무 메시지 없이 skip 된다)")
    try:
        from poc.workflow_3.config import load_workflow3_settings
        settings = load_workflow3_settings()
    except Exception as exc:
        print(f"[ERROR] settings 로드 실패: {exc}")
        return {"enabled": None, "timeout": None}

    enabled = settings.rcp_msr_gather_enabled
    timeout = settings.rcp_gather_timeout_sec
    print(f"    rcp_msr_gather_enabled = {enabled}  (env ALIGN_FAIL_GATHER_RCP_MSR)")
    print(f"    rcp_gather_timeout_sec = {timeout}  (env ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC)")
    if not enabled:
        print("[WARNING] 게이트가 꺼져 있습니다 - rcp 다운로드를 아예 시도하지 않습니다.")
    return {"enabled": enabled, "timeout": timeout}


def _check_elsewhere(eqp_id: str, recipe_id: str) -> dict:
    """다른 루트/다른 class 폴더에 같은 recipe 가 있는지 수색한다."""
    print("[5] 다른 위치 수색 (경로 불일치/폴더명 불일치 판별)")
    recipe_leaf = recipe_id.split("/")[-1]
    found: list[str] = []

    # (a) 과거 MES 루트에 같은 leaf 가 있는가.
    legacy_leaf = LEGACY_ROOT.joinpath(eqp_id, *recipe_id.split("/"))
    legacy_rcp = legacy_leaf / FROM_RCP_DIRNAME
    print(f"    과거 MES 루트 = {LEGACY_ROOT} (존재={'예' if LEGACY_ROOT.is_dir() else '아니오'})")
    if legacy_rcp.is_dir():
        names = _list_dir(legacy_rcp)
        print(f"[WARNING] 과거 루트에 rcp 가 있습니다! -> {legacy_rcp}")
        _show(names, "    거기 내용")
        found.append(f"legacy:{len(names)}")

    # (b) 현재 루트에서 recipe leaf 이름만으로 수색 (class 폴더명이 다른 경우).
    for label, root in (("current", ALIGN_IMAGES_DIR), ("legacy", LEGACY_ROOT)):
        if not root.is_dir():
            continue
        hits = [p for p in root.glob(f"*/*/{recipe_leaf}") if p.is_dir()]
        for hit in hits[:MAX_LIST]:
            rel = hit.relative_to(root)
            has_rcp = (hit / FROM_RCP_DIRNAME).is_dir()
            print(f"    [{label}] 같은 이름 leaf 발견: {rel} "
                  f"({FROM_RCP_DIRNAME}={'있음' if has_rcp else '없음'})")
            if has_rcp:
                found.append(f"{label}:{rel}")

    # (c) eqp 폴더 아래 실제 class 목록 (대소문자/오타 판별).
    eqp_dir = ALIGN_IMAGES_DIR / eqp_id
    print(f"    eqp 폴더 = {eqp_dir} (존재={'예' if eqp_dir.is_dir() else '아니오'})")
    if eqp_dir.is_dir():
        _show(_list_dir(eqp_dir), "    실제 class 폴더")
    elif ALIGN_IMAGES_DIR.is_dir():
        _show(_list_dir(ALIGN_IMAGES_DIR), "    루트의 실제 eqp 폴더")

    return {"found": found}


def _check_consensus(eqp_id: str, recipe_id: str) -> dict:
    """consensus 캐시 경로/적재량을 확인한다 (보정의 1차 템플릿 경로).

    중요: consensus 는 rcp 를 **대체하지 않는다**. crop sizing 을 rcp 등록 key 에서
    가져오므로(`consensus_crops.build_center_tpls_for_sizing` -> `build_templates_from_assets`),
    rcp 가 0장이면 consensus 도 0장이 되어 `resolve_templates` 가 {} 를 돌려준다.
    그때 경고조차 찍히지 않는다(빈 dict 는 예외가 아니므로). rcp 가 consensus 의
    전제라는 뜻이며, 이 섹션은 rcp 가 채워진 뒤에야 의미가 있다.
    """
    print("[6] consensus 캐시 (보정의 1차 템플릿, eqp 무관 경로)")
    try:
        from poc.workflow_3 import ALIGN_CONSENSUS_CACHE_DIR
        from poc.workflow_3.align.consensus_gather import (
            _events_dir_for,
            count_staged_events,
        )
        from poc.workflow_3.monitor.success_gather import DOWNLOADER_AVAILABLE
        from poc.workflow_3.config import load_workflow3_settings
        settings = load_workflow3_settings()
    except Exception as exc:
        print(f"[ERROR] consensus 모듈 로드 실패: {exc}")
        return {"events": 0, "images": 0, "downloader": False}

    cache_env = os.environ.get("ALIGN_CONSENSUS_CACHE_DIR", "").strip()
    print(f"    ALIGN_CONSENSUS_CACHE_DIR = {ALIGN_CONSENSUS_CACHE_DIR}")
    print(f"    env 지정? {'예' if cache_env else '아니오 (기본값 사용)'}")

    # 경로는 반드시 단일 출처(_events_dir_for)로 만든다 - 손으로 조립하면 eqp 를
    # 끼워 넣는 실수가 재발한다(ADR 0004: pool 은 eqp 무관).
    events_dir = _events_dir_for(recipe_id, ALIGN_CONSENSUS_CACHE_DIR)
    print(f"    events 경로 = {events_dir}")
    print(f"    (eqp_id 는 경로에 들어가지 않는 것이 정상 - pool 은 장비 무관)")
    print(f"    존재? = {'예' if events_dir.is_dir() else '아니오'}")

    n_events, n_images = count_staged_events(eqp_id, recipe_id,
                                             cache_root=ALIGN_CONSENSUS_CACHE_DIR)
    print(f"    적재된 event={n_events}, S 이미지={n_images}")
    print(f"    consensus_enabled={settings.consensus_enabled}, "
          f"min_s={settings.consensus_min_s}, max_events={settings.gather_max_events}")
    print(f"    success downloader 사용 가능? = {'예' if DOWNLOADER_AVAILABLE else '아니오'}")

    if not DOWNLOADER_AVAILABLE:
        print("[WARNING] office_success_downloader 가 없습니다 - S 이미지가 자동으로 "
              "모이지 않습니다(consensus 는 영구히 rcp 폴백).")
    if n_images and n_images < settings.consensus_min_s:
        print(f"[WARNING] S 이미지가 min_s({settings.consensus_min_s}) 미만이라 "
              "consensus 빌드가 거부되고 rcp 로 폴백합니다.")

    return {"events": n_events, "images": n_images, "downloader": DOWNLOADER_AVAILABLE}


def _verdict(leaf: dict, dl: dict, gate: dict, elsewhere: dict) -> str:
    """가장 가능성 높은 원인 하나를 지목한다 (조치가 서로 다르므로)."""
    if leaf["om"] and leaf["sem"]:
        return "assets_ok"
    if elsewhere["found"]:
        return "wrong_root_or_class"       # 이미지는 있는데 다른 경로에 있다
    if gate["enabled"] is False:
        return "gather_gate_off"
    if not dl["available"]:
        return "no_downloader_adapter"
    if leaf["unmatched"]:
        return "stem_mismatch"
    if not leaf["leaf_exists"]:
        return "leaf_missing"
    if not leaf["from_rcp_exists"]:
        return "from_rcp_missing"
    return "rcp_empty"                     # 폴더는 있는데 비었다


_NEXT_STEP = {
    "assets_ok": "rcp 는 정상이다. no_assets 가 났다면 다른 recipe/eqp 를 본 것이다.",
    "wrong_root_or_class": "이미지가 다른 경로에 있다. env ALIGN_IMAGES_DIR 로 그 루트를 "
                           "가리키거나 MES 출력 위치를 옮길 것.",
    "gather_gate_off": "ALIGN_FAIL_GATHER_RCP_MSR=1 로 켤 것 (workflow_3_config.py 또는 셸 env).",
    "no_downloader_adapter": "office_rcp_msr_downloader.py 를 poc/workflow_3/monitor/ 에 "
                             "두거나, MES 가 align_images 트리에 직접 적재하게 할 것.",
    "stem_mismatch": "파일명이 IMAP0001/IMAP0002 가 아니다. 다운로더의 저장 파일명을 맞출 것.",
    "leaf_missing": "recipe 폴더 자체가 없다. RECIPE_ID 의 '<class>/<recipe>' 표기가 "
                    "MES 폴더명과 같은지 확인할 것.",
    "from_rcp_missing": "leaf 는 있으나 align_img_from_rcp 가 없다. 다운로드가 한 번도 "
                        "성공하지 않았다는 뜻 - 루프 콘솔에서 '다운로드 예외/시간 초과' 를 볼 것.",
    "rcp_empty": "폴더는 있으나 비었다. 그 recipe 에 등록 align key 가 실제로 있는지 "
                 "MES 에서 확인할 것.",
}


def main() -> int:
    """진단 1회 실행. 부작용 없음(읽기 전용)."""
    eqp_id, recipe_id = _resolve_target()
    print("=" * 74)
    print(f"[INFO] rcp 자산 진단 (read-only): EQP_ID={eqp_id} RECIPE_ID={recipe_id}")
    print("=" * 74)

    _check_root()
    print()
    leaf = _check_leaf(eqp_id, recipe_id)
    print()
    dl = _check_downloader()
    print()
    gate = _check_gate()
    print()
    elsewhere = _check_elsewhere(eqp_id, recipe_id)
    print()
    cons = _check_consensus(eqp_id, recipe_id)
    print()

    verdict = _verdict(leaf, dl, gate, elsewhere)
    print("=" * 74)
    print(f"[INFO] 판정: {verdict}")
    print(f"[INFO] 조치: {_NEXT_STEP[verdict]}")
    if not (leaf["om"] or leaf["sem"]):
        print("[INFO] 참고: rcp 가 0장이면 consensus 도 만들어지지 않습니다 "
              "(crop sizing 을 rcp 등록 key 에서 가져오므로). rcp 를 먼저 채울 것.")
    print(
        f"[DIGEST] rcp_verify eqp={eqp_id} recipe={recipe_id} verdict={verdict} "
        f"root_env={'yes' if os.environ.get('ALIGN_IMAGES_DIR', '').strip() else 'no'} "
        f"leaf={int(leaf['leaf_exists'])} from_rcp={int(leaf['from_rcp_exists'])} "
        f"imgs={leaf['n_images']} om={int(leaf['om'])} sem={int(leaf['sem'])} "
        f"downloader={int(bool(dl['available']))} gate={gate['enabled']} "
        f"elsewhere={len(elsewhere['found'])} "
        f"cons_events={cons['events']} cons_imgs={cons['images']} "
        f"cons_dl={int(bool(cons['downloader']))}"
    )
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
