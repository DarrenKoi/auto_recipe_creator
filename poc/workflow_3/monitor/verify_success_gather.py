"""success gather 1회 검증 — office 에서 실제 DB stage 가 계약대로 되는지 확인.

office PC 전용 검증 스크립트(success downloader 가 없으면 즉시 종료).

**부작용 있음**: 실제로 S 이미지를 내려받아 consensus 캐시(events/)에 stage 한다.
읽기 전용 점검이 필요하면 align/diagnostics/verify_rcp_assets.py 를 쓸 것.

확인 항목: reason=ok / event 디렉토리별 S*.jpeg + 숨김폴더 cond(.<이미지명>/cond.txt) 짝 /
cond 파싱(crosshair 좌표, modality 추론 — msr cond 는 Scope 가 없어 msr_modality() 키로
가른다) / modality 별 장수가 consensus_min_s 에 닿는지 / staging 정리. 통과하면 events/ 는
그대로 캐시로 남는다(추가 정리 불필요).

**임계값에 대하여**: 이 경로에는 "align 실패 몇 건 이상이어야 받는다" 같은 게이트가 없다.
gather 는 *성공(S)* 측정만 받으며 align fail 이력이 0 건인 recipe 도 정상 대상이다.
관련 숫자는 두 개뿐이고 둘 다 하한이 아니다 - `MAX_EVENTS`(최근 성공 측정 몇 건까지
받을지, 상한)와 `consensus_min_s`(modality 별 template 을 *빌드*할 최소 S 장수. 다운로드가
아니라 빌드 단계 판정이라 미달이면 rcp 폴백일 뿐 다운로드는 그대로 된다).

**다운로더 예외 진단**: gather_success_images 는 예외를 삼켜 `reason="error:<Type>: <msg>"`
로만 보고한다(모니터 루프를 안 죽이려는 설계). 그러면 어느 줄에서 터졌는지 알 수 없어,
이 스크립트는 다운로더를 _TracingDownloader 로 감싸 호출 인자와 **전체 traceback** 을
콘솔에 찍는다. traceback 의 마지막 프레임이 office_success_downloader.py 의 실제 원인 줄이다.

사용법 (venv 활성화 후, 저장소 루트에서):
  1) 아래 EQP_ID / CLASS_NAME / RECIPE_NAME 상수를 채운다.
  2) python poc/workflow_3/monitor/verify_success_gather.py

  CLI 인자는 쓰지 않는다. 1회성으로 다른 대상을 볼 때만 동명의 env
  (ALIGN_EQP_ID / ALIGN_CLASS_NAME / ALIGN_RECIPE_NAME)를 붙이면 되고, env 가 상수를
  이긴다 - assets 해석(resolve_assets_auto)과 같은 env 규약이다.
"""

import os
import sys
import traceback
from pathlib import Path

# venv 에서 파일 경로로 직접 실행할 때 저장소 루트를 sys.path 에 얹는다.
# (monitor/manual_align_correction.py 와 같은 규약)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poc.workflow_3.monitor.success_gather import (  # noqa: E402
    DOWNLOADER_AVAILABLE,
    _DOWNLOADER,
)
from poc.workflow_3.align.cond_file import load_cond, msr_modality  # noqa: E402
from poc.workflow_3.align.consensus_gather import gather_success_images  # noqa: E402
from poc.workflow_3.config import load_workflow3_settings  # noqa: E402

# ===========================================================================
# 검증 대상 - 여기만 고쳐서 쓴다 (동명의 ALIGN_* env 가 이 상수를 이긴다).
# CLASS_NAME 과 RECIPE_NAME 은 따로 준다 - 스크립트가 '<class>/<recipe>' 로 합친다.
# ===========================================================================

EQP_ID = "MCD513"
CLASS_NAME = "RJ1BXXX"
RECIPE_NAME = "RJ1B_ISOLINERPOLY_R1"

# 최근 성공(S) 측정을 몇 건까지 받을지 - **상한이지 하한이 아니다**. 측정 1건이 OM 2장 +
# SEM 3장이라 4건이면 OM 8 / SEM 12 장이 되어 consensus_min_s(4) 를 넉넉히 넘는다.
# 이 recipe 의 측정 이력이 적어 modality 별 장수가 모자랄 때만 올린다(예: 8).
# env ALIGN_FAIL_GATHER_MAX_EVENTS 가 이 상수를 이긴다.
MAX_EVENTS = 4

# 캐시 신선도 TTL(초). 0 = 항상 새로 받는다(검증 스크립트 기본값). 프로덕션 루프는
# settings.consensus_refresh_ttl_sec(6h)를 쓴다 - 여기서 흉내내면 "reason=fresh" 로
# 빠져 실제 다운로드를 검증하지 못한다.
REFRESH_TTL_SEC = 0


class _TracingDownloader:
    """실 다운로더를 감싸 호출 인자와 예외 traceback 을 콘솔에 남기는 진단용 래퍼.

    gather_success_images 가 예외를 삼켜 타입/메시지만 남기기 때문에(모니터 루프를
    지키려는 의도적 설계) 검증 실행에서는 여기서 traceback 을 먼저 찍고 그대로
    re-raise 한다 - gather 의 정리/보고 경로는 원래대로 돌게 둔다.
    """

    def __init__(self, inner):
        self._inner = inner

    def download_recent_successes(self, recipe_id, *, max_events, dest_dir):
        print(f"[INFO] downloader 호출: recipe_id={recipe_id!r} "
              f"max_events={max_events!r}({type(max_events).__name__}) "
              f"dest_dir={str(dest_dir)!r}")
        try:
            staged = self._inner.download_recent_successes(
                recipe_id, max_events=max_events, dest_dir=dest_dir
            )
        except Exception as exc:
            print(f"[ERROR] downloader 예외: {type(exc).__name__}: {exc}")
            # stderr 가 아니라 stdout 으로 찍는다 - 오피스 콘솔을 통째로 복사해 올 때
            # [ERROR] 마커 사이에 순서대로 들어가야 어느 호출의 traceback 인지 알 수 있다.
            print("[ERROR] --- traceback (마지막 프레임이 실제 원인 줄) ---")
            print(traceback.format_exc())
            print("[ERROR] --- traceback 끝 ---")
            if isinstance(exc, TypeError) and "/" in str(exc):
                print("[ERROR] 힌트: Path 를 float 로 나눈 자리다 - office_success_downloader "
                      "안에서 dest_dir/<DB 값> 처럼 경로를 만들 때 그 값이 문자열이 아니라 "
                      "float(예: 결측 컬럼의 NaN, 숫자형 lot/event id)로 온 경우다. "
                      "해당 줄에서 값을 str() 로 감싸고, NaN/None 이면 그 event 를 건너뛰도록 고칠 것.")
            raise
        print(f"[INFO] downloader 반환: {type(staged).__name__} "
              f"len={len(staged) if staged is not None else 'None'}")
        return staged


def _print_threshold_banner(settings):
    """이 경로에 어떤 임계값이 있고 없는지 콘솔에 못박는다(오해 방지)."""
    print("[INFO] --- 임계값 점검 ---")
    print(f"[INFO]   MAX_EVENTS={MAX_EVENTS} (최근 성공 측정 건수 *상한*. 하한 아님)")
    print(f"[INFO]   REFRESH_TTL_SEC={REFRESH_TTL_SEC} (0 = 캐시 무시하고 항상 새로 받음)")
    print(f"[INFO]   consensus_min_s={settings.consensus_min_s} "
          f"(modality 별 template *빌드* 최소 S 장수, floor 3. 다운로드 게이트가 아니다)")
    print("[INFO]   align 실패율/실패건수 임계값: 없음. gather 는 성공(S) 측정만 받으므로 "
          "align fail 이력 0 건인 recipe 도 정상 대상이다.")


def main():
    if not DOWNLOADER_AVAILABLE:
        print("[ERROR] success downloader 없음 - office PC 에서 실행하세요.")
        return 1

    eqp_id = (os.environ.get("ALIGN_EQP_ID", "").strip() or EQP_ID).strip()
    class_name = (os.environ.get("ALIGN_CLASS_NAME", "").strip() or CLASS_NAME).strip()
    recipe_name = (os.environ.get("ALIGN_RECIPE_NAME", "").strip() or RECIPE_NAME).strip()
    if not eqp_id or not class_name or not recipe_name:
        print("[ERROR] 검증 대상이 비었습니다. 이 파일 상단의 EQP_ID / CLASS_NAME / "
              "RECIPE_NAME 상수를 채우거나, 동명의 ALIGN_* env 를 지정하세요.")
        print("  예: EQP_ID = \"MCD513\" / CLASS_NAME = \"RJ1BXXX\" / "
              "RECIPE_NAME = \"RJ1B_ISOLINERPOLY_R1\"")
        return 1
    recipe_id = f"{class_name}/{recipe_name}"

    settings = load_workflow3_settings()
    # env 가 상수를 이기는 규약(파일 상단 상수 > 코드 기본값, 실제 셸 env 가 최우선).
    max_events = int(os.environ.get("ALIGN_FAIL_GATHER_MAX_EVENTS", "").strip() or MAX_EVENTS)

    print(f"[INFO] gather 1회 실행: EQP_ID={eqp_id} recipe={recipe_id}")
    _print_threshold_banner(settings)

    res = gather_success_images(
        eqp_id, recipe_id,
        downloader=_TracingDownloader(_DOWNLOADER),
        max_events=max_events,
        refresh_ttl_sec=REFRESH_TTL_SEC,
    )
    print(f"[INFO] reason={res.reason} events={res.n_events} images={res.n_images}")
    print(f"[INFO] events_dir={res.events_dir}")
    if res.reason != "ok":
        print("[ERROR] stage 실패 - reason 으로 downloader/DB 를 점검하세요. "
              "(empty=반환 리스트 비었음, error:=다운로드 예외, error:swap:=반환값/swap 문제)")
        if res.reason.startswith("error:"):
            print("[ERROR] 위의 traceback 마지막 프레임이 office_success_downloader.py 의 "
                  "고칠 줄이다. 이 경로에는 임계값 거부가 없으므로 reason=error 는 "
                  "'자격 미달'이 아니라 '코드가 터졌다'는 뜻이다.")
        return 1

    problems = 0
    modality_counts = {}
    for ev_dir in sorted(p for p in res.events_dir.iterdir() if p.is_dir()):
        images = sorted(ev_dir.glob("S*.jpeg")) + sorted(ev_dir.glob("S*.jpg"))
        pair_ok = len(images) > 0
        if not pair_ok:
            problems += 1
        print(f"[{'INFO' if pair_ok else 'ERROR'}] {ev_dir.name}: images={len(images)}")
        for img in images:
            cond = load_cond(img)  # .<이미지명>/cond.txt 를 해석.
            if cond is None:
                problems += 1
                print(f"[ERROR]   {img.name}: cond 없음 (.{img.name}/cond.txt 미존재)")
                continue
            modality = cond.scope or msr_modality(cond)
            cond_ok = cond.crosshair_xy is not None and modality is not None
            if not cond_ok:
                problems += 1
            if modality:
                key = modality.strip().lower()
                modality_counts[key] = modality_counts.get(key, 0) + 1
            print(f"[{'INFO' if cond_ok else 'ERROR'}]   {img.name}: "
                  f"crosshair={cond.crosshair_xy} modality={modality}")

    # modality 별 장수 vs consensus_min_s - "받긴 받았는데 consensus 가 안 잡히는" 경우를
    # 여기서 분간한다(그건 다운로드 실패가 아니라 빌드 미달이고, 레버는 MAX_EVENTS 상향).
    min_s = settings.consensus_min_s
    print("[INFO] --- modality 별 S 장수 (consensus build 판정) ---")
    if not modality_counts:
        print(f"[WARNING]   modality 판별된 이미지 없음 - consensus build 불가(min_s={min_s})")
    for key in sorted(modality_counts):
        n = modality_counts[key]
        ok = n >= min_s
        print(f"[{'INFO' if ok else 'WARNING'}]   {key.upper()}={n} "
              f"({'>=' if ok else '<'} min_s={min_s}) "
              f"{'consensus build 가능' if ok else 'consensus 미달 -> rcp 폴백. MAX_EVENTS 를 올려 재실행'}")

    staging = res.events_dir.parent / ".events_staging"
    staging_clean = not staging.exists()
    if not staging_clean:
        problems += 1
    print(f"[{'INFO' if staging_clean else 'ERROR'}] staging 정리됨: {staging_clean}")

    if problems:
        print(f"[ERROR] 검증 실패 {problems}건")
        return 1
    print("[INFO] 검증 통과 - events/ 캐시가 계약대로 stage 되었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
