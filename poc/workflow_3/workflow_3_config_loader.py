# poc/workflow_3/workflow_3_config_loader.py
"""workflow_3 모니터 공용 config 로더 — workflow_3_config.py 상수를 env 로 브리지.

왜 env 브리지인가
-----------------
workflow_3 루프는 동작을 `ALIGN_FAIL_*`/`SAFE_MODE` env 로 제어한다(`load_workflow3_settings`).
이 로더는 실편집 파일 workflow_3_config.py 의 상수를 그 env 이름들로 `setdefault` 해, 사용자가
env 를 직접 만지지 않고 한 파일만 편집하면 되게 한다. 모니터 진입점 맨 위(특히
`load_workflow3_settings()` 호출 전)에서 `seed_env()` 를 부른다.

규약
----
  * 상수 = None              → 해당 env 미설정(코드 기본값 유지).
  * 상수 = 0/1/숫자/비어있지않은 문자열 → str() 로 env setdefault.
  * setdefault 라 **실제 OS env 가 이미 있으면 그게 우선**(워크플로 한 줄 오버라이드 가능, 하위호환).
  * workflow_3_config.py 가 없으면 `_cfg=None` → 전부 skip(드라이버 기본 동작).

idempotent: setdefault 라 여러 번 불려도 안전.
"""

import os

try:
    from poc.workflow_3 import workflow_3_config as _cfg
except ImportError:   # 실편집 파일 부재 — workflow_3_config.example.py 복사해서 생성.
    _cfg = None
except Exception as _exc:   # 손편집 scratch 파일의 SyntaxError/NameError 등 — 모니터를
    # 죽이지 않고 코드 기본값으로 진행한다(파일이 깨졌다고 production 루프가 시작 실패하면 곤란).
    print(f"[WARNING] workflow_3_config.py 로드 실패 - 코드 기본값으로 진행: {_exc}")
    _cfg = None


# (config 상수명, env 이름) 순서 매핑. 새 토글은 여기 한 줄만 추가하면 된다.
# env 이름은 poc/workflow_3/config.py 의 load_workflow3_settings 가 읽는 것과 일치해야 한다.
_CONST_TO_ENV = (
    # [1] foreground takeover
    ("SAFE_MODE", "SAFE_MODE"),
    ("BLOCK_INPUT", "ALIGN_FAIL_BLOCK_INPUT"),
    ("KEEP_AWAKE", "ALIGN_FAIL_KEEP_AWAKE"),
    ("RCS_WINDOW_MAX_TRIALS", "ALIGN_FAIL_RCS_WINDOW_MAX_TRIALS"),
    # [2] 알람 폴링 / 소스
    ("POLL_SEC", "ALIGN_FAIL_POLL_SEC"),
    ("WINDOW_SEC", "ALIGN_FAIL_WINDOW_SEC"),
    ("ALARM_SOURCE", "ALIGN_FAIL_ALARM_SOURCE"),
    ("REPLAY_CSV", "ALIGN_FAIL_REPLAY_CSV"),
    # [3] 알림
    ("POPUP", "ALIGN_FAIL_POPUP"),
    ("RICH_NOTIFY", "ALIGN_FAIL_RICH_NOTIFY"),
    # [4] 점검 모니터 — 보정 가능성 / zoom 탐색
    ("FEASIBILITY_MARK", "ALIGN_FAIL_FEASIBILITY_MARK"),
    ("REPOSITION_PREVIEW", "ALIGN_FAIL_REPOSITION_PREVIEW"),
    ("ZOOM_PROBE", "ALIGN_FAIL_ZOOM_PROBE"),
    ("ZOOM_METHOD", "ALIGN_FAIL_ZOOM_METHOD"),
    ("SEM_BOX_DETECT", "ALIGN_FAIL_SEM_BOX_DETECT"),
    ("PM_DROPDOWN", "ALIGN_FAIL_PM_DROPDOWN"),
    # [5] CV 보정 (production)
    ("CORRECTION", "ALIGN_FAIL_CORRECTION"),
    ("CORRECTION_DRY_RUN", "ALIGN_FAIL_CORRECTION_DRY_RUN"),
    # [6] 과거 데이터 수집 / consensus
    ("GATHER_RCP_MSR", "ALIGN_FAIL_GATHER_RCP_MSR"),
    ("RCP_GATHER_TIMEOUT_SEC", "ALIGN_FAIL_RCP_GATHER_TIMEOUT_SEC"),
    ("GATHER_SUCCESS", "ALIGN_FAIL_GATHER_SUCCESS"),
    ("CONSENSUS", "ALIGN_FAIL_CONSENSUS"),
    ("CONSENSUS_MIN_S", "ALIGN_FAIL_CONSENSUS_MIN_S"),
    # [7] 재시도 정책
    ("FAILURE_COOLDOWN_SEC", "ALIGN_FAIL_FAILURE_COOLDOWN_SEC"),
    # [8] VLM 로케이터 조합 (config.py 가 아니라 vlm/ui_venus_mai_locator.py 가 읽는다)
    ("LOCATOR_COMBO", "VLM_LOCATOR_COMBO"),
)


def seed_env():
    """workflow_3_config 상수 → os.environ(setdefault). 모니터 진입점 맨 위에서 호출.

    None 은 skip(코드 기본값), 빈 문자열도 skip(미설정 의도). 0 은 명시적 off 로 보고 "0" 주입.
    이미 설정된 OS env 는 setdefault 라 보존(우선).
    """
    if _cfg is None:
        return
    seeded = []
    ignored = []
    for const_name, env_name in _CONST_TO_ENV:
        value = getattr(_cfg, const_name, None)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        before = os.environ.get(env_name)
        if before is None:
            os.environ[env_name] = str(value)
            seeded.append(f"{env_name}={os.environ[env_name]}")
        elif before != str(value):
            # 실제 OS env 가 우선(workflow_2 패턴과 동일). 값이 다르면 조용히 무시되지
            # 않게 명시한다 — 예: shell 에 SAFE_MODE=1 이 export 된 채 config 에서 0 으로
            # 바꿔도 안 먹는 흔한 함정을 진단으로 드러낸다.
            ignored.append(f"{env_name}: env={before} 우선(config={value} 무시)")
    if seeded:
        print(f"[INFO] workflow_3_config 적용: {', '.join(seeded)}")
    if ignored:
        print(f"[INFO] workflow_3_config 무시(실제 env 우선): {', '.join(ignored)}")


__all__ = ["seed_env"]
