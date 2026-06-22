# poc/workflow_2/golden_eval_config_loader.py
"""golden eval 3개 드라이버 공용 config 로더 — golden_eval_config.py 상수를 env 로 브리지.

왜 env 브리지인가
-----------------
combined/consensus/localization 세 드라이버가 같은 설정을 *서로 다른 시점*에 읽는다:
  - ALIGN_GOLDEN_ROOT        : 각 run() 호출 시점
  - ALIGN_ENSEMBLE_LAB_MODE  : gle._matcher_for_eval() 호출 시점
  - CONSENSUS_MIN_S          : golden_consensus_eval_cond 의 *import 시점*(모듈 상수)
드라이버 *맨 위*(특히 gce/glec import 전에)에서 seed_env() 를 부르면, 그 뒤의 모든 env read 가
golden_eval_config.py 값을 본다. setdefault 라 실제 env 가 있으면 그게 우선(하위호환). 사용자는
env 를 직접 만지지 않고 golden_eval_config.py(gitignore) 한 곳만 편집한다.

golden_eval_config.py 가 없으면 기본값(None/""/None)으로 폴백 — env 안 건드리니 드라이버 기본 동작.
"""

import os

try:
    from poc.workflow_2.golden_eval_config import GOLDEN_ROOT, LAB_MODE, MIN_S
    try:
        from poc.workflow_2.golden_eval_config import HISTORY_ROOT
    except ImportError:   # 구버전 config(HISTORY_ROOT 없음) 하위호환.
        HISTORY_ROOT = None
    try:
        from poc.workflow_2.golden_eval_config import CONSENSUS_BOX_CROP
    except ImportError:   # 구버전 config(CONSENSUS_BOX_CROP 없음) 하위호환.
        CONSENSUS_BOX_CROP = None
    try:
        from poc.workflow_2.golden_eval_config import REREGISTER_BOX_SUGGEST, REREGISTER_TOPN
    except ImportError:   # 구버전 config(REREGISTER_* 없음) 하위호환.
        REREGISTER_BOX_SUGGEST = 1
        REREGISTER_TOPN = 0
    try:
        from poc.workflow_2.golden_eval_config import REREGISTER_ACCEPT_MARGIN
    except ImportError:   # 구버전 config(REREGISTER_ACCEPT_MARGIN 없음) 하위호환.
        REREGISTER_ACCEPT_MARGIN = None
except ImportError:   # 실편집 파일 부재 — golden_eval_config.example.py 참고(복사해서 생성).
    GOLDEN_ROOT, LAB_MODE, MIN_S, HISTORY_ROOT, CONSENSUS_BOX_CROP = None, "", None, None, None
    REREGISTER_BOX_SUGGEST, REREGISTER_TOPN, REREGISTER_ACCEPT_MARGIN = 1, 0, None


def seed_env():
    """golden_eval_config 상수 → os.environ(setdefault). 드라이버 맨 위, gce/glec import 전에 호출.

    idempotent: setdefault 라 여러 번 불려도(드라이버끼리 import 연쇄) 안전하고, OS env 우선.
    """
    if GOLDEN_ROOT:
        os.environ.setdefault("ALIGN_GOLDEN_ROOT", str(GOLDEN_ROOT))
    if HISTORY_ROOT:
        os.environ.setdefault("ALIGN_MSR_HISTORY_ROOT", str(HISTORY_ROOT))
    if LAB_MODE:
        os.environ.setdefault("ALIGN_ENSEMBLE_LAB_MODE", str(LAB_MODE))
    if MIN_S is not None:
        os.environ.setdefault("CONSENSUS_MIN_S", str(MIN_S))
    if CONSENSUS_BOX_CROP is not None:
        os.environ.setdefault("CONSENSUS_BOX_CROP", str(CONSENSUS_BOX_CROP))
    os.environ.setdefault("REREGISTER_BOX_SUGGEST", str(REREGISTER_BOX_SUGGEST))
    os.environ.setdefault("REREGISTER_TOPN", str(REREGISTER_TOPN))
    if REREGISTER_ACCEPT_MARGIN is not None:
        os.environ.setdefault("REREGISTER_ACCEPT_MARGIN", str(REREGISTER_ACCEPT_MARGIN))
