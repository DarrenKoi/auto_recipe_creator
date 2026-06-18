# poc/workflow_2/golden_eval_config.example.py
"""golden eval 설정 템플릿 — 이 파일을 golden_eval_config.py 로 복사해서 쓴다.

왜 분리 파일인가
----------------
실험 토글(LAB_MODE off/edge_ncc, MIN_S 3/4/5, 골든 경로)을 자주 바꾸므로 driver 로직 파일
(golden_combined_eval_cond.py)을 건드리지 않게 상수만 떼어냈다. 실편집 파일 golden_eval_config.py 는
gitignore 라 "지금 뭘 돌리는 중인가" 스크래치가 git 에 안 남는다(템플릿만 추적).

쓰는 법
-------
    1) 이 파일을 같은 폴더에 golden_eval_config.py 로 복사.
       (Windows)  copy poc\\workflow_2\\golden_eval_config.example.py poc\\workflow_2\\golden_eval_config.py
       (bash)     cp   poc/workflow_2/golden_eval_config.example.py poc/workflow_2/golden_eval_config.py
    2) 아래 3개 값만 고친다.
    3) uv run python poc/workflow_2/golden_combined_eval_cond.py

golden_eval_config.py 가 없으면 driver 는 아래와 동일한 기본값으로 동작한다(import 폴백).
env(ALIGN_GOLDEN_ROOT/ALIGN_ENSEMBLE_LAB_MODE/CONSENSUS_MIN_S)가 설정돼 있으면 env 가 우선.
"""

# 골든 데이터 루트. 예: r"C:\\data\\align_images". None = 기본 경로(glec.GOLDEN_ROOT).
GOLDEN_ROOT = None

# rcp-only arm 매처 채널. "" = production ensemble, "edge_ncc" = C4 레버 평가.
LAB_MODE = ""

# consensus 최소 S(바닥 3). None = consensus 드라이버 기본값.
MIN_S = None
