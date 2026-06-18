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

# 골든 데이터 루트(align_images: <eqp>/<class>/<recipe>/...). 예: r"C:\\data\\align_images".
# None = 기본 경로(glec.GOLDEN_ROOT).
GOLDEN_ROOT = None

# consensus 과거 성공 S 풀의 *별도* root. **class/recipe 로만 매칭(eqp 무관)** — 같은 recipe 면
# 장비 달라도 공유. 레이아웃: <HISTORY_ROOT>/<class>/<recipe>/events/<event_id>/S*.jpeg (+ .<img>/cond.txt,
# office_success_downloader 포맷 그대로). None/부재 = LOO 폴백(from_msr 안에서 leave-one-out).
HISTORY_ROOT = None

# rcp-only arm 매처 채널. "" = production ensemble, "edge_ncc" = C4 레버 평가.
LAB_MODE = ""

# consensus 최소 S(바닥 3). None = consensus 드라이버 기본값.
MIN_S = None

# === OM/SEM split 판정 임계 (golden_combined_eval_cond 전용; 오피스에서 데이터 보고 튜닝) ===
# 이 블록은 combined 드라이버만 읽는다(env 브리지 X). 실편집 golden_eval_config.py 에 복사해 조정.
SPLIT_MIN_FRAMES = 30      # modality당 최소 채점 프레임(미달 → verdict=insufficient)
SPLIT_MIN_RECIPES = 5      # modality당 최소 recipe
SPLIT_RANK1_GAP = 0.10     # |rank1(OM)-rank1(SEM)| 이 이상이면 split 후보(10pp)
SPLIT_RANK1_FLOOR = 0.70   # 약한 쪽 routed rank1 이 이 밑이면 split 후보
SPLIT_DOMINANCE = 0.40     # 지배 실패유형 최소 비중(총 실패 중)
