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
    2) 아래 경로 값(GOLDEN_ROOT·HISTORY_ROOT)을 채운다(LAB_MODE/MIN_S/SPLIT_* 는 선택).
    3) uv run python poc/workflow_2/golden_combined_eval_cond.py

golden_eval_config.py 가 없으면 driver 는 아래와 동일한 기본값으로 동작한다(import 폴백).
env(ALIGN_GOLDEN_ROOT/ALIGN_ENSEMBLE_LAB_MODE/CONSENSUS_MIN_S)가 설정돼 있으면 env 가 우선.
"""

# 골든 데이터 루트(align_images: <eqp>/<class>/<recipe>/...). 예: r"C:\\data\\align_images".
# None = 기본 경로(glec.GOLDEN_ROOT).
GOLDEN_ROOT = None

# Phase 2 E-frame confirmation 전용 데이터셋 루트(S-only golden 과 분리). 설정 시 reregister 드라이버가
# GOLDEN_ROOT 대신 이 루트를 walk 한다(미설정/None 이면 GOLDEN_ROOT 폴백). 레이아웃은 golden 과 동일하나
# recipe 마다 align_img_from_rcp + align_img_from_msr 에 **S 와 E 둘 다** 필요(rcp 키로 S/E free-search →
# score collapse 판정). consensus_history(HISTORY_ROOT)는 reregister 와 무관하므로 불필요.
EFRAME_ROOT = None

# consensus 과거 성공 S 풀의 *별도* root. **class/recipe 로만 매칭(eqp 무관)** — 같은 recipe 면
# 장비 달라도 공유. 레이아웃: <HISTORY_ROOT>/<class>/<recipe>/events/<event_id>/S*.jpeg (+ .<img>/cond.txt,
# office_success_downloader 포맷 그대로). None/부재 = LOO 폴백(from_msr 안에서 leave-one-out).
# 수집 단위 = 측정 건수: 1건 = OM 2장 / SEM 3장(같은 마크·다른 stage 위치). consensus 는 이제
# modality별로 독립 평가되므로([[project_om_sem_positions_per_measurement]]) modality당 ≥MIN_S 장 필요 —
# 건수로 환산하면 OM ≥2건·SEM ≥1건. scaling(A) 곡선까지 보려면 recipe마다 건수를 다양화해 받는다.
HISTORY_ROOT = None

# rcp-only arm 매처 채널. "" = production ensemble, "edge_ncc" = C4 레버 평가.
LAB_MODE = ""

# consensus 최소 S(*장수* 기준, 바닥 3). 측정 건수로 환산: OM 2장/건 → ≥2건, SEM 3장/건 → ≥1건이라
# 같은 MIN_S 라도 modality별 건수 게이트가 다르다. None = consensus 드라이버 기본값.
MIN_S = None

# consensus arm whitebox box-crop A/B (center vs box, OM/SEM 층화). 1 이면 box arm 측정.
# 0(기본) 이면 기존 동작 그대로(digest 미출력, box_crop=False).
CONSENSUS_BOX_CROP = 0

# === OM/SEM split 판정 임계 (golden_combined_eval_cond 전용; 오피스에서 데이터 보고 튜닝) ===
# 이 블록은 combined 드라이버만 읽는다(env 브리지 X). 실편집 golden_eval_config.py 에 복사해 조정.
SPLIT_MIN_FRAMES = 30      # modality당 최소 채점 프레임(미달 → verdict=insufficient)
SPLIT_MIN_RECIPES = 5      # modality당 최소 recipe (OM 이 보통 binding — 희박 recipe 서 먼저 탈락)
SPLIT_RANK1_GAP = 0.10     # |rank1(OM)-rank1(SEM)| 이 이상이면 split 후보(10pp)
SPLIT_RANK1_FLOOR = 0.70   # 약한 쪽 routed rank1 이 이 밑이면 split 후보
SPLIT_DOMINANCE = 0.40     # 지배 실패유형 최소 비중(총 실패 중)

# === re-registration 리포트 (golden_reregister_report_cond) ===
# 1 이면 박스 제안(C2)까지, 0 이면 랭킹 리포트(C1)만.
REREGISTER_BOX_SUGGEST = 1
# DIGEST/overlay 상위 N 제한(0=무제한).
REREGISTER_TOPN = 0
# fast A/B: 앞 N개 recipe 만 처리(box-suggestion sweep 가 무거워 전체 >10분). 0=전체.
REREGISTER_MAX_RECIPES = 0
# fidelity 매칭 scale band(A/B). "" 면 코드 기본 tight band(0.85,1.0,1.15) — 작은 box crop 이
# 최소 scale(0.6) distractor 로 빠지는 걸 막음. 옛 동작 복원은 "0.6,0.75,0.85,1.0".
REREGISTER_FIDELITY_SCALES = ""
# fidelity hit tolerance(patch 단변 비율). 참 localization 0.20~0.24, distractor >=0.42 라
# 0.30 이 둘을 가른다. 옛 동작은 0.20.
REREGISTER_GT_TOL_NORM = 0.30
# rank-1 변별력 floor(Phase 3 worklist). rcp_rank1>=floor=OK, <floor 이고 cons_rank1>=floor=
# FRESH_SNAPSHOT, 둘 다 <floor=NEW_REGION. SEM ~0.5 군집이라 0.70 이 대부분 SEM flag. office 보정.
REREGISTER_DISTINCT_FLOOR = 0.70

# === Template-bank matcher bench (golden_consensus_eval_cond) ===
# heatmap=primary(soft-voting), rrf=extra arm. 1/0 토글.
TBANK_HEATMAP = 1
TBANK_RRF = 1
TBANK_PEAK_NMS_FRAC = 0.5      # heatmap peak-NMS 반경 = 이 비율 * 템플릿 단변.
TBANK_CLUSTER_TOL_FRAC = 0.10  # RRF arm 공간 클러스터 허용 = 이 비율 * 템플릿 단변.
TBANK_RRF_K = 60               # RRF 상수.

# === Phase 2: E-frame confirmation (golden_reregister_report_cond) ===
# 1 이면 flagged recipe 를 E(fail) 프레임 score-collapse 로 confirm, 0 이면 Phase 1 만.
REREGISTER_E_CONFIRM = 1
# collapse 규칙 임계(점수 ~0.6 압축 분포 기준 출발점, office 보정 대상).
REREGISTER_S_FLOOR = 0.6           # S best-score median 이 이 이상이어야 'collapse' 전제 성립.
REREGISTER_E_FLOOR = 0.5           # E best-score median 이 이 밑이면 (delta 작아도) collapse.
REREGISTER_COLLAPSE_MARGIN = 0.15  # S_rep - E_rep 이 이 이상이면 collapse.
