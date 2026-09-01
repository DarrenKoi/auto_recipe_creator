[DIGEST] VLM grounding fine-tune로 (rcp, msr) 이미지에서 crosshair 좌표를 "후보 하나로 맞히는" 것은 좌표 토큰화 양자화 하한(5120px 기준 약 5px/bin)과 벤치마크 실측(작은 타깃 정확도 8~40%, 고해상도 좌표 드리프트) 때문에 비현실적이며, VLM은 feasibility/설명/분류 역할에 머무는 것이 맞다(7월 결론 유지, 근거 대폭 보강). 매처 밖의 실질 레버는 8월 벤치에서 확정된 87% recall miss와 "재등록이 답" 결론 위에서 재정렬되며, cond.txt 좌표가 수백만 장 규모의 무료 라벨로 확보된 것이 7월 이후 가장 큰 변화다. 상위 3 추천: (1) distinctiveness predictor 기반 재등록 후보 자동 산출, (2) cond.txt 라벨로 학습하는 proposer(dense heatmap 회귀 + self-supervised pretrain), (3) engineer_done 수동 보정을 라벨로 회수하는 HITL 루프.

---
status: research
date: 2026-09-02
scope: 브리프 D. VLM fine-tune 타당성 판정과 매처 밖의 대안 조사. 오피스 데이터 없이 Mac에서 작성, 웹 검색 근거 포함.
related:
  - ../cv/align_fail_vlm_deep_learning_addendum_ko.md (2026-07-10, 이 문서의 델타 기준선)
  - ../cv/align_fail_cv_methods_research_ko.md (2026-07-10)
  - ../../adr/0006-template-bank.md
  - ../reranker_ab_failure_analysis.md
  - ../../../project_progress/03_workflow_2.md

# 브리프 D: SEM 이해 VLM fine-tune 및 매처 밖의 대안

## 0. 선행 문서(2026-07-10) 대비 델타

7월 문서(addendum, CV methods)와 판단이 같거나 다른 지점을 먼저 고정한다. 반복은 피하고 아래 4개가 이 문서의 새로운 내용이다.

**델타 1** — 87% recall miss가 확정되면서 ranker/verifier 축의 우선순위가 붕괴했다.
7월 addendum는 P0-E patch-pair ranker를 "가장 현실적인 학습 실험"으로 두었고 "top-K에 없는 정답은 ranker의 실패가 아니라 proposer 한계"로 별도 표에 두라고 이미 명시했다(addendum 150행). 이 구분 자체는 옳았다. 그러나 8월 벤치에서 남은 실패의 약 87%가 정답이 top-K에 아예 없는 recall miss로 확정됐다(docs/project_progress/03_workflow_2.md:118-120). ranker가 고칠 수 있는 영역(gt_in_topk=true)은 남은 실패의 ~13%뿐이므로 7월의 P0-E/P1-G(LoFTR, RoMa 등 learned correspondence verifier)는 방향이 틀린 것까지는 아니지만 표적의 크기가 작아졌다. reranker 축은 이미 실험적으로도 사망했다(MI rerank -0.013, contour -0.167, reranker_ab_failure_analysis.md:15-16). 결론: 학습 투자는 ranker가 아니라 proposer와 proposer 밖으로 간다.

**델타 2** — 라벨 경제가 golden 수백 장 전제에서 cond.txt 수백만 장으로 바뀌었다.
7월 addendum의 P3-I(self-supervised pretraining)는 "unlabeled S/E frame은 충분하지만 target label이 적으면"이라는 전제 위에 있었다(addendum 192-196행). 지금은 msr 이미지의 cond.txt가 crosshair 좌표(2숫자, px@5120)를 주고(common.md 5-8행, 코드 근거 poc/workflow_3/align/cond_file.py:6-15) 오피스 MES에서 수백만 장에 접근 가능하다(common.md 8행). 즉 self-supervised를 먼저 할 필요가 없다. supervised proposer 학습(밀도맵/좌표 회귀)이 곧바로 현실적 선택지가 됐고, 이것이 P3-I에 대한 7월 판정과 다른 지점이다. 단 rcp 쪽 등록 박스 4좌표(cond_file.py:6-15의 elements[6]-[9])도 같은 형식이므로 distinctiveness 라벨(아래 3.1)의 재료도 된다.

**델타 3** — VLM-ROI 축이 구조적으로 종결됐다는 것을 코드와 문서에서 재확인했다.
SEM은 고배율에서 key가 프레임의 80~100%를 채운다. 그래서 ROI 축소가 무용하다는 판정이 문서로 남아 있다(poc/workflow_2/docs/specs/2026-06-22-oracle-roi-ceiling-design.md:3-9, SUPERSEDED 표기). 구현물 자체는 남아 있다: `vlm_align_key_region.py`(대형 VLM 멀티이미지 region probe, :9-25), `vlm_align_key_box.py`(소형 grounding 모델 feasibility probe, :3-12), `probe_multi_image_vlm.py`(멀티이미지 capability probe, :9-22). reranker 레버 사망 후 채택된 escalation이 VLM-region + CV 탐색공간 축소였지만(reranker_ab_failure_analysis.md:118-121) SEM에서는 oracle ROI 상한 분석이 이 축 자체를 무의미하게 만들었다. 웹 리서치 결과(2절)와 합쳐 보면 이 판정은 외부 문헌과도 일치한다.

**델타 4** — 7월 문서의 안전 경계, VLM은 좌표를 내지 않는다는 원칙은 유지한다.
7월 결론의 불변 조건(addendum 48-54행)과 CLAUDE.md:308("OpenCV produces quantitative scores and final coordinates; VLM only identifies regions, explains ambiguous FOVs, and assesses feasibility")은 이번 리서치에서 오히려 강하게 재확인됐다(1절). 바뀐 것은 없다.

## 1. VLM fine-tune 타당성: grounding 좌표 회귀는 비현실적인가

브리프 질문: Qwen3-VL / InternVL / Florence-2 / PaliGemma 계열을 (rcp 박스 이미지, msr 이미지) -> crosshair 좌표로 grounding fine-tune 해서 후보 하나로 맞히기를 해낼 수 있는가.

### 1.1 좌표 표현과 양자화 하한

| 모델 | 좌표 표현 | 양자화 | 5120px 기준 이론 하한 |
|---|---|---|---|
| Qwen2-VL | [0,1000) 정규화 | 1000 bin | 약 5.12px/bin |
| Qwen2.5-VL | 리사이즈(28px patch) 후 절대좌표 | 사실상 연속 | 리사이즈 상한(약 1.0M px, 정방형 ~1002px)로 약 5.1x downscale, 1 VLM px = 약 5.1 source px |
| Qwen3-VL | [0,1000] 정규화 재전환 | 1000 bin | 약 5.12px/bin |
| PaliGemma/PaliGemma 2 | <loc0000>~<loc1023> location token | 1024 bin | 약 5.0px/bin (입력 896px면 약 5.7x downscale) |
| Florence-2 | location token 1000 bin | 1000 bin | 약 5.12px/bin |
| InternVL 2/2.5/3 | 0~1000 정규화 box | 1000 bin | 약 5.12px/bin (dynamic tile, 최대 4K) |

근거: Qwen2-VL 좌표 정규화(https://arxiv.org/pdf/2409.12191), Qwen2.5-VL은 resized 절대좌표와 28px patch(https://github.com/QwenLM/Qwen2.5-VL/issues/721, https://github.com/QwenLM/Qwen3-VL/issues/676), Qwen3-VL [0,1000] 재전환(https://arxiv.org/abs/2511.21631, https://github.com/QwenLM/Qwen3-VL/issues/1486), PaliGemma location token(https://arxiv.org/pdf/2407.07726, PaliGemma 2 https://arxiv.org/html/2412.03555), Florence-2 1000 bins(CVPR 2024 논문), InternVL 0~1000(https://arxiv.org/abs/2410.16261).

구조적 한계 3가지:
1. **양자화 하한**: 모든 모델이 1000~1024 bin 좌표 토큰을 쓴다. 5120px 좌표계에서 최선이어도 약 5px/bin이고 이는 우리 요구(픽셀 단위 rank1)와 차원이 다르다. Qwen 팀 자신이 "이미지가 >4k*4k이면 bbox가 biased 될 수 있다"고 공개 답변했다(https://github.com/QwenLM/Qwen2.5-VL/issues/721).
2. **고해상도 positional encoding 열화**: 고해상도 입력에서 VPE가 열화되어 임의 노이즈가 아니라 방향성 바이어스(모델 내부 공간 사전으로 회귀)가 생긴다는 2026년 분석이 있다("Mitigating Coordinate Prediction Bias from Positional Encoding Failures", ACL Findings 2026, https://aclanthology.org/2026.findings-acl.1034.pdf). GUI-CURSOR의 프로브 실험에서도 Qwen2.5-VL-7B가 이미지 중앙만 맞추고 가장자리에서 방향성 실패를 보였다(https://arxiv.org/html/2509.21552).
3. **좌표 생성의 brittleness**: "generating exact numerical coordinates is a challenging task for language-centric architectures"이며 고해상도에서 quantization error가 명시적으로 지적됐다(VLM-FO1, https://arxiv.org/html/2509.25916). 좌표 토큰 cross-entropy는 작은 요소일수록 예측 박스 중심이 GT 밖으로 나가는 저품질 패턴을 보인다(R-VLM, ACL Findings 2025, https://aclanthology.org/2025.findings-acl.501.pdf). 좌표 문자열을 아예 버리고 region token retrieval로 가는 흐름(VLX-Seek, https://github.com/om-ai-lab/VLX-Seek)도 이 문제의 인정이다.

### 1.2 벤치마크 실측: fine-tune 해도 어디까지 오르는가

GUI grounding이 가장 직접적인 유사 문제(스크린샷 위 좌표)다.

- ScreenSpot-v2(일반 해상도): SFT 후 88~93%로 포화(https://arxiv.org/html/2505.13227v3, OSWorld-G/Jedi 논문 표).
- ScreenSpot-Pro(고해상도 전문 소프트웨어, 작은 타깃): UI-TARS-7B 35.7%, Qwen2.5-VL-7B 27.6%, 400만 셋으로 SFT한 Jedi-7B도 39.5%. icon(작은 요소)만 보면 Qwen2.5-VL-7B 평균 7.6%(같은 논문 Table 3).
- OSWorld 실패 분석: 실패 550건 중 75% 이상이 마우스 클릭 좌표 오류(https://arxiv.org/html/2404.07972v2).
- zoom-in 2단계 계층으로만 의미 있는 상승: GMS(Scanner+Locator 계층 탐색)가 ScreenSpot-Pro 35.7%(단독 좌표 예측은 2~4% 수준, https://arxiv.org/html/2509.24133v1), R-VLM의 zoom-in 재예측(+3.5% icon, https://arxiv.org/html/2507.05673v1), ScreenSeekeR(18.9% -> 48.1%, 계층 탐색, https://arxiv.org/html/2504.07981). 문헌 전체가 단일 forward로 좌표 하나를 맞히는 쪽이 아니라 "계층 탐색으로 좌표 문제를 회피"하는 방향이다.

우리 문제와의 대응: SEM msr 이미지에서 crosshair 좌표는 5120x5120 화면 위 작은 특정 지점 1픽셀을 요구한다. 이는 ScreenSpot-Pro의 icon 케이스보다 훨씬 가혹하다(타깃이 시각적으로 크지 않고, 주변이 반복/평탄 구조). 그리고 우리 문제의 어려움은 어디쯤인가가 아니라 어느 junction인가라는 aperture problem이다. 그래서 VLM이 잘하는 coarse 제안은 이미 oracle ROI 분석에서 무용 판정을 받았다(델타 3).

### 1.3 판정

- 좌표 회귀용 fine-tune은 하지 않는다. 근거는 1.1(양자화/해상도 구조 한계)과 1.2(SFT 후에도 작은 타깃 8~40% 수준, 그것도 GUI 도메인). SEM domain shift가 추가되므로 이보다 나아질 근거가 없다. 5120px 전체를 VLM 입력으로 넣는 것 자체가 유효 해상도를 약 1000px 수준으로 떨어뜨린다.
- VLM이 CV proposer보다 나을 근거는 없다. 반대로 문헌은 VLM 단독 좌표가 template/correspondence 계열보다 정밀도에서 열위라는 방향이다. sub-pixel 정밀 좌표를 grounding VLM으로 해낸 공개 사례는 확인되지 않았다(추측이 아니라 문헌 부재 확인 결과임).
- VLM은 feasibility/설명/분류 역할에 머무는 것이 맞다. 7월 P2-D("반도체 VLM fine-tune은 분류/설명에 한정", addendum 114-122행)와 동일 판정이며 이번에 외부 문헌으로 근거가 보강됐다. 7월과 판단이 다른 부분은 없다. 다만 P2-D 당시에는 근거가 사례 논문(https://arxiv.org/abs/2409.07463) 중심이었는데 이제는 좌표 토큰화 구조와 벤치마크 실측으로 뒷받침된다.

## 2. VLM이 유효한 다른 자리

좌표 결정이 아니라 좌표 밖의 판정에서 VLM을 쓸 자리. 현재 자산: mai-ui-8b(grounding, flask_vlm.py:39-45, ui_venus_mai_locator.py:59-60), paddleocr-vl-1.5(OCR), 대형 VLM 멀티이미지 경로(Kimi-K2.6, vlm_align_key_region.py:82-84). 서빙은 vLLM, H200 140GB x2, 호스트 RAM 16GB가 프로세스 수 제약(docs/project_progress/01_vlm_deployment.md:49-54).

### 2.1 (a) 재등록 권고 시 "어디를 key로 다시 등록해야 하나" 제안

- **가치**: 8월 결론의 종착점이 재등록이므로(docs/project_progress/03_workflow_2.md:170-171) 재등록하라고만 알리는 것과 이 영역(unique junction 조합, 주변 반복 구조와 대비되는 곳)을 key로 등록하라고 짚어주는 것은 엔지니어 체감 가치가 다르다. 현재 reregister_recommended는 second_ratio 임계 기반 verdict만 준다(CLAUDE.md:316, feasibility_check.py).
- **형태**: 대형 VLM 멀티이미지(Kimi-K2.6)에 rcp 원본과 주변 FOV를 주고 "변별 가능한 작은 영역 후보 3곳을 relative_1000으로" 요청하는 것. 좌표 권위는 CV distinctiveness 점수(3.1)가 갖는다. VLM은 후보 설명과 feasibility만.
- **비용**: 프롬프트/평가 설계 1~2일 수준, 멀티이미지 경로는 이미 probe로 검증됨(probe_multi_image_vlm.py:9-22). 단 payload 제약(WebP, 1MB) 확인 필요.
- **정직한 평가**: 좋은 key를 VLM이 언어로 골라낼 수 있다는 검증이 아직 없으므로 이것은 CV distinctiveness predictor(3.1)의 보조 설명자이지 결정자가 되어서는 안 된다. 우선순위는 3.1보다 낮다.

### 2.2 (b) 엔지니어 수동 보정 녹화 해석

- **가치**: workflow_3에는 반자동 모드(OK 보류 후 engineer 알림, correction.py:81-84,435-449의 awaiting_engineer_ok)와 엔지니어 done 감지(monitor/engineer_done_align_adjustment.py:1-22)가 이미 있다. 엔지니어가 어디를 클릭했는지, 어떤 pan/zoom 경로를 갔는지를 녹화에서 자동 파싱하면 3.3의 HITL 라벨 회수에 입력으로 들어간다.
- **형태**: 화면 녹화/스크린샷 시퀀스에서 (i) 엔지니어가 재등록을 했는지 vs 그냥 reposition했는지 분류, (ii) 재등록했다면 새 key 영역의 위치를 GUI grounding VLM(mai-ui)으로 식별. 이건 VLM의 원래 용도(GUI grounding)다.
- **비용**: 녹화 인프라가 이미 있고 click 좌표 추출은 화면 좌표계만 맞추면 되므로 낮음. 이 역할의 가치는 자체 성능보다 3.3 루프의 데이터 파이프라인에서 나온다.

### 2.3 (c) 실패 원인 분류(공정 변화 vs 초점 vs 오염 vs 반복구조 비변별)

- **가치**: 남은 실패 87%의 성격 규명은 다음 투자 결정을 바꾼다. 현재는 not_distinctive(반복구조), drift, occlusion 등을 CV 점수 조합으로 추정하는 수준이다. 7월 addendum의 P1-C state classification(usable_for_matching / occluded / wrong_panel_or_mode, addendum 103-112행)과 P2-H anomaly gate가 이미 설계돼 있다.
- **형태**: strict JSON enum 분류(key_visible / focus_blurred / contaminated_or_process_changed / repeating_non_distinctive). 반도체 EM 도메인 VLM 연구가 분류/설명에 한정해 유효하다는 선례가 있다(https://arxiv.org/abs/2409.07463).
- **비용**: 라벨이 필요하다. SEM 이미지 특성상 공정 변화 vs 오염 구분은 사람도 애매할 수 있어 golden 라벨 구축 비용이 생각보다 크다. 반면 not_distinctive 여부는 consensus 실패 기록과 재등록 실적(reranker_ab_failure_analysis.md:151-156의 재등록 우선순위 리포트)에서 부분적으로 회수 가능하다.
- **정직한 평가**: full 원인 분류는 라벨 비용 대비 불확실성이 크다. matching 가능 상태인가를 묻는 이진 gate(P1-C)만 먼저 하고 세부 원인 분류는 HITL 데이터가 쌓인 뒤로 미루는 것이 합리적이다.

## 3. 매처 밖의 대안

### 3.1 문제 정의 변경: 1점 localization 대신 재등록 후보 자동 산출 (distinctiveness predictor)

현 결론이 재등록이 답이라는 점(common.md 16행, docs/project_progress/03_workflow_2.md:170-171, ADR 0006 "레버는 align-key distinctiveness")에서 이것이 가장 큰 레버다. 질문은 주어진 rcp key가 매칭에 적합한가(distinctive한가)를 이미지에서 예측하는 학습이다.

- **학습 방법 문헌**: 이 문제는 풍부한 선례가 있다.
  - R2D2: keypoint의 repeatability와 별도로 reliability(이 descriptor가 고신뢰 매칭 가능한가)를 학습. 반복 패턴(checkerboard 등)은 repeatable해도 matching이 불가능하다는 문제 정의가 SEM aperture problem과 동형이다(https://arxiv.org/abs/1906.06195, NeurIPS 2019).
  - QATM: matching pair의 1-to-1 품질(유일성)을 soft-ranking으로 채점. 1-to-many(반복 패턴)를 낮은 점수로 자동 반영한다(https://arxiv.org/abs/1903.07254).
  - matchability prediction: dense CNN descriptor로 템플릿의 matchability map을 예측, 반복 패턴(풀 등)에 낮은 가중치(3DV 2015, https://cris.fbk.eu/retrieve/ddb241a5-3545-ba8a-e053-3a05fe0afd55/3DV%202015.pdf).
  - SiLK: keypoint 정의 자체를 "matching으로 reliably 식별 가능성"으로 학습(https://openaccess.thecvf.com/content/ICCV2023/papers/Gleize_SiLK_Simple_Learned_Keypoints_ICCV_2023_paper.pdf).
  - SLAM 쪽: 정보량(observability/Max-logDet) 기반 feature 선택이 pose 오차를 크게 줄인다는 선례(https://arxiv.org/abs/1905.07807, CVPR 2015 Good Features to Track).
- **우리 데이터에 붙이는 법**: 라벨은 2종으로 만든다. (i) rcp 박스 4좌표가 있는 수백만 장에 대해 같은 recipe의 S 이미지들과의 실제 matcher 결과(consensus rank1 성공/실패)를 outcome 라벨로 회수. (ii) 엔지니어가 실제로 재등록한 사례(새 key 위치)를 positive 위치 라벨로 회수(2.2의 녹화 파싱). 최종 산출물은 rcp 이미지 각 patch의 locatability score map이고 점수가 낮은 recipe를 재등록 워크리스트에 올린다(기존 워크리스트 설계: poc/workflow_2/docs/specs/2026-06-25-reregister-rank1-distinctiveness-worklist-design.md).
- **기대 이득**: SEM 104개 실패 recipe 중 84개(81%)가 비변별 키였다(docs/project_progress/03_workflow_2.md:151-156). 이들이 사전에 걸러지면 재등록 사이클이 자동화되고 재등록 효과는 실측돼 있다(recall +0.282, rank1 +0.269, reranker_ab_failure_analysis.md:33-35).
- **Mac 준비 가능성**: 합성 이미지로 파이프라인 구축과 유닛 테스트는 가능하다(기존 합성 smoke 테스트 선례: test_align_key_match.py, AGENTS.md 빌드 명령 절). 실제 검증은 오피스 데이터가 필요하다.

### 3.2 cond.txt 대규모 라벨로 학습하는 proposer (recall miss 직격)

87% recall miss에 정면으로 쓰는 방법. (rcp template, msr 이미지) 쌍에 cond.txt crosshair 좌표를 정답으로 두고 heatmap 회귀(dense localization) 모델을 학습해 현재 Chamfer/RRF proposer의 recall을 높인다.

- **왜 지금 가능한가**: 라벨이 무료 대규모(common.md 8행, 델타 2). 7월 P3-I의 전제("target label이 적으면")가 무너졌다.
- **모델 선택지**: (i) 소형 CNN heatmap 회귀(template crop + live frame -> localization heatmap). (ii) frozen DINOv2 등 foundation feature + 얇은 head(P1-F/P3-I의 보수적 경계 유지, addendum 152-161행). (iii) detector-free dense matcher(LoFTR 계열)를 같은 데이터로 fine-tune하는 것은 P1-G의 recall-miss 버전 변형이다. 자연 이미지 사전학습 모델의 EM domain mismatch 경고(https://arxiv.org/abs/2602.08505) 때문에 (i)이 가장 도메인 독립적이다.
- **위험**: 반복 패턴의 정보적 모호성(addendum 30행의 "진짜 위치와 decoy가 정보적으로 구별되지 않으면 어떤 모델도 근거 없는 확신을 낸다")은 학습으로도 사라지지 않는다. 학습 proposer는 없는 recall을 되찾는 것이지 not_distinctive를 해결하는 것이 아니므로 3.1과 짝으로 가야 한다. 추측 포함: 학습 proposer가 decoy 위치까지 높은 확신을 낼 위험이 있으므로 출력은 top-K 후보 추가 기여로 제한하고 좌표 권위는 유지한다.
- **검증**: gt_in_topk=false subset에서만 proposer recall 개선을 별도 집계(CV methods 문서 4.2의 A/B matrix와 동일 규약, align_fail_cv_methods_research_ko.md:174-203).

### 3.3 active learning / human-in-the-loop 라벨 회수

- **기존 자산**: awaiting_engineer_ok(correction.py:81-84,435-449), engineer_done_align_adjustment.py:1-22. 즉 엔지니어 개입 지점이 이미 코드에 계측돼 있다.
- **문헌 근거**: 템플릿 매칭 후보를 classifier로 정오분류할 때 batch-mode active learning으로 36개 라벨(약 25초)로 수렴, 전체 GT 생성 대비 약 10배 처리량(Würzburg, https://www.informatik.uni-wuerzburg.de/fileadmin/10030100/paper-clean.pdf). 검사 도메인 HITL에서 엔지니어 이진 판정을 모델에 직접 반영해 라벨 비용 43%로 절감한 사례(https://arxiv.org/html/2608.17775, 단 uncertainty sampling이 random보다 나았다는 근거는 없었다는 반례 포함). 불확실 샘플만 수동 리뷰하는 xAI+AL 파이프라인(https://arxiv.org/abs/2307.05508).
- **우리 루프 설계**: (1) matcher가 low-confidence/abstain한 사례를 engineer_review 큐에 넣는다(기존 safety router, addendum 20행). (2) 엔지니어 행동(그냥 OK / 재등록 / 포기)을 engineer_done 시점에 기록한다. (3) 재등록이면 새 key 위치를 자동 추출해 3.1의 positive 라벨, matcher 실패 상황 전체를 3.2의 학습 데이터로 회수한다.
- **가치**: 이 방법 자체의 rank1 향상 폭은 간접적이지만 3.1/3.2를 지속 가능하게 하는 데이터 인프라다. 구현 비용이 가장 낮고 Mac에서 코드 작성이 가능하다(녹화 파서는 합성 테스트로 검증).

### 3.4 장비 측 신호를 활용한 탐색 전략

- **기존 자산**: grid_search.py의 search-around 재설계(FOV_um = 135,000/Mag, :47; registered_magnification은 cond.txt에서, :110-116), zoom ladder(monitor/cycle.py:2020), PM dropdown 배율 탐색(:2367, :2724), PM mode 판정(sem_box_detect.py:114-135).
- **문헌 근거**: review SEM 업계의 표준은 저배율 근사 위치에서 고배율 addressing으로 이동하는 계층 구조다. SEM addressing 특허는 low-magnification AP에서 템플릿 매칭으로 변위를 추정하고 beam shift로 고배율 정밀 이동하며 AP 후보 선정 기준으로 "X/Y 방향 모두 고유한 패턴(uniqueness)"을 명시한다(US 20080159609, https://www.patents-review.com/a/20080159609-sem-system-method-producing-recipe-imaging-or-measuring-sem.html). Hitachi review SEM 특허도 stage 좌표 근처 템플릿 매칭 + 주기(period) 자동 검출로 정렬 위치를 선정한다(US12210338).
- **판정**: 이 레버는 이미 workflow_3에서 상당 부분 구현돼 있으므로 추가 학습 과제가 아니라 튜닝 과제다. 다만 uniqueness 기준의 AP 후보 선정은 3.1의 distinctiveness predictor와 같은 문제의 장비 특허 버전이므로 3.1의 우선순위를 뒷받침한다.

### 3.5 synthetic data / simulation

- **문헌 근거**: SuperPoint의 homographic adaptation은 합성 도형 사전학습 + 실영상 무작위 호모그래피로 pseudo-GT를 만드는 표준(self-supervised synthetic-to-real, https://arxiv.org/pdf/1712.07629). 가장 직접적인 선례는 SEM 도메인: mask SEM 분석 도구셋이 CAD 기반 SEM 디지털 트윈으로 70만 정렬 쌍을 생성해 CAD-SEM 정합망을 학습, 실영상 185장에서 평균 NCC 0.9(https://design2silicon.com/wp-content/uploads/2020/10/A-deep-learning-mask-analysis-toolset-using-mask-SEM-digital-twins.pdf). SEM->CAD GAN 변환 후 NCC 매칭으로 4종 패턴에서 약 100% 정렬 정확도, pre-aligned 쌍 20장만으로 학습한 사례도 있다(https://doi.org/10.1109/tsm.2022.3171788). 제조 검사 도메인에서 domain randomization으로 합성 전용 학습이 실데이터 학습을 능가했다는 2025년 결과들(https://arxiv.org/html/2506.07539v1, https://doi.org/10.1016/j.procir.2025.02.205).
- **우리 문제에 적용**: 3.1/3.2의 학습을 오피스 데이터 없이 Mac에서 준비하는 유일한 경로다. SEM 물리를 보존하는 augmentation만 유효하다는 7월 경계(P3-I의 augmentation 제약, addendum 196행: random crop/약한 blur는 가능, 90도 회전/elastic warp/defect 삽입은 stage·magnification contract를 깬다)를 그대로 유지한다. rcp 박스 crop 합성(조명/노이즈/스케일 변화, crosshair 합성)으로 학습 파이프라인을 먼저 구축하고 오피스 실데이터로 교체하는 순서.
- **판정**: 단독으로는 레버가 아니라 3.1/3.2의 활성화 조건이다.

## 4. 비교표와 상위 3 추천

| 후보 | 예상 이득(rank1 향상 폭) | 구현 비용 | 검증 가능성(Mac 준비) | 근거 |
|---|---|---|---|---|
| 3.1 distinctiveness predictor(재등록 후보 자동 산출) | 크. 비변별 SEM 키 81%가 재등록 대상이고 재등록 실측 효과 rank1 +0.269. "후보 하나로 맞히기"의 근본 해법 | 중. score map 모델 + 워크리스트 연계. 문헌 선례(R2D2/QATM)가 구체적 | 합성 파이프라인으로 구축 가능, 실검증은 오피스 | 03_workflow_2.md:151-156, reranker_ab_failure_analysis.md:33-35, R2D2, QATM, matchability(3DV15) |
| 3.2 cond.txt 라벨 learned proposer | 중~크. 남은 실패 87%가 recall miss이므로 직격. 단 정보적 모호성 케이스는 회복 불가(상한 존재) | 중~고. 학습 인프라 필요하나 라벨 무료 | 합성으로 파이프라인 구축 가능, 실검증은 오피스 | common.md 8행, cond_file.py:6-15, 03_workflow_2.md:118-120 |
| 3.3 HITL 라벨 회수 루프 | 간접. 자체 rank1 향상은 없으나 3.1/3.2의 데이터 공급 | 저. 계측 지점이 이미 코드에 존재 | Mac에서 파서/기록 파이프라인 작성 가능 | correction.py:81-84, engineer_done_align_adjustment.py:1-22, Würzburg AL 10x |
| 3.4 장비 신호 탐색 계층화 | 중. 이미 구현된 grid_search/zoom ladder의 튜닝 이득 | 저~중(튜닝) | 오피스 Windows에서만 실검증 가능 | grid_search.py:47,110-116, US 20080159609 |
| 3.5 synthetic data | 간접. 3.1/3.2의 활성화 조건 | 중(CAD/generation 파이프라인) | Mac에서 합성 생성 가능(이 레버의 핵심 가치) | design2silicon 70만 쌍, SuperPoint |
| VLM 좌표 fine-tune | 음수 가능. 양자화 하한 약 5px + 작은 타깃 8~40% 실측 | 고(GPU 학습 + 데이터 파이프라인) | Mac에서 불가(GPU), 되더라도 하한에 막힘 | 1절 전체 |
| VLM feasibility/분류 확장(2.1c, 2.3) | 소~중. gate/설명 품질 개선, rank1 직접 기여 없음 | 저~중. 라벨 필요한 부분과 아닌 부분 구분 | 프롬프트/평가 설계는 Mac 가능 | addendum P1-C/P2-D, arXiv 2409.07463 |

**상위 3 추천** (예상 이득 x 비용 x 검증 가능성 종합):

1. **3.1 distinctiveness predictor**: 현 결론(재등록이 답)을 학습으로 자동화하는 유일한 후보이고 81% 비변별 키 + 재등록 실측 효과(rank1 +0.269)가 이득 상한을 보증한다. 문헌 선례(R2D2 reliability, QATM 유일성 채점, matchability prediction)가 방법 설계를 구체적으로 지원한다.
2. **3.2 cond.txt 라벨 learned proposer**: 87% recall miss에 직격탄이 되는 유일한 학습 접근. 3.1과 짝으로 간다(3.2는 recall 회복, 3.1은 회복 불가 케이스의 조기 분류). 먼저 frozen-encoder + 얇은 head부터 검증하는 7월 P1-F/P3-I의 보수적 경계를 유지한다.
3. **3.3 HITL 라벨 회수 루프**: 구현 비용이 가장 낮고(계측 지점이 이미 있음) 1, 2를 지속 가능하게 만드는 데이터 인프라. 즉시 착수 가능.

제외/유예: VLM 좌표 fine-tune(1절, 구조적 하한), VLM-ROI(델타 3, SEM에서 구조적으로 무용), pair-ranker 재시도(델타 1, 표적이 작아짐), full 실패 원인 분류(2.3, 라벨 비용 대비 불확실, 이진 gate만 우선).

## 참고 자료 (웹)

- Qwen2-VL 좌표 정규화: https://arxiv.org/pdf/2409.12191
- Qwen2.5-VL 절대좌표/28px patch/고해상도 bias: https://github.com/QwenLM/Qwen2.5-VL/issues/721, https://github.com/QwenLM/Qwen3-VL/issues/676
- Qwen3-VL grounding [0,1000]: https://arxiv.org/abs/2511.21631, https://github.com/QwenLM/Qwen3-VL/issues/1486
- PaliGemma location tokens: https://arxiv.org/pdf/2407.07726, PaliGemma 2: https://arxiv.org/html/2412.03555
- ScreenSpot-Pro(고해상도 소형 타깃에서의 부진): https://arxiv.org/html/2504.07981
- OSWorld-G/Jedi(grounding SFT 상한): https://arxiv.org/html/2505.13227v3, https://osworld-grounding.github.io/
- OSWorld 실패 75% 이상이 클릭 좌표 오류: https://arxiv.org/html/2404.07972v2
- VLM-FO1(좌표 생성 brittleness): https://arxiv.org/html/2509.25916
- R-VLM(좌표 토큰 품질, zoom-in 2단계): https://aclanthology.org/2025.findings-acl.501.pdf, https://arxiv.org/html/2507.05673v1
- GUI-CURSOR(positional bias): https://arxiv.org/html/2509.21552
- GMS(계층 탐색): https://arxiv.org/html/2509.24133v1
- ScreenSeekeR: https://arxiv.org/html/2504.07981
- 좌표 PE 열화 보정(ACL Findings 2026): https://aclanthology.org/2026.findings-acl.1034.pdf
- R2D2 reliability/repeatability: https://arxiv.org/abs/1906.06195
- QATM(유일성 채점): https://arxiv.org/abs/1903.07254
- Matchability prediction(3DV 2015): https://cris.fbk.eu/retrieve/ddb241a5-3545-ba8a-e053-3a05fe0afd55/3DV%202015.pdf
- SiLK: https://openaccess.thecvf.com/content/ICCV2023/papers/Gleize_SiLK_Simple_Learned_Keypoints_ICCV_2023_paper.pdf
- SLAM 정보량 feature 선택: https://arxiv.org/abs/1905.07807, https://openaccess.thecvf.com/content_cvpr_2015/papers/Zhang_Good_Features_to_2015_CVPR_paper.pdf
- SuperPoint homographic adaptation: https://arxiv.org/pdf/1712.07629
- SEM CAD 디지털 트윈 70만 쌍: https://design2silicon.com/wp-content/uploads/2020/10/A-deep-learning-mask-analysis-toolset-using-mask-SEM-digital-twins.pdf
- SEM->CAD GAN 정합: https://doi.org/10.1109/tsm.2022.3171788
- 제조 합성데이터 domain randomization: https://arxiv.org/html/2506.07539v1, https://doi.org/10.1016/j.procir.2025.02.205
- 템플릿 매칭 후보 active learning: https://www.informatik.uni-wuerzburg.de/fileadmin/10030100/paper-clean.pdf
- HITL memory bank 보정: https://arxiv.org/html/2608.17775
- 반도체 EM VLM(분류/설명 한정): https://arxiv.org/abs/2409.07463
- EM foundation model domain mismatch: https://arxiv.org/abs/2602.08505
- SEM addressing 특허(저배율 AP, uniqueness 기준): https://www.patents-review.com/a/20080159609-sem-system-method-producing-recipe-imaging-or-measuring-sem.html
- Hitachi review SEM alignment 특허: US12210338

RESEARCH_DONE
