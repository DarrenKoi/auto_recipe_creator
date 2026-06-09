# Proposer Recall A/B 입문 — recall@K · canny/scharr/orient · RRF fusion

> 대상: `poc/workflow_2/proposer_recall_ab.py` 와 그 아래 `ensemble_proposer.py`
> 목적: (1) 본인 학습용, (2) 상사 질문 대응용 설명 자료
> 선행 문서: [`match_algorithms_intro.md`](match_algorithms_intro.md) (Chamfer/NMS/ORB/MI 기초)
> 작성일: 2026-06-09

---

## 0. 한 줄 요약

이 스크립트는 align key 탐색의 **첫 단계(proposer)만 격리해서**, "정답을 후보 바구니 안에
담는 능력"을 baseline(canny 1채널)과 ensemble(canny+scharr+orient 3채널)로 **A/B 비교**합니다.
맞췄는지(accuracy)가 아니라 **담았는지(recall)**만 잽니다.

생소했던 용어들의 정체부터 한 줄로:

```
recall@8/16/24 : 정답이 상위 8/16/24 후보 안에 들어있던 비율 (proposer 의 KPI)
canny  (C1)    : 밝기 엣지 기반 채널 = baseline (기존 매칭 엔진)
scharr (C2)    : 그래디언트 '세기' 기반 채널 (밀도 맞춘 이진화)
orient (C3)    : 엣지 '방향' 8분할 directional chamfer (polarity 불변)
fusion (RRF)   : 세 채널의 '순위'를 합산해 최종 후보 리스트 만들기
scales         : template 을 여러 배율로 매칭 (magnification 차이 흡수)
```

---

## 1. proposer vs reranker — 왜 recall 만 재는가

align key 탐색은 두 단계입니다.

```
[현재 SEM frame]  +  [recipe 등록 template]
        │
        ▼
   ① PROPOSER  ──→ 후보 K개 (xy 좌표 리스트)   ← 이 스크립트가 측정
        │
        ▼
   ② RERANKER  ──→ 후보 재점수 → 최종 1개
```

- **Proposer 의 임무**: 정답을 *어딘가 K개 안에* 넣기만 하면 성공. 1등으로 못 올려도 됨.
- **Reranker 의 임무**: 그 K개 중 진짜를 1등으로 끌어올리기.

둘을 한꺼번에 평가하면 실패 원인이 안 보입니다 — "proposer 가 정답을 아예 안 줬나, reranker 가
순위를 못 올렸나?" 그래서 이 스크립트는 proposer 만 떼어, reranker 재정렬을 **금지**하고
순수하게 "후보 안에 정답이 있나"만 봅니다 (파일 docstring `:1-5`).

> 설계 배경: 메모리 `matcher_flat_chamfer_distinctiveness` 참고 — 실데이터에서 chamfer 점수면이
> 평평해(변별력 부족) reranker 가 약합니다. 그러면 "proposer recall 부터 끌어올리고 rerank vs
> proposer 우선순위를 정하자"는 전략이 나오고, 그 recall 측정 도구가 바로 이 파일입니다.

---

## 2. recall@8 / @16 / @24 의 의미

**Recall@N** = "정답이 상위 N개 후보 안에 들어있던 frame 의 비율". 핵심 코드 두 개:

```python
# _gt_rank : 후보 리스트에서 정답이 처음 나타난 1-base rank (없으면 None=miss)
ax, ay = c.xy[0] + dx, c.xy[1] + dy          # 후보 중심 + align_offset
if math.hypot(ax - gt_x, ay - gt_y) <= tol * short:
    return i                                  # i번째 후보가 정답권

# _recall_at : rank<=n 인 비율
sum(1 for r in ranks if r is not None and r <= n) / len(ranks)
```

세 N값은 각각 다른 질문에 답합니다.

| N | 별명 | 질문 |
|---|------|------|
| **@8**  | budgeted | reranker 에 실제로 넘길 예산이 8개일 때 정답이 그 안에 있나? → **실전 KPI** |
| **@16** | 중간 | 예산을 2배로 늘리면 얼마나 더 건지나? |
| **@24** | shadow | 후보를 최대(`SHADOW_N=24`)로 풀면 이론 상한은? → **proposer 잠재력** |

### 결과표 읽는 법

```
  variant       recall@8  recall@16  recall@24
  baseline(C1)     ...        ...        ...
  ensemble         ...        ...        ...
  solo:canny       ...        ...        ...     ← 채널별 단독 성능(attribution)
  solo:scharr      ...        ...        ...
  solo:orient      ...        ...        ...

  >>> budgeted recall@8 lift = +X  | shadow recall@24 lift = +Y
```

- **@24↑ & @8↑** → ensemble 진짜 효과 (정답을 새로 발견 + 상위로 끌어올림)
- **@24↑ & @8≈** → 정답은 찾는데 fusion 이 상위로 못 올림 (RRF 튜닝 문제)
- **@24≈**        → 채널 무효 (정답을 아예 새로 못 찾음)

`lift8`/`lift24` 가 ensemble − baseline 차이입니다.

---

## 3. 공통 본체 — Chamfer matching (세 채널이 공유)

canny/scharr/orient 의 차이를 이해하려면, 먼저 셋이 **공유하는** 매칭 본체를 알아야 합니다.
([기초는 `match_algorithms_intro.md` §1 — 여기선 요점만.])

```
template ─[엣지 검출]→ edge map (0/255 윤곽선)
frame    ─[엣지 검출]→ edge map ─[distance transform]→ DT
                                    DT[y,x] = 그 픽셀에서 가장 가까운 엣지까지의 거리
                                    (엣지 위=0, 빈 영역=큰 값)

매칭: template 엣지를 frame DT 위에서 슬라이딩
      → 각 위치에서 'template 엣지가 깔고 앉은 DT 값의 평균(mean_dt)'
      → 평균이 작다 = 윤곽이 잘 포개진다 = 좋은 매칭
점수: score = exp(-mean_dt / DT_TAU_PX)     # 0~1, 클수록 좋음 (DT_TAU_PX=10)
```

**왜 distance transform 인가?** 두 엣지 맵을 직접 XOR 하면 1픽셀만 어긋나도 0점(너무 깐깐).
DT 는 "거의 맞으면 거의 좋은 점수"라는 부드러운 거리를 줘서, SEM 공정 변화로 윤곽이 살짝
흔들려도 견딥니다 — 메모리 `align_key_matching_constraint` 의 "픽셀 동일성 가정 불가" 구현.

**왜 `matchTemplate(TM_CCORR)` 인가?** template mask(엣지=1) 를 frame DT 와 correlation 하면
`Σ DT[엣지 위치]` 가 한 번에 나옵니다(엣지 수로 나누면 평균). FFT 기반이라 모든 위치를 동시에
계산 — 픽셀 루프보다 수백 배 빠름.

### 3-1. scales — 여러 배율로 매칭

template 과 frame 의 magnification 이 정확히 같지 않을 수 있어, template 을 여러 크기로
줄였다 늘렸다 하며 각각 매칭합니다.

- `DEFAULT_SCALES = (0.7, 0.85, 1.0, 1.2, 1.4)` — 일반 검색
- `COMPARE_SCALES = (0.6, 0.75, 0.85, 1.0)` — 정적 비교 (rcp/msr 배율 비슷 가정) ← **이 A/B 가 사용**
- `BROAD_SCALES = (0.15~0.5)` — 저배율 미니어처 검색

> ⚠️ **A/B 의 결정적 디테일**: baseline 과 ensemble 이 *같은 scale 밴드*를 써야 공정합니다.
> 안 그러면 "채널 차이"가 아니라 "scale 차이"를 재게 됩니다. 그래서 `proposer_recall_ab.py`
> `:132,134` 가 둘 다 `COMPARE_SCALES` 를 명시로 넘깁니다.

### 3-2. NMS — 각 scale 에서 후보 K개 뽑기

score map 에서 1등만 뽑으면 후보 1개뿐. K개가 필요하니 **NMS**(Non-Maximum Suppression)로:
최고점 기록 → 주변 반경 억제 → 다음 최고점 … 반복 → "서로 떨어진 봉우리 K개".

---

## 4. 세 채널 — 같은 본체, 다른 눈

이제 canny/scharr/orient 의 차이는 **"엣지 맵을 어떻게 만드느냐"** 단 하나입니다.

### C1 = canny (baseline)

`cv2.Canny(blurred, 60, 160)`. 밝기가 급변하는 곳을 0/255 윤곽선으로. 표준적·빠름.
**A/B 의 baseline** 이며 `_baseline_candidates()` 가 이것만 씁니다.

한계: 임계값에 민감하고, SEM/OM 처럼 대비가 흐릿하거나 밝기가 반전되면 윤곽을 놓치거나 끊깁니다.

### C2 = scharr (gradient magnitude)

`cv2.Scharr` 로 x/y 그래디언트 → magnitude `√(gx²+gy²)`. "엣지냐 아니냐"의 이진 판정 대신
**그래디언트 세기**를 봅니다.

핵심 트릭 — **density matching binarization**: Otsu 같은 자동 임계값 대신 "상위 r% 픽셀만
엣지"로 자르고, 이 `r` 을 **Canny 가 만든 엣지 밀도에 맞춥니다** (3~15% clamp).

> **왜 밀도를 맞추나?** chamfer 점수 `exp(-mean_dt/τ)` 는 엣지 개수에 민감합니다. C1 이 엣지 5%,
> C2 가 30% 면 mean_dt 스케일이 달라 채널 비교가 불공정. 밀도를 똑같이 맞추면 세 채널 점수가
> 같은 잣대 위에 놓입니다. Scharr 는 Sobel 보다 회전 대칭성이 좋아 방향 정확도가 높아, C3 의
> 방향 계산에도 재사용됩니다.

### C3 = orient (orientation-binned directional chamfer)

가장 정교한 채널. 엣지를 **그래디언트 방향**에 따라 8개 bin 으로 쪼갭니다 (0–180° 를 8등분,
각 엣지 픽셀을 자기 각도 bin 에 배정 → bin 별 엣지 맵 8장). 매칭할 때 **같은 방향 bin 끼리만**
chamfer 를 재고(가로는 가로끼리, 세로는 세로끼리) 엣지 개수로 가중평균합니다.

> **왜 방향을 나누나?** 일반 chamfer 는 "근처에 아무 엣지나 있으면" 점수를 줍니다(가로선이
> 세로선에 우연히 가까워도 매칭). 방향을 분리하면 "같은 방향 구조가 같은 위치에" 있어야만
> 점수가 나와 훨씬 까다롭고 정확합니다.
>
> **0–180° half-angle** (`% 180`): 방향만 보고 부호(밝→어두 vs 어두→밝)는 버립니다. SEM/OM 은
> 밝기 반전이 잦은데(메모리 `align_fail_modality_om_vs_sem`), 이렇게 하면 **polarity 불변** —
> 반전돼도 같은 구조로 인식합니다.
>
> **성능 함정의 역사**: 과거엔 scale 루프 *안에서* frame 의 8개 DT 를 매번 재계산해 orient 가
> 비용의 ~78% 를 먹고 대량 프레임에서 hang 처럼 보였습니다. 지금은 frame DT 가 scale 과
> 무관하다는 점을 이용해 루프 *밖에서 1회*만 계산(`_directional_context`) — 커밋 `deb76aa`/
> `2a924aa` 의 핵심.

---

## 5. fusion = RRF (Reciprocal Rank Fusion)

세 채널이 각자 후보 24개를 냈습니다. 점수를 그냥 더하면 안 됩니다 — C1 의 0.8 과 C3 의 0.8 은
스케일이 달라 의미가 다르니까요.

**RRF 는 점수 대신 순위만 씁니다:**

```
fused(후보) = Σ_채널  1 / (k0 + rank)        # k0 = 10
```

각 채널에서 1등이면 `1/11`, 2등이면 `1/12` … 기여. **여러 채널이 공통으로 상위에 올린 후보**가
합산점이 높아집니다. 같은 위치를 가리키는 후보(center 거리 ≤ match_radius, Chebyshev/L∞)는 한
클러스터로 묶고 대표는 가장 높은 chamfer score 멤버.

> **왜 점수가 아니라 순위?** RRF 는 스케일 제각각인 점수들을 정규화 없이 합치는 견고한 방법
> (정보검색에서 검증). 한 채널이 비정상적으로 큰 점수를 내도 순위로 환산되니 결과를 독점하지
> 못합니다 — 메모리 `ensemble_proposer_and_consensus_race` 의 recall@8 0.557→0.698 도약이 이 융합 덕.

### solo vs fused — attribution

스크립트는 fused 뿐 아니라 채널별 **solo** recall 도 따로 잽니다. "어느 채널이 정답을 단독으로
잡았나"를 보기 위함 — 만약 `solo:orient` 가 `ensemble` 만큼 좋으면 나머지 두 채널은 비용만
먹는 셈이고, 거꾸로 fused 가 모든 solo 보다 높으면 융합이 실제로 가치를 더한 것입니다.

---

## 6. 정답 판정 (membership) 과 누락 사유

"정답을 담았다"의 정의:

```python
ax, ay = 후보_xy + (dx, dy)                  # align_offset 보정
정답 = hypot(ax-gt_x, ay-gt_y) <= GT_TOL_NORM * short
```

- **GT**: `cond.txt` 의 crosshair 좌표 = 엔지니어가 표시한 진짜 align point
  (메모리 `align_cond_files_and_coords`).
- **align_offset (dx,dy)**: template 흰 박스 중심 ≠ align point 라 보정 필요
  (메모리 `rcp_white_box_unique_area`).
- **GT_TOL_NORM = 0.20**: template 짧은 변의 20% 이내면 정답. 픽셀 절대값이 아니라 **상대값**이라
  해상도가 바뀌어도 일관 → 그래서 `PROPOSER_MAX_DIM` 으로 이미지를 줄여도 recall 보존.

### drop 카운터 — "왜 frame 이 빠졌나"

`drop` dict 가 누락 frame 을 사유별로 셉니다. 오피스에서 "S 데이터가 희박해서"인지 "라우팅/cond
가 frame 을 버려서"인지 구분하는 진단용입니다.

| 사유 | 의미 |
|------|------|
| `no_box_tpl`   | template 빌드 실패(박스 못 만듦) |
| `non_S`        | tool 라벨이 S(success)가 아님 — E/? 프레임 제외 (메모리 `e_images_no_crosshair`) |
| `routing_miss` | modality(OM/SEM) 라우팅 실패 (메모리 `align_fail_modality_om_vs_sem`) |
| `no_crosshair` | cond 에 십자(GT) 없음 |
| `load_failed`  | 이미지 로드 실패 |

> S 프레임만 쓰는 이유: E(fail) 프레임은 crosshair(=GT)가 없어 정답을 매길 수 없습니다. 단,
> S 라벨도 도구의 self-report 라 의심 대상(메모리 `doubt_s_labels`) — tool_label 은 metadata 일 뿐
> CV 입력에 넣지 않습니다.

---

## 7. 실행

```bash
# 오피스(golden 데이터 필요)
uv run python poc/workflow_2/proposer_recall_ab.py

# 옵션 (환경변수)
ALIGN_GOLDEN_ROOT=<경로>   # golden 데이터 루트 override
PROPOSER_MAX_DIM=2000      # frame 최대변 캡(속도↑, recall 보존). 0=끔(기본, 측정 순수)
```

결과는 `DEBUG_IMAGE_DIR/proposer_recall_ab/<timestamp>/summary.json` + 콘솔 표.
진행 heartbeat(25 frame 마다)로 "느린 것"과 "hang"을 구분합니다.

---

## 부록 — 코드 지도

| 개념 | 위치 |
|------|------|
| recall 측정, drop 카운트, A/B 루프 | `proposer_recall_ab.py` |
| baseline(canny only) 후보 | `proposer_recall_ab.py:_baseline_candidates` |
| chamfer 본체 (edge→DT→matchTemplate→exp) | `align_key_matcher.py:preprocess_for_matching`, `_chamfer_score_map_at_scale` |
| 멀티스케일 NMS 후보 수집 | `align_key_matcher.py:_collect_candidates`, `_extract_peaks` |
| scharr 채널 + 밀도 매칭 이진화 | `ensemble_proposer.py:_scharr_edges` |
| orient 채널 (8-bin directional) | `ensemble_proposer.py:_orientation_bin_edges`, `_directional_score_at_scale` |
| RRF 융합 | `ensemble_proposer.py:_rrf_fuse` |
| 3채널 엔트리포인트 | `ensemble_proposer.py:compute_ensemble_candidates` |
| scale 밴드 / 허용오차 상수 | `align_similarity.py:COMPARE_SCALES`, `GT_TOL_NORM` |
