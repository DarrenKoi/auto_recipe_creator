# consensus A/B 점수 해석 + blur 가드 의미·확인 보류

날짜: 2026-06-09 04:58
대상: `poc/workflow_2/golden_consensus_eval_cond.py` 의 어제(260608) A/B 결과 해석 및 blur 가드

---

## 0. 한 줄 요약

어제 검증된 consensus A/B 수치(in_topk 0.434→0.876, rank1 0.318→0.764)를 해석하고 **blur 가드의
의미·필요성**을 정리했다. 결론: blur 가드는 **eval 에선 보조(rank1 이 이미 blur 아님을 증명),
production 에선 필수(recipe 별 ground-truth 없이 bad consensus 를 거르는 유일한 런타임 신호).**
단 **어제 blur 수치는 어디에도 보존돼 있지 않고 이 머신엔 golden 데이터가 없어 재확인 보류** —
데이터 붙는 날 1회 재실행으로 확정. 코드 변경 없음(해석/문서화 세션).

---

## 1. A/B 점수 해석 (어제 260608 run)

| 지표 | rcp(center, baseline) | consensus | lift |
|---|---|---|---|
| **in_topk** | 0.434 | **0.876** | **+0.442** |
| **rank1** | 0.318 | **0.764** | **+0.446** |

조건: `recipes=134`, `S_loo=403`, `min_s=3`, co-registration ON.

- **in_topk = PROPOSER_WALL 그 자체.** 정답(align point)이 matcher top-N 후보에 들어온 비율(LOO).
  0.434→0.876 = "예전엔 절반 넘게 후보에서 정답이 빠졌는데, consensus 로 거의 9할이 들어온다" → 벽이 깨짐.
- **rank1 = 후보 중 정답이 1등인 비율.** 0.318→0.764.
- **rank1 동반 상승이 핵심 신뢰 근거**: median 이 흐린 generic blob 이면 in_topk 만 오르고 rank1 은
  안 올라야 한다. rank1 이 +0.446 으로 같이 올랐으니 median 이 **진짜 변별력**을 가졌다는 증거.
- baseline rcp 0.434 가 localization eval `gt_in_topk` 0.433 과 독립 재현 → 측정 파이프라인 정상.

판정: lift +0.442 ≫ 채택 임계(+0.05). **CONSENSUS 채택 권장.** (저널 260608_163302 와 동일 결론)

## 2. blur 가드 — 무엇을 재나

출력 두 줄(`golden_consensus_eval_cond.py:351-353`)은 **비율**이다:

```
edge_ratio_to_S = (consensus 의 Canny edge 픽셀 비율) / (개별 S crop 들의 edge 비율 median)
lap_ratio_to_S  = (consensus 의 Laplacian 분산)        / (개별 S crop 들의 Laplacian 분산 median)
```

근거: `align_similarity.py:808-815`. `_edge_density`=Canny(60,160) edge 픽셀 비율(matcher 와 동일 임계),
`_lap_var`=Laplacian 분산(선명도). **분모가 "consensus 를 만든 재료 S 한 장"** 이라 apples-to-apples.

- **비율 ≈ 1.0** → median 인데도 개별 프레임만큼 선명 = crosshair 정렬이 잘 맞아 윤곽이 또렷이 겹침. 좋음.
- **비율 낮음(예: 0.5)** → 평균내며 윤곽 뭉개짐(정렬 어긋남 → ghosting/blur). 나쁨.

겨냥하는 실패 모드: consensus 의 **유일한 구조적 위험** = "여러 장 median 이 흐린 blob 됨". crosshair
정렬이 sub-pixel 로 안 맞으면 ghost 가 되는데 edge/lap 비율이 그 손실량을 정량화한다.

## 3. blur 가드, 꼭 필요한가 — 맥락별

| 맥락 | 필요성 | 이유 |
|---|---|---|
| **eval (A/B)** | 보조(없어도 결론 섬) | rank1 0.764 가 이미 "blur 아님"을 증명 → 다른 각도 확인사살 |
| **production** | **필수** | recipe 한 개의 rank1 을 알 방법이 없음(라벨 없음). blur 비율이 런타임에 "이 consensus 믿어도 되나"를 판단하는 **유일하게 싼 신호** → 낮으면 그 recipe 만 rcp 폴백 |

productization 폴백 임계(저널 260608_163302 4-A #7): `edge_ratio < 0.70` **또는** `lap_ratio < 0.50`
면 consensus 버리고 rcp center 로 폴백.

판정 기준:
- `edge_ratio ≥ 0.70` **그리고** `lap_ratio ≥ 0.50` → median 이 선명 = **min_s=3 caveat 완전 해소**,
  프로덕션 임계를 이 값 근거로 확정.
- 둘 중 하나라도 미만 → in_topk 가 흐림 덕에 부풀었을 가능성 점검. **단 rank1 0.764 가 이미 강하게
  반증** → 만약 비율이 낮게 나오면 "흐린데 rank1 높음"이라는 모순이라 오히려 측정 파이프라인을 의심해야 함.

예상: edge/lap 둘 다 ≥0.7 (rank1 이 강하게 시사). 그래서 이 확인은 결론을 바꾸기보다 **결론의 일관성
교차검증** 용도.

## 4. 확인 보류 — blur 수치가 어디에도 없음

어제 오피스 run 의 blur 숫자를 찾으려 했으나 보존돼 있지 않음:

- ❌ `summary.json` — 출력 디렉터리 `debug_images/golden_consensus_eval_cond/` 가 repo 에 없음(debug 산출물, git 미추적)
- ❌ 로그 — consensus A/B stdout 이 `logs/`·`*.log` 어디에도 없음
- ❌ 저널 — 260608_163302 에 "blur 수치 아직 못 봄"으로 명시(4-D)
- ❌ golden 데이터 — 이 머신(`DESKTOP-P0CU31O`)엔 `align_images_golden/` 도 `align_images/` 도 없음

사용자 확인(260609): **지금은 데이터 없음.** → 데이터 붙는 날 재실행으로 확정.

## 5. 데이터 붙는 날 — 실행 카드 (코드 변경 불필요)

```powershell
# golden 데이터를 기본 위치 poc/workflow_1/align_images_golden/ 에 두거나
$env:ALIGN_GOLDEN_ROOT = "D:\...\align_images_golden"   # 다른 곳이면
uv run python poc/workflow_2/golden_consensus_eval_cond.py
```

stdout 맨 아래 블록(`golden_consensus_eval_cond.py:340-357`)의 이 줄만 읽으면 됨:

```
  blur 가드(낮으면 median 흐림): edge_ratio_to_S=____  lap_ratio_to_S=____
```

영구 사본: `debug_images/golden_consensus_eval_cond/<ts>/summary.json` 의
`cons_edge_density_ratio_to_S_median` / `cons_lap_var_ratio_to_S_median`.

→ 2~3절 판정 기준으로 해석. ≥0.7/≥0.5 이면 productization(저널 260608_163302 4절) 바로 진행.

---

## 6. 다음 단계

- (대기) golden 데이터 확보 → blur 가드 1회 재확인(위 실행 카드).
- (이어서) productization: 검증된 consensus 를 matcher/`live_align_search` 의 실제 등록 template 로 승격
  — 통합점·TDD 순서는 저널 260608_163302 4절 참조.
