# Deep Learning 활용 방안 — align key 탐색 (탐색 메모, 판단 보류)

> 상태: **보류(parked)**. 2026-06-01 대화에서 나온 아이디어를 기록만 한다. 결정/착수 아님.
> 성격: 옵션 정리 + 위험/전제 명시. 아래 "의견"은 참고용이며 채택된 방향이 아니다.
> 재논의: 사용자가 더 고민한 뒤 다시 꺼낸다.
> 관련: `poc/workflow_2/docs/study/cv/align_evaluation_metrics_intro.md`,
> `poc/workflow_2/align_point_correction.py`(`_detect_white_box`/`_inner_crop_for_box`),
> `poc/workflow_2/align_similarity.py`(center vs box 템플릿), consensus A/B 핸드오프 저널.

---

## 0. 한 줄 요약

1만 개 규모 recipe 의 rcp(IMAP0001=OM, IMAP0002=SEM)에는 엔지니어가 그린 흰 unique-area
box 가 있다. 이걸 **정밀 crop** 할 수 있으면 대규모 (template, 위치) 라벨 자산이 되어 DL 로
`align_img_from_msr` 에서 align key 위치 찾기를 도울 수 있다. **단, "box 정밀 crop" 이 모든
후속의 게이트**이고, DL 종류 선택과 데이터(msr 동반 확보 여부)가 효과를 좌우한다.

---

## 1. 출발 자산과 게이트

- **자산:** ~1만 recipe × {IMAP0001(OM), IMAP0002(SEM)} + 각 이미지의 흰 unique-area box.
  - box 는 엔지니어가 등록 시 이미지에 픽셀로 그려 넣음(흰색, 회색 SEM 위라 검출 강건).
  - **타깃은 box 가 아니라 이미지 정중앙**(= align point). box 는 보조 단서이고, 매칭해야 할
    것은 box *안쪽 shape*. box 흰 테두리는 msr 엔 없으므로 학습/매칭에서 artifact → 안쪽만 써야 함.
- **게이트(전제):** box 정밀 crop 이 안 되면 1만 개는 라벨 없는 이미지 더미. 모든 DL 갈래의
  공통 1단계.
  - crop 기계는 **이미 존재**: `align_point_correction.py` 의 `_detect_white_box`(top-hat→Otsu→
    contour, 면적/짧은변/종횡비/hollow-outline 필터), `_inner_crop_for_box`(stroke inset 회피),
    미검출 시 center crop fallback, `_draw_rcp_overlay` 시각화.
  - **미해결은 "구현" 이 아니라 "1만 규모 정확도 검증 + 임계 calibration"**: `RCP_BOX_*` 임계가
    전부 cold-start 추정치. 실데이터로 맞춰진 적 없음.
  - 공짜 검증 신호: ① overlay 육안 확인, ② 물리 제약(align point=이미지 중심 → box 중심이 중심에서
    크게 벗어나면 오검출 의심) → 자동 flag 가능.

---

## 2. DL 갈래 (문제 적합도 순) — 참고 의견

> ⚠️ 핵심 프레이밍: align key 위치 찾기는 **일반 object detection 이 아니라 recipe-specific
> one-shot detection**. key 모양이 recipe 마다 달라, 평범한 detector 는 처음 보는 recipe 의 key 를
> 못 찾는다. 이 구분이 방향을 가른다.

| # | 갈래 | 무엇 | 적합도 / 리스크 | 기존 설계와의 정합 |
|---|---|---|---|---|
| ① | Learned template matching / descriptor | (query=rcp box crop, search=msr frame)→위치 heatmap. SiamFC/OS2D, LoFTR/SuperGlue, DINOv2+correlation | **가장 적합·저리스크.** 처음 보는 recipe 도 one-shot. 현재 약점 `topk_not_rank1=0.179`(정답이 후보엔 있는데 1등 아님)를 직접 공략 | "좌표는 CV 가 결정" 원칙 유지(점수만 학습된 것으로 교체). MI reranker 자리 대체 |
| ② | Self-supervised domain backbone | 1만 box(+msr)로 SEM/OM 전용 feature embedding 학습 | 중간. drift(대비/밝기/형상) 강건성 데이터로 학습. Chamfer/ORB 보강 | 기존 matcher 의 feature 단계 교체 |
| ③ | Box detector | 1만 rcp 의 흰 box 자체 검출(YOLO 등) | localization 아님 = **게이트(crop 자동화)** 해결만. ①②의 전제 | `_detect_white_box` 휴리스틱의 학습형 대체 |

- 평범한 YOLO 로 "align key 클래스" 학습 → **실패 패턴**(미학습 recipe 일반화 안 됨). 피할 것.
- **라벨이 공짜**: S 프레임=align 성공(key 중앙)→positive 위치, E 프레임=틀린 위치→hard negative.
  1만 recipe × 다수 프레임 = 대규모 weakly-labeled set. DL 시도의 가장 강한 근거.

---

## 3. 결정적 변수 — 데이터 동반 확보 여부

| 보유 자산 | 학습 가능 | 못 하는 것 |
|---|---|---|
| **rcp 만 1만** (IMAP0001/0002 + box) | ③ box detector, ② backbone(template 표현) | ① supervised localizer(=msr 위치 정답 쌍 없음) |
| **rcp 1만 + 각 recipe 의 msr(S/E)** | ①②③ 전부. ceiling 돌파 정공법 | — |

- **열린 질문(미확정):** 1만 recipe 에 대해 `align_img_from_msr`(S/E) 도 대량으로 받을 수 있나?
  - 예 → ① learned localizer/reranker 학습 가능 → 잔여 ~28% proposer recall 한계(현재 천장 0.718)
    돌파 가능. 핸드오프의 "2차 proposer" 자리와 동일.
  - 아니오(rcp 만) → 1만 개는 ③ crop 자동화 + ② backbone 에 쓰고, localizer 는 소수 office S/E
    fine-tune. 가치는 있으나 ceiling 상승폭은 작음.

---

## 4. 만약 진행한다면 — 권장 시퀀스 (참고)

1. **box 정밀 crop 검증/calibration (게이트, 무조건 선행)**
   - 기존 `_detect_white_box` 를 1만(또는 office 표본)에 돌려 **검출률·실패유형 측정** + 임계 보정.
   - 산출물 예:
     ```
     box_crop_audit/
     ├─ ok/        <recipe>__OM.jpg / __SEM.jpg   # 검출+overlay 자동 통과
     ├─ flagged/   <recipe>__*.jpg                # 중심 이탈/미검출/fill 비정상
     └─ report.json  { detect_rate, fail_reasons{...}, flagged:[...] }
     ```
   - 이 단계가 동시에 DL 데이터셋 1차 라벨링.
2. **다음 한 발: full DL detection 말고 ① learned descriptor/reranker** (저리스크, 기존 CV 권위에
   그대로 삽입, MI reranker 로 풀려던 18% 직접 공략).
3. **통하면 end-to-end one-shot localizer 로 확장**(잔여 28% miss 돌파).

---

## 5. 위험 / 주의 (재논의 시 검토)

- **도메인 갭 rcp→msr**: rcp 는 고배율·클린, msr 는 live·노이즈·대비/밝기/배율 상이. 매칭이 어려운
  근본 이유. DL 은 *실 msr 외형으로 학습할 때만* 이걸 메움.
- **누설 방지**: train/test 를 **recipe 단위로 분리**(프레임 단위 금지). align_similarity 의 LOO 규율과 동일.
- **OM vs SEM** 은 다른 modality → 별도 모델 또는 modality-conditioned.
- **box 테두리 제외 crop** 필수(흰 stroke 학습 금지).
- **비용/인프라**: full DL detection 은 라벨 파이프라인+학습 인프라 필요. ① descriptor 는 상대적 저비용.
- **설계 원칙 충돌 점검**: 프로젝트 규칙 "OpenCV 가 좌표/점수, VLM 은 영역/판단만". 학습된 매처는
  여전히 "점수 내는 CV" 로 볼 수 있어 원칙 유지 가능하나, end-to-end detection 은 이 경계 재검토 필요.

---

## 6. 재논의 트리거

- 사용자가 더 고민 후 방향 결정 시.
- 또는 (a) msr 대량 동반 확보 가능 여부 확인되거나, (b) box crop audit 검출률 수치가 나오면
  — 둘 중 하나만 채워져도 §3 표의 분기가 확정되어 논의를 진전시킬 수 있음.

---

## 7. (관련 아이디어) align key 위치 재배치 = 예방 (parked)

> 2026-06-01 대화에서 파생. §1~6(DL)과 **같은 목표(align fail 감소)의 다른 층위**라 한 문서에 둔다.
> 판단 보류. 착수 아님.

### 7.1 핵심 구분 — 사후 회복 vs 사전 예방

| | 현재 workstream (재등록·workflow_2) | **위치 재배치 (이 절)** |
|---|---|---|
| 성격 | 사후(reactive) — align fail 후 *우리* 파이프라인이 회복 | **사전(preventive)** — *Tool* 이 fail 을 덜 내게 |
| 누구의 알고리즘 | workflow_2 의 CV 매처 | **장비(CD-SEM/VeritySEM)의 native align 알고리즘** |
| 효과 | fail 당 복구 시간↓ | **fail 발생 건수 자체↓** |
| 천장 | recall 0.718 (회복률) | 근본 원인 감소 (훨씬 높음) |

- 아이디어: 이미지가 degrade/약간 변형돼도 **장비 알고리즘이 계속 찾아낼 수 있는 위치**(변별력 ×
  degrade robustness × 시간 안정성)에 align key 를 두면 align fail 자체가 준다.
- 문제 분류: **align key site selection / optimization** (현재는 엔지니어가 수동으로 box 선택).
- **프로젝트 정합성:** CLAUDE.md 의 Project Purpose("CD-SEM recipe **setup** 자동화로 수동 recipe
  생성 대체")와 더 가깝다. workflow_1/2(회복)는 현재 focus 일 뿐, 위치 재배치는 *명시된 목적*과 정합 →
  곁가지가 아니라 본류일 수 있음.

### 7.2 재등록과 다른 점 (혼동 주의)

- **재등록(확정):** 같은 위치, template *외형*만 consensus 로 갱신. align point/box 안 바꿈. staleness 해결.
- **위치 재배치(이 절):** align point 를 더 robust-unique 한 곳으로 *옮겨* 재등록. **장비가 실제 이동하는
  물리 좌표 변경** = recipe re-teach. 전혀 다른 무게.
- 코드의 distinctiveness gate(`test_align_key_distinctiveness.py`)는 모호 매칭을 **reject 만** 함 —
  **더 나은 위치를 *제안*하진 않음**. 즉 site selection 은 미구현.

### 7.3 냉정한 crux (진전 전 반드시 검토)

1. **"Tool 알고리즘"은 벤더 black box.** 우리가 고르는 건 *위치*뿐, *알고리즘*은 모름. 우리 매처
   distinctiveness 가 높아도 장비 알고리즘에 robust 한지는 별개 → **반드시 장비 실제 성공/실패로 검증**.
   우리 점수는 proxy.
   - **우회로:** 내부를 몰라도 **과거 fail 이력**으로 "어떤 위치/패턴이 자주 fail 하나"를 학습 →
     알고리즘 무지 상태에서도 경험적 proxy 가능. ← §1~6 DL 의 실제 응용처가 여기.
2. **후보 위치는 임의가 아님.** "가장 unique 한 픽셀"이 아니라 **모든 웨이퍼/die 에 재현되는 실제 물리
   구조**여야 함. FOV 도달 영역 + 칩 레이아웃 제약. 변별력만 높고 공정변동에 흔들리면 오히려 더 fail.
3. **재배치 = governed recipe re-teach.** reference 이미지 교체(재등록)와 달리 엔지니어 승인·재검증 수반
   → 현실적 도구 형태는 "자동 재등록"보다 **"setup 시 더 나은 align site 추천"**일 가능성.

### 7.4 DL 과의 합류점

- 1만 recipe × **과거 fail/success 이력** → "align site 품질 점수기"(변별력 × degrade robustness ×
  시간 안정성) 학습. → §1~6 의 데이터-엔진이 곧 이 절의 응용. 두 메모는 **같은 목표(예방)의 엔진과 응용**.

### 7.5 진전 게이트 (답이 갈리는 두 질문)

1. **과거 align fail 이력**(recipe/위치별 fail 빈도)을 받을 수 있나? → 있으면 알고리즘 무지에도 proxy
   학습·검증 가능. 없으면 "추측한 robust 위치"를 검증할 길이 없어 막힘.
2. 재배치 위치를 **실제 장비에 등록해 fail 률 before/after 측정**할 권한·절차가 있나? → 없으면 추천까지만.
