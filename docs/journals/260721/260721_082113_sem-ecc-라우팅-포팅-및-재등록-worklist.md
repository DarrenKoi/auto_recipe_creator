# SEM=ecc / OM=mind 라우팅 포팅 + 재등록 worklist 진입

- 날짜: 2026-07-21 08:21
- 브랜치: main
- 관련 커밋: `e36bfa2` `7cf686a` `4cf4d78` `0da674b`
- 선행 저널: `260720_163416_mind-rerank-포팅-및-커버리지-집계.md` (MIND 포팅 + pm/re 커버리지)

---

## 0. 이 세션의 한 줄 요약

registration verifier A/B 를 오피스에서 두 차례 더 돌려(4·5차) modality별 arm 성적을
받아가며, **SEM 에서는 ecc, OM 에서는 mind 를 쓰는 modality-aware 재정렬을 workflow_3
프로덕션에 포팅**했다. 이로써 verifier(순위 교정) 축은 사실상 종료됐고, 남은 개선 몫
(pm=39, 87%)을 겨냥해 **재등록 후보 worklist** 를 registration 벤치에 추가했다.

세션 말미 사용자 지시로 방향 전환: **재등록 worklist 오피스 대조는 보류**하고, 이제
**workflow_3 실제 구동의 robustness** 로 넘어간다(§3).

---

## 1. 진행 사항 (시간 순)

### 1.1 SEM arm 실측 → ecc 가 SEM 최고임을 발견

3차까지 mind 만 포팅했는데, 커버리지 집계에서 SEM oracle(0.817) > prod_mind SEM(0.759)
gap 이 남았다. 오피스에서 SEM arm 행을 받아 원인 확인:

- SEM 단독: **ecc 0.79** > prod_mind 0.759 > mind 0.743 (b0 0.702)
- ecc 는 OM 에서 유해(-0.084)라 전체선 churn 처리됐으나, **modality 는 PM box 로 판별
  가능**하므로 SEM 한정으로 켜면 함정을 피한다.

### 1.2 SEM-aware ecc 라우팅 A/B (route3 / route2) — `e36bfa2`

modality-aware 결합 의사-arm 을 벤치에 추가:
- route3 = SEM `sel⊕mind⊕ecc` / OM `sel⊕mind`
- route2 = SEM `sel⊕ecc` / OM `sel⊕mind`

4차 실측 결과:
```
route3 0.820 [om+0.021/sem+0.062] CI[+0.002,+0.084] p/r=27/12  ← prod_mind(0.817) 이김
route2 0.814 [om+0.021/sem+0.052]                              ← route3 보다 나쁨(mind 도 필요)
```
route3 가 prod_mind 를 통계적으로 유의하게(CI 하한 첫 0 돌파) 이겼고 OM 무손실 확인.

### 1.3 결합이 ecc 를 희석함을 발견 → route_sw — `7cf686a`

**그런데 route3 의 SEM(0.764) < ecc 단독 SEM(0.79).** RRF 가 SEM 을 압도하는 ecc 를
약한 심판(sel/mind)과 섞어 오히려 희석시킨 것(2차 fuse=±0 재현). → 결합이 아니라
**완전 전환** 의사-arm 추가:
- route_sw = SEM `ecc 단독 순위` / OM `prod_mind`

### 1.4 route_sw 판정 통과 → workflow_3 포팅 — `4cf4d78`

5차 실측:
```
route_sw 0.826 (SEM ref=0.775, OM ref=0.895) > route3 0.820 > prod_mind 0.817
OM d_ref +0.021(무손실), SEM d_ref +0.073 (p/r 22/8)
raw == ref (OM 0.895=0.895, SEM 0.775=0.775)  ← 이득 전부 순위 전환, ecc shift 기여 0
```
판정 통과(route_sw > route3, OM 무손실, 순위-only) → **workflow_3 engine 에 포팅**:
- OM: sel ⊕ mind RRF (기존)
- SEM: ecc 단독 순위로 전환(결합 아님)

교훈: **한 심판이 특정 국면을 지배하면 결합(RRF) 대신 전환** 이 낫다.

### 1.5 verifier 축 종료 → 재등록 worklist 진입 — `0da674b`

커버리지 pm=39(87%)는 후보에 GT 가 아예 없어 어떤 재정렬로도 못 고치는 재등록 몫.
registration 벤치에 recipe 단위 pm 집계 + 재등록 후보 worklist 를 추가(옵션 A, 사용자
선택). 이 벤치는 프로덕션과 동일 매칭(consensus + SEM=ecc/OM=mind)을 쓰므로 "지금
프로덕션이 실제로 못 잡는" 재등록 1순위 명단이다.

조사(Explore subagent)로 기존 `golden_reregister_report_cond.py`(Phase 1-3)와의 관계 확인:
독립 신호(reregister=rcp 템플릿 STRONG tier, registration=consensus LOO pm). 두 명단
대조가 후속 단계.

---

## 2. 수정 내용 (파일별)

**workflow_3 (프로덕션 포팅)**
- `poc/workflow_3/align/matching/mind_rerank.py` — `ecc_score`(cc)/`ecc_rerank_order`/
  `is_sem_template`/`ecc_rerank_enabled` 추가(registration_lab bit-parity). 모듈 docstring 을
  modality-aware 로 개정.
- `poc/workflow_3/align/matching/engine.py` — `compute_align_key_score_ensemble` 의 selection 을
  `template.key_type=='sem' → ecc 단독 순위, else → sel⊕mind RRF` 로 분기.
- `poc/workflow_3/align/matching/test_mind_rerank.py` — ecc/SEM 경로 + OM/SEM e2e 테스트
  추가(8/8).
- `CLAUDE.md` — mind_rerank 모듈 설명을 modality-aware 로 갱신.

**workflow_2 (벤치 A/B + 재등록 worklist)**
- `poc/workflow_2/golden_registration_eval_cond.py`:
  - route3/route2 의사-arm(`e36bfa2`), route_sw(`7cf686a`)
  - `_RegAccum.recipe_bucket` + `_reregister_worklist` + `_print_reregister_worklist` +
    `reregister_candidates.json` 출력(`0da674b`)
- `poc/workflow_2/test_golden_registration_eval_cond.py` — route/worklist 테스트로 16/16 확장.

**검증**: workflow_3 smoke 8/8, engine 10/10, ensemble 19, correction 13/13, lab 24/24;
workflow_2 driver pytest 16/16.

---

## 3. 다음 단계 — workflow_3 robustness (사용자 지시로 방향 전환)

세션 말미 사용자 지시:
> "이 테스트는 나중에 이어서 하자 이제는 workflow_3로 가서 실제 구동이 문제 없게
> 돌아가도록 workflow를 robust하게 만들어야 해"

**보류(나중에 이어서)**:
- 재등록 worklist 오피스 대조(registration pm 명단 vs reregister STRONG 명단 교집합/차집합).
- registration 벤치 4차 이후 추가 A/B.

**새 무게중심 — workflow_3 실구동 robustness**:
CV 매칭 정확도(verifier)는 소진에 가까우니, 이제 **실제 루프가 오피스에서 안정적으로
도는지**가 초점. 다음 세션에서 착수할 후보 영역(사용자와 우선순위 확인 필요):
1. **실패/예외 경로 점검** — 알람→접속→캡처→보정→teardown 루프의 각 단계에서 예외가
   나도 `try/finally` teardown(툴 닫기/녹화 중지/팝업 백스톱)이 항상 보장되는지, 부분
   실패 시 다음 알람 대기로 안전 복귀하는지.
2. **포팅한 rerank 의 실구동 확인** — 다음 SEM/OM align fail 에서 ecc/mind 분기가
   로그상 의도대로 타는지, ecc(findTransformECC) 예외/타임아웃이 루프를 멈추지 않는지
   (킬스위치 ALIGN_FAIL_ECC_RERANK/MIND_RERANK 폴백 검증).
3. **office 어댑터/경로 견고성** — 데이터 경로 mismatch, office 모듈 부재, RCS 창 탐색
   실패, 점유 팝업 등 알려진 취약점의 방어 상태 재점검(check-only 모니터 startup 리포트).
4. **장시간 무인 구동** — keep-awake, foreground 탈취, 엔지니어 수동조작 중 녹화 등
   real-time 루프의 장시간 안정성.

→ **다음 세션 첫 작업: 위 1~4 중 어디부터 볼지 사용자와 우선순위 합의 후 진행.**

---

## 4. 메모리 업데이트

- `project_registration_verifier_lab.md` — route_sw 포팅 완료 + 재등록 worklist 진입 기록
  갱신 완료.
- `MEMORY.md` — 인덱스 줄을 "modality-aware rerank 포팅 완료(OM=mind/SEM=ecc)"로 갱신 완료.
- workflow_3 robustness 는 다음 세션에서 착수 시 별도 project 메모리 검토(현재는 이 저널이
  방향 보관).
