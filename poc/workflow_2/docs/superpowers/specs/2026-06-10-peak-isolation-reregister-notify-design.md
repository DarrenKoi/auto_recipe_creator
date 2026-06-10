# peak-isolation 모호 키 → 엔지니어 재등록 알림 (workflow_3 포팅 spec)

> **상태: ✅ 구현 완료 — 2026-06-11, commits `456f65c`..`5dcdaca` (origin/main).**
> A(CorrectionOutcome 모호도 surface + `_with_key_ambiguity` stamp) · B(`Workflow3Settings.
> reregister_second_ratio_threshold` 0.98 + env `ALIGN_FAIL_REREGISTER_RATIO`) ·
> C(`notify` 재등록 권고 + `corrected_but_ambiguous` audit) · D(`cycle` 배선) 전부 랜딩.
> 회귀 green: notify 7/7 · align_fail_correct 9/9 · align_key_match 10/10 ·
> consensus_gather 8/8 · success_gather 6/6. 보정 동작·0.94 visibility 게이트 불변(read-only).
> 미구현(의도된 범위 밖, §2·§8): fail-frame τ 재보정(consensus-gather fail 데이터 축적 후) ·
> act/abstain 3-way 결정(상위 production-trust plan).

날짜: 2026-06-10
대상: lab 검증된 **match-time peak-isolation 모호도**(`second_ratio`)를 workflow_3 의 실시간
보정 경로에 노출해, 만성적으로 모호한 align key 를 엔지니어가 **재등록**하도록 알린다.
선행 검증: `poc/workflow_2/ensemble_lab.py`(peak-isolation 변형) + `golden_consensus_eval_cond.py`
(office digest); 관련: `2026-06-10-production-trust-consensus-cache-and-decision-layer.md`
(이 spec 의 상위 비전 — act/abstain 3-way 결정. **본 spec 은 그 부분집합이 아니라 read-only
알림 1차 포팅만** 다룬다).

> **핵심 전제(중요):** 포팅할 신호는 *새로 만드는 것이 아니다*. workflow_3 의 matcher 가 이미
> `AlignKeyMatchResult.second_ratio`(= 2nd/best chamfer, "1.0 에 가까울수록 모호")를 fail 시점에
> 계산하고 있다(2026-05-29 도입). lab 작업(B)이 한 일은 그 신호를 **검증**(AUC 0.91)하고
> **임계를 보정**(tau\* 0.9825)한 것. 따라서 이 포팅은 *이미 계산되고 버려지는 값을 엔지니어
> 경로로 끌어올리는 surface + calibrate* 작업이며, 새 CV 가 아니다.

---

## 1. 배경·목표

**문제.** 단일 rcp 등록 key 가 공정 드리프트로 모호해지면(반복 패턴/대칭 → 유일 위치 없음)
matcher 의 점수면이 평평해져(top1≈top2) align fail 이 재발한다. 이런 키는 *재등록*이 답이지만,
어느 키가 만성적으로 모호한지 production 에서 가릴 신호가 없었다.

**검증(lab, golden consensus S-LOO, office digest).**
- **template-내재 축은 죽음**: periodicity(자기상관) 는 어떤 기하 정제(선형/window/lag)로도
  miss 예측 AUC 천장 ≈ 0.61. "닮은 구조가 있다"는 miss 의 필요조건이지 충분조건이 아님.
- **match-time peak-isolation 은 강함**: `second_ratio`(= 2nd/best chamfer) 의 per-point
  miss 예측 **AUC 0.9108** (n=403, miss 58/hit 345; mean miss 0.9915 / hit 0.8851;
  Youden tau\* 0.9825, TPR 0.9655 / FPR 0.2348 → recall 97% @ flag 34%, precision ~41%).
- **형태 robust**: 변형 ablation 에서 chamfer top2-family(ratio_top2 0.9105 · margin_abs 0.9236 ·
  median_rest 0.8994)는 AUC 표준오차(~0.036, n_miss=58) 안에서 동률. `ncc_ratio_top2` 만
  0.7198 로 확연히 약함 → 도메인 제약(픽셀 동일성 NCC 금지) 재확인, C4-as-NCC 형태 기각.
  → 포팅 형태 = **scale-free 인 `second_ratio`(=ratio_top2)**. 이미 workflow_3 가 쓰는 값.

**목표.** fail 시점에 이미 계산된 `result.second_ratio` 를 `CorrectionOutcome` 에 실어,
notify 경로가 **모호 키를 엔지니어에게 재등록 권장**으로 알린다. 보정 동작·기존 게이트는 불변.

**성공 기준.** (1) 실패 cube 알림에 `second_ratio` 와 임계 초과 시 "재등록 권장" 한 줄이 포함된다.
(2) 보정 성공이지만 모호한 키는 `work2.log` audit 트레일에 남는다(cube spam 없음). (3) 보정
동작(primary/fallback)·visibility 게이트(0.94)·기존 호출부는 바이트 단위로 불변. (4) 합성 테스트로
세 경로(실패+모호 / 성공+모호 / 데이터 없음) 검증.

---

## 2. 범위·불변 원칙

**포함:** `CorrectionOutcome` 에 모호도 필드 surface, `Workflow3Settings` 재등록 임계(env override),
`notify.py` 의 알림/audit 에리치, `monitor/cycle.py` 배선, 합성 테스트.

**비목표(YAGNI — 명시적 범위 밖):**
- 재등록 worklist/큐/데이터 경로 (이번엔 알림·로그까지만; 큐는 별도 결정에서 다룸).
- visibility 게이트(`MatchPolicy.max_second_ratio = 0.94`) 또는 어떤 correction 동작 변경.
- act/abstain 3-way 결정(상위 production-trust plan 의 영역) — 본 spec 은 **read-only 알림만**.
- 임계의 fail-frame 재보정 (아래 §7; consensus-gather fail 데이터가 쌓인 뒤 별도 후속).

**불변 제약(CLAUDE.md):** CLI 인자 금지 · Korean docstring · `[INFO]/[ERROR]/[WARNING]` print
(+ audit 는 `log_work2_event`) · `from __future__` 금지 · 절대 임포트 · `print()` 내 em-dash(U+2014)
금지(office cp949) · **workflow_3 는 legacy(wf1/wf2) import 금지**.

---

## 3. 설계 근거 — 측정과 정책의 분리

peak-isolation 은 *측정*(CV)이고 임계는 *정책*(monitor)이다 — 프로젝트 핵심 규칙 "OpenCV 가
점수를 내고, 결정은 다른 데서"와 일치. 그래서:

- **채택:** CV 층(`align_fail_correct`)은 raw `second_ratio` 를 `CorrectionOutcome` 에 *기록만*
  한다. monitor 층(`notify`)이 설정 가능한 임계를 적용해 "재등록 권장"을 판정한다.
- **기각 (i):** `AlignKeyMatchResult` 를 notify 에 그대로 넘김 → monitor 가 matcher 타입에 결합.
- **기각 (ii):** 권장 판정을 CV 층에서 계산 → 튜닝 가능한 정책 임계가 CV 에 묻히고, 목적이
  다른 0.94 visibility 게이트와 혼동.

세 가지 사용자 확정 결정:
1. **범위 = 알림 에리치만**(read-only, 동작·0.94 게이트 불변).
2. **위치 = 실패 cube 알림 + corrected-but-ambiguous 는 work2.log audit**(성공 시 cube spam 없음).
3. **임계 = `Workflow3Settings` 설정값**(기본 ≈0.98, tau\* 유래), raw `second_ratio` 항상 표기,
   초과 시 "재등록 권장" 강조. 0.94 와 **별개**. fail-frame 재보정 대상.

---

## 4. 컴포넌트별 변경

### A. CV 층 — `poc/workflow_3/vision/align_fail_correct.py`
- `CorrectionOutcome`: 옵션 필드 추가 — `second_ratio: float | None = None`,
  `score_gap: float | None = None`, `distinctive: bool = True`.
  (`AlignKeyMatchResult` 의 옵션-필드 패턴과 동일; 기존 필드/의미 불변 → 하위호환.)
- `correct_align_fail`: `result` 계산 이후의 각 outcome 반환부에서 작은 헬퍼
  `_with_key_ambiguity(outcome, result)` 로 세 필드를 stamp. match 이전 반환(`no_assets`)은
  기본값 유지(`second_ratio=None`, `distinctive=True`). `key_visibility_gate`·`max_second_ratio`
  (0.94)·decision 경로는 **무수정**(read-only surface).

### B. 설정 — `poc/workflow_3/config.py` (`Workflow3Settings`)
- 추가: `reregister_second_ratio_threshold: float = 0.98`.
  주석: tau\*(S-LOO golden 보정, AUC 0.91) 유래 · **fail-frame 재보정 대상** · 0.94 visibility
  게이트와 별개. env override `ALIGN_FAIL_REREGISTER_RATIO`(legacy env-name 패턴, `load_workflow3_settings`).

### C. monitor 층 — `poc/workflow_3/monitor/notify.py`
- `build_outcome_summary(outcome, *, reregister_ratio_threshold: float | None = None, ...)`:
  `second_ratio` 가 있으면 `second_ratio={x:.3f}` 추가; 임계가 주어지고 `second_ratio > 임계` 면
  `재등록 권장(모호 키)` 한 줄 추가(em-dash 없이). 임계 `None`(구 호출부) → 권장 줄 skip.
- `notify_correction_outcome(..., reregister_ratio_threshold: float | None = None)`:
  - **실패**(status != "corrected"): summary 에 이미 모호도 줄이 포함 → 기존 cube 발송에 실려 나감.
  - **corrected**: 모호(`second_ratio is not None and > 임계`)면 `log_work2_event(message=
    "corrected_but_ambiguous", second_ratio=..., recipe_id=..., eqp_id=...)` audit 만(cube 없음);
    아니면 기존 동작("corrected_no_notify").
  - 임계 None/`second_ratio` None 비교는 모두 guard.

### D. 배선 — `poc/workflow_3/monitor/cycle.py`
- `notify_correction_outcome(...)` 호출에 `reregister_ratio_threshold=settings.
  reregister_second_ratio_threshold` 전달. (build_outcome_summary 도 동일 임계 사용.)

---

## 5. 데이터 흐름

```
fail frame
  → compute_align_key_score_ensemble  (이미 second_ratio/score_gap/distinctive 계산)
  → correct_align_fail: _with_key_ambiguity 로 CorrectionOutcome 에 stamp
  → cycle: notify_correction_outcome(outcome, threshold=settings.reregister_second_ratio_threshold)
      ├─ status != corrected  → cube 알림 summary 에 "second_ratio=X · 재등록 권장"
      └─ status == corrected & second_ratio>임계 → work2.log "corrected_but_ambiguous"(cube 없음)
```

---

## 6. 에러 처리·하위호환

- `second_ratio is None`(후보 0개 / `no_assets` / 기존 호출부 / 구버전 result) → 모호도 줄·권장
  없음, **오늘과 완전히 동일** 동작. `distinctive` 기본 `True` 라 데이터 결손이 false-flag 되지 않음.
- 모든 임계 비교는 `None` guard. notify 가 임계 인자를 못 받으면(구 호출부) 권장 판정 skip(기본 동작).
- `CorrectionOutcome` 신규 필드는 전부 기본값 보유 → 기존 5개 생성부·테스트 무변경 통과.

---

## 7. 테스트 (합성, office 데이터 불요; `test_align_fail_correct.py` 패턴)

- `CorrectionOutcome` 가 신규 필드를 보유하고 기본값으로 기존 생성부가 깨지지 않는다.
- `_with_key_ambiguity(outcome, result)` 가 `result` 의 `second_ratio/score_gap/distinctive` 를 stamp.
- `build_outcome_summary`: `second_ratio > 임계` 면 "재등록 권장" 포함, 이하면 미포함, `None` 이면
  모호도 줄 자체 없음.
- `notify_correction_outcome`(office notify + `log_work2_event` mock):
  corrected+모호 → audit 이벤트 1건·cube 0건 / corrected+distinct → audit "corrected_no_notify"·cube 0 /
  실패+모호 → cube summary 에 권장 포함.

---

## 8. caveat·후속

- **임계는 S-LOO 유래.** tau\* 0.9825 는 *성공(S) 프레임 LOO* 매칭에서 나왔다. 실제 fail/E 프레임은
  분포가 다르므로 신호의 *유효성*은 전이되어도 *임계*는 시작점일 뿐 — `Workflow3Settings` 설정값으로
  두고, consensus-gather 가 실제 fail-time `second_ratio` 를 모으면 재보정한다(별도 후속).
- **precision ~41%(@tau\*)**: 재등록 *triage*(recall 97% @ flag 34%) 용으론 탁월하나 무인 자동
  트리거론 부족 — 그래서 본 spec 은 *엔지니어 알림*까지만. 자동 큐/3-way 결정은 상위 plan 영역.
- **상위 비전과의 관계**: `production-trust-consensus...` plan 의 act/abstain 3-way 결정이 같은
  peak-isolation 신호를 쓴다. 본 spec(알림 1차)은 그 결정 층을 바꾸지 않으며, 같은 raw 신호를
  공유한다 — 3-way 결정은 본 포팅 이후 별도로 다룬다.
