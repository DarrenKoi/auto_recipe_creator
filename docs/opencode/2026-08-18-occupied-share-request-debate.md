# 점유 tool 화면 공유 요청 설계 — opencode oc-discuss record

- Run date: 2026-08-18
- Skill: oc-discuss (3 rounds)
- Model: opencode-go/glm-5.3 variant=high (tier=heavy)
- Target: `poc/workflow_3/docs/superpowers/specs/2026-08-18-occupied-share-request-recording-design.md`
- Elapsed: 90s + 55s + 34s · Session: `ses_fed223a6bffekUmXXkIQ4rFN3u`

압박 테스트한 설계: 점유 `Select` 팝업을 검출만 하고 포기하던 경로를 바꿔, 화면 공유를
요청하고 승낙되면 엔지니어의 수동 align 작업을 녹화한다.

## What the model said

### Round 1

> ## 1. The view-only nets fail in both directions — and the bad direction is suppressed
>
> Net (b) fails **open**: on OCR failure the design keeps correcting (설계 ③ "읽기 실패 시… 기존 동작 유지"). So on the exact uncalibrated path (restart, cache gone, occupant column geometry unknown), an already-granted view-only window gets open-loop clicks that silently no-op → `corrected` → cube suppressed. The doc's own "문제의 핵심" hole stays open on the path the cache was supposed to cover. Worse, the inverse: the 2h grant cache marks an EQP view-only even after the colleague **finishes and releases** — next entry is fully controllable, but you skip correction, suppress the notification, and (⑥) pin it in `active_tools` so you never re-attach. Result: a real alarm, uncorrected, unnotified, unretried. That's the strongest falsifier, just mirrored.
>
> ## 2. Suppression assumes occupant == responsible engineer
>
> The cube alert goes to the engineer owning *this alarm*; the person at the tool may be someone else entirely (or nobody, after they walk away). `view_only_observation` + "한 번 녹화했으면 충분하다" means if the occupant leaves without clearing the alarm, no human is ever told and the loop won't retry. You've converted an escalation channel into a heuristic guess about who's standing at the tool.
>
> ## 3. Strict default + uncalibratable tokens = dead feature *plus* a 90s tax
>
> You develop on macOS with zero access to the popup; the OCR tokens ("share"+"screen", language, line-wrap) are explicitly listed as office-unknown. Strict means first office runs most likely land `confirm_failed` → the flagship feature never fires. Meanwhile every occupied alarm now burns locator+OCR attempts **plus up to 90s of blocking wait** inside `_exec_wait_tool_window` — in a single process with one shared RCS cursor serially serving all tools. Occupied-tool churn (retries via `_RETRY_LATER`) now taxes every other alarm's latency.
>
> ## 4. The Cancel click is ungated
>
> Radio and Request get OCR gates; the recovery/action "Cancel 로 닫기" (error table, ②) is a bare click on the same mislocatable popup, executed precisely when localization has just failed. Worst case is hitting Request twice, but you've violated your own fail-closed rule at the worst moment.
>
> ## Considered and rejected
>
> Detector/actuator split by error policy: sound; mixing fail-open swallowing into a click path is a real hazard. Merged `denied_or_timeout` is fine — action is identical.
>
> Verdict: the architecture is right, but 1 and 2 are design-level, not marginal — fix the fail-open of net (b), the cache's stale-view-only error, and the suppression/retry assumptions before implementing.

### Round 2

> **Objection 3: dropped.** The asymmetric-cost argument is sound, and the mandatory diagnostic artifact does make strict self-debugging. Iterative geometry tuning may take 2–3 round-trips, but each failed round-trip is a non-click — the cost is delay, not harm. Not worth pressing.
>
> **New objection (strongest remaining): `corrected_unverified` has an unspecified lifecycle, and that's where a silent failure hides.**
>
> Your fix makes uncertainty visible at *notification* time, but the design doesn't say how the loop treats the new status afterward. Current consumers branch on `status != "corrected"` (engineer watch, notify) and on outcome/`failure_class` for retry-vs-active (`align_fail_monitor`). `corrected_unverified` is a third thing: the cycle *ran to completion* with an outcome that might be real (tool was free) or a complete no-op (view-only, clicks ignored). You cannot distinguish the two — that's the point of the tag. So:
>
> - If any consumer treats it as success (`active_tools`, prefix/normalized matching against "corrected", manifest success semantics), the alarm is never retried and one cube message is the only signal — a colleague dismisses it, the tool stays uncorrected, and the loop believes it's handled. That is your original silent-failure hole, one level up.
> - Conversely, if it always goes to cooldown-retry, a genuinely-free-but-OCR-flaky tool gets re-corrected repeatedly — visible, mostly harmless, but worth stating as accepted cost.
>
> The spec needs one explicit line: `corrected_unverified` never registers success anywhere, and every string comparison against "corrected" in `cycle.py`/`notify.py`/`align_fail_monitor.py` is enumerated and verified (there are several, including `outcome.status != "corrected"` checks where substring-adjacent statuses could sneak past).
>
> Weaker, noting only: a persistent false-positive occupant read on a free tool → infinite view_only cooldown loop with cube spam. Visible, not silent — fix is just a retry cap.
>
> Otherwise the three-state + always-notify + cooldown combination closes the paths I found.

### Round 3

> **Yes — the widening attacks the mismatch detector, which is the load-bearing part of the existing gate.**
>
> Concrete case: `classify_tokens` (tool_row_verify.py:147) declares **mismatch** on any token that is alnum, 5–12 chars, with both digits and letters, when the target ID wasn't read. Engineer occupant IDs are exactly that shape (`KIM0234`, `HYN1A2B`). Today the strip contains only the ID column, so an OCR flake on the target yields `unreadable` → lenient passes → click proceeds. With the occupant column inside the crop, the same flake now reads the occupant token cleanly → `confirmed` fails (target absent) → occupant token passes `_looks_like_tool_id` → verdict flips to `mismatch` → gate rejects a **correct** row → `connect_not_clicked` → whole cycle dies on the shared cursor. You've converted the benign failure mode (unreadable) into the destructive one (mismatch) at exactly the rate the occupant column is populated — which, by design, is every case you built this for.
>
> Second, `PointTextRead.tokens` is position-free (`tokens_from_text` discards geometry), so you cannot partition tokens by column afterward. One crop means one undifferentiated token soup.
>
> Third, the module's own history says PaddleOCR-VL works only because the crop is a single-line document fragment; a two-column strip moves it back toward the whole-list garbage regime the docstring documents.
>
> If you keep single-OCR: restrict `classify_tokens` to tokens whose substring matches the ID column's known format *and* exclude occupant-format tokens — but that's calibrated guessing. The honest fix is two narrow crops (or one crop, two recognize calls) — the extra ~1 OCR round-trip is cheaper than breaking the gate every occupied row.

## Judgement

### 수용 (5건)

1. **R1-1a — net (b) 의 fail-open 근거가 틀렸다.** 수용. "(a) 확정 판별이 대부분을 덮는다"는
   전제가 프로세스 재시작 후 무너진다. (b)는 이전 승낙만으로 도달 가능하므로 이번 프로세스에
   요청 이력이 없다. → 점유를 3-상태(`occupied_by_other`/`free`/`unknown`)로 바꾸고, `unknown`
   은 보정을 막는 대신 `corrected_unverified` 로 표시해 **알림이 반드시 나가게** 했다. 클릭을
   막는 게 아니라 불확실성을 보이게 하는 방향이다.
2. **R1-1b — grant 캐시의 stale view-only.** 수용. `share_grant_cache` 와
   `ALIGN_FAIL_SHARE_GRANT_TTL_SEC` 를 **삭제**했다. cross-cycle 기억 없이 매 사이클 새로
   판별한다.
3. **R1-2 — cube 생략이 점유자와 알람 담당자를 동일시한다.** 수용. 생략을 철회하고 status 별
   문구만 구분한다. 또한 view-only 사이클은 `active_tools` 가 아니라 cooldown 으로 보내,
   점유자가 tool 을 놓으면 실제 보정이 돌아가게 했다.
4. **R1-4 — Cancel 좌표 클릭이 무방비.** 수용. `close_window()` 로 대체. 모델의 지적이 더 나은
   해법을 끌어냈다 — 창 핸들이 이미 있으므로 VLM 도 좌표도 필요 없다.
5. **R3 — 행 strip crop 확장이 기존 게이트를 깬다.** 수용. 코드로 확인했다:
   `_looks_like_tool_id` 는 `KIM0234` 형태를 통과시키고 `accepts()` 는 `lenient` 에서도
   `mismatch` 를 거부하므로, 지금은 무해한 `unreadable` 이 파괴적인 `mismatch` 로 승격된다.
   → `read_row_occupant()` 는 자기 crop 과 자기 OCR 호출을 갖고, `tool_row_verify` 는 손대지
   않는다.

### 부분 수용 (1건)

6. **R1-3 — strict 기본값 + 90초 블로킹.** 지연 부분만 수용해 `share_wait_sec` 를 90 → 45초로
   낮췄다(단일 RCS 커서를 모든 알람이 직렬 공유하므로 대기가 그대로 전가된다). "dead feature"
   부분은 반박했다 — 확인 실패 시 진단 산출물(팝업 crop + OCR 원문 + 좌표) 저장을 필수로 하면
   한두 번의 오피스 왕복으로 자기 진단이 되고, 실패의 비용은 지연인 반면 `terminate existing
   user` 오클릭의 비용은 동료의 진행 중 측정 파괴로 복구 불가다. **모델이 R2 에서 철회했다.**

### 신규 수용 (2건)

7. **R2 — `corrected_unverified` 의 생애주기 미정의.** 수용. 코드 확인 결과 `corrected` 비교는
   `notify.py:286` 과 `cycle.py:740` 두 곳뿐이고 모두 정확 비교(`==`/`!=`)라 접두사 스밈은
   없었다. 그러나 재시도 쪽 지적은 실재했다 — `_cycle_failed()` 가 `run_status`/`failed_step`
   만 보므로 완주한 사이클은 `active_tools` 로 가 영영 재시도되지 않는다. →
   `_RETRY_LATER_OUTCOME_STATUSES = {view_only_observation, corrected_unverified}` 를 추가.
8. **R2 (weaker) — false-positive 점유 판독의 cube spam.** 수용. cooldown 300초 + 사이클당
   녹화 300초라 시간당 약 6회가 된다. `ALIGN_FAIL_SHARE_MAX_ATTEMPTS`(기본 2) 상한을 두고
   초과 시 `active_tools` 로 넘긴다.

### 기각 (0건)

없음. 모델이 명시적으로 인정한 항목: detector/actuator 분리, `denied_or_timeout` 통합.

## Follow-up

- spec 갱신 커밋: 이 문서와 함께 커밋.
- 구현 계획: `poc/workflow_3/docs/superpowers/plans/2026-08-18-occupied-share-request-recording.md`
