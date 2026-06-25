# Re-registration Phase 3 — 스펙 작성 + fix-type/distinct_floor 자체 리뷰 (2026-06-25, 이어서)

> 내일 이어서 진행할 작업의 메모. 같은 날 앞선 저널(`260625_124639_template-bank-matcher-concluded-rejected.md`)에서 template-bank 기각 → re-registration 스레드 시작으로 넘어온 흐름.

## 1. 진행 사항

- **re-registration Phase 3 스펙 작성·커밋·push** (`5d9c987`): `poc/workflow_2/docs/specs/2026-06-25-reregister-rank1-distinctiveness-worklist-design.md`. brainstorming 스킬로 설계(질문 3개: thread 목표=rank-1 sharpen→worklist / rank-1 출처=consensus summary.json / 두 신호 사용=snapshot-vs-region 진단 둘 다). 스펙에 `§10 NEXT SESSION — start here`(task 순서 + office 실행 명령) 포함.
- **핵심 발견(스펙 근거):** consensus eval 의 `summary.json` `per_recipe` row 가 이미 recipe·modality별 rank-1 을 가짐 — `rcp_rank1_rate`(등록 key 변별력)와 `cons_rank1_rate`(영역 변별력=median). consensus eval 무수정으로 join 만 하면 됨.
- **Codex rescue(`/codex:rescue`)로 스펙 리뷰 시도 → 34분 progress 멈춤(stall)으로 사용자 cancel.** Codex 출력 없음(취소 전 미완). 그래서 아래는 **내 자체 리뷰**.

## 2. fix-type + distinct_floor 자체 리뷰 (Codex 대체)

리뷰 대상: §4.2 fix-type 분류표 + `REREGISTER_DISTINCT_FLOOR=0.70` 기본값.

### 결론: 설계는 건전. 단, 스펙에 반영할 2가지 발견.

**fix-type 표는 보이는 것보다 강함 (단일 floor 가 의외로 잘 작동):**
ADR 0006 에서 median-consensus 도 SEM rank-1 ~0.5(distractor 가 noise 아니라 구조적이라 median 이
못 지움). 따라서 ambiguous SEM 영역에선 `cons_rank1` 도 0.70 *아래*로 떨어져 → `NEW_REGION` 이 제대로
발화(빈 카테고리로 degenerate 안 됨). OM 은 median 이 도와 `cons_rank1` 높게 유지 → `FRESH_SNAPSHOT`/`OK`.
modality 물리로 snapshot-vs-region 진단이 자연히 갈림. 좋음.

**발견 1 — `FRESH_SNAPSHOT` 이 과약속(over-promise):**
"영역 괜찮음" 근거는 `cons_rank1`=**median**(N장 success crop)인데, 권고하는 fix(재등록)는 **단일**
새 snapshot 설치 → median 의 denoising 효과 없음. median 은 1등인데 *개별* snapshot 은 아닌 영역은
한 장 다시 찍어도 안 고쳐짐. 함의:
- FRESH_SNAPSHOT 의 정직한 변별력 테스트는 median 이 아니라 *개별* crop 의 rank-1.
- 더 유용: **FRESH_SNAPSHOT row 는 재등록보다 기존 consensus-live-correction**(런타임의 median, 이미
  구현됨 — `project_consensus_live_correction_landed`) 활성화가 더 싸고 나은 fix 일 수 있음. worklist 의
  권고를 그 row 들에 대해 재구성. → 스펙 §4.2/§4.4 에 caveat + 대안 반영.

**발견 2 — `distinct_floor=0.70` 이 한 knob 으로 두 일:**
0.70 은 합리적 *flag* 임계(SEM ~0.5 라 대부분 SEM key flag = 의도대로)지만, 같은 knob 이 (i) flag 여부
+ (ii) FRESH/NEW 분기를 동시에 결정. 둘은 다른 값을 원함 — `rcp_rank1` flag floor(~0.7) + 더 낮은
`cons_rank1` region floor(~0.5, "median 도 못 꼽음"). 스펙 §6 은 별도 cons floor 를 YAGNI 라 했으나
`NEW_REGION` 정확성이 cons 분포에 직결되므로 사실상 선택 아님. → **1차 office run 이 rank-1 *히스토그램*
(평균 말고)을 내보내 0.70/2nd-floor 를 실분포로 보정**. 스펙 §6 의 YAGNI 톤 완화.

(나머지: OK row 가 cons 무시 = 등록 key 가 잘 되면 OK 라 정당, 문제없음. NO_DATA = tier 로만 랭킹,
coverage line 으로 가시화 = 정당.)

### 두 발견의 처리(내일 결정)
스펙에 지금 fold 할지 vs 구현 plan 의 calibration note 로 둘지는 **미결 — 사용자가 내일 결정**. 둘 다
worklist 의 *권고*를 날카롭게 할 뿐 *구조*는 안 바꿈 → 어느 쪽이든 착수에 지장 없음.

## 3. 다음 단계 (내일)

1. (선택) 발견 1·2 를 스펙에 fold 할지 결정 → fold 하면 §4.2/§4.4/§6 수정 후 재커밋.
2. `superpowers:writing-plans` 를 스펙 대상으로 → TDD plan → subagent-driven 실행.
3. plan task 1 = **join key 핀**(reregister `f"{class}/{recipe}"` (line 918) vs consensus `per_recipe["recipe"]` 형식 일치 확인 + `[INFO] rank1-join: matched M/N` coverage line). read-only, Mac-safe.
4. office: consensus eval 먼저(summary.json) → reregister 드라이버가 consume → worklist. 1차 run 에서 **rank-1 히스토그램** 확보해 floor 보정.

핸드오프: `.remember/remember.md` 에 active thread = Phase 3 으로 기록됨. 스펙 §10 이 cold-start 가이드.

## 4. 메모리 업데이트

신규 없음(template-bank 기각 메모리 2건은 앞 저널에서 기록 완료). 본 저널이 Phase 3 스펙 + 미결 리뷰
2건의 기록. 내일 fold 결정 후 필요시 `project_reregister_report_phase1` 메모리에 Phase 3 진행 추가.
