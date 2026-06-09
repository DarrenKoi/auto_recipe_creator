# consensus S-image gather — fail 루프 통합 설계 (spec)

날짜: 2026-06-10
대상: 검증된 consensus(최근 S median) 레버의 **데이터 수집 첫 단계**를, 이미 도는
`poc/workflow_1/align_fail_alarm_record.py` 의 fail 감지 루프에 통합하는 구조.
선행: `2026-06-09-consensus-productization-design.md`(전체 productization),
`consensus_template.py`(게이트·라우터, ✅ 구현됨).

> 이 spec 은 productization 의 §3-C 흐름 중 **"fail 감지 → S 이미지 stage"** 한 조각만
> 다룬다. queue + async worker 분리는 production mode 로 의도적으로 **deferred**.

---

## 1. 배경·목표

PROPOSER_WALL: 단일 rcp 등록 key 는 공정 드리프트로 stale 해져 matcher 후보(top-N)에
진실(align point)이 자주 빠진다. recipe 의 *최근 성공(S)* crop median(consensus)이 현재
외형을 추종해 후보 진입률을 크게 올린다(cond A/B: in_topk 0.434→0.876). `consensus_template.py`
의 게이트·라우터는 이미 구현·검증(7 tests)됐으나, **consensus 를 만들 재료(S 이미지)를 모으는
경로가 없다.**

**전제(이번에 해결됨):** 사용자가 측정 이력 DB 를 직접 구축해, recipe 의 측정이 align fail
이었는지 즉시 판별하고 **성공(S) 이미지 + cond.txt(crosshair 좌표)를 함께 다운로드**할 수 있다.
DB 조회가 빠르고 cond 가 함께 오므로(=consensus 빌드가 요구하는 정렬 좌표 확보), gather 를
fail 루프 안에 둘 수 있다.

**목표:** align fail 감지 시, 그 recipe 의 최근 성공 측정 이미지(S-only)를 **stage(다운로드+저장)**
해 두어 consensus 빌드의 재료를 확보한다. 루프 응답성·기존 동작은 건드리지 않는다.

**비목표(이번 범위 아님):** consensus template *빌드*(workflow_2 의 lazy 작업), rolling
window/eviction/source-set hash/TTL(production worker), `live_align_search` 라우팅 배선,
office downloader *구현*(사용자 담당).

---

## 2. 범위·불변 원칙

**포함:** `SuccessDownloader` Protocol 정의, gather orchestration(disk layout 의 주인),
fail 루프 통합(daemon thread·env flag·guarded import), cache layout(productization spec 호환).

**제외:** 위 비목표 전부. 특히 build 는 명시적으로 workflow_2 가 나중에 lazy 로 수행.

**불변 원칙**
- **경계 유지:** workflow_1 = 데이터 *수집*, workflow_2 = consensus *빌드·소비*. 이 작업은
  workflow_1 이 raw S 이미지를 stage 하는 데서 끝난다(template 생성 안 함).
- **graceful degrade:** 어떤 실패도 루프를 죽이지 않고, 최악의 경우 consensus 재료가 없어
  나중에 rcp 베이스라인으로 폴백(회귀 위험 0).
- **inline-now / separate-later:** gather 는 **단일 함수 seam**(`gather_success_images`)이라,
  production 전환 시 `consensus_prep.py` 워커로 그대로 들어 올린다. cache layout 을 지금부터
  spec 그대로 써서 **데이터 마이그레이션 0**.

---

## 3. 아키텍처

### 3-A. 컴포넌트 (4 touch-points)

의존 방향: `align_fail_alarm_record`(workflow_1) → `consensus_gather`(workflow_2) →
`__init__`(workflow_1, 경로 상수만). 순환 없음. 신규 의존은 전부 guarded import.

| 파일 | 상태 | 책임 |
|---|---|---|
| `poc/workflow_2/consensus_gather.py` | 신규 | Protocol + dataclass + `gather_success_images()` orchestration. **순수·Mac 테스트 가능.** 추후 `consensus_prep.py` 가 흡수. |
| `poc/workflow_1/office_success_downloader.py` | 신규(사용자 구현, office-only) | DB 조회 + S 이미지/cond 다운로드. `SuccessDownloader` 준수. `office_*` 컨벤션(`office_align_fail_alarm.py` 등) 따름. Mac 에선 import 안 됨. |
| `poc/workflow_1/__init__.py` | 수정 | `ALIGN_CONSENSUS_CACHE_DIR` 루트 상수 추가(`align_images/` 와 물리 분리). |
| `poc/workflow_1/align_fail_alarm_record.py` | 수정 | guarded import(`GATHER_AVAILABLE`), `_gather_success_async`(daemon thread), call site(notify 직후·record cycle 직전), env flag·param. |

### 3-B. 인터페이스 (consensus_gather.py)

```python
@dataclass
class StagedEvent:
    event_id: str
    image_paths: list[Path]   # 쓰여진 S*.jpg
    cond_paths: list[Path]    # 쓰여진 S*.txt(cond)

@dataclass
class GatherResult:
    eqp_id: str
    recipe_id: str
    events_dir: Path
    n_events: int
    n_images: int
    reason: str               # "ok" | "empty" | "skipped" | "error:<msg>"

class SuccessDownloader(Protocol):
    def download_recent_successes(self, recipe_id, *, max_events, dest_dir) -> list[StagedEvent]:
        """recipe 의 최근 성공(S) 측정 max_events 건을 dest_dir/<event_id>/ 에
        S*.jpg + S*.txt(cond)로 쓰고, 쓴 내역을 반환한다. (office 구현)

        dest_dir 는 orchestration 이 넘기는 *임시 staging dir* 다(최종 events/ 아님).
        orchestration 이 ≥1 event 일 때만 events/ 로 swap(replace-if-non-empty)."""

GATHER_MAX_EVENTS = 5         # 모듈 기본; 루프가 env 값을 주입.

def gather_success_images(eqp_id, recipe_id, *, downloader,
                          max_events=GATHER_MAX_EVENTS,
                          cache_root=ALIGN_CONSENSUS_CACHE_DIR) -> GatherResult:
    """disk layout 의 주인. events_dir = cache_root/<eqp_id>/<recipe_id>/events.
    downloader 를 호출해 stage 하고, replace-if-non-empty 로 교체, 건수 집계·로그."""
```

**역할 분리(확정):** orchestration 이 **disk layout 의 주인**(events_dir 결정·생성·교체·집계·로그),
downloader 가 **DB 조회 + 파일 쓰기**(넘겨받은 dir 안에). office 특화는 downloader 한 곳에.

### 3-C. cache layout (productization spec 호환)

```
poc/workflow_1/align_consensus_cache/<eqp_id>/<class>/<recipe>/events/<event_id>/
   ├─ S0001.jpg  S0001.txt
   └─ S0002.jpg  S0002.txt
```

`recipe_id` 가 `"<class>/<recipe>"` 형태라 `<eqp_id>/<recipe_id>` 가 그대로 3단계가 된다
(`align_images/` 와 동일 규약, `rcs_screenshot.py` 와 일치). `align_images/` 와 **물리 분리**
(race/혼동 없음). productization spec 의 `events/` 트리와 동일 → 추후 worker 가 `state.json`·
`template/`·eviction 을 *같은 트리 위에* 얹으면 됨.

### 3-D. data flow / 통합 지점

`process_fail_rows` 의 per-eqp 루프, notify(`_send_rich_notify_async`) **직후 · `run_record_cycle`
직전**. fire-and-forget 들과 그룹화되고, daemon thread 라 뒤따르는 동기 record cycle 과 **겹쳐
실행**(DB/디스크 vs 마우스/창 — 자원 비경합).

```
new align-fail edge (eqp_id, recipe_id="<class>/<recipe>")
  → append_alarm_record / popup / rich_notify
  → [NEW] if gather_enabled and recipe_id:
        _gather_success_async(eqp_id, recipe_id)          # daemon thread, 루프 비차단
            └─ gather_success_images(eqp_id, recipe_id, downloader=_success_downloader, max_events=N)
                 ├─ events_dir = ALIGN_CONSENSUS_CACHE_DIR/<eqp_id>/<recipe_id>/events
                 ├─ temp 에 stage → downloader.download_recent_successes(...)
                 ├─ replace-if-non-empty: ≥1 event 면 events/ 교체, 아니면 기존 보존
                 └─ 건수 로그 → GatherResult
  → run_record_cycle (동기 GUI — gather thread 와 겹침)
  → append_record_manifest
```

---

## 4. 파라미터·semantics

- **`N` = `ALIGN_FAIL_GATHER_MAX_EVENTS` (기본 5):** 끌어올 **측정 event 수**(이미지 수 아님;
  event 당 보통 2~3장 → ~10–15장). gather/저장량 수도꼭지. event 단위인 이유 = modality(OM/SEM)
  balance(이미지 단위면 한쪽 쏠림 위험).
- **`min_s` = `ConsensusPolicy.min_s` (기본 3):** consensus *빌드* 게이트, **modality별**. 이 작업
  (gather)엔 게이트 없음 — 받은 건 다 stage 하고, "부족" 판정은 **나중 build 시점**에서
  per-modality 로 graceful 폴백(< min_s → 그 modality 만 rcp). gather 는 게이트 안 함.
- **replace-if-non-empty:** downloader 가 ≥1 event 반환 시에만 `events/` 를 최신 snapshot 으로
  교체(temp→swap), 0건/실패면 기존 보존. consensus 가 *현재* 외형을 추종해야 하므로 stale +
  fresh 누적은 median 을 오염시킨다 → 교체가 정답. rolling-window/eviction-by-event_id/source-set
  hash 는 production worker 로 deferred.

---

## 5. error handling

기존 파일 철학("실패는 삼켜 루프가 안 죽게") 그대로:

1. **루프 불사:** gather 전체가 daemon thread try/except. 예외 → `[WARNING]` 후 계속.
2. **guarded import:** `consensus_gather`/`office_success_downloader` import 실패 →
   `GATHER_AVAILABLE=False` → skip. Mac 은 항상 이 경로(office 모듈 없음).
3. **off 스위치:** env `ALIGN_FAIL_GATHER_SUCCESS`(기본 on) → `process_fail_rows(gather_enabled=)`.
4. **recipe_id 게이트:** 없으면 skip(경로 불가) — record cycle 과 동일.
5. **빈/실패 fetch → 기존 보존:** replace-if-non-empty(§4). transient 실패로 멀쩡한 set 안 날림.
6. **부분 fetch 통과:** 적게 받아도 stage·로그. 부족 판정은 build 시점 `min_s` per-modality.
7. **동시성:** 다수 eqp fail → 각 daemon thread. 저장 경로가 `<eqp_id>/<recipe_id>/` 라 디렉터리
   분리 → 충돌 없음. lockfile 불필요(production pooling/dedup 시 재검토).

actuation 없음(순수 DB+파일 IO) → `SAFE_MODE` 무관, Mac 안전.

---

## 6. 테스트 (전부 Mac 합성, office 데이터 불필요)

**신규 `poc/workflow_2/test_consensus_gather.py`** — `FakeSuccessDownloader`(DI, `tmp_path` 에
합성 event 파일 write):

1. **stage basic** — `events/<event_id>/` 에 S*.jpg + S*.txt, `GatherResult` 건수 정확.
2. **layout** — 경로 = `cache_root/<eqp_id>/<recipe_id>/events/...`(슬래시 → class/recipe 중첩).
3. **replace-if-non-empty** — 2차 gather 가 새 set 교체 / 빈 반환 시 기존 보존.
4. **downloader raises** — `gather_success_images` 가 삼키고 `GatherResult(reason="error:...")` 반환.
5. **mkdir** — events_dir 없으면 생성.

**`align_fail_alarm_record` 게이팅** — `_gather_success_async` monkeypatch:
`gather_enabled` + recipe_id 있을 때만 호출, off / recipe_id 없음 / `GATHER_AVAILABLE=False` 면 미호출.

> cond.txt 합성 포맷은 eval 의 cond sidecar 스키마(crosshair 좌표)에 맞춘다 — build 는 scope
> 밖이지만 같은 계약으로 stage 해야 추후 crop_pipeline + `build_consensus_template` 이 추가 변환
> 없이 소비한다.

---

## 7. 구현 순서

1. `consensus_gather.py` — Protocol + dataclass + `gather_success_images`(replace-if-non-empty
   포함). `FakeSuccessDownloader` 로 TDD(§6 1–5). **office 의존 0 → Mac 완결.**
2. `poc/workflow_1/__init__.py` — `ALIGN_CONSENSUS_CACHE_DIR` + `__all__`.
3. `align_fail_alarm_record.py` — guarded import + `_gather_success_async` + call site +
   `process_fail_rows(gather_enabled=)` + `monitor_loop` env 배선. 게이팅 테스트.
4. (사용자, office) `office_success_downloader.py` 구현 + 오피스 1회 실행으로 stage 확인.

1–3 은 Mac 에서 완결·검증. 4 만 office.

---

## 8. deferred (production mode 에서)

- queue(`consensus_prep_queue.jsonl`) + async worker(`consensus_prep.py`) 분리.
- rolling window(N event) + eviction-by-event_id + source-set hash + TTL throttle.
- temp→swap 의 완전 원자성(현재 POC 는 rmtree+rename 비원자 윈도우 허용).
- consensus *빌드*(crop_pipeline 추출 + `build_consensus_template`) 및 `live_align_search`
  `select_routing_templates` 라우팅 배선.
- E(fail) 타깃에서 consensus 가 실제 align point 를 더 잘 잡는지 라이브 검증(현재 모든 검증은
  S 타깃 proposer recall 기준).

---

## 9. 제약 (불변)

CLI 인자 금지 / Korean docstring / `[INFO]·[ERROR]·[WARNING]` print(logging 모듈 금지) /
`from __future__` 금지 / 절대 임포트(`from poc.workflow_x...`) / main 직접 commit·push(Mac→office
pull) / commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` /
설계규칙: CV 가 좌표·점수 결정, VLM 은 영역/타당성만(이 작업엔 VLM 무관).
