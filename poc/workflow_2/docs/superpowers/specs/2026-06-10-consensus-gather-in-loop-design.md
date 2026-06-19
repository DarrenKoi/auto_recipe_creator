# consensus S-image gather — workflow_3 monitor 루프 통합 설계 (spec)

날짜: 2026-06-10 (workflow_3 재정렬 rev)
대상: 검증된 consensus(최근 S median) 레버의 **데이터 수집 첫 단계**를, workflow_3 의
실시간 모니터 루프(`poc/workflow_3/monitor/align_fail_monitor.py`)에 통합하는 구조.
선행: `2026-06-09-consensus-productization-design.md`(전체 productization),
`poc/workflow_2/consensus_template.py`(게이트·라우터, ✅ 구현됨, 아직 legacy 위치).

> **재정렬 메모(중요):** 본 spec 초판은 legacy `workflow_1/align_fail_alarm_record.py` +
> `workflow_2` 를 겨냥했으나, 세션 중 진행된 **workflow_3 이전**(stage 1–5 commit + monitor
> stage 작업트리)으로 타겟을 옮긴다. workflow_3 는 workflow_1+2 의 production 경로를 통합한
> 패키지이며 **legacy 를 import 하지 않는다**(`workflow_3/__init__.py`). 따라서 신규 코드는
> 전부 workflow_3 안에 둔다. 설계 자체(Approach A: stage-only, daemon thread,
> replace-if-non-empty)는 불변 — 배치/배선만 바뀐다(monitor 루프도 `process_fail_rows`
> 동일 shape 유지).

---

## 1. 배경·목표

PROPOSER_WALL: 단일 rcp 등록 key 는 공정 드리프트로 stale 해져 matcher 후보(top-N)에
진실(align point)이 자주 빠진다. recipe 의 *최근 성공(S)* crop median(consensus)이 현재
외형을 추종해 후보 진입률을 크게 올린다(cond A/B: in_topk 0.434→0.876). 게이트·라우터
(`consensus_template.py`)는 구현·검증(7 tests)됐으나 **consensus 재료(S 이미지)를 모으는
경로가 없다.**

**전제(이번에 해결됨):** 사용자가 측정 이력 DB 를 직접 구축해, recipe 의 측정이 align fail
이었는지 즉시 판별하고 **성공(S) 이미지 + cond.txt(crosshair 좌표)를 함께 다운로드**할 수 있다.
DB 가 빠르고 cond 가 함께 오므로(consensus 빌드가 요구하는 정렬 좌표 확보) gather 를 모니터
루프 안에 둘 수 있다.

**목표:** align fail 감지 시, 그 recipe 의 최근 성공 측정 이미지(S-only)를 **stage(다운로드+
저장)** 해 consensus 빌드 재료를 확보한다. 루프 응답성·기존 동작은 건드리지 않는다.

**비목표(범위 아님):** consensus template *빌드*(vision 의 lazy 작업; `consensus_template`
vision 이전 포함), rolling window/eviction/hash/TTL(production worker), `live_align_search`
라우팅 배선, office downloader *구현*(사용자 담당).

---

## 2. 범위·불변 원칙

**포함:** `SuccessDownloader` Protocol + gather orchestration(vision, 순수·Mac 테스트),
office-loader + 비차단 fire(monitor glue), monitor 루프 통합(`Workflow3Settings` flag),
cache layout(productization spec 호환), cache root 상수.

**제외:** 위 비목표 전부. build 는 명시적으로 vision 이 나중에 lazy 수행.

### 불변 원칙

- **경계 유지:** monitor = 폴링/수집/배선, vision = consensus 빌드·소비(추후), office_* = DB/장비
  접점. 이 작업은 monitor 가 vision 의 순수 gather 를 호출해 raw S 이미지를 stage 하는 데서 끝난다
  (template 생성 안 함). 의존 방향 `monitor → vision` 준수.
- **graceful degrade:** 어떤 실패도 루프를 죽이지 않고, 최악의 경우 재료 부재 → 나중에 rcp
  베이스라인 폴백(회귀 위험 0).
- **inline-now / separate-later:** gather 는 **단일 함수 seam**(`gather_success_images`)이라
  production 전환 시 `consensus_prep.py` 워커로 그대로 이식. cache layout 을 spec 그대로 써서
  **데이터 마이그레이션 0**.

---

## 3. 아키텍처

### 3-A. 컴포넌트 (전부 workflow_3)

의존 방향: `align_fail_monitor`(monitor) → `success_gather`(monitor) → `consensus_gather`(vision)
→ `workflow_3` 루트 상수. office downloader 는 importlib 2단 fallback. 순환 없음.

| 파일 | 상태 | 책임 |
| --- | --- | --- |
| `poc/workflow_3/vision/consensus_gather.py` | 신규 | Protocol + dataclass + `gather_success_images()` orchestration. **순수·Mac 테스트 가능**(office/threading import 0). 추후 `consensus_prep.py` 흡수. |
| `poc/workflow_3/monitor/success_gather.py` | 신규 | glue: `load_success_downloader()`(office 2단 fallback, `alarm_source`/`notify` 패턴) + `gather_success_async()`(daemon thread). |
| `poc/workflow_3/monitor/office_success_downloader.py` | 신규(사용자, office-only, `**/office_*` gitignore) | DB 조회 + S 이미지/cond 다운로드. `SuccessDownloader` 준수. Mac 미존재. |
| `poc/workflow_3/__init__.py` | 수정 | `ALIGN_CONSENSUS_CACHE_DIR` 루트 상수(env override, `ALIGN_IMAGES_DIR` 패턴). |
| `poc/workflow_3/config.py` | 수정 | `Workflow3Settings.gather_enabled/gather_max_events` + `load_workflow3_settings` env 배선. |
| `poc/workflow_3/monitor/align_fail_monitor.py` | 수정 | `process_fail_rows` 에 `gather_success_async` 1회 호출(popup 직후·`run_alarm_cycle` 직전). |

### 3-B. 인터페이스 (vision/consensus_gather.py)

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

GATHER_MAX_EVENTS = 5         # 모듈 기본; 루프가 settings 값을 주입.

def gather_success_images(eqp_id, recipe_id, *, downloader,
                          max_events=GATHER_MAX_EVENTS,
                          cache_root=ALIGN_CONSENSUS_CACHE_DIR) -> GatherResult:
    """disk layout 의 주인. events_dir = cache_root/<eqp_id>/<recipe_id>/events.
    임시 dir 에 downloader 로 stage → ≥1 event 면 events/ 로 swap, 아니면 보존. 집계·로그."""
```

**역할 분리(확정):** orchestration(vision) 이 **disk layout 의 주인**(cache root·events_dir·temp·
swap·집계·로그), downloader(office) 가 **DB 조회 + 파일 쓰기**(넘겨받은 dest_dir 안). office 특화는
downloader 한 곳에. → 사용자는 "어디에"가 아니라 "무엇을 DB에서 가져와 어떻게 파일로 쓰나"만 구현.

### 3-C. cache layout (productization spec 호환)

```text
poc/workflow_3/align_consensus_cache/<eqp_id>/<class>/<recipe>/events/<event_id>/
   ├─ S0001.jpg  S0001.txt
   └─ S0002.jpg  S0002.txt
```

`recipe_id` 가 `"<class>/<recipe>"` 형태라 `<eqp_id>/<recipe_id>` 가 그대로 3단계(기존
`align_images/` 규약과 동일, `rcs_screenshot` 일치). cache 루트는 `WORKFLOW_3_DIR /
"align_consensus_cache"` 기본, env `ALIGN_CONSENSUS_CACHE_DIR` 로 override(MES 가 아니라 우리가
만드는 산출물이라 위치 자유). productization spec 의 `events/` 트리와 동일 → 추후 worker 가
`state.json`·`template/`·eviction 을 같은 트리 위에 얹는다.

### 3-D. data flow / 통합 지점

`align_fail_monitor.process_fail_rows` 의 per-eqp 루프, popup(`notify_align_fail_popup`)
**직후 · `run_alarm_cycle` 직전**. fire-and-forget(daemon thread)이라 뒤따르는 동기 cycle
(접속·녹화·보정·engineer_watch 최대 수백초)과 **겹쳐 실행**(DB/디스크 vs RCS GUI — 자원 비경합).
gather 는 monitor 의 데이터 수집 책임(cycle 의 RCS/보정과 독립)이라 cycle 이 아니라 루프 레벨에 둔다.

```text
process_fail_rows (per new eqp edge)
  → append_alarm_record / popup
  → [NEW] if settings.gather_enabled and recipe_id:
        gather_success_async(eqp_id, recipe_id, settings)          # monitor/success_gather, daemon thread
            └─ downloader = load_success_downloader()  (1회 캐시, office 2단 fallback)
            └─ gather_success_images(eqp_id, recipe_id, downloader=, max_events=settings.gather_max_events)  # vision, 순수
                 ├─ events_dir = ALIGN_CONSENSUS_CACHE_DIR/<eqp_id>/<recipe_id>/events
                 ├─ temp 에 stage → downloader.download_recent_successes(recipe_id, max_events=, dest_dir=temp)
                 ├─ replace-if-non-empty: ≥1 event 면 events/ swap, 아니면 보존
                 └─ 건수 로그 → GatherResult
  → run_alarm_cycle (동기 — gather thread 와 겹침)
  → append_cycle_manifest
```

---

## 4. 파라미터·semantics

- **`gather_max_events` (= `ALIGN_FAIL_GATHER_MAX_EVENTS`, 기본 5):** 끌어올 **측정 event 수**
  (이미지 수 아님; event 당 보통 2~3장 → ~10–15장). event 단위 이유 = modality(OM/SEM) balance
  (이미지 단위면 한쪽 쏠림 위험).
- **`min_s` = `ConsensusPolicy.min_s` (기본 3):** consensus *빌드* 게이트, **modality별**. 이
  작업(gather)엔 게이트 없음 — 받은 건 다 stage, "부족" 판정은 *나중 build 시점*에서 per-modality
  graceful 폴백(< min_s → 그 modality 만 rcp). build 는 범위 밖(consensus_template, 아직 legacy).
- **replace-if-non-empty:** downloader 가 ≥1 event 반환 시에만 `events/` 를 최신 snapshot 으로
  교체(temp→swap), 0건/실패면 기존 보존. consensus 가 *현재* 외형 추종해야 하므로 stale+fresh
  누적은 median 오염 → 교체가 정답. rolling-window/eviction/hash 는 production worker 로 deferred.
- **`gather_enabled` (= `ALIGN_FAIL_GATHER_SUCCESS`, 기본 on):** off 스위치.

---

## 5. error handling

기존 workflow_3 모듈 철학("실패는 삼켜 루프가 안 죽게", `notify`/`recording` 동일):

1. **루프 불사:** `gather_success_async` 의 daemon thread 본체가 try/except. 예외 →
   `[WARNING]`/`log_work2_event` 후 계속.
2. **office 미존재:** `load_success_downloader()` 가 2단 fallback 실패 시 None → gather skip
   (Mac/개발 PC. `alarm_source`/`notify` 의 None 폴백과 동일).
3. **off 스위치:** `settings.gather_enabled`(`ALIGN_FAIL_GATHER_SUCCESS`).
4. **recipe_id 게이트:** 없으면 skip(경로 불가) — cycle 의 보정 게이트와 동일 정책.
5. **빈/실패 fetch → 기존 보존:** replace-if-non-empty(§4). transient 실패로 멀쩡한 set 안 날림.
6. **부분 fetch 통과:** 적게 받아도 stage·로그. 부족 판정은 build 시점 `min_s` per-modality.
7. **동시성:** 다수 eqp fail → 각 daemon thread. 저장 경로가 `<eqp_id>/<recipe_id>/` 라 디렉터리
   분리 → 충돌 없음. lockfile 불필요(production pooling/dedup 시 재검토).

actuation 없음(순수 DB+파일 IO) → `SAFE_MODE`/`correction_dry_run` 무관, Mac 안전.

---

## 6. 테스트 (전부 Mac 합성, office 데이터 불필요)

**신규 `poc/workflow_3/vision/test_consensus_gather.py`** — `FakeSuccessDownloader`(DI,
`tmp_path` 에 합성 event 파일 write):

1. **stage basic** — `events/<event_id>/` 에 S*.jpg + S*.txt, `GatherResult` 건수 정확.
2. **layout** — 경로 = `cache_root/<eqp_id>/<recipe_id>/events/...`(슬래시 → class/recipe 중첩).
3. **replace-if-non-empty** — 2차 gather 가 새 set 교체 / 빈 반환 시 기존 보존.
4. **downloader raises** — `gather_success_images` 가 삼키고 `GatherResult(reason="error:...")` 반환.
5. **mkdir** — events_dir 없으면 생성.

**신규 `poc/workflow_3/monitor/test_success_gather.py`** — `gather_success_async` 게이팅·해석:
`gather_enabled` + recipe_id 있을 때만 `gather_success_images` 호출(monkeypatch),
off/recipe_id 없음/downloader None 이면 미호출. `load_success_downloader` 2단 fallback(가짜 모듈).

> cond.txt 합성 포맷은 eval cond sidecar 스키마(crosshair 좌표,
> `poc/workflow_3/vision/cond_file.py`)에 맞춘다 — build 는 범위 밖이지만 같은 계약으로 stage
> 해야 추후 crop_pipeline + `build_consensus_template` 이 추가 변환 없이 소비한다.

---

## 7. 구현 순서

1. `vision/consensus_gather.py` — Protocol + dataclass + `gather_success_images`(replace-if-non-
   empty). `FakeSuccessDownloader` 로 TDD(§6 1–5). **office 의존 0 → Mac 완결.**
2. `workflow_3/__init__.py` — `ALIGN_CONSENSUS_CACHE_DIR`(env override) + `__all__`.
3. `config.py` — `gather_enabled`/`gather_max_events` 필드 + `load_workflow3_settings` env.
4. `monitor/success_gather.py` — `load_success_downloader`(2단) + `gather_success_async`. TDD(§6 monitor).
5. `monitor/align_fail_monitor.py` — `process_fail_rows` 호출 1줄.
6. (사용자, office) `monitor/office_success_downloader.py` 구현 + 오피스 1회 실행 stage 확인.

1–5 는 Mac 에서 완결·검증. 6 만 office.

---

## 8. deferred (production mode 에서)

- queue(`consensus_prep_queue.jsonl`) + async worker(`consensus_prep.py`) 분리.
- rolling window(N event) + eviction-by-event_id + source-set hash + TTL throttle.
- temp→swap 완전 원자성(현재 POC 는 rmtree+rename 비원자 윈도우 허용).
- consensus *빌드*: `consensus_template`/crop_pipeline 의 vision 이전 + `build_consensus_template`
  소비 + `live_align_search` `select_routing_templates` 라우팅 배선.
- E(fail) 타깃에서 consensus 가 실제 align point 를 더 잘 잡는지 라이브 검증(현재 모든 검증은
  S 타깃 proposer recall 기준).

---

## 9. 제약 (불변)

CLI 인자 금지 / Korean docstring / `[INFO]·[ERROR]·[WARNING]` print(+ `log_work2_event` audit) /
`from __future__` 금지 / 절대 임포트(`from poc.workflow_3...`) / **workflow_3 는 legacy(wf1/wf2)
import 금지** / main 직접 commit·push(Mac→office pull) / commit trailer
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` / 설계규칙: CV 가 좌표·점수
결정, VLM 은 영역/타당성만(이 작업엔 VLM 무관).
