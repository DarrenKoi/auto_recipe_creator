# consensus 프로덕션화 설계 (spec)

날짜: 2026-06-09
대상: 검증된 consensus(최근 S median) template 을 matcher/`live_align_search` 의 실제 등록
template 로 승격하는 productization 의 **데이터 수집·저장·소비** 구조.
선행 저널: `260608_163302`(검증·핸드오프), `260609_045819`(blur 가드 해석),
`260609_053556`(수집/저장 §1·§2 합의). 이 spec 은 그 §3(가드·에러·테스트·구현순서)을 확정한다.

---

## 1. 배경·목표

PROPOSER_WALL: 단일 rcp 등록 key 는 공정 드리프트로 stale 해져 matcher 후보(top-N)에 진실
(align point)이 자주 빠진다([[project_matcher_flat_chamfer_distinctiveness]]). recipe 의 *최근
성공(S)* crop 들을 crosshair 로 정렬해 median 을 뜨면 현재 외형을 추종 → 후보 진입률↑.

cond A/B 검증(`260608_163302`): **in_topk 0.434→0.876 (+0.442), rank1 0.318→0.764 (+0.446)**
(recipes=134, S_loo=403, min_s=3). baseline rcp 0.434 가 localization eval `gt_in_topk` 0.433 과
독립 재현 → 측정 파이프라인 정상. → **CONSENSUS 채택 권장.**

**목표**: 이 검증된 레버를 프로덕션 경로(`live_align_search` 의 등록 template)로 옮기되,
recipe 가 **수만 개**·한 측정 이벤트당 success 이미지 **2~3장뿐**이라는 제약 아래
on-demand 수집·롤링 캐시·rcp 폴백으로 안전하게 통합한다.

## 2. 범위

**포함**: prep 큐 → 워커 → 롤링 캐시 → consensus 빌드(게이트 포함) → 소비(consensus-or-rcp 라우팅).
**제외**: download 구현(사용자 담당, Protocol 만 정의), 실시간 SEM Monitor end-to-end 액추에이션.

**불변 원칙**: consensus 는 rcp 위에 얹는 *옵션*이지 대체가 아니다. 모든 실패 경로는 rcp 로
graceful degrade → **최악의 경우 = 현재 검증된 rcp 베이스라인(in_topk 0.434) → 회귀 위험 0.**

**검증 범위 caveat (★ 잊지 말 것)**: 모든 검증은 **S 타깃 proposer recall** 기준이다. 실제
fail(E) 타깃 정확도는 미검증 — E 이미지엔 crosshair GT 가 없다([[project_e_images_no_crosshair]]).
consensus 가 proposer recall 을 올린다는 것은 검증됐고, fail 프레임에서 최종 align point 를 더 잘
잡는지는 라이브 검증 과제로 남는다.

## 3. 아키텍처

### 3-A. 컴포넌트

| 파일 | 상태 | 책임 |
| --- | --- | --- |
| `consensus_template.py` | ✅ 구현됨 | `build_consensus_template`(게이트: median→blur 가드→template\|None), `select_routing_templates`(consensus-or-rcp 라우팅 dict) |
| `crop_pipeline.py` | 신규(리팩터) | clean(crosshair/box 제거)+crosshair중심 crop+co-registration 을 `golden_consensus_eval_cond.py` 에서 추출 → eval/prod 공유(표류 방지, §2-F) |
| `consensus_cache.py` | 신규 | 캐시 레이아웃 + 롤링 윈도우: `topup_evict()`, `resolve_consensus_template()`, source-set hash, TTL throttle |
| `consensus_prep.py` | 신규 | 워커 진입점(인자 없음): 큐 drain→coalesce→(TTL?)download 훅→cache top-up/evict→modality별 build→저장+meta |
| workflow_1 fail 핸들러 | 수정(1줄) | fail 시 `{eqp,class,recipe,ts}` 를 `consensus_prep_queue.jsonl` 에 append |
| `live_align_search.py` 호출부 | 수정(소) | `select_routing_templates(...)` 결과를 `templates=` 로 주입 |

### 3-B. 캐시 레이아웃

로컬 SSD 기본, `align_images/`·`align_img_from_rcp` 와 **물리 분리**(race/혼동 없음, §2-C):

```text
align_consensus_cache/<eqp>/<class>/<recipe>/
├─ events/<event_id>/  S0001.jpg + S0001.txt(cond sidecar, 필수)   # 진실 = 원본 이미지
├─ template/  OM.png OM.json  SEM.png SEM.json                      # 파생물(median 재계산 쌈)
└─ state.json   # 롤링 윈도우: event 목록, source-set hash, last_topup_ts, blur 수치, grouping 태그
```

- **저장 키 = `<eqp>`** (provenance: 어느 tool/wafer/lot — eviction·audit·blur 진단). MES recipe
  identity 는 글로벌 공유라 같은 `(class,recipe)` 는 tool 무관 동일 물리 타깃 → pooling 본질적 안전(§2-D).
- **그룹핑 키**(어떤 이미지를 한 median 에 묶나) = **정책 파라미터**: `per-eqp` / `pooled` / `hybrid`.
  default 는 measurement 로 확정(§6).

### 3-C. 데이터 흐름

```text
fail 감지 ─enqueue(hot loop, download 없음)→ consensus_prep_queue.jsonl
                                                  │
consensus_prep.py(워커, 비동기 분리) ─────────────┘
  drain → coalesce(eqp/class/recipe) →
  if now - last_topup_ts > TTL:  download_recent_successes(...) → events top-up, oldest evict(>N=5)
  if source-set hash 변경:  crop_pipeline → build_consensus_template(modality별) → template+meta 저장
                            (또는 None 이면 fallback 마킹)
live_align_search ─ resolve_consensus_template(eqp,class,recipe,modality) → AlignKeyTemplate|None
                   → select_routing_templates(consensus_by_mod, rcp_by_mod) → route_template → 매칭
```

워커는 **prep(준비) 전용, 보정과 비동기 분리**(§2-A): hot loop 무지연·download 격리·재개 가능.

### 3-D. 인터페이스 경계

사용자가 download 구현, Claude 가 Protocol·캐시·빌드·소비 구현:

```python
# 사용자 구현 (MES 쿼리+다운로드)
class SuccessDownloader(Protocol):
    def download_recent_successes(
        self, class_name, recipe_id, *, eqp_scope, max_events, dest_event_dir
    ) -> list[DownloadedEvent]: ...
# DownloadedEvent: {eqp_id, event_id, images[], conds[], timestamp} — cond sidecar 필수
#   (crosshair 좌표 없으면 정렬·clean 불가 → 빌드 실패)

# Claude 구현 (poc/workflow_2)
consensus_cache.topup_evict(class_name, recipe_id, *, downloader, grouping, window, ttl) -> CacheView
consensus_template.build_consensus_template(crops, *, recipe_id, modality, policy) -> ConsensusResult   # ✅
consensus_cache.resolve_consensus_template(eqp, class_name, recipe_id, modality) -> AlignKeyTemplate | None
```

`event_id` = download 가 반환하는 식별자 → 그대로 dedup/eviction 단위.

## 4. 결정사항

### 확정(설계-시점)

- 큐: 별도 `consensus_prep_queue.jsonl`(append-only 한 줄/이벤트). `align_fail_records.csv` 는
  audit log 로 유지(혼합 안 함).
- 윈도우 단위 = 측정 **이벤트**, 크기 **N=5 이벤트**(~10-15장), env 튜닝.
- top-up = **TTL throttle**(~24h, env): TTL 내면 MES 쿼리 skip; 아니면 쿼리→evict→rebuild.
- rebuild = **source-set hash 변경 시에만**(중복 median 연산 회피).
- 원자성: template/state 는 temp→`os.replace`(Windows 원자적).
- 단일 워커 가정(`uv run`, 인자 없음). lockfile 은 YAGNI-deferred.

### 측정-대기(empirical, 파라미터화됨 — 블로커 아님)

- 그룹핑 default(`per-eqp`/`pooled`/`hybrid`) → §6 A/B 로 확정. 검증 전까진 validated 체제(per-eqp).
- blur 임계 `edge_ratio<0.70` / `lap_ratio<0.50` → golden 데이터 1회 실측(저널 `260609_045819` 실행 카드).
  현재 `ConsensusPolicy` 기본값으로 박혀 있고 env 로 조정 가능.

## 5. 에러·엣지 처리

지침: **모든 실패 → rcp graceful degrade.** eval 의 drop taxonomy(`_precrop_drop_reason`)를
`crop_pipeline.py` 로 공유해 재사용.

| 상황 | 워커 동작 | 소비 효과 |
| --- | --- | --- |
| download 전체 실패 | 로그; 캐시 유지; 다음 enqueue 가 TTL 후 재시도 | 이전 윈도우(있으면) 서빙, 없으면 None→rcp |
| download 부분 실패 | 받은 것으로 빌드 | modality ≥min_s 면 빌드, 아니면 None→rcp |
| cond 없음 / crosshair 없음 | 프레임 drop(`missing_cond`/`missing_crosshair`) 집계 | crop 감소 → min_s 미달 가능 |
| OOB / 너무 작은 crop | drop(`crop_failed`) | 〃 |
| modality 미상 | drop(`missing_modality`) | 〃 |
| drop 후 modality < min_s | `build_consensus_template`→None | 그 modality 만 rcp; 다른 modality 는 consensus 가능 |
| blur 가드 실패 | None, reason `blurry`, 비율 meta 기록 | →rcp |
| 워커 크래시 중 빌드 | 멱등 재시작: template 은 파생물 → state.json hash≠event-set hash 면 rebuild | 손상 없음 |
| 중복 enqueue | batch 전 `(eqp,class,recipe)` coalesce | — |

## 6. 테스트 전략 (TDD, 전부 Mac 합성 데이터 — 오피스 데이터 불필요)

- `consensus_template.py` — ✅ 7 tests(게이트 a/b/c/d + 라우팅 폴백).
- `crop_pipeline.py`(리팩터) — 기존 `test_golden_consensus_eval_cond.py`(28)가 **회귀 가드**
  (추출이 crop 산출을 바꾸지 않음); + 합성 프레임 단위 테스트.
- `consensus_cache.py` — `tmp_path` + **fake `SuccessDownloader`**(in-memory 합성 events, DI):
  topup 추가, evict(>N=5 oldest), hash 는 event-set 변경 시에만 바뀜, TTL throttle skip, resolve→template|None.
- `consensus_prep.py` — 큐 drain + coalesce + **멱등 재시작**(두 번 처리→동일 state).
- `live_align_search` 통합 — 기존 mock 으로 consensus template 주입 동작.

**측정-대기(unit 아님, 오피스 golden + 별도 eval)**: 그룹핑 A/B(eval 하니스 재키잉
`eqp/class/recipe`→`class/recipe`, in_topk best); blur 임계 1회 확인.

## 7. 구현 순서 (build order)

1. **`crop_pipeline.py` 추출** — eval 의 clean+crop+co-reg 를 공유 모듈로. 회귀 가드(28 tests)로 안전망.
2. **`consensus_cache.py`** — 레이아웃·롤링 윈도우·hash·TTL·resolve. fake downloader 로 TDD.
3. **`consensus_prep.py`** — 워커: 큐 drain·coalesce·멱등. fake downloader + tmp cache 로 TDD.
4. **`live_align_search` 호출부 배선** — `select_routing_templates` 주입 + 통합 테스트.
5. **workflow_1 fail 핸들러 enqueue** — 1줄(오피스에서 검증).
6. **(측정-대기)** 그룹핑 A/B + blur 임계 확인(golden 데이터 붙는 날).

`consensus_template.py`(게이트·라우터)는 1 이전에 이미 ✅ 완료.

## 8. 미해결/세부 (구현 중 확정)

- 윈도우 크기 N·TTL 의 실측 튜닝(초기값 5/24h 로 시작).
- pooled 시 template 저장 위치(per-eqp dir + meta grouping 태그 권장 vs recipe-level).
- 큐 jsonl 라인 스키마 최종(잠정: `{eqp,class,recipe,ts,alid?}`).
- 네트워크 공유(SMB) 캐시 루트 시 경량 인덱스 — YAGNI-deferred(로컬 SSD 가정).
