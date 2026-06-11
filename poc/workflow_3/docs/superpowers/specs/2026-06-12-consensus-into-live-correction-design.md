# Consensus 템플릿을 실시간 보정 경로에 투입 — 설계

> 날짜: 2026-06-12
> 대상: `poc/workflow_3/align/` 실시간 align-fail 보정 경로
> 상태: 설계 확정(brainstorming) → 구현 계획(writing-plans) 직전
> 관련 검증: `poc/workflow_2/docs/journals/260608/260608_163302_consensus-validated-and-productization-handoff.md`,
> `poc/workflow_2/docs/journals/260529/260529_152818_consensus-ab-verdict-and-next-steps.md`,
> 관련 메모리: `project_consensus_gather_in_loop`, `project_consensus_sparse_golden_and_recipe_id_collision`,
> `project_matcher_flat_chamfer_distinctiveness`, `feedback_ensemble_dev_in_workflow2_then_port`

---

## 0. 한 줄 요약

검증된 consensus(최근 성공 S 이미지의 co-registered median)를 실시간 보정의 *등록 template* 으로
승격한다. recipe 의 해당 modality S 가 `min_s`(=4) 이상이고 blur 가드를 통과하면 consensus 를,
아니면 기존 rcp align key 를 쓴다. matcher(coordinate authority)는 손대지 않는다 — consensus
template 은 rcp 와 동일한 `raw/edge/dt` 필드를 갖는다.

근거(실데이터 cond A/B, recipes=134, S_loo=403, min_s=3): in_topk **0.434 → 0.876 (+0.442)**,
rank1 **0.318 → 0.764 (+0.446)**. 병목은 stale 단일 rcp key 였고, 최근 S median 이 *현재* 공정
외형을 추종해 해소했다.

---

## 1. 범위 / 비범위

**범위(Mac 에서 빌드·테스트 가능):**
- 신규 모듈 `poc/workflow_3/align/consensus_template.py` — bench 의 검증된 순수 build/gate/select
  로직 포팅(bit-parity) + workflow_3 event-cache 레이아웃을 읽는 adapter + resolver.
- `correction.py` / `live_search.py` 의 template 조립 호출부 1곳씩 교체.
- `success_gather.py` 에 cold-cache 동기 대기 헬퍼(`wait_for_gather`) + gather 의 TTL 새로고침.
- `consensus_gather.gather_success_images` 에 TTL freshness 가드.
- `config.py` 에 consensus 노브 추가.
- 합성 downloader/cache 기반 단위·통합 테스트.

**비범위(오피스에서 사용자가 구현/검증):**
- `office_success_downloader.py` 의 FTP 연결/리스팅/복사 실제 구현 — 본 설계는 *계약*만 정의한다
  (이미 `consensus_gather.SuccessDownloader` Protocol 로 명세됨, §5 에서 FTP 관점으로 구체화).
- consensus 의 실데이터 재검증(오피스 A/B) — `feedback_ensemble_dev_in_workflow2_then_port` 규약대로
  bench 에서 이미 검증됨. 본 작업은 *포팅*이며 알고리즘 변경 없음.

---

## 2. 아키텍처 (Approach A — Resolver 모듈)

선택 로직을 한 곳(resolver)에 두고, primary(`correction.py`)와 fallback(`live_search.py`)이
모두 그 resolver 를 통해 template dict 을 받는다. matcher 와 template 머티리얼라이즈(rcp)는 불변.

```
신규  poc/workflow_3/align/consensus_template.py
  ├─ (포팅·bit-parity) build_consensus_template(crops, *, recipe_id, modality, policy) -> ConsensusResult
  │     · blur 가드 내장: reason ∈ {ok, insufficient_s, blurry}; template=None 이면 호출부가 rcp 폴백
  │     · select_routing_templates(consensus_by_mod, rcp_by_mod) -> {"OM":tpl, "SEM":tpl}
  │     · DEFAULT_CONSENSUS_POLICY(min_s, edge_ratio_min=0.70, lap_ratio_min=0.50), ConsensusResult,
  │       _consensus, _sharpness_ratio/_edge_density/_lap_var, CONSENSUS_VERSION
  ├─ (신규) load_coregistered_crops(cache_root, eqp_id, recipe_id, rcp_center_tpls) -> {mod: [crop,...]}
  │     · event-cache 레이아웃 어댑터(bench _build_cond_by_recipe 의 cache 판)
  └─ (신규) resolve_templates(assets, settings) -> {"OM":tpl, "SEM":tpl}
        · cold-cache 시 1회 bounded sync(wait_for_gather) → 재로드 → build → select

수정  align/correction.py        : build_templates_from_assets(...) → resolve_templates(...)
수정  align/live_search.py        : 호출자 template dict 조립 → resolve_templates(...) 경유
수정  monitor/success_gather.py   : + wait_for_gather(eqp, recipe, timeout); gather 에 TTL 적용
수정  align/consensus_gather.py   : gather_success_images 에 refresh TTL freshness 가드
수정  config.py                   : consensus_enabled/min_s/sync_timeout/refresh_ttl, gather_max_events 5→8
불변  align/matching/engine.py    : 좌표 권위 — 변경 없음(consensus template = 동일 필드)
불변  align/templates.py          : rcp 머티리얼라이즈 — 변경 없음(resolver 가 호출만)
```

---

## 3. 데이터 흐름 (실시간 보정)

```
알람 감지
  └─ gather_success_async(eqp, recipe)   [daemon: TTL 미경과면 skip; 경과면 FTP→최신 N(=8) 복사→cache/events/ 교체]
사이클 진행 … run_correction 도달
  └─ resolve_templates(assets, settings):
       if not settings.consensus_enabled: return rcp_by_mod                  # 킬스위치
       cache_key    = f"{assets.class_name}/{assets.recipe_name}"            # ⚠ assets.recipe_id 는 leaf만 — gather 가 쓴 <eqp>/<class>/<recipe> 와 맞추려면 class/recipe (Codex#5)
       rcp_by_mod   = templates.build_templates_from_assets(assets)          # 항상 확보(없으면 no_assets 에스컬레이션). 런타임 라우팅 키=대문자 OM/SEM, box-crop 가능
       center_tpls  = build_center_tpls_for_sizing(assets)                   # consensus crop sizing 전용: 소문자 om/sem, (tpl, offset) center crop — rcp 라우팅 template 과 별개 (Codex#3)
       crops_by_mod = load_coregistered_crops(cache, eqp, cache_key, center_tpls)
       if (관심 modality 의 len(crops) < min_s) and cache_cold:
           if success_gather.wait_for_gather(eqp, cache_key, sync_timeout_sec) is True:  # bool 분기 (Codex#1)
               crops_by_mod = load_coregistered_crops(...)                   # 채워졌을 때만 1회 재로드
           # False(timeout/실패) → 재로드 안 함, 그대로 rcp 폴백(reason=sync_timeout)
       results_by_mod = {mod: build_consensus_template(crops, recipe_id=cache_key, modality=mod, policy)
                         for mod, crops in crops_by_mod.items()}             # ConsensusResult: gate 내장(insufficient_s/blurry → template=None)
       # 로그는 ConsensusResult 로, 라우팅엔 .template 만 (Codex#4)
       tpl_by_mod = {mod: r.template for mod, r in results_by_mod.items() if r.template is not None}
       templates  = select_routing_templates(tpl_by_mod, rcp_by_mod)        # modality별 consensus or rcp (AlignKeyTemplate 만)
       return templates
  └─ correct_align_fail_auto: route_template(templates, read_mode()) → match → reposition + OK
```

**타이밍 race 처리:** correction 은 알람 후 수 초 내(여러 GUI step 뒤)에 돈다. 이 fail 의 async gather
완료를 보장하지 않으므로, 기본은 *이전에 캐시된* consensus 를 쓴다(이 fail 의 gather 는 다음을 위해
캐시를 데움). cache cold(해당 recipe 최초 fail) 일 때만 §4-C 의 bounded sync 로 한 번 채운다.

**modality 의존:** consensus build 는 rcp center template 으로 crop size/box-offset 을 정한다
(bench `_build_cond_by_recipe` 가 `center_tpls` 를 받는 것과 동일). 따라서 rcp 자산은 항상 먼저
resolve 한다 — rcp 가 없으면 consensus 도 없고 폴백도 없다(기존 `no_assets` 에스컬레이션 유지).

---

## 4. 컴포넌트 상세

### 4-A. 포팅(bit-parity) — build/gate/select
bench `poc/workflow_2/consensus_template.py` 의 순수 함수를 workflow_3 로 가져온다. 알고리즘 변경
금지(`feedback_ensemble_dev_in_workflow2_then_port`). 포팅 surface:
- `build_consensus_template(crops, *, recipe_id, modality, policy)` — median + blur 가드 내장,
  `ConsensusResult(template, modality, n, edge_ratio, lap_ratio, reason)` 반환.
- `select_routing_templates(consensus_by_mod, rcp_by_mod)` — modality별 `consensus or rcp` dict 조립
  (route_template 은 불변; dict 조립 시점 폴백 — 저널 163302 §4-A #6). **입력은 `AlignKeyTemplate`(또는
  None)만** — resolver 가 `ConsensusResult.template` 만 넣고 `ConsensusResult` 객체 자체를 넣지 않는다
  (그러면 truthy non-template 이 matcher 로 흘러감 — Codex#4). 결정 로그는 `ConsensusResult` 로 따로 남긴다.
- `DEFAULT_CONSENSUS_POLICY`(min_s, edge_ratio_min=0.70, lap_ratio_min=0.50), `_consensus`,
  `_sharpness_ratio`/`_edge_density`/`_lap_var`, `CONSENSUS_VERSION`.
- min_s 는 `policy` 로 주입 — resolver 가 `settings.consensus_min_s` 로 policy 구성.

### 4-B. 신규 — event-cache adapter (bench `_build_cond_by_recipe` 의 cache 판, bit-parity)
`load_coregistered_crops(cache_root, eqp_id, cache_key, center_tpls) -> {mod: [gray_crop,...]}`
- `cache_key = "<class>/<recipe>"` — `consensus_gather._events_dir_for` 로 events/ 경로 →
  `<event_id>/S*` 순회(event_id 시각 prefix = 이름정렬=시간정렬, 최신 우선 cap = `gather_max_events`).
- **modality 분류는 bench 와 동일하게 `_resolve_mod(cond, recipe_mod)`** — bare `msr_modality(cond)` 가
  아니다. `recipe_mod` = center_tpls 가 단일 modality 면 그것(둘 다면 None). 이 폴백이 없으면 단일-modality
  recipe 의 S 가 조용히 drop 된다(Codex#3). `recipe_mod` 는 center_tpls 로 산출.
- **crop 은 clean→crosshair중심→고정 size 순서로 bench 와 동일**: 각 S 를 정제(crosshair/직선 제거)한 뒤
  `_cond_crosshair_xy(cond)` 중심으로 해당 modality center tpl 크기(`size_wh`)로 `_cond_consensus_crop`
  (OOB/과소 → None drop). center tpl 은 **소문자 om/sem 키의 (tpl, offset) center crop** 으로, 런타임
  rcp 라우팅 template(대문자, box-crop 가능)과 *별도로* 만든다(`build_center_tpls_for_sizing`) — 둘을
  섞으면 crop geometry 가 달라져 bit-parity 깨짐(Codex#3).
- modality별로 `coregister_crops`(sub-pixel 정렬)로 median blur 저감(bench 와 동일; crop 내용만
  변경, full-frame/GT 불변).
- drop 사유(modality 미해결/crosshair 없음/OOB/`load_gray` 실패)는 조용히 버리지 않고 count + `[INFO]`
  1줄. 디코드 실패 S 는 drop(전부 실패면 캐시 보존 — §4-D).
- **공유 헬퍼 처리:** 정확히 bench 의 `_resolve_mod`/`_cond_crosshair_xy`/`_cond_consensus_crop`/
  `coregister_crops`/`_precrop_drop_reason` 을 재사용한다. 이들은 현재 bench 드라이버
  (`golden_consensus_eval_cond.py`)·`align_similarity.py` 에 있으므로, workflow_3 align 의 공용 위치로
  단일화하고 bench 가 거기서 import 하도록 정리한다(중복/재구현 금지 — 재구현하면 bit-parity 위험).
  정확한 배치/이동은 plan 에서 확정.

### 4-C. 신규 — resolver + cold-cache bounded sync
`resolve_templates(assets, settings) -> {"OM":tpl, "SEM":tpl}` (§3 의사코드).
- `consensus_enabled=False` → 즉시 rcp dict 반환(킬스위치, 현행 동작).
- cold-cache 정의: 관심 modality crop 수 < `min_s` *그리고* events/ 부재/희박.
- `success_gather.wait_for_gather(eqp, cache_key, timeout) -> bool`: **lock 안에서 thread 스냅샷만 뜨고
  lock 을 푼 뒤 `join(timeout)`** 한다. lock 을 쥔 채 join 하거나 `gather_success_async`(같은 `_IN_FLIGHT_LOCK`
  취득)를 재호출하면 데드락(Codex#1). 스냅샷된 thread 가 살아있으면 join; 없고 cache cold 면
  `gather_success_async` 를 lock 밖에서 1회 fire 후 그 thread join. 반환 bool = join 후 캐시가
  실제로 채워졌는지(events/ 에 ≥min_s 가능). 이미 알람 시점에 async fire 됐으므로 보통은 *그 스레드를
  join*(중복 fetch 방지). resolver 는 correction 당 최대 1회 호출.
- 반환이 `True` 일 때만 crop 재로드. `False`(timeout/실패) → 재로드 없이 rcp(`reason=sync_timeout`),
  백그라운드 gather 가 다음을 위해 캐시 데움.
- modality별 선택 결과를 `[INFO]` 1줄로(`consensus` / `rcp:insufficient_s(n)` / `rcp:blurry` /
  `rcp:sync_timeout` / `rcp:disabled`) 남겨 manifest/콘솔에서 결정이 보이게 한다.

### 4-D. gather TTL 새로고침 + 원자적 교체
`gather_success_images(..., refresh_ttl_sec=...)`: 다운로드 전 events/ mtime 확인 → TTL 미경과면
다운로더 호출 없이 `reason="fresh"` 조기반환(기존 캐시 재사용). 경과면 replace-if-non-empty
(최신 N 으로 통째 교체) 수행. 교체 의미는 유지 — recency 추종이 consensus 의 핵심이라 누적 금지.

**원자적 교체(Codex#2):** 현행은 `rmtree(events_dir)` 후 `staging.replace(events_dir)` 라, 그 사이
동시 correction 의 adapter 가 events/ 부재를 관측할 수 있다. 수정:
- swap 을 reader/writer 관점에서 안전하게 — events/ 를 지우지 말고 `staging → events` 를 **단일
  rename(os.replace)** 로 덮어쓰거나(같은 볼륨), 불가하면 `events.new` 로 rename 후 기존을 `events.old`
  로 비키고 `events.new→events` 한 뒤 `events.old` 정리(중간에 항상 유효한 events/ 존재).
- adapter 는 events/ 부재/일시적 빈 상태를 "0 crop → rcp 폴백" 으로 안전 처리(읽기 중 OSError 무시).
- **"non-empty" 정의를 실제 stage 된 S 이미지 수로** — `len(staged)>0` 만으로 판단하면 이미지 0장짜리
  event 가 멀쩡한 옛 캐시를 덮어쓸 수 있다(Codex#2). `n_images >= 1`(가능하면 modality별 ≥1) 일 때만 교체,
  아니면 옛 캐시 보존.

---

## 5. Downloader 계약 (FTP, 오피스 구현)

`consensus_gather.SuccessDownloader` Protocol 을 FTP 관점으로 구체화(구현은 오피스):
- `download_recent_successes(recipe_id, *, max_events, dest_dir) -> list[StagedEvent]`.
- 공유/FTP 경로에서 recipe 의 최근 성공(S) 측정 디렉터리를 나열 → 최신 `max_events` 건을
  `dest_dir/<event_id>/` 에 `S*.jpeg` + 숨김 `.<img>/cond.txt` 로 복사. align-fail 측정 자체는 제외.
- `event_id = yyyymmdd_hhmmss_<recipe>_<lot>`(시각 prefix), Windows 금지문자 치환.
- modality 추론 키 보존: cond 에 `!OM_Brightness`/`Accelerating_voltage`/`Magnification` 유지
  (`cond_file.msr_modality` 가 소비; msr 에는 `Scope` 없음 — 2026-06-08 확인).
- build 로직 금지 — downloader 는 파일 stage 만. median/가드는 workflow_3 가 담당.
- 부재/예외 시 `gather_success_async` 가 조용히 skip(현행) → consensus 휴면 → rcp 폴백.

---

## 6. Config (Workflow3Settings, 모두 env override)

| field | default | env | 용도 |
|---|---|---|---|
| `consensus_enabled` | `True` | `ALIGN_FAIL_CONSENSUS` | 마스터 토글; off → 순수 rcp(현행). 롤아웃 킬스위치 |
| `consensus_min_s` | `4`(floor 3) | `ALIGN_FAIL_CONSENSUS_MIN_S` | modality별 build·신뢰 최소 S |
| `gather_max_events` | `5 → 8` | `ALIGN_FAIL_GATHER_MAX_EVENTS` | OM/SEM split 후도 ≥4 남도록(=GATHER_MAX_EVENTS 동기) |
| `consensus_sync_timeout_sec` | `8.0` | `ALIGN_FAIL_CONSENSUS_SYNC_TIMEOUT` | cold-cache bounded 대기 |
| `consensus_refresh_ttl_sec` | `21600`(6h) | `ALIGN_FAIL_CONSENSUS_REFRESH_TTL` | gather 재fetch TTL |

blur 가드 임계(0.70/0.50)는 검증된 값이라 `consensus_template.py` 모듈 상수로 둔다(env 비노출, YAGNI).

**⚠ min_s 정책 변경 명시(Codex#5):** §0 의 검증 수치(+0.442 in_topk / +0.446 rank1)는 **`min_s=3`**
오피스 A/B 결과다. production 기본 `consensus_min_s=4` 는 *의도된 롤아웃 정책 변경* 이다 — 더 강한 median
(S=3 의 2장 median 보다 변별력↑)을 노리되, S=3 만 모이는 recipe 는 consensus 대신 rcp 로 폴백한다
(커버리지↓ 가능). bench 수치를 4 에 그대로 귀속하지 말 것. 오피스 롤아웃 시 `ALIGN_FAIL_CONSENSUS_MIN_S`
로 3↔4 를 A/B 해 커버리지 대 정확도 trade-off 를 실측 후 확정. floor 는 3(LOO 바닥 fm≥3).

---

## 7. 에러 처리 — 모든 실패는 rcp 로 강등, 사이클 비중단

| 상황 | 동작 |
|---|---|
| `consensus_enabled=False` / downloader 부재 / 빈 캐시 | 순수 rcp(consensus 휴면) |
| FTP/gather 예외 | 캐시 보존, 로그, rcp 폴백 |
| build 예외 | 로그, rcp 폴백 |
| `S < min_s` | rcp(`reason=insufficient_s`) |
| blur 가드 미통과 | rcp(`reason=blurry`) |
| cold-cache sync timeout | 이번 rcp, 백그라운드가 다음 데움 |
| rcp 자산 자체 없음 | 기존 `no_assets` 에스컬레이션(불변) |
| 캐시 S 전부 디코드 실패 | 옛 캐시 보존(§4-D), adapter 0 crop → rcp |
| adapter/build 예외(modality별) | per-modality try/except → 그 modality 만 rcp |
| `cache_key` 불일치(leaf vs class/recipe) | resolver 가 `f"{class_name}/{recipe_name}"` 사용 — gather/downloader 와 동일 키(Codex#5) |

→ office downloader 가 없어도 안전하게 선배포 가능(캐시 안 차면 자동으로 rcp).

**recipe_id 키 일관성(Codex#5):** `AlignFailAssets.recipe_id` 는 leaf(`recipe_name`)만 반환한다. 그러나
gather(monitor 알람 경로)는 RECIPE_ID=`<class>/<recipe>` 로 cache 를 적재하므로
(`project_recipe_id_class_recipe_format`), resolver/adapter 는 반드시 `f"{assets.class_name}/{assets.recipe_name}"`
를 cache/downloader 키로 써야 한다. leaf 만 쓰면 캐시를 못 찾거나(조용한 miss) eqp 내 동명 recipe 끼리
충돌한다. build_consensus_template 의 `recipe_id` 인자도 같은 키로 통일.

---

## 8. 테스트 (Mac, 합성 — 오피스 의존 0)

- 합성 `SuccessDownloader`: modality별 cond 가진 가짜 S event 를 temp cache 에 적재.
- adapter: crop 수/modality 필터/cap/ drop-count; co-registration 호출.
- build/gate(포팅): bench 픽스처와 bit-parity(재export 확인); sharp 통과 / blurry 기각 / insufficient_s.
- select: modality별 `consensus or rcp` 정확.
- resolver: enough→consensus, `<min_s`→rcp, blurry→rcp, cold→`wait_for_gather` 호출(mock),
  TTL fresh→fetch skip, `consensus_enabled=False`→rcp.
- `wait_for_gather`: in-flight join + timeout(False) 경로 + lock 밖 join(데드락 회귀 방지) + 반환
  bool 에 따른 재로드 분기.
- gather TTL: 미경과 skip(`reason=fresh`) / 경과 replace; **이미지 0장 event 는 옛 캐시 보존**(non-empty=
  S 이미지 수); 교체 중 events/ 항상 유효(원자 swap).
- cache_key: leaf 가 아닌 `class/recipe` 로 cache 경로 구성 — gather 가 쓴 위치와 일치 / leaf 충돌 미발생.
- adapter: `_resolve_mod` 폴백으로 단일-modality recipe S 가 안 드롭됨; 디코드 실패 S drop; 전부 실패→0 crop.
- routing: `ConsensusResult.template`(AlignKeyTemplate) 만 select 에 들어감(ConsensusResult 객체 미유입).
- 통합: `correct_align_fail_auto` 가 합성 cache 로 consensus 선택 / 빈 cache 로 rcp 폴백.

---

## 9. 롤아웃

1. consume 측(adapter/build/resolver/routing/TTL/config) 포팅 + 합성 테스트 통과 → main.
   `consensus_enabled=True` 여도 office downloader 가 없으면 캐시가 안 차 자동 rcp 폴백 → 무해.
2. 오피스에서 `office_success_downloader.py`(FTP, §5 계약) 구현 → 캐시 적재 시작.
3. 오피스 점검: 캡처 옆 feasibility 마킹 + manifest 의 modality별 결정 로그로 consensus 채택률·폴백
   사유 모니터. 이상 시 `ALIGN_FAIL_CONSENSUS=0` 즉시 롤백.
