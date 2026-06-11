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
       rcp_by_mod   = templates.build_templates_from_assets(assets)          # 항상 확보(없으면 no_assets 에스컬레이션)
       crops_by_mod = load_coregistered_crops(cache, eqp, recipe, rcp_by_mod)
       if (관심 modality 의 len(crops) < min_s) and cache_cold:
           success_gather.wait_for_gather(eqp, recipe, sync_timeout_sec)     # in-flight thread join(bounded), 1회만
           crops_by_mod = load_coregistered_crops(...)                       # 1회 재로드
       cons_by_mod = {mod: build_consensus_template(crops, recipe_id=key, modality=mod, policy)
                      for mod, crops in crops_by_mod.items()}                # 내부 gate: insufficient_s/blurry → template=None
       templates   = select_routing_templates(cons_by_mod, rcp_by_mod)       # modality별 consensus or rcp
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
  (route_template 은 불변; dict 조립 시점 폴백 — 저널 163302 §4-A #6).
- `DEFAULT_CONSENSUS_POLICY`(min_s, edge_ratio_min=0.70, lap_ratio_min=0.50), `_consensus`,
  `_sharpness_ratio`/`_edge_density`/`_lap_var`, `CONSENSUS_VERSION`.
- min_s 는 `policy` 로 주입 — resolver 가 `settings.consensus_min_s` 로 policy 구성.

### 4-B. 신규 — event-cache adapter
`load_coregistered_crops(cache_root, eqp_id, recipe_id, rcp_center_tpls) -> {mod: [gray_crop,...]}`
- `consensus_gather._events_dir_for` 로 events/ 경로 → `<event_id>/S*` 순회(event_id 시각 prefix =
  이름정렬=시간정렬, 최신 우선 cap = `gather_max_events`).
- 각 S: `cond_file.load_cond` → `cond_file.msr_modality(cond)` 로 modality 분류 → 해당 modality
  rcp center tpl 크기로 crosshair-중심 고정-size crop(bench `_cond_consensus_crop` 동등) →
  modality별 누적.
- modality별로 `coregister_crops`(sub-pixel 정렬)로 median blur 저감(bench 와 동일; crop 내용만
  변경, full-frame/GT 불변).
- drop 사유(modality 미해결/crosshair 없음/OOB/로드실패)는 조용히 버리지 않고 count + `[INFO]` 1줄.
- **공유 헬퍼 처리:** `_cond_consensus_crop`/crosshair-xy/`coregister_crops` 등 crop 생성 헬퍼는
  현재 bench 드라이버(`golden_consensus_eval_cond.py`)·`align_similarity.py` 에 있다. 포팅 시
  workflow_3 align 의 공용 위치(예: `clean_align_image.py`/`cond_template.py` 또는 신규 helper)로
  단일화하고 bench 가 거기서 import 하도록 정리한다(중복 구현 금지). 정확한 배치/이동은 plan 에서 확정.

### 4-C. 신규 — resolver + cold-cache bounded sync
`resolve_templates(assets, settings) -> {"OM":tpl, "SEM":tpl}` (§3 의사코드).
- `consensus_enabled=False` → 즉시 rcp dict 반환(킬스위치, 현행 동작).
- cold-cache 정의: 관심 modality crop 수 < `min_s` *그리고* events/ 부재/희박.
- `success_gather.wait_for_gather(eqp, recipe, timeout) -> bool`: `_IN_FLIGHT[(eqp,recipe)]` 살아있으면
  `join(timeout)`; 없고 cache cold 면 1회 fire+join. 이미 알람 시점에 async fire 됐으므로 보통은
  *그 스레드를 join* 한다(중복 fetch 방지). resolver 는 correction 당 최대 1회만 호출.
- timeout 초과 → 이번엔 rcp, 백그라운드 gather 가 다음을 위해 캐시 데움.
- modality별 선택 결과를 `[INFO]` 1줄로(`consensus` / `rcp:insufficient_s(n)` / `rcp:blurry` /
  `rcp:sync_timeout` / `rcp:disabled`) 남겨 manifest/콘솔에서 결정이 보이게 한다.

### 4-D. gather TTL 새로고침
`gather_success_images(..., refresh_ttl_sec=...)`: 다운로드 전 events/ mtime 확인 → TTL 미경과면
다운로더 호출 없이 `reason="fresh"` 조기반환(기존 캐시 재사용). 경과면 현행 replace-if-non-empty
(최신 N 으로 통째 교체) 수행. 교체 의미는 유지 — recency 추종이 consensus 의 핵심이라 누적 금지.

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

→ office downloader 가 없어도 안전하게 선배포 가능(캐시 안 차면 자동으로 rcp).

---

## 8. 테스트 (Mac, 합성 — 오피스 의존 0)

- 합성 `SuccessDownloader`: modality별 cond 가진 가짜 S event 를 temp cache 에 적재.
- adapter: crop 수/modality 필터/cap/ drop-count; co-registration 호출.
- build/gate(포팅): bench 픽스처와 bit-parity(재export 확인); sharp 통과 / blurry 기각 / insufficient_s.
- select: modality별 `consensus or rcp` 정확.
- resolver: enough→consensus, `<min_s`→rcp, blurry→rcp, cold→`wait_for_gather` 호출(mock),
  TTL fresh→fetch skip, `consensus_enabled=False`→rcp.
- `wait_for_gather`: in-flight join + timeout 경로.
- gather TTL: 미경과 skip(`reason=fresh`) / 경과 replace.
- 통합: `correct_align_fail_auto` 가 합성 cache 로 consensus 선택 / 빈 cache 로 rcp 폴백.

---

## 9. 롤아웃

1. consume 측(adapter/build/resolver/routing/TTL/config) 포팅 + 합성 테스트 통과 → main.
   `consensus_enabled=True` 여도 office downloader 가 없으면 캐시가 안 차 자동 rcp 폴백 → 무해.
2. 오피스에서 `office_success_downloader.py`(FTP, §5 계약) 구현 → 캐시 적재 시작.
3. 오피스 점검: 캡처 옆 feasibility 마킹 + manifest 의 modality별 결정 로그로 consensus 채택률·폴백
   사유 모니터. 이상 시 `ALIGN_FAIL_CONSENSUS=0` 즉시 롤백.
