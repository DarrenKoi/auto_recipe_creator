# consensus 프로덕션화 — 수집/저장 설계 결정사항 (brainstorming 중간 정리)

날짜: 2026-06-09 05:35
대상: 검증된 consensus(최근 S median)를 matcher/`live_align_search`의 실제 등록 template로 승격하는
productization 의 **데이터 수집·저장 전략** 설계. brainstorming §1·§2 합의 완료, §3 이전 중간 저장.
선행: 같은 날 `260609_045819_blur-guard-interpretation-and-pending-check.md`, consensus 검증
저널 `260608_163302_consensus-validated-and-productization-handoff.md`.

---

## 0. 한 줄 요약

on-demand(fail 트리거) + 분리된 prep 워커 + 롤링 캐시 구조로 합의. 핵심 제약: recipe 수만 개라
**미리 다 못 만든다**, 한 측정 이벤트당 success 이미지 **2~3장뿐**이라 과거 여러 이벤트를 합쳐 9~10장.
**§3(가드·에러·테스트·golden 검증 단계)는 다음 세션.**

---

## 1. 문제 정의 (사용자가 제기한 실제 제약)

- consensus 소스 = **golden S(MES 성공 이력)**. fail 폴더 S(편향)·실시간 S(라벨 불신) 배제 — 확정.
- recipe가 **수만 개** 구동 → 모든 recipe의 consensus를 **미리 모아둘 수 없다**. 반드시 on-demand.
- 한 측정 이벤트의 success 이미지는 **2~3장뿐** → 9~10장 모으려면 과거 측정 **여러 건**(서로 다른
  wafer/lot/시간, 그리고 여러 tool)을 합쳐야 함.
- **download(MES 쿼리+다운로드)는 사용자가 구현 가능.** Claude 는 캐시 관리+빌드+소비 담당.
- `align_fail_alarm_record.py` 는 "production 에 근접한 진입점의 *예시*"로 지목된 것 — 그 hot loop
  안에 반드시 넣자는 의미 아님.

## 2. 결정된 사항 (이번 세션 합의)

### 2-A. 오케스트레이션 = **분리된 prep 큐 (Approach A)**
- fail 핸들러(production-near)는 실패 recipe를 **prep 큐에 등록만** (download/build 안 함).
- 별도 prep 워커(`consensus_prep.py`, `uv run`, 인자 없음)가 큐 소비: ①download 훅 → ②캐시
  top-up/evict → ③modality별 consensus 빌드 → ④저장/폴백 마킹.
- 보정(`live_align_search`)은 캐시된 template만 읽고, 없거나 가드 실패면 **rcp 폴백**.
- **목적 = prep(준비), 보정과 비동기 분리** (사용자 선택). hot loop 무지연, download 격리, 재개 가능.
- 반려: B(fail 핸들러 인라인 — download 의존성·지연을 알람 루프에 재유입), C(보정 시점 lazy build —
  지연을 없애려던 자리에 재도입).

### 2-B. 신선도 정책 = **최신 N건 롤링 캐시**
- recipe별 저장소에 success를 쌓되, prep 때 새 성공만 top-up, 가장 오래된 건 evict(롤링 윈도우).
- 항상 최근 외형 유지(consensus의 drift 추종 본질) + 전체 재다운로드 회피.
- **윈도우 단위 = 이미지가 아니라 측정 *이벤트*** → wafer/lot/시간/tool 분산이 자연 확보
  (plan §6 "한 wafer 10장 ❌" 충족).
- 반려: 매번 전체 재빌드(재다운로드 비용), 축적 후 동결(stale → drift 장점 상실).

### 2-C. 저장 내용/위치 = **원본 이미지 = 진실, 전용 루트**
- 신규 루트 `align_consensus_cache/<eqp>/<class>/<recipe>/` 에 success 원본(+cond.txt) 보관.
- template은 **파생물**(median이라 재계산 쌈); meta에 grouping·source event 목록·source-set hash·
  blur 수치 기록.
- `align_images/`·`align_img_from_rcp` 와 **물리 분리** → race/혼동 없음(저널 4-B #3 제약 충족).
- 레이아웃:
  ```
  align_consensus_cache/<eqp>/<class>/<recipe>/
  ├─ events/<event_id>/  S0001.jpg…  S0001.txt…   # 한 이벤트의 success + cond sidecar(필수)
  ├─ template/  OM.png OM.json  SEM.png SEM.json   # 파생물
  └─ state.json                                     # 롤링 윈도우 상태
  ```
- **캐시 루트는 로컬 디스크(SSD) 기본.** 네트워크 공유(SMB)면 stat-heavy 글롭이 느려짐 → 그 경우만
  경량 인덱스(`recipe→eqp 목록`) 추가. 인덱스는 **YAGNI/deferred**.

### 2-D. eqp 키 — **저장 키와 그룹핑 키 분리**
사용자 지적: 같은 class/recipe를 여러 tool이 구동.
- **저장 키 = `<eqp>` 유지** — download가 tool별 이벤트 단위라 provenance(어느 tool/wafer/lot) 필요
  (eviction·audit·blur 진단용).
- **그룹핑 키(어떤 이미지를 한 median에 묶나) = 정책 파라미터로 분리.** per-eqp / pooled(recipe로
  합침) / hybrid.
- **MES recipe identity = 글로벌 공유(동일 타깃)** 확인 → 같은 (class,recipe)는 tool 무관 같은 물리
  타깃 → **pooling 본질적 안전.** rcp align-key 교차매칭은 필수 게이트가 아니라 값싼 sanity/audit로
  격하, blur 가드가 최종 backstop.
- **sister-tool 발견 메커니즘**: 진실은 MES 쿼리(download가 (class,recipe)로 tool 횡단 질의).
  캐시 글롭 `align_consensus_cache/*/<class>/<recipe>/` 는 *이미 캐시된* 이벤트 인벤토리(재다운로드
  회피)용. 글롭은 와일드카드가 eqp 레벨 1곳뿐(`**` 재귀 아님) → `O(tool 대수)` stat → 로컬 수 ms,
  게다가 decoupled 워커라 무해.

### 2-E. 인터페이스 경계 (당신 ↔ Claude = download 훅 하나)
```python
# 사용자 구현. Protocol 만 Claude 정의.
class SuccessDownloader(Protocol):
    def download_recent_successes(
        self, class_name, recipe_id, *, eqp_scope, max_events, dest_event_dir
    ) -> list[DownloadedEvent]: ...   # {eqp_id, event_id, images[], conds[], timestamp}
# Claude 구현 (poc/workflow_2)
consensus_cache.topup_evict(class_name, recipe_id, *, grouping, window) -> CacheView
build_consensus_template(events, modality, *, policy=DEFAULT_POLICY) -> ConsensusResult
resolve_consensus_template(eqp, class_name, recipe_id, modality) -> AlignKeyTemplate | None
```
- **download 훅은 cond.txt sidecar까지 필수 반환** (crosshair 좌표 없으면 정렬·clean 불가 → 빌드 실패).

### 2-F. 컴포넌트 (신규 3 + 수정 2 + 리팩터 1)
| 종류 | 대상 | 책임 |
|---|---|---|
| 신규 | `consensus_cache.py` | 캐시 레이아웃 + 롤링 윈도우(이벤트 단위 top-up/evict/dedup) |
| 신규 | `consensus_template.py` | `build_consensus_template` — crop(clean+정렬+co-reg)→median→가드, modality 분리 |
| 신규 | `consensus_prep.py` | 워커 진입점(큐 소비→download 훅→cache→build→저장) |
| 수정 | `poc/workflow_1/` fail 핸들러 | fail 시 prep 큐 enqueue(한 줄) |
| 수정 | `live_align_search.py` | `route_template`/dict 구성에서 `consensus or rcp` 우선 |
| 리팩터 | `golden_consensus_eval_cond.py` 의 crop 파이프라인 | crosshair clean+crop+co-reg 를 공유 모듈로 추출 → eval/prod 표류 방지 |

## 3. 다음에 결정/진행할 사항 (§3 이후)

1. **§3 — 가드·에러 처리·폴백 상세** (다음 세션 시작점):
   - min_s≥3 + blur 가드(edge<0.70 or lap<0.50 → None=rcp 폴백) 적용 지점/임계 확정.
   - download 실패/부분 실패, cond 없음, OOB crop, modality 미상 시 동작.
   - 큐 중복 coalesce, 워커 재시작/중단·재개, template stale 판정(source-set hash).
2. **테스트 설계(TDD)**: 합성 crop 으로 build_consensus_template RED→GREEN(S≥3/ S<3 None/ blur 낮음
   None/ modality 분리), route_template consensus 우선·rcp 폴백, 워커 큐 처리.
3. **★ golden A/B — 그룹핑 default 확정** (구현 전 측정): eval 하니스를 재키잉
   (`eqp/class/recipe` → `class/recipe`)해 **per-eqp / pooled / hybrid** 비교. blur 가드 유지하며
   in_topk best인 걸 default. 검증 전까진 validated 체제(per-eqp) 유지.
   - hybrid 후보 정의: 실패 tool 롤링 윈도우가 modality별 ≥min_s → 그 tool만; 미달 → 같은
     class/recipe 의 sister-tool 이벤트로 보충.
4. **blur 가드 수치 1회 확인**(별도, golden 데이터 붙는 날 — `260609_045819` 참조): default 임계
   0.70/0.50 의 실측 근거.
5. **세부 미정**: 윈도우 크기 N(이벤트 수, 잠정 ~4~5 이벤트→9~12장), event_id dedup 키 정의,
   pooled 시 template 저장 위치(per-eqp dir + meta grouping 태그 vs recipe-level), prep 큐 파일
   포맷(신규 jsonl vs 기존 `align_fail_records.csv` 재사용 — §1 미확정으로 남김).

## 4. 다음 세션 재개 지점

brainstorming 스킬 진행 중(HARD-GATE: 설계 승인 전 구현 금지). §1(컴포넌트)·§2(데이터 흐름·인터페이스)
사용자 승인 완료. **§3(가드·에러·테스트·golden 검증 단계) 제시 → 승인 → 설계 doc
작성(`docs/superpowers/specs/`) → writing-plans** 순으로 이어가면 됨.
