# consensus S-image gather — workflow_3 루프 통합 (2026-06-10)

plan 문서 `poc/workflow_2/docs/superpowers/plans/2026-06-10-consensus-gather-in-loop.md` 의
Task 1–5 를 subagent-driven development(작업별 구현 → 스펙 리뷰 → 품질 리뷰 → 수정 → 재리뷰)로
전부 실행하고 main 에 push 까지 완료한 세션.

## 1. 진행 사항

- **Task 1** — `ALIGN_CONSENSUS_CACHE_DIR` 루트 상수 추가 (`poc/workflow_3/__init__.py`,
  env override 가능, 기본 `poc/workflow_3/align_consensus_cache`). (581065a)
- **Task 2** — vision 순수 orchestration `gather_success_images` + self-test.
  disk layout 의 주인: events_dir 결정 → `.events_staging` 에 다운로더로 stage →
  ≥1 event 면 `events/` 로 atomic swap(replace-if-non-empty). (2fabc67)
- **Task 3** — `Workflow3Settings.gather_enabled`(env `ALIGN_FAIL_GATHER_SUCCESS`, 기본 on) /
  `gather_max_events`(env `ALIGN_FAIL_GATHER_MAX_EVENTS`, 기본 5). (2cb29c4)
- **Task 4** — monitor glue `gather_success_async`: office 다운로더 2단 fallback 로딩
  (`monitor/office_success_downloader` → legacy `workflow_1/`), daemon thread 비차단 fire,
  게이트(enabled/recipe_id/downloader) 내부 판정, 예외 삼킴 + `log_work2_event`. (e7d760b)
- **Task 5** — 루프 통합: `process_fail_rows` popup 직후 `gather_success_async(eqp_id,
  info["recipe_id"], settings)` 1줄. (7e152df)
- **리뷰 루프가 plan 을 넘어 개선한 것:**
  - swap/카운트 예외도 `GatherResult` 로 흡수 — `error:swap:<Type>` reason 신설.
    Windows 잠금 파일로 `Path.replace` 가 raise 해도 루프/계약 불사. (2e158b4)
  - 같은 (eqp, recipe) in-flight gather dedupe — `_IN_FLIGHT` 레지스트리 + lock,
    `thread.start()` 까지 lock 안(등록-시작 TOCTOU 창 제거). (00a1403, d3158e8)
  - office 핸드오프 마감: SuccessDownloader cond 계약 docstring, README env/산출물/체크리스트,
    `align_consensus_cache/` gitignore(staged S*.txt = fab 데이터 유출 방지). (a44f51f)
- **설계 결정 2건 (사용자 확정):**
  - `event_id` = msr 측정이력의 유니크 string 그대로: `yyyymmdd_hhmmss_<recipe_name>_<lot_id>`.
    시각 prefix 라 이름 정렬 = 시간 정렬, 같은 측정 재수집 시 같은 id(stable).
  - 레이아웃은 **`events/` 유지** (flat 안 기각): `events/` 는 장식이 아니라 통째 atomic swap 의
    단위 — 빼면 event 단위 swap + 이름 패턴 prune 으로 재설계해야 해서 비용 > 이득.
- 테스트 전부 green: vision `test_consensus_gather.py` **8/8**, monitor
  `test_success_gather.py` **6/6**, `align_fail_monitor` import 스모크, 기존 회귀
  `test_align_fail_correct` 6/6 · `test_align_key_match` 10/10.
- main push 완료 (최종 45a15c1). 병행 세션의 workflow_2 eval 커밋(d31558c, 67ac825)과
  같은 repo 에서 충돌 없이 진행(파일별 stage 만 사용).

## 2. 수정 내용

| 파일 | 변경 |
|---|---|
| `poc/workflow_3/__init__.py` | `ALIGN_CONSENSUS_CACHE_DIR` 상수 + `__all__` |
| `poc/workflow_3/vision/consensus_gather.py` | 신규 — `gather_success_images` / `StagedEvent` / `GatherResult` / `SuccessDownloader` Protocol (+ event_id·cond 계약 docstring) |
| `poc/workflow_3/vision/test_consensus_gather.py` | 신규 — 합성 self-test 8 케이스 |
| `poc/workflow_3/config.py` | gather 게이트 필드 2개 + env 배선 + cross-ref 주석 |
| `poc/workflow_3/monitor/success_gather.py` | 신규 — office loader 2단 fallback + daemon fire + in-flight dedupe |
| `poc/workflow_3/monitor/test_success_gather.py` | 신규 — self-test 6 케이스 |
| `poc/workflow_3/monitor/align_fail_monitor.py` | import 1줄 + `process_fail_rows` 호출 1줄 |
| `poc/workflow_3/README.md` | gather 단락 + env 3종 + 산출물 경로 + office 체크리스트 항목 |
| `.gitignore` | `poc/workflow_3/align_consensus_cache/` |

## 3. office_success_downloader 다운로드 경로 (계약 명세)

**다운로더는 경로를 스스로 정하지 않는다.** 호출부(`gather_success_images`)가 넘기는
`dest_dir` 인자 아래에만 쓴다. `dest_dir` 의 실체는 임시 staging 디렉토리:

```
poc/workflow_3/align_consensus_cache/<eqp_id>/<class>/<recipe>/.events_staging/
```

다운로더가 써야 하는 구조 (event 별 서브폴더, **flat layout**):

```
<dest_dir>/<event_id>/S0001.jpg     # 성공 측정 이미지
<dest_dir>/<event_id>/S0001.txt     # cond — parse_cond() 형식 (Scope + !Cursor_info 필수)
<dest_dir>/<event_id>/S0002.jpg
<dest_dir>/<event_id>/S0002.txt
...
```

- `event_id` = msr 유니크 string `yyyymmdd_hhmmss_<recipe_name>_<lot_id>`
  (Windows 디렉토리명 금지 문자는 office 구현이 치환).
- cond 는 이미지 **옆**에 flat 으로 — `align_images/` 의 `.<파일명>/cond.txt`
  숨김폴더 규약과 **다름**.
- 반환: `list[StagedEvent]` (없으면 빈 리스트 → 호출부가 기존 캐시 보존).
- swap 후 최종 위치 (다운로더 관여 없음):
  `poc/workflow_3/align_consensus_cache/<eqp_id>/<class>/<recipe>/events/<event_id>/`
- 캐시 루트는 `ALIGN_CONSENSUS_CACHE_DIR` env 로 override 가능.

## 4. 다음 단계

1. **office PC (사용자):** `poc/workflow_3/monitor/office_success_downloader.py` 구현
   (gitignore, 커밋 안 됨). 스켈레톤 = plan 문서 Task 6, 계약 = `SuccessDownloader`
   Protocol docstring (위 §3 경로 명세 포함). `make_success_downloader()` 인자 없는
   팩토리 노출 필수.
2. **office 1회 검증:** 실제/replay 알람으로 루프를 돌려
   `align_consensus_cache/.../events/<event_id>/` 에 S*.jpg+S*.txt stage 확인,
   로그 `consensus gather: ... reason=ok events=N images=M` 확인,
   legacy 경고 없이 정위치 로드되는지 확인.
3. **deferred:** consensus *빌드* (staged 재료 → consensus template) — 이번 범위 밖.
   빌드 시 cond 는 flat S*.txt 라 `cond_file.cond_path_for()` 경로 해석을 그대로 못 쓰고
   직접 짝 매칭 필요함을 유의.

## 5. 메모리 업데이트

`memory/project_consensus_gather_in_loop.md` 신규 작성 + `MEMORY.md` 인덱스 1줄 추가 완료
(landed 커밋 범위, office 후속 조건, cond/flat 계약, 게이트 env). 이번 저널에서 추가 변경 없음.
