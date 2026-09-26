# 알람 1건 = 이벤트 폴더 1개

> CLAUDE.md 에서 옮겨 온 원문(2026-09-27). 이 기능을 고치기 전에 끝까지 읽을 것 - 계약은 오피스 실측과 설계 결정에서 나왔다.

- **알람 1건 = 이벤트 폴더 1개** (2026-09-17, `util/event_dir.py`): `<EVENTS_DIR>/<eqp_id>-<tag>/[attempt_<n>/]` 에
  `console.log`(stdout/stderr 전사, 줄마다 시각) / `work2.log`·`vlm_calls.log` 사본 / `runs/`(runner 저널) /
  `debug_images/` / `recording/` / `event.json`(eqp·recipe·CycleResult) / Episode·Guard JSON 이 모인다.
  `EVENTS_DIR` 기본값은 `ALIGN_IMAGES_DIR` 의 **형제** `align_fail_events/` 다(녹화가 같은 드라이브에 쌓이게;
  override `ALIGN_FAIL_EVENTS_DIR`). 원리: `event_scope(take)` 가 프로세스 전역 take 를 세우고 `sys.stdout`
  을 tee 하며, `debug_root()`/`runs_root()` 가 **호출 시점에** 그 take 를 가리킨다 - 모듈 import 시점에
  `DEBUG_IMAGE_DIR / "x"` 를 굳히면 이 경로를 못 따라오므로 새 debug 저장 지점은 반드시 `debug_root()` 를 쓸
  것. 모니터가 감지 시점에 열고 `run_alarm_cycle`/`run_check_only_cycle` 이 같은 폴더로 재진입한다(동시 사이클
  전제 없음). take 식별은 `cycle.take_dir_for` 와 `recovery_episode.episode_root_for` 한 곳. 보관은
  `prune_events`(오래된 take 의 recording/·debug_images/ 만 삭제, 텍스트·Episode 는 유지). 사후 복사로
  모으던 `cycle_images`(mtime 추정)와 `prune_recordings` 는 이것으로 대체되어 삭제됐다.
