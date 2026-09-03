# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

AI-powered automation system for CD-SEM/VeritySEM recipe setup. Uses VLM (Vision Language Models) for screen understanding and classical CV for coordinate decisions, driving GUI automation of the RCS metrology client to replace manual recipe creation.

## Active Workstreams

**`poc/workflow_3/` is the production package and current focus.** It consolidates the former workflow_1 (RCS GUI automation) and workflow_2 (CV align-key correction) into one real-time loop:

```
alarm detection (ALID=9006) → connect to tool via RCS → CV align-fail correction
→ on failure: cube rich notification to engineer → always-on screenshot recording
  (captures engineer manual operations too) → close tool → wait for next alarm
```

Subpackages — 4-layer DAG: `util` (leaf) → `{vlm, runner}` (services) → `{align, rcs, sem_monitor, recording_filter}` (capabilities) → `monitor` (orchestrator). workflow_3 never imports workflow_1/2.

- **`monitor/`** — the loop. `align_fail_monitor.py` (primary entry: polling + edge-trigger + manifest), `align_fail_monitor_only_check.py` (light "check-only" variant: connect → capture one frame → close, no correction actuation / no recording / no engineer watch). **두 진입점은 목적으로 갈린다 — 시험 성격에 따라 골라 쓴다(2026-08-12):** production 은 `_apply_live_mode_defaults()` 가 `SAFE_MODE=0` + `ALIGN_FAIL_CORRECTION_DRY_RUN=0` 을 진입점에서 못박아(seed_env 보다 **먼저** — 오피스 `workflow_3_config.py` 사본의 `CORRECTION_DRY_RUN=1` 이 조용히 덮는 것을 막는다) env 없이 실클릭으로 뜨고, 시작 시 실운전 배너를 찍는다. 되돌리려면 셸 `SAFE_MODE=1`(setdefault 라 셸 env 가 항상 이기고, `config.py` 의 `dry_run or safe_mode` 이중 게이트도 그대로). check-only 는 **의도적으로 그 기본값을 갖지 않는다** — "안전하게 한 번만 보고 싶다"는 요구를 받는 쪽이라 같은 기본값을 복사하면 안 된다, `cycle.py` (per-alarm WorkflowRunner steps + guaranteed teardown; also the check-only cycle), `recording.py` (always-on RecordingSession), `notify.py` (popup + outcome-based cube notify), `engineer_done_align_adjustment.py` (detects engineer finishing manual align via Recipe Monitor measurement counter N→ stops recording early so teardown closes the tool), `success_gather.py` (non-blocking office glue around `align.consensus_gather`), `alarm_source.py` (office module 2-stage fallback + replay CSV), `rcs_recovery.py` (**RCS 부재 시 재실행+재로그인**, see below), `integration_loader.py` (office adapter loading logs), `manual_record.py` + `frame_meta.py` (**alarm-free manual recording session**, see below), `share_request.py` (**점유 tool 화면 공유 요청 actuator**, see below), `make_demo_video.py` + `demo_log_panel.py` (**오프라인** 시연 영상 조립 — recording 프레임 -> mp4; 루프와 무관, `DEMO_VIDEO_*` 네임스페이스. 파일명의 `elapsed_ms` 로 실시간 축을 복원하고 정지 구간을 압축한다. 녹화는 **tool 창 rect 그랩**이라 터미널이 원리상 안 찍히므로, `demo_log_panel` 이 manifest `started_at` 기준으로 `work2.log`/`step_*.json`/콘솔 tee 를 시간 정렬해 프레임 옆에 합성한다 — 녹화 시작 30s 전까지 포함해 `connect_tool` 구간도 되살린다. 한글이라 cv2 가 아닌 PIL+TrueType 렌더. 촬영 대본은 `docs/runbooks/demo_video_shot_list.md`. **접속 구간 prelude** (2026-08-19): 본 녹화가 tool 창 rect 라 'RCS 실행->로그인->tool 진입' 구간은 원리상 프레임이 없다 - `ALIGN_FAIL_RECORD_PRELUDE=1` 이면 사이클 시작부터 **화면 전체**를 `recording/prelude/` 에 따로 녹화하고(그래서 터미널 콘솔도 프레임에 담긴다), `make_demo_video` 가 manifest 의 `started_epoch` 차이로 시간축을 맞춰 앞에 잇는다. 하위 폴더인 이유는 `recording_filter` 의 비재귀 glob 에 안 걸리게 하려는 것 - 그 파이프라인은 tool 창 rect 를 전제한다. 기본 off. 편집은 `DEMO_VIDEO_SEGMENTS="0-30,120-260"`(START/END 보다 우선, 음수=끝에서부터)이고 실행하면 먼저 타임라인 미리보기(소스 구간 + 30s 프레임 밀도)를 찍어 자를 지점을 고르게 한다). **회차 이어붙이기** (2026-08-24, `make_demo_video_combined.py`, `DEMO_COMBINED_*`): `manual_align_correction.py` 를 같은 tool/recipe 로 여러 번 돌리면 실행마다 별도 tag 폴더가 생기는데, 이 스크립트가 그것들을 시간 순으로 이어 `1st/2nd/3rd Trial` 타이틀 카드 + 프레임 좌상단 회차 라벨을 붙여 한 편으로 만든다. prelude 접합과 달리 **회차는 하나의 절대 시간축에 올리지 않는다** - 사이에 수십 분~수 시간 공백이 있어 그러면 영상 대부분이 정지 화면이 된다. 각 회차는 t=0 부터 다시 세고 화면의 `t=` 는 '그 회차의 경과시간'이다. 리샘플링/압축/letterbox/인코딩은 `make_demo_video` 함수를 그대로 import 해 쓴다(포크 금지 - 시간축 규약이 갈리면 같은 녹화가 스크립트마다 다르게 보인다). 렌더 env 는 `DEMO_VIDEO_*` 를 공유하고 조합 관련만 `DEMO_COMBINED_*`(INPUT_DIRS/ROOT/LABELS/TITLE_SEC/SEGMENTS[_n]). 로그 패널은 미지원 - 회차마다 기준 시각이 달라 한 벌로 못 맞춘다. 인자는 `manual_align_correction.py` 와 같은 규약으로 **파일 상단 상수**를 고쳐 쓰고 (`seed_env_from_constants` 가 `setdefault` 로 env 에 흘린다: 실제 셸 env > 파일 상수 > 코드 기본값, 무시된 상수는 콘솔에 남는다), 경로/라벨은 리스트로도 적을 수 있다. `None`/`""` 이 미설정이고 `0` 은 유효한 값이다(TITLE_SEC=0=카드 없음).

- **Recovery Episode 수집** (2026-08-30, `ALIGN_FAIL_EPISODE_COLLECT=1`, **기본 off**):
  ALID=9006 active interval 하나 = Recovery Episode 하나. `monitor/recovery_episode.py`
  의 `EpisodeTracker` 가 알람 row 처리(`process_fail_rows`) 부수효과로 capture 폴더 루트에
  `recovery_episode.json` 을 **첫 GUI step 전에** 원자적으로 쓰고, cooldown 재시도는 같은
  Episode 의 `attempt_2, 3...` 이 되며, 알람이 poll 에서 사라지면 clearance 이벤트와 함께
  닫힌다. identity 는 uuid4 이고 tag(알람 UTC9)는 **위치일 뿐 identity 가 아니다**;
  재개 판정은 fingerprint(장비+alid+recipe+UTC9) **완전 일치**뿐이며 프로세스당 1회
  capture tree 스캔이 유일한 디스크 재구성 경로다(알람 없는 open Episode 는
  `incomplete(alarm_gone_during_restart)`). attempt 산출물은 `<tag>/attempt_<n>/` 아래로
  갈린다 - **cooldown 재시도가 같은 `recording/` 에 두 테이크를 섞던 tag 충돌 결함이 이
  구조로 닫힌다**(별도 수정 금지). 수집 on 이면 attempt 폴더에 `guards.json`
  (`monitor/guard_readings.py`, Episode-level Guard **정확히 셋**: 화면 관측 가능성 /
  점유·제어 / SEM mode+align key 가시성·유일성. 값은 `True/False/None` 이고 **관측 실패는
  전부 unknown** - `false` 로 새지 않는다. 읽은 OM/SEM 은 detail 에만, OK 컨트롤 가용성은
  Guard 가 아니라 precondition), `measurement_verification.json`
  (`monitor/measurement_verification.py`, primary Verification 3상태 record. 자동 reader 는
  **unknown-only stub** 이고 Assist 패널 crop 만 근거로 남긴다 - 열 분리 판독은 오피스
  캘리브레이션 gate 다), `numerator_reads.jsonl`(분자 per-read 판독; fallback Verification 은
  detector 의 boolean 이 아니라 이 기록을 읽는다)이 함께 남는다. Episode 를 닫을 때
  `poc/workflow_4/playbook` 의 순수 evaluator 가 Outcome 을 파생하고 `[DIGEST] episode ...`
  한 줄을 찍는다. 저장 경로는 **Episode-relative 만**(절대/`..` 는 로드가 거부).
  off 면 녹화 폴더·manifest·사이드카·Guard 파일 모두 종전과 동일하다. 스펙/티켓은
  `docs/issues/align-fail-recovery-playbook/`.

- **RCS 자동 조작 시연** (2026-08-19, `monitor/demonstration_rcs_control.py`, `DEMO_RCS_*`): 알람과 무관한 시연 전용 진입점. `RCS 실행 -> 로그인 -> View 탭+휠 훑기 -> List 탭 -> 장비별 [접속 -> 창 안 조작 -> tool 창 닫기]` 를 재생한다. **장비마다 다른 조작**을 보여주는 것이 요점이다(같은 동작 반복은 스크립트로 보이고, 창을 열어 메뉴를 타고 빠져나오면 자동화로 보인다): `MCD019=memo_print`(Utility -> 위로 열리는 드롭다운의 Memo Print -> MemoPrint popup 편집 영역에 세 줄 메모 입력 -> Close, 2026-08-24 추가), `MCDC10=worksheet`(Work Sheet 아래 버튼 -> File -> Exit; 2026-08-24 MCDC22 에서 교체 - MCD019 의 memo_print 가 오피스 확인됐으므로 두 번째 장비는 같은 것을 반복하지 않고 'File' 쪽만 본다). `optics`(Optics... -> Memory 탭 -> Close)는 기본 배정에서 빠졌을 뿐 **등록은 유지**한다 - 설명문 좌표가 오피스 실측이라 지우면 Mac 에서 되살릴 수 없고, `requires_previous=False` 를 뒤 step 에 가진 유일한 흐름이라 엔진 계약 ③의 유일한 커버리지다. 배정은 `DEMO_RCS_FLOWS="TOOL=flow,..."`(memo_print/optics/worksheet). 세 흐름은 모양이 같아 한 엔진(`run_in_tool_flow` + `InToolFlow`/`FlowStep` 데이터)으로 돈다 - 차이는 **`requires_previous`** 하나다: Optics 의 Close 는 대화상자 상시 버튼이라 Memory 실패와 무관하게 누르지만, Work Sheet 의 Exit 는 **File 이 드롭다운을 열어야만 존재**해 File 이 실패하면 건너뛴다(안 열린 메뉴 자리를 클릭하지 않기 위해). RCS 기동은 production 과 같은 `rcs_preflight.ensure_rcs_session_ready` 배선(이미 로그인돼 있으면 재실행 생략). 클릭은 전부 `share_request` 와 **같은 fail-closed 확인 게이트**(VLM 이 좌표, OCR 이 라벨 확인, 미확인 시 클릭 금지)를 지난다. 계약 셋 — ① **tool 창 닫기는 어떤 경로로든 시도한다** ② **첫 step 의 라벨이 '창이 떴다'는 유일한 증거이며, 확인 전에는 뒤 요소를 찾지 않는다** ③ **창 확인 후에는 독립 step 을 끝까지 시도한다**. 1회차 오피스 실행이 이 설계를 만들었다 — 처음엔 Optics 대화상자를 `find_window_by_title_prefix("Optic")` 로 찾고 못 찾으면 tool 창으로 폴백했는데, **Remote Monitoring 창은 장비 화면의 원격 뷰라 그 창들이 로컬 top-level 창으로 뜨지 않는다**. 폴백이 "못 찾았어도 계속"을 만들어 화면 어딘가의 **다른 Close** 를 눌렀다. **클릭이 실제로 먹게 하는 데 세 가지가 필요했다**(오피스 실측 2회차: 커서는 정확히 가는데 클릭만 안 먹음) — `perform_remote_click` 이 순서를 계약으로 고정한다: ① **전면화**(`foreground_window`, 실패 시 클릭 안 함) - 포커스 없는 창의 첫 클릭은 창 활성화에 쓰이고 버튼에 닿지 않는다. `sem_monitor.controller._ensure_actionable` 이 제스처마다 하는 일과 같으며 시연 경로가 이걸 빠뜨렸던 것이 원인이다. ② **이동 후 체류**(`DEMO_RCS_PRE_CLICK_SETTLE_SEC` 0.6) - 원격이 커서를 따라올 시간. ③ **누름 유지**(`DEMO_RCS_CLICK_HOLD_SEC` 0.15 -> `click_at_screen(hold_sec=)` 가 press/유지/release) - 즉시 press/release 쌍은 원격의 입력 샘플링 두 틱 사이로 통째로 빠져나갈 수 있다. `hold_sec` 기본값 0.0 이라 기존 호출부는 무영향. `FlowStep.required` 가 비면 **forbidden 만 보고 읽힌 토큰을 콘솔에 남긴다** - 실제 문구를 모르는 요소를 첫 실행에서 알아내기 위한 장치이며, 알아낸 뒤에는 required 로 승격해 게이트를 온전하게 만든다(Work Sheet 버튼이 그 경로를 거쳤다: 라벨이 `Work Sheet` 로 확인되어 `("work", "sheet")` 로 승격). required 는 **언어별 묶음을 통째로** 만족해야 하므로 `Work` 만 읽히면 확인이 아니다 - OCR 이 `WorkSheet` 로 붙여 읽는 경우는 두 needle 이 모두 부분 일치해 통과한다. **타이핑 step 은 그 `required=()` 구멍을 쓰지 않는다**(`FlowStep.input_text`, memo_print 의 편집 영역): 빈 required 는 `_confirm_point` 가 정책을 건너뛰고 조기 반환하므로 strict 에서도 무검증 통과가 되는데, 클릭은 대개 한 번이면 끝나지만 타이핑은 상태를 남기고 포커스가 어긋나면 엉뚱한 필드에 글자가 들어간다. 그래서 편집 영역의 확인 근거는 **popup 제목**(`("memo",)` 한 needle - `MemoPrint`/`Memo Print` 모두 부분 일치)이고, 제목이 crop 밖이면 unreadable 이라 기본 lenient 에서는 그대로 진행하되 strict 만 거부한다. 입력은 `type_multiline_text` 가 `\n`->Enter 로 바꿔 **글자마다** 흘려보내며(`DEMO_RCS_CHAR_TYPE_DELAY_SEC` 0.08 - 원격이 입력을 샘플링해서 한 번에 보내면 삼켜진다), **대문자는 문구 전체를 Caps Lock 한 쌍으로 감싸서 만든다**(`DEMO_RCS_SHIFT_MODE` 기본 `caps_all`: 켜기 -> 모든 글자를 소문자 기본 키로 -> 끄기 = **전부 대문자**). 4회차(2026-08-24)에서 대문자마다 토글하던 `caps` 모드가 memo 를 깨뜨렸다 - 토글은 **상태**라서 쥐는 수정자와 위험의 성질이 다르다: (a) 장비가 토글을 적용하기 전에 글자가 도착하거나 'caps off' 가 글자를 추월하고, (b) 24번 보내는 토글 중 **하나만 유실되면** 장비의 caps 상태가 뒤집혀 그 뒤 글자가 전부 반대 case 가 된다(Shift 유실은 한 글자, 토글 유실은 나머지 전부). 그래서 토글 수를 문구당 2번으로 줄였고, 실패해도 '전부 소문자' 라는 읽을 수 있는 형태로 어긋난다. **문구에 대문자가 하나도 없으면 토글을 아예 보내지 않는다** - 소문자만 적으면 수정자 0개 경로가 되어 가장 안전하다(1회차에서 소문자는 유실이 0이었다). 토글 체류는 Shift 와 별도로 더 길다(`DEMO_RCS_CAPS_SETTLE_SEC` 0.4 vs `SHIFT_SETTLE` 0.12 - 토글은 링크를 건너 장비가 적용해야 한다). 끄기는 `finally` 보장이고, `SendInput` 이 전역이라 **로컬 PC 의 Caps 도 같이 켜지므로** 끝나고 로컬 상태를 읽어 켜져 있으면 되돌린다(`_local_caps_on` - 모르면 None 이고 그때는 건드리지 않는다: 추측 토글은 꺼져 있던 것을 켜는 쪽이 될 수 있다). 오피스 실측 2회가 이 방식을 강제했다. **1회차**: "Infra. Tech Center!!" 가 "nfra. ech enter" 로 들어갔고 사라진 것이 정확히 `I T C !! O S S` = Shift 조합 전부, 그리고 그것만이었다. 원인은 pynput win32 구현(`pynput/keyboard/_win32.py:83-92`) - `VkKeyScan(char)` 이 'Shift 필요' 라고 답하면 `vk=0 / scan=유니코드 코드포인트 / flags=UNICODE` 로 보내는데, **vk 도 scan code 도 없는 이벤트**라 vk/scancode 를 중계하는 RCS 원격에는 중계할 것이 없다(소문자는 진짜 vk+scan 이라 전부 정상이었다). **2회차**: Shift 를 직접 쥐고 기본 키를 누르니 글자는 도착했지만 **소문자로** 들어왔다 - 이 원격은 키를 개별 타건으로 넘기고 쥐고 있는 수정자를 함께 실어 보내지 않는다. pynput 의 `Key.shift` 는 Windows 에서 **이미 `VK.LSHIFT` + scancode 0x2A**(`_win32.py:160`), 즉 실제 왼쪽 Shift 와 같은 신호다 - '다른 Shift 를 쓰면 된다' 는 선택지는 없고, 쥐는 수정자는 이 경로로 불가능하다고 봐야 한다. 그래서 필요한 것은 '쥐는 수정자' 가 아니라 **상태를 남기는 키**다(Caps Lock 은 평범한 vk 타건이라 중계되고 상태는 장비 OS 가 기억한다). 진단용으로 `caps`(4회차 경로)/`shift`(2회차)/`type`(1회차)을 남겨 뒀다. **Caps Lock 은 글자만 바꾸므로 Shift 기호는 못 건넌다**(US 기호 배열 가정, `SHIFTED_CHARS` 표): 기본 키가 그대로 들어와 `!`->`1`, `"`->`'` 가 된다. 조용히 틀린 글자를 넣지 않으려고 `shift_symbols` 가 입력 **전에** '무엇이 무엇으로 들어올지' 를 콘솔에 경고한다 - Mac 에서 화면을 볼 수 없으니 오피스에서 대조할 유일한 근거다. 고치는 방법은 코드가 아니라 `DEMO_RCS_MEMO_TEXT`(`\\n` 이 줄바꿈) 로 그 기호를 문구에서 빼는 것이다 - **기본 문구에서도 그렇게 했다**(2026-08-24 사용자 결정: `!!` 와 큰따옴표 제거). 지금 기본 문구에 남은 수정자는 전체를 감싸는 Caps 토글 한 쌍뿐이며 `shift_symbols` 경고도 비어 있다. 새 문구에 Shift 기호를 다시 넣지 말 것. 입력을 끝내면 `DEMO_RCS_POST_TYPE_WAIT_SEC`(2.0) 머문 뒤 popup 의 **Close** 를 누른다(사용자 지시) - 이 Close 는 `requires_previous=True` 다: 편집 영역 클릭이 popup 존재의 유일한 증거라 그게 실패한 뒤 'Close' 를 찾아 나서면 화면 어딘가의 다른 Close 를 누른다(엔진 계약 ②와 같은 이유). 그때 popup 은 열린 채 남지만 엔지니어가 손으로 닫는 편이 낫다. 글자마다 `abort_switch` 를 확인한다(수십 글자면 수 초간 키가 나가므로 여기서 안 보면 긴급 단축키가 안 먹는 것으로 느껴진다). 클릭은 두고 입력만 끄는 롤백은 `ACTION_LOGIN_TYPING_ENABLED=0`. **Utility 는 tool 모니터 오른쪽 아래에 있고 다른 창에 가려질 수 있다**(사용자 보고 2026-08-24) - 그때는 그 자리를 **Alt+click** 하면 덮은 창이 뒤로 밀려 되살아난다. 그래서 '여는 버튼을 못 찾으면 즉시 포기'(다시 눌러도 같은 화면이라는 전제) 규칙이 이 경우에만 깨진다: `_confirm_point` 가 실패 이유를 `not_located`/`label_rejected` 로 갈라 주고, **좌표 미검출일 때만** `reveal_fn` 이 돌아 `opener_not_visible` 로 끝난다(라벨 불일치는 이미 보이는 화면을 잘못 짚은 것이라 창을 밀어내면 엉뚱한 창만 뒤로 간다). 가림 해제 예산(`DEMO_RCS_REVEAL_ATTEMPTS` 2 - 창이 여러 장 겹칠 수 있다)은 클릭 재시도 예산과 **분리**한다(서로 다른 실패를 고친다). 누를 지점은 가린 창이 아니라 **Utility 가 있어야 할 자리**이며(그 위를 덮은 것이 밀어낼 창이다) 창 밖으로는 절대 나가지 않게 자른다 - Mac 에서 화면을 볼 수 없으므로 `DEMO_RCS_REVEAL_X_RATIO`/`_Y_RATIO`(0.88/0.92)로 옮기고 콘솔의 px/screen 값으로 맞춘다. **Alt 는 커서가 도착한 뒤에 잡는다**(순서: 전면화 -> 이동 -> 체류 -> Alt down -> `DEMO_RCS_ALT_SETTLE_SEC` 0.3 -> 클릭). 세 제약이 이 자리를 정한다 - ① 전면화보다 뒤(`window_utils.foreground_window` 가 foreground-lock 우회로 더미 Alt down/up 을 주입하므로 먼저 잡으면 그 up 이 우리 Alt 를 놓아 평범한 클릭이 된다 = 커서는 맞는데 창이 안 밀리는 실패), ② 커서 이동보다 뒤(Alt 쥔 채 커서를 끌면 그 이동 전체가 Alt 눌린 상태가 된다), ③ 누름보다 한 틱 앞(원격이 입력을 샘플링해 Alt down 과 버튼 down 이 같은 틱이면 수정자 없는 클릭으로 넘어간다 - `click_at_screen(hold_sec=)` 이 생긴 것과 같은 이유). `click_at_screen` 의 ±3px jiggle 만은 Alt 를 쥔 채 일어나지만 버튼이 안 눌려 Alt+drag 는 되지 않는다(`ALIGN_FAIL_CURSOR_JIGGLE_PX=0` 로 끌 수 있다). 해제는 `finally` 보장 - 눌린 채 남으면 이후 모든 클릭이 Alt+click 으로 변질된다. 전면화 실패 시엔 Alt 를 잡지도 않는다. 롤백 `DEMO_RCS_REVEAL=0`. **메뉴 항목의 형제 이름을 forbidden 에 두면 안 된다**(오피스 1회차 Work Sheet 실패 원인): OCR crop 이 클릭 지점 좌우 30% 를 담으므로 메뉴 바에서는 Edit/View/Help 가, File 드롭다운에서는 Save/Print/Export 가 **반드시** 함께 읽힌다. `classify_label` 은 forbidden 을 required 보다 **먼저** 보고 forbidden 은 lenient 에서도 막으므로, 그 목록이 File/Exit 클릭을 스스로 막았다. 둘 다 `forbidden=()` 으로 비우고 확인은 `required` 에 맡긴다(File 은 드롭다운만 여는 무해한 클릭이다). File 은 **'Work Sheet' 창 제목 바로 아래의 작은 라벨**이라 설명문에 'SMALL'/'title bar' 단서를 넣어야 VLM 이 찾는다. OK/Print 는 실제로 인쇄/저장될 수 있어 어떤 흐름에서도 누르지 않는다. **타이밍 상수는 파일 상단 블록에 있고 두 종류로 갈린다**(2026-08-24 사용자 요청으로 ①만 30% 단축): ① 동작 사이 간격 = 관객이 보는 속도(체류 3.0->2.1, 장비 간격 3.0->2.1, step settle 1.5->1.05, 글자 0.08->0.056, 입력후 2.0->1.4, 휠 0.6->0.42 - 장비 2대 기준 대기 합계 ~42s->~30s) ② **원격 입력 성사 조건 = 줄이지 않는다**(`PRE_CLICK_SETTLE` 0.6 / `CLICK_HOLD` 0.15 / `ALT_SETTLE` 0.3 / `SHIFT_SETTLE` 0.12) - 오피스 실측 3회로 얻은 값이고, 깎으면 커서는 가는데 클릭이 안 먹거나 대문자가 사라진다. 시연이 30% 빠른 것과 시연이 안 되는 것은 비교 대상이 아니다. `SAFE_MODE=0` 이 진입점 기본값(setdefault 라 셸 env 가 이김)이고 보정 actuation 은 없다. 확인 정책 기본값은 **lenient**(`DEFAULT_CONFIRM_POLICY`) — 이 흐름이 누르는 버튼은 돌고 있는 장비에 영향을 주지 않는다고 오피스에서 확인됐으므로(2026-08-19), 여기서 지키는 것은 장비 안전이 아니라 시연 흐름이고 OCR 이 한 번 못 읽어 멈추는 쪽이 더 큰 손해다. lenient 는 '못 읽음' 만 통과시키며 **금지 토큰(cancel/exit/terminate)은 어떤 정책에서도 막힌다**(`share_request.accepts_label`), 좌표 미검출도 정책과 무관하게 클릭 금지다. 라벨 문구를 확정하는 진단 실행만 `DEMO_RCS_CONFIRM=strict`. 반면 **전면화 실패 시 클릭 금지는 완화하지 않는다** — 그건 안전 게이트가 아니라 '이 클릭이 어디로 가는지 아는가' 의 문제다. 확인 실패 시 crop/OCR 원문이 `debug_images/demo_rcs_flow/<tag>/` 에 남는다 — Mac 에서 이 화면을 볼 수 없어 실제 문구를 아는 유일한 경로다. 테스트(124)는 순서·확인 게이트·의존 step 건너뛰기·재시도·클릭 순서·흐름 정의·타이핑 게이트(라벨 근거/긴급 해제/dry-run)와 가림 해제(이유 분기/예산 분리/Alt 순서·전용 체류·해제 보장/지점 클램프)·속도 상수 2분류와 대문자 입력(caps_all/caps/shift/type 4방식·토글 1쌍 계약·전용 체류·기호 매핑·예외 시 Caps 복원·로컬 Caps 되돌리기·Enter 는 비-수정자)을 덮는다. **VLM 좌표는 오피스 검증됨**(커서가 Optics/Work Sheet 에 정확히 도달), 클릭 성사 여부는 재검증 대기. Utility/Memo Print 흐름은 **MCD019 에서 오피스 확인됨**(2026-08-24). Work Sheet/File 흐름은 MCDC10 에서 재검증 대기.
- **점유 tool 화면 공유 요청** (2026-08-18, `ALIGN_FAIL_SHARE_*`): 점유 `Select` 팝업을 검출만 하고 포기하던 경로를 바꿔, "화면 공유"를 골라 `Request` 를 눌러 관전 세션을 얻고 엔지니어의 수동 align 작업을 녹화한다. `occupied_popup.py` 는 fail-**open** detector 로 그대로 두고, 클릭은 `share_request.py` 의 fail-**closed** actuator 가 한다 (오류 정책이 정반대라 파일을 나눴다). 안전은 env 게이트가 아니라 **확인 게이트** — 좌표는 VLM 이 찍고 그 자리 라벨을 OCR 로 읽어 `share`+`screen` 이 확인될 때만 클릭하며, `control`/`terminat`/`cancel` 이 읽히면 정책과 무관하게 클릭하지 않는다. 점유는 **3-상태** (`rcs/row_occupant.py`: `occupied_by_other`/`free`/`unknown`) 이며 `unknown` 은 보정을 막는 대신 outcome 을 `corrected_unverified` 로 강등해 **cube 가 반드시 나가게** 한다 — `correct_align_fail_auto` 가 open-loop 라, 먹지 않은 클릭을 `corrected` 로 보고하면 알림까지 생략되어 아무도 모르는 미보정이 남기 때문이다. 두 새 status(`view_only_observation`, `corrected_unverified`)는 `_RETRY_LATER_OUTCOME_STATUSES` 로 **`active_tools` 가 아니라 cooldown 재시도**로 가며(점유가 풀리면 실제 보정이 돌아야 한다), `share_max_attempts`(2) 상한이 cube 반복과 커서 독점을 끊는다. `row_occupant` 는 반드시 **자기 crop** 을 쓴다 — `tool_row_verify` 의 strip 을 넓히면 점유자 ID 가 `_looks_like_tool_id` 를 통과해 `unreadable`(lenient 통과)이 `mismatch`(무조건 거부)로 승격되어 정상 행의 클릭이 거부된다. 설계 `docs/superpowers/specs/2026-08-18-occupied-share-request-recording-design.md`, 적대적 검토 `docs/opencode/2026-08-18-occupied-share-request-debate.md`.
- **RCS 부재 시 자동 복구** (2026-08-19, `monitor/rcs_recovery.py`, 기본 **on**): `ensure_rcs_ready` step 이 메인 창을 못 찾으면 재실행+재로그인한다. 협력자(프로세스 조회/실행/로그인/창 대기)를 전부 주입받아 Mac 에서 실장비 없이 시험된다(`share_request.py` 와 같은 규약). 두 가지가 계약이다 — ① **프로세스가 이미 있으면 재실행하지 않는다**. 이 분기의 진입 조건은 "창을 못 찾았다"이지 "프로세스가 없다"가 아니라서(스플래시/멈춘 창) 무조건 띄우면 `RcsMainHD.exe` 가 2개가 된다. `find_existing_rcs_processes` 는 psutil 이 없으면 빈 리스트를 주므로 "없음"과 "모름"이 같은 값이 되는데, 그대로 믿으면 가드가 **겉보기에만** 존재한다 — `cycle._scan_rcs_processes` 어댑터가 모름을 `None` 으로 구분하고 그때는 실행을 보류한다(psutil 은 이 커밋에서 `pyproject.toml` 에 처음 선언됐다. 그전엔 선언이 없어 `open_rcs.main()` 의 중복 실행 가드도 오피스에서 내내 무력이었다). ② **복구 로그인은 tool 에 접속하지 않는다**. `run_login_workflow` 는 `target_tool_name` 이 비지 않으면 `open_target_tool` step 을 붙여 그 tool 을 여는데, 인자를 생략하면 env(`ACTION_TARGET_TOOL_NAME`)가 지목한 엉뚱한 tool 이 열려 알람의 tool 은 `wrong_tool_opened` 로 깨진다. 어느 tool 인지는 알람이 정한다. 빈 문자열을 넘기는 것만으로는 안 고쳐진다 — 기존 `target_tool_name or load_target_tool_name()` 에서 `""` 가 falsy 라 그대로 env 로 흘렀다. `resolve_login_tool_name` 이 **`""`=접속 안 함 / `None`=env 조회**로 가른다. 실패는 예외가 아니라 status(`rcs_recovery_error`/`rcs_recovery_no_window`)로 나가 `failure_class` 가 되므로 manifest 에서 "복구가 깨졌다"와 "복구를 안 했다"(`rcs_unavailable`)가 구분된다. ③ **창 없는 좀비 프로세스**(2026-08-19 오피스: 작업 표시줄엔 없고 PID 만 남음)는 ①의 가드에 걸려 재실행이 막히고 로그인만 시도하다 `rcs_recovery_no_window` 로 끝난다 - 가드가 원래 막으려던 것과 정반대 상황에서 발목을 잡는 것이다. `classify_existing_processes` 가 PID 별 창 보유로 `unknown`/`none`/`windowed`/`stale` 을 가르고, 창 조회가 안 되거나 예외면 **`windowed`(살아 있음)로 본다** - 종료는 되돌릴 수 없으므로 모를 때는 죽이지 않는다. 창을 가진 프로세스가 하나라도 있으면 좀비로 보지 않는다(그게 엔지니어 세션일 수 있다). 자동 종료+재실행은 opt-in `ALIGN_FAIL_RCS_KILL_STALE=1`, 기본은 경고만 하고 수동 종료를 안내한다. 롤백 `ALIGN_FAIL_RCS_RECOVERY=0`. **실장비 미검증** — 테스트는 판정 로직만 덮고 pywinauto 가 실제 로그인 다이얼로그를 상대하는 동작은 안 덮는다. 리뷰 `docs/opencode/2026-08-19-rcs-recovery-review.md`.
- **`rcs/`** — RCS GUI automation: open/login (`login_rcs_common`, `login_rcs_ui_venus_mai`)/tool select+match (`tool_name_match`)/close/screenshot. Tool-row click is coarse→fine 2-VLM (coarse bbox → fine point; **both stages default to `mai-ui`** since 2026-08-07) + a **row confirm gate** (`tool_row_verify`): the two VLMs are *not* independent votes (fine only sees the crop coarse chose), so after the point is picked a **single-row strip** is cropped and OCR'd to confirm the text is the target ID. Policy via `SELECT_TOOL_ROW_CONFIRM` = `lenient` (default; reject only on reading a *different* ID) | `strict` (require confirmation) | `off`. Crop tightness needs all three of `SELECT_TOOL_ROW_VERTICAL_PAD_RATIO` (0.35) / `SELECT_TOOL_ROW_VERTICAL_PAD_MIN_PX` (10) / `SELECT_TOOL_ROW_MIN_CROP_HEIGHT` (56) — lowering only the ratio is a no-op because the two floors dominate (list rows are ~24px). A mis-click now reports `failure_class="wrong_tool_opened"` (was indistinguishable from `rcs_occupied`), closes the stray tool window, and retries after the occupied cooldown. Model choice is benchmarked by `bench_tool_locator.py` (no alarm, no clicking).
- **`align/`** — Align fail correction domain. Flat domain modules + two subpackages:
  - `matching/` — coordinate authority: `engine` (match engine, `AlignKeyTemplate`/`build_template`), `ensemble`.
  - `diagnostics/` — offline review/compare entrypoints (`compare_align_images`, `crosshair_detect`, `search_align_key`, `align_review`, `feasibility_check`, `verify_cond_box_crop`, `test_match_on_captured_frames`).
  - domain: `assets` (reads the `align_images/...` tree), `correction` (primary entry: `correct_align_fail_auto(controller, ...) -> CorrectionOutcome`), `live_search` (legacy fallback + `SEMMonitorController` Protocol + Mac mock), `grid_search` (**2026-08-28 search-around 재설계, 기본 경로**: PM 드롭다운 절대 배율 zoom-out + 2R 박스 FOV 격자 sweep(collect-then-chase) + phase-correlation odometry; 배율 변경은 Protocol 이 아니라 주입 함수 `MagnificationControl` — cycle.py `_PMDropdownSelector` 가 채우고 Mac 은 mock. 등록 배율은 cond.txt `Magnification`. 판독 실패/등록 배율 없음이면 legacy 로 degrade. env `ALIGN_FAIL_SEARCH_*`, 롤백 `ALIGN_FAIL_SEARCH_MODE=legacy`. 스펙 `docs/superpowers/specs/2026-08-28-search-around-zoomout-grid-design.md`, 물리 `docs/study/hitachi_mag_fov_pixel_260828.md`; **오피스 실장비 미검증**), `templates` (recipe align image → `AlignKeyTemplate`, cond-aware), `ok_button` (VLM OK-button locator), `search_pattern` (square-spiral pan primitive), `cond_file`/`cond_template`/`clean_align_image`/`consensus_gather` (cond + consensus helpers).
- **`sem_monitor/`** — `panel_locator.py` (landmark 기반 SEM Monitor panel locator) + `controller.py` (real `RCSSEMMonitor` adapter — double-click recenter / wheel zoom / OK click). **Panel ROI 확보는 2단(2026-08-12)**: `build_rcs_sem_monitor(vlm_client=...)` 가 먼저 `detect_sem_box`(check-only 에서 오피스 검증된 live SEM box 검출)로 ROI 를 잡고, 실패 시에만 landmark 템플릿 매칭으로 폴백한다 — `templates/sem_panel_landmarks/` 는 여전히 비어 있고(캘리브레이션 없음), 이전에는 그 때문에 보정 사이클이 step 6 `panel_not_found` 에서 항상 멈췄다. 같은 검출의 `pm_mode` 가 `mode_hint` 로 주입되어 `read_mode()` 가 OM/SEM 을 화면에서 읽은 값으로 답한다(우선순위 `ALIGN_SEM_MODE_OVERRIDE` > `mode_hint` > `sem_mode_default`). 검증: `test_controller.py` (7/7, VLM/실장비 없이 Mac 실행).
- **`recording_filter/`** — offline, on-demand frame-filter package (NOT in the loop hot path). Turns `RecordingSession` frames into `interaction_timeline.json`; `run_filter` orchestrates, `settings` = `RecordingFilterSettings`. Four stages: **1** `frame_reduce` (cv2 change-detection) → **1.5** `region_gate` (**VLM-free per frame**: demotes changes confined to the live SEM box to `ambient`; live-box location detected once per *layout generation*, so cost scales with generations, not frames) → **2a** `click_detect` (VLM cursor locate + ROI change → click) → **2c** `element_label` (click-point crop → PaddleOCR, VLM fallback → *what* was clicked). Stage 1.5/2c exist for the manual-recording use case (see below) and degrade to no-ops on sidecar-less alarm recordings.
- **`vlm/`** — Flask VLM client/config/prompts (`flask_vlm`, `vlm_client`, `ui_venus_mai_locator`, `ocr_spotting`). **`runner/`** — WorkflowRunner/step types/settings. **`util/`** — shared helpers. Top-level: `config.py` (`Workflow3Settings`), `logger.py` (audit trail), `debug_artifacts.py` (debug-file saver, no per-save console spam).

**Extension:** `poc/workflow_3e/` adds new MES-alarm jobs *on top of* workflow_3 without editing its core (imports workflow_3 one-way). First job: **measurement-fail abort** (MES fires a consecutive-fail threshold alarm → connect + abort the running measurement). Runs via a **unified supervisor** (`poc/workflow_3e/monitor.py`) that polls MES once and dispatches align rows to workflow_3's `process_fail_rows` and abort rows to workflow_3e's `process_abort_rows` — one process, so the single RCS cursor stays serialized (no lock; abort "can queue"). Ships **notify-only** behind a double gate (`SAFE_MODE=0` **and** `MEAS_FAIL_ABORT_DRY_RUN=0` to actually click). `MEAS_FAIL_*` env namespace (not `ALIGN_FAIL_*`). See `poc/workflow_3e/README.md` + spec/plan under `poc/workflow_3/docs/superpowers/`.

**State-machine layer:** `poc/workflow_4/` (2026-08-28) is a self-contained hand-rolled FSM framework (`framework/`: graph + validate, bounded engine with failure_class→fallback routing / per-node + global retry budgets / abort polling, `RunState` JSON, mermaid + self-contained HTML live view). It never imports workflow_3 except inside `adapters/`. **What touches production today is only the read-only mirror** (`adapters/workflow3_cycle.py`, `ALIGN_FAIL_GRAPH_VIEW=1`, default off): `cycle.py` hands it the same step list it gives the runner and `context["run_dir"]`, and it renders the runner journal as `workflow_graph.html` next to the journal. The engine is demo-only — it is **not** a second production runner and `WorkflowRunner` is not to be given routing; its first real consumer is the align-correction sub-flow nested inside the `run_correction` executor (ADR `poc/workflow_4/docs/study/adr/0003-*.md`, debate `docs/opencode/2026-08-28-workflow4-engine-vs-runner-debate.md`). **`playbook/`** (2026-08-30) 은 그 옆에 새로 선 **순수 도메인 계층**이다 - workflow_3 를 import 하지 않고 plain data 만 받으며, 첫 조각인 `outcome.py` 가 Verification 우선순위(primary Measurement, unknown 일 때만 분자 fallback)와 Recovery Outcome 파생의 유일한 소유자다. Tests: `uv run pytest poc/workflow_4/` (49).

**Frozen:** `poc/workflow_1/` keeps only the CCTV/DVR path + early experiments (no active work; still the `align_images` data root).

**Active offline CV bench:** `poc/workflow_2/` is *not* frozen — it is the eval / A-B / tuning harness where matching, ensemble, threshold, and consensus changes are validated against golden sets, then ported into `workflow_3/align`. It imports the engine from `poc.workflow_3.align` (never the reverse) and forks it bit-parity for experiments via `ensemble_lab.py`; golden drivers are `golden_localization_eval_cond.py` (rcp localization), `golden_consensus_eval_cond.py` (consensus A/B), and `golden_combined_eval_cond.py` (**production routed pipeline** — consensus-if-eligible else rcp, reusing both drivers; 3 axes: (A) consensus scaling by `cons_pool_n`, (B) rcp-only arm = `edge_ncc`/lab testbed, (C) routed overall; prints a one-line `[DIGEST]` + `digest.txt` to relay results without re-typing the console). **Current transition:** prove a CV change in workflow_2 → port only the verified change into workflow_3; primary build focus is workflow_3 (the real-time loop).

- **Bench config (shared, no env/CLI):** the 3 golden drivers read `poc/workflow_2/golden_eval_config.py` (gitignored edit-often scratch; copy from `golden_eval_config.example.py`). `golden_eval_config_loader.seed_env()` bridges its constants into env at each driver's top (before `gce`'s import-time `CONSENSUS_MIN_S` read); real env still wins. Constants: `GOLDEN_ROOT` (align_images eval root), `HISTORY_ROOT` (consensus pool root), `LAB_MODE` (`""`|`edge_ncc`), `MIN_S`.
- **Consensus history pool:** lives in a **separate root keyed by `<class>/<recipe>` only (eqp-independent — same recipe shares one pool across tools)**: `<HISTORY_ROOT>/<class>/<recipe>/events/<event_id>/S*.jpeg` (+ `.<img>/cond.txt`), the same format `office_success_downloader` writes. Production `align/assets.py` is untouched; `gce._history_images` reads this root directly. `_consensus_template_ab` is **history-first + LOO fallback**: history pool ≥ `min_s` → consensus from that disjoint pool (eval on `from_msr` S, no leakage, no LOO); else the byte-identical `from_msr` leave-one-out path. Office collects **class·recipe·modality-wise ~8–10 most-recent S (rolling, S only)**.

The filesystem contract (office MES writes, `align` reads):

```
align_images/<eqp_id>/<class>/<recipe>/
├─ align_img_from_rcp/      IMAP0001.*(OM)  IMAP0002.*(SEM)   # recipe-registered align key (office MES)
├─ align_img_from_msr/      S*/E*                             # measurement trajectory (E = fail) (office MES)
└─ captured_img_from_rcs/   <tag>/…                           # fail-time captures + recording/ (workflow_3 writes)
```

- **Runtime no longer consumes `align_img_from_msr`** (2026-06-18): correction/feasibility match consensus(preferred)/rcp(fallback) templates into the live capture, so the production loop (`align_fail_monitor`, `align_fail_monitor_only_check`) downloads **rcp only** (`gather_rcp_msr(..., include_msr=False)`). msr is offline-bench-only — fetch it on demand with `poc/workflow_3/monitor/fetch_msr_offline.py` (`include_msr=True`).

- **Production consensus cache is eqp-independent** (`ALIGN_CONSENSUS_CACHE_DIR`, distinct from the eqp-keyed `align_images` tree above): `<cache_root>/<class>/<recipe>/events/<event_id>/S*.jpeg` — **no `<eqp_id>`** (same recipe pools across tools; matches the bench `HISTORY_ROOT` keying + what `office_success_downloader` writes). `consensus_gather._events_dir_for(recipe_id, cache_root)` is the **single** path-construction point and deliberately omits `eqp_id` so it can't be re-added (re-adding splits the pool per-tool and misses the eqp-less office writer → silent permanent rcp fallback). Coupled guard: `monitor/success_gather._IN_FLIGHT` dedupe key is `recipe_id` alone (shared per-recipe staging would otherwise race across tools). The `align_images/<eqp_id>/…` tree (rcp/msr + captures/recordings) stays eqp-keyed — separate MES-contract root. Verify at office (read-only, no RCS/download): `uv run python poc/workflow_3/align/diagnostics/verify_consensus_path.py` (prints `[DIGEST]`). Fixed 2026-06-26.

- Root constant: `ALIGN_IMAGES_DIR` in `poc/workflow_3/__init__.py` (env-overridable). **Default now resolves to `poc/workflow_3/align_images`** (moved 2026-06-11; `.gitignore` tracks the new location). Office MES historically writes align keys to `poc/workflow_1/align_images`, so at the office you MUST either repoint MES output to the workflow_3 tree or set `ALIGN_IMAGES_DIR` to the MES path — otherwise the code reads an empty root and rcp/msr appear absent (captures still land because the loop writes those itself). The check-only monitor prints a path-health report at startup (`_report_data_paths`) to surface this mismatch.
- `align/assets.resolve_assets_auto()` is the single reader (override via `ALIGN_EQP_ID` / `ALIGN_CLASS_NAME` / `ALIGN_RECIPE_NAME` or kwargs).
- `office_*` modules (`office_align_fail_alarm`, `office_rich_notify`) are gitignored and exist only on the office PC; copy them into `poc/workflow_3/monitor/` (the canonical location — workflow_3 loads office adapters only from there; the old `poc.workflow_1.office_*` import fallback has been removed, so a missing adapter just disables that integration with a warning). See `poc/workflow_3/README.md` for the office migration + staged-enablement checklist.

**Authoritative docs:** `poc/workflow_3/README.md` (loop, env, office checklist). New workflow_3 loop/ops docs (specs, ADRs, journals, runbooks) live under `poc/workflow_3/docs/` (authored + git-tracked; generated artifacts go to `debug_images/`, never `docs/`). CV procedure history stays in the bench: `poc/workflow_2/docs/study/runbooks/workflow_2_procedure.md` + ADRs under `poc/workflow_2/docs/study/adr/` (paths in older docs predate the workflow_3 migration).

## Setup & Dependencies

`uv` with `pyproject.toml` (Python >= 3.10). Use uv-managed workflows by default.

```bash
uv sync --extra dev                      # Core project + dev tools
uv pip install -r requirements.txt       # All-in-one
uv pip install -r test/video_frame_parser/requirements.txt  # torch, opencv, pymongo, faiss
```

## Running Modules

All scripts run with just `uv run python <script>.py` (no CLI args — see Code Conventions).

```bash
# workflow_3 — production loop (office Windows)
uv run python poc/workflow_3/monitor/align_fail_monitor.py   # Real-time align-fail monitoring loop
uv run python poc/workflow_3/monitor/align_fail_monitor_only_check.py  # Check-only variant: connect + 1 capture + close (no correction/recording)

# dev-PC dry-run (no office modules; replay one synthetic alarm through the cycle)
SAFE_MODE=1 ALIGN_FAIL_ALARM_SOURCE=replay ALIGN_FAIL_REPLAY_CSV=<fixture.csv> \
  uv run python poc/workflow_3/monitor/align_fail_monitor.py

# workflow_3 — 엔지니어 수동 조작 녹화 (알람 불필요; office Windows, tool 창을 먼저 열어둘 것)
uv run python poc/workflow_3/monitor/manual_record.py        # 열린 Remote Monitoring 창에 붙어 녹화 (기본 600s)
RECORDING_FILTER_INPUT_DIR=<recording 경로> RECORDING_FILTER_MAX_VLM_CALLS=300 \
  uv run python poc/workflow_3/recording_filter/filter_recording.py   # 녹화 -> interaction_timeline.json
WORKFLOW_EXTRACT_INPUT_DIR=<recording_filter 경로> \
  uv run python poc/workflow_3/workflow_extract/extract_workflow.py   # timeline -> workflow.json + workflow.md

# workflow_3 — RCS 자동 조작 시연 (알람 불필요; office Windows). 실클릭이 기본
uv run python poc/workflow_3/monitor/demonstration_rcs_control.py
DEMO_RCS_TOOL_IDS="MCD019,MCDC10" DEMO_RCS_DWELL_SEC=10 \
  uv run python poc/workflow_3/monitor/demonstration_rcs_control.py
SAFE_MODE=1 uv run python poc/workflow_3/monitor/demonstration_rcs_control.py  # 리허설(클릭 차단)

# workflow_3 — 녹화 프레임 -> 시연 mp4 (오프라인; Mac/dev PC 에서도 실행 가능)
uv run python poc/workflow_3/monitor/make_demo_video.py      # 최근 recording 폴더 자동 선택
DEMO_VIDEO_INPUT_DIR=<recording 경로> DEMO_VIDEO_SPEED=2 \
  uv run python poc/workflow_3/monitor/make_demo_video.py    # 구간/배속 지정
DEMO_VIDEO_SEGMENTS="0-30,120-260" \
  uv run python poc/workflow_3/monitor/make_demo_video.py    # 필요 없는 구간 잘라내기

# workflow_3 — 여러 회차(trial) 녹화를 자막과 함께 한 편으로 (오프라인)
# 인자는 파일 상단 상수(ROOT/INPUT_DIRS/LABELS/...)를 고쳐 쓴다. 셸 env 는 1회성 override.
uv run python poc/workflow_3/monitor/make_demo_video_combined.py

# workflow_3 — RCS building blocks (office Windows; each runnable standalone)
uv run python poc/workflow_3/rcs/open_rcs.py                 # Start RcsMainHD.exe only
uv run python poc/workflow_3/rcs/workflow_login.py           # RCS login workflow
uv run python poc/workflow_3/rcs/view_list_tab_rcs.py        # Locate + click the List tab
uv run python poc/workflow_3/rcs/workflow_select_tool.py     # Find a tool in List tab and double-click it
uv run python poc/workflow_3/rcs/workflow_close_tool.py      # Close the opened tool window by tool id in title
uv run python poc/workflow_3/rcs/rcs_screenshot.py           # Capture tool window into captured_img_from_rcs, then close

# workflow_3 — CV engine demos (run on Mac/dev PC, synthetic data)
uv run python poc/workflow_3/align/diagnostics/compare_align_images.py  # static CV compare (falls back to synthetic self-test)
uv run python poc/workflow_3/align/correction.py                       # primary reposition+OK demo (mock, dry-run)
uv run python poc/workflow_3/align/live_search.py                      # two-phase live search demo (mock)

# legacy workflow_1 — CCTV/DVR path only
uv run python poc/workflow_1/monitor_align_fail.py           # Align-fail + open Tool DVR (CCTV) + capture CH4 frames

# Video frame parser
uv run python -m test.video_frame_parser.example_usage
```

`runner/workflow_runner.py` is a library, not an entry point: `WorkflowRunner` runs a `list[WorkflowStep]` sequentially and `ConditionChecker` evaluates step pre/post conditions; runs are journaled under `poc/workflow_3/logs/workflow_runs/`. The per-alarm cycle (`monitor/cycle.py`) is built on it; cleanup (stop recording / close tool / popup backstop) is guaranteed by `try/finally`, not steps.

## Testing

```bash
# align engine — synthetic smoke tests
uv run python poc/workflow_3/align/matching/test_engine.py
uv run python poc/workflow_3/align/test_correction.py                 # incl. error paths
uv run python poc/workflow_3/align/matching/test_engine_ensemble.py
uv run python poc/workflow_3/align/matching/test_ensemble.py
uv run python poc/workflow_3/align/diagnostics/test_match_on_captured_frames.py  # needs office capture fixtures
uv run python poc/workflow_3/rcs/test_tool_name_match.py              # 9/9
uv run python poc/workflow_3/rcs/test_tool_row_verify.py              # 42/42 (row confirm gate + crop tightness)
uv run pytest poc/workflow_3/align/test_grid_search.py                # 31 (search-around: zoom-out 단 선택/격자/odometry/추격 confirm/degrade/cycle 주입)
uv run pytest poc/workflow_3/align/test_fallback_kill_switch.py       # 9 (fallback kill switch + pan 예산 10 이 streak 에 안 잘림)
uv run pytest poc/workflow_3/rcs/test_row_occupant.py                 # 14 (점유 3-상태 판별)
uv run pytest poc/workflow_3/monitor/test_share_request.py            # 35 (확인 게이트/승낙 대기/클릭 경로)
uv run pytest poc/workflow_3/monitor/test_share_cycle_wiring.py       # 30 (occupancy->outcome->notify->retry 배선 + CORRECT_WHEN_OCCUPIED on-branch)
uv run pytest poc/workflow_3/monitor/test_rcs_recovery.py             # 27 (RCS 재실행 중복 가드 + 복구 로그인이 tool 안 여는 계약 + 창 없는 좀비 판정)
uv run pytest poc/workflow_3/monitor/test_recovery_episode.py         # 21 (Episode 식별/attempt 폴더/재시작 재개/Outcome+digest)
uv run pytest poc/workflow_3/monitor/test_guard_readings.py          # 12 (Guard 3종 3상태 + attempt 기록)
uv run pytest poc/workflow_3/monitor/test_measurement_verification.py # 7 (Verification record + unknown-only stub)
uv run pytest poc/workflow_3/monitor/test_numerator_records.py       # 6 (분자 per-read 기록/판정 분류)
uv run pytest poc/workflow_3/monitor/test_frame_meta_recorder.py     # 5 (알람 녹화 사이드카 + manifest additive)
uv run pytest poc/workflow_3/monitor/test_prelude_recording.py        # 4 (접속 구간 화면 녹화 게이트/저장 위치/인계)
uv run pytest poc/workflow_3/monitor/test_demonstration_rcs_control.py  # 124 (시연 흐름 + 확인 게이트 + 클릭/대문자 입력 + Alt+click 가림 해제)
uv run pytest poc/workflow_3/monitor/test_make_demo_video.py          # 16 (prelude 시간축 접합 + 편집 구간 + letterbox)
uv run pytest poc/workflow_3/monitor/test_make_demo_video_combined.py # 17 (회차 정렬/번호/시간축 리셋/공통 캔버스)
uv run python poc/workflow_3/vlm/test_label_verify.py                 # 23/23 (shared point->label OCR verifier)

# tool locator VLM combo bench (office; RCS logged in, List tab visible; no alarm, no clicking)
uv run python poc/workflow_3/rcs/bench_tool_locator.py
BENCH_REPEATS=1 uv run python poc/workflow_3/rcs/bench_tool_locator.py   # smoke first (48 runs); full default = 4 combos x 12 tools x 3 = 144 runs / ~432 VLM calls

# tool WINDOW reader bench (office; a tool already open). buttons arm = no click, no mouse move.
uv run python poc/workflow_3/rcs/bench_tool_window_reader.py
BENCH_CURSOR_ARM=1 SAFE_MODE=0 uv run python poc/workflow_3/rcs/bench_tool_window_reader.py  # + cursor-tracking arm (moves mouse, never clicks)

# recording_filter — offline frame-filter unit tests (pytest-style, 71 tests: Stage 1/1.5/2a/2c + wiring)
uv run pytest poc/workflow_3/recording_filter

# workflow_extract — 그룹핑/렌더 단위 테스트 (VLM 불필요, Mac 실행 가능)
uv run pytest poc/workflow_3/workflow_extract

# monitor — engineer-done + success-gather + manual-record smoke tests (run directly)
uv run python poc/workflow_3/monitor/test_engineer_done_align_adjustment.py
uv run python poc/workflow_3/monitor/test_success_gather.py
uv run python poc/workflow_3/monitor/test_manual_record.py                    # 47 (EQP 파싱/예산/가림 판정/teardown)

# Video frame parser unit tests
uv run pytest test/video_frame_parser/tests/

# vlm_input_control integration (safe mode by default; toggle via SAFE_MODE in .env)
uv run python -m test.vlm_input_control.integration_test
```

## Code Conventions

- **Korean docstrings** throughout all modules. **예외: `flask_api/vlm_serve/*.py` 의
  route-template stub 들은 영어 한 줄 docstring 을 유지한다** (기존 5개 파일의 선례,
  사용자 판정 2026-09-03). 새 서비스를 추가할 때도 형제 파일과 같은 영어 한 줄로 맞출 것 -
  이 패키지만 섞이면 일관성이 더 나빠진다. 그 밖의 모든 모듈은 한국어 docstring 이다.
- **No `__future__` imports by default**: do not add `from __future__ import annotations` (or any `__future__` import) unless explicitly asked.
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (never the `logging` module). Exception: `poc/workflow_3/logger.py` uses Python `logging` with `RotatingFileHandler` for the audit trail (`poc/workflow_3/logs/vlm_calls.log` for VLM calls, `work2.log` for general events). Avoid em-dash (U+2014) inside `print()` strings — the office console is cp949 and cannot encode it (docstrings are fine).
- **Absolute imports** within `poc/`: use `from poc.workflow_3.xxx import ...`; legacy packages import from workflow_3, never the reverse.
- **`__all__` in `__init__.py` is optional**: only add it when it provides clear value for a curated package API.
- **Image format convention**: save debug screenshots locally as **JPEG**; convert to **WebP** (quality=90) when sending to VLM APIs to cut payload size without hurting accuracy.
- **Safe mode**: interactive modules respect `SAFE_MODE` (blocks real mouse/keyboard output). `action_enabled`/`typing_enabled` default to the inverse of `SAFE_MODE` in `WorkflowSettings`. CV correction has a second gate: real reposition/OK clicks require `SAFE_MODE=0` **and** `ALIGN_FAIL_CORRECTION_DRY_RUN=0`.
- **No CLI arguments**: do not use `argparse` or flags. Configuration comes from `Workflow3Settings` (`poc/workflow_3/config.py`, extends `WorkflowSettings`), `vlm/flask_vlm.py` constants, or environment variables. Scripts must run with just `uv run python <script>.py`.
- **진입점 상단 상수 블록이 "인자"다** (2026-08-31): 시나리오 하나 = 진입점 `.py` 하나이므로, 그 실행의 knob 은 **실행하는 파일 맨 위**에 산다. 목적별 설정 폴더를 새로 만들지 않는다 - 폴더는 knob 개수를 안 줄이면서 공유 knob 사본을 N개로 늘리고(`_apply_live_mode_defaults` 가 오피스 사본의 `CORRECTION_DRY_RUN=1` 을 막으려고 생긴 것과 같은 사고), gitignored 사본을 Claude 없는 오피스 PC 에서 N개 관리하게 만든다. 두 가지 모양이 있다 - ① **모듈이 자기 env 를 직접 읽으면 상수를 기본값 인자로**(`_env_float("DEMO_RCS_DWELL_SEC", DWELL_SEC)`; `demonstration_rcs_control`/`manual_record`/`make_demo_video`) ② **다른 모듈이 읽으면**(`config.py` 의 `load_workflow3_settings`) `util/env_utils.seed_env_from_constants(globals(), _CONST_TO_ENV, label=...)` 로 시딩(`align_fail_monitor`/`_only_check`/`make_demo_video_combined`). 어느 쪽이든 `setdefault` 라 **셸 env > 파일 상수 > 코드 기본값**이고, 셸 env 에 밀려 무시된 상수는 반드시 콘솔에 찍힌다(사본 파일에 없는 자기고발 장치 - 이것 때문에 파일 상수가 폴더 사본보다 안전하다). `align_fail_monitor` 의 시딩 순서는 `_apply_live_mode_defaults` → 상수 블록 → `workflow_3_config.py` 이며, 상수 블록이 오피스 사본보다 앞서는 이유는 이 파일만 git 에 추적되어 리뷰를 거치기 때문이다. **`.env` 는 남긴다 - 비밀값 전용**(`ACTION_LOGIN_PASSWORD`); `rcs/` standalone 의 `load_dotenv()` 는 그 경로라 상수 블록으로 옮기지 않는다. 규칙은 "`.env`=비밀값, 상수 블록=동작". `align_fail_monitor` 는 **구현이 끝난 기능을 켜고 시작**하되(사용자 결정), 되돌릴 수 없거나 남에게 영향을 주는 넷(`BLOCK_INPUT`/`RCS_KILL_STALE`/`ACCESS_GRANT`/`CORRECT_WHEN_OCCUPIED`)과 반자동 계약인 `OK_CLICK` 은 `[위험]` 주석과 함께 off 다. 계약 검증은 `util/test_env_utils.py` (10) - 특히 상수 표의 env 이름이 실제 reader 가 읽는 이름 집합에 있는지 대조한다(오타는 조용히 아무 일도 안 하므로).

## Development Workflow

Development is **mixed macOS + Windows**:

- On **macOS**, Claude Code cannot see or drive the actual RCS application. Windows-only paths (RCS, pywinauto, pynput mouse/keyboard) are edited on Mac, pushed via git, pulled at the office, and run there; debugging relies on the user reporting console output and debug screenshots in `poc/workflow_3/debug_images/` (per-model subdirs).
- On **Windows** (office machine), Claude Code runs directly and can execute the automation scripts itself.

Pure-CV and synthetic-data work in `workflow_3/align` (e.g. `diagnostics/compare_align_images.py`, `matching/test_engine.py`) and the replay-source loop dry-run run and are verified on any dev machine without RCS.

## Architecture Notes

### Flask Proxy VLM Architecture

VLM calls route through a Flask proxy at the company server, which provides unified health discovery and per-service routing.

- **Service registry (server side)**: `flask_api/vlm_serve/config.py`, one `VLMServiceEntry` dataclass per model.
- **Registered services**: mai-ui (8002), paddleocr-vl-1.5 (8004), qwen3.8-27b (8006) are enabled and served; mai-ui-2b (8007) is registered `enabled=False` (A/B bench only). **ui-venus / ui-tars / got-ocr are gone as of 2026-09-03** — weights deleted from the server, so their registry entries, route stubs, deploy env files and start scripts were all removed on both server and client. A call to those slugs now fails at slug resolution (`get_service_by_slug` returns `None`), not at the proxy. Reviving one needs the checkpoint re-imported first; git history is the restore path (the route stub is a 13-line `service_template` copy).
- **Health endpoint**: `GET /api/vlm_serve/health`.
- **Proxy URL pattern**: `{flask_base}/api/vlm_serve/{service_slug}/v1/chat/completions`.

### `flask_api/model_upload/` — 모델 가중치 청크 업로드 (2026-08-22)

사내 private cloud 의 Flask 서버로 openweight 모델을 올리는 경로. code-server 웹
드래그앤드롭이 1GB 근처에서 깨지던 것을 대체한다 - 요청 하나에 파일 하나를 싣지 않고
청크로 쪼갠다. 3계층: `store.py` (HTTP 를 모르는 파일시스템/상태 계층 - 이어받기·무결성
로직이 전부 여기 산다), `routes.py` (blueprint, store 주입식), `config.py` (env + 배선).
클라이언트는 `deploy_vlms/scripts/upload_model.py` (stdlib + requests 만, 서버 코드를
import 하지 않는다). 계약 셋 — ① **committed offset 은 기록보다 `.part` 실제 크기가
우선**(상태만 남고 파일이 없으면 seek 이 0 으로 채운 구멍을 만든다) ② **청크 sha256 이
안 맞으면 offset 을 전진시키지 않는다**(소켓이 예외 없이 일찍 닫히는 짧은 바디가 조용히
구멍을 만드는 것을 막는 유일한 방어선) ③ **완료 시 전체 파일 재해싱**(청크 검증은
네트워크만 덮고 디스크/조립 손상은 못 잡는다). staging 은 반드시 목적지 루트 안쪽 -
`os.replace` 가 원자적이려면 같은 파일시스템이어야 한다. 앞단 프록시의
`client_max_body_size` 가 청크보다 작으면 Flask 에 닿기도 전에 413 이 나므로
`/health` 가 서버 상한을 알리고 클라이언트가 청크를 자동으로 줄인다. 운영 문서
`deploy_vlms/UPLOAD.md`. 테스트 50개 전부 Mac 에서 실장비/실서버 없이 돈다
(`uv run pytest flask_api/model_upload deploy_vlms/scripts`) - 마지막 3개는 로컬에
진짜 werkzeug 서버를 띄워 클라이언트<->서버 계약을 왕복 검증한다.

### `poc/workflow_3/vlm/flask_vlm.py` — client config hub

Defines `ALL_VLM_SERVICES` (a `list[VLMServiceEntry]`) plus `DEFAULT_*` service/model constants. Two connection modes:

- **`proxy`** — Flask-routed UI/OCR models: `mai-ui-8b` (**primary grounding model — all VLM defaults, 2026-08-07**), `paddleocr-vl-1.5` (OCR assist), `qwen3.8-27b` (general reasoning, not a grounding/OCR default). `ui-venus` / `ui-tars` / `got-ocr` were **removed from the client registry too** (2026-09-03) — the `*_SERVICE` rollback env vars can no longer name them.
- **`direct`** — company LLM gateway (`http://common.llm.skhynix.com/v1`): `Kimi-K2.5`, `Qwen3-VL-30B-Instruct`.

Helpers: `get_service_by_slug()`, `resolve_service_proxy_url()`, `resolve_service_api_key()`. Per-model debug dirs live under `debug_images/<model-slug>/` (slug via `resolve_debug_model_name()` in `poc/workflow_3/__init__.py`).

Run/step tuning lives in `Workflow3Settings` (`poc/workflow_3/config.py`, extends `WorkflowSettings` in `runner/workflow_config.py`): retry budget, settle/poll timings, verify service (`paddleocr-vl-1.5`), `service_fallback_order` (`mai-ui` alone — no fallback since 2026-09-03), plus loop fields (poll/recording/watch intervals, correction toggles, alarm source). Build it with `load_workflow3_settings()` (env overrides applied; legacy `ALIGN_FAIL_*` env names preserved).

- **Local config (`workflow_3_config.py`, edit-often scratch — distinct from `config.py`):** `config.py` is the authoritative **schema/reader** (defines `Workflow3Settings` defaults + the `ALIGN_FAIL_*`/`SAFE_MODE` env names it reads); `workflow_3_config.py` is a **gitignored convenience front-end** of plain constants (copy from `workflow_3_config.example.py`) that `workflow_3_config_loader.seed_env()` bridges into env *before* `load_workflow3_settings()` runs — so you set toggles in one file instead of a long `ALIGN_FAIL_X=… uv run …` line. One-way flow: `workflow_3_config.py` constants → `seed_env()` (`os.environ`) → `config.py` reads env. **Precedence: real shell env > `workflow_3_config.py` > `config.py` defaults** (seed is setdefault; the loader prints which config values were ignored because env already set them). It can only set vars `config.py` already reads — it never adds a setting, and deleting it just falls back to `config.py` defaults (a malformed scratch file warns + falls back, doesn't crash). `seed_env()` is called in both monitors' `__main__` (`align_fail_monitor.py`, `align_fail_monitor_only_check.py`). Same pattern as workflow_2's `golden_eval_config.py`. (`ALIGN_IMAGES_DIR` is read at package import, *before* `seed_env()`, so it must come from real env or its default — not controllable here.)

**VLM 모델 통일 (2026-08-07):** every VLM default is now **`mai-ui`** — the project goal is to retire `ui-venus`. Switched in two steps: the 2-stage locator (`vlm/ui_venus_mai_locator.py` `DEFAULT_COARSE_SERVICE`/`DEFAULT_REFINE_SERVICE`, commit `64ef936`) and then every single-call service (`sem_box`/`ok_button`/`occupied_popup`/`engineer_done`/3e `abort_button`, `d0b0a8a`). Office-verified with `SAFE_MODE=0`: login / View→List tabs / select tool / screenshot / close tool, both benches (`bench_tool_locator`, `bench_tool_window_reader` acc=1.000), and a replay check-only cycle (SEM box + PM box/modality correct). Still unexercised: OK button, occupied popup, engineer-done counter, 3e abort — each needs its situation to occur. **그 롤백 경로는 2026-09-03 에 닫혔다** — ui-venus 가중치를 삭제했으므로 `VLM_LOCATOR_COMBO="ui-venus>mai-ui"` 같은 복귀는 불가능하다. per-service env (`ALIGN_FAIL_{SEM_BOX,OCCUPIED_POPUP,ENGINEER_DONE_VLM}_SERVICE` / `ALIGN_OK_BUTTON_VLM_SERVICE` / `MEAS_FAIL_ABORT_BUTTON_SERVICE`) 는 그대로 있지만 고를 수 있는 값은 살아 있는 slug 뿐이다. Note `VLM_LOCATOR_COMBO` is read at call time and `rcs/` standalone scripts never call `seed_env()`, so for those it must come from real shell env, not `workflow_3_config.py`.

**Replay dry-run without a real alarm** (the only way to exercise in-tool VLM paths on demand): copy `poc/workflow_3/monitor/replay_fixture.example.csv`, set `EQP_ID`/`RECIPE_ID`, then `ALIGN_FAIL_ALARM_SOURCE=replay` + `ALIGN_FAIL_REPLAY_CSV=<path>`. `ALID` must be `9006`; rows are emitted on the **first poll only** (then empty, so the edge-trigger release path runs too).

**엔지니어 수동 조작 녹화 (2026-08-10, `MANUAL_RECORD_*` env namespace):** 알람과 무관한 별도 진입점. 엔지니어와 "지금부터 녹화하겠다"고 약속한 뒤 **이미 열려 있는** Remote Monitoring 창에 붙어 수동 작업을 녹화한다 — 접속(tool 더블클릭)은 하지 않는다. 목적은 모방 학습/절차 분석용 원천 데이터 확보이며, 지금 단계의 산출물은 자동화가 아니라 **"의미 있는 데이터가 나오는가"에 대한 판단 근거**다. 설계/계획: `docs/superpowers/{specs,plans}/2026-08-10-manual-recording-session*.md`.

- **런처** `monitor/manual_record.py` — 창 제목 `"Remote Monitoring System - <EQP>"` 에서 EQP 를 뽑아 `align_images/<EQP>/_manual/<tag>/recording/` 에 적재. 창이 2개 이상이면 목록만 출력하고 종료한다(`MANUAL_RECORD_EQP_ID` 로 지정; **부분 일치가 모호하면 임의 선택하지 않고 거부** — 엉뚱한 장비를 10분 녹화하느니 다시 실행하는 편이 낫다). `RecordingSession` 은 **감싸기만** 하고 동작을 바꾸지 않는다(`capture_fn` 주입점). 상한: `MANUAL_RECORD_MAX_SEC` (600, 실질 상한) / `MAX_FRAMES` (기본은 `max_sec/poll_sec x 1.25` 파생, 15000) / `MAX_DISK_MB` (4000) — 뒤 둘은 백스톱이며, **샘플링 주기에서 파생**되므로 poll 을 올려도 실질 상한보다 먼저 걸리지 않는다(고정 4000 이던 시절 0.05s 로 바꾸자 10분 세션이 ~3분에 `frame_budget` 으로 끊겼다). 예산 판정은 `RecordingSession` 이 프레임을 쓰는 자리에서 직접 한다. 그 외 `POLL_SEC` (0.05), `JPEG_QUALITY` (85; 알람 녹화는 종전대로 95), `META` (1). 정지 사유는 manifest 에 `user_interrupt`/`max_sec`/`window_gone`/`frame_budget`/`disk_budget`/`watch_error` 로 남고, **어느 경로로 끝나도 teardown 은 완료된다**.
- **사이드카** `monitor/frame_meta.py` → `frame_meta.jsonl` (프레임당 1줄: 창 rect, 전면 창 제목, 가림 여부, 로컬 커서 좌표). `capture_window` 가 창 핸들이 아니라 **창 rect 의 mss 스크린 그랩**이라 다른 앱이 위에 뜨면 그 앱이 찍히므로, 가림을 프레임 단위로 기록해 분석에서 걸러낸다. 가림 판정은 창 영역 5점에서 `WindowFromPoint` → **`GetAncestor(.., GA_ROOT)` 로 정규화 후** 우리 창인지 비교(정규화를 빼면 자식 컨트롤 HWND 가 잡혀 **전 프레임이 `full` 로 오판**된다). 커서는 `GetCursorPos` 폴링이며 **입력 후킹이 아니다 — 키 입력은 기록하지 않는다**. 기록 실패는 1회 경고 후 영구 비활성화(초당 5회 호출이라 콘솔 범람 방지).
- **Stage 2a 사이드카 커서 + Stage 2b 타이핑** (2026-08-11) — 사이드카에 커서가 있으면
  Stage 2a 가 VLM 커서 탐지를 건너뛴다(`cursor_source` 필드로 구분; 알람 녹화는 사이드카가
  없어 기존 VLM 경로 그대로). Stage 2b 는 **커서 정지 + 국소 반복 변화**로 타이핑 구간을 찾아
  구간 시작/끝 OCR 2콜로 값을 복원하고, before == after 면 캐럿 깜빡임으로 보고 버린다.
  `MANUAL_RECORD_*` 가 아니라 `RECORDING_FILTER_TYPING_*` 네임스페이스다.
- **분석 접합** — 사이드카와 프레임은 **`t_sec` 최근접**으로 조인한다(캡처 순번과 저장 seq 는 어긋난다: 변화 없는 샘플은 저장되지 않음). 조인 상한 `META_MAX_JOIN_GAP_SEC` (10.0) 를 넘으면 meta 없음으로 취급 — 사이드카가 중간에 죽어도 낡은 rect/커서에 영구히 조인되지 않는다. **화면→프레임 커서 변환은 반드시 frame/rect 배율 보정**을 거친다(오피스 125/150% 배율에서 단순 뺄셈은 어긋나 라이브 박스 좌·상단 20% 구간의 실제 조작이 `ambient` 로 버려진다; `util/window_utils.image_point_to_screen` 과 같은 규약). 사이드카가 없는 기존 알람 녹화는 게이트가 전량 통과로 degrade 한다(실패 아님).
- **타임라인 스키마** (`interaction_timeline.json`) — `element` / `element_source` (`ocr`|`vlm`|`none`) / `target_kind` (`ui_control`|`live_image`|`unknown`) / `region` / `generation` / `occlusion`. `target_kind` 는 **A 장비 → B 장비 이식 가능성** 표시다: 같은 RCS exe 라 라벨은 재탐색 가능하지만 좌표는 창 위치가 달라 무의미하고, 라이브 영상 위 조작은 CV 재해석이 필요하다. `element_source` 를 따로 두는 이유는 OCR 로 읽은 라벨과 VLM 이 서술한 라벨의 신뢰 수준이 다르기 때문(이식성 판단 시 `ocr` 만 신뢰하는 식으로 필터 가능).
- **첫 오피스 실행 시 주의** — Stage 2a 는 `max_vlm_calls` 기본 0(무제한)이라 10분 세션이 수백~수천 콜이 될 수 있다. 첫 회는 `RECORDING_FILTER_MAX_VLM_CALLS=300` 으로 상한을 걸 것(잘린 양은 `summary.json` 의 `truncated`/`skipped_due_to_cap` 에 정직하게 보고된다). 확인 포인트 3가지: manifest 의 `sampled_count/경과시간` (실측 샘플링 주기, 목표 ~20/s), `region_map_gen0.jpg` 의 시안 박스가 실제 라이브 SEM 영역과 맞는지(**틀리면 이후 게이팅 전부 무효 — 여기서 멈출 것**), `summary.json` 의 `gate_passed/total_change_events` (90%+ 제거면 정상, 0% 면 사이드카 조인 의심). 전량 폐기 시 `run_filter` 는 성공이 아닌 상태를 반환한다.

런타임 env 플래그 레퍼런스(반자동 보정 게이트, foreground takeover, SEM-box/PM mode 검출,
occupied popup, 실패경로 쿨다운, zoom ladder + PM dropdown)는 `workflow3-env-flags` 스킬에
있다 - 기본값/튜너블/롤백 스위치가 필요할 때 불러 쓴다.

### `poc/workflow_3/vlm/prompts/` prompt builders

Each builder returns a `(system_message, user_message)` tuple and takes image `width`/`height` plus target params.

- `prompt_login_rcs_ui_venus.py` — coarse bbox for Server / UserID / Password / Login / Cancel / Shortcut.
- `prompt_login_rcs_mai_ui.py` — refined click point on the cropped+zoomed region (2-stage locator).
- `prompt_ocr_assist.py` — OCR text extraction.
- `prompt_recipe_monitor_counter.py` — grounds the Recipe Monitor measurement counter (N/M) for engineer-done detection.

### `poc/workflow_3/align/` — align-key engine

Design rule (confirmed 2026-05-25): **OpenCV produces quantitative scores and final coordinates; VLM only identifies regions, explains ambiguous FOVs, and assesses feasibility.** Never let a VLM answer override a low CV score or decide a repeatable stage transition.

- `matching/engine.py` — match engine (the coordinate authority). Ensemble path (`compute_align_key_score_ensemble`: C1/C2/C3 proposer RRF + NCC rerank + MIND self-similarity rerank, Youden-calibrated thresholds 0.6053/0.4727) for paused/static frames; lightweight `compute_align_key_score` for live broad-scan. `MatchPolicy` / `DEFAULT_POLICY` / `STRUCTURE_POLICY`; scale bands `DEFAULT_SCALES` (immutable) and `BROAD_SCALES` (low-mag miniature search).
- `matching/mind_rerank.py` — **modality-aware rerank** on top of the NCC selection inside `compute_align_key_score_ensemble` (ported 2026-07-20~21 from the workflow_2 registration A/B, 67 recipes/334 pts). Branches on `template.key_type` (`is_sem_template`): **OM** = sel order ⊕ MIND(self-similarity) order via RRF (`prod_mind`, d=+0.042 > NCC-only +0.009); **SEM** = ECC(cc) rank **alone**, not RRF-combined (`route_sw` 0.826 > route3 combined 0.820 — ECC dominates SEM so mixing dilutes it). Rank-only in both paths (never emits new coordinates — picks among existing candidates; sub-pixel proven moot by route_sw raw==ref); all-rejected → NCC selection unchanged. Kill switches `ALIGN_FAIL_MIND_RERANK=0` (OM), `ALIGN_FAIL_ECC_RERANK=0` (SEM). Keep constants bit-parity with `poc/workflow_2/registration_lab.py` (the bench measures against this implementation).
- `assets.py` — resolves/loads the `align_images/...` tree (see Active Workstreams).
- `templates.py` — materializes a recipe align image into an `AlignKeyTemplate` (cond-aware via `cond_template`: box-crop + decoupled `align_offset_xy`, gated by `ALIGN_FAIL_COND_BOX_CROP`).
- `ok_button.py` — VLM locator for the Align Fail dialog's OK button (screen-absolute coords; VLM identifies the button region only, never the align coordinate).
- `correction.py` — **primary correction entry** (`correct_align_fail_auto`): `key_visibility_gate` decides primary (reposition best_xy + OK click) vs fallback; `CorrectionOutcome.status` ∈ {corrected, **awaiting_engineer_ok**, fallback_*, escalated_ambiguous_key, escalated_no_ok, ok_detect_error, no_assets} drives the cube-notify decision in `monitor/notify.py`. **반자동 모드** (`CorrectionConfig.ok_click_enabled=False`; 운영 루프 기본값): reposition 더블클릭까지만 자동으로 하고 OK 는 누르지 않은 채 `awaiting_engineer_ok` 로 끝낸다. 이 상태값이 따로 있는 이유는 `notify_correction_outcome` 이 `corrected` 면 cube 를 생략하기 때문 — `require_ok_button=False` 로 OK 만 건너뛰면 `corrected` 가 반환되어 "OK 눌러달라"는 알림이 조용히 사라진다(회귀 방지 테스트: `test_correction.py:test_awaiting_engineer_ok*`). `corrected` 가 아니므로 엔지니어 watch 도 계속 돌아 OK 를 누르는 장면까지 녹화된다.
- `diagnostics/feasibility_check.py` (`mark_align_feasibility` → `FeasibilityResult`) — beyond the verdict/`[NON-DISTINCT]` banner it now draws the **2nd-best candidate** (magenta box+"2nd" from `result.candidates[1].xy`, the look-alike that drives the ambiguity) on `_marked.jpg`, and sets `reregister_recommended` (= verdict `ambiguous`, i.e. `second_ratio > reregister tau` — a chronic-ambiguous align key). `_feasibility.json` gains `second_xy`/`reregister_recommended`; `monitor/cycle.py` surfaces the recommendation to `result.notes` + a `reregister_recommended` audit-log line so the engineer sees which recipes need their align key re-registered on a more distinctive region.
- `live_search.py` — two-phase fallback search. Physical conventions: **double-click = recenter on click point, wheel = discrete FOV-centered zoom, template routing by OM/SEM mode.** Phase A broad zoom-out + spiral pan (budget 10); Phase B recenter → zoom-in → confirm. Real equipment is isolated behind the `SEMMonitorController` Protocol (Mac mock in same file; real adapter = `sem_monitor/controller.RCSSEMMonitor`).
- Office calibration **done** (2026-07-07): the former gaps — SEM panel landmarks (`poc/workflow_3/templates/sem_panel_landmarks/`), double-click/wheel↔magnification calibration, `read_mode()` real implementation, zoom/click-coordinate + engineer-done-detection tuning — are calibrated on the office PC. Still open: real-data accuracy/threshold confirmation on office data (진행 중) and the joint evaluation with field engineers (실전 테스트, 2026-07~08); see `docs/project_progress/00_executive_summary.md` §7.

### `test/video_frame_parser/`

CLIP-based video frame extraction and analysis for GPU cluster environments. MongoDB for metadata, FAISS for similarity search. For imports across `test/` siblings, use `from video_frame_parser.xxx import Yyy` with `PYTHONPATH=./test`.

## Agent skills

### Issue tracker

Issues are tracked as markdown files under `docs/issues/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + ADRs (root `docs/adr/` and per-workflow `poc/workflow_*/docs/study/adr/`). See `docs/agents/domain.md`.
