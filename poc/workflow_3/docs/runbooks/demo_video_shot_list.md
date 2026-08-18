# Align Fail Monitor 시연영상 촬영 대본

- **목적**: 동작 증명 (성과 보고용)
- **시나리오**: replay 고정 (알람 타이밍만 우리가 고름)
- **반출 범위**: 오피스 내부 전용 (블러 불필요, 실제 EQP/recipe ID 노출 무방)
- **대상 진입점**: `poc/workflow_3/monitor/align_fail_monitor.py`
- **작성일**: 2026-08-18

---

## 0. 이 대본의 대전제 — 꼭 먼저 읽을 것

`ALIGN_FAIL_ALARM_SOURCE=replay` 가 가짜로 만드는 것은 **알람 row 한 줄뿐**이다.
그 뒤의 RCS 접속 / tool 창 인식 / 상시 녹화 / SEM panel ROI 검출 / 템플릿 매칭 /
reposition 더블클릭은 **전부 실제 장비를 상대로 실제로 실행된다.**

그래서 성립 조건은 하나다:

> **fixture 의 `EQP_ID` 는 촬영 시점에 실제로 align fail 로 멈춰 있는 장비여야 한다.**

정상 가동 중인 장비에 replay 를 쏘면 `run_correction` 이 align fail 화면이 아닌 것을
매칭하게 되고, 그 컷은 동작 증명이 아니라 반증 자료가 된다. 이 조건만 지키면
"알람이 오는 순간만 우리가 고르고, 나머지는 전부 진짜"가 되어 재현성과 진정성을
동시에 얻는다.

### 촬영 창(window) 확보 방법

align fail 이 나면 장비는 엔지니어가 개입할 때까지 멈춰 있다. 즉 알람 발생 후
**수 분~수십 분의 촬영 가능 구간**이 생긴다. 그 구간에 들어가면:

1. 엔지니어에게 "이 장비 지금 손대지 말아달라" 를 먼저 확보한다.
2. fixture CSV 의 `EQP_ID` / `RECIPE_ID` 를 그 장비 값으로 바꾼다.
3. 화면 녹화를 켜고, 그 다음 모니터를 띄운다. (순서 중요 — 기동 배너부터 담아야 한다)

---

## 0.5 실알람 자동 녹화를 함께 쓴다 (권장)

"그냥 실제 align fail 났을 때 녹화하는 게 더 확실하지 않나" 는 **반은 맞다.**
그리고 그건 이미 **공짜로 돌아가고 있다.**

사이클의 `start_recording` step 은 모든 align fail 에서 tool 창을 상시 녹화한다.
즉 production 모니터를 띄워두기만 하면, **아무도 안 보고 있어도** 실알람마다
프레임이 쌓인다. 재현 노력 0, 진정성 100.

### 다만 자동 녹화가 담지 못하는 것

녹화는 `start_recording` 부터 시작하고 **tool 창 rect 만** 찍는다. 그래서 빠지는 것:

- 터미널 콘솔 (기동 배너, step 로그) - "구동 방식" 설명의 절반
- RCS 메인 창 / List 탭 / **tool 더블클릭**(`connect_tool` 은 녹화 시작 *이전*이다)
- cube 알림이 도착하는 화면

> 앞의 두 개는 **로그 패널(§5)로 상당 부분 되살릴 수 있다** - 화면 그림은 못 되살리지만
> "그 시각에 시스템이 무슨 판단을 했는가" 는 프레임 옆에 텍스트로 붙는다. 이미 지나간
> 실알람 녹화에도 소급 적용되므로, 화면 녹화를 못 걸어둔 건에 대한 유일한 수단이다.

### 결론: 대체가 아니라 역할 분담

| | 실알람 자동 녹화 | replay 촬영 |
|---|---|---|
| 얻는 것 | 진짜 fail → 진짜 보정 (결과 증명) | 콘솔·접속·안전장치 (구동 방식) |
| 비용 | 0 (이미 쌓임) | 촬영 조율 필요 |
| 타이밍 | 통제 불가 | 통제 가능 |

**권장 순서**: ① 먼저 이미 쌓인 녹화가 있는지 확인한다 → ② 쓸 만한 게 있으면
그것으로 컷 ④ 본체를 만들고, replay 촬영은 컷 ②③⑤⑥⑦(맥락)만 담당하게 한다.
이러면 촬영 부담이 크게 줄고 영상의 진정성은 올라간다.

### 이미 쌓인 녹화 확인

```
uv run python poc/workflow_3/monitor/analyze_cycle_manifest.py
```

`align_fail_cycles.csv` 의 `outcome_status` 와 `frame_count` / `recording_dir` 을 보고
쓸 만한 사이클을 고른다. `awaiting_engineer_ok` 또는 `corrected` 이면서
`frame_count` 가 충분한 것이 좋은 후보다. 고른 폴더를 §5 의 스크립트에 넣으면 된다.

> 자동 녹화만으로 영상을 만들 계획이면 이 문서의 §2 env 조정과 §4 컷 ②③은 건너뛰고,
> §5 (프레임 → mp4) 와 §6 (산출물) 만 보면 된다.

---

## 1. 촬영 전 체크리스트

- [ ] 대상 장비가 지금 align fail 상태인가 (엔지니어 확인)
- [ ] 해당 장비를 **다른 사람이 점유하고 있지 않은가** — 점유 중이면 화면 공유 요청
      경로로 빠져 `view_only_observation` 으로 끝나고, 보정 컷을 못 얻는다
- [ ] RCS 로그인 세션 상태 (이미 로그인돼 있으면 로그인 컷이 안 나온다.
      로그인 장면도 담고 싶으면 RCS 를 미리 닫아둘 것 — 루프가 자동 재실행+재로그인한다)
- [ ] `ALIGN_IMAGES_DIR` 이 MES 적재 경로를 가리키는가 (rcp 가 없으면 보정이 `no_assets`)
- [ ] 관리자 권한으로 터미널 실행 (시작 로그의 권한 진단 줄이 경고면 전면화가 불안정)
- [ ] 터미널 폰트 크게 (발표 화면에서 `[INFO]` 줄이 읽혀야 한다)
- [ ] 다른 창 정리 — 녹화 중 알림 팝업(메신저 등) 금지

---

## 2. env 세팅

fixture CSV 를 복사해서 만든다.

```
cp poc/workflow_3/monitor/replay_fixture.example.csv <작업경로>/demo_fixture.csv
```

내용 (헤더 그대로, EQP_ID / RECIPE_ID / UTC9 만 실제 값으로):

```csv
EQP_ID,RECIPE_ID,ALID,UTC9,ALARM_NAME,OPERATION_DESC,LOT_TYPE_CD
<실제장비>,<class/recipe>,9006,<촬영직전시각 YYYY-MM-DD HH:MM:SS>,Align Fail,,PROD
```

> `UTC9` 는 `detection_window_sec` 윈도우 안에 들어와야 한다. 촬영 직전 시각으로 넣을 것.
> `ALID` 는 반드시 `9006`.

실행 env (셸 env 가 `workflow_3_config.py` 보다 우선한다):

```
ALIGN_FAIL_ALARM_SOURCE=replay
ALIGN_FAIL_REPLAY_CSV=<작업경로>/demo_fixture.csv
ALIGN_FAIL_POLL_SEC=5              # 기본 10s -> 감지까지 대기 단축
ALIGN_FAIL_ENGINEER_WATCH_SEC=60   # 기본 300s. 안 줄이면 5분 정지 화면이 생긴다
```

**`SAFE_MODE` 와 `ALIGN_FAIL_CORRECTION_DRY_RUN` 은 건드리지 않는다.**
진입점의 `_apply_live_mode_defaults()` 가 둘 다 `0` 으로 못박아 실클릭 모드로 뜬다 —
이게 동작 증명 영상에서 우리가 원하는 상태다.

---

## 3. 화면 배치 — 나누지 말 것

RCS 는 모니터 하나를 거의 다 차지하므로 **좌우 분할은 성립하지 않는다.**
분할하려 들지 말고 전체 화면을 통째로 녹화한다.

### 권장 (단일 모니터): 전면 창 전환을 그대로 담는다

전체 화면 1개를 녹화하면 전면 창이 이렇게 저절로 바뀐다.

```
터미널(기동/감지)  ->  [ensure_rcs_ready 가 RCS 를 강제 전면화]  ->  RCS / tool 창(사이클)
                   ->  [사이클 종료, tool 닫힘]                  ->  터미널(결과/중단)
```

이 전환 자체가 **"자동화가 스스로 창을 확보한다"** 는 증거 컷이다. 억지로 나눠 담는
것보다 이쪽이 서사가 자연스럽고, 성과 보고에서도 더 설득력 있다.

사이클이 도는 동안 콘솔이 안 보이는 것은 **컷 ⑥에서 스크롤백으로 되짚어** 보완한다.
"지금 보신 화면이 이 로그로 남습니다" 가 오히려 증명 영상에 맞는 순서다.

### 절대 하지 말 것: 터미널을 RCS 위에 always-on-top 으로 겹치기

두 가지가 동시에 깨진다.

1. **클릭이 터미널로 간다.** 자동화는 화면 절대좌표로 클릭하므로, 터미널이 클릭
   지점을 덮고 있으면 그 클릭을 터미널이 먹는다.
2. **녹화 프레임이 오염된다.** `RecordingSession` 은 창 핸들이 아니라 **창 rect 의
   스크린 그랩**이다. 위에 겹친 창이 프레임에 그대로 찍혀, 프레임으로 만드는 확대
   클립(§5)이 못 쓰게 된다.

### 듀얼 모니터가 있다면

보조 모니터에 터미널, 주 모니터에 RCS 를 둔다. 단 PowerPoint 화면 녹화는 한 화면
영역만 잡으므로 두 모니터를 한 컷에 담으려면 별도 도구가 필요하다. 안 되면 주
모니터만 녹화하고 콘솔은 편집에서 PiP 로 얹거나 컷 ⑥에서 스크롤백으로 처리한다.

어느 쪽이든 **RCS 를 축소하지 말 것.** 창을 줄이면 로케이터가 보는 화면이 촬영
때문에 달라져, 평소와 다른 조건에서 찍은 영상이 된다.

---

## 4. 샷 리스트

### 컷 ① 개요 (30초, 사후 편집)

`docs/project_progress/Align_Tuning_Agent.bento.html` 의 루프 다이어그램 슬라이드 캡처.
나레이션: "알람 감지 → 접속 → 녹화 → CV 보정 → 실패 시 엔지니어 인계, 이 한 바퀴를 자동으로 돕니다."

### 컷 ② 기동 (60초)

녹화 시작 → 터미널에 아래를 타이핑해서 실행한다. (미리 붙여넣지 말고 **타이핑하는 장면**을
담을 것. "이 한 줄이면 뜬다"가 메시지다.)

```
uv run python poc/workflow_3/monitor/align_fail_monitor.py
```

**여기서 화면에 뜰 것 (나레이션 포인트):**

```
======================================================================
[WARNING] 실운전 모드: 실제 마우스 클릭이 발생합니다 (접속 더블클릭 + align point reposition).
[WARNING] 점검만 하려면 중단 후 'SAFE_MODE=1' 을 붙여 다시 실행하세요.
======================================================================
[INFO] Align Fail 모니터링 시작 (소스=replay, 주기=5s, 윈도우=...s, 팝업=..., cube알림=on, 사이클=on, 보정=on, OK클릭=off(엔지니어))
[INFO] engineer watch: 상한=60s, 작업완료 감지(카운터+Assist)=on
[INFO] 알람 로그: .../align_fail_alarms.txt
[INFO] 사이클 manifest: .../align_fail_cycles.csv
[INFO] VLM 로케이터 조합: ...
[INFO] 각 신규 Align Fail: RCS 확보 → 접속 → 상시 녹화 → SEM panel → CV 보정 → ...
```

나레이션에서 **`OK클릭=off(엔지니어)` 를 반드시 짚을 것.** 반자동 모드라 좌표만 잡아주고
최종 OK 는 사람이 누른다 — 성과 보고에서 "완전 무인이냐"는 질문이 반드시 나온다.

### 컷 ③ 감지 (20초)

첫 poll 에서 fixture row 가 나온다.

```
[INFO] YYYY-MM-DD HH:MM:SS - 알람 조회 (최근 ...s 윈도우)
[WARNING] Align Fail 감지: EQP_ID=..., ALID=9006, RECIPE_ID=..., LOT_TYPE=PROD, 시각=...
[INFO] 기록 완료 → .../align_fail_alarms.txt
```

### 컷 ④ 사이클 본편 (2~3분) — **영상의 핵심**

7개 step 이 순서대로 흐른다. 각 step 이 콘솔에 찍히는 동안 우측 RCS 화면이 실제로 움직인다.

| # | step_id | 화면에서 보일 것 |
|---|---|---|
| 1 | `ensure_rcs_ready` | RCS 메인 창이 전면으로 올라옴 (없으면 재실행+재로그인) |
| 2 | `close_alert_popup` | 감지 팝업이 닫힘 |
| 3 | `connect_tool` | **List 탭 클릭 → 대상 tool 행 더블클릭** (커서가 스스로 움직인다) |
| 4 | `wait_tool_window` | Remote Monitoring 창이 뜸 |
| 5 | `start_recording` | `[INFO] 녹화 시작: dir=...` |
| 6 | `locate_sem_panel` | SEM box 검출 → controller 생성 |
| 7 | `run_correction` | **align point 로 커서 이동 + reposition 더블클릭** |

> 컷 ④에서 **절대 마우스/키보드를 건드리지 말 것.** 자동화가 커서를 독점하며,
> 사람이 개입하면 클릭이 빗나가고 그 실패가 그대로 영상에 남는다.

`connect_tool` 과 `run_correction` 두 장면은 **편집 시 확대(zoom-in)** 를 걸 것.
증명력의 대부분이 이 두 컷에 있다.

### 컷 ⑤ 결과와 인계 (60초)

보정이 끝나면 `awaiting_engineer_ok` 로 끝나고 cube 알림이 나간다.

- 콘솔의 outcome status
- **cube 알림이 실제로 도착한 화면** (휴대폰/PC 알림 — 이 컷이 "인계까지 자동"의 증거)
- 엔지니어가 OK 를 누르는 장면 (watch 가 그것까지 녹화 중임을 언급)

### 컷 ⑥ 산출물 (60초)

루프가 남긴 것을 파일로 보여준다. 여기가 "증명"의 마무리다.

1. `poc/workflow_3/logs/align_fail_cycles.csv` — 이번 알람 한 줄
   (`run_status`, `outcome_status`, `key_decision`, `best_xy`, `frame_count`)
2. `align_images/<eqp>/<class>/<recipe>/captured_img_from_rcs/<tag>/recording/`
   — 프레임과 `recording_manifest.json`
3. `debug_images/` 의 `_marked.jpg` — 매칭된 align key 와 2nd 후보가 그려진 이미지
   (CV 가 무엇을 근거로 좌표를 정했는지 한 장으로 설명된다)

### 컷 ⑦ 안전장치와 중단 (30초)

```
Ctrl+C
```

```
[INFO] 감지 중단 (Ctrl+C)
[INFO] keep-awake OFF (원복)
[INFO] 감지 종료
```

그리고 점검 모드를 한 번 보여준다:

```
SAFE_MODE=1 uv run python poc/workflow_3/monitor/align_fail_monitor.py
```

```
[INFO] 점검 모드: SAFE_MODE=1, ALIGN_FAIL_CORRECTION_DRY_RUN=0 (셸 env 가 기본값을 덮었습니다)
```

성과 보고에서 "실수로 장비를 건드리면?" 질문에 대한 답이 이 컷이다.
배너 문구가 바뀌는 것을 확대해서 보여줄 것.

---

## 5. 촬영 수단

**주 화면**: PowerPoint `삽입 > 화면 녹화` → 나레이션 동시 녹음 → `미디어로 저장`(mp4).
오피스 PC 에 추가 설치가 필요 없는 가장 확실한 경로.
Xbox Game Bar(Win+G)는 단일 창만 잡고 사내 정책으로 막혀 있을 수 있어 비추천.

**보조 (핵심)**: 컷 ④의 보정 순간은 루프가 **이미 저장해 둔** `recording/` JPEG
프레임으로 별도 클립을 만든다. 외부 녹화보다 화질이 좋고, 무엇보다 **실알람 때 아무도
안 보고 있어도 이미 쌓여 있다**(§0.5).

```
# 가장 최근 recording 폴더를 자동으로 골라 실시간 배속 mp4 생성
uv run python poc/workflow_3/monitor/make_demo_video.py

# 폴더를 지정하고, 보정 순간만 잘라 2배속으로
DEMO_VIDEO_INPUT_DIR=<recording 경로> \
DEMO_VIDEO_START_SEC=120 DEMO_VIDEO_END_SEC=180 DEMO_VIDEO_SPEED=2 \
  uv run python poc/workflow_3/monitor/make_demo_video.py
```

이 스크립트가 하는 일:

- 파일명(`..._{seq}_{elapsed_ms}ms.jpg`)의 경과시간으로 **실제 시간축을 복원**한다.
  파일 순서대로 고정 fps 로 붙이면 빠른 조작 구간은 늘어지고 정지 구간은 순식간에
  지나가 실제와 정반대인 영상이 되는데, 그걸 피한다.
- **정지 구간을 압축**한다(`DEMO_VIDEO_MAX_HOLD_SEC`, 기본 2s). align fail 은 장비가
  멈춰 있어 수십 초씩 정지 화면이 이어지는데, 잘라낸 시간은 화면에
  `>> +38s skipped` 로 표시되므로 몰래 편집한 영상이 되지 않는다.
- 경과시간/프레임번호를 burn-in 한다(`DEMO_VIDEO_OVERLAY=0` 으로 끌 수 있음).
  **ASCII 만 렌더링된다** — 한글 자막은 편집 도구에서 얹을 것.

주요 env 는 스크립트 docstring 참고 (`SPEED`/`FPS`/`MAX_HOLD_SEC`/`START_SEC`/
`END_SEC`/`MAX_WIDTH`/`OVERLAY`/`LABEL`).

### 로그 패널 - 프레임에 없는 콘솔을 되살린다

녹화 프레임은 **tool 창 rect** 만 담으므로 터미널이 절대 찍히지 않는다(§0.5).
`DEMO_VIDEO_LOG_PANEL=1` 을 켜면 프레임 오른쪽에 그 시각의 로그를 합성한다.

```
DEMO_VIDEO_INPUT_DIR=<recording 경로> \
DEMO_VIDEO_LOG_PANEL=1 \
DEMO_VIDEO_RUN_DIR=poc/workflow_3/logs/workflow_runs/<run_id>_align_fail_cycle_<EQP> \
  uv run python poc/workflow_3/monitor/make_demo_video.py
```

시각 정합은 `recording_manifest.json` 의 `started_at` + 파일명의 `elapsed_ms` 로 잡는다.
녹화 시작 **30초 전**까지 포함하므로, 프레임에 원리상 없는 `connect_tool`(더블클릭)
구간도 패널에는 나온다.

읽는 소스 3종:

| 소스 | env | 성격 |
|---|---|---|
| 감사 로그 `logs/work2.log` | `DEMO_VIDEO_LOG_FILE` | 기본값. **소급 적용 가능** - 이미 녹화된 알람에도 쓸 수 있다 |
| 실행 저널 `step_*.json` | `DEMO_VIDEO_RUN_DIR` | `STEP connect_tool [success]` - 데모에서 가장 읽기 좋은 줄 |
| 콘솔 tee 파일 | `DEMO_VIDEO_CONSOLE_LOG` | 진짜 stdout 전사. 촬영 전에 걸어둬야 한다 |

> **감사 로그는 stdout 전사가 아니다.** 의미 있는 이벤트(step 결과, 보정 outcome,
> 알림)는 다 있지만 콘솔의 모든 `[INFO]` 줄이 있지는 않다. 콘솔 원문 그대로가
> 필요하면 촬영 시 **줄마다 시각을 붙여** tee 한다(시각이 없으면 정합 불가):
>
> ```powershell
> uv run python poc/workflow_3/monitor/align_fail_monitor.py 2>&1 |
>   ForEach-Object { "{0:HH:mm:ss} {1}" -f (Get-Date), $_ } |
>   Tee-Object -FilePath console.log
> ```

한글은 PIL+TrueType 으로 그리므로 정상 출력된다(좌하단 burn-in 은 cv2 라 ASCII 만 -
두 경로가 다르다). 폰트 자동 탐색이 실패하면 `DEMO_VIDEO_FONT` 로 지정한다.

패널을 못 붙이는 조건은 **조용히 넘어가지 않고 사유를 남긴다**: manifest 의
`started_at` 이 없거나, 녹화 구간에 걸치는 로그 줄이 0이면 경고 후 패널 없이 영상만
만든다(시각이 어긋난 로그를 붙이면 없느니만 못하기 때문).

---

## 6. 촬영이 어긋났을 때

| 증상 | 원인 | 대응 |
|---|---|---|
| `점유 cooldown 중` / `view_only_observation` | 다른 사람이 tool 점유 | 점유자에게 해제 요청 후 재촬영 |
| `wrong_tool_opened` | List 행 오클릭 | cooldown 후 자동 재시도 — 그대로 두고 다음 사이클을 담아도 된다 |
| `no_assets` | rcp align 이미지 없음 | `ALIGN_IMAGES_DIR` 확인. 촬영 중단하고 경로부터 고칠 것 |
| `panel_not_found` | SEM box 검출 실패 | 화면 배율/창 크기 변경 후 재시도 |
| 신규 없음만 반복 | fixture `UTC9` 가 윈도우 밖 | `UTC9` 를 현재 시각으로 갱신 후 재실행 |

> **replay row 는 첫 poll 에만 나온다.** 그 뒤 poll 부터는 비어서 알람 해제 경로가
> 자동으로 돈다(`[INFO] Align Fail 해제: EQP_ID=...`). 재촬영하려면 모니터를 재실행할 것.

---

## 7. 실패 컷도 유효한 자료다

이 시스템의 설계 의도는 "항상 성공"이 아니라 **실패하면 cube 로 인계하고 그 장면까지
녹화하는 것**이다. 보정이 fallback 으로 끝나 엔지니어가 수동 조작하는 장면이 담기면,
그것도 설계대로 동작한 증거다. 굳이 성공 컷만 고집하다 촬영 기회를 놓치지 말 것.
