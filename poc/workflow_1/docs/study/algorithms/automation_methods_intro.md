# Workflow 1 자동화 기법 입문 — VLM 좌표 · DPI 보정 · ID 정규화 · 알람 폴링

> 대상: `poc/workflow_1/` 의 RCS GUI 자동화에 쓰이는 핵심 기법들
> 목적: (1) 본인 학습용, (2) 상사 질문 대응용 설명 자료
> 작성일: 2026-06-07

---

## 0. 먼저 — 왜 "그냥 좌표 클릭"으로는 안 되는가

가장 먼저 받을 질문: **"버튼 위치는 정해져 있을 텐데, 그 좌표를 그냥 클릭하면 되지 않나?"**

안 됩니다. 이유:

- **RCS 는 legacy GUI 라 내부 컨트롤이 표준 방식으로 노출되지 않습니다.** pywinauto 의 UIA/win32
  backend 로 ComboBox·Button 을 찾으려 해도 컨트롤이 트리에 잡히지 않습니다. (그래서 "그 버튼의
  좌표를 프로그램에 물어보는" 방법 자체가 불가능합니다.)
- **창 위치·크기·DPI 배율이 PC마다 다릅니다.** 오피스 PC 는 125%/150% 배율이 흔한데, 하드코딩한
  픽셀 좌표는 다른 배율에서 빗나갑니다.
- **레이아웃이 상황에 따라 바뀝니다.** 팝업, 점유 상태(select 팝업), 스크롤 위치에 따라 같은 요소가
  다른 자리에 나타납니다.

그래서 workflow_1 은 **사람이 화면을 보고 누르는 과정을 그대로 흉내** 냅니다.

```
화면을 본다       → 스크린샷 캡처 (mss)
어디인지 안다     → VLM 이 좌표를 찾는다 (ui-venus → mai-ui)
맞는지 확인한다   → OCR 로 그 자리 텍스트를 읽어 검증 (PaddleOCR, 확인 전용)
실제로 누른다     → 화면 좌표로 변환(DPI 보정) 후 클릭/입력 (pynput)
```

핵심 원칙(프로젝트 전역): **좌표는 VLM(coarse→fine)이 제안하고, OCR 은 확인만 한다. 미확인 시 클릭하지 않는다.**
(workflow_2 의 "좌표는 CV 가 결정, VLM 은 식별만" 원칙과 짝을 이루는, workflow_1 판 안전 규칙입니다.)

아래 다섯 기법이 그 도구입니다. 한 줄 요약:

```
2단계 VLM 로케이터 : 전체화면 bbox(coarse) → crop 확대 후 정밀 클릭점(fine)  → 좌표 찾기
DPI 좌표 변환       : 이미지 픽셀 → 실제 화면 픽셀 (rect/screenshot 비율 보정)  → 좌표 옮기기
OCR 확인(Spotting)  : 찾은 자리의 텍스트를 읽어 "맞는 요소인지" 검증           → 안전장치
Tool ID 정규화      : O↔0, I↔1 같은 OCR 혼동을 정규화해 정확 매칭             → ID 매칭
알람 폴링 루프      : ALID=9006 을 주기적으로 보고, 새 알람만 한 번 처리        → 트리거
```

---

## 1. 2단계 VLM 로케이터 — coarse → fine → confirm

### 한 문장
전체 스크린샷에서 **대략적 박스(coarse)** 를 먼저 잡고, 그 주변만 **잘라서 확대(crop+zoom)** 한 뒤
**정밀한 클릭점(fine)** 을 다시 찾는, 2단계 좌표 탐색.

### 직관
사람도 "로그인 버튼 어디 있지?" → (대충 아래쪽) → 거기를 자세히 본 뒤 → 버튼 한가운데를 누릅니다.
전체 화면에서 픽셀 단위 정밀 좌표를 한 번에 맞추라고 하면 큰 모델도 흔들립니다. 그래서 **범위를 좁혀
가며** 정밀도를 올립니다.

작동 순서:

1. **Stage 1 (coarse, `ui-venus`)**: 전체 스크린샷 + "User ID 입력칸을 찾아라" → 0–1000 정규화 bbox.
2. **crop**: 그 bbox 둘레에 패딩을 줘 잘라낸다 (`left_pad_ratio=1.25` 등 요소별 설정).
3. **zoom**: 잘린 조각을 최소 960×320 이 되도록 LANCZOS 로 확대 (최대 `MAX_UPSCALE=4.0`).
4. **Stage 2 (fine, `mai-ui`)**: 확대된 조각만 보고 "이 안에서 클릭할 한 점" → 0–1000 좌표.
5. **역변환**: fine 좌표를 crop 좌표로, 다시 full image 좌표로 되돌린다.
6. **confirm(선택)**: 그 자리 텍스트를 OCR 로 읽어 기대 라벨과 맞는지 확인. 안 맞으면 클릭 안 함.

### 비유
지도에서 도시를 먼저 찾고(coarse), 그 도시를 확대해 골목 주소를 짚는 것(fine).

### 장점 / 단점
| 장점 | 단점 |
|---|---|
| 전체화면 정밀 좌표보다 훨씬 안정적 | VLM 호출이 2번 (지연·비용 ↑) |
| 작은 요소도 확대 후라 잘 잡힘 | coarse 가 완전히 빗나가면 fine 도 실패 |
| 모델별 강점 분담 (식별 vs 정밀) | crop 패딩이 부족하면 대상이 잘릴 수 있음 |

### 우리 프로젝트에서
로그인 5개 요소(Server/UserID/Password/Login/Cancel), List 탭, Tool 행 클릭 등 **모든 GUI 클릭의 기본기**.
자세한 건 → `two_stage_vlm_locator.md`.

---

## 2. DPI 좌표 변환 — 이미지 픽셀을 실제 화면 픽셀로

### 한 문장
스크린샷은 **물리 픽셀**(예: 150% 배율이면 2880px)로 찍히는데, pywinauto 가 보고하는 창 사각형은
**논리 픽셀**(1920px)이라, 그 비율로 좌표를 보정해야 클릭이 빗나가지 않는다.

### 직관
같은 창이라도 "사진 속 좌표"와 "마우스가 가야 할 좌표"는 축척이 다릅니다. VLM 은 사진(이미지) 위의
좌표를 주지만, pynput 마우스는 화면 좌표를 필요로 합니다. 그래서 둘 사이의 **scale = 창 rect 크기 / 캡처 이미지 크기**
를 곱해 변환합니다.

```python
scale_x = rect_w / img_w
screen_x = rect.left + image_point["x"] * scale_x   # y 도 동일
```

### 왜 중요한가 (상사 질문 포인트)
이 보정을 빼먹으면 **150% 배율 PC 에서 클릭이 일관되게 우하단으로 빗나갑니다.** "왜 우리 PC 에서만
안 되지?"의 90% 는 코드 문제가 아니라 바로 이 배율 문제입니다. 그래서 import 시점에 **DPI awareness
를 먼저 선언**(`_enable_dpi_awareness()` in `__init__.py`)해, 캡처와 rect 가 같은 픽셀 기준을 쓰도록 맞춥니다.

### 우리 프로젝트에서
`util/window_utils.py` 의 `image_point_to_screen()`. 모든 클릭 직전에 통과하는 좌표 변환기.
자세한 건 → `dpi_coordinate_mapping.md`.

---

## 3. OCR 확인 (Spotting) — 누르기 전에 "맞는 자리인지" 검증

### 한 문장
VLM 이 찾은 좌표 위/주변의 **텍스트를 OCR 로 읽어**, 기대한 라벨과 일치할 때만 클릭하는 안전장치.

### 직관
VLM 이 가끔 비슷하게 생긴 다른 요소를 가리킬 수 있습니다. 누르기 전에 "거기 진짜 'Log In' 이라고
쓰여 있나?"를 한 번 더 확인하면 오클릭을 막을 수 있습니다. **OCR 은 결정하지 않고 확인만** 합니다.

`Spotting:` 태스크는 평문 OCR(`OCR:`)과 달리 **검출 텍스트마다 bbox 좌표를 같이** 줍니다. 그 결과를
`ocr_spotting.parse_spotting_items()` 가 다양한 형식(4원소 배열/dict/폴리곤/wrapper)을 견디며
`{"text", "bbox"}` 리스트로 정규화합니다.

### 왜 중요한가
**"미확인 시 클릭 금지"** 가 규칙입니다. OCR 이 못 읽거나 엉뚱한 텍스트를 주면, 클릭하지 않고
debug 아티팩트만 남긴 뒤 재시도하거나 에스컬레이션합니다. 자동화가 틀린 곳을 누르는 것이 가장 위험하기 때문입니다.

### 우리 프로젝트에서
전체 RCS 화면에 OCR 을 직접 돌리면 환각이 납니다(프로젝트 메모리에 기록됨). 그래서 **VLM 으로 영역을
좁힌 뒤 그 crop 에만** OCR 을 씁니다. 자세한 건 → `../cv/ocr_spotting_intro.md`, `../paddleOCR/README.md`.

---

## 4. Tool ID 정규화 — OCR 혼동 글자를 정규화해 정확 매칭

### 한 문장
OCR/VLM 이 자주 헷갈리는 글자쌍(O↔0, I↔1, B↔8…)을 **하나의 표준 글자로 치환**한 뒤, fuzzy 매칭이
아니라 **정확(exact) 토큰 매칭** 으로 Tool ID 를 찾는다.

### 직관
Tool ID 는 `MCD916` 처럼 **고정 길이 코드** 입니다. 여기에 edit-distance(편집거리) 같은 fuzzy 매칭을
쓰면 한 글자 차이로 엉뚱한 ID 와 매칭됩니다. 대신 "어차피 OCR 이 O 를 0 으로 잘못 읽는다"는 점을
역이용해, **양쪽 모두 0 으로 정규화** 한 뒤 정확히 같은지만 비교합니다.

```python
_CONFUSION_MAP = {"O":"0","Q":"0", "I":"1","L":"1", "B":"8", "S":"5","Z":"2","G":"6"}
# "Tool_O1b" → 대문자/영숫자화 "TOOLO1B" → 치환 "TOOL018"
```

보수적으로, 의미 있는 접두사를 망가뜨리는 치환(D→0, T→7, A→4)은 **안 합니다.**

### 왜 중요한가 (상사 질문 포인트)
정확 매칭이라도 **같은 정규형이 여러 행에 걸치면 모호**해집니다. 이때는 자체 매칭을 포기하고 VLM
grounding 으로 넘깁니다(`_distinct_row_count > 1` → None). 같은 행 안에 후보가 여럿이면 **가장 작은
bbox**(가장 타이트한 검출)를 채택합니다. 틀린 행을 무턱대고 더블클릭하는 일을 막는 장치입니다.

### 우리 프로젝트에서
`tool_name_match.py` 의 `canonicalize()` / `best_match()`. List 탭에서 OCR fallback 으로 Tool 을
찾을 때 사용. 자세한 건 → `tool_name_canonicalization.md`.

---

## 5. 알람 폴링 루프 — ALID=9006 을 보고, 새 알람만 한 번 처리

### 한 문장
일정 주기로 알람 목록을 읽어 **Align Fail(ALID=9006)** 을 거르고, **이미 처리한 장비는 다시 안
울리도록(edge-triggered)** 중복을 막는 루프.

### 직관
알람은 해제될 때까지 계속 목록에 남아 있습니다. 폴링할 때마다 알림을 쏘면 60초마다 같은 팝업이
반복됩니다. 그래서 **처음 나타난 순간** 딱 한 번만 처리합니다. 집합(set) 차집합으로 구현합니다:

```python
new_tools     = current_tools - active_tools   # 이번에 새로 뜬 것만 처리
cleared_tools = active_tools - current_tools   # 사라진 것은 active 에서 제거(=재발 시 다시 처리 가능)
```

폴링 주기(`POLL_INTERVAL_SEC=10s`)와 **탐지 윈도우(`DETECTION_WINDOW_SEC=60s`)** 를 분리해, 보고
지연이 있어도 최근 60초 내 알람을 놓치지 않습니다(UTC9 시각 컬럼 기준 필터).

### 탐지 시 동작
1. 텍스트 로그 기록 (`logs/align_fail_alarms.txt`)
2. Windows MessageBox 팝업 (`MessageBoxTimeoutW`, 60초 후 자동 닫힘)
3. **RECIPE_ID 가 있을 때만** Tool 자동 접속 1회(느슨하게) — 없으면 엔지니어가 직접
4. (record 변형) Tool 접속 → 팝업 닫기 → 화면 캡처 → Tool 창 닫기

### 우리 프로젝트에서
`align_fail_alarm.py` / `align_fail_alarm_record.py`. workflow_1 의 **트리거이자 workflow_2 로의
핸드오프 시작점.** 자세한 건 → `alarm_polling_loop.md`.

---

## 6. 전체 파이프라인 — 어떻게 한 줄로 이어지나

```
 (상시) 알람 폴링 루프  ── ALID=9006 감지 ──┐
                                            ▼
 RCS 로그인  →  List 탭 진입  →  Tool 검색·더블클릭  →  Tool 화면 캡처  →  Tool 창 닫기
   │              │                │                      │
   └─ 2단계 VLM ──┴── 2단계 VLM ───┤                      └─ mss 캡처 → JPEG 저장
      (+OCR 확인)    (+OCR fallback,│                         → align_images/.../captured_img_from_rcs/
                      Tool ID 정규화)│
                                    └─ 모든 클릭은 image_point_to_screen() 으로 DPI 보정 후 pynput
                                            ▼
                                    captured_img_from_rcs/  ──► [workflow_2 가 읽음]
```

**역할 분담 요약:**

| 단계 | 기법 | 역할 |
|---|---|---|
| 트리거 | 알람 폴링 루프 | ALID=9006 을 edge-triggered 로 한 번만 처리 |
| 좌표 찾기 | 2단계 VLM 로케이터 | coarse bbox → fine 클릭점 |
| 검증 | OCR Spotting | 그 자리 텍스트 확인 (확인 전용) |
| ID 매칭 | Tool ID 정규화 | OCR 혼동 보정 + 정확 매칭 |
| 실행 | DPI 좌표 변환 + pynput | 이미지 좌표 → 화면 좌표 → 클릭/입력 |

---

## 7. 상사 예상 질문 & 답변

**Q. pywinauto 로 컨트롤을 직접 조작하면 더 안정적이지 않나?**
A. RCS 는 legacy 라 ComboBox/Button 이 UIA/win32 트리에 잡히지 않습니다. pywinauto 는 **창 띄우기·창
제목 탐색·foreground** 에만 쓰고, 내부 클릭은 스크린샷→VLM 좌표→pynput 경로를 탑니다. (ADR
`0001` 참고.)

**Q. VLM 이 틀리면 어떻게 하나?**
A. (1) coarse→fine 으로 정밀도를 높이고, (2) OCR 로 클릭 전 확인하고, (3) **미확인이면 클릭하지
않습니다.** Tool ID 는 정규화 후 정확 매칭하고, 모호하면(여러 행) 자동 선택을 포기합니다.

**Q. 왜 모델을 2개(ui-venus, mai-ui) 쓰나?**
A. ui-venus 는 전체화면에서 "어느 영역인지" 식별하는 데 강하고, mai-ui 는 확대된 crop 에서 "정확한
점"을 잡는 데 강합니다. 싸고 거친 식별로 범위를 좁힌 뒤, 정밀 좌표는 확대한 다음에 뽑는 구조입니다.

**Q. 다른 PC 에서 클릭이 빗나가는 건?**
A. 거의 DPI 배율(125/150%) 문제입니다. import 시 DPI awareness 를 선언하고, 클릭 직전
`image_point_to_screen()` 으로 rect/screenshot 비율 보정을 합니다. (`dpi_coordinate_mapping.md`)

**Q. 같은 알람이 계속 울리지 않나?**
A. edge-triggered dedup 으로 **처음 뜬 순간만** 처리합니다. 해제되면 active 집합에서 빠지고, 재발
시 다시 한 번 처리됩니다.

---

## 8. 더 읽을거리 (이 저장소 내부)

- 전체 절차: `../runbooks/workflow_1_procedure.md`
- 모듈 실행법: `../runbooks/module_run_guide.md`
- 2단계 로케이터 deep-dive: `two_stage_vlm_locator.md`
- DPI 보정 deep-dive: `dpi_coordinate_mapping.md`
- Tool ID 정규화 deep-dive: `tool_name_canonicalization.md`
- 알람 루프 deep-dive: `alarm_polling_loop.md`
- OCR/영상 처리: `../cv/ocr_spotting_intro.md`, `../cv/cursor_detection.md`, `../cv/image_capture_pipeline.md`
- 설계 결정: `../adr/`
