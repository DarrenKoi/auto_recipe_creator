# RCS List 탭 UI 구조 레퍼런스

> 이 폴더는 workflow_2 의 `study/sem_box/`(SEM 화면 영역의 도메인 시각 구조 레퍼런스)에 **대응**한다.
> workflow_1 의 도메인 UI 구조는 **RCS List 탭** 이므로 여기 정리한다.
> 대상 코드: `view_list_tab_rcs.py`, `workflow_select_tool.py`, `tool_name_match.py`

---

## 1. 왜 레이아웃을 문서화하나

Tool 을 찾으려면 "화면 어디에 Tool ID 가 있는지"를 알아야 crop·검색 영역을 좁힐 수 있다. RCS List
탭은 좌측에 장비 목록, 우측에 부가 컬럼이 놓인 표 형태이므로 **검색은 좌측에 집중** 한다.

---

## 2. List 탭 그리드 구조

```
┌──────────────────────────────────────────────────────────────┐
│ [List] 탭                                                       │
├──┬───────────┬──────┬─────────┬───────┬────────┬─────┬────────┤
│●│ MC ID      │ RCS  │ Location│ Model │ Status │ ... │ User   │
│  │ (장비 ID)  │ IP   │         │       │        │     │        │
├──┼───────────┼──────┼─────────┼───────┼────────┼─────┼────────┤
│●│ MCD916     │ ...  │ ...     │ ...   │ ...    │ ... │ ...    │  ← 더블클릭 대상 행
│●│ MCD917     │ ...                                              │
│  └ 신호등                                                       │
└──┴───────────┴────────────────────────────────────────────────┘
   ▲
   신호등(녹=On / 검=Off), ID 텍스트 왼쪽의 작은 사각형
```

- **최좌측 컬럼 = MC ID(장비 ID)** — 세로로 나열. 우리가 매칭할 대상.
- **ID 텍스트 왼쪽의 작은 사각형 = 신호등** — **녹색=Tool On, 검정=Tool Off.**
- **우측 컬럼들(무시)**: RCS IP, Location, Model, Status, Count, DVR, Connection User.

(프로젝트 메모리: project_rcs_tool_list_layout)

---

## 3. 검색 영역 crop 비율

`workflow_select_tool.py` 는 좌측 패널만 잘라 VLM/OCR 입력으로 쓴다:

```python
LIST_REGION_LEFT_RATIO   = 0.00   # 좌측 끝
LIST_REGION_TOP_RATIO    = 0.10   # 타이틀바 스킵
LIST_REGION_RIGHT_RATIO  = 0.42   # ~폭의 40%(좌측 패널)
LIST_REGION_BOTTOM_RATIO = 0.98   # 거의 전체 높이
```

> 우측 컬럼까지 포함하면 반복 텍스트·잡음이 늘어 OCR 환각·오매칭도 늘어난다. **좌측 집중** 이 핵심.

---

## 4. 검색 흐름 (레이아웃이 결정하는 것)

1. **1차 VLM**: 좌측 crop 에서 ui-venus(coarse)→mai-ui(fine)로 Tool 행 클릭점. 최대
   `COARSE_FINE_MAX_ITERS=2` 회.
2. **fallback OCR**: 실패 시 좌측 crop 에 OCR → 라인 추출 → `tool_name_match` 정규화 매칭.
   - **모호 시 포기**: 같은 정규형이 여러 행이면 자동선택 중단(`_distinct_row_count>1`).
3. **스크롤**: 못 찾으면 List 영역 안에서 아래로 스크롤한다. 픽셀 변화
   `mean_diff > LIST_CHANGE_THRESHOLD=2.0` 로 목록 변경을 감지하고, 최대 `MAX_SCROLL_ITERS=8` 회까지 반복한다. 변화가
   멈추면(스크롤바 끝) 종료한다.
4. **더블클릭**: 찾은 점을 full image→screen(DPI 보정) 후 `click_count=2`.

---

## 5. 점유 상태 — select 팝업

다른 사용자가 Tool 을 쓰는 중이면 더블클릭 시 **"select"(공유/종료) 팝업** 이 뜬다. 이 팝업은 무시하고,
Tool 창 탐색을 `RCS_WINDOW_MAX_TRIALS=10` 회로 제한한 뒤 다음 알람 대기로 넘어간다.
(프로젝트 메모리: project_rcs_occupied_select_popup)

---

## 6. 신호등 활용 가능성 (참고)

신호등 색(녹/검)은 Tool On/Off 를 나타내는 **시각 신호** 다. 현재는 ID 텍스트 매칭이 1차지만, 향후 "꺼진
Tool 은 후보에서 제외" 같은 사전 필터에서 신호등 색을 보조 신호로 쓸 여지가 있다.

---

## 7. 핵심 상수 한눈에

| 상수 | 값 | 의미 |
|---|---|---|
| `LIST_REGION_*_RATIO` | 0.00 / 0.10 / 0.42 / 0.98 | 좌측 패널 crop |
| `COARSE_FINE_MAX_ITERS` | 2 | VLM 좌표 재시도 |
| `LIST_CHANGE_THRESHOLD` | 2.0 | 스크롤 변화 감지 |
| `MAX_SCROLL_ITERS` | 8 | 스크롤 상한 |
| `RCS_WINDOW_MAX_TRIALS` | 10 | Tool 창 탐색 상한(점유 대비) |
