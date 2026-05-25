# Handoff — Workflow 2 Align Key 탐색 (2026-05-25)

다음 에이전트가 이어받을 수 있도록 이번 세션의 맥락·결정·남은 일을 정리한다.
세부 절차/파일/상태표는 중복하지 않는다 → **`poc/workflow_2/docs/workflow_2_procedure.md`** 참조(이번 세션에서 작성한 권위 문서).

## 이번 세션에서 한 일

1. `/grill-me` 로 workflow_2 의 7단계 계획을 끝까지 인터뷰해 설계를 확정.
2. 확정된 설계대로 코드 작성·검증(아래 "검증 상태").
3. `docs/` 정리: 기술 status 중복본 `sem_monitor_control_implementation_map.html` 을
   `workflow_2_procedure.md` 로 통합·삭제. 매니저 슬라이드(`generate_status_report.py`+html+pptx)와
   알고리즘 해설 HTML(chamfer/orb/hamming/filter)은 성격이 달라 유지.

## 확정된 핵심 설계 결정 (grill 결과)

- 좌표는 **CV 가 결정**, VLM 은 영역 식별·feasibility 평가에 한정.
- Align fail = live key 가 등록 이미지와 **다르게 보이는** 경우 → 구조(Chamfer) 위주, hard match 강요 안 함, **best candidate 를 엔지니어에 보고**.
- 다운로드 자산: `align_fail_downloads/<recipe_id>/{recipe_om,recipe_sem,current_sem}.*` (OM·SEM·현재 SEM, 메타 없음).
- 물리 규약: **더블클릭=클릭점 recenter(배율 불변), wheel=FOV중심 discrete 배율**, mode(OM/SEM) 별 template routing.
- Budget: **pan 만 10회 카운트**, recenter/zoom 은 별도 상한.
- Step 1·2 는 **throwaway VLM probe**(루프에 안 들어감) — "VLM 이 miniature/drift key 를 식별 가능한가"를 평가하기 위함.

## 검증 상태 (이번 세션, Mac/win32에서 직접 실행)

- `align_key_matcher.py` 확장(MatchPolicy/STRUCTURE_POLICY/BROAD_SCALES/scales override) — smoke test **10/10 유지**.
- `compare_align_images.py` (Step 3) — 합성 self-test **통과**(drift key localize, score 0.864).
- `live_align_search.py` (Step 4~7) — 배율 mock 에서 two-phase **통과**(scale 1.0, ORB 0.942 에서 확정 match).
- `vlm_align_key_box.py` (Step 1·2) — import OK, 자산/Flask 없을 때 graceful. **VLM 호출은 오피스 전용(미검증)**.
- `align_fail_assets.py` — 신규 공용 로더.

## 발견한 중요한 한계 (반드시 인지)

저배율 miniature 단계의 **chamfer 단독 변별력이 낮다**(축소 template 은 배경에서도 고득점, ORB 무력).
broad 는 *후보 제안*일 뿐, 진짜 판정은 confirm 단계(zoom-in 후 scale~1.0 + ORB)로 미룬다.
terminal match 가드 `best_scale≥0.6 AND orb>0` 가 거짓 종료를 막는다.
→ 이 한계가 Step 1·2 VLM probe 의 존재 이유. probe 결과가 좋으면 broad 단계에 VLM `roi_hint` 도입 검토.

## 다음에 할 일 (우선순위 순, 상세는 procedure.md §4.3)

1. (오피스, 최우선) workflow_1 다운로드 핸들러 — 자산을 designated path 에 저장, **파일명 규약 확정**.
2. (오피스, 최우선) 실장비 adapter — 신규 `poc/workflow_2/rcs_sem_controller.py` 로
   `SEMMonitorController`(capture/move_to_point/zoom/read_mode) 구현.
3. (최우선) Safety gate — SAFE_MODE 기본 on, ROI 밖 클릭 차단, 이동/zoom 상한, emergency stop.
4. (높음) 실데이터 calibration — `STRUCTURE_POLICY` 임계값·`MIN_CONFIRM_SCALE`·`candidate_score` 보정(현재 cold-start).
5. (중간) Engineer escalation 3-pane 산출물 — `live_align_search.notify_fn` 구현.
6. (probe 후 결정) VLM 보조 broad spotting — 옵션 플래그(기본 off), VLM box→roi_hint.

## 미해결/확인 필요

- 오피스 실제 다운로드 **파일명**이 `__init__.py` 의 STEM 상수와 일치하는지.
- `read_mode()` 를 무엇으로 구현할지(VLM box 의 mode_label vs OCR).
- 직전에 사용자에게 던진 열린 질문: "probe 출력을 live loop 의 roi_hint 로 지금 연결(플래그 off)할지, 오피스 probe 후로 미룰지" — **미결정**.

## 다음 세션 추천 스킬

- `superpowers:brainstorming` — 실장비 adapter/safety gate 설계를 시작하기 전.
- `superpowers:test-driven-development` — calibration·adapter 구현 시.
- `feature-dev:code-explorer` — workflow_1 다운로드 경로/RCS 캡처 기존 코드 추적.
