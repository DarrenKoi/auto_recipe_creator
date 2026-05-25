# Handoff — Workflow 2 자산 레이아웃·다운로드 경로 재설계 (2026-05-26)

이번 세션은 `/grill-with-docs` 로 "workflow_1 의 align_fail 코드를 workflow_2 로 가져오기 +
다운로드 경로에 날짜 삽입" 제안을 인터뷰하며, 기존 자산 모델과의 모순을 드러내고 재설계를 확정했다.
직전 핸드오프(`2026-05-25-workflow2-align-search.md`)의 **미해결 항목 중 다운로드 경로/파일명**을
이번에 갱신했다 → 그 문서의 §"미해결" 첫 두 줄은 이 문서로 대체됨.

## 권위 산출물 (중복 금지, 경로 참조)

- **`CONTEXT.md`** (repo 루트) — 도메인 용어집. `align_recipe`/`current_sem` 의미가 두-폴더 모델로 정정됨.
- **`poc/workflow_2/docs/adr/0001-current-sem-is-live-captured-not-downloaded.md`** — 핵심 결정 ADR.
- 절차/상태표: `poc/workflow_2/docs/workflow_2_procedure.md` (여전히 권위. 단 §2 의 자산 모델은 아래로 갱신 필요).

## 이번 세션에서 확정된 설계 결정

1. **마이그레이션 범위** — workflow_1 의 detection+download 를 **통째로 workflow_2 로 이관**.
   workflow_2 가 알람 감지→다운로드까지 소유. workflow_1 사본은 제거/동결.
   (대상: `align_fail_alarm.py`, `monitor_align_fail.py`, office 의존 모듈 `office_align_fail_alarm`/`office_rich_notify`.)
2. **다운로드 경로** — `align_fail_downloads/<YYYYMMDD-로컬>/<eqp_id>/<class_name>/<recipe_name>/`.
   - `RECIPE_ID == "{class_name}/{recipe_name}"` 이므로 디스크상 `download_root/date/eqp_id/recipe_id` 로 자연 중첩.
   - `eqp_id`(=EQP_ID), `recipe_id` 는 알람 row 에서 직접 옴. **세그먼트는 Windows 금지문자 sanitize 필요.**
   - 같은 (date,tool,class,recipe) 폴더는 **덮어쓰기**. retention 은 **수동**(날짜 폴더 삭제), 자동 prune 없음.
   - date 는 **다운로드 시점 로컬 날짜**(`datetime.now()`), 알람 UTC9 아님.
3. **이벤트 폴더 하위 2개 서브폴더** (생산 주체가 다름):
   - `align_recipe/` — **다운로드**되는 등록 align key step 이미지 **시퀀스**(`XX_001_XX.jpeg`...). OM 단계 먼저, SEM 나중(`OM,OM,SEM,SEM`), fail 단계에서 멈춤. **matcher 의 reference template 출처.**
   - `current_sem/` — **다운로드 아님.** workflow_2 가 fail 후 SEM Monitor 를 **라이브 캡처해 저장**하는 출력 폴더.
     (장비가 fail 시점 live 이미지를 파일로 남기지 않음 → ADR 0001.)
4. **ROI 전략** (`rcs_sem_controller.capture()`) — **1회 locate → rect 캐시 → 재크롭 → K프레임마다 landmark confidence 재검증**, 창 이동/신뢰도 하락 시에만 relocate. (대안 "매 프레임 relocate"는 jitter/비용으로 기각.)

## 다음에 할 코드 작업 (office-confirm 게이트 있음)

1. `align_fail_assets.py` 재작성 — 고정 stem 3개 → **2개 서브폴더(각 시퀀스) 해석**, **교체 가능한 OM/SEM 분류기**.
   현재 코드는 `resolve_assets(recipe_id)` + `recipe_om/recipe_sem/current_sem` 고정 stem (구 모델).
2. `poc/workflow_2/__init__.py` 경로 상수 갱신 — `ALIGN_FAIL_DOWNLOAD_DIR` 및 `<date>/<eqp_id>/<class>/<recipe>/` 스킴.
   구 `RECIPE_OM_STEM/RECIPE_SEM_STEM/CURRENT_SEM_STEM` 폐기.
3. ROI 안정성 테스트 — `test_match_on_captured_frames.py` 확장: static-window 녹화 시퀀스에서 캐시 ROI 가 고정 유지되고 재검증 confidence > `LANDMARK_CONF_MIN` 임을 assert.
4. (직전 핸드오프와 동일) workflow_2 로 이관된 download 핸들러 + `rcs_sem_controller.py` 실장비 adapter + safety gate.

## 미해결 (오피스 실제 파일명 확인 필요)

- `align_recipe/` 실제 파일명 패턴과 **OM/SEM 구분 규칙**(파일명 token vs 순서·개수) — **미정**. 분류기를 교체 가능하게 둘 것.
- `current_sem/` 라이브 캡처 파일 **네이밍 규약** — 미정(workflow_2 가 정함).
- matcher 가 **OM 단계 reference 도** 쓰는지(OM-mode 매칭), **SEM-only** 인지 — 미정.

## 다음 세션 추천 스킬

- `superpowers:test-driven-development` — `align_fail_assets.py` 재작성·ROI 안정성 테스트 시(분류기·경로 해석은 합성 fixture 로 TDD 가능).
- `feature-dev:code-explorer` — workflow_1 download/office 모듈 의존성을 이관 전 추적.
- `superpowers:brainstorming` — `rcs_sem_controller`/safety gate 설계 착수 전.

## 운영 메모

- 솔로 프로젝트: 커밋 후 `git push origin main` 직접 push (PR/브랜치 불필요). auto-commit 훅이 파일 쓰기마다 커밋함.
