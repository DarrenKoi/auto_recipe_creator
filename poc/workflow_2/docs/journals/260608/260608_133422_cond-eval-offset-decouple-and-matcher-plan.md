# 260608 세션 저널 — cond GT eval 오프셋 분리 · matcher 다음 단계 계획

## 0. 한 줄 요약
`golden_localization_eval_cond.py` 를 cond.txt 기하 기반으로 재설계(offset 을 crop 과 분리, box-stroke-only 청소, measure-first 게이트)하고, /codex:rescue 자문 + /code-review 로 검증·수정한 뒤 matcher 성능 개선의 다음 단계를 "1회 오피스 실행으로 결정"하도록 만들었다.

---

## 1. 진행 사항

### 1-1. green box vs yellow box 발산 원인 규명
- 사용자 관찰: `rcp_templates/` 오버레이에서 **초록 box(template)** 가 **노란 box(cond white box)** 와 크기가 크게 다른 경우가 절반가량.
- 원인 추적: 노란 box = `_cond_box_to_xywh(cond.box_ltrb)`(정확), 초록 box = `_inner_crop_for_box(gray, det)` 의 **내용 검출 기반 축소** 결과.
  - `_inner_hole_bbox`(ring 안쪽 hole) → 실패 시 고정 inset 3px 폴백
  - `_trim_bright_border`(밝은 픽셀 사라질 때까지 한 줄씩 깎기, 하한 8px)
  - `RCP_BOX_SAFETY_PX`(1px)
- 결론: cond.txt 로 box 위치를 **정확히** 아는데도 검출형 축소가 돌아가, box 내부에 밝은 구조가 있으면 초록 box 가 과도 축소·off-center → 그 off-center 가 **offset(= image_center − inner_center)을 오염**(절반 miss 의 한 원인).

### 1-2. /codex:rescue 자문 (fresh thread)
- Codex 결론 4건 채택:
  1. inpaint 후 **대칭 2~3px inset**(edge-smear 차단).
  2. 검출형 trim 은 box 가 정확하면 무가치+위험(crop 중심이 내용 의존).
  3. **offset 을 crop 과 분리** — `align_offset = image_center − box_center` 를 cond.txt 만으로.
  4. 가드: 내부 <16px skip, offset_norm(÷대각선) >0.25 warn / ~0.38 skip, 경계/유효성 체크.
- "우리가 놓친 점" = #3(offset 결합). 원본 `golden_localization_eval_cond.py:111` 이 inner-crop 중심에서 offset 을 뽑던 것 → 검출 오류가 offset 을 오염.

### 1-3. cond eval 재설계 (TDD)
- 새 순수 헬퍼: `cond_align_offset`, `cond_offset_norm`(÷대각선), `check_cond_box`(ok/warn/skip), `cond_template_crop`(box-stroke inpaint + 대칭 inset).
- `_build_offset_templates_cond` 의 box 분기를 검출(`_inner_crop_for_box`) → cond 기하로 교체. offset 은 `cond_align_offset` 로 분리.
- 검증: 합성 E2E 에서 offset (96,0)=image_center−box_center, crop 중심=box 중심, stroke 제거 확인.

### 1-4. matcher 성능 개선 — 다음 레버 조사
- Explore 에이전트로 "예전에 계획한 ensemble" 전수 조사.
  - 계획했던 것 = **reranker**(MI / contour) → 둘 다 실데이터에서 음수 lift 로 **이미 폐기**(2026-06-02). 이유: 진실이 후보(top-N)에 없으면 재정렬 무력. proposer recall(gt_in_topk) 천장 **0.594**.
  - 검증된 레버 = consensus 재등록(in_topk +0.282).
- 핵심 통찰: 그 숫자들은 모두 **흰box/crosshair 제거 전** 측정 → cond 정제 후엔 proposer membership 이 올랐을 가능성. 그래서 **measure-first**(먼저 gt_in_topk 재측정)가 옳은 다음 행동.
- 사용자 선택(AskUserQuestion): **"Measure first, then decide"**.

### 1-5. measure-first 게이트 구현 (TDD)
- `lever_verdict(box__inpaint 셀)` → `PROPOSER_WALL | RERANKER_ALIVE | NEAR_CEILING | no_data` 한 줄 판정.
- `run()` 말미에 "이 블록만 읽어주면 됨" 출력(타이핑 부담 제거 — 사용자 요청 반영).

### 1-6. /code-review (high effort, 7 angle finders + 검증)
- 3개 finder 에이전트(line scan / removed-behavior / cross-file tracer). tracer 가 핵심(offset 분리, 가드 순서, None-deref, overlay 포맷)은 **SAFE** 로 적대적 확인.
- 발견 5건 중 실수정 3건 채택(아래 2장).

---

## 2. 수정 내용

### 변경 파일
- `poc/workflow_2/golden_localization_eval_cond.py` (대폭 재설계 + 리뷰 수정)
- `poc/workflow_2/test_golden_localization_eval_cond.py` (신규 → 19 테스트)
- `poc/workflow_2/docs/journals/260608/cond_sample.txt` (사용자 작성, 실 cond 샘플)

### 커밋 (main 직접 push, Mac→office pull)
1. `decouple align-offset from crop + self-interpreting lever verdict`
2. `code-review fixes — preserve rcp crosshair, fix offset-norm scale, persist summary early`

### code-review 수정 3건
- **#1 (MEDIUM-HIGH) rcp template crosshair 보존**: `cond_template_crop` 이 `clean_image(gray, cond)` 로 box + crosshair 를 둘 다 inpaint → rcp cond 에 crosshair 있으면 box 내부 **실제 내용**을 지워 매칭 신호(lever 가 읽는 box__inpaint 셀) 손상. → `crosshair_xy=None` box-only `CondInfo` 로 **box 테두리만** 제거. (msr 프레임 crosshair 제거는 별개 — distractor 라 지움)
- **#2 (MEDIUM) offset_norm 척도 불일치**: `cond_offset_norm`(÷대각선) 값을 `gle._offset_diag`(GT_TOL_NORM 0.20 short-side)에 넣어 '가정민감' ~4.6× 과소계상. → 대각선 전용 `_offset_diag_cond`(tol=OFFSET_WARN 0.25) 신설.
- **#3 (LOW-MED) summary.json 유실 위험**: write 가 프린트 뒤로 이동 → 프린트 예외 시 산출물 통째 유실. → `lever_verdict` 계산 후 **출력 전에** 먼저 write.

### 설계 상수
- `CROP_INSET_PX=2`, `MIN_INNER_PX=16`, `WARN_INNER_PX=24`, `OFFSET_WARN=0.25`, `OFFSET_SKIP=0.38`
- `OLD_PROPOSER_CEILING=0.594`, `PROPOSER_WALL=0.62`, `RERANKER_MIN_HEADROOM=0.08`

### 테스트
- 19/19 통과. (offset/norm/guard/crop/verdict + crosshair 보존 + offset_diag 척도/empty)
- 형제 스위트(`test_clean_align_image.py`, `test_cond_file.py`) 16/16 무영향.
- 실행 환경 주의: cv2 가 기본 인터프리터에 없어 **`uv run --extra dev pytest ...`** 로 실행.

---

## 3. 다음 단계

### 즉시 (오피스에서 사용자 실행)
```
uv run python poc/workflow_2/golden_localization_eval_cond.py
```
- 맨 아래 **`>>> 판정:`** 단어 + **`gt_in_topk`** 숫자만 보고하면 다음 matcher 경로 결정:
  - **PROPOSER_WALL** → ensemble *proposer*(Chamfer+NCC/region+edge-orient 후보 합집합) + consensus 재등록 (rerank 은 여전히 무력)
  - **RERANKER_ALIVE** → 정제로 reranker 레버가 살아남 → ensemble reranker 구축
  - **NEAR_CEILING** → 남은 미스는 proposer membership 몫 → proposer/재등록 집중

### 후속 (열린 항목)
- **1024-px OVERSAMPLE 비율 검증**: 현재 OVERSAMPLE=10 은 512-px 만 확인. 1024-px cond 샘플이 오피스에 생기면 ÷10 비율 재확인 필요(틀리면 box 가 좌상단 1/2 스케일로 들어가 out_of_bounds 걸러지지 않고 조용히 오작동 가능).
- (확인 필요) cond.txt 에 rcp **align/measure point** 를 직접 담는 필드가 있는가? 있으면 "rcp align point = 이미지 중심" 가정을 아예 제거 가능. → **사용자 확인 완료: rcp align point = 이미지 중심 (가정 아님, 확정)**.

---

## 4. 메모리 업데이트
- **업데이트함**: `memory/project_align_cond_files_and_coords.md` 에 cond eval 재설계 블록 추가
  - offset 을 crop 과 분리(image_center − box_center)
  - rcp template 청소는 **box-only**(crosshair 보존), msr 는 crosshair 제거
  - offset_norm 대각선 정규화 + `_offset_diag_cond`(척도 섞지 말 것)
  - measure-first `lever_verdict` 게이트 + 다음 단계 분기(rerank 폐기, proposer 가 벽이면 ensemble proposer)
- MEMORY.md 인덱스: 기존 항목 그대로(해당 메모리 파일 내용만 갱신, 신규 파일 없음).
