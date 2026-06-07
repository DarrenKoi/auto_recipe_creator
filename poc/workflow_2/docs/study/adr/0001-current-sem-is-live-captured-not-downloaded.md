---
status: superseded
---

# fail 시점 current SEM 이미지는 다운로드가 아니라 workflow_2 가 라이브 캡처한다

> **SUPERSEDED (2026-05-27):** 본 ADR 의 `align_recipe`/`current_sem` 폴더 + `align_fail_downloads`
> 경로 + "라이브 캡처" 모델은 폐기되었다. 정본 자산 모델은 `align_images/<eqp>/<class>/<recipe>/`
> 아래 **`align_img_from_rcp`**(IMAP0001=OM, IMAP0002=SEM) / **`align_img_from_msr`**(S*/E*) 이며,
> align fail 순간 이미지는 `align_img_from_msr` 최신 `E*` 에서 확인한다(라이브 캡처가 아님). 단,
> 여러 align 위치를 순차로 잡아야 하므로 *보정 과정*은 여전히 live SEM 에 의존한다. CONTEXT.md 및
> `align_fail_assets.py` 가 정본. 아래 본문은 히스토리로만 보존.

## 결정

Align fail 이벤트 자산은 이벤트별 폴더 `align_fail_downloads/<YYYYMMDD>/<eqp_id>/<class_name>/<recipe_name>/`
아래 두 서브폴더로 나눈다.

- `align_recipe/` — **다운로드**되는 등록 align key step 이미지 시퀀스(번호 순, 예: `XX_001_XX.jpeg`).
  align 은 OM 단계를 먼저, SEM 단계를 나중에 수행하므로 `OM, OM, SEM, SEM` 처럼 순차 저장되고
  fail 단계에서 멈춘다. matcher 가 비교할 **등록 기준(reference) template** 의 출처.
- `current_sem/` — **다운로드가 아니라**, align fail 로 장비가 멈춘 뒤 workflow_2 가 SEM Monitor 를
  **라이브 캡처해 저장**하는 출력 폴더.

## 맥락 / 이유

장비는 fail 시점의 live 화면을 **파일로 남기지 않는다.** align fail 로 구동이 멈춘 뒤부터는
SEM Monitor 의 실시간 캡처에 의존할 수밖에 없다. 따라서 "현재(실패) 이미지"는 다운로드 자산이 될 수
없으며, workflow_2 의 `rcs_sem_controller.capture()` 가 직접 만들어내는 산출물이다.

## 결과 (Consequences)

- `align_fail_assets.py` 는 "고정 stem 3개 파일(`recipe_om`/`recipe_sem`/`current_sem`)" 모델을 버리고
  **두 서브폴더(각각 이미지 시퀀스)** 를 해석하는 구조로 바뀐다.
- 다운로드 핸들러(과거 workflow_1, 현재 workflow_2 로 통째 이관)는 `align_recipe/` 만 채운다.
  `current_sem/` 의 생성·기록 책임은 라이브 탐색 쪽(`live_align_search` / `rcs_sem_controller`)에 있다.
- 미해결: `align_recipe/` 안에서 OM/SEM 을 구분하는 규칙(파일명 token vs 순서·개수)은 오피스 실제
  파일명 확인 후 확정한다. 그 전까지 분류기는 교체 가능한 형태로 둔다.
