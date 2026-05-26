# CONTEXT

CD-SEM/VeritySEM 레시피 자동 셋업 시스템의 도메인 용어집(glossary).
구현 세부는 담지 않는다 — 용어의 **의미**만 정의한다.

## 용어

### Align Fail
CD-SEM 장비가 레시피에 등록된 align key 와 현재 wafer 화면을 정렬하지 못해 발생하는 알람.
알람 코드 **ALID=9006** 으로 식별한다. workflow_2 파이프라인의 트리거.

### align key
레시피에 등록된, 정렬 기준이 되는 화면 패턴(주변 layout 포함). OM 버전과 SEM 버전이 따로 있다.
등록 이미지(`align_img_from_rcp`)에는 엔지니어가 유니크한 위치를 **박스로 그려** 두며(이미지에
burned-in), 그 **박스 안의 모양**이 정렬 기준이다. live SEM 에도 그 모양이 보여야 align 성공.

### target point (= recipe-box center)
엔지니어가 그린 박스의 중심(보통 이미지 정중앙). crosshair 가 가야 할 정답 위치. matcher 의
`best_xy` 가 live 프레임에서 이 점을 추정한다. live 는 픽셀 동일이 아니라 contrast/brightness/
shape 가 조금씩 다르므로 edge 구조(Chamfer)로 매칭한다.

### live crosshair (가로/세로 십자선)
Align Fail 로 멈춘 paused SEM Monitor 화면이 그리는, *현재(=잘못된)* align 시도 위치 표시.
보정에서 옮겨야 할 대상이며, 위치는 매번 달라진다.

### reposition gesture (double-click recenter)
clicked point 를 화면 중심으로 만들고 crosshair 를 그 점에 놓는 확정 제스처. 코드의
`SEMMonitorController.move_to_point()` 와 같은 의미. (single click 은 crosshair 만 옮기고,
double-click 은 recenter — 결국 둘 다 같은 best_xy 를 가리킴.)

### OK confirm
reposition 후 dialog 의 OK(확인) 버튼을 single-click(screen 좌표, SEM ROI 밖)해 align 을
진행시키는 동작. 버튼 위치는 VLM 으로 찾는다(`vlm_ok_button_box`).

### key visibility gate (키 가시성 게이트)
paused 프레임에서 recipe key 가 인식되면 PRIMARY(즉시 reposition+OK), 아니면 FALLBACK(pan/zoom
탐색)으로 가르는 단일 분기 기준(`align_fail_correct.key_visibility_gate`).

### primary vs fallback path
PRIMARY: key 가 잘못된 crosshair 근처에 이미 보임 → 즉시 reposition+OK. FALLBACK
(`live_align_search`): 아무것도 안 보일 때만 SEM Monitor 를 pan/zoom 하며 key 를 헌트.

### tool_id (= EQP_ID)
CD-SEM 장비 한 대의 식별자. 알람 row 의 `EQP_ID` 필드. 같은 recipe 라도 장비별로 자산을
구분하기 위해 다운로드 경로에 포함한다.

### RECIPE_ID
알람 row 가 제공하는 레시피 식별자. 값 자체가 `"{class_name}/{recipe_name}"` 형태로,
슬래시로 class_name 과 recipe_name 두 부분으로 분해된다.

### class_name
레시피가 속한 분류. `RECIPE_ID` 의 슬래시 앞부분.

### recipe_name
레시피 고유 이름. `RECIPE_ID` 의 슬래시 뒷부분.

### align_recipe (folder)
한 align fail 이벤트 폴더 아래 서브폴더. **다운로드된** 등록 align key step 이미지들의
**시퀀스**(번호 순, 예: `XX_001_XX.jpeg`)를 담는다. align 은 OM 단계를 먼저, SEM 단계를
나중에 수행하므로 이미지가 `OM, OM, SEM, SEM` 처럼 순차 저장되고 fail 단계에서 멈춘다.
matcher 가 비교할 **등록 기준(reference) template** 의 출처. 한 시퀀스 안에서 OM/SEM 을
구분하는 규칙(파일명 token vs 순서/개수)은 **오피스 실제 파일명 확인 후 확정**(미정).

### current_sem (folder)
한 align fail 이벤트 폴더 아래 서브폴더. **다운로드가 아니라** workflow_2 가 align fail 로
장비가 멈춘 뒤 SEM Monitor 를 **라이브 캡처해 저장**하는 출력 폴더. fail 시점의 live 이미지는
장비가 파일로 남기지 않으므로 workflow_2 가 직접 캡처한다.

> 이전 모델의 고정 stem `recipe_om`/`recipe_sem`/`current_sem` (각 1개 파일)은 폐기되었다.
> 실제는 위 두 **폴더**(각각 이미지 시퀀스) 구조다.

> ⚠️ **드리프트 주의(2026-05-27 확인, 미해결):** 위 `align_recipe`/`current_sem` 폴더 모델과
> `align_fail_downloads` 경로(ADR 0001)는 **현재 코드·CLAUDE.md 와 어긋난다.** 현 구현은
> `align_images/<eqp>/<class>/<recipe>/` 아래 **`align_img_from_rcp`**(IMAP0001=OM, IMAP0002=SEM)
> + **`align_img_from_msr`**(S*/E*, E=fail) 레이아웃을 쓰고(`align_fail_assets.py`,
> `workflow_2/__init__.py`), `current_sem` 도 라이브 캡처가 아니라 `from_msr` 최신 E* 에서 읽는다.
> 즉 ADR 0001 의 "라이브 캡처 + align_recipe/current_sem 폴더" 결정은 코드에 미반영. 어느 쪽을
> 정본으로 할지(폴더 모델 복원 vs IMAP 모델 추인 + ADR 갱신)는 오피스 실파일 확인 후 사용자가 결정.

### OM / SEM (mode)
SEM Monitor 의 두 관찰 모드. Optical(OM) vs 전자빔(SEM). template routing 이 mode 별로 갈린다.
