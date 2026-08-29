# CONTEXT

CD-SEM/VeritySEM 레시피 자동 셋업 시스템의 도메인 용어집(glossary).
구현 세부는 담지 않는다 — 용어의 **의미**만 정의한다.

## 용어

### Align Fail
CD-SEM 장비가 레시피에 등록된 align key 와 현재 wafer 화면을 정렬하지 못해 발생하는 알람.
알람 코드 **ALID=9006** 으로 식별한다. workflow_2 파이프라인의 트리거.

### Recovery Episode
하나의 Align Fail 발생에서 시작해 정렬 복구 성공 또는 엔지니어 escalation 으로 끝나는 한 번의
복구 사건.

### Recovery Trace
하나의 Recovery Episode 동안 관측된 화면 상태와 엔지니어 행동의 시간순 기록. 실행 절차가
아니라 Recovery Playbook 을 만드는 근거다.

### Recovery Playbook
여러 Recovery Trace 를 통합해 만든 조건부 복구 모델. 관측 가능한 사전 상태, 행동,
검증 결과, 실패 시 대안 경로의 관계를 담는다.

### Recovery Guard
Recovery Playbook 에서 특정 Recovery Action 을 선택할 수 있게 하는 관측 가능한 사전
상태. 판정값은 `true`, `false`, `unknown` 이며, 관측 불가는 `false` 가 아니라 `unknown`
이다. 엔지니어의 의도나 관측되지 않은 이유는 Recovery Guard 가 아니다.

### Recovery Action
Recovery Playbook 이 선택하는 장비 독립적 행동 의미. 화면 좌표나 녹화에서 읽은 라벨은
행동 자체가 아니라 그 행동을 뒷받침하는 근거다.

### Recovery Verification
Recovery Action 후 기대한 관측 가능한 상태 변화가 실제로 일어났는지 확인하는 판정.
행동을 시도했다는 기록이나 GUI 클릭 자체는 Verification 이 아니며, 관측할 수 없으면
실패로 단정하지 않고 `unknown` 이다.

### Recovery Outcome
Recovery Episode 의 종료 결과. `recovered`, `escalated`, `aborted`, `unknown` 중 하나이며,
`recovered` 는 측정 재개나 품질 신호처럼 관측 가능한 성공 근거가 있을 때만 성립한다.

### Recovery Annotation
행동 수행자가 Recovery Trace 의 특정 근거(frame 또는 event 범위)를 가리키며 남기는 판독·분류·
설명 기록. 종류는 관측 Guard 값의 판독/정정, Recovery Action 의 도메인 의미 분류, Verification
판독, rationale 넷뿐이다. append-only 이며 정정은 이전 기록을 supersedes 하는 새 기록이다. 근거를
가리키지 않는 annotation 은 관측값을 바꾸지 못하고 rationale 로만 남는다. 관측되지 않은 Guard 나
counterfactual 성공을 만들 수 없다.

### 검토 묶음 (Review Packet)
Recovery Episode 하나와 그 Episode 를 근거로 삼는 Recovery Playbook rule 하나에 대해, 엔지니어가
검토해야 할 것만 모은 파생 view. Guard 판정, 정규화 step 과 대표 전·후 frame, Verification,
Outcome, 분기 이유, 적용 범위, 시스템이 묻는 열린 질문을 담는다. 저장 정본이 아니라
Episode·Playbook·Recovery Annotation 에서 매번 다시 만든다.

### 행동 수행자 (Recovery Actor)
Recovery Episode 에서 실제로 Recovery Action 을 한 엔지니어. 검토 묶음의 질문에 답하고 Recovery
Annotation 을 남길 수 있는 유일한 역할.

### Playbook 승인자 (Playbook Approver)
candidate Recovery Playbook 의 특정 버전을 승인·반려하거나 근거를 요청하는 역할. 관측이나 행동
의미를 대신 판독하지 않으며, 승인 기록은 Recovery Annotation 과 분리해 Playbook 버전에 고정된다.

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

### align_img_from_rcp (folder)
한 align fail 이벤트 폴더(`align_images/<eqp_id>/<class>/<recipe>/`) 아래 서브폴더. 오피스 MES 가
생성하는 **레시피 등록 align key** 이미지: `IMAP0001.*`(OM), `IMAP0002.*`(SEM). matcher 가
비교할 **등록 기준(reference) template** 의 출처. 엔지니어가 그린 박스(burned-in)가 여기 있다.

### align_img_from_msr (folder)
같은 이벤트 폴더 아래 서브폴더. 오피스 MES 가 생성하는 **측정 궤적** 이미지 시퀀스(`S*`=정상
step, `E*`=fail step). **align fail 이 발생한 순간의 이미지는 여기서 확인할 수 있다**(최신 `E*`).
`align_fail_assets` 의 `current_sem` 은 이 폴더의 최신 `E*` 다.

### live SEM 의존 (sequential align)
`align_img_from_msr` 의 fail 이미지는 *그 순간 한 장*일 뿐이다. align fail 복구는 한 위치만이
아니라 **여러 align 위치를 순차적으로 다시 잡아줘야 할 수 있어**, 정지된 파일이 아니라 진행 중인
**live SEM 화면에 의존**하며 단계마다 매칭·reposition 을 반복한다. 그래서 workflow_2 는 정적
비교(Step 3)만으로 끝나지 않고 live 보정(Step 4 PRIMARY) / live 탐색(Step 5~8 FALLBACK)을 둔다.

> 정본(canonical) 자산 모델은 위 `align_img_from_rcp` / `align_img_from_msr` (IMAP) 레이아웃이다
> (`align_fail_assets.py`, `workflow_2/__init__.py`, CLAUDE.md 와 일치). 과거의 고정 stem 3개 파일
> 모델, 그리고 ADR 0001 의 `align_recipe`/`current_sem` 폴더 + 라이브 캡처 모델은 **폐기**되었다.

### OM / SEM (mode)
SEM Monitor 의 두 관찰 모드. Optical(OM) vs 전자빔(SEM). template routing 이 mode 별로 갈린다.
