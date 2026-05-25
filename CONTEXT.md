# CONTEXT

CD-SEM/VeritySEM 레시피 자동 셋업 시스템의 도메인 용어집(glossary).
구현 세부는 담지 않는다 — 용어의 **의미**만 정의한다.

## 용어

### Align Fail
CD-SEM 장비가 레시피에 등록된 align key 와 현재 wafer 화면을 정렬하지 못해 발생하는 알람.
알람 코드 **ALID=9006** 으로 식별한다. workflow_2 파이프라인의 트리거.

### align key
레시피에 등록된, 정렬 기준이 되는 화면 패턴(주변 layout 포함). OM 버전과 SEM 버전이 따로 있다.

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

### recipe_om / recipe_sem / current_sem
한 align fail 이벤트에 대해 내려받는 세 이미지의 표준 이름.
- **recipe_om** — 레시피 등록 OM align key
- **recipe_sem** — 레시피 등록 SEM align key
- **current_sem** — fail 시점의 live SEM 모니터 이미지

### OM / SEM (mode)
SEM Monitor 의 두 관찰 모드. Optical(OM) vs 전자빔(SEM). template routing 이 mode 별로 갈린다.
