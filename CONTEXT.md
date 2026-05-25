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

### OM / SEM (mode)
SEM Monitor 의 두 관찰 모드. Optical(OM) vs 전자빔(SEM). template routing 이 mode 별로 갈린다.
