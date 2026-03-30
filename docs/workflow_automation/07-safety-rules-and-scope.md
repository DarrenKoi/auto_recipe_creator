# Safety Rules + v1 범위

## Safety Rules

- 검증 실패의 기본 동작은 **중단(HALT)**이지, 예산 초과 재시도가 아닙니다
- `idempotent=False` step은 act phase 이후 검증 실패 시 **즉시 HALT**합니다 (자동 재실행 금지)
- jitter 좌표는 doc 04의 safe zone **및** element bounding box 범위 내에 있어야 합니다
- 모델 fallback은 safety tier 검사를 우회하지 않습니다
- 사람 에스컬레이션이 최종 fallback이며, 무한 재시도는 없습니다
- 모든 재시도는 **전체 증거 trail**을 남깁니다 (before/after 스크린샷, VLM 응답, 좌표)
- Tier 3 step은 액션 **전과 후** 모두 검증이 필요합니다
- `SAFE_MODE` 토글은 워크플로 레벨에서도 존중됩니다
- password / credential step 의 artifact 와 로그는 기본적으로 redact 합니다
- unstable / occluded frame 은 "모델 실패"가 아니라 "입력 화면 품질 실패"로 취급합니다
- 매 step 전 **foreground 윈도우 검사**를 수행하여 예상치 못한 윈도우를 감지합니다
- 알려진 interrupt(에러 팝업, 시스템 알림)는 자동 처리하되, `safety_tier <= 1` 동작만 허용합니다

## v1 범위

**v1에 포함:**
- `WorkflowStep` / `StepCondition` / `ConditionType` / `StepResult` / `WorkflowRun` dataclass
- `ConditionChecker` 기반 타입 안전 조건 평가
- 순차 실행기 (dependency 체크 포함)
- precondition / success_condition / skip_condition 기반 step 계약
- phase별 타임아웃 (detect / act / verify 독립 관리)
- foreground 윈도우 검사 + 알려진 interrupt 자동 처리
- poll-until-stable 안정화 대기 (pHash 기반)
- hybrid 후행 검증 (window title / UIA / OCR 우선, 필요한 곳만 VLM)
- failure-aware 재시도: recapture, jitter (bbox 경계 포함), model fallback
- non-idempotent step HALT 처리 (act 이후 자동 재실행 차단)
- per-step + per-workflow 재시도 예산 (reserved_retry_budget 포함)
- per-step 결과 JSON 로깅
- RCS 로그인 워크플로 구현

**v1에 미포함:**
- Checkpoint / Resume (상태 저장 포맷만 준비)
- 실패 이력 기반 서비스 자동 선택
- 병렬 step 실행
- VLM 기반 동적 워크플로 생성
- doc 06 align-fail 모니터링 통합
- doc 07 occlusion recovery 통합
- password artifact masking 자동화 고도화
