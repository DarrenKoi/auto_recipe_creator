# 00. 임원 요약 (Executive Summary)

## 1. 프로젝트 목적

CD-SEM / VeritySEM 계측 장비의 **recipe setup을 사람이 수동으로 만드는 과정을 AI로 자동화**합니다.
특히 계측 중 빈번하게 발생하는 **Align Fail(정렬 실패, ALID=9006)** 을 무인으로 감지·보정하여
엔지니어 개입과 장비 idle 시간을 줄이는 것이 1차 목표입니다.

**왜 어려운가 (문제 정의):**

- RCS(Recipe Control System)는 legacy GUI 클라이언트(`RcsMainHD.exe`)로, accessibility tree가
  부실하여 일반적인 UI 자동화(pywinauto/UIA)로 안정적으로 제어하기 어렵습니다.
- Align Fail의 핵심은 "현재 SEM 화면에서 recipe에 등록된 **align key**가 어디 있는가"를 찾아
  그 좌표로 재정렬하는 것인데, 화면은 contrast·밝기·패턴이 매 측정마다 변하고 반복 패턴이 많아
  단순 좌표 인식이 통하지 않습니다.
- 수동 대응은 야간·주말 무인 운영이 불가능하고, 실패 원인 분석용 화면 증거도 사후에 사라집니다.

## 2. PoC 핵심 방향 — VLM과 CV의 역할 분리

이 프로젝트의 일관된 설계 원칙(2026-05-25 확정)은 다음과 같습니다.

> **OpenCV(CV)는 정량 점수와 최종 좌표를 만듭니다. VLM은 영역을 식별하고, 모호한 화면을 설명하고,
> 보정 가능성(feasibility)을 판단합니다. 낮은 CV 점수를 VLM 답변이 덮어쓰거나, 반복 가능한 단계 전환을
> VLM이 결정하게 하지 않습니다.**

| 구분 | 담당 | 이유 |
|------|------|------|
| **VLM** (Vision Language Model) | "어디를 볼까" — UI 요소 위치, align key 후보 영역, OM/SEM 모드 판독, 모호성 설명 | 화면 의미 이해에 강함. 단, stateless·픽셀 정확도 한계 |
| **CV** (OpenCV) | "얼마나 닮았나 / 정확히 어디" — template matching 점수, 최종 클릭 좌표 | 반복 가능·정량적. stateless VLM의 "기억"을 외부 상태로 대신 |

이 분리 덕분에 화면 이해(VLM)와 정밀 좌표(CV)를 각자의 강점에 맡겨, 무인 자동화의 신뢰도를 확보했습니다.

## 3. 단계별 진행 (Workstream Timeline)

| 단계 | 워크스트림 | 한 일 | 상태 |
|------|-----------|-------|------|
| 0 | **deploy_vlms** | 오픈소스 VLM 5종 조사·선정 → 사내 HCP(H200 140GB×2)에 vLLM으로 설치, Flask proxy로 통합 운영 | 운영 중 |
| 1 | **workflow_1** | RCS GUI 자동화(2-stage VLM 로케이터) + Align Fail 감지 + CCTV/DVR 화면 자동 캡처 PoC | 동결(frozen), 검증 완료 |
| 2 | **workflow_2** | 오프라인 CV 평가 벤치 — golden set으로 matching/ensemble/consensus 정확도 A/B·튜닝 | 활성(검증 harness) |
| 3 | **workflow_3** | 위를 통합한 **production 실시간 align-fail 모니터링 루프** (현재 주력) | 구현 완료, 오피스 활성화 진행 중 |

설계 흐름은 **VLM 인프라 확보(0) → GUI 자동화 가능성 증명(1) → CV 정확도 확보(2) → 실시간 통합 루프(3)** 입니다.

## 4. 핵심 성과 (Highlights)

- **VLM 운영 효율**: GUI grounding + OCR에 필요한 능력을 특화 모델로 분담하여, **운영 4종 + 대안 경로 1종(UI-TARS,
  현재 비활성)** 구성을 **H200 2장**으로 운영합니다. 단일 프런티어 모델(Kimi-K2 계열, 1.04T MoE)은 동일 작업에
  H200 약 8장이 필요하므로 → **하드웨어 약 4배**, **가중치 풋프린트·GPU당 모델 밀도 약 20배**(처리량 효율이 아닌
  메모리·밀도 기준)의 절감입니다. (근거: `docs/setup_vlms/05`)
- **GUI 자동화 가능성 증명(workflow_1)**: coarse(UI-Venus) → fine(MAI-UI) → OCR 검증의 다중 모델
  파이프라인으로, UIA 없이도 RCS 화면 좌표를 안정적으로 클릭 가능함을 입증했습니다.
- **정렬 정확도 도약(workflow_2)**: 등록 align key(rcp) 단독 대비, **최근 성공(S) 이미지의 consensus
  template** 라우팅으로 align key 탐색 recall(in_topk)이 **0.434 → 0.876**, rank1이 **0.318 → 0.764**
  로 향상되었습니다. **단, 이 수치는 현재 오프라인 golden set 벤치(min_s=3) 기준이며, 오피스 실데이터
  현장 검증은 진행 예정**입니다(production 기본값은 보수적으로 min_s=4 운영). (근거: `poc/workflow_3/README.md`, workflow_2 bench)
- **통합 루프 구현(workflow_3)**: 알람 감지 → RCS 접속 → CV 보정 → 실패 시 cube 알림 →
  상시 녹화 → 자동 종료의 end-to-end 루프와, 4-layer 모듈 아키텍처를 구축했습니다.

## 5. 기대 효과 (Business Impact)

> **[작성 필요 — 아래 항목을 오피스 실측치/추정치로 채워 주시기 바랍니다. 추정치는 "추정"으로 명시.]**
> 임원 보고의 핵심 질문은 "그래서 무엇을 절감하는가"입니다. 현재 본 보고서는 정성적 효과만 기술되어 있어,
> 아래 정량 항목 중 확보 가능한 것을 채우면 설득력이 크게 높아집니다.

| 항목 | 값 | 비고 |
|------|-----|------|
| 대상 장비 수 (CD-SEM/VeritySEM) | `[N대]` | pilot → 전사 확대 기준 |
| 하루 평균 Align Fail 발생 건수 | `[N건/일]` | ALID=9006 알람 로그 기준 |
| 1건당 수동 대응 시간 (감지~보정) | `[N분]` | 엔지니어 인터뷰/로그 기준 |
| 야간·주말 무인화로 회수되는 대응 지연 | `[N시간/주]` | 현재는 익일 오전까지 idle |
| 연간 절감 환산 (장비 idle + 인시) | `[추정 N]` | 위 값들의 단순 환산, "추정" 명시 |

- 정량 수치 확보 전이라도 **정성적 효과는 분명**합니다: ① 야간/주말 무인 모니터링·보정, ② 실패 시점
  화면 증거 100% 자동 보존(원인 분석 시간 단축), ③ 수집 데이터의 학습 자산화.

## 6. 확장성 (Scalability)

- **모델 추가 용이**: Flask proxy 서비스 레지스트리에 `.env` + 모듈 1개 등록으로 6번째 모델 추가가 가능합니다
  (GPU 1에 ~50 GiB 헤드룸 보유).
- **장비·recipe 일반화**: consensus 이력 풀은 `<class>/<recipe>` 키(장비 무관)로 설계되어, 같은 recipe면
  여러 장비가 학습 데이터를 공유합니다. 신규 장비 onboarding 시 코드 변경이 최소화됩니다.
- **데이터 자산화**: 상시 녹화가 엔지니어의 수동 보정 조작까지 프레임으로 보존하므로,
  향후 모방학습/원인 자동 분류용 학습 데이터로 재활용됩니다(recording_filter로 interaction timeline 추출).
- **벤치→프로덕션 파이프라인**: workflow_2에서 CV 변경을 golden set으로 검증한 뒤
  bit-parity로 workflow_3에 포팅하여, 회귀 위험을 통제한 채 정확도 개선을 반복합니다.

## 7. 현재 상태 한눈에

| 영역 | 상태 |
|------|------|
| VLM 배포·운영 (deploy_vlms + Flask proxy) | ✅ 운영 중 |
| workflow_1 GUI 자동화 + CCTV PoC | ✅ 검증 완료(동결) |
| workflow_2 CV 평가 벤치 | ✅ 활성 (golden 데이터로 오피스 실측 대기) |
| workflow_3 루프 골격·보정·녹화·알림 | ✅ 코드 완료 |
| workflow_3 consensus 라이브 보정 활성화 | 🟡 `office_success_downloader` 구현 대기 |
| 오피스 PC 캘리브레이션(zoom/click 좌표, engineer-done) | 🟡 진행 중 |
| pilot 실보정(actuation) | 🟡 dry-run 후 단일 장비부터 |

상세 현황·로드맵·일정은 [05_status_roadmap.md](05_status_roadmap.md)를 참조하시기 바랍니다.
