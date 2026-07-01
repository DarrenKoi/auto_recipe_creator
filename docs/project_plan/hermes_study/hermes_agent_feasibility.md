# Hermes Agent 도입 타당성 검토 — workflow_3 구현 시점에서

> 상태: **검토 완료, 권고 = 도입 보류(프로덕션 루프에는 미채택)**. 2026-07-01 요청으로 작성.
> 성격: 외부 도구(Hermes Agent) 사실 정리 + workflow_3 정합성 분석 + 판단 근거. 아래 "권고"는 근거 기반 의견이며, 최종 채택은 사용자 결정.
> 관련: `CLAUDE.md`(workflow_3 설계 원칙), `poc/workflow_3/README.md`, `poc/workflow_3/vlm/flask_vlm.py`(기존 모델 게이트웨이).
> 출처: 문서 하단 참조.
>
> **[2026-07-01 정정 — 오피스 제약]** Hermes는 **오피스에서 사용 예정**이고 **오피스에는 Claude 모델/Claude Code가
> 없다**(로컬 GLM 5.2·Kimi-K2.6만 가용). 따라서 아래 §4의 "이미 Claude Code가 커버함"은 **Mac 개발 장비 기준으로만
> 참이고 오피스 기준으로는 거짓**이다. 이 제약을 반영한 Claude Code↔Hermes 정면 비교는 **§7**에 추가했다.
> 요약: **프로덕션 제어 루프 미채택 결론은 그대로**, 다만 **오피스-사이드 비안전 보조 도구**로서 Hermes는
> "오피스에서 Claude Code가 못 채우는 자리"를 채운다(로컬 모델 위 상시 에이전트).

---

## 0. 한 줄 요약

Hermes Agent는 **런타임에 스스로 skill을 만들고 고치는 "자기개선형 범용 비서 프레임워크"**다.
workflow_3는 **fab 계측장비(CD-SEM/VeritySEM)를 실시간 구동하는 결정론적·감사가능·이중 안전게이트
제어 루프**다. 두 시스템의 근본 성격이 정반대라, Hermes의 대표 강점(자기수정·메모리·범용성)이
workflow_3에서는 이득이 아니라 리스크로 작동한다. 게다가 **Hermes가 해결하는 문제(에이전트 오케스트레이션·
세션 메모리·범용 모델 라우팅)는 workflow_3의 실제 병목이 아니다.** workflow_3의 병목은 CV 변별력
(SEM aperture/junction), 물리 calibration(더블클릭·휠↔배율), 오피스 데이터 수집(consensus pool)이며,
이건 에이전트 프레임워크로 풀리지 않는다.

**결론: workflow_3 구현 시점에 Hermes를 프로덕션 루프에 얹지 않는다.** "더 효율적·더 빠름"의 근거가 없고,
오히려 통합 비용과 안전성 훼손 위험만 늘린다. 자기개선형 에이전트가 궁금하면 프로덕션과 **완전히 분리된
오프라인 실험**(벤치 튜닝·지식 검색)으로만 좁혀서 다루되, 그마저도 지금 스택(Claude Code + 자체 루프)이
이미 커버한다.

---

## 1. Hermes Agent란 무엇인가 (사실)

Nous Research가 2026년 2월 공개한 오픈소스(MIT) **자기개선형 AI 에이전트**. Python 3.11+/`uv` 기반,
공개 후 빠르게 성장(GitHub ~46k star). 핵심 구성:

| 구성 | 내용 |
|---|---|
| **Agent loop + 학습** | 대화/작업 처리 루프에 학습 메커니즘 내장 |
| **3층 메모리(closed learning loop)** | ① agent-curated 메모리 + 주기적 nudge + FTS5 교차세션 검색(LLM 요약), ② **자율 skill 생성 + 사용 중 self-improvement**(agentskills.io 표준 호환), ③ Honcho dialectic 방식 "사용자 모델링" |
| **Gateway(메시징)** | Telegram/Discord/Slack/WhatsApp/Signal/Matrix/Teams/Email/SMS 등 20+ 플랫폼 |
| **모델 연결** | model-agnostic. Nous Portal/OpenRouter/OpenAI/**z.ai(GLM)/Kimi(Moonshot)** + **OpenAI 호환 endpoint**(Ollama·vLLM 로컬 포함). `hermes model` 로 코드 수정 없이 전환 |
| **배포/실행** | local/Docker/SSH/Modal/Daytona/Singularity, cron 스케줄러, subagent 병렬 위임, MCP 서버 연동 |
| **능력** | 웹 검색·추출·브라우징, **vision *analysis*(이미지 해석)**, 이미지 생성, TTS, cron 자동화 |

**중요한 공백 두 가지 (사실 확인됨):**

1. **GUI 자동화·컴퓨터비전이 설계 목적이 아니다.** 문서상 "vision"은 이미지 *해석*을 뜻하며,
   workflow_3가 요구하는 **"CV가 좌표 권위, VLM은 영역 식별만"** 파이프라인과는 층위가 다르다.
   데스크톱 제어는 범용 `computer-use-linux` MCP / cloud browser(Browser Use)로만 언급되고,
   **Windows 네이티브 앱(RCS) 정밀 구동은 대상이 아니다**(Windows 지원 자체가 신규·엣지케이스 존재).
2. **GLM 5.2 / Kimi-K2.6는 Hermes 전용 기능이 아니다.** 이들은 OpenAI 호환 텍스트/추론 모델이고,
   workflow_3는 이미 `flask_vlm.py`의 **`direct` 게이트웨이(`common.llm.skhynix.com`)로 Kimi-K2.5·
   Qwen3-VL을 직접 호출**한다. 로컬 모델을 쓰려고 Hermes가 필요하지 않다.

---

## 2. workflow_3의 성격 재확인 (왜 이게 판단을 가르나)

CLAUDE.md에 못박힌 설계 원칙들이 곧 Hermes와의 정합성 판정 기준이다.

- **결정론(no self-modification):** 코드 변경은 Mac 작성 → git push → 오피스 pull → 실행. 모든 동작
  변화는 code review·버전관리·게이트를 거친다. **런타임에 로직이 바뀌면 안 된다.**
- **좌표 권위는 CV, VLM은 영역 식별만:** "낮은 CV 점수를 VLM 답으로 덮지 않는다. 반복 stage 전이를
  VLM이 결정하지 않는다"(2026-05-25 확정).
- **이중 안전게이트:** 실제 마우스/키보드 출력은 `SAFE_MODE=0` **그리고** `ALIGN_FAIL_*_DRY_RUN=0`
  둘 다여야 발화. 미확인 시 클릭 금지.
- **감사 추적:** `logger.py`가 VLM 호출·이벤트를 audit log로 남긴다(실장비 구동 기록).
- **No CLI args / 고정 config:** 모든 설정은 `Workflow3Settings`·env. 스크립트는 `uv run python x.py`로만.
- **자체 오케스트레이션 존재:** monitor 루프(polling+edge-trigger+manifest), `WorkflowRunner`
  (결정론적 step 시퀀싱 + journaling + try/finally 강제 teardown), cube rich-notify.

즉 workflow_3는 **이미 "안전 모델에 맞춰 최소한으로 특화된 결정론적 오케스트레이터"**를 갖고 있다.

---

## 3. 정합성 분석 — Hermes 강점 vs workflow_3 요구

| Hermes가 내세우는 것 | workflow_3에서의 실제 가치 | 판정 |
|---|---|---|
| **런타임 자기수정(skill 자율 생성/개선)** | 실장비 제어에서 최대 금기. 에이전트가 세션 사이 자기 제어 로직을 소리 없이 바꾸면 감사·재현·안전인증이 무너짐 | ❌ **부채** |
| **3층 메모리 / 세션 학습 / 사용자 모델링** | 개인 비서용 가치. 제어 루프는 "누구인지 학습"할 대상이 없음(무인 알람 대응) | ❌ 무관 |
| **20+ 메시징 gateway** | 이미 cube rich-notify로 엔지니어 통지. Slack/Telegram은 한계효용 | △ 미미 |
| **cron 스케줄러 / subagent 병렬** | monitor 루프·WorkflowRunner가 결정론적으로 이미 수행. 범용 프레임워크로 바꾸면 결정론을 되찾으려 싸워야 함 | ❌ 퇴보 |
| **model-agnostic(GLM/Kimi/로컬)** | 기존 `direct` 게이트웨이가 이미 로컬 LLM 호출. grounding은 ui-venus/mai-ui/PaddleOCR가 담당 → GLM/Kimi로 대체 불가 | ⭕ 이미 보유 |
| **vision analysis / 이미지 생성 / TTS** | workflow_3 vision은 "CV 좌표 권위 + 특정 grounding 모델". Hermes vision은 층위가 다름 | ❌ 무관 |
| **MCP 연동** | 필요하면 workflow_3가 직접 MCP를 붙이면 됨. Hermes를 낄 이유 아님 | △ 중립 |
| **Windows RCS 정밀 구동** | Hermes 비대상(Linux computer-use 중심). workflow_3의 핵심 난제는 그대로 남음 | ❌ 미해결 |

### 3.1 핵심 논지 — Hermes는 workflow_3가 "없는 문제"를 푼다

workflow_3의 실제 병목(메모리·git 이력에 다수 기록)은 다음이며, **전부 에이전트 아키텍처가 아니라
CV·데이터·calibration 문제**다. Hermes는 이 중 어느 것도 건드리지 못한다.

1. **CV 변별력(가장 큰 레버):** SEM align은 aperture problem — junction만 unique하고 주변 line/flat이
   희석. 매칭 점수면이 평평(rank-1 ≈ 0.5, coin flip). template-bank/ensemble 실험이 "member-fusion으로는
   못 넘는 벽"을 확인 → 진짜 레버는 **align key 재등록**이지 오케스트레이터가 아님.
2. **물리 calibration:** 더블클릭 recenter·휠↔배율·`read_mode()` 실측이 전부 cold-start. 오피스에서
   실장비로 맞춰야 함. LLM 에이전트로 대체 불가.
3. **오피스 데이터 수집:** consensus history pool(class·recipe·modality별 최근 S 8~10장) 적재가
   활성화 게이트. `office_success_downloader` 적재량이 레버(첫 실행 join 8/193 데이터게이트).
4. **Windows 포그라운드/UIPI/DPI:** 관리자 권한·foreground takeover·DPI 좌표 보정 등 OS 밀착 문제.

> 요컨대, Hermes를 얹어도 위 4개는 1도 나아지지 않는다. "더 빠르고 효율적"의 근거가 여기서 무너진다.

---

## 4. Hermes가 그래도 유용할 수 있는 곳 (공정하게)

프로덕션 루프 밖, **오프라인·비안전 영역**이라면 이야기가 달라진다. 다만 대부분 이미 커버된다.

- **개발/벤치 생산성:** workflow_2 오프라인 CV 벤치의 eval/A-B/튜닝 반복을, 메모리 가진 자기개선형
  비서가 도울 수 있음. → **그러나 지금 이 일을 하는 게 Claude Code**이고, `.remember/`·MEMORY.md·
  skill 시스템이 이미 세션 메모리+절차기억+subagent를 제공. Hermes는 보완재가 아니라 경쟁 제너럴리스트.
- **오피스 지식/운영:** B1 RAG(2000쪽 매뉴얼)를 Hermes 메모리·검색으로 다룰 수도 있으나, 이미
  OpenSearch+bge-m3로 구축됨. 중복 투자.
- **통지/스케줄:** Slack/Telegram 통지, cron — 필요하면 채택 가능하나 cube-notify·monitor 루프와 중복.

**공정한 결론(장소 의존):**
- **Mac 개발 장비에서는** Hermes의 "탐나는" 능력(메모리·skill·subagent·cron·로컬 모델 라우팅)이 이미
  Claude Code + 자체 루프 인프라로 보유 중 → 여기선 경쟁 제너럴리스트일 뿐 보완재 아님.
- **오피스에서는 이야기가 다르다.** 오피스엔 Claude Code가 없으므로(§7), 오피스에서 로컬 모델 위에 도는
  비안전 보조 에이전트 자리는 Claude Code가 채울 수 없다 → 여기선 Hermes가 **빈 자리를 채우는 후보**가 된다.
  단 그 자리는 "workflow_3 대체/가속"이 아니라 "workflow_3 주변 비안전 잡무 자동화"다.

---

## 5. 비용 · 리스크

- **통합 표면:** 독자 agent loop + SQLite 메모리 DB + gateway를 가진 무거운 Python 프레임워크. workflow_3의
  결정론·게이트·감사 모델에 맞추려면 Hermes의 자율성(자기수정·자율 skill)을 **끄면서** 써야 함 = 프레임워크
  철학과 상시 충돌.
- **안전성:** 실장비 제어에 자기수정 에이전트를 두는 것 자체가 인증·재현·감사에 대한 위협.
- **운영 마찰:** 혼합 Mac-dev / Windows-office 워크플로우, 관리자 권한/UIPI 이슈. Hermes 데스크톱 제어는
  Linux 지향.
- **학습·유지비:** 새 프레임워크 학습 곡선 + 의존성 무게 + 버전 추종. workflow_3 빌드 도중 도입 시
  집중력 분산.
- **기회비용:** 같은 시간에 4장(재등록 레버·calibration·오피스 수집·Windows 포그라운드)을 진전시키는 게
  실제 성능에 직결.

---

## 6. 권고

1. **workflow_3 프로덕션 루프에 Hermes를 도입하지 않는다.** 근본 성격 불일치(자기수정 ↔ 결정론·안전게이트)
   + 실제 병목 미해결 + 능력 중복.
2. **로컬 모델(GLM 5.2/Kimi-K2.6)이 목적이라면 Hermes 불필요** — 기존 `flask_vlm.py` `direct` 게이트웨이로
   바로 호출. 필요 시 `ALL_VLM_SERVICES`에 엔트리 추가로 끝.
3. **자기개선형 에이전트 자체가 궁금하면**, 프로덕션과 물리적으로 분리된 **오프라인 개인 실험**(예: 벤치
   튜닝 조수, 사내 문서 검색)으로만 좁혀서 1회성으로 평가. 단, 그마저도 Claude Code가 이미 커버함을 전제로.

### 재검토 트리거 (아래가 실제로 발생하면 다시 연다)

- 지금 스택으로 **감당 안 되는, 경계가 분명한 구체 요구**가 등장(예: 다수 메시징 플랫폼 동시 운영 봇,
  대규모 비안전 배치 자동화).
- Hermes가 **Windows 네이티브 정밀 GUI 구동 + CV 좌표 권위 원칙**을 1급으로 지원하는 방향으로 진화.
- 프로덕션과 분리된 오프라인 벤치에서 **Claude Code 대비 측정 가능한 우위**가 실증됨.

---

## 7. 보론 — Claude Code vs Hermes Agent (오피스 = Claude 불가 제약)

Hermes는 오피스에서 쓸 예정이고, **오피스에는 Claude 모델/Claude Code가 없다**(로컬 GLM 5.2·Kimi-K2.6만).
이 문맥에서 두 도구를 정면 비교한다.

### 7.1 정면 비교

| 축 | Claude Code | Hermes Agent |
|---|---|---|
| **구동 모델** | Claude 전용(Anthropic/Bedrock/Vertex 전부 Claude). 비-Claude 로컬 모델 1급 지원 없음 | **model-agnostic** — GLM/Kimi/로컬(Ollama·vLLM) OpenAI 호환 endpoint |
| **오피스(무 Claude) 가용성** | ❌ **사실상 불가**(Claude 접근 없음) | ⭕ **로컬 GLM 5.2/Kimi-K2.6 구동** |
| **데이터 경계** | Anthropic API로 전송 | 완전 self-host, 데이터 온프레 유지(fab egress 금지 환경 적합) |
| 실행 형태 | 대화형 개발 도구(터미널/IDE), 사람이 앞에서 구동 | 배포형 **상시 서비스**(VPS/클러스터/서버리스), idle-cheap |
| 인터페이스 | 터미널/IDE 세션 | TUI + **20+ 메시징 플랫폼**(Telegram/Slack…) gateway |
| 자율성 | 세션 내, 사용자 주도 | 상시 autonomous + **cron 스케줄러** + subagent |
| 메모리/학습 | CLAUDE.md/.remember(dev 장비-로컬, Claude 구동) | 3층 메모리 + **자율 skill 생성/자기개선**(SQLite 로컬) |
| 성숙도 | 성숙·안정, 광범위 도구(hooks/skills/MCP) | 신생(2026-02), 빠르게 성장하나 Windows 등 엣지케이스 |
| Windows 네이티브 GUI 구동(RCS) | 대상 아님 | **대상 아님**(Linux computer-use 중심) |

### 7.2 "Claude Code로는 안 되고 Hermes로 되는" 경우 (오피스 문맥)

1. **오피스 로컬 모델로 에이전트를 아예 구동** ← 결정적. 오피스에 Claude가 없어 Claude Code 자체가 못 돎;
   Hermes는 GLM/Kimi 로컬로 돎.
2. **온프레 완전 자급(데이터 반출 0).** fab 데이터 egress 금지 환경에서, 클라우드로 나가는 Claude Code는
   부적합; self-host Hermes는 적합.
3. **상시 무인 서비스 + 메시징 봇 운영.** 엔지니어가 Slack/Telegram으로 질의·지시, cron으로 도는 배포형 봇.
   Claude Code는 앉아서 쓰는 개발 도구지 배포형 멀티유저 봇이 아님.
4. **오피스에 귀속된 지속 메모리/자기개선.** 오피스 배포본이 자기 자리에서 로컬 모델로 학습·skill 축적.
   Claude Code 메모리는 dev 장비-로컬 + Claude 구동.

### 7.3 이 제약이 권고를 바꾸나 — 절반만

- **실시간 안전제어 루프에 넣는 판단: 그대로 미채택.** 자기수정 부채·진짜 병목(CV/calibration/데이터)·
  **오피스에서도 Windows RCS 미구동**(Linux computer-use). 제어 루프는 결정론 자체 코드 유지.
- **오피스-사이드 비안전 보조 도구 판단: 열림.** 오피스에서 Claude Code가 못 채우는 자리를 Hermes가 채울 수
  있음 — 로그 분류, consensus 데이터 수집 babysitting, 통지, 2000쪽 매뉴얼 RAG 질의 등 **workflow_3 주변 잡무**.

### 7.4 냉정한 단서 — 품질은 구동 모델에 종속

Claude Code가 강한 이유의 절반은 Claude 모델이 agentic tool-use에 강하기 때문이다. Hermes를 GLM 5.2/
Kimi-K2.6 위에서 돌리면 "오피스에 에이전트가 생긴다"는 사실이지만, **자기개선 루프·skill 생성의 안정성은
구동 모델에 종속**된다. K2.6은 agentic·멀티모달에 쓸 만하고 GLM은 코드에 강하나, Claude 수준을 기대하면 안 된다.
따라서 오피스 도입 시 **기대치를 "비안전 잡무 자동화"로 낮춰 잡고**, 소규모 오프라인 PoC로 실제 모델 품질을
먼저 검증할 것.

---

## 참조

- [Hermes Agent Documentation | Nous Research](https://hermes-agent.nousresearch.com/docs/)
- [GitHub — NousResearch/hermes-agent](https://github.com/nousresearch/hermes-agent)
- [AI Providers | Hermes Agent (GLM·Kimi·로컬 endpoint 지원)](https://hermes-agent.nousresearch.com/docs/integrations/providers)
- [Run Hermes Agent with Local Models | NVIDIA DGX Spark](https://build.nvidia.com/spark/hermes-agent)
- [Hermes Unlocks Self-Improving AI Agents | NVIDIA Blog](https://blogs.nvidia.com/blog/rtx-ai-garage-hermes-agent-dgx-spark/)
- 내부: `CLAUDE.md`(workflow_3 설계 원칙·안전게이트·모델 게이트웨이), `poc/workflow_3/README.md`
