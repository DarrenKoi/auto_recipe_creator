# Weekly Report — workflow_3

**기간:** 2026.06.11 ~ 06.18
**대상:** `poc/workflow_3/` (실시간 align-fail 모니터링 루프)
**커밋:** 약 60개, 전량 `main` 직접 반영

---

## 요약

workflow_3 실시간 align-fail 루프의 **구조 통합과 현장 진단 기능**을 집중 개발한 한 주.
핵심 성과 4가지:

1. **align 모듈 구조 재편 완료** (`vision/` → `align/`, matching/diagnostics 서브패키지)
2. **consensus 템플릿을 라이브 보정 경로에 정식 배선** (활성화는 office downloader 대기)
3. **check-only 모니터에 SEM-box 검출 + PM 배율 줌 래더 진단 기능 탑재**
4. **만성 모호 recipe 재등록 플래깅 자동화**

---

## 1. 구조 재편 — align 도메인 정리 · **완료**

- `vision/` → `align/` 분리, `matching/`(엔진 + ensemble) · `diagnostics/`(오프라인 리뷰) 서브패키지로 재편 (`3c93b2f`)
- workflow_1 레거시 경로 import fallback 제거 — office 어댑터는 `monitor/`만이 정식 위치 (`de8e11b`)
- CLAUDE.md 구조 동기화 (`98b5214`)

> 4-layer DAG (util → {vlm, runner} → capabilities → monitor) 확립.

## 2. Cond-aware 박스-크롭 템플릿 · **완료**

- CV 흰박스 검출을 **cond.txt 기반 기하 박스-크롭**으로 대체 — `AlignKeyTemplate`이 `align_offset_xy` 운반, reposition 시 `offset * best_scale` 적용 (`1ca3d87`, `34028e3`, `4296c59`)
- `ALIGN_FAIL_COND_BOX_CROP` kill-switch 추가, 7-task TDD로 포팅 (`e16dfb6`, `0b5fb5e`)

> workflow_2에서 검증 후 workflow_3 포팅 원칙대로 진행.

## 3. Consensus → 라이브 보정 배선 · **코드 완료 / 활성화 대기**

- consensus CV 프리미티브 · build/gate/select를 bit-parity로 포팅 (`4b2cff4`, `4578c7e`)
- `resolve_templates`: **consensus 우선, rcp 폴백** 라우팅 + cold-cache bounded sync, TTL 신선도, atomic swap (`4be70e3`, `acc3808`, `6214307`)
- `correct_align_fail_auto`가 resolve 경로 경유, 모든 실패 → rcp 폴백, `ALIGN_FAIL_CONSENSUS` kill-switch (`acc3808`)

> **남은 활성화 게이트 = office_success_downloader 구현** (S 이미지 공급원).

## 4. Check-only 모니터 + 현장 진단 · **구현 완료 / 오피스 캘리브레이션 중**

경량 변형(접속 → 1프레임 캡처 → 종료, 보정·녹화 없음) 위에 진단 스택 구축:

- **SEM-box 검출 + PM OM/SEM 모드 판독** — 화면 PM 배율(104/210 = OM, K 접미사 = SEM)로 modality 결정, live SEM box ROI 내 매칭 (`c59bfc3`, `2329d75`)
- **점유 'select' 팝업 검출** — 타 엔지니어 사용 중 감지 시 공유/종료 옵션 안 건드리고 백오프(300s 쿨다운) (`2329d75`)
- **줌 인/아웃 래더** — feasibility가 `ambiguous` / `not_visible`일 때 mouse wheel 양방향 스윕 + 매 rung 재매칭 (`f01a053`, `9d708a7`)
- **PM-dropdown 폴백** — 이 장비는 wheel이 배율을 안 바꿈 확인 → **'PM' 버튼(2-stage VLM locate) → 드롭다운**으로 value-space 배율 제어가 **기본값** (`81397ea`, `546ba35`, `3058617`)
- **RCS 커서 추적 수정** — teleport → glide + jiggle, RCS가 커서를 live SEM box까지 따라오게 (`ed93bc5`)
- 클릭 검증 오버레이 디버그 이미지(PM 박스 / 버튼점 / 드롭다운 영역) 저장

## 5. 만성 모호 recipe 재등록 플래깅 · **완료**

- feasibility가 `ambiguous`(2nd/best ratio > τ)이면 `reregister_recommended` 플래그 + audit-log, marked 이미지에 **2nd-best 위치(magenta)** 표시 (`cc32991`, `ca5a604`)
- 엔지니어가 어떤 recipe의 align key를 더 변별력 있는 영역으로 재등록할지 식별 가능

## 6. VLM 2-image align-point 폴백 · **실험 단계**

- CV가 모호할 때 rcp vs live 2장 비교 프롬프트 (`prompt_align_compare`, Kimi-K2.6) — VLM은 영역 식별만, 최종 좌표는 CV (2026-05-25 원칙) (`5fca221`)
- workflow_2 probe에서 Kimi 검증(Gemma 실패), 빈 응답 진단 강화
- Kimi-K2.5 → **K2.6** 승급, deprecated Qwen3-VL 제거 (`0abaf7a`)

> workflow_2 A/B 검증 중, 프로덕션 미적용.

## 7. MSR를 프로덕션 루프에서 제거 · **진행 중**

- 분석 결과 `align_img_from_msr`가 프로덕션 루프에선 dead I/O → spec/plan 작성 후 `include_msr` 프로토콜 플래그로 **rcp-only 프로덕션 gather** 구현 (`8eeffb6`, `f4cb97e`, `66a5784`)

> 잔여: assets 경고 정리 · 오프라인 msr-fetch 스크립트 · 테스트.

## 8. 기타 인프라

- **recording_filter** 패키지 신설 — RecordingSession 프레임 → interaction timeline (cv2 change-detect + VLM 커서 클릭 검출), 18 테스트 (`5eed267`~`8ead575`)
- 엔지니어 align 완료 시 **N>5 자동 종료**, watch 5분 캡 (`ac31c6a`)
- 시작 시 **경로 헬스 리포트** — align_images MES 경로 불일치 조기 경고 (`11b03e1`)
- rcp/msr office downloader 훅을 notify에서 분리 (`cdbcff5`)

---

## 다음 주 우선순위

1. **office_success_downloader 구현** — consensus 라이브 보정 활성화의 마지막 게이트
2. **PM-dropdown 줌 + SEM-box 검출 오피스 실측 캘리브레이션** (클릭 좌표 · 배율 매핑 검증)
3. MSR 제거 잔여 정리 + 오프라인 msr-fetch 스크립트
4. VLM 2-image 폴백 workflow_2 A/B 결론 → 채택 여부 결정

---

## 진행 현황 한눈에

| 항목 | 상태 |
| --- | --- |
| align 모듈 구조 재편 | ✅ 완료 |
| Cond-aware 박스-크롭 템플릿 | ✅ 완료 |
| Consensus 라이브 보정 배선 | 🟡 코드 완료 / downloader 대기 |
| Check-only 진단(SEM-box·PM 줌) | 🟡 구현 완료 / 캘리브레이션 중 |
| 재등록 플래깅 | ✅ 완료 |
| VLM 2-image 폴백 | 🔬 실험 중 |
| MSR 제거 | 🟡 진행 중 |
