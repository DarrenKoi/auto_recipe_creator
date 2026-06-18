<!--
Marp 슬라이드 소스. 렌더링:
  npx @marp-team/marp-cli@latest weekly_report_slides.md -o weekly_report_slides.pdf
  npx @marp-team/marp-cli@latest weekly_report_slides.md -o weekly_report_slides.html
VS Code 의 "Marp for VS Code" 확장으로 미리보기도 가능.
-->
---
marp: true
theme: default
paginate: true
size: 16:9
header: 'workflow_3 Weekly Report'
footer: '2026.06.11 ~ 06.18'
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Weekly Report — workflow_3

### 실시간 align-fail 모니터링 루프

**2026.06.11 ~ 06.18** · 커밋 ~60개 · 전량 `main` 직접 반영

---

## 이번 주 핵심 4가지

1. **align 모듈 구조 재편 완료**
   `vision/` → `align/`, matching / diagnostics 서브패키지
2. **consensus 템플릿을 라이브 보정 경로에 정식 배선**
   (활성화는 office downloader 대기)
3. **check-only 모니터에 현장 진단 기능 탑재**
   SEM-box 검출 + PM 배율 줌 래더
4. **만성 모호 recipe 재등록 플래깅 자동화**

---

## 1. 구조 재편 — align 도메인 정리

**상태: ✅ 완료**

- `vision/` → `align/` 분리
  - `matching/` (엔진 + ensemble)
  - `diagnostics/` (오프라인 리뷰)
- workflow_1 레거시 import fallback 제거 (office 어댑터는 `monitor/`만)
- CLAUDE.md 구조 동기화

> 4-layer DAG 확립: util → {vlm, runner} → capabilities → monitor

---

## 2. Cond-aware 박스-크롭 템플릿

**상태: ✅ 완료**

- CV 흰박스 검출 → **cond.txt 기반 기하 박스-크롭**으로 대체
- `AlignKeyTemplate`이 `align_offset_xy` 운반
  - reposition 시 `offset × best_scale` 적용
- `ALIGN_FAIL_COND_BOX_CROP` kill-switch, 7-task TDD 포팅

> workflow_2 검증 → workflow_3 포팅 원칙 준수

---

## 3. Consensus → 라이브 보정 배선

**상태: 🟡 코드 완료 / 활성화 대기**

- consensus CV 프리미티브 · build/gate/select bit-parity 포팅
- `resolve_templates`: **consensus 우선, rcp 폴백**
  - cold-cache bounded sync, TTL 신선도, atomic swap
- 모든 실패 → rcp 폴백, `ALIGN_FAIL_CONSENSUS` kill-switch

> **남은 게이트 = office_success_downloader 구현** (S 이미지 공급원)

---

## 4. Check-only 모니터 + 현장 진단

**상태: 🟡 구현 완료 / 오피스 캘리브레이션 중**

- **SEM-box 검출 + PM OM/SEM 모드 판독**
  104/210 = OM, K 접미사 = SEM → modality 결정
- **점유 'select' 팝업 검출** — 타 엔지니어 사용 중 백오프(300s)
- **줌 인/아웃 래더** — ambiguous / not_visible 시 양방향 스윕
- **PM-dropdown 폴백** — wheel 무효 장비용 value-space 배율 제어
- **RCS 커서 추적 수정** — teleport → glide + jiggle

---

## 5. 재등록 플래깅 · 6. VLM 2-image 폴백

**재등록 플래깅 — ✅ 완료**
- `ambiguous`(2nd/best > τ) 시 `reregister_recommended` + 2nd-best 위치(magenta) 표시
- 어떤 recipe align key를 재등록할지 식별

**VLM 2-image align-point 폴백 — 🔬 실험 중**
- rcp vs live 2장 비교 (Kimi-K2.6), VLM은 영역만·좌표는 CV
- Kimi-K2.5 → K2.6 승급, Qwen3-VL 제거

---

## 7. MSR 제거 · 8. 기타 인프라

**MSR 프로덕션 제거 — 🟡 진행 중**
- `align_img_from_msr` dead I/O 확인 → `include_msr` 플래그로 rcp-only gather

**기타**
- **recording_filter** 패키지 신설 (프레임 → interaction timeline, 18 테스트)
- 엔지니어 align 완료 시 N>5 자동 종료, watch 5분 캡
- 시작 시 경로 헬스 리포트 (MES 경로 불일치 조기 경고)

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

---

## 다음 주 우선순위

1. **office_success_downloader 구현**
   → consensus 라이브 보정 활성화의 마지막 게이트
2. **PM-dropdown 줌 + SEM-box 검출 오피스 실측 캘리브레이션**
   → 클릭 좌표 · 배율 매핑 검증
3. MSR 제거 잔여 정리 + 오프라인 msr-fetch 스크립트
4. VLM 2-image 폴백 A/B 결론 → 채택 여부 결정

---

<!-- _paginate: false -->

# 감사합니다

**workflow_3** — 실시간 align-fail 모니터링 루프
