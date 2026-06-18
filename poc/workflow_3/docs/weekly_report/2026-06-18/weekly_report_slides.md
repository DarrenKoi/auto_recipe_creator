---
marp: true
theme: default
paginate: true
size: 16:9
backgroundColor: '#FFFFFF'
header: 'workflow_3 Weekly Report'
footer: '2026.06.11 ~ 06.18'
style: |
  section { font-family: 'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif; color:#212121; font-size: 22px; background:#FFFFFF; }
  h1 { color:#1a1a1a; }
  h2 { color:#455A64; border-bottom:2px solid #CFD8DC; padding-bottom:4px; font-size: 28px; }
  code { background:#ECEFF1; padding:1px 5px; border-radius:4px; font-size:.85em; }
  .badge { display:inline-block; padding:0 7px; border:1px solid; border-radius:12px; font-size:13px; font-weight:700; }
  .b-done{ background:#E8F5E9; border-color:#2E7D32; color:#1B5E20; }
  .b-prog{ background:#FFF3E0; border-color:#ED6C02; color:#B25500; }
  .b-plan{ background:#F3E5F5; border-color:#6A1B9A; color:#4A148C; }
  .b-new { background:#E3F2FD; border-color:#1565C0; color:#0D47A1; }
  .flow-row{ display:flex; gap:26px; margin:8px 0; }
  .node{ position:relative; flex:1; background:#fff; border:2px solid #CFD8DC; border-radius:12px; padding:8px 12px; }
  .node.done{ border-color:#66BB6A; background:#F4FBF4; }
  .node.prog{ border-color:#FFA726; background:#FFFBF3; }
  .node .step{ font-size:11px; font-weight:700; color:#90A4AE; }
  .node .ttl{ font-size:16px; font-weight:700; margin:2px 0 4px; }
  .node .sub{ font-size:12px; color:#607D8B; line-height:1.3; }
  .node:not(:last-child)::after{ content:"\203A"; position:absolute; right:-19px; top:50%; transform:translateY(-50%); font-size:30px; color:#B0BEC5; font-weight:700; }
  .branch{ margin-top:6px; padding:8px 12px; background:#FAFCFF; border:1.5px dashed #90CAF9; border-radius:10px; font-size:14px; color:#37474F; }
  .legend{ font-size:13px; color:#455A64; margin-top:6px; }
  table{ font-size:17px; }
  th{ background:#ECEFF1; color:#37474F; }
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Weekly Report — workflow_3

### 실시간 align-fail 모니터링 루프

**2026.06.11 ~ 06.18** · 커밋 ~60개 · 전량 `main` 직접 반영

---

## 이번 주 요약

구조 통합과 현장 진단 능력을 집중 개발한 한 주.

- **align 모듈 구조 재편 완료** — `vision/` → `align/` (matching·diagnostics 서브패키지)
- **consensus 템플릿을 라이브 보정 경로에 정식 배선** — 코드 완료, 활성화는 downloader 대기
- **check-only 진단 탑재** — SEM-box 검출 · PM 배율 줌 래더
- **만성 모호 recipe 재등록 플래깅 자동화**

> 남은 활성화 게이트 = **office_success_downloader 구현** 하나.

---

## 실시간 루프 — 단계별 진행

<div class="flow-row">
  <div class="node done"><div class="step">STEP 1</div><div class="ttl">알람 감지</div><div class="sub">ALID=9006 polling + edge-trigger</div></div>
  <div class="node done"><div class="step">STEP 2</div><div class="ttl">RCS 접속·툴 매칭</div><div class="sub">login→List→더블클릭<br><span class="badge b-new">NEW</span> 점유 팝업 백오프</div></div>
  <div class="node prog"><div class="step">STEP 3</div><div class="ttl">CV 매칭·feasibility</div><div class="sub">consensus→rcp 매칭<br><span class="badge b-new">NEW</span> SEM-box·PM 모드</div></div>
  <div class="node prog"><div class="step">STEP 4</div><div class="ttl">보정 reposition+OK</div><div class="sub">key 보이면 1차 보정<br><span class="badge b-new">NEW</span> PM 줌 래더</div></div>
</div>
<div class="flow-row">
  <div class="node done"><div class="step">STEP 5</div><div class="ttl">실패 시 알림</div><div class="sub">cube notify<br><span class="badge b-new">NEW</span> 재등록 플래그</div></div>
  <div class="node done"><div class="step">STEP 6</div><div class="ttl">상시 화면 녹화</div><div class="sub">수동 조작까지 기록 · N&gt;5 조기 종료</div></div>
  <div class="node done"><div class="step">STEP 7</div><div class="ttl">툴 종료</div><div class="sub">try/finally teardown</div></div>
  <div class="node done"><div class="step">STEP 8</div><div class="ttl">다음 알람 대기</div><div class="sub">↻ STEP 1 로 순환</div></div>
</div>
<div class="branch"><b>check-only 변형</b>: 접속 → 1프레임 캡처 → 종료. 이번 주 진단 기능을 이 경로에 먼저 탑재해 오피스 실측 중.</div>
<div class="legend"><span class="badge b-done">●</span> 안정 동작 &nbsp; <span class="badge b-prog">●</span> 코드 완료·캘리브레이션 중 &nbsp; <span class="badge b-new">NEW</span> 이번 주 추가</div>

---

## 작업 항목 진행 현황

| 항목 | 상태 | 요지 |
| --- | :--: | --- |
| align 모듈 구조 재편 | <span class="badge b-done">완료</span> | `vision/`→`align/`, 4-layer DAG 확립 |
| Cond-aware 박스-크롭 템플릿 | <span class="badge b-done">완료</span> | cond.txt 기하 크롭 + `align_offset_xy`, 7-task TDD |
| Consensus 라이브 보정 배선 | <span class="badge b-prog">진행</span> | consensus→rcp 라우팅, **downloader가 게이트** |
| Check-only 진단 (SEM-box·PM 줌) | <span class="badge b-prog">진행</span> | 양방향 줌 래더 + PM-dropdown 폴백, **캘리브레이션 중** |
| 모호 recipe 재등록 플래깅 | <span class="badge b-done">완료</span> | `ambiguous` 시 권고 + 2nd-best 표시 |
| VLM 2-image align-point 폴백 | <span class="badge b-plan">실험</span> | rcp vs live 비교(K2.6), 좌표는 CV |
| MSR 프로덕션 제거 | <span class="badge b-prog">진행</span> | `include_msr` 플래그 rcp-only gather |
| recording_filter 패키지 | <span class="badge b-done">완료</span> | 녹화→interaction timeline, 18 테스트 |

---

## 다음 주 우선순위

1. **office_success_downloader 구현** — consensus 라이브 보정 활성화의 마지막 게이트
2. **PM-dropdown 줌 · SEM-box 검출 오피스 실측 캘리브레이션** — 클릭 좌표·배율 매핑 검증
3. MSR 제거 잔여 정리 + 오프라인 msr-fetch 스크립트
4. VLM 2-image 폴백 A/B 결론 → 채택 여부 결정

---

<!-- _paginate: false -->

# 감사합니다

**workflow_3** — 실시간 align-fail 모니터링 루프

<!--
빌드: npx -y @marp-team/marp-cli@latest weekly_report_slides.md --html --pptx -o weekly_report_slides.pptx
HTML(div) 렌더링을 위해 --html 플래그 필수.
-->
