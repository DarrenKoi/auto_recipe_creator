---
marp: true
theme: default
paginate: true
size: 16:9
backgroundColor: '#FFFFFF'
header: '재등록 리포트 Weekly Report'
footer: '2026.06.19 ~ 06.24'
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
  .b-stop{ background:#FFEBEE; border-color:#C62828; color:#B71C1C; }
  .flow-row{ display:flex; gap:28px; margin:8px 0; }
  .node{ position:relative; flex:1; background:#fff; border:2px solid #CFD8DC; border-radius:12px; padding:8px 12px; }
  .node.done{ border-color:#66BB6A; background:#F4FBF4; }
  .node.prog{ border-color:#FFA726; background:#FFFBF3; }
  .node.stop{ border-color:#EF5350; background:#FFF6F6; }
  .node .step{ font-size:11px; font-weight:700; color:#90A4AE; }
  .node .ttl{ font-size:16px; font-weight:700; margin:2px 0 4px; }
  .node .sub{ font-size:12px; color:#607D8B; line-height:1.3; }
  .node:not(:last-child)::after{ content:"\203A"; position:absolute; right:-20px; top:50%; transform:translateY(-50%); font-size:30px; color:#B0BEC5; font-weight:700; }
  .branch{ margin-top:6px; padding:8px 12px; background:#FFF8F8; border:1.5px dashed #EF9A9A; border-radius:10px; font-size:14px; color:#37474F; }
  .branch b{ color:#B71C1C; }
  .legend{ font-size:13px; color:#455A64; margin-top:6px; }
  table{ font-size:17px; }
  th{ background:#ECEFF1; color:#37474F; }
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Weekly Report — 재등록 리포트 Phase 1·2

### align-key 재등록 우선순위 리포트 (오프라인 CV 벤치)

**2026.06.19 ~ 06.24** · 커밋 ~25개 · 전량 `main` 직접 반영

---

## 이번 주 요약

만성 모호 align-key를 **데이터로 식별해 재등록 우선순위를 매기는 리포트**를 2단계로 개발.

- **Phase 1 — S-only 스크리닝 완료** — success 프레임만으로 약한 key 랭킹 + 대체 박스 제안
- **Phase 1 — box-fidelity 버그 해결** — off-center 박스 fidelity 0 → `w_sugg 0→1`
- **Phase 2 — E-frame confirmation 실험 종료** — 오피스 실측서 신호 신뢰 불가
- **다음 단계 — matcher 개선(template-bank) 착수**

> 근본 병목 = **rcp align-key가 success에서도 변별력 부족** → collapse 신호 자체가 성립 불가.

---

## 작업 흐름 — Phase 1 → Phase 2 → 다음 단계

<div class="flow-row">
  <div class="node done"><div class="step">PHASE 1</div><div class="ttl">S-only 스크리닝</div><div class="sub">약한 key latent-risk 랭킹 + 박스 제안<br><span class="badge b-done">완료</span></div></div>
  <div class="node done"><div class="step">PHASE 1·디버그</div><div class="ttl">box-fidelity 버그</div><div class="sub">offset 보정+tight band<br><b>w_sugg 0→1</b></div></div>
  <div class="node stop"><div class="step">PHASE 2</div><div class="ttl">E-frame confirm</div><div class="sub">S→E collapse 승급(60 tests)<br><span class="badge b-stop">실험 종료</span></div></div>
  <div class="node prog"><div class="step">NEXT</div><div class="ttl">matcher 개선</div><div class="sub">template-bank 벤치<br><span class="badge b-new">착수</span></div></div>
</div>
<div class="branch"><b>왜 멈췄나:</b> 오피스 실측 <b>confirmed 0</b>(실제 점수 ~0.2-0.3, 임계 ~0.6 가정) → 임계 낮추면 <b>5/6 false positive</b>(delta 0.005~0.033 = collapse 아님). rcp key가 <b>success에서도 약함</b>(s_rep 0.15-0.31)이라 "S에서 높다가 E에서 무너지는" 신호 불가.</div>
<div class="legend"><span class="badge b-done">완료</span> 반영·검증 &nbsp; <span class="badge b-stop">실험 종료</span> 코드 보존·신호 불가 &nbsp; <span class="badge b-new">착수</span> 이번 주 시작</div>

---

## 오피스 실측 — Phase 2 종료 근거

dataset health: **117 recipes · confirm-capable 28 · E-bearing 28 · incomplete 89**(대부분 E 없음)

| row | s_rep → e_rep | delta | 판정 |
| --- | :--: | :--: | --- |
| OM | 0.244 → 0.187 | 0.057 | 유일한 진짜 collapse 후보 (noise floor 미측정) |
| SEM | 0.155 → 0.122 | 0.033 | <span class="badge b-stop">false positive</span> E-floor 분기로만 |
| SEM | 0.182 → 0.162 | 0.020 | <span class="badge b-stop">false positive</span> |
| SEM | 0.158 → 0.153 | 0.005 | <span class="badge b-stop">false positive</span> |

> `E_FLOOR` 절대임계가 점수대 전부 ~0.15-0.25인 데이터서 무효 → '낮은 점수'를 'collapse'로 오판.

---

## 작업 항목 진행 현황

| 항목 | 상태 | 요지 |
| --- | :--: | --- |
| Phase 1 — S-only 스크리닝 | <span class="badge b-done">완료</span> | tier+risk+랭킹, C1 스크리닝 + C2 박스 제안 |
| Phase 1 — box-fidelity 버그 | <span class="badge b-done">완료</span> | offset 보정+tight band+tol 0.30, **w_sugg 0→1** |
| Phase 2 — E-confirm 구현 | <span class="badge b-done">완료</span> | SDD 6태스크, `E_CONFIRMED` tier, 48 tests |
| Phase 2 — 오피스 검증 | <span class="badge b-stop">종료</span> | confirmed 0 → 5/6 false positive, **신호 불가** |
| EFRAME_ROOT + dataset-health | <span class="badge b-done">유지</span> | 전용 루트 + 사전점검(12 tests), 보존 |
| 다음 단계 — template-bank matcher | <span class="badge b-plan">착수</span> | heatmap-primary + RRF arm 벤치 spec |

---

## 다음 주 우선순위

1. **template-bank matcher 벤치 구현** — heatmap-primary + RRF arm, distinctiveness 정면 개선
2. **re-registration 신호 활용** — Phase 1 랭킹으로 더 distinctive 영역에 align-key 재등록
3. rcp **이미지** 약함 vs **matcher** 약함 분리 진단
4. (정리) `REREGISTER_E_CONFIRM` 기본 0으로 — 오해 소지 출력 방지

---

<!-- _paginate: false -->

# 감사합니다

**재등록 리포트** — align-key 변별력 진단 → matcher 개선으로

<!--
빌드: npx -y @marp-team/marp-cli@latest weekly_report_slides.md --html --pptx -o weekly_report_slides.pptx
HTML(div) 렌더링을 위해 --html 플래그 필수.
-->
