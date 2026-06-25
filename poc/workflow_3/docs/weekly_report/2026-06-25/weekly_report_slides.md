---
marp: true
theme: default
paginate: true
size: 16:9
backgroundColor: '#FFFFFF'
header: '측정 실패 abort + matcher 벤치 Weekly Report'
footer: '2026.06.25'
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
  .branch{ margin-top:6px; padding:8px 12px; background:#FAFCFF; border:1.5px dashed #90CAF9; border-radius:10px; font-size:14px; color:#37474F; }
  .branch b{ color:#0D47A1; }
  .legend{ font-size:13px; color:#455A64; margin-top:6px; }
  table{ font-size:17px; }
  th{ background:#ECEFF1; color:#37474F; }
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Weekly Report — 측정 실패 abort 잡 + matcher 벤치

### 효율적 장비 운영 — 측정 연속 실패 시 자동 abort (`workflow_3e`)

**2026.06.25** · 커밋 ~15개 · 전량 `main` 직접 반영

---

## 이번 주 요약

효율적 장비 운영을 위해 **두 번째 MES 알람 — "측정 연속 실패" 자동 abort** 잡을 프로덕션 루프에 추가.

- **측정 실패 abort 잡 착수** — 정렬 성공 후 포인트 연속 실패(예: 100점 중 ~20점)면 run 중단 → wafer 손실 방지
- **`workflow_3` core 무수정** — 별도 확장 패키지 `workflow_3e` 단방향 import
- **이중 게이트 + notify-only 기본** — CV 보정과 동일 안전 모델, 6 test 파일 통과
- **template-bank matcher 벤치 구현 완료** — heatmap-primary + RRF arm (오피스 평가 대기)

> 핵심: 단일 RCS 커서라 **두 잡은 한 프로세스에서 직렬**. abort 큐잉 가능 → 락 불필요. 실패 카운팅은 **MES 소유**.

---

## 통합 루프 — 한 프로세스가 두 MES 알람을 직렬 처리

<div class="flow-row">
  <div class="node done"><div class="step">DETECT</div><div class="ttl">MES 1회 polling</div><div class="sub">tick 당 한 번 조회 → 두 필터 분배<br>streak은 <b>MES 소유</b></div></div>
  <div class="node done"><div class="step">JOB A (기존)</div><div class="ttl">align fail → 보정</div><div class="sub">ALID=9006 → CV 보정<br>workflow_3 무수정</div></div>
  <div class="node prog"><div class="step">JOB B (신규)</div><div class="ttl">측정 실패 → abort</div><div class="sub">임계 알람 → 접속·캡처·Stop<br><span class="badge b-new">착수</span></div></div>
  <div class="node done"><div class="step">SERIALIZE</div><div class="ttl">단일 커서=직렬</div><div class="sub">커서 1개·abort 큐잉<br><b>락 불필요</b></div></div>
</div>
<div class="branch"><b>왜 별도 <code>workflow_3e</code>:</b> 잡 추가 때마다 <code>ALIGN_FAIL_*</code> 플래그·분기 증식을 격리. <b>단방향 import</b>(3e→3, 역방향 금지)로 core 편집 0건. connect/창대기/캡처/teardown 재사용, <code>_exec_abort_measurement</code>+버튼 locator 만 신규.</div>
<div class="legend"><span class="badge b-done">기존·재사용</span> workflow_3 무수정 &nbsp; <span class="badge b-new">착수</span> 이번 주 신규</div>

---

## abort 잡 — 안전(이중 게이트, notify-only) & 환경변수

파괴적·외부 영향이라 **이중 게이트**(`SAFE_MODE=0` **그리고** `MEAS_FAIL_ABORT_DRY_RUN=0`) + **notify-only 기본**. dry-run 이 VLM locate 포함 전체 경로 태우고 **클릭만** 게이트.

| Env (`MEAS_FAIL_*`) | 기본 | 의미 |
| --- | :--: | --- |
| `MEAS_FAIL_ABORT_ENABLED` | `1` | abort 잡 마스터 토글(감지+알림) |
| `MEAS_FAIL_ALID` | `""` | 임계 알람 ALID — **오피스 확인 필요** |
| `MEAS_FAIL_ABORT_DRY_RUN` | `1` | 클릭 게이트(`SAFE_MODE=0` **AND** `=0`) |
| `MEAS_FAIL_ABORT_BUTTON_SERVICE` | `ui-venus` | Stop/Abort locator route_slug |

> 기본 경로: 감지 → 접속 → 캡처 → 버튼 locate → **cube 알림**, 클릭 없음.

---

## template-bank matcher 벤치 (병행)

직전 "다음 단계"였던 매처 개선 — rcp 가 **success 에서도** 약하게(~0.2–0.3) localize, SEM recall ~68% 병목.

- **가설 H1** — N 개 S-crop 을 **개별(sharp)** 유지 + **dense 응답을 멤버 합의로 융합**이 median 보다 낫다
- **primary = heatmap soft-voting** — top-K 에 못 든 약한 응답도 합산해 `gt_not_in_topk` 정면 공략. RRF 는 extra arm
- **반대가설 H0(먼저 배제)** — distractor 가 S 간 일관되면 합의가 **distractor 강화** → median 보다 나쁠 수도
- **구현** — `bank_build`(개별, no median) + heatmap/RRF 2-arm + **kill-test**(lattice-period·GT-bucket) + eval helpers

> bench 전용 bit-parity fork(`workflow_3` 무수정). **오피스 골든셋 평가 미실행** → H1/H0 판정 후 포팅 게이트.

---

## 작업 항목 진행 현황

| 항목 | 상태 | 요지 |
| --- | :--: | --- |
| abort — `workflow_3e` 패키지 | <span class="badge b-new">코드 shipped</span> | core 무수정 단방향 확장, 6 test 파일 통과 |
| abort — 안전 게이트 | <span class="badge b-done">적용</span> | 이중 게이트 + notify-only 기본 |
| abort — dry-run 경로 | <span class="badge b-done">검증</span> | 캡처 → 좌표 로깅 → notify, **클릭 없음** |
| abort — 오피스 detection 입력 | <span class="badge b-prog">오피스 게이트</span> | `office_meas_many_fails.py`/`MEAS_FAIL_ALID` |
| abort — 버튼 calibrate + 무장 | <span class="badge b-prog">예정</span> | dry-run 검증 후 명시 단계로 무장 |
| template-bank 벤치 | <span class="badge b-done">구현 완료</span> | heatmap+RRF 2-arm + kill-test + helpers |
| template-bank — 오피스 평가 | <span class="badge b-plan">대기</span> | digest 미실행, 포팅 게이트 |

---

## 다음 주 우선순위

1. **abort 잡 오피스 활성화** — `office_meas_many_fails.py` 구현 + `MEAS_FAIL_ALID` 확정 + 실장비 버튼 calibrate
2. **dry-run → 무장 단계** — 캡처로 버튼 검증 후 `SAFE_MODE=0 MEAS_FAIL_ABORT_DRY_RUN=0`
3. **template-bank 골든셋 평가 실행** — heatmap vs RRF vs baseline digest, H1/H0 판정
4. rcp **이미지** 약함 vs **matcher** 약함 분리 진단

---

<!-- _paginate: false -->

# 감사합니다

**효율적 장비 운영** — 측정 연속 실패 자동 abort + matcher distinctiveness 개선

<!--
빌드: npx -y @marp-team/marp-cli@latest weekly_report_slides.md --html --pptx -o weekly_report_slides.pptx
HTML(div) 렌더링을 위해 --html 플래그 필수.
-->
