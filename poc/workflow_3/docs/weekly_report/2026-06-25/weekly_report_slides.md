---
marp: true
theme: default
paginate: true
size: 16:9
backgroundColor: '#FFFFFF'
header: '재등록 리포트 · 측정 실패 abort · matcher 벤치 Weekly Report'
footer: '2026.06.19~25'
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
  .branch.warn{ background:#FFF8F8; border-color:#EF9A9A; }
  .branch.warn b{ color:#B71C1C; }
  .branch.ok{ background:#F4FBF4; border-color:#A5D6A7; }
  .branch.ok b{ color:#1B5E20; }
  .legend{ font-size:13px; color:#455A64; margin-top:6px; }
  table{ font-size:17px; }
  th{ background:#ECEFF1; color:#37474F; }
---

<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# Weekly Report — 재등록 리포트 · 측정 실패 abort · matcher 벤치

### 하나의 병목(align-key 변별력) → 세 갈래 공략 → "레버는 재등록"

**2026.06.19~25** · 커밋 ~40개 · 전량 `main` 직접 반영 · 용어는 `concepts_explained.md`

---

## 이번 주 요약 — 무엇을 시도하고 무엇이 나왔나

**"align-key 변별력이 약하다"는 근본 병목**을 세 갈래로 공략 → **매처 개선은 막혀 있고 진짜 레버는 align-key 재등록**임을 확인.

- **재등록 Phase 1** — S-only 변별력 스크리닝 + 대체 박스 제안 → <span class="badge b-done">완료</span>
- **재등록 Phase 2** — E-frame S→E collapse 승급 → 오피스 실측서 신호 신뢰 불가 → <span class="badge b-stop">종료</span>
- **측정 실패 abort 잡** — 두 번째 MES 알람 자동 abort(`workflow_3e`), notify-only 기본 → <span class="badge b-new">코드 shipped</span>
- **template-bank matcher 벤치** — heatmap+RRF 융합 구현·평가 → **rank-1 0.5, 출하 불가** → <span class="badge b-stop">기각(ADR 0006)</span>

> 핵심: 매처를 어떻게 융합해도 **1등을 못 꼽음**(rank-1 0.5). 등록된 key 가 애초에 변별력이 없으면 1등 근거가 없다 → **재등록이 유일한 레버**.

---

## ① 재등록 리포트 Phase 1·2 — 시도 → 결과

**Phase 1 (완료)** — success 프레임만으로 약한 align-key 를 latent-risk 랭킹 + 대체 박스 제안(3단 evidence tier).
→ box-fidelity 전 recipe 0 버그 추적(후보 xy=patch 중심, offset 미적용)·해결 → **`w_sugg 0 → 1`**.

**Phase 2 (종료)** — fail(E) 프레임에서 **S→E 점수 collapse** 로 confirmed 승급(60 tests, `EFRAME_ROOT`+health).

<div class="branch warn">
<b>오피스 실측 결과:</b> 실제 점수대 <b>~0.2-0.3</b>인데 임계 ~0.5 가정 → confirmed 0; 임계 낮추니 <b>5/6 false positive</b>. <b>근본 원인 = key 가 S 에서도 약함 → 무너질 높이가 없는 구조적 모순.</b> 재등록 리포트가 약하다고 flag 하는 recipe 는 애초에 success 에서도 약한 key.
</div>

---

## ② 측정 실패 abort 잡 — 한 프로세스가 두 MES 알람을 직렬 처리

<div class="flow-row">
  <div class="node done"><div class="step">DETECT</div><div class="ttl">MES 1회 polling</div><div class="sub">tick 당 한 번 조회 → 두 필터 분배<br>streak은 <b>MES 소유</b></div></div>
  <div class="node done"><div class="step">JOB A (기존)</div><div class="ttl">align fail → 보정</div><div class="sub">ALID=9006 → CV 보정<br>workflow_3 무수정</div></div>
  <div class="node prog"><div class="step">JOB B (신규)</div><div class="ttl">측정 실패 → abort</div><div class="sub">임계 알람 → 접속·캡처·Stop<br><span class="badge b-new">코드 shipped</span></div></div>
  <div class="node done"><div class="step">SERIALIZE</div><div class="ttl">단일 커서=직렬</div><div class="sub">커서 1개·abort 큐잉<br><b>락 불필요</b></div></div>
</div>
<div class="branch"><b>왜 별도 <code>workflow_3e</code>:</b> 잡 추가 때마다 <code>ALIGN_FAIL_*</code> 플래그·분기 증식을 격리. <b>단방향 import</b>(3e→3)로 core 편집 0건. connect/창대기/캡처/teardown 재사용, <code>_exec_abort_measurement</code>+버튼 locator 만 신규.</div>
<div class="branch warn"><b>안전:</b> 파괴적 행동이라 <b>이중 게이트</b>(<code>SAFE_MODE=0</code> AND <code>MEAS_FAIL_ABORT_DRY_RUN=0</code>) + <b>notify-only 기본</b>. 6 test 파일 통과. 남은 건 오피스 detection 입력 + 실장비 버튼 calibrate → 무장.</div>

---

## ③ template-bank matcher 벤치 — 가설과 3단 판정

**동기** — rcp 가 **success 에서도** 약하게(~0.2-0.3) localize, SEM recall ~68% 병목.
**H1(주력)** = S-crop 을 **개별(sharp)** 유지 + dense 응답을 **heatmap soft-voting(SUM)** 융합이 median 보다 낫다. RRF 는 extra arm.
**H0(반대가설)** = distractor 가 S 간 일관되면 합의가 **distractor 강화** → median 보다 나쁠 수도.

| 검증 단계 | 판정 | 내용 |
| --- | :--: | --- |
| 1. kill-test (H0 배제) | <span class="badge b-done">통과</span> | near_periodic om 0.014/sem 0.052 → distractor 강화 안 일어남 |
| 2. in_topk (천장) | <span class="badge b-done">이김</span> | 정답을 후보 안엔 잘 넣음 |
| 3. **rank-1 (출하 성능)** | <span class="badge b-stop">막힘</span> | **OM/SEM 둘 다 ≈ 0.5(동전 던지기)** |

---

## ③ template-bank — 결론: 기각 (ADR 0006)

<div class="branch warn">
실전은 후보 8개가 아니라 <b>1등 좌표 하나를 클릭</b> → <b>rank-1 이 실제 출하 성능</b>(<code>in_topk</code> 는 천장일 뿐). rank-1 0.5 는 출하 불가. heatmap·RRF·기타 3가지 융합이 <b>모두 같은 벽</b>에 막힘 → <b>SEM 은 어떤 멤버 융합으로도 못 푸는 ranking/distinctiveness 문제</b>. 매처-융합 소진, 레버는 <b>upstream 재등록</b>.
</div>

<div class="branch ok"><b>가장 비싸게 배운 규율:</b> 벤치 A/B 는 <code>in_topk</code>(천장)가 아니라 <b>rank-1</b>(출하 성능)으로 비교한다. 코드는 <code>TBANK_HEATMAP=0</code> kill switch 뒤 보존(16/16 tests), <code>workflow_3</code> 포팅 안 함.</div>

> **두 갈래(재등록 Phase 2 + template-bank)가 같은 결론**을 가리킴: 문제는 매처가 아니라 **align-key 이미지의 변별력**.

---

## ④ 작업 항목 진행 현황

| 항목 | 상태 | 요지 |
| --- | :--: | --- |
| 재등록 Phase 1 — S-only + box-fidelity | <span class="badge b-done">완료</span> | `w_sugg 0→1` 복구 |
| 재등록 Phase 2 — E-frame confirmation | <span class="badge b-stop">종료</span> | 60 tests, 신호 신뢰 불가 |
| abort — `workflow_3e` 패키지 | <span class="badge b-new">코드 shipped</span> | core 무수정 단방향, 6 test 파일 |
| abort — 안전 게이트 + dry-run | <span class="badge b-done">검증</span> | 이중 게이트 + notify-only |
| abort — 오피스 입력 + 무장 | <span class="badge b-prog">오피스 게이트</span> | provider/ALID + 버튼 calibrate |
| template-bank — 벤치 + 평가 | <span class="badge b-done">완료</span> | heatmap+RRF + kill-test |
| template-bank — 출하 판정 | <span class="badge b-stop">기각</span> | rank-1 0.5, ADR 0006 |

---

## ⑤ 다음 주 우선순위

1. **재등록이 유일한 레버** — Phase 1 신호로 약한 align-key 를 더 distinctive 한 영역으로 재등록(Phase 3 worklist)
2. **abort 잡 오피스 활성화** — `office_meas_many_fails.py` + `MEAS_FAIL_ALID` 확정 + 실장비 버튼 calibrate → 무장
3. **규율 정착** — 벤치 A/B 는 `in_topk`(천장)가 아니라 **rank-1**(출하 성능)으로 비교
4. rcp **이미지** 약함 vs **matcher** 약함 분리 — 무게추는 **이미지(재등록)** 쪽

---

<!-- _paginate: false -->

# 감사합니다

**하나의 병목, 세 갈래, 한 결론** — align-key **재등록**이 진짜 레버

<!--
빌드: npx -y @marp-team/marp-cli@latest weekly_report_slides.md --html --pptx -o weekly_report_slides.pptx
HTML(div) 렌더링을 위해 --html 플래그 필수.
-->
