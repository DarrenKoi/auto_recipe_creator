# Align Fail Correction 방법론 개선안

> 2026-09-21. 대상 = `poc/workflow_3/align` + `monitor/cycle.py` 의 보정 경로.
> 기준 사건 = OM search-around 오피스 성공(`a89a852`), 다음 목표 = SEM search-around + OM-D.

---

## 0. 요약

지금 방법론의 병목은 **matcher 의 점수 자체가 아니라, 점수를 낼 수 없는 상태로 들어가는 입력**이다.
2026-06~07 workflow_2 벤치가 rank-1 ~0.5 벽(SEM)에서 member-fusion 3종을 모두 소진했고, 그 뒤
실제로 보정을 막은 것은 전부 "매칭 이전" 조건이었다 - valid-mode footprint(09-19 수정), 배율 band,
그리고 아직 안 본 **극성(polarity)** 이다.

제안 6개, 우선순위 순:

| # | 제안 | 무엇을 푸는가 | 크기 | 상태 |
|---|---|---|---|---|
| 1 | **극성 정합 (OM-D)** | 반전 화면에서 sel 이 구조적으로 0.5 를 못 넘는다 | 중 | 액추에이터 **오피스 OM-D/SEM 변경 성공**(09-22), OM 행은 'OM ABC' 버튼 혼동으로 재시험 대기; 감지·라우팅 미구현 |
| 2 | **SEM search-around = frac 목표 zoom-out** | SEM 은 footprint 가 0 이라 격자가 성립하지 않는다 | 중 | 미구현 |
| 3 | **극성·배율을 '사전 조건' 으로 승격** | 실패를 보정 도중이 아니라 진입 전에 가른다 | 소 | 미구현 |
| 4 | **재등록 루프 닫기** | 측정된 진짜 벽(key 변별력) | 소(코드)·중(운영) | 신호는 이미 나옴, 전달 경로 없음 |
| 5 | **consensus 조용한 rcp 폴백 차단** | template 품질이 말없이 떨어진다 | 소 | 의심 단계 |
| 6 | **Verification 을 stub 에서 올린다** | `corrected` 가 자기보고다 | 중 | unknown-only stub |

1~3 은 **OM-D/SEM 을 풀기 위해 지금 해야 하는 것**이고, 4~6 은 그 뒤 정확도를 끌어올리는 축이다.

---

## 1. 지금 방법론 (한 장)

```
알람 -> 접속 -> paused 프레임 1장
  -> template 선택   consensus(우선) / rcp(폴백),  OM|SEM 라우팅
  -> ensemble 매칭   C1/C2/C3 chamfer proposer -> RRF -> NCC rerank -> MIND(OM)/ECC(SEM) rerank
  -> key_visibility_gate
        act              -> closed-loop reposition(<=3회, tol 1% 폭) -> OK
        engineer_review  -> (fallback on) search-around 로 판별
        fallback_search  -> search-around(grid: PM 절대 zoom-out + footprint 격자 + odometry)
  -> 탐색 match -> correct_align_fail 재진입 -> reposition -> OK
  -> 그 외 -> cube 알림 + 엔지니어 watch + 녹화
```

판정의 단일 지점은 `key_visibility_gate` 하나다(`align/correction.py:175`). 점수는
`sel = 0.5·chamfer + 0.5·max(0, ncc)` 이고 임계는 Youden 보정값 **match 0.6053 / adjust 0.4727**
(`matching/engine.py:160`), 배율비 하한은 `MIN_CONFIRM_SCALE = 0.6`(`live_search.py:72`).

---

## 2. 제안 1 - 극성 정합 (OM-D)

### 2.1 무엇이 깨지는가 (계산으로 증명됨)

OM-D 는 OM 의 명암이 **반전된** 화면이다. 반전은 두 채널에 정반대로 작용한다.

- **chamfer 채널(C1/C2/C3)** 은 Canny edge -> distance transform 위에서 돈다. edge 는 gradient
  **크기**라 반전에 불변이다. 즉 proposer 는 정상 동작하고 **후보 좌표는 여전히 옳다.**
- **NCC rerank** 는 부호에 민감하다. 반전된 진짜 정답의 NCC 는 **≈ −1** 이고, 코드는
  `ncc_pos = max(0.0, ncc)`(`matching/engine.py:967`)로 음수를 0 으로 눌러 버린다.

따라서 **완벽하게 맞는 key 라도** 반전 상태에서는

```
sel = 0.5 · chamfer + 0.5 · 0  =  0.5 · chamfer  ≤ 0.5
```

`chamfer = 1.0` 인 이상적인 경우에도 sel 상한이 **0.5** 다. match 임계 0.6053 을 **구조적으로**
넘을 수 없고, adjust 임계 0.4727 대역(0.4727 ≤ sel ≤ 0.5)에 갇힌다. 그 뒤 게이트는

- `distinctive` 가 없으면 -> `engineer_review` (`correction.py:214`)
- 있으면 -> `act` 로 가지만 **좌표를 chamfer 절반의 정보로만 고른 것**이다.

오피스 OM 실측이 `score ≈ 0.53` 이었다는 사실과 이 상한(0.5)이 **수상하리만치 가깝다.**
OM-D 화면에서 찍힌 프레임이 섞여 있었다면 설명이 된다 - 지금은 확인할 방법이 없다.

### 2.2 더 중요한 것: 극성 불일치는 align fail 의 *원인*일 수 있다

recipe 가 OM-D 로 등록됐는데 화면이 OM 이면(또는 그 반대), 장비 자신의 align 루틴도 같은 반전
영상을 보게 된다. 즉 이것은 "우리 matcher 가 불리해지는 조건" 이 아니라 **align fail 자체의
한 실패 계급(failure class)** 일 수 있다. 그렇다면 올바른 조치는 점수를 우회하는 것이 아니라
**화면 모드를 recipe 에 맞추는 것**이고, 그것이 곧 보정이다.

### 2.3 지금 시스템이 OM-D 를 못 보는 이유

| 지점 | 현재 | 문제 |
|---|---|---|
| modality 판독 | PM 배율 문자열(104/210=OM, K접미사=SEM) | **배율로는 OM 과 OM-D 를 못 가른다.** 둘 다 같은 저배율 단이다 |
| template 라우팅 | `route_template`(`live_search.py:344`) 키가 `"OM"`/`"SEM"` 뿐 | `"OM-D"` 는 `"OM" in mode` 로 걸려 **밝은 OM template** 을 쓴다 |
| 자산 | `align_img_from_rcp/IMAP0001(OM) / IMAP0002(SEM)` | OM-D 전용 IMAP 이 없다 - 등록 이미지의 극성은 **파일만 봐서는 모른다** |

### 2.4 제안

**(a) 화면 모드를 읽는다 - 새 센서.**
오늘 만든 `monitor/manual_image_mode_change.py` 의 `read_mode()` 가 그 센서다. Image 콤보의
**화살표 왼쪽**을 OCR 로 읽어 `OM | OM-D | SEM` 을 단어 전체로 확정한다(부분 일치 금지 -
`om` 은 `om-d` 의 접두다). 이 함수를 `cycle.py` 가 사전 조건 판독으로 재사용한다.

**(b) recipe 의 등록 극성을 안다.**
두 경로 중 오피스에서 확인되는 쪽을 쓴다.
1. `cond.txt` 에 모드/극성 필드가 있는지 확인 (**오피스 확인 필요** - 있으면 이것이 정본)
2. 없으면 등록 이미지에서 추정: 배경/구조 밝기 분포. 다만 이것은 추정이므로 **판정이 아니라
   경고**로만 쓰고, 실제 판별은 (c) 의 probe 로 한다.

**(c) 극성 probe - 판별의 정공법.**
chamfer best 후보 자리에서 **NCC 의 부호**를 본다. 지금 코드는 `max(0, ncc)` 로 이 정보를
버리고 있다. 버리지 말고 기록한다.

```
ncc(best) ≈ +높음  -> 극성 일치
ncc(best) ≈ −높음  -> 극성 반전 (|ncc| 는 오히려 '잘 맞는다' 는 증거다)
|ncc| 낮음         -> 자리가 틀렸다 (극성 문제 아님)
```

이 세 갈래는 **한 번의 추가 계산도 필요 없다** - 이미 계산한 `ncc` 의 부호를 버리지만 않으면 된다.

**(d) 조치는 화면 쪽으로.**
probe 가 반전을 말하면 `manual_image_mode_change.change_image_mode` 와 같은 경로로
**화면 모드를 recipe 쪽으로 바꾸고 다시 캡처**한다. 그 뒤는 기존 경로 그대로(정상 sel 이 나온다).
바꿀 수 없으면(콤보 미검출/항목 미확인) 보정하지 않고 cube - **추측해서 누르지 않는다.**

**(e) 안전망: 극성 불변 매칭은 폴백으로만.**
`abs(ncc)` 로 바꾸는 1줄 수정은 유혹적이지만 **기본 경로로 삼지 않는다.** 이유 둘:
- 임계 0.6053/0.4727 은 `max(0,ncc)` 분포에 **묶여 있다**(`engine.py:158` 경고). 부호를 풀면
  배경 후보의 sel 도 같이 올라가 오탐이 늘고, 재캘리브레이션 없이는 두 임계가 무의미해진다.
- workflow_2 lab 과의 bit-parity 가 깨진다.
그러므로 극성 불변은 **`ALIGN_FAIL_POLARITY_INVARIANT=1` 킬스위치 뒤의 진단용**으로만 두고,
운영 기본은 (d) 의 화면 정합이다.

### 2.5 작업 목록

1. `read_mode` 를 `align/` 또는 `monitor/` 공용으로 옮기고 cycle 의 사전 판독에 배선
2. `AlignKeyMatchResult` 에 `best_ncc`(부호 유지) 필드 추가 - 기록만, 판정 불변
3. `route_template` 키를 3-state 로 확장(`OM-D` -> OM template + `inverted=True` 표식)
4. 보정 진입 전 `screen_mode != recipe_mode` 면 모드 변경 액추에이터 1회 -> 재캡처
5. 실패 시 새 status `image_mode_mismatch`(cube 로 나가고 `corrected` 아님)

### 2.6 오피스에서 가져올 숫자

- `[DIGEST] image_mode ...` 의 `current=` 와 `arrow_center=` (다음 실행부터 등록 영역으로 쓴다)
- OM-D 화면에서 잡은 paused 프레임의 `paused_match` history: `score`, `chamfer`, 그리고
  (2번 작업 후) `best_ncc` 의 **부호**
- 같은 recipe 를 OM 과 OM-D 에서 각각 한 장씩 - 이 한 쌍이 2.1 의 상한 0.5 를 실측으로 확정한다

---

## 3. 제안 2 - SEM search-around 는 frac 목표 zoom-out 이 전제다

### 3.1 OM 에서 통한 것이 SEM 에서 통하지 않는 이유

09-19 에 넣은 보폭은 **검출 footprint** 다.

```
stride = max(0.1 · frame,  (frame − template) · 0.9)
```

- **OM**: key 가 프레임의 10~20% -> footprint ≈ 0.8~0.9 FOV -> stride ≈ 0.75 FOV. 격자가 성립한다.
  그래서 OM 이 오피스에서 살아났다.
- **SEM**: key 가 프레임의 **80~100%** -> `frame − template ≈ 0` -> stride 가 **하한 0.1 FOV** 로
  떨어진다. 예산 10 셀이면 도달 반경이 **±0.2 FOV** 뿐이다. 즉 SEM 은 지금 격자를 아무리 돌려도
  실질적으로 제자리다.

이것은 셀 수를 늘려 풀 문제가 아니다(반경은 예산의 제곱근으로만 자란다). **zoom-out 으로 frac 을
내리는 것이 유일한 레버**다.

### 3.2 제안: zoom-out 목표를 `min_key_px` 가 아니라 **목표 frac** 으로 정한다

현재 `choose_zoom_out_mag(options, reg_mag, min(tw0,th0), config.min_key_px)` 는 key 가 너무
작아지지 않게만 본다(하한). 필요한 것은 **상한**이다 - key 가 프레임의 몇 %를 차지해야 격자가
성립하는가.

```
stride = FOV · (1 − frac) · 0.9
반경 R 을 덮는 데 필요한 셀 수  n² ≥ (2R / stride)²

목표 frac* 을 정하면   target_mag ≈ reg_mag · (frac* / frac_now)
```

권고 `frac* = 0.30` (stride ≈ 0.63 FOV). 예산 10 이면 3×3 격자로 ±1 FOV 를 덮는다.
`frac* ` 을 낮출수록 반경은 커지지만 key 가 작아져 chamfer 가 약해지므로, **`min_key_px` 하한과
`frac*` 상한 둘 다** 만족하는 단을 고른다. 둘이 충돌하면(= 드롭다운에 그런 단이 없으면)
degrade 로 빠지되 **이유를 남긴다**(`reason="no_mag_step_for_frac"`) - 지금처럼 조용히 좁은
격자를 도는 것보다 낫다.

### 3.3 SEM 고유 주의점

- **휠 금지, PM 절대 단만.** OM 은 `_om_to_registered_step` 이 휠로 등록 단을 맞추지만 SEM 의
  배율 범위에서 휠 한 칸은 FOV 를 크게 흔든다. SEM 은 반드시 드롭다운 절대값 + 판독.
- **ECC rerank 는 SEM 단독 순위**다(`key_type=='sem'`). ECC 도 상관 기반이라 극성에 민감하지만
  **SEM 에는 dark 모드가 없다**(2026-09-22 사용자 확인) - 극성 반전은 OM <-> OM-D 사이에서만
  일어나므로 제안 1 은 SEM 경로에 해당하지 않는다. SEM 은 제안 2(frac) 만 필요하다.
- **aperture problem**: SEM key 는 junction 이 유일점이고 주변 line/flat 이 점수를 희석한다.
  zoom-out 하면 junction 이 픽셀 몇 개로 줄어 이 문제가 **악화**된다. 그래서 `frac*` 를 0.1 처럼
  과하게 낮추면 안 된다 - 0.25~0.35 대역이 타협점이다.

### 3.4 오피스에서 가져올 숫자

`grid search: reg=… search=… scale=… stride=(x,y)FOV cells=…` 한 줄 + `template … 이 프레임을
거의 채운다` 경고의 유무. 이 두 줄이 3.2 의 `frac*` 를 실측으로 정한다.

---

## 4. 제안 3 - 극성·배율을 '사전 조건' 으로 승격

지금 보정은 **실패해 봐야** 조건이 틀렸다는 것을 안다. 09-17 에 넣은 `probe_align_dialog`
(클릭 전 다이얼로그 확인)와 같은 자리에, 같은 성격의 확인을 둘 더 넣는다.

| 사전 조건 | 판독 | 불일치 시 |
|---|---|---|
| 다이얼로그가 살아 있는가 | (구현됨) `ok_button.probe_align_dialog` | `align_fail_cleared` |
| **화면 image mode == recipe mode** | 제안 1(a) | 모드 변경 1회 -> 재캡처, 안 되면 cube |
| **PM 배율이 등록 배율과 같은 단인가** | 이미 읽고 있음(`grid_reg_mag`) | 등록 단으로 복귀 후 재캡처 |

이유: 셋 다 **매칭 결과로는 서로 구분되지 않는다**(전부 "점수가 안 나온다" 로 보인다).
사전에 갈라 두면 `[DIGEST]` 가 실패 계급을 직접 말해 주고, 잘못된 화면에서 reposition 을
누르는 일이 사라진다.

---

## 5. 제안 4 - 재등록 루프를 닫는다

측정된 사실: SEM 은 **rank-1 ≈ 0.5**(동전 던지기)이고, template-bank/RRF/ensemble 3종이 같은
벽에서 멈췄다(2026-06-25 결론: member-fusion 으로는 넘을 수 없다). 남은 레버는 matcher 가 아니라
**어떤 자리를 key 로 등록했는가**다. 재등록 커버리지는 pm=39 / 87% 로 이미 계산돼 있다.

지금도 `feasibility_check` 가 `reregister_recommended` 를 내고 `cycle.py` 가 `result.notes` 와
감사 로그에 남긴다. **없는 것은 그 신호가 사람에게 도달하는 경로**다.

제안: Episode 수집(`ALIGN_FAIL_EPISODE_COLLECT=1`)을 켜서 알람 1건마다 `second_ratio` /
`decision` / outcome 을 쌓고, 주 1회 `golden_reregister_report_cond.py` 형식의 worklist 를
엔지니어에게 넘긴다. 코드 작업은 거의 없고 **운영 절차**가 본체다. (첫 Episode = 티켓 18,
`docs/runbooks/first_recovery_episode.md`.)

---

## 6. 제안 5 - consensus 의 조용한 rcp 폴백을 막는다

열린 의심: `a33f8c4` 이후 consensus 가 rcp 와 크기/배율이 다른 S 이미지를 버리는데, 그 결과
pool 이 `min_s`(3) 미만이 되면 **경고 없이** rcp 로 떨어진다. 이것이 오피스에서 본
`match` -> `adjust` 강등의 후보 원인이다.

제안(소):
1. 드롭 사유를 세어 한 줄로 남긴다 - `[INFO] consensus[OM] n=8 -> kept=2 (size_mismatch=5, mag=1)`
2. `kept < min_s` 로 폴백할 때는 `[WARNING]` 으로 승격한다(지금은 조용하다)
3. `verify_consensus_path.py` 를 오피스에서 1회 돌려 경로 keying(eqp 무관) 확인

`consensus` 는 `rcp` 를 전제로 한다(crop 크기를 등록 key 에서 가져온다). rcp 가 0장이면
consensus 도 0장이고 경고조차 없으므로, `no_assets` 진단에서는 **항상 rcp 부터** 본다.

---

## 7. 제안 6 - Verification 을 stub 에서 올린다

지금 `corrected` 는 **자기보고**다. reposition 잔차가 tol 안에 들어왔고 OK 를 눌렀다는 뜻이지,
측정이 실제로 진행됐다는 증거가 아니다. `measurement_verification.py` 는 unknown-only stub 이고
Assist 패널 crop 만 남긴다.

제안: Assist 패널의 **분자(측정 카운터)** 판독을 오피스 캘리브레이션으로 확정하고, primary
Verification 3-상태를 실제로 채운다. 그러면
- `corrected` 인데 측정이 안 도는 경우를 잡는다(지금은 영원히 모른다)
- Outcome 파생(`workflow_4/playbook/outcome.py`)이 fallback(분자 per-read) 대신 primary 를 쓴다

이것이 완성되기 전까지 "보정 성공률" 이라는 숫자는 **우리가 누른 횟수**일 뿐이라는 점을 문서에
명시해 둔다.

---

## 8. 실행 순서와 정지점

```
1) 제안 1(a)+(c)  화면 모드 센서 + ncc 부호 기록      <- 코드 소, 위험 없음(기록만)
      정지점: OM / OM-D 같은 recipe 한 쌍의 score·best_ncc 부호를 오피스에서 받는다
2) 제안 1(d)      모드 불일치 시 모드 변경 -> 재캡처   <- 첫 실제 액추에이션
      정지점: [DIGEST] image_mode result=changed 1건
3) 제안 2         SEM frac 목표 zoom-out               <- SEM 격자가 성립하는 최소 조건
      정지점: stride=(x,y)FOV 가 0.5 FOV 이상으로 나오는 실행 1건
4) 제안 3         사전 조건 승격
5) 제안 5         consensus 드롭 사유 계측
6) 제안 4 / 6     재등록 운영 + Verification 캘리브레이션
```

1~2 를 3 보다 먼저 두는 이유: 극성이 틀린 채로 SEM 격자를 시험하면 **격자가 잘못된 건지 극성이
잘못된 건지 구분되지 않는다.** 한 번에 한 변수만 바꾼다.

---

## 9. 하지 않기로 한 것 (재제안 금지)

| 안 하는 것 | 이유 |
|---|---|
| matcher 채널/fusion 추가 | 2026-06-25 결론 - 3종이 같은 rank-1 ≈ 0.5 벽에서 멈췄다. 벽은 ranking 이 아니라 key 변별력 |
| `abs(ncc)` 를 기본값으로 | 임계 0.6053/0.4727 이 `max(0,ncc)` 분포에 묶여 있다. 킬스위치 뒤 진단용만 |
| 탐색 셀 수 늘리기로 SEM 해결 | 반경은 √예산으로만 자란다. frac 을 내리는 것이 레버 |
| cond box ROI / C4 채널 포팅 | ADR 0003/0004 기각 |
| Qwen 으로 PaddleOCR 대체 | 확인 게이트 false confirm + 좌표계 불일치(2026-09-03 기각) |
| 첫 글자 anchor 프롬프트 | 같은 글자로 시작하는 라벨이 많아 이웃을 짚는다 |

---

## 10. 한 줄 요약

**"점수를 올리려 하지 말고, 점수가 나올 수 있는 화면을 먼저 만든다."**
극성(제안 1)과 배율/frac(제안 2)은 matcher 의 입력 조건이지 matcher 의 성능이 아니다.
그 둘을 사전 조건으로 승격(제안 3)한 뒤에야 재등록·검증(제안 4·6)의 숫자가 의미를 갖는다.
