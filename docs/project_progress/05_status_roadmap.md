# 05. 현황 & 로드맵 (Status & Roadmap)

> 목적: 완료/대기 항목을 명확히 구분하고, production 활성화까지 남은 단계와 확장 방향·리스크를 정리한다.

근거: `poc/workflow_3/README.md`(이전 체크리스트), `docs/journals/260618/`.

## 1. 현황 한눈에

| 영역 | 상태 | 비고 |
|------|------|------|
| VLM 배포·운영 (deploy_vlms + Flask proxy) | ✅ 운영 중 | H200×2, 5 모델 |
| workflow_1 GUI 자동화 + CCTV PoC | ✅ 검증 완료(동결) | production은 wf3로 이전 |
| workflow_2 CV 평가 벤치 (4 driver) | ✅ 활성 | 오피스 golden 데이터 실측 대기 |
| OM/SEM modality-split 평가 | ✅ 코드 완료 | split verdict + Youden delta + failure-mode |
| Job 2 box-crop 실험 | ❌ 기각 (ADR 0005) | 오피스 실데이터 열세, production 미포팅 |
| 재등록 우선순위 리포트 (Phase 1) | 🟡 캘리브레이션 중 | STRONG tier 게이트 조정 중, Phase 2=E-frame |
| workflow_3 루프 골격·녹화·알림 | ✅ 코드 완료 | — |
| workflow_3 primary 보정 + fallback search | ✅ 코드 완료 | dry-run 게이트 |
| box-crop cond-aware 템플릿 | ✅ 코드 완료 | TDD 검증 |
| consensus 라우팅 (코드) | ✅ 코드 완료 | downloader 없으면 자동 rcp |
| **consensus 라이브 보정 활성화** | 🟡 대기 | `office_success_downloader` 구현 필요 |
| engineer-done 감지 | 🟡 캘리브레이션 중 | 기본 off |
| SEM-box 검출 + zoom/click 캘리브레이션 | 🟡 진행 중 | 모델별 landmark·배율 비율 |
| align_images 루트 이전 (wf1→wf3) | 🟡 진행 중 | MES 출력 경로 먼저 변경 |
| pilot 실보정(actuation) | 🟡 대기 | dry-run 후 단일 장비부터 |

## 2. 오피스 PC 이전 체크리스트 (요약)

1. **office_* 복사** — `office_align_fail_alarm.py`, `office_rich_notify.py`를 `poc/workflow_3/monitor/`로.
2. **`office_success_downloader.py` 신규 작성** (사용자 담당, gitignore) — `make_success_downloader()`
   팩토리로 `SuccessDownloader` Protocol 구현. **consensus 라이브 보정 활성화 게이트.**
3. **`office_rcp_msr_downloader.py`** (선택) — MES가 `ALIGN_IMAGES_DIR` 트리에 직접 적재 못 하는 환경에서만.
4. **import sweep** — 경고 없이 로드되는지 확인.
5. **SAFE_MODE=1 dry-run** — 클릭 0회, journal/알림/manifest만 확인.
6. **record-only 패리티** — `ALIGN_FAIL_CORRECTION=0`로 접속→캡처→닫기 + 상시 녹화 재현.
7. **보정 dry-run** — `ALIGN_FAIL_CORRECTION=1`(DRY_RUN 유지): 좌표 계산·overlay·cube 알림까지, 클릭은 로그만.
8. **캘리브레이션** (모델별):
   - SEM panel landmark crop + meta.json → `templates/sem_panel_landmarks/<model_id>/`
   - 더블클릭 recenter 이동량, wheel 1단계↔배율(`ALIGN_SEM_ZOOM_SCROLL_DY`) 측정
   - `read_mode` 실제 판독(모드 라벨 OCR/픽셀 휴리스틱)
9. **pilot actuation** — 단일 장비에서 `ALIGN_FAIL_CORRECTION_DRY_RUN=0`.

## 3. workflow_2 → 정확도 확정 다음 단계

1. office `<HISTORY_ROOT>/<class>/<recipe>/events/`에 class·recipe·modality별 최근 S 8~10장 rolling 적재.
2. `golden_eval_config.py`에 경로 채우고 `golden_combined_eval_cond.py` 실행 → `[DIGEST]` 한 줄 회신.
3. **판정 1**: OM rank1 ≪ SEM이고 실패 유형이 다르면 → modality별 Canny/Youden/proposer 레버 A/B.
4. **판정 2**: consensus 점수가 S 수에 따라 단조 증가하면 "consensus 많을수록 좋음" 확정.
5. **판정 3**: `edge_ncc`로 rcp-only rank1 상승 + 회귀 없으면 production 포팅.

## 4. 확장성 (Scalability)

- **모델 확장**: Flask proxy에 `.env` + 모듈 1개로 6번째 모델 추가(GPU 1에 ~50 GiB 헤드룸).
  multi-GPU data/tensor parallel로 처리량 확장.
- **장비·recipe 일반화**: consensus 이력 풀이 `<class>/<recipe>` 키(장비 무관) → 신규 장비 onboarding 시
  같은 recipe의 학습 데이터 자동 공유, 코드 변경 최소.
- **데이터 자산화**: 상시 녹화 + `recording_filter`로 엔지니어 수동 조작을 interaction timeline으로 추출 →
  향후 모방학습/원인 자동 분류 학습 데이터.
- **재등록·template bank 축**: SEM에서는 ROI/box-crop 탐색 범위 축소가 구조적으로 무효임을 확인
  (distractor가 align key **내부**의 periodic 구조라 frame 밖이 아님 — Job 2 box-crop 기각, ADR 0005).
  남은 어려운 케이스는 chronic-ambiguous align key의 **재등록 권고**(`golden_reregister_report_cond.py`)와
  recipe별 template bank로 풀어간다.
- **검증된 변경만 포팅**: workflow_2 golden 검증 → workflow_3 bit-parity 포팅으로 회귀 통제.

## 5. 리스크 & 완화

| 리스크 | 완화 | 상태 |
|--------|------|------|
| office downloader 부재 | 자동 비활성 → 기존 rcp 경로 동일 동작 | ✅ |
| cold consensus cache | bounded sync 8s 후 rcp 폴백 | ✅ |
| S 이미지 부족(< min_s) | modality별 rcp 폴백 | ✅ |
| 정확도 수치가 벤치 기준 | 오피스 실데이터 `[DIGEST]`로 확정 예정 | 🟡 |
| SEM-box zoom/click 미캘리브레이션 | dry-run 게이트(실클릭 차단) | 🟡 |
| engineer-done 오탐 | N>5 연속 2회 확인 + CV gate | ✅ |
| 실보정 사고 | 2단계 게이트(SAFE_MODE + DRY_RUN), pilot 단일 장비 | ✅ |

## 6. 다음 주 우선순위

1. `office_success_downloader` 구현 → consensus 라이브 보정 활성화.
2. 오피스 캘리브레이션(zoom/click 좌표, engineer-done) 완료.
3. `golden_combined_eval_cond.py` 오피스 실행 → OM/SEM 층화·라우팅 정확도 확정.
4. `golden_reregister_report_cond.py` STRONG tier 게이트 오피스 캘리브레이션 마무리 → 재등록 권고 리스트 산출.
5. pilot 장비 1대 dry-run → 실보정 단계적 전환.
