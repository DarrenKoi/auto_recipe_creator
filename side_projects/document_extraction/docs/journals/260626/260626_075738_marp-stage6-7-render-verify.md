# Marp roundtrip Stage 6(렌더) + Stage 7(SSIM 검증/자동 강등) 구현

- **날짜**: 2026-06-26
- **영역**: `side_projects/document_extraction/marp/`
- **방식**: TDD (red-green, vertical slice) — 순수 결정 로직 우선, I/O 는 graceful degrade

## 배경

`marp_roundtrip_design.md` 는 슬라이드 캡처 -> 구조 추출 -> Marp -> 재렌더
라운드트립을 정의한다. 직전까지 **Stage 5(evidence -> Marp Markdown 생성)** 만
구현되어 있었고(`generate.py`/`build_marp.py`), Stage 6(marp-cli 렌더) + Stage 7
(재렌더 SSIM 검증 + 저충실도 자동 강등)은 "office TODO" 로 남아 있었다.

이번 작업은 그 두 단계를 **Mac 에서 빌드 가능한 형태**로 구현했다. 핵심은
설계를 "순수 결정 로직(집에서 검증) + 얇은 I/O 껍데기(office/marp 필요 시)"로
나눈 것. 오피스 데이터 없이도 파이프라인 전체가 동작·검증된다.

## 한 일

### Stage 6 — `marp/render.py` (deck.md -> 렌더 산출물)
- `build_render_args(deck, out_dir, fmt)` — **순수**. png/pptx/pdf/html 별 marp-cli
  인자. png 은 `--images png` 로 슬라이드별 이미지. 알 수 없는 fmt 는 `ValueError`.
- `resolve_marp_command()` — PATH 의 `marp` 우선, 없으면 `npx --yes
  @marp-team/marp-cli`, 둘 다 없으면 `None`(graceful).
- `render_deck(...)` — subprocess 호출, `RenderResult(available, ok, outputs, stderr)`.
  marp 바이너리 부재(FileNotFoundError)/timeout 모두 예외 없이 graceful degrade.
  산출물은 포맷 확장자로 필터(소스 deck.md 제외 — e2e 에서 발견한 wart 수정).

### Stage 7 — `marp/verify.py` (재렌더 vs 원본 SSIM + 자동 강등)
- `ssim(a, b)` — **순수 numpy** (Wang et al. 2004, 전역 단일 윈도우). **skimage
  비의존** (office 패키지 추가 마찰 회피). 동일 -> 1.0, 반전 -> 낮음.
- `slide_fidelity(orig, rendered)` — 렌더 해상도가 달라도 원본 크기로 리사이즈 후
  SSIM(marp 출력 DPI 무관). PIL 우선, 없으면 numpy 최근접 폴백.
- `flag_low_fidelity(scores, threshold=0.90)` — floor 미만 슬라이드 인덱스. floor
  동률은 통과. **DEFAULT_SSIM_FLOOR=0.90** (사용자 확정).
- `plan_downgrade(...)` -> `DowngradePlan` — **안전망 결정**. 네이티브 데이터표로
  렌더된 차트(crop 가용)는 래스터로 부분 강등(편집성 보존), 살릴 게 없으면 슬라이드
  전체를 원본 캡처 래스터로 강등(최후 안전망, 사용자 확정).
- `whole_slide_marp(capture)` — Marp `![bg fit]` 전면 배경(전체 래스터).
- `apply_downgrade_plans(...)` — 강등 계획 반영해 보정 deck 재작성(`generate.py`
  의 emitter 재사용 -> Stage 5 와 일관).
- `verify_and_downgrade(...)` — **I/O 오케스트레이터**. render -> 슬라이드별 SSIM
  -> flag -> plan -> 보정 deck 작성. marp/이미지 부재면 `rendered=False` graceful.

### 패키지/문서
- `marp/__init__.py` — render/verify 심볼 export, "Stage 5만" 문구 -> "Stage 5+6+7".
- `build_marp.py` — Stage 6/7 TODO 주석을 실제 함수 호출 가이드로 교체.
- `docs/marp_roundtrip_design.md` — 구현 상태 블록 + §8 갱신(Stage 6/7 구현됨).

## 검증

- 스모크 테스트 **25개 통과**: `test_render_smoke`(6) + `test_verify_smoke`(11)
  + 기존 `test_marp_smoke`(8). 순수 함수 전수 + I/O graceful-degrade 경로 커버.
- **Mac e2e (실제 marp 렌더)**: `npx @marp-team/marp-cli` 로 합성 2-슬라이드 deck
  -> `deck.001.png`/`deck.002.png` 렌더 성공. 그 PNG 를 `verify_and_downgrade` 에
  자기비교 입력으로 -> `rendered=True scores=[1.0, 1.0] flagged=[]`. Stage 5->6->7
  전 구간이 오피스 데이터 없이 실제로 동작함을 확인.

## 설계 메모 / 결정

- **floor=0.90, 최후 안전망=슬라이드 전체 래스터** — 사용자 확정(편집성 ↔ 시각
  충실 trade-off 에서 "균형" 정책에 부합; 한 슬라이드만 편집성 포기).
- **skimage 도입 안 함** — SSIM 을 numpy 로 직접. 의존성 표면 평탄 유지.
- 텍스트/레이아웃 mismatch 의 OCR 재합성 분기는 design 의 풍부한 경로지만 이번엔
  전체 래스터 안전망으로 보증(office 후속).

## 남은 일 (office)

1. 9장 미니 벤치로 실제 캡처 대상 `verify_and_downgrade` 돌려 **floor=0.90 보정** +
   강등 빈도/유형 측정(`benchmark_plan.md` 재활용).
2. crop 분기 임계(Stage 3) 튜닝 — 어느 type 을 선제적으로 래스터로 강등할지.
3. Marp 커스텀 테마 1종(시각 보정).
4. Kimi Platform 엔드포인트 + base64 이미지 1콜 스모크(Stage 4 합성 품질).
