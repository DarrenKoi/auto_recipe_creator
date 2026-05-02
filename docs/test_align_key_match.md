# Align Key 매칭 — 합성 데이터 기반 smoke test 계획

본 문서는 `docs/search_align_key.md` §6 의 두 번째 단계 — **classical matcher 의
오프라인 프로토타입** — 을 위한 테스트 설계와 구성 파일을 정리한다. 사무실
실장비 데이터 없이 Mac 에서 알고리즘 파이프라인의 정상 동작과 positive /
negative 점수 분리를 검증하는 것이 목적이다.

## 목적

- `align_key_matcher.compute_align_key_score()` 가 end-to-end 로 돌아가는지
  확인.
- positive (template 가 frame 에 embedding) 과 negative (template 가 frame 에
  없음) 사이에 깨끗한 점수 분리가 발생하는지 확인.
- §7.3 의 `0.6 × chamfer + 0.4 × orb_inlier_ratio` 합성 점수가 의도대로
  분류 (match / adjust / low) 를 만들어내는지 확인.
- 디버그 산출물 (overlay JPG + result JSON + summary JSON) 이 운영 단계에서
  필요한 형태로 저장되는지 확인.

이 테스트는 **임계값 캘리브레이션이 아니다**. cold-start 임계값
(`MATCH_THRESHOLD = 0.75`, `ADJUST_THRESHOLD = 0.55`) 이 합성 데이터에서
합리적으로 동작하는지를 sanity check 할 뿐이다. 실제 임계값 튜닝은 사무실
데이터 수집 (§6 단계 1) 이후 진행한다.

## 본 테스트에서 사용/생성하는 파일

### 새로 작성

| 파일 | 역할 |
|------|------|
| `poc/workflow_2/__init__.py` | 패키지 진입점 — `WORKFLOW_2_DIR`, `DEBUG_IMAGE_DIR`, `LOG_DIR` 상수 |
| `poc/workflow_2/align_key_matcher.py` | 매칭 엔진 — `AlignKeyTemplate`, `AlignKeyMatchResult`, `build_template`, `compute_chamfer_score`, `compute_orb_inlier_ratio`, `compute_align_key_score`, `save_overlay_jpeg` |
| `poc/workflow_2/test_align_key_match.py` | 합성 데이터 생성 + 10 케이스 실행 + 디버그 산출물 저장 |
| `poc/workflow_2/search_align_key.py` | search loop 오케스트레이션 — `AlignKeySearchConfig`, `AlignKeySearchState`, `AlignKeySearchOutcome`, `search_align_key()`. 사무실 (RCS 캡처+pywinauto 이동) 과 Mac (가상 wafer+mock 이동) 양쪽에서 동일 흐름으로 돌아가도록 `capture_fn`/`move_stage_fn` injection. `__main__` 에 가상 wafer 데모 포함 |
| `docs/test_align_key_match.md` | 본 문서 (테스트 계획 + 파일 목록) |

### 의존하는 기존 파일 (수정하지 않음)

| 파일 | 사용 이유 |
|------|----------|
| `docs/search_align_key.md` | §3 알고리즘, §7 운영 결정사항 (특히 §7.3 임계값, §7.6 인터페이스) |
| `poc/workflow_2/__init__.py` | `DEBUG_IMAGE_DIR` 상수 import |
| `pyproject.toml` | `opencv-contrib-python>=4.8.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `python-pptx>=0.6.21` 의존성 추가 |

### 산출물 (실행 시마다 생성)

```
poc/workflow_2/debug_images/align_search/<YYMMDD_HHMMSS>/
    case_01_pos_identity_frame.jpg
    case_01_pos_identity_template.jpg
    case_01_pos_identity_overlay.jpg     # 매칭 박스 + 점수 텍스트가 그려진 결과
    case_01_pos_identity_result.json     # score / decision / best_xy / 등
    ...                                  # 케이스 02~10 동일 4종 산출물
    summary.json                         # 전체 요약 (n_pass, 케이스별 결과)
```

## 합성 데이터 설계

### 템플릿 (`make_synthetic_template`)

128×128 grayscale 패턴. 회색 배경 (~120) 위에 어두운 (~40) 굵은 stroke 으로
fiducial 을 그린다. 가장자리는 Gaussian blur (σ=0.6), 표면에 mild Gaussian
noise (σ=4) 를 추가하여 ORB 가 keypoint 를 잡을 수 있도록 한다.

지원 종류:

- `cross` — 굵은 `+` 마크.
- `box` — 3-box-in-box (중첩 사각형 3개) + **비대칭 orientation cluster**.
  rotational symmetry 를 깨고 ORB 가 distinctive keypoint 를 다수 검출할 수
  있도록 좌상단에 3-dot + L-tick, 우하단에 1-dot + tick 을 배치.
- `checker` — 4×4 alternating 체커보드 (negative case 8 의 "다른 패턴" 에
  사용).

본 테스트에서는 `box` 를 기본 템플릿으로 사용한다 (사용자 도메인 노트
"큰 박스 3~4개" 와 일치).

### 프레임 배경 (`make_wafer_background`)

512×768 grayscale. 16× 다운샘플 후 INTER_CUBIC 으로 업샘플한 random Gaussian
field 로 저주파 밝기 변화 (~60–200) 를 시뮬레이션. feature-sparse 한 웨이퍼
영역을 흉내낸다.

### 변환 파이프라인 (`embed_pattern`)

템플릿을 frame 에 합성할 때 적용되는 변환 (positive 케이스에서):

1. 스케일 (`cv2.resize` linear).
2. 회전 (`cv2.warpAffine`, `BORDER_REPLICATE`).
3. 콘트라스트 + 밝기 — 평균 주변에서 contrast 곱한 뒤 brightness offset.
4. 배경의 random 위치에 합성 (margin 20px 보장).

추가 distortion:

- `add_charging_gradient(max_delta)` — 좌우/상하 선형 밝기 ramp.
- `add_random_blobs(count)` — random 위치/크기/진폭의 Gaussian blob (negative
  case 7 에서 ORB false-positive 유도용).
- `add_gaussian_noise(sigma)` — 모든 케이스 마지막에 적용.

### 재현성

`np.random.default_rng(42)` 로 RNG 고정. 케이스별 `embed_pattern(rng_seed=...)`
도 고정 → 매 실행마다 동일한 합성 데이터.

## 테스트 케이스 (10개)

### Positives — template 가 frame 안에 존재 (5개)

| # | 라벨 | 회전 | 스케일 | 밝기 | 콘트라스트 | charging | 노이즈 | 의도 |
|---|------|------|--------|------|----------|---------|--------|------|
| 1 | `pos_identity` | 0° | 1.0 | 0 | 1.0 | ✗ | σ=4 | 이상적 조건 — 알고리즘 자체 동작 확인 |
| 2 | `pos_mild` | 1° | 1.0 | +10 | 0.95 | ✗ | σ=8 | 통상적 SEM 변동 |
| 3 | `pos_scale_down` | 0° | 0.85 | -5 | 1.0 | ✗ | σ=8 | multi-scale fallback 검증 (더 작은 스케일) |
| 4 | `pos_scale_up` | 0° | 1.2 | +5 | 1.05 | ✗ | σ=8 | multi-scale fallback 검증 (더 큰 스케일) |
| 5 | `pos_hard` | 2° | 0.9 | -10 | 0.85 | ✓ (Δ=25) | σ=12 | 다중 distortion 스택 — 보수적이지만 현실적 worst case |

### Negatives — template 가 frame 에 없음 (5개)

| # | 라벨 | 구성 | 의도 |
|---|------|------|------|
| 6 | `neg_plain_wafer` | 배경만 | 가장 간단한 negative |
| 7 | `neg_random_blobs` | 배경 + 18개 random Gaussian blob | ORB false-positive 함정 |
| 8 | `neg_wrong_pattern` | checker 패턴이 frame 에 있고 box 템플릿으로 매칭 | 다른 fiducial → 점수 낮아야 |
| 9 | `neg_strong_charging` | 배경 + 강한 charging (Δ=60) | 큰 밝기 ramp 만으로는 매칭 X |
| 10 | `neg_out_of_scale` | 템플릿이 scale=2.0 으로 frame 에 있음 | DEFAULT_SCALES (0.7~1.4) 범위 밖 → 매칭 실패해야 |

## Pass criteria

- **Positive**: `decision in {"match", "adjust"}` (즉 `score >= ADJUST_THRESHOLD = 0.55`)
  AND `best_xy` 가 ground truth 로부터 20px 이내.
- **Negative**: `decision == "low"` (즉 `score < ADJUST_THRESHOLD`).

## 실행 방법

```bash
uv run python poc/workflow_2/test_align_key_match.py
```

CLI 인자 없음 (프로젝트 컨벤션). 출력 디렉토리 경로는 stdout 에 출력된다.

## 최근 실행 결과 (참고용)

```
[INFO] 템플릿 생성 완료: shape=(128, 128), edges=1778px
[PASS] case=01           pos_identity  score=0.850  chamfer=0.973  orb=0.667  decision=match   ok (err=0.0px)
[PASS] case=02               pos_mild  score=0.873  chamfer=0.956  orb=0.750  decision=match   ok (err=0.0px)
[PASS] case=03         pos_scale_down  score=0.754  chamfer=0.954  orb=0.455  decision=match   ok (err=1.4px)
[PASS] case=04           pos_scale_up  score=0.804  chamfer=0.941  orb=0.600  decision=match   ok (err=1.4px)
[PASS] case=05               pos_hard  score=0.773  chamfer=0.878  orb=0.615  decision=match   ok (err=1.4px)
[PASS] case=06        neg_plain_wafer  score=0.402  chamfer=0.670  orb=0.000  decision=low     ok (low)
[PASS] case=07       neg_random_blobs  score=0.442  chamfer=0.737  orb=0.000  decision=low     ok (low)
[PASS] case=08      neg_wrong_pattern  score=0.458  chamfer=0.763  orb=0.000  decision=low     ok (low)
[PASS] case=09    neg_strong_charging  score=0.401  chamfer=0.668  orb=0.000  decision=low     ok (low)
[PASS] case=10       neg_out_of_scale  score=0.456  chamfer=0.760  orb=0.000  decision=low     ok (low)
[INFO] Prototype validated: 10/10 cases passed
```

점수 분포 요약:

- Positives: 0.754 ~ 0.873 (모두 `match`)
- Negatives: 0.401 ~ 0.458 (모두 `low`)
- Margin: ≈ 0.30 — 합성 데이터 기준 충분히 분리됨.

## 합성 데이터의 한계 (실제 SEM 데이터로 반드시 재검증)

본 smoke test 가 통과했다고 알고리즘이 production-ready 라는 뜻이 **아니다**.
다음은 합성 데이터로 커버되지 않는 영역이다.

1. **실제 SEM 노이즈 통계** — Poisson noise (electron count) + detector
   electronic noise. 본 테스트는 Gaussian 노이즈만 사용.
2. **실제 charging 패턴** — 본 테스트의 선형 ramp 는 단순화. 실제 charging
   은 wafer 표면 도전성에 따라 spatial pattern 이 다양.
3. **실제 align key 패턴 다양성** — 사무실 실 recipe 의 align key 가 단순
   box-in-box 보다 복잡할 수 있음. ORB 의 keypoint 다양성이 합성보다 풍부할
   수도 있고 빈약할 수도 있음.
4. **해상도 정규화** — 본 테스트는 `nm_per_pixel` 이 둘 다 None 이므로
   §7.2 Case B (multi-scale fallback) 만 검증. Case A (서버 메타 제공 시
   단일 스케일) 는 실제 메타가 들어왔을 때 별도 검증 필요.
5. **임계값 (0.75 / 0.55)** — cold-start 보수값. 실 데이터에서 false
   positive / false negative ROC 로 recipe 별 재캘리브레이션 필요 (§7.4).

따라서 본 smoke test 통과 후 다음 단계는 다음과 같다.

- 사무실에서 실 (recipe align key, SEM FOV) 쌍 10~30 개 + 부정 FOV 10~30 개
  수집.
- 동일 매칭 엔진을 실 데이터에 돌려 점수 분포 확인.
- 분포가 충분히 분리되면 임계값 캘리브레이션 후 `monitor_align_fail.py` 통합.

## 변경된 의존성

`pyproject.toml` 의 `dependencies` 에 다음을 추가했다.

```toml
"opencv-contrib-python>=4.8.0",
"numpy>=1.24.0",
"pandas>=2.0.0",
"python-pptx>=0.6.21",
```

이전에는 `requirements.txt` 에만 있어 `uv sync` 후 누락되던 패키지들. 이번 통합으로 `uv sync` 만으로 본 테스트와 기존 `build_report_pptx.py`, `office_align_fail_alarm.py` 가 모두 동작한다.

## 참고 문서

- `docs/search_align_key.md` — 알고리즘 설계, 운영 결정사항 (정본).
- `docs/setup_vlms/05-resource-comparison-vs-kimi-k2.md` — VLM 리소스 비교
  (본 테스트와 직접 관련 없으나 같은 워크스트림).
- `CLAUDE.md` — 프로젝트 코딩 컨벤션 (Korean docstrings, no argparse, etc.)
