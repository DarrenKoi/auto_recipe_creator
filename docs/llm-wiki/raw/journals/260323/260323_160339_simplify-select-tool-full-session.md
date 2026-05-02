# Simplify select_tool.py: 코드 리뷰, 중복 제거, OCR 토큰 한도 수정

**날짜**: 2026-03-23 16:03
**트리거**: `/simplify @poc/work/select_tool.py` → 후속 OCR 토큰 이슈 대응

---

## 1. 진행 사항

### Phase 1: 코드 리뷰 및 중복 제거
- `poc/work/select_tool.py` 경로가 이미 삭제되었음을 확인 (`poc/work/` 폴더 자체가 없음)
- 실제 파일 `poc/work2/select_tool.py` (531줄) 대상으로 리뷰 진행
- 3개 병렬 리뷰 에이전트 (Code Reuse / Code Quality / Code Efficiency) 실행
- 중복 헬퍼 3개를 `poc/work2/util/`로 추출, 6개 파일에서 로컬 복사본 제거
- 코드 품질 이슈 수정 (불필요한 별칭, 취약한 로직, 과도한 반환값)

### Phase 2: OCR 토큰 한도 증가
- 사용자 보고: Tool List OCR 시 하단 tool ID가 누락됨
- `OCR_MAX_TOKENS`를 1024 → 4096으로 증가, `normalize_lines max_items`를 120 → 300으로 증가

### Phase 3: PaddleOCR-VL 모델 컨텍스트 확장
- 사용자 보고: `max_tokens=4096` 요청 시 모델 컨텍스트 초과 에러 발생
- 원인 분석: `MAX_MODEL_LEN=4096` (vLLM 배포 설정)이 입력+출력 합산 한도
- Tool List 화면에는 tool ID 외에도 IP, location, model name, status 등 다량의 텍스트가 있어 OCR 출력이 큼
- `MAX_MODEL_LEN`을 4096 → 8192로 증가 (0.9B 모델이라 VRAM 부담 미미)

## 2. 수정 내용

### 신규 공유 헬퍼 추출 (3개)

| 헬퍼 | 추출 위치 | 제거된 원본 파일 |
|---|---|---|
| `normalize_lines()` | `util/json_utils.py` | `select_tool.py`, `login_rcs_paddleocr.py`, `login_rcs_got_ocr.py` |
| `image_point_to_screen()` | `util/window_utils.py` | `select_tool.py`, `action_login.py`, `view_list_tab_rcs.py` |
| `crop_image()` | `util/image_utils.py` | `select_tool.py`, `ui_venus_mai_locator.py` |

### 수정 파일 목록 (11개)

- `poc/work2/util/json_utils.py` — `normalize_lines()` 추가
- `poc/work2/util/window_utils.py` — `image_point_to_screen()` 추가
- `poc/work2/util/image_utils.py` — `crop_image()` 추가
- `poc/work2/util/__init__.py` — 신규 3개 헬퍼 export 추가
- `poc/work2/select_tool.py` — 로컬 중복 3개 제거, `COMPONENT_NAME` 별칭 제거, `target_visible` 로직 수정, `_run_list_ocr` 반환값 축소, full-window WebP 저장 제거, `OCR_MAX_TOKENS` 1024→4096, `normalize_lines` max_items 120→300
- `poc/work2/action_login.py` — 로컬 `_image_point_to_screen` 제거
- `poc/work2/view_list_tab_rcs.py` — 로컬 `_image_point_to_screen` 제거
- `poc/work2/login_rcs_paddleocr.py` — 로컬 `_normalize_lines` 제거
- `poc/work2/login_rcs_got_ocr.py` — 로컬 `_normalize_lines` 제거
- `poc/work2/ui_venus_mai_locator.py` — 로컬 `_crop_image` 제거, 변수명 충돌 해소 (`crop_image` → `cropped`)
- `deploy_vlms/config/models/paddleocr-vl-1.5.env` — `MAX_MODEL_LEN` 4096→8192

### 커밋 3건

1. `refactor: extract shared helpers to util/ and simplify select_tool.py` — 중복 제거 (-79줄)
2. `fix: increase OCR max tokens and line limit for long tool lists` — OCR_MAX_TOKENS, max_items 증가
3. `fix: increase PaddleOCR-VL context to 8192 for long tool lists` — 배포 설정 변경

## 3. 다음 단계

- **PaddleOCR-VL vLLM 서비스 재시작 필요** — `MAX_MODEL_LEN=8192` 적용을 위해
- Windows 환경에서 select_tool 파이프라인 통합 테스트 (OCR → VLM → 더블클릭)
- `login_rcs_paddleocr.py`와 `login_rcs_got_ocr.py`에서 `normalize_lines` 호출 시 기존 `max_items=80`이 기본값 120으로 변경됨 — 의도된 동작인지 확인 필요
- `GPU_MEMORY_UTILIZATION=0.10`이 8192 컨텍스트에서 충분한지 확인 필요 (부족하면 0.15로 증가)

## 4. 메모리 업데이트

변경 없음
