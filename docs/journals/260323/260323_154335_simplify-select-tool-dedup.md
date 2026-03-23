# Simplify: select_tool.py 코드 리뷰 및 중복 제거

**날짜**: 2026-03-23 15:43
**트리거**: `/simplify @poc/work/select_tool.py`

---

## 1. 진행 사항

- `poc/work/select_tool.py` 경로가 더 이상 존재하지 않음을 확인 → 실제 파일은 `poc/work2/select_tool.py` (완전히 다른 리라이트 버전)
- 3개 병렬 리뷰 에이전트 실행 (Code Reuse / Code Quality / Code Efficiency)
- `poc/work2/select_tool.py` 대상 통합 리뷰 실행
- 중복 헬퍼 3개를 `poc/work2/util/` 로 추출하여 6개 파일에서 제거
- 코드 품질 이슈 수정 (불필요한 별칭, 취약한 로직, 과도한 반환값)

## 2. 수정 내용

### 신규 공유 헬퍼 추출 (3개)

| 헬퍼 | 추출 대상 | 제거 원본 파일 |
|---|---|---|
| `normalize_lines()` | `util/json_utils.py` | `select_tool.py`, `login_rcs_paddleocr.py`, `login_rcs_got_ocr.py` |
| `image_point_to_screen()` | `util/window_utils.py` | `select_tool.py`, `action_login.py`, `view_list_tab_rcs.py` |
| `crop_image()` | `util/image_utils.py` | `select_tool.py`, `ui_venus_mai_locator.py` |

### 수정 파일 목록 (10개)

- `poc/work2/util/json_utils.py` — `normalize_lines()` 추가
- `poc/work2/util/window_utils.py` — `image_point_to_screen()` 추가
- `poc/work2/util/image_utils.py` — `crop_image()` 추가
- `poc/work2/util/__init__.py` — 신규 3개 헬퍼 export 추가
- `poc/work2/select_tool.py` — 로컬 중복 3개 제거, `COMPONENT_NAME` 별칭 제거, `target_visible` 로직 수정 (`normalized_joined` → `bool(matched_lines)`), `_run_list_ocr` 반환값 축소, 불필요한 full-window WebP 저장 제거
- `poc/work2/action_login.py` — 로컬 `_image_point_to_screen` 제거, import 전환
- `poc/work2/view_list_tab_rcs.py` — 로컬 `_image_point_to_screen` 제거, import 전환
- `poc/work2/login_rcs_paddleocr.py` — 로컬 `_normalize_lines` 제거, import 전환
- `poc/work2/login_rcs_got_ocr.py` — 로컬 `_normalize_lines` 제거, import 전환
- `poc/work2/ui_venus_mai_locator.py` — 로컬 `_crop_image` 제거, import 전환, 변수명 충돌 해소 (`crop_image` → `cropped`)

### 결과

- **-79 라인** (149 삭제, 70 추가)
- `target_visible` 로직의 cross-line 매칭 false positive 제거

## 3. 다음 단계

- Windows 환경에서 전체 파이프라인 통합 테스트 (select_tool → OCR → VLM → 더블클릭)
- `login_rcs_paddleocr.py`의 `normalize_lines` 호출 시 `max_items=80` 인자가 생략됨 → 기본값 120으로 변경됨. 의도된 동작인지 확인 필요
- `login_rcs_got_ocr.py`도 마찬가지로 `max_items=80` → 120으로 변경됨

## 4. 메모리 업데이트

변경 없음
