## 1. 진행 사항
- PaddleOCR-VL-1.5 프롬프트를 `"OCR:"` 태스크 키워드 방식으로 전면 단순화했다. 0.9B 모델이 학습된 6가지 태스크 키워드(`OCR:`, `Table Recognition:` 등)에 맞춰 시스템 메시지와 복잡한 JSON 지시를 모두 제거했다.
- `poc/work2/__init__.py`에 모델 기반 디버그 이미지 디렉터리 유틸리티(`_slugify_model_name()`, `resolve_debug_model_name()`, `debug_image_dir()`, `debug_image_path()`)를 추가했다. 모델명을 폴더명에 안전한 slug로 변환해 하위 디렉터리를 자동 생성한다.
- `poc/work2/rcs_utils.py`의 `debug_image_path()`를 `poc.work2.debug_image_path()`에 위임하도록 변경하고 `model_name` 파라미터를 추가했다.
- 모든 디버그 이미지 저장 코드(`automate_rcs_login.py`, `check_tool_screen.py`, `click_rcs_view_mode.py`, `reading_check.py`)를 모델별 하위 폴더 구조로 전환했다.
- `poc/work2/pipeline_ocr.py`에서 JSON 파싱(`extract_json`) 의존을 제거하고, plain text 줄 단위 파싱(`_parse_ocr_lines()`)과 포커스 워드 매칭(`_match_focus_words()`)으로 교체했다.
- `poc/work/vlm_openai_client.py`의 `OpenAICompatibleVLMClient`와 `LangChainOpenAICompatibleVLMClient` 모두에서 `system_message`가 빈 문자열이면 messages 리스트에서 제외하도록 수정했다.
- `poc/work2/reading_check.py`의 `build_paddleocr_reading_prompt()`도 `"OCR:"` 키워드만 반환하도록 단순화하고, 결과 요약 출력 로직을 개선했다.

## 2. 수정 내용
- 변경 파일: `poc/work2/__init__.py`
  모델명 slug 변환과 디버그 이미지 경로 유틸리티 함수 4개를 추가하고 `__all__`을 업데이트했다.
- 변경 파일: `poc/work2/prompts/ocr_assist.py`
  JSON 프롬프트 전체를 삭제하고 `return "", "OCR:"` 로 교체했다. 모듈 docstring에 PaddleOCR-VL-1.5 태스크 키워드 설명을 추가했다.
- 변경 파일: `poc/work2/pipeline_ocr.py`
  `extract_json` 임포트 제거, `_normalize_texts()` → `_parse_ocr_lines()` 교체, `_match_focus_words()` 신규 추가.
- 변경 파일: `poc/work2/rcs_utils.py`
  `debug_image_path()`가 `poc.work2.debug_image_path()`에 위임하도록 변경, `model_name` 파라미터 추가.
- 변경 파일: `poc/work2/automate_rcs_login.py`
  `_run_benchmark()` 내 디버그 이미지 저장에 `model_name` 파라미터를 전달하도록 수정. 수동 safe name 변환 제거.
- 변경 파일: `poc/work2/check_tool_screen.py`
  `debug_image_path` 임포트 추가, `pipeline_config` 로딩을 앞으로 이동, 디버그 이미지와 VLM overlay 이미지 모두 모델 기반 경로로 전환.
- 변경 파일: `poc/work2/click_rcs_view_mode.py`
  `debug_image_path()` 호출에 `model_name=VLM_MODEL` 추가.
- 변경 파일: `poc/work2/reading_check.py`
  `build_paddleocr_reading_prompt()` 단순화, 소스 JPEG/전송 WebP를 공유 디버그 디렉터리에 저장, 각 모델 결과물을 개별 모델 디렉터리에 저장, 요약 출력 개선.
- 변경 파일: `poc/work/vlm_openai_client.py`
  `OpenAICompatibleVLMClient.send_chat_image()`와 `LangChainOpenAICompatibleVLMClient.send_chat_image()` 모두 `system_message` 조건부 포함으로 수정.

## 3. 다음 단계
- 사무실 Windows 환경에서 `reading_check.py`를 실행해 PaddleOCR-VL-1.5가 `"OCR:"` 키워드에 올바르게 plain text 응답하는지 확인한다.
- 디버그 이미지가 모델별 하위 폴더(`debug_images/<model-slug>/`)에 정상 저장되는지 실제 실행 후 디렉터리 구조를 점검한다.
- `pipeline_ocr.py`의 `_parse_ocr_lines()`가 실제 PaddleOCR-VL 응답에서 텍스트를 정확히 추출하는지 검증한다.
- `automate_rcs_login.py` → `click_rcs_view_mode.py` → `check_tool_screen.py` 전체 파이프라인을 연결해 모델 기반 디버그 폴더에 각 단계 이미지가 분리 저장되는지 확인한다.

## 4. 메모리 업데이트
- 변경 없음
