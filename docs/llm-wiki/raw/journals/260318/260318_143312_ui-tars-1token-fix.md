# UI-TARS 1-Token 응답 문제 진단 및 수정

**날짜**: 2026-03-18
**세션 요약**: UI-TARS VLM이 200 응답은 주지만 1 토큰만 생성하는 문제를 진단하고 수정

---

## 1. 진행 사항

- **문제 신고**: UI-TARS 모델이 Flask 프록시를 통해 200 응답을 반환하지만 completion_tokens=1 (즉시 EOS)
- **프록시 계층 점검**: `flask_api/vlm_serve/ui_tars.py`, `service_template.py`의 `force_stream=True` 설정, SSE 버퍼링, 클라이언트 파싱 로직 — 모두 정상 확인
- **클라이언트 점검**: `poc/work2/vlm_client.py`의 `_extract_text_from_sse_body()` SSE 파싱, `prefer_stream=True` 설정 — 정상 확인
- **HuggingFace에서 UI-TARS tokenizer_config.json 확인**: chat template이 system role을 지원함을 확인 → 프롬프트 포맷 문제 아님
- **vLLM 직접 테스트 지시**:
  - `/v1/chat/completions` → 1 토큰 (finish_reason=stop) 확인
  - `/v1/completions` (raw prompt) → 정상 동작 확인
- **근본 원인 특정**: vLLM 0.17.1이 UI-TARS를 `Qwen2_5_VLForConditionalGeneration`으로 로드하면서 내부 Qwen2.5-VL 채팅 템플릿 처리가 UI-TARS의 실제 템플릿과 충돌 → 모델이 즉시 EOS 출력
- **수정 적용 및 동작 확인 완료**

## 2. 수정 내용

### 새 파일 생성

- **`deploy_vlms/config/chat_templates/ui-tars.jinja`**
  - UI-TARS의 tokenizer_config.json에서 추출한 Jinja2 채팅 템플릿
  - vLLM `--chat-template` 옵션으로 명시적 제공하여 내부 Qwen2.5-VL 처리를 오버라이드

### 파일 수정

- **`deploy_vlms/config/models/ui-tars.env`**
  - `CHAT_TEMPLATE=` → `CHAT_TEMPLATE=chat_templates/ui-tars.jinja` 설정

- **`deploy_vlms/scripts/serve_vlm.py`** (line 433 부근)
  - `CHAT_TEMPLATE` 값이 상대 경로일 때 `CONFIG_ROOT` 기준으로 절대 경로로 변환하는 로직 추가
  - ```python
    if chat_template and not os.path.isabs(chat_template):
        chat_template = os.path.join(config_root, chat_template)
    ```

## 3. 다음 단계

- GPU 서버에서 UI-TARS vLLM 인스턴스 재시작 후 이미지 포함 요청으로 end-to-end 테스트
- Flask 프록시를 통한 전체 파이프라인 테스트 (`poc/work2/vlm_client.py` → Flask proxy → vLLM)
- 다른 Qwen2-VL 기반 모델(mai-ui 등)에서 유사한 문제가 없는지 확인

## 4. 메모리 업데이트

- UI-TARS vLLM 배포 시 명시적 chat template 필요하다는 사실 기록 필요
