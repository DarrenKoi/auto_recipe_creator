# PaddleOCR Usage Guide For Workflow 1

작성일: 2026-06-07

이 노트는 workflow_1 에서 PaddleOCR-VL 을 **GUI 자동화의 "확인(confirm)" 도구** 로만 쓰는 이유와
방법을 정리한다. 원칙은 workflow_2 의 같은 가이드(`poc/workflow_2/docs/study/paddleOCR/README.md`)와
동일하며, 여기서는 workflow_1 의 쓰임(로그인 confirm, Tool 행 verify)에 맞춰 간추린다.

## 핵심 규칙

**PaddleOCR 는 OCR 증거 엔진이지, 최종 GUI 행위자가 아니다.**

workflow_1 의 클릭 파이프라인:

```
ui-venus(coarse) → mai-ui(fine) 가 좌표를 "결정"
PaddleOCR-VL 이 그 자리 텍스트를 "확인"
확인 안 되면 클릭하지 않음
```

(프로젝트 메모리: feedback_click_pipeline_coarse_fine_confirm)

## 왜 전체 화면 OCR 을 금지하나

- PaddleOCR-VL 은 **문서 파서** 다. 전체 RCS 스크린샷(여러 패널·아이콘·반복 라벨)에 직접 돌리면
  **환각 텍스트** 가 다발로 쏟아진다. (메모리: project_paddleocr_vl_screenshot_hallucination)
- 정상 경로는 **layout → crop → recognize** 다. 즉 VLM/CV 로 영역을 좁힌 뒤 **그 crop 에만** OCR 을 돌린다.
- 현재 라우트(`paddleocr-vl-1.5`)는 OpenAI 호환 chat 의 VLM 컴포넌트만 노출할 뿐, 완전한 PaddleOCR
  문서 파이프라인이 아니다. 따라서 **task-prompted OCR sidecar** 로 취급한다.

## 태스크 프롬프트

| 태스크 | 프롬프트 | 용도 |
|---|---|---|
| 평문 OCR | `OCR:` | 고유 문구 존재 확인 |
| Spotting | `Spotting:` | 텍스트 + bbox (행 위치) |

`temperature=0.0`, 작은 crop 은 `max_tokens` 128~512 면 충분.

## workflow_1 use cases

### A. 로그인 요소 confirm
- mai-ui 가 클릭점을 찾으면, 그 주변 crop 의 텍스트를 `OCR:` 로 읽어 기대 라벨(예: "User ID",
  "Log In")과 근접·일치하는지 확인한 뒤에만 클릭한다.
- 짧은 라벨(`OK`)만으로 확정하지 말 것 — 화면 여러 곳에 있어 증거가 약하다. 가능하면 긴 고유 문구를 쓴다.

### B. Tool 행 verify (fallback)
- VLM 이 제안한 행에 좁은 horizontal strip 을 만들고 `Spotting:` 1회.
- `ocr_spotting.parse_spotting_items()` 로 정규화 → `tool_name_match` 의 confusion-map 정규화 매칭.
- **List 전체나 4개 큰 crop 에 루프로 Spotting 돌리지 말 것** — 과거에 느린 데다 결과도 garbage 였다.
  (메모리: project_rcs_tool_list_layout)

### C. debug 증거
- 실패 run 에 사람이 읽을 OCR 텍스트를 아티팩트로 저장(자동화 명령 아님).

## 검증 체크리스트 (클릭 전)

- 대상이 텍스트/표 중심인가?
- 입력을 한 영역으로 crop 했는가?
- 확인용 **긴 고유 문구** 가 있는가?
- `max_tokens` 캡 / `temperature=0.0` 인가?
- raw OCR 출력을 저장하는가?
- 박스를 정규화·경계검사했는가?
- OCR 실패 시 **no-click** 경로가 있는가?
- 결과가 증거/게이트일 뿐 단독 권위가 아닌가?

하나라도 No 면 PaddleOCR 가 1차 도구로 부적절.

## 레포 패턴

```python
from poc.workflow_1.prompts import build_ocr_assist_prompt, build_spotting_prompt
from poc.workflow_1.vlm_client import Workflow1VLMClient
from poc.workflow_1.ocr_spotting import parse_spotting_items

client = Workflow1VLMClient(service_slug="paddleocr-vl-1.5", timeout_sec=120.0, log_name="workflow_1_ocr")

# 존재 확인
sys_msg, user_text = build_ocr_assist_prompt(width=0, height=0)   # "OCR:"
resp = client.chat_with_image_b64(image_b64=crop_b64, system_message=sys_msg,
                                  user_text=user_text, image_mime="image/webp",
                                  temperature=0.0, max_tokens=256)

# 위치 포함
sys_msg, user_text = build_spotting_prompt()                       # "Spotting:"
resp = client.chat_with_image_path(image_path=strip_crop_path, system_message=sys_msg,
                                   user_text=user_text, image_mime="image/webp",
                                   temperature=0.0, max_tokens=512)
items = parse_spotting_items(resp.text)
```

## 참고

- 상세 운영/실험 가이드는 workflow_2 판: `poc/workflow_2/docs/study/paddleOCR/README.md`.
- 관련 해설: `../cv/ocr_spotting_intro.md`, `../sem_box/rcs_list_tab_layout.md`.
