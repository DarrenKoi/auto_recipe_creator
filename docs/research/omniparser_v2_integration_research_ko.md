# 리서치: OmniParser V2 통합 — OCR 사이드카 대체 및 UI 요소 감지 강화

## 배경

현재 `poc/work2/` 파이프라인은 2단계 구조로 동작한다:

```
스크린샷 → PaddleOCR-VL (텍스트 힌트) → ui-venus (좌표 + 액션 결정)
```

PaddleOCR-VL 사이드카는 **텍스트만** 추출하며, 인터랙티브 요소의 위치나 타입 정보는 제공하지 않는다. OmniParser V2 (Microsoft, 2025년 2월)는 **텍스트 + 인터랙티브 요소 바운딩 박스 + 아이콘 기능 설명**을 한 번에 추출하는 복합 파이프라인이다. 본 리서치는 OmniParser V2를 기존 OCR 사이드카 위치에 통합하여 UI 요소 감지를 강화할 수 있는지를 평가한다.

---

## 1. OmniParser V2 아키텍처

OmniParser V2는 단일 모델이 아니라 **3개의 독립 모델을 순차 체이닝**하는 복합 파이프라인이다:

### 내부 구성 모델

| 단계 | 모델 | 파라미터 | 가중치 크기 | 라이선스 |
|------|------|----------|-------------|----------|
| **1. OCR 텍스트 감지** | EasyOCR (기본) 또는 PaddleOCR (옵션) | 경량 | — | Apache-2.0 |
| **2. 인터랙티브 요소 감지** | Fine-tuned YOLOv8 Nano | 3.2M | 40.6 MB | **AGPL-3.0** |
| **3. 아이콘 캡셔닝** | Fine-tuned Florence-2-base-ft | ~230M | 1.08 GB | MIT |

**처리 흐름:**

```
스크린샷 입력
    │
    ├──→ [1] EasyOCR/PaddleOCR → 텍스트 바운딩 박스 + 텍스트 내용
    │
    ├──→ [2] YOLOv8n (1280×1280) → 인터랙티브 요소 바운딩 박스 (단일 클래스)
    │         │
    │         └──→ [3] Florence-2 (각 아이콘 64×64 크롭) → 기능 캡션
    │
    └──→ 오버랩 제거 + 병합 → 구조화된 요소 리스트 + SoM 이미지 출력
```

### YOLO 감지기 세부사항

- **클래스 수: 1** — 단순히 "아이콘/인터랙티브 영역"만 감지. 버튼/체크박스/드롭다운 등 시맨틱 분류는 하지 않음
- **학습 데이터**: 인기 웹 URL 10만 개에서 추출한 63,641장의 스크린샷, DOM 트리로 바운딩 박스 라벨링
- **학습 설정**: 20 에폭, batch 64, GPU 4장
- **이미지 크기**: 1280×1280으로 리사이즈 후 추론
- **감지 임계값**: confidence 0.05 (기본값), IoU 0.7

### Florence-2 캡셔닝 세부사항

- **아키텍처**: DaViT (Dual Attention Vision Transformer) 인코더 + 6-layer Transformer 디코더
- **입력**: 각 감지된 아이콘을 **64×64로 크롭/리사이즈** → 배치 추론
- **출력**: 최대 20 토큰의 기능 설명 (예: "A gear icon for settings")
- **V2 핵심 개선**: V1에서 더 큰 해상도를 사용하던 것을 64×64로 줄여 **레이턴시 60% 감소**
- **한계**: 64×64 크롭에서 주변 컨텍스트가 제거되므로 아이콘 의미 해석이 부정확할 수 있음

---

## 2. 입출력 포맷

### 입력

FastAPI 서버 엔드포인트 (`POST /parse/`):

```json
{
  "base64_image": "<base64 인코딩 이미지>"
}
```

HuggingFace 추론 엔드포인트 변형:

```json
{
  "inputs": {
    "image": "<URL 또는 base64>",
    "image_size": {"w": 1920, "h": 1080},
    "bbox_threshold": 0.05,
    "iou_threshold": 0.7
  }
}
```

### 출력

```json
{
  "som_image_base64": "<번호 매긴 바운딩 박스가 오버레이된 PNG>",
  "parsed_content_list": [
    {
      "type": "text",
      "bbox": [0.12, 0.34, 0.25, 0.38],
      "interactivity": false,
      "content": "Settings",
      "source": "box_ocr_content_ocr"
    },
    {
      "type": "icon",
      "bbox": [0.08, 0.33, 0.11, 0.39],
      "interactivity": true,
      "content": "A gear icon for settings",
      "source": "box_yolo_content_yolo"
    },
    {
      "type": "icon",
      "bbox": [0.40, 0.10, 0.45, 0.15],
      "interactivity": true,
      "content": "Search",
      "source": "box_yolo_content_ocr"
    }
  ],
  "latency": 0.65
}
```

**바운딩 박스 포맷**: `[x1, y1, x2, y2]` — **비율 좌표** (0.0~1.0, 이미지 크기로 정규화)

**source 필드 해석**:
- `box_ocr_content_ocr`: OCR로 감지 + OCR로 텍스트 추출
- `box_yolo_content_yolo`: YOLO로 감지 + Florence-2로 캡션 생성
- `box_yolo_content_ocr`: YOLO로 감지 + 해당 영역 안의 OCR 텍스트 상속

**SoM (Set-of-Mark) 이미지**: 원본 스크린샷에 색상 바운딩 박스와 순번 ID (0, 1, 2, ...)가 오버레이된 이미지. 다운스트림 에이전트는 "Box ID 3을 클릭하라"는 식으로 참조.

---

## 3. 벤치마크 성능

| 벤치마크 | OmniParser V2 + GPT-4o | GPT-4o 단독 | 비고 |
|----------|-------------------------|-------------|------|
| ScreenSpot (Mobile) | 93.9% | 22.6% | 모바일 UI |
| ScreenSpot (Desktop) | 91.3% | 20.2% | 데스크톱 UI |
| ScreenSpot (Web) | 81.3% | 9.2% | 웹 UI |
| **ScreenSpot Pro** | **39.6** | **0.8** | 고해상도 + 작은 아이콘 |
| Mind2Web | 39.4% | — | 웹 자동화 |
| AITW | 57.7% | 53.0% (w/히스토리) | Android |

**핵심 수치**: ScreenSpot Pro에서 GPT-4o 단독 0.8 → OmniParser 추가 시 39.6. 구조화된 파싱이 VLM의 GUI 그라운딩을 극적으로 향상시킴.

---

## 4. 배포 방법 및 GPU 요구사항

### 배포 옵션

| 방식 | 설명 | 적합성 |
|------|------|--------|
| **FastAPI 서버** | `omniparserserver.py` — uvicorn 기반, `POST /parse/`, `GET /probe/` | **권장** — 기존 Flask proxy와 유사한 패턴 |
| Gradio 데모 | 웹 UI, 포트 7861, 임계값 조정 가능 | 테스트/데모용 |
| HuggingFace 추론 | `EndpointHandler` 패턴 | 클라우드 배포 시 |

**vLLM 불가**: OmniParser는 표준 LLM/VLM이 아닌 복합 파이프라인이므로 vLLM으로 서빙할 수 없음. 자체 FastAPI 서버가 필요.

**Docker**: OmniParser 자체에는 전용 Dockerfile이 없음. OmniTool (Windows 11 VM 환경)은 Docker 사용하지만 별개.

### GPU VRAM 요구사항

| 구성 요소 | VRAM | 비고 |
|-----------|------|------|
| YOLOv8n | ~0.5 GB | 3.2M 파라미터, 매우 경량 |
| Florence-2-base-ft | ~2–4 GB | 230M 파라미터 (float16), batch_size=128일 때 ~4 GB |
| EasyOCR | ~0.5 GB | CPU에서도 실행 가능 |
| **합계** | **~4–6 GB** | 단일 GPU에서 다른 모델과 공존 가능 |

**현재 GPU 배치와의 호환성**: 24 GB GPU 기준 OmniParser가 ~4-6 GB 사용 시 18-20 GB가 다른 VLM 모델에 활용 가능. 기존 ui-venus (8B)나 PaddleOCR-VL과 같은 GPU에 배치하기에 충분한 여유.

### 추론 레이턴시

| GPU | 프레임당 레이턴시 |
|-----|-------------------|
| A100 | ~0.6초 |
| RTX 4090 | ~0.8초 |

**레이턴시 분해**:
1. OCR 추출: <100ms
2. YOLO 아이콘 감지: <10ms (YOLOv8n은 매우 빠름)
3. 오버랩 제거/박스 병합: 무시 가능 (CPU)
4. **Florence-2 배치 캡셔닝**: 병목 지점 — 감지된 아이콘 수에 비례. 64×64 리사이즈로 개선됨

---

## 5. 현재 파이프라인과의 비교

### 현재: PaddleOCR-VL 사이드카

```
pipeline_ocr.py:
  스크린샷 → PaddleOCR-VL → OCRHintResult(texts, focus_hits)
                                  │
                                  ▼
  build_ocr_extra_instructions() → "OCR observed these visible texts: ..."
                                       │
                                       ▼
  primary VLM (ui-venus) → 최종 좌표/액션 결정
```

**출력**: 텍스트 줄 리스트 + focus word 매칭만 제공.

### 제안: OmniParser V2 사이드카

```
pipeline_omniparser.py (신규):
  스크린샷 → OmniParser V2 → OmniParserResult(elements, som_image)
                                  │
                                  ▼
  build_omniparser_extra_instructions() →
    "Detected interactable elements:
     [0] Button at (120,45)-(200,65) label='Login' interactable=true
     [1] Input at (120,80)-(300,100) label='Server' interactable=true
     [2] Text at (50,45)-(100,65) content='Server:' interactable=false
     ..."
                                       │
                                       ▼
  primary VLM (ui-venus) → 최종 좌표/액션 결정 (바운딩 박스 힌트 활용)
```

**출력**: 바운딩 박스 + 인터랙티브 여부 + 텍스트/아이콘 설명 — 질적으로 다른 수준의 힌트.

### 기능 비교

| 기능 | PaddleOCR-VL (현재) | OmniParser V2 (제안) |
|------|---------------------|----------------------|
| 텍스트 추출 | O | O (EasyOCR/PaddleOCR 내장) |
| 바운딩 박스 | X | O (비율 좌표) |
| 인터랙티브 요소 감지 | X | O (YOLO 기반) |
| 아이콘 기능 설명 | X | O (Florence-2) |
| SoM 이미지 출력 | X | O (번호 매긴 오버레이) |
| 레이턴시 | PaddleOCR-VL 1회 호출 | ~0.6-0.8초 (3모델 체인) |
| VRAM | ~2 GB (VLM 서빙) | ~4-6 GB (3모델 동시) |
| 라이선스 | Apache-2.0 | **AGPL-3.0** (YOLO 컴포넌트) |

---

## 6. 통합 설계

### A. Flask proxy 서비스 등록

`flask_api/vlm_serve/config.py`에 새 서비스 추가:

```python
VLMServiceEntry("omniparser", "OmniParser-V2", "omniparser-v2", 8006, enabled=True),
```

단, OmniParser는 OpenAI-compatible API가 아닌 자체 FastAPI (`POST /parse/`)를 사용하므로, 기존 vLLM proxy 패턴과 다른 전용 프록시 블루프린트가 필요하다.

**선택지 A**: OmniParser FastAPI 서버를 8006 포트로 직접 노출하고, `pipeline_omniparser.py`에서 직접 호출.
**선택지 B**: Flask proxy에 `/api/vlm_serve/omniparser/parse` 라우트 추가, OmniParser 8006을 업스트림으로 프록시.

**권장**: 선택지 A — OmniParser는 이미 FastAPI 서버이므로 Flask 프록시를 한 번 더 거칠 이유가 없음. health 체크만 기존 시스템에 통합.

### B. pipeline_omniparser.py 설계

`pipeline_ocr.py`를 대체하는 새 모듈. 핵심 함수:

```python
@dataclass(frozen=True)
class OmniParserElement:
    """OmniParser가 감지한 단일 UI 요소."""
    element_id: int           # SoM 번호
    element_type: str         # "text" | "icon"
    bbox: tuple[float, ...]   # (x1, y1, x2, y2) 비율 좌표
    interactivity: bool       # 인터랙티브 여부
    content: str              # 텍스트 내용 또는 아이콘 캡션
    source: str               # "box_ocr_content_ocr" 등

@dataclass(frozen=True)
class OmniParserResult:
    """OmniParser V2 파싱 결과."""
    elements: tuple[OmniParserElement, ...]
    som_image_b64: str        # SoM 오버레이 이미지 (디버그용)
    focus_hits: tuple[str, ...]  # focus_words 매칭 결과
    latency_sec: float

def collect_omniparser_result(...) -> OmniParserResult | None:
    """OmniParser V2 호출 후 결과 반환."""

def build_omniparser_extra_instructions(result: OmniParserResult | None) -> tuple[str, ...]:
    """OmniParser 결과를 primary VLM용 프롬프트 힌트로 변환."""
```

### C. 프롬프트 힌트 변환 전략

현재 OCR 힌트 포맷:
```
OCR observed these visible texts: Server, User ID, Password, Log In.
OCR confirmed these target texts are visible: Log In.
```

OmniParser 힌트 포맷 (제안):
```
Screen parsing detected 12 UI elements:
  Interactable: [0] icon (120,45)-(200,65) "Login button"
  Interactable: [1] icon (120,80)-(300,100) "Text input field"
  Interactable: [2] icon (120,115)-(300,135) "Text input field"
  Text: [3] (50,45)-(110,65) "Server"
  Text: [4] (50,80)-(110,100) "User ID"
  Text: [5] (50,115)-(110,135) "Password"
Target texts confirmed: "Log In" at element [0].
Use element bounding boxes as reference. Final coordinates must come from actual pixels.
```

**핵심 원칙 유지**: "Use OCR only as auxiliary context. Final coordinates and layout judgment must still come from the actual pixels." — OmniParser 힌트도 동일하게 보조 정보로만 사용. ui-venus가 최종 판단.

### D. SHARED_PIPELINE_SETTINGS 확장

`flask_vlm.py`에 OmniParser 설정 추가:

```python
SHARED_PIPELINE_SETTINGS: dict[str, str | bool] = {
    # ... 기존 설정 ...

    # OmniParser V2 사이드카 (OCR 파이프라인 대체)
    "omniparser_enabled": False,          # True로 변경 시 OCR 대신 OmniParser 사용
    "omniparser_api_url": "",             # 예: "http://gpu-server:8006"
    "omniparser_bbox_threshold": 0.05,    # YOLO 감지 임계값
    "omniparser_iou_threshold": 0.7,      # NMS IoU 임계값
}
```

**전환 전략**: `omniparser_enabled=True`이면 `pipeline_omniparser.py` 사용, `False`이면 기존 `pipeline_ocr.py` 유지. 점진적 마이그레이션 가능.

### E. VLMScreenAnalyzer 수정

`vlm_screen_analysis.py`의 `_build_pipeline_instructions()`에서 설정에 따라 OCR 또는 OmniParser 호출:

```python
def _build_pipeline_instructions(self, ...):
    if self.pipeline_config.get("omniparser_enabled"):
        omni_result = collect_omniparser_result(...)
        instructions.extend(build_omniparser_extra_instructions(omni_result))
    else:
        ocr_result = collect_ocr_hint_result(...)
        instructions.extend(build_ocr_extra_instructions(ocr_result))
```

---

## 7. SoM (Set-of-Mark) 이미지 활용 옵션

OmniParser의 가장 강력한 부가 가치 중 하나는 **SoM 이미지** 출력이다:

```
원본 스크린샷 → OmniParser → [번호 매긴 바운딩 박스 오버레이 이미지]
```

기존 연구에서 SoM 오버레이는 그라운딩 정확도를 **70.5% → 93.8%** 로 향상시킨 결과가 있다 (기존 리서치 문서 참조).

**활용 방안**:

| 방안 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **A. 텍스트 힌트만** | `parsed_content_list`를 텍스트로 변환 | 토큰 효율적, 기존 파이프라인과 호환 | 시각적 그라운딩 정보 미활용 |
| **B. SoM 이미지 전송** | 원본 대신 SoM 이미지를 primary VLM에 전송 | 최대 그라운딩 정확도 | VLM이 번호 매긴 박스를 이해해야 함 |
| **C. 둘 다** | SoM 이미지 + 텍스트 힌트를 함께 전송 | 시각적 + 텍스트 보강 동시 | 토큰 사용량 증가 |

**권장**: 1차 통합에서는 **방안 A** (텍스트 힌트만)로 시작. `pipeline_ocr.py` 대체 수준의 최소 변경. 이후 방안 B/C를 벤치마크로 비교.

---

## 8. 알려진 한계 및 리스크

### 기술적 한계

| 한계 | 영향 | 완화 방안 |
|------|------|-----------|
| **학습 데이터가 웹 UI 중심** | RCS 같은 산업용 레거시 UI에서 감지 정확도 저하 예상 | 기존 ui-venus를 최종 판단자로 유지; OmniParser는 힌트만 제공 |
| **단일 클래스 감지** | 버튼/입력필드/체크박스 구분 불가 | Florence-2 캡션에 의존하되, 캡셔닝 정확도도 64×64 크롭 한계 있음 |
| **인터랙티비티 하드코딩** | YOLO 감지 = interactable, OCR 감지 = non-interactable (학습된 분류 아님) | RCS에서는 대부분의 버튼이 YOLO로 감지되므로 실용적으로 작동할 가능성 |
| **동일 요소 반복** | 같은 아이콘이 여러 개일 때 구분 불가 | 텍스트 라벨과의 근접성으로 구분 (OCR 텍스트 + YOLO 아이콘 병합 로직 활용) |
| **OCR 언어 제한** | EasyOCR 기본 영어만 (`['en']`). 한국어 UI 요소 있을 경우 설정 변경 필요 | RCS는 주로 영어 UI이므로 큰 문제 아님 |
| **PaddleOCR GPU 충돌** | PaddleOCR 옵션 사용 시 `use_gpu=False` (PyTorch CUDA 충돌 방지) | EasyOCR 기본값 사용 권장 |

### AGPL-3.0 라이선스 리스크

OmniParser의 YOLO 컴포넌트(YOLOv8 + Ultralytics 런타임)는 **AGPL-3.0** 라이선스다:

- **AGPL 13조 (원격 네트워크 상호작용)**: 네트워크 서비스로 제공 시 **전체 서버 애플리케이션의 소스 코드를 공개**해야 함
- 내부 폐쇄망에서만 사용하더라도 네트워크를 통해 접근하는 사용자에게 소스 코드를 제공해야 하는 의무 발생
- **상용 라이선스**: Ultralytics에서 별도 상용 라이선스 구매 가능

**완화 선택지**:

| 선택지 | 설명 | 난이도 |
|--------|------|--------|
| Ultralytics 상용 라이선스 구매 | 가장 단순 | 비용 발생 |
| YOLO를 RT-DETR 또는 Grounding DINO로 대체 | 감지 모델만 교체 (Apache-2.0) | 중간 — fine-tuning 필요할 수 있음 |
| OmniParser를 참고하여 자체 파이프라인 구축 | 아키텍처만 참고, 코드/가중치 미사용 | 높음 |
| 내부 연구/평가 목적으로만 사용 | AGPL이 적용되지 않는 범위 | 프로덕션 배포 불가 |

**권장**: 1차 PoC 단계에서는 내부 평가 목적으로 사용. 프로덕션 배포 시 Ultralytics 상용 라이선스 또는 YOLO 대체 모델 검토.

---

## 9. 배포 계획

### Phase 1: 평가 (1-2주)

1. GPU 서버에 OmniParser V2 FastAPI 서버 배포 (포트 8006)
   ```bash
   # 가중치 다운로드
   huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model

   # 서버 실행
   python -m omniparserserver \
     --som_model_path weights/icon_detect/model.pt \
     --caption_model_name florence2 \
     --caption_model_path weights/icon_caption_florence \
     --device cuda \
     --BOX_TRESHOLD 0.05 \
     --port 8006
   ```

2. RCS 스크린샷 10-20장으로 감지 품질 평가
   - 현재 PaddleOCR-VL 텍스트 추출 vs OmniParser 요소 리스트 비교
   - RCS 레거시 UI 컨트롤에서의 YOLO 감지율 측정
   - Florence-2 아이콘 캡션이 RCS 버튼/아이콘을 올바르게 설명하는지 확인

3. 평가 지표:
   - `element_detection_rate`: 실제 인터랙티브 요소 중 YOLO가 감지한 비율
   - `false_positive_rate`: 인터랙티브 아닌 것을 인터랙티브로 감지한 비율
   - `caption_accuracy`: Florence-2 캡션이 실제 기능과 일치하는 비율
   - `text_recall`: OmniParser 내장 OCR vs PaddleOCR-VL 텍스트 추출 재현율

### Phase 2: 통합 (1-2주)

1. `poc/work2/pipeline_omniparser.py` 구현
2. `flask_vlm.py`에 `omniparser_enabled` 설정 추가
3. `vlm_screen_analysis.py`에서 분기 로직 추가
4. 기존 `automate_rcs_login.py`, `click_rcs_view_mode.py`에서 A/B 비교

### Phase 3: 최적화 (선택)

1. SoM 이미지를 primary VLM에 직접 전송하는 방안 B/C 벤치마크
2. Florence-2 캡셔닝 결과를 RCS 도메인에 맞게 fine-tune (선택)
3. YOLO 감지기를 RCS UI 학습 데이터로 fine-tune (선택)

---

## 10. 결론 및 권장사항

### OmniParser V2는 OCR 사이드카의 **상위 호환**이다

현재 PaddleOCR-VL이 제공하는 텍스트 추출은 OmniParser V2에 포함되어 있으며, 추가로 바운딩 박스 + 인터랙티비티 + 아이콘 캡션을 제공한다. 질적으로 다른 수준의 힌트를 primary VLM에 전달할 수 있다.

### 그러나 **ui-venus를 대체하지는 않는다**

OmniParser V2는 파서(전처리기)이지 에이전트(의사결정자)가 아니다. 구조화된 요소 리스트를 생성할 뿐, 최종 좌표 결정과 액션 선택은 여전히 VLM(ui-venus)이 담당해야 한다. 특히 RCS 같은 니치 산업용 UI에서는 전용 GUI VLM이 범용 파서보다 우수할 가능성이 높다.

### 권장 아키텍처

```
                 ┌─────────────────────────────────────────┐
                 │         Enhanced Pipeline (Phase 2)      │
                 │                                          │
스크린샷 ──────→ │  OmniParser V2 (GPU, ~4-6 GB)           │
                 │    ├─ EasyOCR → 텍스트 + 바운딩 박스     │
                 │    ├─ YOLOv8n → 인터랙티브 요소 박스     │
                 │    └─ Florence-2 → 아이콘 기능 캡션      │
                 │           │                              │
                 │           ▼                              │
                 │  build_omniparser_extra_instructions()   │
                 │    "Element [0] Button (120,45)-(200,65) │
                 │     label='Login' interactable=true"     │
                 │           │                              │
                 │           ▼                              │
                 │  ui-venus (Primary VLM, 8B)              │
                 │    → 최종 좌표 결정 + 액션 선택           │
                 │    (바운딩 박스 힌트 참고, 픽셀 기준)     │
                 └─────────────────────────────────────────┘
```

### 우선순위 판단

| 항목 | 우선순위 | 이유 |
|------|----------|------|
| OmniParser V2 평가 (Phase 1) | **P1** | RCS UI에서의 실제 감지 품질을 먼저 확인해야 함 |
| pipeline_omniparser.py 구현 | P2 | Phase 1 결과가 긍정적일 때만 진행 |
| SoM 이미지 직접 전송 | P2 | 텍스트 힌트만으로도 충분할 수 있음 |
| YOLO/Florence-2 RCS fine-tune | P3 | 범용 모델로 충분하면 불필요 |
| AGPL 라이선스 해결 | **P1** (프로덕션 시) | 내부 평가는 괜찮지만 서비스 배포 전 반드시 해결 |

---

## 레퍼런스

| 자료 | 링크 |
|------|------|
| OmniParser GitHub | https://github.com/microsoft/OmniParser |
| OmniParser V2 Microsoft Research | https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/ |
| OmniParser V2 HuggingFace | https://huggingface.co/microsoft/OmniParser-v2.0 |
| Azure AI Foundry Labs | https://labs.ai.azure.com/projects/omniparserv2/ |
| OmniParser V2 Replicate API | https://replicate.com/microsoft/omniparser-v2 |
| ScreenSpot Pro 벤치마크 | OmniParser V2 논문 내 참조 |
| 기존 리서치: VLM GUI 자동화 | `docs/research/vlm_gui_automation_for_engineering_ko.md` |
| 기존 리서치: GUI VLM 벤치마크 | `docs/research/gui_vlm_benchmark_report_ko.md` |
