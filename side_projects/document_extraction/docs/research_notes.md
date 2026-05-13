# 조사 노트

## 질문

DRM 보호 때문에 원본 PowerPoint, PDF, Excel 파일을 직접 파싱할 수 없을 때, 현재 회사에서 사용할 수 있는 모델들이 스크린샷에서 유용한 정보를 추출할 수 있는가?

현재 판단은 "가능하지만, 화면에 보이는 정보에 한정된다"입니다. 가장 강한 설계는 OCR/document parsing 모델이 evidence를 제공하고, GUI/layout 모델이 visual structure를 해석하며, large VLM이 추출된 evidence 위에서 final synthesis를 수행하는 hybrid extraction pipeline입니다.

## 모델 Capability 정리

### PaddleOCR-VL-1.5

PaddleOCR-VL-1.5는 GUI action보다 document parsing에 초점이 맞춰진 모델이므로, 이 side project의 first-pass 모델로 가장 적합합니다.

공식 문서와 model card에서 확인한 관련 capability는 다음과 같습니다.

- OCR 및 page-level document parsing
- Table recognition
- Formula recognition
- Chart recognition
- Text spotting
- Seal recognition
- Screen photography, illumination variation, skew, scanning, warping 같은 robustness case

Sources:

- PaddleOCR-VL-1.5 docs: <https://www.paddleocr.ai/latest/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.html>
- PaddleOCR-VL-1.5 model card: <https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5>

권장 역할:

- Raw text와 reading order를 추출합니다.
- 보이는 table과 formula를 parse합니다.
- Chart text, legend, axis, visible data label을 식별합니다.
- 고수준 VLM 해석 전에 첫 번째 evidence layer를 생성합니다.

### UI-Venus-1.5-8B

UI-Venus는 GUI grounding, navigation, real-world application screen의 visual understanding을 위해 설계된 GUI agent model family입니다. Model card는 vLLM을 통한 OpenAI-compatible serving과 GUI grounding 성능을 설명합니다.

Source:

- UI-Venus-1.5-8B model card: <https://huggingface.co/inclusionAI/UI-Venus-1.5-8B>

권장 역할:

- Full screenshot에서 high-level region을 감지합니다.
- Screenshot이 slide, PDF page, spreadsheet, dialog, mixed screen 중 무엇인지 식별합니다.
- title, subtitle, body, table, chart, legend, toolbar, footnote, page number 같은 visual block을 labeling합니다.
- refinement를 위한 crop box를 제공합니다.

### MAI-UI-8B

MAI-UI도 GUI-focused model family입니다. 8B model card는 OpenAI-compatible serving과 GUI benchmark 기반 grounding 성능을 설명합니다.

Source:

- MAI-UI-8B model card: <https://huggingface.co/Tongyi-MAI/MAI-UI-8B>

권장 역할:

- Dense layout에서 crop을 refine합니다.
- 작은 chart legend, Excel table header, footer, label을 검사합니다.
- Region이 모호할 때 UI-Venus region proposal을 cross-check합니다.

### Kimi-K2.5

Repo의 `poc/work2/flask_vlm.py`에는 `kimi-k2.5` direct company API model entry가 이미 정의되어 있습니다. 응답 시간이 느릴 수 있으므로 모든 crop에 실행하지 않습니다.

권장 역할:

- OCR과 visual evidence를 final structured answer로 merge합니다.
- OCR output과 layout model interpretation이 충돌할 때 해결합니다.
- Slide intent와 chart meaning을 요약합니다.
- 추출된 table을 일관된 schema로 normalize합니다.
- Confidence note를 포함한 final Markdown과 JSON을 생성합니다.

이 모델은 더 저렴한 pass가 evidence를 충분히 모은 뒤, 또는 extraction confidence가 낮을 때만 사용합니다.

## Hybrid Pipeline이 필요한 이유

단일 모델이 전체 작업을 소유하면 안 됩니다.

- OCR model은 visible text와 document element에는 강하지만, high-level business meaning은 놓칠 수 있습니다.
- GUI grounding model은 region detection과 visual hierarchy에는 유용하지만, 정확한 text의 가장 신뢰할 수 있는 source는 아닙니다.
- Large VLM은 synthesis에 유용하지만, OCR evidence 없이 dense screenshot을 직접 읽게 하면 느리고 hallucination 가능성이 커집니다.

따라서 pipeline은 다음 책임을 분리해야 합니다.

- Evidence extraction
- Region detection
- Crop refinement
- Final reasoning

## Data Type별 예상 난이도

| Source type | 쉬운 case | 어려운 case |
| --- | --- | --- |
| PowerPoint | 큰 title/body text, 단순 chart, visible table | dense slide, 작은 footnote, slide 안에 포함된 screenshot |
| PDF | text-heavy page, numbered section, visible table | low-resolution scan page, multi-column reading order, equation |
| Excel | visible grid, header, cell에 보이는 simple formula | hidden row/column, wide sheet, tiny value, merged cell, filter |

## 안전 경계

이 프로젝트는 screen-observation 범위에 머물러야 합니다.

- 허용: 사용자가 볼 수 있는 content의 user-captured screenshot
- 허용: visible pixel에 대한 OCR 및 VLM analysis
- 금지: DRM removal, protected file parsing, hidden-content extraction, credential bypass, 접근 제어 우회 목적의 automation

## 연구 가설

1. PaddleOCR-VL만으로도 대부분의 visible text는 복원하지만, useful business-level summary는 항상 만들지 못할 수 있습니다.
2. UI-Venus는 slide, spreadsheet, mixed UI screenshot의 region routing을 개선할 것입니다.
3. MAI-UI는 full screenshot보다 crop에 사용할 때 small-region extraction을 개선할 것입니다.
4. Kimi-K2.5는 final document summary와 conflict resolution을 개선하지만, latency 때문에 confidence gate 또는 final synthesis step으로 제한해야 합니다.
5. 가장 큰 개선은 특정 모델 하나를 바꾸는 것보다 crop retry와 evidence merging에서 나올 가능성이 큽니다.
