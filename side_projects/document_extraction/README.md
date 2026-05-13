# document_extraction — 문서 페이지 단위 이미지 추출

폴더 안의 PPT / Excel / Word / PDF 파일을 순차적으로 열어,
**파일명을 딴 하위 폴더에 페이지별 JPEG**로 떨궈주는 도구.

기존 `side_projects/screenshot_document_extraction/`의 v2 — PowerPoint를
COM `Slide.Export()` 대신 **실제 슬라이드쇼 모드를 띄워 화면 캡처**하여,
사용자가 발표 시 보게 될 픽셀과 동일한 결과를 얻는다.

## 출력 구조

```
<OUTPUT_DIR>/
├── presentation_A/
│   ├── page_001.jpg
│   ├── page_002.jpg
│   └── ...
├── workbook_B/
│   ├── page_001.jpg   # sheet1 page 1
│   ├── page_002.jpg   # sheet1 page 2 (인쇄 시 분할)
│   ├── page_003.jpg   # sheet2 page 1
│   └── ...
└── report_C/          # PDF
    └── page_001.jpg ...
```

## 핸들러별 동작

| 확장자 | 핸들러 | 방식 |
|---|---|---|
| `.ppt / .pptx / .pptm` | `ppt_handler.py` | PowerPoint COM → `SlideShowSettings.Run()` (단일 모니터 강제) → primary 모니터 mss 캡처 → `View.Next()` |
| `.xls / .xlsx / .xlsm` | `excel_handler.py` | xlwings로 워크북 열기 → 시트별 `ExportAsFixedFormat(0)` (xlTypePDF) → PyMuPDF 200 DPI 페이지 분할 |
| `.doc / .docx / .docm` | `word_handler.py` | Word COM → `ExportAsFixedFormat(ExportFormat=17)` (wdExportFormatPDF) → PyMuPDF 200 DPI |
| `.pdf` | `pdf_handler.py` | PyMuPDF 200 DPI 직접 렌더 (앱 안 열림) |

## 듀얼 모니터 처리 (PPT)

PowerPoint는 듀얼 모니터 연결 시 발표자 보기 / 슬라이드쇼를 분리한다.
이 도구는 항상 **단일 모니터 모드**로 강제:

- `SlideShowSettings.ShowPresenterView = False` (msoFalse)
- `SlideShowSettings.ShowType = 1` (ppShowTypeSpeaker)
- `SlideShowSettings.AdvanceMode = 1` (ppSlideShowManualAdvance) — 애니메이션 자동 진행 방지

캡처 대상은 항상 `mss().monitors[1]` (primary 모니터).

## 실행

CLI 인자도 환경변수도 사용하지 않는다. **매 실행 전 `extract.py` 상단의 상수를 직접 수정**:

```python
# === 실행 전 매번 채워 넣을 것 =================================================
INPUT_DIR: Path = Path(r"C:\Users\me\Documents\문서더미")
OUTPUT_DIR: Path = Path(r"C:\Users\me\Documents\extracted")
OVERWRITE: bool = False   # True면 기존 출력 폴더 덮어쓰기 (기본: 스킵)
RECURSIVE: bool = False   # True면 하위 폴더 재귀 탐색
# ==============================================================================
```

```bash
uv run python side_projects/document_extraction/extract.py
```

`INPUT_DIR` 또는 `OUTPUT_DIR` 가 비어 있으면 실행 시 에러로 중단된다.

## 의존성

```bash
uv sync --extra dev --extra windows
```

- `pymupdf` (모든 핸들러)
- `Pillow`, `mss` (PPT 스크린샷)
- `pywin32` (Windows COM, PPT/Word)
- `xlwings` (Excel)

PDF 핸들러만 사용하면 macOS/Linux에서도 동작한다.
다른 핸들러는 import 시점에 lazy import 하므로, 지원하지 않는 OS에서
PDF만 처리하는 워크플로우는 막히지 않는다.

## 주의 사항

- **PPT 캡처 중 사용자 입력 금지**: 슬라이드쇼는 primary 모니터를 전면 차지하므로,
  추출이 끝날 때까지 키보드/마우스를 건드리지 말 것.
- **비밀번호 보호 파일**: 에러가 발생하면 해당 파일만 스킵하고 다음으로 진행.
- **이미 존재하는 출력 폴더**: 기본은 스킵. 강제 재추출은 `OVERWRITE=1`.
