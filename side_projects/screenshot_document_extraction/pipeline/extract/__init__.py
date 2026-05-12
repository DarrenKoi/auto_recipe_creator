"""Step 2 — 캡처된 페이지 JPEG 를 paddleocr-vl-1.5, ui-venus 로 추출한다.

페이지 하나당 다음 두 파일을 만든 뒤 머지한다.
- page_NNN.paddleocr.json  : OCR raw 응답
- page_NNN.uivenus.json    : 영역(region) raw 응답
- page_NNN.raw.json        : 두 응답을 합친 머지본
"""
