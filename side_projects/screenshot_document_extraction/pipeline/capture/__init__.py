"""Step 1 — Office/PDF 파일을 페이지별 JPEG 로 캡처한다.

핸들러는 native export 를 기본으로 사용한다 (PowerPoint 는 Slide.Export,
Word/Excel 은 ExportAsFixedFormat → PyMuPDF rasterize, PDF 는 PyMuPDF 직접).
GUI screenshot 모드는 추후 옵션으로 추가 가능하다.
"""
