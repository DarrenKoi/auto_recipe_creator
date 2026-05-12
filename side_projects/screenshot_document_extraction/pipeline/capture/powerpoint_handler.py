"""PowerPoint 캡처 핸들러 — COM 으로 Slide.Export("JPG") 호출.

Windows + pywin32 + Office 가 모두 있어야 동작한다.
다른 플랫폼에서는 ImportError 가 발생하므로, 상위 runner 는 try/except 로 받아
doc 의 capture.status 를 failed 로 표시한다.
"""

import tempfile
from pathlib import Path
from typing import Iterator

from PIL import Image

from pipeline.settings import (
    JPEG_QUALITY,
    POWERPOINT_EXPORT_WIDTH,
    POWERPOINT_EXPORT_HEIGHT,
)
from pipeline.capture.common import is_windows


def _import_com():
    """pywin32 import 를 lazy 하게 시도한다."""
    if not is_windows():
        raise ImportError("PowerPoint 캡처는 Windows 에서만 동작한다.")
    try:
        import pythoncom  # noqa: F401
        from win32com import client  # type: ignore
    except ImportError as exc:  # pragma: no cover - Windows 전용
        raise ImportError(
            "pywin32 가 필요합니다. `uv pip install pywin32` 로 설치하세요."
        ) from exc
    return client


def iter_pages(source_path: Path) -> Iterator[tuple[int, bytes, int, int]]:
    """PPT/PPTX 슬라이드마다 (page_index, jpeg_bytes, width, height) 를 yield."""
    if not source_path.exists():
        raise FileNotFoundError(f"PowerPoint 파일을 찾을 수 없다: {source_path}")

    client = _import_com()
    import pythoncom

    pythoncom.CoInitialize()
    try:
        app = client.Dispatch("PowerPoint.Application")
        # WithWindow=False 로 보이지 않는 인스턴스를 띄운다.
        presentation = app.Presentations.Open(
            str(source_path.resolve()),
            ReadOnly=True,
            Untitled=False,
            WithWindow=False,
        )
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="ppt_export_"))
            try:
                for index, slide in enumerate(presentation.Slides, start=1):
                    tmp_jpg = tmp_dir / f"slide_{index:03d}.jpg"
                    # Slide.Export(FileName, FilterName, ScaleWidth, ScaleHeight)
                    slide.Export(
                        str(tmp_jpg),
                        "JPG",
                        POWERPOINT_EXPORT_WIDTH,
                        POWERPOINT_EXPORT_HEIGHT,
                    )
                    if not tmp_jpg.exists():
                        raise RuntimeError(
                            f"PowerPoint Slide.Export 결과 파일이 없다: {tmp_jpg}"
                        )
                    jpeg_bytes, width, height = _normalize_jpeg(tmp_jpg)
                    yield index, jpeg_bytes, width, height
                    try:
                        tmp_jpg.unlink()
                    except OSError:
                        pass
            finally:
                try:
                    for leftover in tmp_dir.iterdir():
                        leftover.unlink(missing_ok=True)
                    tmp_dir.rmdir()
                except OSError:
                    pass
        finally:
            presentation.Close()
        app.Quit()
    finally:
        pythoncom.CoUninitialize()


def _normalize_jpeg(jpeg_path: Path) -> tuple[bytes, int, int]:
    """PowerPoint 가 만든 JPEG 의 크기를 읽고 bytes 로 돌려준다.

    필요한 경우 quality 를 재인코딩해서 파일 크기를 줄일 수도 있지만,
    첫 버전에서는 원본 그대로 사용한다.
    """
    data = jpeg_path.read_bytes()
    with Image.open(jpeg_path) as image:
        width, height = image.size
    if jpeg_path.stat().st_size > 8 * 1024 * 1024:
        # 너무 큰 경우 재인코딩
        from io import BytesIO

        with Image.open(jpeg_path) as image:
            buffer = BytesIO()
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
            data = buffer.getvalue()
    return data, width, height
