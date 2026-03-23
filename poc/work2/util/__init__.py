"""poc.work2 전용 경량 유틸리티 묶음."""

from .debug_image_utils import (
    debug_image_path,
    save_debug_jpeg,
    save_debug_webp,
    save_marked_bboxes,
    save_marked_image,
)
from .json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    coerce_float,
    extract_json,
    normalize_bbox_1000,
    normalize_lines,
    parse_coords,
)
from .time_utils import format_elapsed_ms, make_timestamp_tag

IMAGE_UTILS_AVAILABLE = False
WINDOW_UTILS_AVAILABLE = False

try:
    from .image_utils import capture_window, crop_image, encode_image_webp

    IMAGE_UTILS_AVAILABLE = True
except ImportError:
    pass

try:
    from .window_utils import (
        WindowRow,
        activate_window,
        collect_window_rows,
        find_window_by_pid_and_title_prefix,
        find_window_by_title_prefix,
        foreground_window,
        get_window_process_id,
        image_point_to_screen,
        read_foreground_window_info,
    )

    WINDOW_UTILS_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    "WindowRow",
    "activate_window",
    "bbox_1000_to_pixels",
    "bbox_center",
    "capture_window",
    "coerce_float",
    "crop_image",
    "collect_window_rows",
    "debug_image_path",
    "encode_image_webp",
    "extract_json",
    "find_window_by_pid_and_title_prefix",
    "find_window_by_title_prefix",
    "format_elapsed_ms",
    "foreground_window",
    "get_window_process_id",
    "image_point_to_screen",
    "make_timestamp_tag",
    "normalize_bbox_1000",
    "normalize_lines",
    "parse_coords",
    "read_foreground_window_info",
    "save_debug_jpeg",
    "save_debug_webp",
    "save_marked_bboxes",
    "save_marked_image",
]
