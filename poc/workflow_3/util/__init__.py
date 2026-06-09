"""workflow_3 공용 경량 유틸리티 묶음 (env/image/json/time + 선택적 mouse/window)."""

from .env_utils import env_flag, env_float, env_int
from .image_utils import (
    build_relative_crop_box,
    capture_window,
    crop_image,
    encode_image_webp,
    ensure_min_span,
    point_to_tiny_bbox,
)
from .json_utils import (
    bbox_1000_to_pixels,
    bbox_center,
    coerce_float,
    extract_json,
    normalize_bbox_1000,
    normalize_lines,
    normalize_tool_text,
    parse_coords,
)
from .time_utils import format_elapsed_ms, make_timestamp_tag

IMAGE_UTILS_AVAILABLE = False
MOUSE_UTILS_AVAILABLE = False
WINDOW_UTILS_AVAILABLE = False

click_at_screen = None
scroll_at_screen = None

WindowRow = None
activate_window = None
close_window = None
collect_window_rows = None
find_window_by_pid_and_title_prefix = None
find_window_by_title_prefix = None
foreground_window = None
get_window_process_id = None
image_point_to_screen = None
is_window_maximized = None
maximize_window = None
read_foreground_window_info = None

IMAGE_UTILS_AVAILABLE = True

try:
    from .mouse_utils import click_at_screen, scroll_at_screen

    MOUSE_UTILS_AVAILABLE = True
except ImportError:
    pass

try:
    from .window_utils import (
        WindowRow,
        activate_window,
        close_window,
        collect_window_rows,
        find_window_by_pid_and_title_prefix,
        find_window_by_title_prefix,
        foreground_window,
        get_window_process_id,
        image_point_to_screen,
        is_window_maximized,
        maximize_window,
        read_foreground_window_info,
    )

    WINDOW_UTILS_AVAILABLE = True
except ImportError:
    pass
