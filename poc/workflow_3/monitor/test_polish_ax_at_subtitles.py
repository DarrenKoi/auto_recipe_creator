"""polish_ax_at_subtitles: txt 파싱 + 구간 안/밖 그리기."""

import numpy as np

from poc.workflow_3.monitor import polish_ax_at_subtitles as ax


def test_parse_and_draw():
    rows = ax.parse_subtitles("# 주석\n\n00:01~00:06 | 제목 | 설명 | 파이프 포함\n01:02~01:03 | B | b\n")
    assert rows == [(1.0, 7.0, "제목", "설명 | 파이프 포함"), (62.0, 64.0, "B", "b")]

    frame = np.zeros((360, 640, 3), np.uint8)
    ax.draw_section(frame, 0.5, rows)   # 첫 구간 전 - 그대로
    assert not frame.any()
    ax.draw_section(frame, 3.0, rows)
    assert frame[:180].any() and frame[180:].any()   # 제목(위) + 설명(아래)


def test_real_subtitle_file_parses():
    rows = ax.parse_subtitles((ax.VIDEO_DIR / ax.SUBTITLES).read_text(encoding="utf-8"))
    assert len(rows) == 7 and rows[0][:2] == (1.0, 7.0) and rows[-1][:2] == (50.0, 59.0)
    assert all(a[1] <= b[0] for a, b in zip(rows, rows[1:]))   # 구간이 겹치지 않는다
