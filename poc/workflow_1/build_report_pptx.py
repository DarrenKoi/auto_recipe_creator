"""Boss 보고용 2-slide PPTX 생성 스크립트.

poc/workflow_1/ 의 Align Fail 모니터링 + CCTV 자동 캡처 작업을 요약한다.
실행:
  uv run python poc/workflow_1/build_report_pptx.py
출력:
  poc/workflow_1/align_fail_capture_report.pptx
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

OUTPUT_PATH = Path(__file__).resolve().parent / "align_fail_capture_report.pptx"

NAVY = RGBColor(0x14, 0x2A, 0x55)
ACCENT = RGBColor(0xE8, 0x6A, 0x1F)
LIGHT = RGBColor(0xF4, 0xF6, 0xFA)
TEXT = RGBColor(0x22, 0x29, 0x33)
MUTED = RGBColor(0x55, 0x60, 0x70)


def _set_run(run, *, size, bold=False, color=TEXT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"


def _add_title_band(slide, title_text, subtitle_text):
    band = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1)
    )
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()

    title_box = slide.shapes.add_textbox(
        Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.55)
    )
    tf = title_box.text_frame
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title_text
    _set_run(run, size=26, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

    sub_box = slide.shapes.add_textbox(
        Inches(0.4), Inches(0.7), Inches(12.5), Inches(0.35)
    )
    p = sub_box.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = subtitle_text
    _set_run(run, size=13, color=RGBColor(0xCD, 0xD7, 0xE6))


def _add_section_card(slide, *, left, top, width, height, heading, bullets):
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = NAVY
    card.line.width = Pt(0.75)
    card.shadow.inherit = False

    head_box = slide.shapes.add_textbox(
        left + Inches(0.25), top + Inches(0.15), width - Inches(0.5), Inches(0.45)
    )
    p = head_box.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = heading
    _set_run(run, size=16, bold=True, color=NAVY)

    body_box = slide.shapes.add_textbox(
        left + Inches(0.25),
        top + Inches(0.65),
        width - Inches(0.5),
        height - Inches(0.8),
    )
    tf = body_box.text_frame
    tf.word_wrap = True
    for idx, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(4)
        run = p.add_run()
        run.text = f"• {bullet}"
        _set_run(run, size=12, color=TEXT)


def _add_pipeline_step(slide, *, left, top, width, height, index, title, detail):
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    box.line.color.rgb = ACCENT
    box.line.width = Pt(1.25)
    box.shadow.inherit = False

    badge = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        left + Inches(0.15),
        top + Inches(0.15),
        Inches(0.4),
        Inches(0.4),
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = ACCENT
    badge.line.fill.background()
    btf = badge.text_frame
    btf.margin_left = btf.margin_right = btf.margin_top = btf.margin_bottom = 0
    p = btf.paragraphs[0]
    p.alignment = 2  # center
    run = p.add_run()
    run.text = str(index)
    _set_run(run, size=12, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

    title_box = slide.shapes.add_textbox(
        left + Inches(0.65), top + Inches(0.12), width - Inches(0.8), Inches(0.4)
    )
    p = title_box.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = title
    _set_run(run, size=13, bold=True, color=NAVY)

    body_box = slide.shapes.add_textbox(
        left + Inches(0.2),
        top + Inches(0.6),
        width - Inches(0.4),
        height - Inches(0.7),
    )
    tf = body_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = detail
    _set_run(run, size=10.5, color=MUTED)


def _add_footer(slide, text):
    box = slide.shapes.add_textbox(
        Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.3)
    )
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = text
    _set_run(run, size=10, color=MUTED)


def build_slide_1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    _add_title_band(
        slide,
        "CD-SEM Align Fail 자동 감지 & 영상 캡처 파이프라인",
        "poc/workflow_1 — Align Fail 발생 시점의 CCTV 화면을 자동으로 보존",
    )

    _add_section_card(
        slide,
        left=Inches(0.4),
        top=Inches(1.4),
        width=Inches(6.25),
        height=Inches(2.7),
        heading="문제 정의",
        bullets=[
            "CD-SEM 장비에서 Align Fail (ALID=9006) 발생 시,"
            " 원인 분석을 위한 CCTV 영상이 사람이 늦게 확인하면 덮어쓰기로 사라짐",
            "엔지니어가 알람 콘솔을 수동 모니터링하고,"
            " EQP_ID 별로 Tool DVR 을 직접 열어야 했음",
            "Align Fail 순간의 챔버 내부 상태(Channel 4 영상)를"
            " 사후에 재현하기 어려움",
        ],
    )

    _add_section_card(
        slide,
        left=Inches(6.85),
        top=Inches(1.4),
        width=Inches(6.1),
        height=Inches(2.7),
        heading="이번에 만든 것",
        bullets=[
            "Align Fail 알람을 1~2분 주기로 자동 폴링 (ALID=9006 필터)",
            "감지 즉시 Windows 팝업 + 누적 텍스트 로그로 엔지니어에게 통보",
            "동일 EQP_ID 의 중복 알람은 edge-trigger 처리 (해제 후 재발 시에만 재알림)",
            "감지된 EQP_ID 의 Tool DVR(CCTV) 창을 자동으로 열고,"
            " Channel 4 를 확대한 뒤 최대 8분간 100ms 간격으로 프레임 저장",
        ],
    )

    _add_section_card(
        slide,
        left=Inches(0.4),
        top=Inches(4.25),
        width=Inches(12.55),
        height=Inches(2.7),
        heading="기대 효과",
        bullets=[
            "Align Fail 발생 시점의 챔버 영상을 100% 자동 보존 →"
            " 원인 분석/리포트 작성 시간 단축",
            "야간/주말 무인 모니터링 가능 — 엔지니어가 알람 콘솔을 지킬 필요 없음",
            "수집된 8분 분량 (~4,800 프레임) 데이터를"
            " 후속 VLM(UI-Venus) 기반 자동 진단의 학습/검증 데이터로 활용",
            "기존 RCS/DVR 시스템에 변경 없이 GUI 자동화로 동작 — 운영 리스크 최소화",
        ],
    )

    _add_footer(slide, "Module: poc/workflow_1/  •  Trigger: ALID=9006 (Align Fail)  •  Capture: CH4 @ 100ms, up to 8 min")


def build_slide_2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    _add_title_band(
        slide,
        "End-to-End 파이프라인 & 구성 모듈",
        "감지 → 알림 → CCTV 진입 → CH4 확대 → 프레임 캡처",
    )

    steps = [
        (
            "Poll",
            "CD-SEM 알람 API 를 1~2분 주기로 조회하여 ALID=9006 (Align Fail) 만 필터",
        ),
        (
            "Notify",
            "Windows MessageBox 팝업 + logs/align_fail_alarms.txt 누적 기록 (edge-trigger)",
        ),
        (
            "Open DVR",
            "RCS Tool List 창에서 해당 EQP_ID 의 Tool DVR(CCTV) 창을 자동 오픈",
        ),
        (
            "Zoom CH4",
            "DVR Player 창에서 Channel 4 를 확대하여 챔버 내부 영상만 단독 표시",
        ),
        (
            "Capture",
            "최대 8분(480초)간 100ms 간격으로 JPEG 프레임 저장 + summary.json/timeline.txt",
        ),
    ]

    step_top = Inches(1.5)
    step_height = Inches(1.55)
    step_width = Inches(2.45)
    gap = Inches(0.07)
    left = Inches(0.4)
    for idx, (title, detail) in enumerate(steps, start=1):
        _add_pipeline_step(
            slide,
            left=left,
            top=step_top,
            width=step_width,
            height=step_height,
            index=idx,
            title=title,
            detail=detail,
        )
        left += step_width + gap

    _add_section_card(
        slide,
        left=Inches(0.4),
        top=Inches(3.3),
        width=Inches(6.25),
        height=Inches(3.55),
        heading="핵심 모듈",
        bullets=[
            "align_fail_alarm.py — 감지 + 팝업 알림 전용 (1분 주기, 자동화 없음)",
            "monitor_align_fail.py — 감지 + RCS/CCTV/캡처 풀 자동화 (2분 주기)",
            "office_align_fail_alarm.py — 사내 알람 API 호출 & ALID=9006 필터",
            "workflow_select_tool_cctv.py — EQP_ID 로 Tool DVR 창 진입",
            "workflow_select_ch4_cctv.py — DVR Player 에서 Channel 4 확대",
            "capture_window_frames_ch4.py — 100ms 간격 JPEG 캡처 + 메타데이터 저장",
        ],
    )

    _add_section_card(
        slide,
        left=Inches(6.85),
        top=Inches(3.3),
        width=Inches(6.1),
        height=Inches(3.55),
        heading="운영 / 출력물",
        bullets=[
            "실행: uv run python poc/workflow_1/monitor_align_fail.py",
            "전제: 엔지니어가 RCS Tool List 창을 미리 열어둔 상태",
            "출력: 캡처 폴더당 ~4,800 JPEG 프레임 + summary.json + timeline.txt",
            "로그: poc/workflow_1/logs/align_fail_alarms.txt (감지 시각/EQP_ID/ALID)",
            "안전: 캡처 종료 후 DVR 창을 닫고 Tool List 창으로 포커스 복귀",
            "다음 단계: 수집된 프레임을 VLM(UI-Venus) 으로 Align Fail 원인 자동 분류",
        ],
    )

    _add_footer(
        slide,
        "Status: 사내 Windows 환경에서 동작 검증 중  •  다음 단계: VLM 기반 원인 자동 분석 연계",
    )


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    build_slide_1(prs)
    build_slide_2(prs)

    prs.save(OUTPUT_PATH)
    print(f"[INFO] PPTX saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
