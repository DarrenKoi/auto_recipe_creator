"""GUI VLM 벤치마크 비교 및 H200 GPU 도입 제안 PPTX 생성 스크립트

현황: Kimi-VL은 사내 서버에서 자체 운영 중 (API 제공)
제안: GUI 자동화 전용 모델 운영을 위한 별도 H200 GPU 2대 도입
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Colors ──
BG_DARK = RGBColor(0x0F, 0x17, 0x2A)
SURFACE = RGBColor(0x1E, 0x29, 0x3B)
SURFACE2 = RGBColor(0x33, 0x41, 0x55)
WHITE = RGBColor(0xE2, 0xE8, 0xF0)
DIM = RGBColor(0x94, 0xA3, 0xB8)
ACCENT = RGBColor(0x38, 0xBD, 0xF8)
ACCENT2 = RGBColor(0x81, 0x8C, 0xF8)
GREEN = RGBColor(0x4A, 0xDE, 0x80)
YELLOW = RGBColor(0xFB, 0xBF, 0x24)
RED = RGBColor(0xF8, 0x71, 0x71)
ORANGE = RGBColor(0xFB, 0x92, 0x3C)


def set_slide_bg(slide, color=BG_DARK):
    """슬라이드 배경색 설정"""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, font_size=18,
                color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Malgun Gothic"):
    """텍스트 박스 추가 헬퍼"""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_shape_box(slide, left, top, width, height, fill_color=SURFACE, border_color=None):
    """배경 사각형 추가"""
    from pptx.enum.shapes import MSO_SHAPE
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape


def add_bullet_frame(slide, left, top, width, height, items, font_size=14, color=DIM):
    """글머리 기호 텍스트 프레임"""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Malgun Gothic"
        p.space_after = Pt(4)
    return txBox


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ════════════════════════════════════════
    # SLIDE 1: 표지
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide)

    add_textbox(slide, 1.5, 1.5, 10, 1.2,
                "GUI 자동화 전용 VLM 모델 도입 제안",
                font_size=36, color=ACCENT, bold=True)
    add_textbox(slide, 1.5, 2.8, 10, 0.8,
                "CD-SEM / RCS 레시피 자동화를 위한 GPU H200 x 2 도입 필요성",
                font_size=22, color=ACCENT2)
    add_textbox(slide, 1.5, 4.0, 10, 0.5,
                "벤치마크 기반 분석 — 범용 VLM vs GUI 전문 모델 성능 비교",
                font_size=16, color=DIM)
    add_textbox(slide, 1.5, 5.0, 10, 0.8,
                "현황: Kimi-VL은 사내 서버에서 자체 운영 중 (API 서빙)\n"
                "제안: GUI 자동화 전용 모델을 별도 GPU에서 운영",
                font_size=14, color=YELLOW)
    add_textbox(slide, 1.5, 6.2, 10, 0.5,
                "2026년 2월",
                font_size=14, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 2: 문제점
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 12, 0.8,
                "문제: 현재 Kimi-VL로는 복잡한 엔지니어링 UI를 처리할 수 없음",
                font_size=28, color=RED, bold=True)

    # 왼쪽 박스
    add_shape_box(slide, 0.8, 1.3, 5.8, 5.5, SURFACE, SURFACE2)
    add_textbox(slide, 1.0, 1.4, 5.4, 0.5,
                "Kimi-VL 현재 성능 한계", font_size=18, color=RED, bold=True)

    items = [
        "  ScreenSpot-Pro 벤치마크 정확도: 34.5%",
        "  전문 UI에서 클릭 타겟 3개 중 2개를 놓침",
        "  범용 모델 — GUI 인터랙션 데이터로 학습되지 않음",
        "  아이콘/위젯 감지 능력이 엔지니어링 소프트웨어에서 거의 0%",
        "  사내 운영 중이나, 범용 모델의 근본적 한계 존재",
        "  RCS/CD-SEM 전용 파인튜닝이 불가능한 구조",
    ]
    add_bullet_frame(slide, 1.0, 2.0, 5.4, 4.5, items, font_size=14, color=DIM)

    # 오른쪽 박스 - 좁은 범위 문제
    add_shape_box(slide, 6.8, 1.3, 5.8, 5.5, SURFACE, RED)
    add_textbox(slide, 7.0, 1.4, 5.4, 0.5,
                "핵심 문제: 좁은 영역 UI 요소 인식 실패", font_size=18, color=RED, bold=True)

    items2 = [
        "  UI 요소가 좁은 영역에 밀집되어 있을 때 (<20px 간격),",
        "  Kimi-VL은 인접 요소를 구분하지 못합니다:",
        "",
        "  - 클릭 좌표가 인접 버튼/라벨로 이탈",
        "  - 툴바 아이콘 클러스터에서 모호한 좌표 출력",
        "  - TreeView / ListView 행 높이가 작으면 행 혼동",
        "  - RCS 레시피 편집기는 <10px 간격의 밀집 패널 보유",
        "",
        "  근본 원인: GUI 공간 학습 부재",
        "  범용 VLM은 자연 이미지/문서로 학습 → 밀집 UI에 약함",
        "  타겟 크기가 줄어들수록 정확도가 급격히 하락",
        "  아이콘 전용 그라운딩은 범용 모델에서 < 5% 정확도",
    ]
    add_bullet_frame(slide, 7.0, 2.0, 5.4, 4.5, items2, font_size=12, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 3: ScreenSpot-Pro 벤치마크
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "ScreenSpot-Pro: 전문 GUI 그라운딩 벤치마크",
                font_size=28, color=ACCENT, bold=True)
    add_textbox(slide, 0.8, 1.0, 11, 0.5,
                "고해상도 전문 애플리케이션 UI (CAD, EDA, 개발 도구)에서의 클릭 좌표 정확도 — "
                "CD-SEM/RCS 유즈케이스에 가장 유사한 공개 벤치마크",
                font_size=13, color=DIM)

    # 테이블 데이터
    models = [
        ("MAI-UI-32B (Alibaba)", "GUI 전용", "32B", "67.9", GREEN),
        ("MAI-UI-8B (Alibaba)", "GUI 전용", "8B", "65.7", GREEN),
        ("Qwen3-VL-30B-A3B", "범용", "30B MoE", "61.8", ACCENT),
        ("UI-TARS-1.5-7B (ByteDance)", "GUI 전용", "7B", "61.6", ACCENT),
        ("MAI-UI-2B (Alibaba)", "GUI 전용", "2B", "57.4", YELLOW),
        ("Qwen3-VL-8B", "범용", "8B", "54.6", YELLOW),
        ("Kimi-K2.5 (Moonshot)", "범용", "~200B+ MoE", "52.8", YELLOW),
        ("GUI-Actor-7B (Microsoft)", "GUI 전용", "7B", "44.6", ORANGE),
        ("UI-TARS-72B v1", "GUI 전용", "72B", "38.1", ORANGE),
        ("Kimi-VL / MoonViT (현재 운영 중)", "범용", "~3.9B MoE", "34.5", RED),
        ("OS-Atlas-7B", "GUI 전용", "7B", "18.9", RED),
        ("Claude Computer Use", "범용", "~200B+", "17.1", RED),
        ("GPT-4o (OpenAI)", "범용", "~200B+", "0.8", RED),
    ]

    # 테이블 헤더
    headers = ["모델", "유형", "파라미터", "ScreenSpot-Pro (%)"]
    col_widths = [4.2, 1.3, 1.8, 2.2]
    x_start = 0.8
    y_start = 1.6
    row_h = 0.36

    add_shape_box(slide, x_start, y_start, sum(col_widths) + 0.4, row_h, SURFACE2)
    x = x_start + 0.1
    for header, w in zip(headers, col_widths):
        add_textbox(slide, x, y_start, w, row_h, header,
                    font_size=11, color=ACCENT, bold=True)
        x += w

    for i, (model, mtype, params, score, color) in enumerate(models):
        y = y_start + row_h * (i + 1)
        bg = SURFACE if i % 2 == 0 else BG_DARK
        if "현재" in model:
            bg = RGBColor(0x3B, 0x1A, 0x1A)
        add_shape_box(slide, x_start, y, sum(col_widths) + 0.4, row_h, bg)

        x = x_start + 0.1
        is_current = "현재" in model
        add_textbox(slide, x, y, col_widths[0], row_h, model,
                    font_size=10, color=RED if is_current else WHITE,
                    bold=is_current)
        x += col_widths[0]
        add_textbox(slide, x, y, col_widths[1], row_h, mtype,
                    font_size=10, color=GREEN if "GUI" in mtype else YELLOW)
        x += col_widths[1]
        add_textbox(slide, x, y, col_widths[2], row_h, params, font_size=10, color=DIM)
        x += col_widths[2]
        add_textbox(slide, x, y, col_widths[3], row_h, score + "%",
                    font_size=12, color=color, bold=True)

    # 인사이트 박스
    add_shape_box(slide, 10.2, 1.6, 2.8, 5.5, SURFACE, RED)
    add_textbox(slide, 10.3, 1.7, 2.6, 0.4,
                "핵심 수치", font_size=14, color=RED, bold=True)
    insight_items = [
        "현재 Kimi-VL: 34.5%",
        "UI-TARS-1.5-7B: 61.6%",
        "",
        "+78% 정확도 향상",
        "(7B 모델로 달성 가능)",
        "",
        "GUI 전용 7B 모델이",
        "범용 200B+ 모델보다",
        "전문 UI에서 월등히 우수",
        "",
        "→ 전용 모델 + 전용 GPU",
        "   가 해답",
    ]
    add_bullet_frame(slide, 10.3, 2.2, 2.6, 4.5, insight_items, font_size=11, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 4: Agent 벤치마크
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "End-to-End Agent 벤치마크: 다단계 데스크톱 자동화",
                font_size=28, color=ACCENT, bold=True)
    add_textbox(slide, 0.8, 1.0, 11, 0.5,
                "OSWorld & WindowsAgentArena = Windows 데스크톱 자동화 직접 테스트 — RCS/CD-SEM과 동일한 환경",
                font_size=13, color=DIM)

    agent_models = [
        ("UI-TARS-2 (ByteDance)", "47.5", "50.6", "73.3", "88.2"),
        ("Claude 4 Sonnet", "43.9", "—", "—", "—"),
        ("OpenAI CUA-o3", "42.9", "—", "52.5", "71.0"),
        ("UI-TARS-1.5 (ByteDance)", "42.5", "42.1", "64.2", "75.8"),
        ("Qwen3-VL-32B", "—", "—", "63.7", "—"),
    ]

    headers2 = ["모델", "OSWorld", "WinAgentArena", "AndroidWorld", "Mind2Web"]
    col_widths2 = [3.5, 1.8, 2.0, 2.0, 1.8]
    x_start2 = 1.0
    y_start2 = 1.7

    add_shape_box(slide, x_start2, y_start2, sum(col_widths2) + 0.4, row_h, SURFACE2)
    x = x_start2 + 0.1
    for header, w in zip(headers2, col_widths2):
        add_textbox(slide, x, y_start2, w, row_h, header,
                    font_size=12, color=ACCENT, bold=True)
        x += w

    for i, (model, *scores) in enumerate(agent_models):
        y = y_start2 + row_h * (i + 1)
        bg = SURFACE if i % 2 == 0 else BG_DARK
        add_shape_box(slide, x_start2, y, sum(col_widths2) + 0.4, row_h, bg)
        x = x_start2 + 0.1
        add_textbox(slide, x, y, col_widths2[0], row_h, model, font_size=11, color=WHITE, bold=True)
        x += col_widths2[0]
        for j, s in enumerate(scores):
            c = GREEN if s not in ("—", "") and float(s) > 45 else (ACCENT if s not in ("—", "") else DIM)
            add_textbox(slide, x, y, col_widths2[j + 1], row_h, s,
                        font_size=12, color=c, bold=(s not in ("—", "")))
            x += col_widths2[j + 1]

    # 하단 설명
    add_shape_box(slide, 1.0, 4.2, 11.3, 2.5, SURFACE, ACCENT)
    add_textbox(slide, 1.2, 4.3, 10.9, 0.4,
                "RCS 자동화에 왜 중요한가", font_size=16, color=ACCENT, bold=True)
    note_items = [
        "  UI-TARS-2가 OSWorld (47.5%) / WindowsAgentArena (50.6%) 모두 1위 — Windows 데스크톱 자동화 직접 벤치마크",
        "  이 모델들은 복잡한 다단계 워크플로우를 자율 수행: 앱 실행 → 컨트롤 탐색 → 클릭 → 입력 → 검증",
        "  RCS 레시피 생성에 필요한 능력과 정확히 일치: 로그인 → 탭 전환 → 도구 선택 → 파라미터 설정",
        "  오픈소스 & vLLM 셀프 호스팅 가능 — 현재 Kimi 사내 서버와 독립적으로 운영 가능",
    ]
    add_bullet_frame(slide, 1.2, 4.8, 10.9, 2.0, note_items, font_size=13, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 5: 왜 H200 x 2인가
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "GPU 리소스 요구사항: H200 x 2 도입 근거",
                font_size=28, color=ACCENT, bold=True)
    add_textbox(slide, 0.8, 0.95, 11, 0.4,
                "기존 Kimi-VL 사내 서버와 별도로, GUI 자동화 전용 모델을 위한 독립 GPU 인프라 필요",
                font_size=13, color=YELLOW)

    # GPU 스펙 박스
    add_shape_box(slide, 0.8, 1.5, 5.8, 2.3, SURFACE, ACCENT)
    add_textbox(slide, 1.0, 1.55, 5.4, 0.4,
                "NVIDIA H200 SXM 사양", font_size=16, color=ACCENT, bold=True)
    specs = [
        "  HBM3e 메모리: GPU당 141 GB",
        "  메모리 대역폭: 4.8 TB/s",
        "  FP16 처리량: 989 TFLOPS",
        "  2x H200 합계: 282 GB VRAM",
    ]
    add_bullet_frame(slide, 1.0, 2.0, 5.4, 1.6, specs, font_size=14, color=DIM)

    # VRAM 요구사항 테이블
    vram_data = [
        ("UI-TARS-1.5-7B", "14 GB", "7 GB", "멀티 모델 + 대량 배치"),
        ("MAI-UI-8B", "16 GB", "8 GB", "멀티 모델 + 대량 배치"),
        ("GUI-Actor-7B", "14 GB", "7 GB", "멀티 모델 + 대량 배치"),
        ("UI-TARS-1.5-72B", "144 GB", "72 GB", "2x H200 필수 (FP16)"),
        ("MAI-UI-32B (SOTA)", "64 GB", "32 GB", "1x H200으로 운영 가능"),
        ("Qwen3-VL-30B-A3B", "60 GB", "30 GB", "1x H200으로 운영 가능"),
    ]

    y_table = 4.0
    headers3 = ["모델", "FP16 VRAM", "INT8 VRAM", "2x H200 배포 전략"]
    col_w3 = [3.0, 1.5, 1.5, 4.0]
    add_shape_box(slide, 0.8, y_table, sum(col_w3) + 0.4, 0.35, SURFACE2)
    x = 0.9
    for h, w in zip(headers3, col_w3):
        add_textbox(slide, x, y_table, w, 0.35, h, font_size=11, color=ACCENT, bold=True)
        x += w

    for i, (model, fp16, int8, note) in enumerate(vram_data):
        y = y_table + 0.35 * (i + 1)
        bg = SURFACE if i % 2 == 0 else BG_DARK
        add_shape_box(slide, 0.8, y, sum(col_w3) + 0.4, 0.35, bg)
        x = 0.9
        add_textbox(slide, x, y, col_w3[0], 0.35, model, font_size=10, color=WHITE)
        x += col_w3[0]
        add_textbox(slide, x, y, col_w3[1], 0.35, fp16, font_size=10, color=DIM)
        x += col_w3[1]
        add_textbox(slide, x, y, col_w3[2], 0.35, int8, font_size=10, color=DIM)
        x += col_w3[2]
        add_textbox(slide, x, y, col_w3[3], 0.35, note, font_size=10, color=GREEN)

    # 오른쪽: 2대 근거
    add_shape_box(slide, 6.8, 1.5, 6.0, 2.3, SURFACE, GREEN)
    add_textbox(slide, 7.0, 1.55, 5.6, 0.4,
                "왜 2대인가 (1대가 아닌 이유)", font_size=16, color=GREEN, bold=True)
    reasons = [
        "1. 멀티 모델 A/B 테스트: UI-TARS + MAI-UI를 동시에",
        "   운영하여 비교 및 자동 폴백 구현",
        "2. 72B 모델 운영: 대형 모델은 141GB 이상 필요",
        "   → 2대 텐서 병렬로 FP16 배치 추론 가능",
        "3. 동시 세션: 다수 RCS 자동화 워크플로우의",
        "   병렬 GPU 추론 처리 필요",
        "4. 파인튜닝: 자사 CD-SEM/RCS 스크린샷으로 학습 시",
        "   추론 대비 2~4배 메모리 필요",
        "5. 미래 대비: 차세대 모델은 더 큰 파라미터 추세",
    ]
    add_bullet_frame(slide, 7.0, 2.0, 5.6, 1.8, reasons, font_size=11, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 6: 현재 vs 제안 비교
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "현행 대비 개선 효과: Kimi-VL (사내 운영) vs GUI 전용 모델 (별도 GPU)",
                font_size=26, color=ACCENT, bold=True)

    # 왼쪽: 현행
    add_shape_box(slide, 0.8, 1.3, 5.8, 5.5, SURFACE, RED)
    add_textbox(slide, 1.0, 1.4, 5.4, 0.4,
                "현행: Kimi-VL (사내 서버 운영 중)", font_size=18, color=RED, bold=True)
    api_items = [
        "운영 환경:",
        "  - 사내 서버에서 Kimi-VL API 서빙 중",
        "  - 범용 VLM으로 다양한 업무에 활용",
        "  - GUI 자동화 외 다른 용도에도 사용 가능",
        "",
        "GUI 자동화 한계:",
        "  - ScreenSpot-Pro 정확도 34.5% (전문 UI 기준)",
        "  - 밀집된 좁은 영역 UI 요소 인식 불가",
        "  - 클릭 좌표가 인접 요소로 빈번하게 이탈",
        "  - 범용 모델의 근본적 한계 — 해결 불가",
        "",
        "결론:",
        "  - Kimi-VL은 범용 업무에는 적합하나",
        "  - 정밀 GUI 자동화에는 전용 모델이 필요",
    ]
    add_bullet_frame(slide, 1.0, 1.9, 5.4, 5.0, api_items, font_size=12, color=DIM)

    # 오른쪽: 제안
    add_shape_box(slide, 6.8, 1.3, 5.8, 5.5, SURFACE, GREEN)
    add_textbox(slide, 7.0, 1.4, 5.4, 0.4,
                "제안: GUI 전용 모델 (별도 H200 x 2)", font_size=18, color=GREEN, bold=True)
    self_items = [
        "운영 구조:",
        "  - 기존 Kimi-VL 서버는 그대로 유지",
        "  - 별도 H200 GPU 2대에 GUI 전용 모델 배포",
        "  - vLLM 기반 OpenAI 호환 API 서빙",
        "",
        "기대 효과:",
        "  - 정확도 61.6~67.9% (UI-TARS, MAI-UI)",
        "  - 현재 대비 +78% 정확도 향상",
        "  - 좁은 영역 밀집 UI 요소도 정밀 인식",
        "  - RCS/CD-SEM 스크린샷으로 파인튜닝 가능",
        "",
        "추가 이점:",
        "  - Kimi-VL과 독립 운영 → 기존 서비스 영향 없음",
        "  - 데이터 보안: 모든 추론이 사내에서 완결",
        "  - 멀티 모델 앙상블/폴백으로 안정성 확보",
    ]
    add_bullet_frame(slide, 7.0, 1.9, 5.4, 5.0, self_items, font_size=12, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 7: 구현 로드맵
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "구현 로드맵",
                font_size=28, color=ACCENT, bold=True)

    phases = [
        ("1단계: 즉시 착수 (1~2주차)", ACCENT,
         ["H200에 UI-TARS-1.5-7B 배포 (vLLM 서빙)",
          "OpenAI 호환 API — 기존 poc/work/ 파이프라인 코드 변경 최소화",
          "A/B 테스트: Kimi-VL vs UI-TARS 병행 비교 (RCS 로그인 + 탭 전환)"]),
        ("2단계: 검증 (3~4주차)", YELLOW,
         ["2번째 GPU에 MAI-UI-8B 배포 (보조 모델)",
          "실제 RCS 스크린샷으로 벤치마크: 로그인, 도구 목록, 레시피 편집기",
          "밀집/좁은 영역 컨트롤에서의 클릭 정확도 측정",
          "모델 앙상블/폴백 구현: UI-TARS 우선 → MAI-UI 보조"]),
        ("3단계: 최적화 (2~3개월차)", GREEN,
         ["자사 RCS/CD-SEM 스크린샷 데이터셋 구축 (~1,000장 이상 라벨링)",
          "UI-TARS-1.5-7B 도메인 파인튜닝 (LoRA, ~16GB VRAM)",
          "72B 모델 평가 (UI-TARS-72B) — 2x H200 텐서 병렬 활용",
          "전체 RCS 레시피 생성 파이프라인에 통합"]),
        ("4단계: 프로덕션 (3개월차~)", ACCENT2,
         ["모니터링 및 자동 복구 기능 포함 프로덕션 배포",
          "추가 엔지니어링 도구로 확장 (VeritySEM, CD-SEM 뷰어)",
          "차세대 모델 (UI-TARS-2, MAI-UI v2) 출시 시 평가 및 교체",
          "내부 벤치마크 데이터셋 구축으로 지속적 모델 평가 체계 확립"]),
    ]

    y = 1.2
    for title, color, items in phases:
        add_shape_box(slide, 0.8, y, 11.7, 0.35 + len(items) * 0.30, SURFACE, color)
        add_textbox(slide, 1.0, y + 0.02, 11, 0.35, title, font_size=14, color=color, bold=True)
        for j, item in enumerate(items):
            add_textbox(slide, 1.3, y + 0.35 + j * 0.28, 11, 0.28,
                        "  " + item, font_size=11, color=DIM)
        y += 0.45 + len(items) * 0.30

    # ════════════════════════════════════════
    # SLIDE 8: 요약 및 요청
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 1.5, 0.8, 10, 1.0,
                "요약 및 요청사항",
                font_size=36, color=ACCENT, bold=True)

    summary = [
        "1.  현재 사내 운영 중인 Kimi-VL은 ScreenSpot-Pro 기준 34.5% 정확도",
        "     — CD-SEM/RCS 레시피 자동화에 신뢰할 수 없는 수준",
        "",
        "2.  특히 좁은 영역에 밀집된 UI 요소 (<20px 간격)를 구분하지 못함",
        "     — RCS 레시피 편집기의 밀집 컨트롤 패널에서 치명적",
        "",
        "3.  GUI 전용 오픈소스 모델 (UI-TARS, MAI-UI)은 61~68% 정확도",
        "     — 현재 대비 +78% 향상, 7B 소형 모델로도 달성 가능",
        "",
        "4.  기존 Kimi-VL 서버와 별도로 운영하여",
        "     기존 서비스에 영향 없이 GUI 자동화 전용 인프라 구축",
        "",
        "5.  H200 GPU 2대 (282GB VRAM)로 멀티 모델 서빙,",
        "     72B 대형 모델 운영, 파인튜닝, 미래 확장성 확보",
    ]
    add_bullet_frame(slide, 1.5, 2.0, 10, 4.0, summary, font_size=15, color=DIM)

    add_shape_box(slide, 1.5, 5.8, 10.3, 1.0, SURFACE, ACCENT)
    add_textbox(slide, 1.7, 5.85, 9.9, 0.9,
                "요청: GUI 자동화 전용 VLM 모델 배포 및 파인튜닝을 위한\n"
                "NVIDIA H200 SXM GPU 2대 도입을 승인해 주시기 바랍니다.",
                font_size=18, color=ACCENT, bold=True)

    # Save
    output_path = "/Users/daeyoung/Codes/auto_recipe_creator/docs/gui_vlm_gpu_h200_proposal.pptx"
    prs.save(output_path)
    print(f"[INFO] PPTX 생성 완료: {output_path}")
    return output_path


if __name__ == "__main__":
    create_presentation()
