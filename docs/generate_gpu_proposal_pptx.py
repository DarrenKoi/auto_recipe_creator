"""GUI VLM 벤치마크 비교 및 H200 GPU 필요성 제안 PPTX 생성 스크립트"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE

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
                color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Segoe UI"):
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
        p.font.name = "Segoe UI"
        p.space_after = Pt(4)
    return txBox


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ════════════════════════════════════════
    # SLIDE 1: Title
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide)

    add_textbox(slide, 1.5, 1.5, 10, 1.2,
                "GUI-Specialized VLM for CD-SEM/RCS Automation",
                font_size=36, color=ACCENT, bold=True)
    add_textbox(slide, 1.5, 2.8, 10, 0.8,
                "Why Kimi-VL Is Not Enough & Why We Need GPU H200 x2",
                font_size=22, color=ACCENT2)
    add_textbox(slide, 1.5, 4.0, 10, 0.5,
                "Benchmark-Driven Proposal for Self-Hosted GUI Agent Models",
                font_size=16, color=DIM)
    add_textbox(slide, 1.5, 5.5, 10, 0.5,
                "February 2026",
                font_size=14, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 2: Problem Statement
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "Problem: Kimi-VL Cannot Handle Complex Engineering UIs",
                font_size=28, color=RED, bold=True)

    # Left box
    add_shape_box(slide, 0.8, 1.3, 5.8, 5.5, SURFACE, SURFACE2)
    add_textbox(slide, 1.0, 1.4, 5.4, 0.5,
                "Kimi-VL Current Performance", font_size=18, color=RED, bold=True)

    items = [
        "  34.5% accuracy on ScreenSpot-Pro benchmark",
        "  Misses 2 out of every 3 click targets on professional UIs",
        "  General-purpose model not trained on GUI interaction data",
        "  Icon/widget detection near-zero on engineering software",
        "  API-only: latency, cost, and vendor lock-in risks",
        "  No fine-tuning possible on proprietary RCS screenshots",
    ]
    add_bullet_frame(slide, 1.0, 2.0, 5.4, 4.5, items, font_size=14, color=DIM)

    # Right box - narrow range issue
    add_shape_box(slide, 6.8, 1.3, 5.8, 5.5, SURFACE, RED)
    add_textbox(slide, 7.0, 1.4, 5.4, 0.5,
                "Critical: Narrow-Range Element Failure", font_size=18, color=RED, bold=True)

    items2 = [
        "  When UI elements are packed in a narrow region (<20px apart),",
        "  Kimi-VL CANNOT distinguish adjacent elements:",
        "",
        "  - Click coordinates drift to neighboring buttons/labels",
        "  - Toolbar icon clusters produce ambiguous outputs",
        "  - Tree-view / list-view rows confused at small row heights",
        "  - RCS recipe editor has dense panels with <10px spacing",
        "",
        "  Root cause: No GUI spatial training. General VLMs are",
        "  trained on natural images, not dense engineering UIs.",
        "  Accuracy degrades rapidly as target size decreases.",
        "  Icon-only grounding drops to < 5% for general models.",
    ]
    add_bullet_frame(slide, 7.0, 2.0, 5.4, 4.5, items2, font_size=12, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 3: ScreenSpot-Pro Benchmark
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "ScreenSpot-Pro: Professional GUI Grounding Benchmark",
                font_size=28, color=ACCENT, bold=True)
    add_textbox(slide, 0.8, 1.0, 11, 0.5,
                "Click-point accuracy on high-resolution professional application UIs (CAD, EDA, dev tools) "
                "— closest benchmark to CD-SEM/RCS use case",
                font_size=13, color=DIM)

    # Table data
    models = [
        ("MAI-UI-32B (Alibaba)", "GUI-Trained", "32B", "67.9", GREEN),
        ("MAI-UI-8B (Alibaba)", "GUI-Trained", "8B", "65.7", GREEN),
        ("Qwen3-VL-30B-A3B", "General", "30B MoE", "61.8", ACCENT),
        ("UI-TARS-1.5-7B (ByteDance)", "GUI-Trained", "7B", "61.6", ACCENT),
        ("MAI-UI-2B (Alibaba)", "GUI-Trained", "2B", "57.4", YELLOW),
        ("Qwen3-VL-8B", "General", "8B", "54.6", YELLOW),
        ("Kimi-K2.5 (Moonshot)", "General", "~200B+ MoE", "52.8", YELLOW),
        ("GUI-Actor-7B (Microsoft)", "GUI-Trained", "7B", "44.6", ORANGE),
        ("UI-TARS-72B v1", "GUI-Trained", "72B", "38.1", ORANGE),
        ("Kimi-VL / MoonViT (CURRENT)", "General", "~3.9B MoE", "34.5", RED),
        ("OS-Atlas-7B", "GUI-Trained", "7B", "18.9", RED),
        ("Claude Computer Use", "General", "~200B+", "17.1", RED),
        ("GPT-4o (OpenAI)", "General", "~200B+", "0.8", RED),
    ]

    # Table header
    headers = ["Model", "Type", "Params", "ScreenSpot-Pro (%)"]
    col_widths = [4.0, 1.5, 1.8, 2.2]
    x_start = 0.8
    y_start = 1.6
    row_h = 0.36

    # Header row
    add_shape_box(slide, x_start, y_start, sum(col_widths) + 0.4, row_h, SURFACE2)
    x = x_start + 0.1
    for header, w in zip(headers, col_widths):
        add_textbox(slide, x, y_start, w, row_h, header,
                    font_size=11, color=ACCENT, bold=True)
        x += w

    # Data rows
    for i, (model, mtype, params, score, color) in enumerate(models):
        y = y_start + row_h * (i + 1)
        bg = SURFACE if i % 2 == 0 else BG_DARK
        if "CURRENT" in model:
            bg = RGBColor(0x3B, 0x1A, 0x1A)
        add_shape_box(slide, x_start, y, sum(col_widths) + 0.4, row_h, bg)

        x = x_start + 0.1
        is_current = "CURRENT" in model
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

    # Insight box on the right
    add_shape_box(slide, 10.2, 1.6, 2.8, 5.5, SURFACE, RED)
    add_textbox(slide, 10.3, 1.7, 2.6, 0.4,
                "Key Insight", font_size=14, color=RED, bold=True)
    insight_items = [
        "Kimi-VL: 34.5%",
        "UI-TARS-1.5-7B: 61.6%",
        "",
        "+78% improvement",
        "with a 7B model",
        "",
        "GUI-trained 7B models",
        "outperform general",
        "200B+ models on",
        "professional UIs",
    ]
    add_bullet_frame(slide, 10.3, 2.2, 2.6, 4.5, insight_items, font_size=11, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 4: Agent Benchmarks
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "End-to-End Agent Benchmarks: Multi-Step Desktop Automation",
                font_size=28, color=ACCENT, bold=True)
    add_textbox(slide, 0.8, 1.0, 11, 0.5,
                "OSWorld & WindowsAgentArena directly test Windows desktop automation — same environment as RCS/CD-SEM",
                font_size=13, color=DIM)

    # Agent table
    agent_models = [
        ("UI-TARS-2 (ByteDance)", "47.5", "50.6", "73.3", "88.2"),
        ("Claude 4 Sonnet", "43.9", "—", "—", "—"),
        ("OpenAI CUA-o3", "42.9", "—", "52.5", "71.0"),
        ("UI-TARS-1.5 (ByteDance)", "42.5", "42.1", "64.2", "75.8"),
        ("Qwen3-VL-32B", "—", "—", "63.7", "—"),
    ]

    headers2 = ["Model", "OSWorld", "WinAgentArena", "AndroidWorld", "Mind2Web"]
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

    # Bottom note
    add_shape_box(slide, 1.0, 4.2, 11.3, 2.5, SURFACE, ACCENT)
    add_textbox(slide, 1.2, 4.3, 10.9, 0.4,
                "Why This Matters for RCS Automation", font_size=16, color=ACCENT, bold=True)
    note_items = [
        "  UI-TARS-2 leads both OSWorld (47.5%) and WindowsAgentArena (50.6%) — direct Windows desktop benchmarks",
        "  These models can autonomously navigate complex multi-step workflows: launch app → find controls → click → type → verify",
        "  Exactly the capability needed for RCS recipe creation: login → switch tabs → select tools → configure parameters",
        "  Open-source & self-hostable via vLLM — no API cost, no vendor dependency, full data privacy",
    ]
    add_bullet_frame(slide, 1.2, 4.8, 10.9, 2.0, note_items, font_size=13, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 5: Why H200 x2
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "GPU Resource Requirement: H200 x 2",
                font_size=28, color=ACCENT, bold=True)

    # GPU specs box
    add_shape_box(slide, 0.8, 1.3, 5.8, 2.5, SURFACE, ACCENT)
    add_textbox(slide, 1.0, 1.4, 5.4, 0.4,
                "NVIDIA H200 SXM Specifications", font_size=16, color=ACCENT, bold=True)
    specs = [
        "  HBM3e Memory: 141 GB per GPU",
        "  Memory Bandwidth: 4.8 TB/s",
        "  FP16 Throughput: 989 TFLOPS",
        "  2x H200 Total: 282 GB VRAM",
    ]
    add_bullet_frame(slide, 1.0, 1.9, 5.4, 1.8, specs, font_size=14, color=DIM)

    # VRAM requirements table
    vram_data = [
        ("UI-TARS-1.5-7B", "14 GB", "7 GB", "Multi-model + large batch"),
        ("MAI-UI-8B", "16 GB", "8 GB", "Multi-model + large batch"),
        ("GUI-Actor-7B", "14 GB", "7 GB", "Multi-model + large batch"),
        ("UI-TARS-1.5-72B", "144 GB", "72 GB", "Requires 2x H200 (FP16)"),
        ("MAI-UI-32B (SOTA)", "64 GB", "32 GB", "Comfortable on 1x H200"),
        ("Qwen3-VL-30B-A3B", "60 GB", "30 GB", "Comfortable on 1x H200"),
    ]

    y_table = 4.0
    headers3 = ["Model", "FP16 VRAM", "INT8 VRAM", "2x H200 Deployment"]
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

    # Right side: Justification
    add_shape_box(slide, 6.8, 1.3, 6.0, 2.5, SURFACE, GREEN)
    add_textbox(slide, 7.0, 1.4, 5.6, 0.4,
                "Why 2x H200 (Not 1x)", font_size=16, color=GREEN, bold=True)
    reasons = [
        "1. Multi-Model A/B Testing: Run UI-TARS + MAI-UI",
        "   simultaneously for comparison and automatic fallback",
        "2. 72B Model Access: Larger models need >141GB total",
        "   for comfortable FP16 serving with batch inference",
        "3. Concurrent Sessions: Multiple RCS automation",
        "   workflows need parallel GPU inference",
        "4. Fine-Tuning: Training on proprietary CD-SEM/RCS",
        "   screenshots requires 2-4x inference memory",
        "5. Future-Proofing: Next-gen models trend larger",
    ]
    add_bullet_frame(slide, 7.0, 1.9, 5.6, 2.0, reasons, font_size=11, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 6: Cost-Benefit & ROI
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "Cost-Benefit Analysis: API vs Self-Hosted",
                font_size=28, color=ACCENT, bold=True)

    # API cost box
    add_shape_box(slide, 0.8, 1.3, 5.8, 5.5, SURFACE, RED)
    add_textbox(slide, 1.0, 1.4, 5.4, 0.4,
                "Current: Cloud API (Kimi-VL)", font_size=18, color=RED, bold=True)
    api_items = [
        "Ongoing Costs:",
        "  - Per-request API fees (vision tokens are expensive)",
        "  - 500-2000ms latency per VLM call (network round-trip)",
        "  - Each RCS step = 3-5 VLM calls = 1.5-10s delay per step",
        "",
        "Risks:",
        "  - Vendor dependency (API deprecation, pricing changes)",
        "  - Data privacy: screenshots sent to external servers",
        "  - Rate limiting during peak automation runs",
        "  - Cannot fine-tune on proprietary engineering UIs",
        "",
        "Performance:",
        "  - 34.5% accuracy on professional GUI grounding",
        "  - Frequent misclicks requiring human intervention",
        "  - Cannot reliably distinguish narrow-range elements",
    ]
    add_bullet_frame(slide, 1.0, 1.9, 5.4, 5.0, api_items, font_size=12, color=DIM)

    # Self-hosted box
    add_shape_box(slide, 6.8, 1.3, 5.8, 5.5, SURFACE, GREEN)
    add_textbox(slide, 7.0, 1.4, 5.4, 0.4,
                "Proposed: Self-Hosted (H200 x2)", font_size=18, color=GREEN, bold=True)
    self_items = [
        "One-Time Investment:",
        "  - 2x H200 GPU hardware (282GB total VRAM)",
        "  - Zero per-request cost after setup",
        "  - vLLM serving: production-ready inference server",
        "",
        "Advantages:",
        "  - Full data privacy: all inference on-premise",
        "  - <100ms latency (local GPU vs cloud round-trip)",
        "  - No rate limits: unlimited concurrent requests",
        "  - Fine-tune on proprietary RCS/CD-SEM screenshots",
        "",
        "Performance:",
        "  - 61.6-67.9% accuracy (GUI-specialized models)",
        "  - +78% improvement over current Kimi-VL",
        "  - Reliable narrow-range element detection",
        "  - Run multiple models for fallback/ensemble",
    ]
    add_bullet_frame(slide, 7.0, 1.9, 5.4, 5.0, self_items, font_size=12, color=DIM)

    # ════════════════════════════════════════
    # SLIDE 7: Recommended Roadmap
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 0.8, 0.3, 11, 0.8,
                "Implementation Roadmap",
                font_size=28, color=ACCENT, bold=True)

    phases = [
        ("Phase 1: Immediate (Week 1-2)", ACCENT,
         ["Deploy UI-TARS-1.5-7B on H200 via vLLM",
          "OpenAI-compatible API — minimal code changes to existing poc/work/ pipeline",
          "A/B test: Kimi-VL vs UI-TARS side-by-side on RCS login + tab switching"]),
        ("Phase 2: Validation (Week 3-4)", YELLOW,
         ["Deploy MAI-UI-8B as secondary model (2nd GPU)",
          "Benchmark on real RCS screenshots: login, tool list, recipe editor",
          "Measure click accuracy, especially on dense/narrow-range controls",
          "Implement model ensemble/fallback: try UI-TARS first, fall back to MAI-UI"]),
        ("Phase 3: Optimization (Month 2-3)", GREEN,
         ["Collect proprietary RCS/CD-SEM screenshot dataset (~1000+ labeled images)",
          "Fine-tune UI-TARS-1.5-7B on domain-specific data (LoRA, ~16GB VRAM)",
          "Evaluate 72B models (UI-TARS-72B) using 2x H200 tensor parallel",
          "Integrate into full RCS recipe creation pipeline"]),
        ("Phase 4: Production (Month 3+)", ACCENT2,
         ["Production deployment with monitoring and auto-recovery",
          "Expand to additional engineering tools (VeritySEM, CD-SEM viewer)",
          "Evaluate next-gen models (UI-TARS-2, MAI-UI v2) as they release",
          "Build internal benchmark dataset for continuous model evaluation"]),
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
    # SLIDE 8: Summary / Ask
    # ════════════════════════════════════════
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)

    add_textbox(slide, 1.5, 0.8, 10, 1.0,
                "Summary & Request",
                font_size=36, color=ACCENT, bold=True)

    summary = [
        "1.  Current Kimi-VL model scores 34.5% on professional GUI grounding",
        "     — insufficient for reliable CD-SEM/RCS recipe automation",
        "",
        "2.  Kimi-VL fails on narrow-range UI elements (<20px spacing)",
        "     — critical limitation for dense engineering control panels",
        "",
        "3.  GUI-specialized open-source models (UI-TARS, MAI-UI) achieve",
        "     61-68% accuracy — a +78% improvement with self-hosted inference",
        "",
        "4.  Self-hosting eliminates API costs, provides data privacy,",
        "     enables fine-tuning on proprietary screenshots, and reduces",
        "     inference latency from seconds to milliseconds",
        "",
        "5.  2x H200 GPUs (282GB total) enables multi-model serving,",
        "     72B model access, fine-tuning, and future-proofing",
    ]
    add_bullet_frame(slide, 1.5, 2.0, 10, 4.0, summary, font_size=15, color=DIM)

    add_shape_box(slide, 1.5, 5.8, 10.3, 1.0, SURFACE, ACCENT)
    add_textbox(slide, 1.7, 5.85, 9.9, 0.9,
                "Request: Approve procurement of 2x NVIDIA H200 SXM GPUs "
                "for self-hosted GUI-specialized VLM deployment and fine-tuning.",
                font_size=18, color=ACCENT, bold=True)

    # Save
    output_path = "/Users/daeyoung/Codes/auto_recipe_creator/docs/gui_vlm_gpu_h200_proposal.pptx"
    prs.save(output_path)
    print(f"[INFO] PPTX 생성 완료: {output_path}")
    return output_path


if __name__ == "__main__":
    create_presentation()
