"use strict";

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const { safeOuterShadow } = require("./pptxgenjs_helpers/util");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const OUT_DIR = __dirname;
const OUT_FILE = path.join(OUT_DIR, "flask_vlm_coworker_intro.pptx");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "SK hynix";
pptx.subject = "동료용 Flask API 및 VLM 소개";
pptx.title = "GUI 자동화를 위한 Flask API + VLM 스택";
pptx.lang = "ko-KR";
pptx.theme = {
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  lang: "ko-KR",
};

const SW = 13.333;
const SH = 7.5;

const COLORS = {
  white: "FFFFFF",
  ink: "0F172A",
  navy: "10233F",
  teal: "0F766E",
  cyan: "0E7490",
  sky: "DCEAF8",
  slate: "475569",
  muted: "64748B",
  panel: "F7F9FC",
  panel2: "F2F6FB",
  line: "D8E1EC",
  pale: "ECF4FB",
  amber: "B45309",
  red: "B42318",
  green: "0B6B50",
};

const FONT_HEAD = "Malgun Gothic";
const FONT_BODY = "Malgun Gothic";
const FONT_MONO = "Consolas";

function addBg(slide) {
  slide.background = { color: COLORS.white };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: SW,
    h: 0.14,
    line: { color: COLORS.navy, transparency: 100 },
    fill: { color: COLORS.navy },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 10.9,
    y: 0,
    w: 2.433,
    h: 0.14,
    line: { color: COLORS.teal, transparency: 100 },
    fill: { color: COLORS.teal },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.62,
    y: 7.02,
    w: 11.95,
    h: 0,
    line: { color: COLORS.line, pt: 1 },
  });
}

function addHeader(slide, section, title, subtitle, pageNo) {
  addBg(slide);
  slide.addText(section.toUpperCase(), {
    x: 0.68,
    y: 0.28,
    w: 2.2,
    h: 0.18,
    fontFace: FONT_BODY,
    fontSize: 9,
    color: COLORS.teal,
    bold: true,
    charSpace: 0.8,
    margin: 0,
  });
  slide.addText(title, {
    x: 0.68,
    y: 0.46,
    w: 9.3,
    h: 0.4,
    fontFace: FONT_HEAD,
    fontSize: 22,
    color: COLORS.ink,
    bold: true,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.68,
      y: 0.92,
      w: 11.2,
      h: 0.26,
      fontFace: FONT_BODY,
      fontSize: 10.5,
      color: COLORS.slate,
      margin: 0,
    });
  }
  slide.addText(String(pageNo), {
    x: 12.3,
    y: 7.06,
    w: 0.35,
    h: 0.16,
    fontFace: FONT_BODY,
    fontSize: 8,
    color: COLORS.muted,
    align: "right",
    margin: 0,
  });
}

function addSource(slide, text) {
  slide.addText(text, {
    x: 0.68,
    y: 7.06,
    w: 10.95,
    h: 0.18,
    fontFace: FONT_BODY,
    fontSize: 6.2,
    color: COLORS.muted,
    margin: 0,
  });
}

function addCard(slide, x, y, w, h, opts = {}) {
  const fill = opts.fill || COLORS.panel;
  const accent = opts.accent || COLORS.navy;
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    line: { color: COLORS.line, pt: 1 },
    fill: { color: fill },
    shadow: safeOuterShadow("000000", 0.05, 45, 1.5, 0.8),
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 0.02,
    y: y + 0.02,
    w: w - 0.04,
    h: 0.08,
    line: { color: accent, transparency: 100 },
    fill: { color: accent },
  });
  if (opts.label) {
    slide.addText(opts.label.toUpperCase(), {
      x: x + 0.18,
      y: y + 0.16,
      w: w - 0.36,
      h: 0.16,
      fontFace: FONT_BODY,
      fontSize: 8.2,
      color: accent,
      bold: true,
      charSpace: 0.6,
      margin: 0,
    });
  }
  const titleYOffset = opts.label ? 0.33 : 0.18;
  const titleHeight = opts.titleH || 0.24;
  if (opts.title) {
    slide.addText(opts.title, {
      x: x + 0.18,
      y: y + titleYOffset,
      w: w - 0.36,
      h: titleHeight,
      fontFace: FONT_HEAD,
      fontSize: opts.titleSize || 14,
      color: COLORS.ink,
      bold: true,
      margin: 0,
    });
  }
  if (opts.body) {
    const minBodyY = opts.title
      ? titleYOffset + titleHeight + 0.16
      : opts.label
        ? 0.4
        : 0.22;
    const bodyYOffset = opts.bodyY !== undefined
      ? Math.max(opts.bodyY, minBodyY)
      : minBodyY;
    slide.addText(opts.body, {
      x: x + 0.18,
      y: y + bodyYOffset,
      w: w - 0.36,
      h: Math.max(0.18, h - bodyYOffset - 0.16),
      fontFace: FONT_BODY,
      fontSize: opts.bodySize || 9.6,
      color: opts.bodyColor || COLORS.slate,
      valign: "top",
      margin: 0,
      breakLine: false,
    });
  }
}

function addMiniList(slide, items, x, y, w, opts = {}) {
  const rowH = opts.rowH || 0.32;
  const dotColor = opts.dotColor || COLORS.teal;
  const fontSize = opts.fontSize || 9.4;
  items.forEach((item, idx) => {
    const yy = y + idx * rowH;
    slide.addShape(pptx.ShapeType.ellipse, {
      x,
      y: yy + 0.1,
      w: 0.08,
      h: 0.08,
      line: { color: dotColor, transparency: 100 },
      fill: { color: dotColor },
    });
    slide.addText(item, {
      x: x + 0.16,
      y: yy,
      w: w - 0.16,
      h: rowH,
      fontFace: FONT_BODY,
      fontSize,
      color: opts.color || COLORS.slate,
      margin: 0,
    });
  });
}

function addPill(slide, x, y, w, text, fill, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.28,
    rectRadius: 0.08,
    line: { color: fill, pt: 1 },
    fill: { color: fill },
  });
  slide.addText(text, {
    x,
    y: y + 0.02,
    w,
    h: 0.18,
    align: "center",
    fontFace: FONT_BODY,
    fontSize: 8.2,
    color: color || COLORS.white,
    bold: true,
    margin: 0,
  });
}

function addCodeBox(slide, x, y, w, h, title, code) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    line: { color: COLORS.navy, pt: 1 },
    fill: { color: COLORS.navy },
  });
  slide.addText(title, {
    x: x + 0.18,
    y: y + 0.14,
    w: w - 0.36,
    h: 0.18,
    fontFace: FONT_BODY,
    fontSize: 8.2,
    color: "A7F3D0",
    bold: true,
    charSpace: 0.6,
    margin: 0,
  });
  slide.addText(code, {
    x: x + 0.18,
    y: y + 0.38,
    w: w - 0.36,
    h: h - 0.5,
    fontFace: FONT_MONO,
    fontSize: 8.3,
    color: "F8FAFC",
    margin: 0,
    valign: "top",
    breakLine: false,
  });
}

function finalize(slide) {
  warnIfSlideHasOverlaps(slide, pptx, {
    muteContainment: true,
    ignoreDecorativeShapes: true,
  });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function newSlide(section, title, subtitle, pageNo) {
  const slide = pptx.addSlide();
  addHeader(slide, section, title, subtitle, pageNo);
  return slide;
}

function addTitleSlide(pageNo) {
  const slide = pptx.addSlide();
  addBg(slide);

  slide.addShape(pptx.ShapeType.ellipse, {
    x: 8.7,
    y: 0.52,
    w: 4.6,
    h: 4.6,
    line: { color: COLORS.sky, pt: 1.2, transparency: 20 },
    fill: { color: COLORS.white, transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.ellipse, {
    x: 9.55,
    y: 1.22,
    w: 2.9,
    h: 2.9,
    line: { color: COLORS.pale, pt: 1.0 },
    fill: { color: COLORS.white, transparency: 100 },
  });

  slide.addText("GUI 자동화 스택", {
    x: 0.7,
    y: 0.68,
    w: 3.0,
    h: 0.22,
    fontFace: FONT_BODY,
    fontSize: 9.6,
    color: COLORS.teal,
    bold: true,
    charSpace: 0.9,
    margin: 0,
  });
  slide.addText("동료 소개용\nFlask API + VLM 스택", {
    x: 0.7,
    y: 1.04,
    w: 5.8,
    h: 1.25,
    fontFace: FONT_HEAD,
    fontSize: 24,
    color: COLORS.ink,
    bold: true,
    margin: 0,
    breakLine: false,
  });
  slide.addText(
    "docs/research, docs/setup_vlms, poc/work2를 기준으로 정리한 벤치마크 중심 소개 자료입니다.\n모델 계보, 강점과 한계, Flask 프록시 사용법, 이미지 전송, OCR, 프롬프트, 파이프라인 설계를 다룹니다.",
    {
      x: 0.72,
      y: 2.5,
      w: 5.6,
      h: 0.82,
      fontFace: FONT_BODY,
      fontSize: 11.2,
      color: COLORS.slate,
      margin: 0,
    }
  );

  addPill(slide, 0.72, 3.48, 1.6, "벤치마크", COLORS.navy);
  addPill(slide, 2.45, 3.48, 1.6, "프록시 API", COLORS.teal);
  addPill(slide, 4.18, 3.48, 1.6, "OCR + GUI", COLORS.cyan);

  slide.addText("현재 설치된 프록시 서빙 모델", {
    x: 7.15,
    y: 0.88,
    w: 3.6,
    h: 0.22,
    fontFace: FONT_HEAD,
    fontSize: 14.5,
    color: COLORS.ink,
    bold: true,
    margin: 0,
  });
  slide.addText("비교용 direct baseline을 포함한 현재 운영 모델 구성입니다.", {
    x: 7.15,
    y: 1.18,
    w: 4.9,
    h: 0.24,
    fontFace: FONT_BODY,
    fontSize: 9.4,
    color: COLORS.slate,
    margin: 0,
  });

  const modelCards = [
    ["ui-venus", "전체 화면 1차 grounding", COLORS.navy],
    ["ui-tars", "주력 대안이자 agent형 planner", COLORS.cyan],
    ["mai-ui", "좁은 crop 재판독 sidecar", COLORS.teal],
    ["paddleocr-vl-1.5", "기본 OCR 및 구조 추출", COLORS.green],
    ["got-ocr", "작은 영역용 hard OCR fallback", COLORS.amber],
  ];
  modelCards.forEach((item, idx) => {
    const y = 1.62 + idx * 0.92;
    slide.addShape(pptx.ShapeType.roundRect, {
      x: 7.12,
      y,
      w: 5.22,
      h: 0.72,
      rectRadius: 0.08,
      line: { color: COLORS.line, pt: 1 },
      fill: { color: COLORS.panel },
      shadow: safeOuterShadow("000000", 0.05, 45, 1.2, 0.6),
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: 7.14,
      y: y + 0.02,
      w: 0.1,
      h: 0.68,
      line: { color: item[2], transparency: 100 },
      fill: { color: item[2] },
    });
    slide.addText(item[0], {
      x: 7.36,
      y: y + 0.16,
      w: 1.6,
      h: 0.2,
      fontFace: FONT_BODY,
      fontSize: 10.2,
      color: item[2],
      bold: true,
      margin: 0,
    });
    slide.addText(item[1], {
      x: 8.98,
      y: y + 0.16,
      w: 3.05,
      h: 0.24,
      fontFace: FONT_BODY,
      fontSize: 9.4,
      color: COLORS.slate,
      margin: 0,
    });
  });

  slide.addText("동료 온보딩용으로 재정리한 research / setup 맵을 기준으로 구성했습니다.", {
    x: 0.7,
    y: 6.55,
    w: 6.2,
    h: 0.22,
    fontFace: FONT_BODY,
    fontSize: 8.6,
    color: COLORS.muted,
    margin: 0,
  });
  slide.addText(String(pageNo), {
    x: 12.3,
    y: 7.06,
    w: 0.35,
    h: 0.16,
    fontFace: FONT_BODY,
    fontSize: 8,
    color: COLORS.muted,
    align: "right",
    margin: 0,
  });
  addSource(
    slide,
    "출처: docs/research/flask_vlm_coworker_research_map.md, docs/setup_vlms/flask_vlm_coworker_setup_map.md, poc/work2/flask_vlm.py"
  );
  finalize(slide);
}

function addSourceMapSlide(pageNo) {
  const slide = newSlide(
    "문서 맵",
    "현재 마크다운 자료 정리 방식",
    "먼저 research map으로 벤치마크와 모델 역할을 잡고, 그다음 setup map으로 배포와 coworker-facing API 사용 흐름을 설명하는 구조입니다.",
    pageNo
  );

  addCard(slide, 0.72, 1.45, 5.9, 4.48, {
    label: "리서치",
    title: "docs/research",
    accent: COLORS.navy,
    body:
      "발표용 기본 읽기 순서:\n1. gui_model_selection_and_benchmark_plan.md\n2. deploy_vlms_model_roles_and_pipeline_research.md\n3. paddleocr_vl_ui_venus_pipeline_research.md\n\n필요할 때만 보는 deep dive:\nvllm_runtime_and_unsloth_finetuning.md\nencode_decode_and_model_architecture.md\npagedattention_and_prefix_caching.md",
    bodySize: 9.8,
  });
  addCard(slide, 6.75, 1.45, 5.88, 4.48, {
    label: "셋업",
    title: "docs/setup_vlms",
    accent: COLORS.teal,
    body:
      "운영 관점 읽기 순서:\n1. README.md\n2. 01-layout-and-settings.md\n3. 03-operations-and-repo-integration.md\n\n모델 기동 상세:\n02-serve-ui-venus-and-mai-ui.md\n05-ui-tars-vs-others.md\n07-paddleocr-vl-1.5.md\n08-got-ocr-2.0-hf.md",
    bodySize: 9.8,
  });

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.72,
    y: 6.12,
    w: 12.0,
    h: 0.5,
    rectRadius: 0.08,
    line: { color: COLORS.line, pt: 1 },
    fill: { color: COLORS.panel2 },
  });
  slide.addText(
    "클라이언트 코드 앵커: flask_vlm.py -> registry와 기본값 | connection_check.py -> 실제 route 확인 | vlm_client.py -> 이미지 요청 | prompts/* -> 작업별 프롬프트 계약",
    {
      x: 0.92,
      y: 6.28,
      w: 11.6,
      h: 0.14,
      fontFace: FONT_BODY,
      fontSize: 8.5,
      color: COLORS.slate,
      margin: 0,
      align: "center",
    }
  );

  addSource(
    slide,
    "출처: docs/research/flask_vlm_coworker_research_map.md, docs/setup_vlms/flask_vlm_coworker_setup_map.md"
  );
  finalize(slide);
}

function addBenchmarkSlide(pageNo) {
  const slide = newSlide(
    "벤치마크",
    "현재 리서치 기준 비교 순서",
    "핵심 원칙은 명확합니다. 설치된 5개 모델을 매 step 모두 호출하지 말고, primary GUI 비교와 조건부 sidecar escalation을 분리해야 합니다.",
    pageNo
  );

  const steps = [
    ["1", "Primary 정면 비교", "Kimi-K2.5, Qwen3-VL-30B-Instruct, UI-Venus, UI-TARS를 같은 스크린샷 세트에서 sidecar 없이 비교합니다."],
    ["2", "MAI-UI 추가", "primary 승자를 고른 뒤에만 MAI-UI를 붙입니다. 작은 버튼, 과밀 탭, local crop retry에 씁니다."],
    ["3", "PaddleOCR 추가", "dense text, parameter panel, table, exact string 검증이 필요할 때 PaddleOCR-VL을 붙입니다."],
    ["4", "GOT fallback", "formatting 민감 OCR이나 hard small-crop 실패에만 GOT-OCR로 escalation합니다."],
  ];
  steps.forEach((item, idx) => {
    const x = 0.72 + idx * 3.05;
    addCard(slide, x, 1.46, 2.82, 1.96, {
      label: `단계 ${item[0]}`,
      title: item[1],
      accent: idx === 0 ? COLORS.navy : idx === 1 ? COLORS.teal : idx === 2 ? COLORS.green : COLORS.amber,
      body: item[2],
      bodySize: 8.8,
      bodyY: 0.62,
    });
  });

  addCard(slide, 0.72, 3.75, 5.65, 2.35, {
    label: "규칙",
    title: "해석 규칙",
    accent: COLORS.navy,
  });
  addMiniList(
    slide,
    [
      "같은 스크린샷 세트와 같은 task 세트를 유지합니다.",
      "같은 단계에서 primary 모델 비교와 sidecar 변경을 섞지 않습니다.",
      "MAI-UI는 전체 화면 기본값이 아니라 crop sidecar로 봅니다.",
      "OCR 모델은 click planner가 아니라 exact-text authority로 둡니다.",
    ],
    0.96,
    4.38,
    5.1,
    { rowH: 0.35, fontSize: 9.3, dotColor: COLORS.navy }
  );

  addCard(slide, 6.68, 3.75, 6.0, 2.35, {
    label: "지표",
    title: "실패 유형을 직접 측정",
    accent: COLORS.teal,
  });
  addMiniList(
    slide,
    [
      "element hit rate (타깃 적중률)",
      "click-point drift (px)",
      "retry count / sidecar escalation rate",
      "step completion rate",
      "small-text OCR recall",
      "latency",
    ],
    6.92,
    4.34,
    5.3,
    { rowH: 0.29, fontSize: 8.95, dotColor: COLORS.teal }
  );

  addSource(
    slide,
    "출처: docs/research/gui_model_selection_and_benchmark_plan.md, docs/research/deploy_vlms_model_roles_and_pipeline_research.md"
  );
  finalize(slide);
}

function addPortfolioSlide(pageNo) {
  const slide = newSlide(
    "모델 구성",
    "설치된 5개 모델의 역할",
    "현재 스택은 의도적으로 비대칭입니다. GUI grounding 모델이 타깃을 고르고, OCR 모델이 exact text를 맡고, MAI-UI가 좁은 crop을 재판독합니다.",
    pageNo
  );

  const x0 = 0.72;
  const y0 = 1.5;
  const rowH = 0.77;
  const cols = [1.7, 2.45, 2.55, 2.6, 2.15];
  const headers = ["서비스", "기반 / 구조", "권장 역할", "주요 한계", "호출 경로"];
  let cursor = x0;
  headers.forEach((header, idx) => {
    slide.addShape(pptx.ShapeType.rect, {
      x: cursor,
      y: y0,
      w: cols[idx],
      h: 0.36,
      line: { color: COLORS.line, pt: 1 },
      fill: { color: COLORS.navy },
    });
    slide.addText(header, {
      x: cursor + 0.08,
      y: y0 + 0.08,
      w: cols[idx] - 0.16,
      h: 0.16,
      fontFace: FONT_BODY,
      fontSize: 8.8,
      color: COLORS.white,
      bold: true,
      margin: 0,
    });
    cursor += cols[idx];
  });

  const rows = [
    ["ui-venus", "Qwen3-VL 계열\nunified GUI agent", "기본 full-screen grounding", "dense exact text는 OCR 도움 필요", "/v1/chat/completions"],
    ["ui-tars", "Qwen2.5-VL\nreasoning-heavy agent", "주력 대안\nplanner 실험", "runtime과 좌표 후처리가 민감", "/v1/chat/completions"],
    ["mai-ui", "Qwen3-VL 계열\naction parser", "작은 local target용 zoom-in sidecar", "최종 OCR authority는 아님", "/v1/chat/completions"],
    ["paddleocr-vl-1.5", "OCR / document parsing VLM\n0.9B급", "기본 OCR 및 구조 추출", "click-grounding planner가 아님", "/v1/chat/completions"],
    ["got-ocr", "transformers 기반 OCR specialist\ndynamic resolution", "작은 영역용 hard OCR fallback", "endpoint와 runtime이 별도", "/v1/ocr"],
  ];

  rows.forEach((row, ridx) => {
    let xx = x0;
    const yy = y0 + 0.42 + ridx * rowH;
    row.forEach((cell, cidx) => {
      slide.addShape(pptx.ShapeType.rect, {
        x: xx,
        y: yy,
        w: cols[cidx],
        h: rowH,
        line: { color: COLORS.line, pt: 1 },
        fill: { color: ridx % 2 === 0 ? COLORS.white : COLORS.panel },
      });
      slide.addText(cell, {
        x: xx + 0.08,
        y: yy + 0.1,
        w: cols[cidx] - 0.16,
        h: rowH - 0.16,
        fontFace: FONT_BODY,
        fontSize: cidx === 0 ? 9.2 : 8.8,
        color: cidx === 0 ? COLORS.ink : COLORS.slate,
        bold: cidx === 0,
        margin: 0,
        valign: "mid",
      });
      xx += cols[cidx];
    });
  });

  addCard(slide, 0.72, 5.92, 12.0, 0.7, {
    label: "운영 원칙 한 줄 요약",
    title: "전체 화면 판단은 GUI 모델이, exact text authority는 OCR 모델이, 작은 ambiguous crop은 먼저 MAI-UI가 맡습니다.",
    accent: COLORS.teal,
    titleSize: 12.2,
  });

  addSource(
    slide,
    "출처: docs/research/deploy_vlms_model_roles_and_pipeline_research.md, poc/work2/flask_vlm.py"
  );
  finalize(slide);
}

function addModelProfileSlide(pageNo, config) {
  const slide = newSlide(
    config.section,
    config.title,
    config.subtitle,
    pageNo
  );

  addCard(slide, 0.72, 1.46, 4.6, 4.95, {
    label: "계보와 구조",
    title: config.modelName,
    accent: config.accent,
    body: config.summary,
    bodyY: 0.72,
    bodySize: 10.5,
  });

  addPill(slide, 0.94, 3.1, 1.18, config.pill1, config.accent);
  addPill(slide, 2.22, 3.1, 1.18, config.pill2, COLORS.navy);
  addPill(slide, 3.5, 3.1, 1.24, config.pill3, COLORS.cyan);

  addCard(slide, 0.94, 3.62, 3.95, 0.96, {
    label: "구축 방식",
    title: "",
    accent: config.accent,
    body: config.builtAs,
    bodyY: 0.28,
    bodySize: 9.2,
  });
  addCard(slide, 0.94, 4.72, 3.95, 0.96, {
    label: "권장 슬롯",
    title: "",
    accent: config.accent,
    body: config.bestSlot,
    bodyY: 0.28,
    bodySize: 9.2,
  });

  addCard(slide, 5.58, 1.46, 3.1, 2.38, {
    label: "강점",
    title: "잘하는 일",
    accent: COLORS.green,
  });
  addMiniList(slide, config.strengths, 5.82, 2.2, 2.62, {
    rowH: 0.38,
    fontSize: 9.2,
    dotColor: COLORS.green,
  });

  addCard(slide, 9.02, 1.46, 3.05, 2.38, {
    label: "약점",
    title: "먼저 흔들리는 지점",
    accent: COLORS.red,
  });
  addMiniList(slide, config.weaknesses, 9.26, 2.2, 2.55, {
    rowH: 0.38,
    fontSize: 9.05,
    dotColor: COLORS.red,
  });

  addCard(slide, 5.58, 4.08, 6.49, 1.6, {
    label: "운영 가이드",
    title: config.guidanceTitle,
    accent: config.accent,
    body: config.guidanceBody,
    bodyY: 0.62,
    bodySize: 9.7,
  });

  addCard(slide, 5.58, 5.92, 3.12, 0.72, {
    label: "권장 상황",
    title: config.useWhen,
    accent: COLORS.teal,
    titleSize: 10.5,
  });
  addCard(slide, 8.95, 5.92, 3.12, 0.72, {
    label: "비권장 상황",
    title: config.avoidWhen,
    accent: COLORS.amber,
    titleSize: 10.2,
  });

  addSource(slide, config.sources);
  finalize(slide);
}

function addArchitectureSlide(pageNo) {
  const slide = newSlide(
    "아키텍처",
    "동료에게 설명할 Flask 프록시 스택 관점",
    "coworker-facing 계약은 route slug -> model alias -> Flask proxy URL입니다. 클라이언트는 서버 내부 flask_api 소스 코드에 의존하지 않아야 합니다.",
    pageNo
  );

  const boxes = [
    [0.78, "동료용 task script", "health check 뒤에 service slug를 고릅니다."],
    [3.25, "poc/work2/flask_vlm.py", "slug, model, URL, 기본값의 single source of truth입니다."],
    [6.16, "vlm_client.py / OCR helper", "이미지 요청을 보내고 응답 텍스트를 정규화합니다."],
    [9.1, "Flask API proxy", "/api/vlm_serve/<service>/v1/* 가 올바른 backend로 라우팅합니다."],
  ];
  boxes.forEach((item, idx) => {
    addCard(slide, item[0], 1.72, idx === 0 ? 2.1 : idx === 1 ? 2.45 : idx === 2 ? 2.45 : 3.32, 1.4, {
      label: `레이어 ${idx + 1}`,
      title: item[1],
      accent: idx === 1 ? COLORS.navy : idx === 2 ? COLORS.teal : COLORS.cyan,
      body: item[2],
      bodySize: 8.8,
      bodyY: 0.58,
    });
  });
  slide.addText("->", {
    x: 2.96,
    y: 2.22,
    w: 0.18,
    h: 0.2,
    fontFace: FONT_HEAD,
    fontSize: 18,
    color: COLORS.muted,
    bold: true,
    margin: 0,
  });
  slide.addText("->", {
    x: 5.91,
    y: 2.22,
    w: 0.18,
    h: 0.2,
    fontFace: FONT_HEAD,
    fontSize: 18,
    color: COLORS.muted,
    bold: true,
    margin: 0,
  });
  slide.addText("->", {
    x: 8.83,
    y: 2.22,
    w: 0.18,
    h: 0.2,
    fontFace: FONT_HEAD,
    fontSize: 18,
    color: COLORS.muted,
    bold: true,
    margin: 0,
  });

  addCard(slide, 0.78, 3.55, 4.15, 2.18, {
    label: "Route 패턴",
    title: "동료에게 공유할 프록시 URL",
    accent: COLORS.navy,
    body:
      "http://<host>/api/vlm_serve/ui-venus\nhttp://<host>/api/vlm_serve/ui-tars\nhttp://<host>/api/vlm_serve/mai-ui\nhttp://<host>/api/vlm_serve/paddleocr-vl-1.5\nhttp://<host>/api/vlm_serve/got-ocr",
    bodySize: 9.1,
    bodyY: 0.62,
  });

  addCard(slide, 5.18, 3.55, 3.45, 2.18, {
    label: "현재 기본값",
    title: "공유 pipeline 스냅샷",
    accent: COLORS.teal,
    body:
      "screen_analysis_service = ui-venus\nmain_tabs_service = ui-venus\nocr_service = paddleocr-vl-1.5\nocr_pipeline_enabled = true",
    bodySize: 9.2,
    bodyY: 0.62,
  });

  addCard(slide, 8.88, 3.55, 3.45, 2.18, {
    label: "중요한 포인트",
    title: "direct baseline 두 개도 같이 존재",
    accent: COLORS.cyan,
    body:
      "Kimi-K2.5와 Qwen3-VL-30B-Instruct도 flask_vlm.py에 등록되어 있지만, 설치된 5개 프록시 route가 아니라 회사 direct LLM base URL을 사용합니다.",
    bodySize: 8.95,
    bodyY: 0.62,
  });

  addSource(
    slide,
    "출처: poc/work2/flask_vlm.py, docs/setup_vlms/README.md, docs/setup_vlms/03-operations-and-repo-integration.md"
  );
  finalize(slide);
}

function addClientCallSlide(pageNo) {
  const slide = newSlide(
    "클라이언트 사용",
    "poc/work2에서 모델 호출하는 방법",
    "안전한 기본 습관은 connection_check.py를 먼저 실행하고, 그다음 명시적인 service slug로 Work2VLMClient를 생성하는 것입니다.",
    pageNo
  );

  addCodeBox(
    slide,
    0.76,
    1.5,
    6.55,
    3.98,
    "표준 OpenAI-compatible 호출",
    `from pathlib import Path
from poc.work2.vlm_client import Work2VLMClient

client = Work2VLMClient(service_slug="ui-venus")
resp = client.chat_with_image_path(
    image_path=Path("debug_images/login.webp"),
    system_message="Respond only with JSON.",
    user_text="Find clickable UI elements.",
)
print(resp.text)
print(resp.token_usage)`
  );

  addCard(slide, 7.55, 1.5, 5.05, 2.08, {
    label: "Route별 호출 방식",
    title: "이 4개는 vlm_client.py로 호출",
    accent: COLORS.teal,
    body:
      "ui-venus -> /v1/chat/completions\nui-tars -> /v1/chat/completions\nmai-ui -> /v1/chat/completions\npaddleocr-vl-1.5 -> /v1/chat/completions\n\ngot-ocr만 예외이며 /v1/ocr를 사용합니다.",
    bodySize: 9.3,
    bodyY: 0.62,
  });

  addCard(slide, 7.55, 3.82, 5.05, 1.66, {
    label: "Health 먼저",
    title: "살아 있는 서비스를 추측하지 않기",
    accent: COLORS.navy,
    body:
      "uv run python poc/work2/connection_check.py\n\n품질 비교 전에 /api/vlm_serve/health와 각 route의 /v1/models를 먼저 확인합니다.",
    bodySize: 9.1,
    bodyY: 0.62,
  });

  addCard(slide, 0.76, 5.82, 11.84, 0.78, {
    label: "설명 포인트",
    title: "먼저 service slug를 설명하고, backend port는 구현 상세로만 설명합니다.",
    accent: COLORS.cyan,
    titleSize: 11.2,
  });

  addSource(
    slide,
    "출처: poc/work2/vlm_client.py, poc/work2/connection_check.py, docs/setup_vlms/03-operations-and-repo-integration.md"
  );
  finalize(slide);
}

function addImageIOSlide(pageNo) {
  const slide = newSlide(
    "이미지 I/O",
    "이미지를 보내는 방식과 결과를 받는 방식",
    "이 저장소의 정책은 명확합니다. 디버그 캡처는 로컬에 JPEG로 저장하고, VLM API 호출 전에는 WebP로 변환해 payload를 줄입니다.",
    pageNo
  );

  addCard(slide, 0.76, 1.5, 4.0, 1.95, {
    label: "아티팩트 정책",
    title: "로컬 디버그와 네트워크 payload 분리",
    accent: COLORS.navy,
  });
  addMiniList(
    slide,
    [
      "로컬 스크린샷과 marked overlay는 JPEG로 저장합니다.",
      "외부 API 전송 이미지는 quality=90 WebP로 변환합니다.",
      "기본 운영 모드는 prompt당 이미지 1장입니다.",
    ],
    1.0,
    2.18,
    3.45,
    { rowH: 0.34, fontSize: 9.2, dotColor: COLORS.navy }
  );

  addCard(slide, 4.96, 1.5, 3.58, 1.95, {
    label: "클라이언트 입력",
    title: "Work2VLMClient가 받는 입력",
    accent: COLORS.teal,
  });
  addMiniList(
    slide,
    [
      "image_path",
      "image_bytes",
      "base64 image string",
      "JPEG / PNG / WebP 자동 MIME 감지",
    ],
    5.2,
    2.12,
    3.02,
    { rowH: 0.28, fontSize: 8.9, dotColor: COLORS.teal }
  );

  addCard(slide, 8.74, 1.5, 3.86, 1.95, {
    label: "전송 포맷",
    title: "OpenAI 스타일 image_url payload",
    accent: COLORS.cyan,
    body:
      "messages -> user -> [text, image_url]\nimage_url.url = data:image/webp;base64,...\nmodel, temperature, max_tokens를 같은 JSON body에 함께 보냅니다.",
    bodySize: 8.95,
    bodyY: 0.62,
  });

  addCodeBox(
    slide,
    0.76,
    3.78,
    5.54,
    2.22,
    "핵심 요청 구조",
    `{
  "model": "ui-venus-1.5-8b",
  "messages": [
    {"role": "system", "content": "Respond only with JSON."},
    {"role": "user", "content": [
      {"type": "text", "text": "<prompt>"},
      {"type": "image_url",
       "image_url": {"url": "data:image/webp;base64,..."}}
    ]}
  ]
}`
  );

  addCard(slide, 6.54, 3.78, 6.08, 2.34, {
    label: "응답 추출",
    title: "vlm_client.py의 파싱 순서",
    accent: COLORS.teal,
  });
  addMiniList(
    slide,
    [
      "choices[].message.content",
      "streamed output용 choices[].delta",
      "top-level output_text / text / content",
      "body가 plain JSON이 아니면 SSE 파싱 fallback",
      "usage는 token_usage로 저장되어 로그와 보고에 활용",
    ],
    6.8,
    4.42,
    5.4,
    { rowH: 0.29, fontSize: 8.85, dotColor: COLORS.teal }
  );

  addSource(
    slide,
    "출처: poc/work2/vlm_client.py, poc/work2/util/image_utils.py, AGENTS.md 저장소 규칙"
  );
  finalize(slide);
}

function addPromptSlide(pageNo) {
  const slide = newSlide(
    "프롬프트",
    "이 저장소에서 좋은 프롬프트 패턴",
    "핵심은 generic한 긴 프롬프트가 아니라 작업별 계약입니다. 좌표계 명시, strict JSON, compact OCR evidence가 중요합니다.",
    pageNo
  );

  addCard(slide, 0.76, 1.5, 4.65, 4.98, {
    label: "프롬프트 규칙",
    title: "안정적인 프롬프트의 5가지 공통점",
    accent: COLORS.navy,
  });
  addMiniList(
    slide,
    [
      "고정된 schema의 JSON만 반환하게 합니다.",
      "pixels 또는 relative_1000처럼 좌표계를 명시합니다.",
      "모호한 중심점이 아니라 click-safe point를 요구합니다.",
      "UI가 과밀하면 target semantics를 더 자세히 적습니다.",
      "OCR은 raw dump가 아니라 compact hint로 넣습니다.",
    ],
    1.0,
    2.18,
    4.0,
    { rowH: 0.43, fontSize: 9.35, dotColor: COLORS.navy }
  );

  addCard(slide, 5.66, 1.5, 3.15, 2.28, {
    label: "화면 분석",
    title: "상태 인식",
    accent: COLORS.teal,
    body:
      "schema는 state_id, state_name, confidence, description, ui_elements, suggested_actions를 요구합니다.\n좌표는 이미지 크기 안의 정수 픽셀입니다.",
    bodySize: 8.95,
    bodyY: 0.62,
  });

  addCard(slide, 9.02, 1.5, 3.58, 2.28, {
    label: "Locator 프롬프트",
    title: "타깃 특화 grounding",
    accent: COLORS.cyan,
    body:
      "RCS 프롬프트는 타깃을 개별 정의하고, relative_1000을 강제하며, control 내부의 safe click point 위치를 명시합니다.",
    bodySize: 8.95,
    bodyY: 0.62,
  });

  addCard(slide, 5.66, 4.04, 3.15, 2.44, {
    label: "PaddleOCR-VL",
    title: "OCR 프롬프트는 최소화",
    accent: COLORS.green,
    body:
      "PaddleOCR-VL은 system_message를 비우고 OCR: 같은 task keyword만 보내는 편이 안정적입니다.\n구조나 위치가 중요하면 Spotting:, Table Recognition:, Chart Recognition:으로 분기합니다.",
    bodySize: 8.8,
    bodyY: 0.62,
  });

  addCard(slide, 9.02, 4.04, 3.58, 2.44, {
    label: "피해야 할 것",
    title: "좋지 않은 프롬프트 패턴",
    accent: COLORS.amber,
    body:
      "OCR 모델에게 action planning을 맡기지 않습니다.\nGUI 모델을 final exact-text authority로 두지 않습니다.\n응답 형식을 모호하게 두지 않습니다.\nOCR raw text 전체를 main prompt에 밀어 넣지 않습니다.",
    bodySize: 8.85,
    bodyY: 0.62,
  });

  addSource(
    slide,
    "출처: poc/work2/prompts/screen_analysis.py, poc/work2/prompts/login_rcs.py, poc/work2/prompts/rcs_main_tabs.py, poc/work2/prompts/ocr_assist.py"
  );
  finalize(slide);
}

function addOCRSlide(pageNo) {
  const slide = newSlide(
    "OCR 사용",
    "OCR 모델 역할을 헷갈리지 않고 쓰는 방법",
    "PaddleOCR-VL은 chat completions 기반 기본 OCR sidecar이고, GOT-OCR은 별도 /v1/ocr route를 쓰는 hard fallback입니다.",
    pageNo
  );

  addCodeBox(
    slide,
    0.76,
    1.48,
    5.95,
    2.25,
    "vlm_client.py로 PaddleOCR-VL 호출",
    `from poc.work2.vlm_client import Work2VLMClient
from poc.work2.prompts.ocr_assist import build_ocr_assist_prompt

system_msg, user_text = build_ocr_assist_prompt(1920, 1080)
client = Work2VLMClient(service_slug="paddleocr-vl-1.5")
resp = client.chat_with_image_path(
    image_path="panel.webp",
    system_message=system_msg,
    user_text=user_text,
)
print(resp.text)`
  );

  addCard(slide, 7.02, 1.48, 5.56, 2.25, {
    label: "keyword 분기 시점",
    title: "모든 상황에 OCR:만 쓰지 않기",
    accent: COLORS.green,
  });
  addMiniList(
    slide,
    [
      "OCR: 넓은 범위 텍스트 추출",
      "Spotting: 텍스트 + 위치",
      "Table Recognition: 구조화된 표와 그리드",
      "Chart Recognition: 차트 레이블과 값",
    ],
    7.28,
    2.18,
    4.95,
    { rowH: 0.34, fontSize: 9.15, dotColor: COLORS.green }
  );

  addCodeBox(
    slide,
    0.76,
    4.02,
    5.95,
    2.26,
    "GOT-OCR 전용 endpoint",
    `import base64, requests

image_b64 = base64.b64encode(open("crop.jpg", "rb").read()).decode()
resp = requests.post(
    "http://<host>/api/vlm_serve/got-ocr/v1/ocr",
    json={
        "image": image_b64,
        "box": [x1, y1, x2, y2],
        "format_output": True,
        "crop_to_patches": False,
    },
    timeout=120,
)
print(resp.json()["text"])`
  );

  addCard(slide, 7.02, 4.02, 5.56, 2.26, {
    label: "판단 규칙",
    title: "어떤 OCR 모델이 답해야 하는가",
    accent: COLORS.amber,
  });
  addMiniList(
    slide,
    [
      "dense text, table, structured extraction은 PaddleOCR-VL을 먼저 사용합니다.",
      "formatting이 중요하거나 crop이 여전히 unreadable하면 GOT-OCR로 갑니다.",
      "exact string authority는 GUI grounding이 아니라 OCR 결과에 둡니다.",
      "OCR과 GUI grounding이 충돌하면 바로 act하지 말고 다시 관찰합니다.",
    ],
    7.28,
    4.72,
    4.95,
    { rowH: 0.34, fontSize: 9.0, dotColor: COLORS.amber }
  );

  addSource(
    slide,
    "출처: poc/work2/prompts/ocr_assist.py, docs/research/paddleocr_vl_ui_venus_pipeline_research.md, deploy_vlms/scripts/serve_got_ocr.py"
  );
  finalize(slide);
}

function addPipelineSlide(pageNo) {
  const slide = newSlide(
    "파이프라인",
    "권장 파이프라인과 escalation 규칙",
    "현재 리서치는 단계형 파이프라인으로 수렴합니다. GUI 모델로 observe하고, MAI-UI로 local escalation하고, OCR로 exact text를 확인한 뒤에만 act합니다.",
    pageNo
  );

  const flow = [
    [0.78, "1. Observe", "전체 화면은 UI-Venus 또는 UI-TARS", COLORS.navy],
    [3.38, "2. Retry local", "작고 과밀한 crop만 MAI-UI", COLORS.teal],
    [6.18, "3. Read exact text", "dense text와 table은 PaddleOCR-VL", COLORS.green],
    [9.05, "4. Fallback OCR", "hard small-region만 GOT-OCR", COLORS.amber],
  ];
  flow.forEach((box) => {
    addCard(slide, box[0], 1.64, 2.28, 1.34, {
      label: box[1],
      title: box[2],
      accent: box[3],
      titleSize: 11.2,
    });
  });
  slide.addText("->", {
    x: 3.12,
    y: 2.06,
    w: 0.18,
    h: 0.2,
    fontFace: FONT_HEAD,
    fontSize: 18,
    color: COLORS.muted,
    bold: true,
    margin: 0,
  });
  slide.addText("->", {
    x: 5.79,
    y: 2.06,
    w: 0.18,
    h: 0.2,
    fontFace: FONT_HEAD,
    fontSize: 18,
    color: COLORS.muted,
    bold: true,
    margin: 0,
  });
  slide.addText("->", {
    x: 8.67,
    y: 2.06,
    w: 0.18,
    h: 0.2,
    fontFace: FONT_HEAD,
    fontSize: 18,
    color: COLORS.muted,
    bold: true,
    margin: 0,
  });

  addCard(slide, 11.55, 1.64, 1.02, 1.34, {
    label: "5. Act",
    title: "모든 action 뒤에 verify",
    accent: COLORS.cyan,
    titleSize: 9.0,
  });

  addCard(slide, 0.78, 3.5, 6.0, 2.5, {
    label: "기본 자동화 흐름",
    title: "Grounding-heavy 작업",
    accent: COLORS.navy,
  });
  addMiniList(
    slide,
    [
      "기본은 full-screen screenshot -> UI-Venus입니다.",
      "타깃이 작거나 과밀하면 crop 후 MAI-UI를 호출합니다.",
      "선택이 exact text에 의존하면 같은 crop을 PaddleOCR-VL로 읽습니다.",
      "formatting 또는 tiny text가 여전히 애매하면 GOT-OCR로 fallback합니다.",
      "좌표는 GUI 모델에서, exact string은 OCR 모델에서 가져옵니다.",
    ],
    1.04,
    4.2,
    5.38,
    { rowH: 0.34, fontSize: 9.2, dotColor: COLORS.navy }
  );

  addCard(slide, 7.02, 3.5, 5.58, 1.28, {
    label: "Extraction-heavy 변형",
    title: "클릭보다 값 추출이 중요하면 OCR을 먼저 사용",
    accent: COLORS.green,
    body:
      "PaddleOCR-VL -> compact OCR anchor -> UI-Venus semantic pass -> hard region만 GOT 검증",
    bodySize: 9.25,
    bodyY: 0.58,
  });

  addCard(slide, 7.02, 4.98, 5.58, 1.02, {
    label: "중재 규칙",
    title: "OCR과 GUI grounding이 충돌하면 action을 멈추고 다시 관찰합니다.",
    accent: COLORS.amber,
    titleSize: 11.1,
  });

  addSource(
    slide,
    "출처: docs/research/deploy_vlms_model_roles_and_pipeline_research.md, docs/research/paddleocr_vl_ui_venus_pipeline_research.md"
  );
  finalize(slide);
}

function addSetupSlide(pageNo) {
  const slide = newSlide(
    "배포",
    "설명해야 할 운영 토폴로지와 셋업 포인트",
    "동료에게는 사용법이 단순해야 하지만, deck 안에서는 포트 계획, runtime 분리, health-check 루틴을 분명히 보여줘야 합니다.",
    pageNo
  );

  addCard(slide, 0.76, 1.48, 5.25, 4.95, {
    label: "현재 토폴로지",
    title: "GPU와 포트 배치",
    accent: COLORS.navy,
  });

  addCard(slide, 1.02, 2.1, 2.1, 1.85, {
    label: "GPU 0",
    title: "Primary GUI",
    accent: COLORS.navy,
    body: "8001 -> UI-Venus\n8003 -> UI-TARS",
    bodySize: 10,
    bodyY: 0.6,
  });
  addCard(slide, 3.38, 2.1, 2.1, 1.85, {
    label: "GPU 1",
    title: "Sidecar",
    accent: COLORS.teal,
    body: "8002 -> MAI-UI\n8004 -> PaddleOCR-VL\n8005 -> GOT-OCR",
    bodySize: 9.6,
    bodyY: 0.6,
  });
  addCard(slide, 1.02, 4.28, 4.46, 1.55, {
    label: "Runtime 분리",
    title: "",
    accent: COLORS.cyan,
    body:
      "vLLM 경로: UI-Venus, UI-TARS, MAI-UI, PaddleOCR-VL\ntransformers wrapper: GOT-OCR",
    bodySize: 9.25,
    bodyY: 0.28,
  });

  addCard(slide, 6.28, 1.48, 6.32, 2.18, {
    label: "운영 루틴",
    title: "최소 bring-up 흐름",
    accent: COLORS.teal,
  });
  addMiniList(
    slide,
    [
      "먼저 docs/setup_vlms/README.md와 01-layout-and-settings.md를 봅니다.",
      "deploy_vlms/scripts/의 wrapper script로 모델을 기동합니다.",
      "품질 테스트 전에 /v1/models 또는 /api/vlm_serve/health를 확인합니다.",
      "raw backend port 대신 Flask proxy URL을 동료에게 공유합니다.",
    ],
    6.52,
    2.18,
    5.55,
    { rowH: 0.33, fontSize: 9.1, dotColor: COLORS.teal }
  );

  addCard(slide, 6.28, 3.96, 6.32, 2.46, {
    label: "실행 명령",
    title: "실무에서 주로 필요한 명령",
    accent: COLORS.navy,
    body:
      "uv run python poc/work2/connection_check.py\npython scripts/start_ui_venus.py\npython scripts/start_mai_ui.py\npython scripts/start_ui_tars.py\npython scripts/start_paddleocr_vl.py\npython scripts/serve_got_ocr.py",
    bodySize: 9.0,
    bodyY: 0.62,
  });

  addSource(
    slide,
    "출처: docs/setup_vlms/README.md, docs/setup_vlms/01-layout-and-settings.md, docs/setup_vlms/02-serve-ui-venus-and-mai-ui.md, docs/setup_vlms/08-got-ocr-2.0-hf.md"
  );
  finalize(slide);
}

function buildDeck() {
  let page = 1;
  addTitleSlide(page++);
  addSourceMapSlide(page++);
  addBenchmarkSlide(page++);
  addPortfolioSlide(page++);

  addModelProfileSlide(page++, {
    section: "모델 1",
    title: "UI-Venus-1.5-8B",
    subtitle: "현재 poc/work2에서 full-screen screenshot grounding 기본값으로 가장 적합한 모델입니다.",
    modelName: "UI-Venus-1.5-8B",
    accent: COLORS.navy,
    pill1: "Qwen3-VL",
    pill2: "GUI 에이전트",
    pill3: "전체 화면",
    summary:
      "공식적으로는 screenshot-driven unified GUI agent로 소개됩니다. 현재 리서치 기준으로는 full-screen 상태 인식, target grounding, 최종 click candidate 선택의 기본값으로 가장 적합합니다.",
    builtAs:
      "Qwen3-VL 계열 GUI agent로 구축되었고, mid-training -> offline RL -> online RL -> model merge 흐름이 공개되어 있습니다.",
    bestSlot:
      "RCS screenshot과 다른 single-frame desktop UI 작업에서 기본 screen_analysis / main_tabs primary로 두는 것이 적합합니다.",
    strengths: [
      "desktop, web, mobile GUI benchmark의 screenshot grounding에 강합니다.",
      "work2가 이미 채택한 screenshot-first 자동화 흐름과 잘 맞습니다.",
      "영역이 과밀할 때 crop 기반 zoom-in retry와 궁합이 좋습니다.",
      "무엇이 중요한지 고른 뒤 act하는 흐름에 강합니다.",
    ],
    weaknesses: [
      "exact text, 작은 숫자, dense table의 최종 authority로 두기엔 약합니다.",
      "과밀한 toolbar와 tiny target에서는 crop retry가 필요할 수 있습니다.",
      "exact-string verification을 혼자 맡기면 안 됩니다.",
      "OCR fidelity보다 semantic grounding이 더 강합니다.",
    ],
    guidanceTitle: "스택 안에서의 권장 역할",
    guidanceBody:
      "full-screen observation은 여기서 시작합니다. exact text evidence는 OCR 모델이 맡고, click 전 불확실한 small crop은 MAI-UI로 다시 확인하는 구성이 적합합니다.",
    useWhen: "가장 안정적인 full-screen grounding pass가 필요할 때",
    avoidWhen: "dense panel에서 exact text 추출이 주목적일 때",
    sources:
      "출처: docs/research/deploy_vlms_model_roles_and_pipeline_research.md, docs/setup_vlms/05-ui-tars-vs-others.md",
  });

  addModelProfileSlide(page++, {
    section: "모델 2",
    title: "UI-TARS-1.5-7B",
    subtitle: "더 agent-style reasoning 모델이 필요할 때 비교할 주력 대안입니다.",
    modelName: "UI-TARS-1.5-7B",
    accent: COLORS.cyan,
    pill1: "Qwen2.5-VL",
    pill2: "리저닝",
    pill3: "플래너",
    summary:
      "공식적으로는 computer use, browser use, phone use, game / virtual-world task를 다루는 end-to-end multimodal agent로 소개됩니다. 이 저장소에서는 UI-Venus와 정면 비교할 가장 강한 대안입니다.",
    builtAs:
      "Qwen2.5-VL 기반이며 grounding / computer-use template가 명확하고, thought-before-action과 multi-step agent behavior를 강하게 전면에 둡니다.",
    bestSlot:
      "UI-Venus와의 primary A/B 비교축, 또는 단순 one-shot grounding보다 action DSL 실험이 필요할 때 적합합니다.",
    strengths: [
      "multi-step action planning 프레이밍이 강합니다.",
      "UI-Venus와 head-to-head benchmark 축을 만들기 좋습니다.",
      "action-template 중심 접근이 후속 agent 실험에 유리합니다.",
      "reasoning-heavy next-step proposal이 필요할 때 적합합니다.",
    ],
    weaknesses: [
      "Qwen2.5-VL 경로 특성상 template와 processor 파일에 민감합니다.",
      "absolute coordinate 후처리 중요도가 더 큽니다.",
      "niche industrial UI에서는 단순 single-shot grounding보다 흔들릴 수 있습니다.",
      "여전히 OCR specialist는 아닙니다.",
    ],
    guidanceTitle: "스택 안에서의 권장 역할",
    guidanceBody:
      "UI-TARS는 primary 대안으로 benchmark하되, 비교 중에는 sidecar 규칙을 고정해야 모델 차이와 escalation 정책이 섞이지 않습니다.",
    useWhen: "primary 대안 또는 planner-style 실험이 필요할 때",
    avoidWhen: "단순하고 low-variance인 full-screen grounding만 필요할 때",
    sources:
      "출처: docs/research/deploy_vlms_model_roles_and_pipeline_research.md, docs/setup_vlms/05-ui-tars-vs-others.md",
  });

  addModelProfileSlide(page++, {
    section: "모델 3",
    title: "MAI-UI-8B",
    subtitle: "항상-on primary보다 zoom-in sidecar로 보는 편이 적합한 모델입니다.",
    modelName: "MAI-UI-8B",
    accent: COLORS.teal,
    pill1: "Qwen3-VL",
    pill2: "액션 파서",
    pill3: "crop retry",
    summary:
      "공식적으로는 vision-centric action parser와 device-cloud collaboration을 강조하는 real-world-centric GUI agent입니다. work2에서는 local crop resolver로 둘 때 가장 가치가 큽니다.",
    builtAs:
      "Qwen3-VL 계열이며, 하나의 거대한 prompt보다 GroundingAgent와 NavigationAgent를 나눠 설명하는 구조입니다.",
    bestSlot:
      "primary 모델이 영역을 이미 좁힌 뒤, 작은 버튼이나 icon cluster, label-plus-input pair에 대한 second opinion이 필요할 때 적합합니다.",
    strengths: [
      "dense local crop의 zoom-in recheck에 매우 유용합니다.",
      "작은 버튼과 icon cluster grounding에 강합니다.",
      "primary 답변 confidence가 낮을 때 second opinion으로 좋습니다.",
      "항상-on full-screen보다 sidecar escalation에 더 잘 맞습니다.",
    ],
    weaknesses: [
      "exact OCR authority는 아닙니다.",
      "strict offline 모드에서는 협업 관련 강점을 거의 쓰지 못합니다.",
      "industrial Windows UI의 full-screen 기본값으로는 불확실성이 있습니다.",
      "primary 모델이 먼저 triage한 뒤에 붙일 때 가치가 가장 큽니다.",
    ],
    guidanceTitle: "스택 안에서의 권장 역할",
    guidanceBody:
      "MAI-UI는 primary full-screen pass 뒤에 붙입니다. 실패 원인이 global state 이해가 아니라 local density일 때 가장 적절한 해답입니다.",
    useWhen: "타깃이 작고 과밀하거나 local ambiguity가 클 때",
    avoidWhen: "exact number나 table text가 필요한 경우",
    sources:
      "출처: docs/research/deploy_vlms_model_roles_and_pipeline_research.md, docs/setup_vlms/05-ui-tars-vs-others.md",
  });

  addModelProfileSlide(page++, {
    section: "모델 4",
    title: "PaddleOCR-VL-1.5",
    subtitle: "dense text, table, structure extraction에 쓰는 기본 OCR sidecar입니다.",
    modelName: "PaddleOCR-VL-1.5",
    accent: COLORS.green,
    pill1: "OCR VLM",
    pill2: "109개 언어",
    pill3: "구조 추출",
    summary:
      "공식적으로는 text spotting, table, formula, chart, 왜곡된 capture까지 다루는 OCR / document parsing VLM입니다. work2에서는 기본 OCR evidence engine입니다.",
    builtAs:
      "task-keyword 중심 OCR parser로 보는 편이 맞습니다. 긴 instruction prompt보다 OCR:, Spotting:, Table Recognition:, Chart Recognition: 같은 task selector가 더 중요합니다.",
    bestSlot:
      "dense text, 작은 값, table, exact label verification이 필요할 때 기본 ocr_service로 두는 것이 적합합니다.",
    strengths: [
      "dense text와 structured extraction에서 GUI 모델보다 훨씬 강합니다.",
      "table, chart, formula, multilingual capture를 지원합니다.",
      "파이프라인의 exact string authority 기본값으로 적합합니다.",
      "GUI 모델과 같은 proxy chat/completions 경로로 호출할 수 있습니다.",
    ],
    weaknesses: [
      "primary click-grounding planner가 아닙니다.",
      "출력을 compact hint로 정규화할 때 가장 좋습니다.",
      "현재 prompt helper는 기본적으로 OCR:만 노출합니다.",
      "semantic control selection을 GUI 모델 대신 맡기면 안 됩니다.",
    ],
    guidanceTitle: "스택 안에서의 권장 역할",
    guidanceBody:
      "PaddleOCR-VL은 화면의 텍스트가 실제로 무엇인지 알려주는 역할입니다. 그다음 어떤 텍스트나 field가 중요한지는 GUI 모델이 결정하게 두는 편이 맞습니다.",
    useWhen: "exact text, table, panel value가 필요할 때",
    avoidWhen: "전체 화면에서 one-shot click target만 찾고 싶을 때",
    sources:
      "출처: docs/research/paddleocr_vl_ui_venus_pipeline_research.md, docs/setup_vlms/07-paddleocr-vl-1.5.md",
  });

  addModelProfileSlide(page++, {
    section: "모델 5",
    title: "GOT-OCR-2.0-hf",
    subtitle: "작고 어려우며 formatting-sensitive한 영역에 쓰는 전용 OCR fallback입니다.",
    modelName: "GOT-OCR-2.0-hf",
    accent: COLORS.amber,
    pill1: "OCR fallback",
    pill2: "영역 OCR",
    pill3: "Transformers",
    summary:
      "공식적으로는 dynamic resolution, global/local module, region-aware OCR 옵션을 가진 scene OCR 모델입니다. 이 스택에서는 작은 crop과 formatting-sensitive case용 fallback입니다.",
    builtAs:
      "메인 vLLM 경로가 아니라 direct transformers inference용입니다. wrapper가 /v1/ocr를 통해 box, color, format_output, crop_to_patches 같은 OCR 옵션을 노출합니다.",
    bestSlot:
      "PaddleOCR-VL만으로 부족하고, 실패가 작은 영역이나 code-like text, formatting-sensitive content에 집중될 때 적합합니다.",
    strengths: [
      "tiny하거나 어려운 local region의 fallback으로 좋습니다.",
      "box와 color 힌트로 region-specific OCR을 지원합니다.",
      "formatting이 중요하고 plain extraction만으로 부족할 때 유용합니다.",
      "chat schema를 강제하지 않고 OCR control을 직접 노출합니다.",
    ],
    weaknesses: [
      "메인 chat 모델과 runtime / endpoint 경로가 다릅니다.",
      "GUI planner도 아니고 full-screen semantic model도 아닙니다.",
      "work2의 chat-only client로는 바로 재사용할 수 없습니다.",
      "항상-on step보다 조건부 fallback으로 쓸 때 적합합니다.",
    ],
    guidanceTitle: "스택 안에서의 권장 역할",
    guidanceBody:
      "GOT-OCR은 surgical tool처럼 다루는 편이 좋습니다. GUI 모델과 PaddleOCR-VL이 문제 영역을 이미 좁힌 뒤에도 더 깨끗한 exact text가 필요할 때만 호출합니다.",
    useWhen: "영역이 tiny하거나 blurred되었거나 formatting이 중요할 때",
    avoidWhen: "일반 UI grounding이나 표준 OCR pass만으로 충분할 때",
    sources:
      "출처: docs/research/deploy_vlms_model_roles_and_pipeline_research.md, docs/setup_vlms/08-got-ocr-2.0-hf.md, deploy_vlms/scripts/serve_got_ocr.py",
  });

  addArchitectureSlide(page++);
  addClientCallSlide(page++);
  addImageIOSlide(page++);
  addPromptSlide(page++);
  addOCRSlide(page++);
  addPipelineSlide(page++);
  addSetupSlide(page++);
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  buildDeck();
  await pptx.writeFile({ fileName: OUT_FILE });
  console.log(`Wrote ${OUT_FILE}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
