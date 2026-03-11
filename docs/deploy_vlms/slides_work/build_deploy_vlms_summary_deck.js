"use strict";

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");
const { svgToDataUri } = require("./pptxgenjs_helpers/svg");

const ROOT = __dirname;
const OUTPUT_DIR = path.join(ROOT, "..", "slides");
const ASSET_DIR = path.join(ROOT, "assets");

const COLORS = {
  bg: "F5EFE6",
  panel: "FFFDF8",
  text: "1F2933",
  muted: "52606D",
  border: "D9D0C3",
  rust: "B85C38",
  rustSoft: "F5D7CA",
  amber: "D98E04",
  amberSoft: "F8E2C2",
  teal: "1F6F78",
  tealSoft: "D8EFF0",
  green: "2E6F40",
  greenSoft: "DDEEE0",
  red: "9E2A2B",
  redSoft: "F5D8D8",
  navySoft: "E2E8F0",
  codeBg: "102A43",
  codeText: "F0F4F8",
};

const FONT_SANS = "Malgun Gothic";
const FONT_MONO = "Menlo";
const SHAPE = new PptxGenJS().ShapeType;

const architectureSvg = fs.readFileSync(
  path.join(ASSET_DIR, "flask_vllm_architecture.svg"),
  "utf8"
);

function addBackground(slide) {
  slide.background = { color: COLORS.bg };
  slide.addShape(SHAPE.rect, {
    x: 0.28,
    y: 0.25,
    w: 0.16,
    h: 6.95,
    line: { color: COLORS.rust, transparency: 100 },
    fill: { color: COLORS.rust },
  });
  slide.addShape(SHAPE.roundRect, {
    x: 0.5,
    y: 0.28,
    w: 12.33,
    h: 6.9,
    line: { color: "E7DDCF", pt: 1.2 },
    fill: { color: "F8F4EC" },
  });
}

function addTitleBlock(slide, section, title, subtitle, sources) {
  addBackground(slide);
  slide.addShape(SHAPE.roundRect, {
    x: 0.74,
    y: 0.44,
    w: 1.46,
    h: 0.34,
    line: { color: COLORS.amberSoft, transparency: 100 },
    fill: { color: COLORS.amberSoft },
  });
  slide.addText(section, {
    x: 0.9,
    y: 0.5,
    w: 1.2,
    h: 0.2,
    fontFace: FONT_SANS,
    fontSize: 9.5,
    bold: true,
    color: COLORS.rust,
    margin: 0,
  });
  slide.addText(title, {
    x: 0.78,
    y: 0.82,
    w: 8.6,
    h: 0.3,
    fontFace: FONT_SANS,
    fontSize: 23,
    bold: true,
    color: COLORS.text,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.8,
      y: 1.26,
      w: 10.8,
      h: 0.18,
      fontFace: FONT_SANS,
      fontSize: 10.6,
      color: COLORS.muted,
      margin: 0,
    });
  }
  slide.addText(`Sources: ${sources}`, {
    x: 0.82,
    y: 6.96,
    w: 11.3,
    h: 0.2,
    fontFace: FONT_SANS,
    fontSize: 8.5,
    color: COLORS.muted,
    margin: 0,
  });
}

function addPanel(slide, x, y, w, h, fill = COLORS.panel, line = COLORS.border) {
  slide.addShape(SHAPE.roundRect, {
    x,
    y,
    w,
    h,
    line: { color: line, pt: 1.1 },
    fill: { color: fill },
  });
}

function addPanelTitle(slide, title, x, y, w, fill = COLORS.tealSoft, color = COLORS.text) {
  slide.addShape(SHAPE.roundRect, {
    x,
    y,
    w,
    h: 0.28,
    line: { color: fill, transparency: 100 },
    fill: { color: fill },
  });
  slide.addText(title, {
    x: x + 0.14,
    y: y + 0.06,
    w: w - 0.28,
    h: 0.16,
    fontFace: FONT_SANS,
    fontSize: 10,
    bold: true,
    color,
    margin: 0,
  });
}

function addBodyText(slide, text, x, y, w, h, fontSize = 12, color = COLORS.text, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: opts.fontFace || FONT_SANS,
    fontSize,
    bold: opts.bold || false,
    italic: opts.italic || false,
    color,
    margin: opts.margin !== undefined ? opts.margin : 0,
    breakLine: opts.breakLine || false,
    align: opts.align || "left",
    valign: opts.valign || "top",
    paraSpaceAfterPt: opts.paraSpaceAfterPt || 0,
  });
}

function addBulletRows(slide, items, x, y, w, rowHeight, bulletColor = COLORS.teal) {
  items.forEach((item, index) => {
    const rowY = y + rowHeight * index;
    slide.addShape(SHAPE.roundRect, {
      x,
      y: rowY + 0.07,
      w: 0.12,
      h: 0.12,
      line: { color: bulletColor, transparency: 100 },
      fill: { color: bulletColor },
    });
    addBodyText(slide, item, x + 0.2, rowY, w - 0.2, rowHeight, 11.5, COLORS.text);
  });
}

function addTag(slide, label, x, y, w, fill, color = COLORS.text) {
  slide.addShape(SHAPE.roundRect, {
    x,
    y,
    w,
    h: 0.26,
    line: { color: fill, transparency: 100 },
    fill: { color: fill },
  });
  addBodyText(slide, label, x + 0.12, y + 0.05, w - 0.24, 0.16, 9.5, color, {
    bold: true,
  });
}

function addCodeBox(slide, title, body, x, y, w, h) {
  addPanel(slide, x, y, w, h, COLORS.codeBg, COLORS.codeBg);
  addTag(slide, title, x + 0.18, y + 0.18, Math.min(1.5, w - 0.36), COLORS.rust, "FFFFFF");
  addBodyText(slide, body, x + 0.18, y + 0.54, w - 0.36, h - 0.7, 10.2, COLORS.codeText, {
    fontFace: FONT_MONO,
  });
}

function addMetricCard(slide, title, value, subtitle, x, y, w, h, fill, accentColor) {
  addPanel(slide, x, y, w, h, fill, fill);
  addBodyText(slide, title, x + 0.18, y + 0.18, w - 0.36, 0.22, 10.5, COLORS.muted, {
    bold: true,
  });
  addBodyText(slide, value, x + 0.18, y + 0.48, w - 0.36, 0.34, 20, accentColor, {
    bold: true,
  });
  addBodyText(slide, subtitle, x + 0.18, y + 0.9, w - 0.36, h - 1.02, 10.5, COLORS.text);
}

function finalizeSlide(slide, pptx) {
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function buildDeck() {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "OpenAI Codex";
  pptx.company = "OpenAI";
  pptx.subject = "deploy_vlms markdown summary";
  pptx.title = "deploy_vlms 운영 요약";
  pptx.lang = "ko-KR";
  pptx.theme = {
    headFontFace: FONT_SANS,
    bodyFontFace: FONT_SANS,
    lang: "ko-KR",
  };
  pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
  pptx.layout = "CUSTOM_WIDE";

  let slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "00 / Overview",
    "deploy_vlms 운영 요약",
    "H200 140GB x2 기준 GUI VLM, Flask proxy, OCR 분기 전략을 10장으로 압축",
    "README, 01-08"
  );
  addBodyText(
    slide,
    "문서 핵심: 1모델=1포트=1서비스로 시작하고, 공통 env / 모델 env / Flask proxy / poc/work 연동 규칙을 먼저 고정한다.",
    0.82,
    1.6,
    6.3,
    0.6,
    14,
    COLORS.text,
    { bold: true }
  );
  addTag(slide, "vLLM 0.17", 0.84, 2.28, 1.0, COLORS.tealSoft, COLORS.teal);
  addTag(slide, "Flask /api proxy", 1.92, 2.28, 1.38, COLORS.amberSoft, COLORS.rust);
  addTag(slide, "poc/work ready", 3.4, 2.28, 1.2, COLORS.greenSoft, COLORS.green);
  addBulletRows(
    slide,
    [
      "권장 baseline은 UI-Venus 8001, MAI-UI 8002, 이후 UI-TARS 8003 순서다.",
      "Flask gateway는 /api/vlm_serve/<service>/v1/* 로 direct port를 감추는 공용 주소를 제공한다.",
      "PaddleOCR-VL-1.5는 기존 vLLM wrapper에 편입하고, GOT-OCR-2.0-hf는 transformers direct runtime으로 분리한다.",
    ],
    0.92,
    2.82,
    6.0,
    0.55,
    COLORS.rust
  );
  addPanel(slide, 7.35, 1.56, 5.1, 4.78, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Source Map", 7.56, 1.76, 1.46, COLORS.tealSoft, COLORS.teal);
  addMetricCard(slide, "Core baseline", "8001 / 8002", "UI-Venus와 MAI-UI를 분리 배치해 장애 분리와 A/B 비교를 우선한다.", 7.62, 2.18, 2.2, 1.46, COLORS.amberSoft, COLORS.rust);
  addMetricCard(slide, "Gateway path", "/api/vlm_serve", "coworkers 에게는 direct port 대신 Flask proxy base URL을 배포한다.", 9.98, 2.18, 2.18, 1.46, COLORS.tealSoft, COLORS.teal);
  addMetricCard(slide, "Research path", "81xx / 82xx / 83xx", "family-size variant는 baseline 포트와 분리된 대역으로 운영한다.", 7.62, 3.9, 2.2, 1.46, COLORS.greenSoft, COLORS.green);
  addMetricCard(slide, "OCR split", "vLLM vs transformers", "PaddleOCR-VL과 GOT-OCR은 런타임 성격이 달라 문서를 분리해 관리한다.", 9.98, 3.9, 2.18, 1.46, COLORS.rustSoft, COLORS.red);
  addBodyText(slide, "Source docs cover layout, startup scripts, Flask integration, offline policy, model comparison, size research, and OCR paths.", 7.62, 5.65, 4.15, 0.44, 10.5, COLORS.muted);
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "01 / Architecture",
    "Flask server와 vLLM 서비스 구조",
    "문서 흐름과 실제 repo 코드 구조를 한 장에 합친 운영 아키텍처",
    "README, 03, flask_api/*, web_main.py"
  );
  slide.addImage({
    data: svgToDataUri(architectureSvg),
    x: 0.76,
    y: 1.46,
    w: 11.92,
    h: 4.72,
  });
  addTag(slide, "imagegen dry-run prompt used", 0.86, 6.34, 1.96, COLORS.amberSoft, COLORS.rust);
  addBodyText(
    slide,
    "실제 이미지 API 생성은 OPENAI_API_KEY 부재로 실행하지 못했고, 대신 같은 prompt spec을 저장한 뒤 그 구조로 SVG를 작성했다.",
    2.96,
    6.38,
    7.9,
    0.16,
    9.2,
    COLORS.muted
  );
  addBodyText(
    slide,
    "Repo current state on March 12, 2026: flask_api/vlm_serve/config.py 에서 enabled=true 인 서비스는 ui-venus(8001)와 paddleocr-vl-1.5(8004)다.",
    0.86,
    6.64,
    11.1,
    0.16,
    9.2,
    COLORS.text,
    { bold: true }
  );
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "02 / Baseline Ports",
    "권장 시작점과 포트 정책",
    "PoC 초반에는 메모리 효율보다 비교 실험과 장애 분리가 우선이다.",
    "README, 02, 03"
  );
  const portCards = [
    { title: "UI-Venus", port: "8001", subtitle: "GPU 0 / 주력 grounding 후보", fill: COLORS.amberSoft, color: COLORS.rust },
    { title: "MAI-UI", port: "8002", subtitle: "GPU 1 / A-B 비교 후보", fill: COLORS.tealSoft, color: COLORS.teal },
    { title: "UI-TARS", port: "8003", subtitle: "다음 단계 추가 / runtime 민감도 높음", fill: COLORS.greenSoft, color: COLORS.green },
    { title: "Canary / OCR", port: "8004", subtitle: "template 검증 또는 PaddleOCR-VL canary", fill: COLORS.rustSoft, color: COLORS.red },
  ];
  portCards.forEach((card, index) => {
    const x = 0.88 + index * 3.05;
    addMetricCard(slide, card.title, card.port, card.subtitle, x, 1.72, 2.65, 1.62, card.fill, card.color);
  });
  addPanel(slide, 0.88, 3.72, 5.82, 2.36, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Port Rules", 1.08, 3.92, 1.1, COLORS.tealSoft, COLORS.teal);
  addBulletRows(
    slide,
    [
      "8000은 비워 둔다.",
      "운영 포트는 모델 식별자처럼 유지한다: 8001=ui-venus, 8002=mai-ui.",
      "신규 실험은 기존 포트를 덮어쓰지 말고 다음 빈 포트 또는 canary 포트에서 먼저 검증한다.",
      "모델 교체는 가능하면 포트 변경이 아니라 MODEL_ID 변경으로 처리한다.",
    ],
    1.08,
    4.28,
    5.24,
    0.44,
    COLORS.teal
  );
  addPanel(slide, 7.0, 3.72, 5.34, 2.36, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Current Repo Config", 7.2, 3.92, 1.62, COLORS.amberSoft, COLORS.rust);
  addBulletRows(
    slide,
    [
      "문서 baseline은 UI-Venus + MAI-UI dual start를 권장한다.",
      "하지만 현재 repo config는 ui-venus와 paddleocr-vl-1.5만 enabled=true 로 등록되어 있다.",
      "즉, 문서와 실제 서비스 활성 세트가 다를 수 있으므로 발표 시 둘을 구분해서 설명하는 편이 안전하다.",
    ],
    7.2,
    4.28,
    4.72,
    0.46,
    COLORS.rust
  );
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "03 / Layout & Config",
    "배포 레이아웃과 env 분리 원칙",
    "모델 파일, 공통 옵션, 모델별 override 를 나눠 두면 교체와 rollback 이 단순해진다.",
    "01, README"
  );
  addMetricCard(slide, "Model root", "/project/.../data/models/", "실제 weight 는 로컬 절대경로로만 관리한다.", 0.86, 1.7, 3.6, 1.42, COLORS.amberSoft, COLORS.rust);
  addMetricCard(slide, "Common env", "config/common.env", "HOST, DTYPE, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION 같은 공통값.", 4.86, 1.7, 3.6, 1.42, COLORS.tealSoft, COLORS.teal);
  addMetricCard(slide, "Model env", "config/models/*.env", "MODEL_ID, SERVED_MODEL_NAME, PORT, GPU_ID, CHAT_TEMPLATE override.", 8.86, 1.7, 3.6, 1.42, COLORS.greenSoft, COLORS.green);
  addCodeBox(
    slide,
    "common.env",
    "HOST=127.0.0.1\nDTYPE=bfloat16\nGPU_MEMORY_UTILIZATION=0.80\nMAX_MODEL_LEN=8192\nMAX_NUM_SEQS=8\nTENSOR_PARALLEL_SIZE=1",
    0.88,
    3.48,
    3.86,
    2.2
  );
  addCodeBox(
    slide,
    "ui-venus.env",
    "MODEL_ID=/project/.../UI-Venus-1.5-8B\nSERVED_MODEL_NAME=ui-venus-1.5-8b\nPORT=8001\nGPU_ID=0\nCHAT_TEMPLATE=\nTRUST_REMOTE_CODE=1",
    4.94,
    3.48,
    3.86,
    2.2
  );
  addPanel(slide, 9.02, 3.48, 3.46, 2.2, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Naming Rules", 9.22, 3.68, 1.28, COLORS.rustSoft, COLORS.red);
  addBulletRows(
    slide,
    [
      "운영 alias 는 ui-venus.env, mai-ui.env 처럼 짧게 유지",
      "size 연구는 family-size 규칙 사용: ui-venus-2b, ui-venus-30b",
      "rollback 은 MODEL_ID 경로만 이전 버전으로 돌리고 프로세스 재시작",
    ],
    9.22,
    4.02,
    2.9,
    0.46,
    COLORS.red
  );
  addBodyText(
    slide,
    "중요: serve_vlm.py 는 common.env 를 먼저 읽고 models/<instance>.env 를 나중에 읽는다. 그래서 size별 MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION, EXTRA_VLLM_ARGS override 가 자연스럽다.",
    0.92,
    6.0,
    11.0,
    0.44,
    10.6,
    COLORS.text,
    { bold: true }
  );
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "04 / Serve Flow",
    "서버 기동, 확인, 유지 절차",
    "긴 vllm serve 명령보다 wrapper 스크립트를 유지하는 편이 운영상 안정적이다.",
    "02, 06"
  );
  addMetricCard(slide, "1. Preflight", "nvidia-smi", "GPU 수, vllm CLI 존재, 8001/8002 포트 비어 있는지 확인", 0.88, 1.72, 2.86, 1.36, COLORS.amberSoft, COLORS.rust);
  addMetricCard(slide, "2. Start", "start_*.py", "start_ui_venus.py / start_mai_ui.py / start_model.py family size", 3.98, 1.72, 2.86, 1.36, COLORS.tealSoft, COLORS.teal);
  addMetricCard(slide, "3. Check", "/v1/models", "curl 또는 check_vlm.py 로 served-model-name 확인", 7.08, 1.72, 2.86, 1.36, COLORS.greenSoft, COLORS.green);
  addMetricCard(slide, "4. Keep simple", "one shell per model", "PoC 단계에서는 각 모델을 별도 셸 / 세션에서 유지", 10.18, 1.72, 2.26, 1.36, COLORS.rustSoft, COLORS.red);
  addCodeBox(
    slide,
    "Preflight",
    "nvidia-smi\npython -c \"import torch; print(torch.cuda.device_count())\"\nvllm --help\nss -ltn | grep 800",
    0.9,
    3.42,
    3.6,
    2.38
  );
  addCodeBox(
    slide,
    "Start",
    "python scripts/start_ui_venus.py\npython scripts/start_mai_ui.py\npython scripts/start_model.py ui-venus 30b",
    4.86,
    3.42,
    3.2,
    2.38
  );
  addCodeBox(
    slide,
    "Check",
    "curl http://127.0.0.1:8001/v1/models\npython scripts/check_vlm.py http://127.0.0.1:8001 ui-venus-1.5-8b",
    8.38,
    3.42,
    4.08,
    2.38
  );
  addBodyText(
    slide,
    "UI-TARS 메모: Qwen2.5-VL runtime 가용성, preprocessor_config.json / tokenizer_config.json / chat_template.json / shard 파일 존재 여부를 먼저 본다. CHAT_TEMPLATE 를 비워 두면 모델 디렉터리의 chat_template.json 을 자동 사용한다.",
    0.92,
    6.06,
    11.0,
    0.56,
    10.4,
    COLORS.text,
    { bold: true }
  );
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "05 / Repo Integration",
    "poc/work 와 Flask gateway 연결 방식",
    "이 저장소는 모델 다운로드가 아니라 endpoint 와 alias 를 바꾸는 방식으로 실험 전환한다.",
    "03, poc/work/config.py, poc/work/vlm_openai_client.py"
  );
  addPanel(slide, 0.88, 1.66, 3.24, 2.56, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Environment keys", 1.08, 1.86, 1.3, COLORS.tealSoft, COLORS.teal);
  addBulletRows(
    slide,
    [
      "VLM_API_URL 또는 VLM_API_BASE_URL",
      "VLM_API_KEY",
      "VLM_MODEL_NAME",
      "SAFE_MODE=true / USE_WEBP=true / MAX_IMAGE_SIZE=1280",
    ],
    1.08,
    2.2,
    2.72,
    0.44,
    COLORS.teal
  );
  addPanel(slide, 4.42, 1.66, 3.88, 2.56, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Direct port", 4.62, 1.86, 1.08, COLORS.amberSoft, COLORS.rust);
  addBodyText(slide, "UI-Venus", 4.64, 2.22, 0.9, 0.16, 11.5, COLORS.text, { bold: true });
  addBodyText(slide, "http://...:8001", 4.64, 2.46, 2.4, 0.16, 11.2, COLORS.muted, { fontFace: FONT_MONO });
  addBodyText(slide, "MAI-UI", 4.64, 2.88, 0.9, 0.16, 11.5, COLORS.text, { bold: true });
  addBodyText(slide, "http://...:8002", 4.64, 3.12, 2.4, 0.16, 11.2, COLORS.muted, { fontFace: FONT_MONO });
  addBodyText(slide, "클라이언트가 /v1 를 자동 보정하므로 base URL 만 넣으면 된다.", 4.64, 3.54, 3.08, 0.34, 10.6, COLORS.text);
  addPanel(slide, 8.62, 1.66, 3.84, 2.56, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Flask gateway", 8.82, 1.86, 1.26, COLORS.greenSoft, COLORS.green);
  addBodyText(slide, "http://.../api/vlm_serve/ui-venus", 8.84, 2.34, 3.0, 0.16, 10.7, COLORS.muted, { fontFace: FONT_MONO });
  addBodyText(slide, "http://.../api/vlm_serve/mai-ui", 8.84, 2.76, 3.0, 0.16, 10.7, COLORS.muted, { fontFace: FONT_MONO });
  addBodyText(slide, "coworkers 에게는 direct port 대신 proxy base URL 을 주는 것이 문서상 권장 패턴이다.", 8.84, 3.22, 3.0, 0.36, 10.6, COLORS.text);
  addCodeBox(
    slide,
    ".env.ui-venus",
    "VLM_API_URL=http://...:8001\nVLM_API_KEY=\nVLM_MODEL_NAME=ui-venus-1.5-8b\n\nSAFE_MODE=true\nUSE_WEBP=true\nMAX_IMAGE_SIZE=1280",
    0.9,
    4.62,
    4.14,
    1.78
  );
  addCodeBox(
    slide,
    "Gateway profile",
    "VLM_API_URL=http://.../api/vlm_serve/ui-venus\nVLM_API_KEY=\nVLM_MODEL_NAME=ui-venus-1.5-8b",
    5.32,
    4.62,
    3.24,
    1.78
  );
  addPanel(slide, 8.88, 4.62, 3.58, 1.78, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Switching rule", 9.08, 4.82, 1.12, COLORS.rustSoft, COLORS.red);
  addBulletRows(
    slide,
    [
      "실험별 .env 를 따로 두고 현재 활성 파일만 poc/work/.env 로 사용",
      "모델 교체와 프롬프트 변경을 한 번에 하지 않음",
      "로그에는 endpoint 와 model_name 을 같이 남김",
    ],
    9.08,
    5.14,
    3.0,
    0.36,
    COLORS.red
  );
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "06 / Model Tradeoffs",
    "UI-TARS, UI-Venus, MAI-UI 비교",
    "세 모델은 모두 GUI VLM 이지만 runtime 민감도와 제품 지향점이 다르다.",
    "05"
  );
  const modelCards = [
    {
      x: 0.9,
      title: "UI-TARS-1.5-7B",
      fill: COLORS.rustSoft,
      accent: COLORS.red,
      bullets: [
        "Qwen2.5-VL 기반으로 다른 두 모델과 베이스 runtime 계열이 다르다.",
        "computer-use / browser / phone / game / virtual-world reasoning 성향이 강하다.",
        "chat_template.json, preprocessor_config.json, shard 파일 등 runtime 준비 상태가 중요하다.",
      ],
    },
    {
      x: 4.42,
      title: "UI-Venus-1.5-8B",
      fill: COLORS.amberSoft,
      accent: COLORS.rust,
      bullets: [
        "Qwen3-VL 계열로 grounding + mobile + web 을 unified GUI agent 관점에서 묶는다.",
        "문서 baseline 의 첫 번째 운영 모델이며, 8001 기준 포트로 유지하는 편을 권장한다.",
        "브라우저/모바일 navigation 과 grounding 완성도 비교의 기준축 역할을 한다.",
      ],
    },
    {
      x: 7.94,
      title: "MAI-UI-8B",
      fill: COLORS.tealSoft,
      accent: COLORS.teal,
      bullets: [
        "Qwen3-VL 계열이지만 real-world deployment, agent-user interaction, MCP tool call 을 더 전면에 둔다.",
        "문서상 GPU 1 / 8002 로 배치해 UI-Venus 와 동일 스크린샷 A/B 비교를 권장한다.",
        "device-cloud collaboration 같은 운영 지향 표현이 많아 실서비스 비교축으로 유용하다.",
      ],
    },
  ];
  modelCards.forEach((card) => {
    addPanel(slide, card.x, 1.72, 3.0, 4.8, COLORS.panel, COLORS.border);
    addPanelTitle(slide, card.title, card.x + 0.18, 1.92, 1.9, card.fill, card.accent);
    addBulletRows(slide, card.bullets, card.x + 0.18, 2.42, 2.5, 0.78, card.accent);
  });
  addPanel(slide, 11.02, 1.72, 1.42, 4.8, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Ops note", 11.18, 1.92, 0.82, COLORS.greenSoft, COLORS.green);
  addBulletRows(
    slide,
    [
      "UI-TARS 는 runtime mismatch 를 먼저 의심",
      "UI-Venus / MAI-UI 는 same prompt A/B 기준축",
      "포트와 served-model-name 을 같이 기록",
    ],
    11.18,
    2.42,
    1.0,
    0.96,
    COLORS.green
  );
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "07 / Multi-size Research",
    "family-size 연구 규칙과 메모리 튜닝",
    "baseline 포트는 유지하고, variant 는 81xx/82xx/83xx 대역으로 분리한다.",
    "06, README, 01"
  );
  addPanel(slide, 0.9, 1.72, 3.16, 2.14, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Naming + scripts", 1.08, 1.92, 1.22, COLORS.amberSoft, COLORS.rust);
  addBulletRows(
    slide,
    [
      "instance = family-size",
      "prepare_research_envs.py 로 env 초안 생성",
      "start_model.py <family> <size> 로 generic start",
    ],
    1.08,
    2.28,
    2.62,
    0.46,
    COLORS.rust
  );
  addCodeBox(
    slide,
    "Commands",
    "python scripts/prepare_research_envs.py ui-venus\npython scripts/start_model.py ui-venus 2b\npython scripts/check_vlm.py http://127.0.0.1:8102 ui-venus-2b",
    0.92,
    4.14,
    3.16,
    1.9
  );
  addPanel(slide, 4.38, 1.72, 3.64, 4.32, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Port bands", 4.56, 1.92, 0.96, COLORS.tealSoft, COLORS.teal);
  addMetricCard(slide, "ui-venus", "8102 / 8107 / 8130", "2B / 7B / 30B", 4.58, 2.32, 3.24, 0.94, COLORS.tealSoft, COLORS.teal);
  addMetricCard(slide, "mai-ui", "8202 / 8207 / 8230", "2B / 7B / 30B", 4.58, 3.44, 3.24, 0.94, COLORS.amberSoft, COLORS.rust);
  addMetricCard(slide, "ui-tars", "8302 / 8307 / 8330", "2B / 7B / 30B", 4.58, 4.56, 3.24, 0.94, COLORS.greenSoft, COLORS.green);
  addPanel(slide, 8.34, 1.72, 4.12, 4.32, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Memory sizing", 8.52, 1.92, 1.16, COLORS.rustSoft, COLORS.red);
  addBulletRows(
    slide,
    [
      "2~3개 소형 모델 동시 구동 시 GPU_MEMORY_UTILIZATION 을 감으로 올리지 않는다.",
      "H200 140GB 기준 reserve = shared 8GiB, process 4GiB 를 기본값으로 둔다.",
      "2개 공유 시작값은 약 0.44, 3개 공유 시작값은 약 0.29.",
      "실패 시 MAX_NUM_SEQS 를 먼저 줄이고, 다음으로 MAX_MODEL_LEN=4096 을 검토한다.",
      "AUTO_TUNE_GPU_MEMORY_UTILIZATION=1 또는 GPU_MEMORY_UTILIZATION=auto 를 쓰면 serve_vlm.py 가 로컬 config.json 과 shard 크기를 읽어 자동 계산한다.",
    ],
    8.52,
    2.3,
    3.44,
    0.48,
    COLORS.red
  );
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "08 / OCR Split",
    "PaddleOCR-VL-1.5 와 GOT-OCR-2.0-hf 운영 분기",
    "두 OCR 모델은 같은 docs/deploy_vlms 아래에서 관리하지만, 배포 런타임은 다르게 보는 편이 맞다.",
    "07, 08, README"
  );
  addPanel(slide, 0.92, 1.72, 5.72, 4.86, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "PaddleOCR-VL-1.5", 1.1, 1.92, 1.5, COLORS.greenSoft, COLORS.green);
  addBulletRows(
    slide,
    [
      "현재 Linux 클라우드의 Python 3.11 + vLLM 0.17.0 + transformers 4.57.6 + torch 2.10.0 조합이면 기존 vLLM wrapper 에 편입 가능하다.",
      "MODEL_ID, PORT, GPU_ID, SERVED_MODEL_NAME 만 맞추면 serve_vlm.py 와 start_paddleocr_vl.py 를 그대로 재사용한다.",
      "STRICT_OFFLINE=1 이면 MODEL_ID 가 ALLOWED_MODEL_ROOT 아래에 있어야 한다.",
      "권장 포트는 8004 canary 이고, VLM_MODEL_NAME=paddleocr-vl-1.5 로 맞춘다.",
    ],
    1.12,
    2.34,
    5.1,
    0.54,
    COLORS.green
  );
  addCodeBox(
    slide,
    "Start",
    "python scripts/start_paddleocr_vl.py\ncurl http://127.0.0.1:8004/v1/models\npython scripts/check_vlm.py http://127.0.0.1:8004 paddleocr-vl-1.5",
    1.12,
    4.88,
    5.08,
    1.36
  );
  addPanel(slide, 6.9, 1.72, 5.56, 4.86, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "GOT-OCR-2.0-hf", 7.08, 1.92, 1.4, COLORS.rustSoft, COLORS.red);
  addBulletRows(
    slide,
    [
      "공식 사용 방식이 AutoProcessor / AutoModelForImageTextToText 중심이라 현재 문서 기준 vLLM 경로와 분리한다.",
      "현 시점 문서에서는 transformers 4.57.6 + torch 2.10.0 direct runtime smoke test 를 우선한다.",
      "run_got_ocr.py 와 config/models/got-ocr-2.0-hf.env 를 사용해 local or dedicated Python runtime 에서 먼저 검증한다.",
      "팀 공용 endpoint 가 필요해지면 작은 Flask/FastAPI wrapper 를 따로 추가하는 순서가 적절하다.",
    ],
    7.1,
    2.34,
    4.98,
    0.54,
    COLORS.red
  );
  addCodeBox(
    slide,
    "Smoke test",
    "python scripts/run_got_ocr.py\n# optional isolated env\nuv venv .venvs/got-ocr-2.0-hf --python python3.11",
    7.1,
    4.88,
    4.96,
    1.36
  );
  finalizeSlide(slide, pptx);

  slide = pptx.addSlide();
  addTitleBlock(
    slide,
    "09 / Ops Standards",
    "운영 표준과 트러블슈팅 기준점",
    "슬라이드 발표의 마무리는 '무엇을 고정하고 무엇을 실험하는가'를 분명히 보여주는 편이 좋다.",
    "03, 04, 06"
  );
  addPanel(slide, 0.88, 1.72, 4.1, 4.76, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Keep fixed", 1.08, 1.92, 1.0, COLORS.amberSoft, COLORS.rust);
  addBulletRows(
    slide,
    [
      "포트는 모델 식별자처럼 다룬다.",
      "baseline 은 8001 / 8002 / 8003 에 고정하고, size 연구는 81xx / 82xx / 83xx 로 분리한다.",
      "모델 revision 교체는 port 변경이 아니라 MODEL_ID 변경으로 처리한다.",
      "실험 로그에는 endpoint 와 model_name 을 함께 기록한다.",
      "SAFE_MODE, prompt, input image, post-processing 을 고정한 뒤 모델만 바꿔 비교한다.",
    ],
    1.08,
    2.28,
    3.42,
    0.5,
    COLORS.rust
  );
  addPanel(slide, 5.3, 1.72, 3.42, 2.2, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "When server does not start", 5.5, 1.92, 1.8, COLORS.rustSoft, COLORS.red);
  addBulletRows(
    slide,
    [
      "--trust-remote-code 누락 확인",
      "모델 경로 존재 여부 확인",
      "UI-TARS 는 qwen2_5_vl import 와 template / shard 파일 점검",
    ],
    5.5,
    2.26,
    2.76,
    0.44,
    COLORS.red
  );
  addPanel(slide, 5.3, 4.28, 3.42, 2.2, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "When memory is tight", 5.5, 4.48, 1.52, COLORS.tealSoft, COLORS.teal);
  addBulletRows(
    slide,
    [
      "1) MAX_MODEL_LEN=4096",
      "2) MAX_NUM_SEQS=4",
      "3) GPU_MEMORY_UTILIZATION=0.70 or auto-tune",
    ],
    5.5,
    4.82,
    2.72,
    0.44,
    COLORS.teal
  );
  addPanel(slide, 9.0, 1.72, 3.46, 2.2, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "When responses look wrong", 9.2, 1.92, 1.92, COLORS.greenSoft, COLORS.green);
  addBulletRows(
    slide,
    [
      "VLM_MODEL_NAME 과 served-model-name 일치 여부",
      "별도 chat template 요구 여부",
      "canary 포트에서 template 실험 후 운영 포트 승격",
    ],
    9.2,
    2.26,
    2.8,
    0.44,
    COLORS.green
  );
  addPanel(slide, 9.0, 4.28, 3.46, 2.2, COLORS.panel, COLORS.border);
  addPanelTitle(slide, "Recommended finish", 9.2, 4.48, 1.62, COLORS.amberSoft, COLORS.rust);
  addBulletRows(
    slide,
    [
      "1. UI-Venus 8001 baseline smoke test",
      "2. MAI-UI 8002 동일 이미지 A/B",
      "3. 더 좋은 쪽을 기준 포트로 유지",
      "4. 이후 UI-TARS 와 size variants 추가",
    ],
    9.2,
    4.82,
    2.82,
    0.4,
    COLORS.rust
  );
  finalizeSlide(slide, pptx);

  return pptx;
}

async function main() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const pptx = buildDeck();
  const outputPath = path.join(OUTPUT_DIR, "deploy_vlms_summary_deck.pptx");
  await pptx.writeFile({ fileName: outputPath });
  console.log(`[INFO] Wrote ${outputPath}`);
}

main().catch((error) => {
  console.error("[ERROR] Failed to build deck");
  console.error(error);
  process.exit(1);
});
