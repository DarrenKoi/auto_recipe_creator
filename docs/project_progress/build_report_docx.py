"""프로젝트 진행 보고서 Word(.docx) 생성 스크립트.

docs/project_progress/ 의 Markdown 보고서 내용을 하나의 임원 보고용 .docx 로 작성한다.
표지 -> 목차 -> 임원 요약 -> 워크스트림별 기술 부록 -> 현황/로드맵 순.

실행:
  uv run python docs/project_progress/build_report_docx.py
출력:
  docs/project_progress/project_progress_report.docx

본문은 한국어, 모델명/CV 기법/env 등 기술 용어는 영문 병기. 한글 폰트는 맑은 고딕.
수치/경로는 저장소 근거 문서(docs/setup_vlms, poc/workflow_2/docs, poc/workflow_3)에 따른다.
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUTPUT_PATH = Path(__file__).resolve().parent / "project_progress_report.docx"

KR_FONT = "맑은 고딕"
MONO_FONT = "Consolas"

NAVY = RGBColor(0x14, 0x2A, 0x55)
ACCENT = RGBColor(0xE8, 0x6A, 0x1F)
MUTED = RGBColor(0x55, 0x60, 0x70)


def _apply_font(run, *, name=KR_FONT, size=None, bold=None, color=None):
    """run 에 폰트를 적용한다. 한글은 w:eastAsia 속성까지 설정해야 제대로 렌더된다."""
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def _para(doc, text="", *, size=10.5, bold=False, color=None, align=None,
          space_after=6, space_before=0, mono=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    if align is not None:
        p.alignment = align
    if text:
        run = p.add_run(text)
        _apply_font(run, name=MONO_FONT if mono else KR_FONT, size=size,
                    bold=bold, color=color)
    return p


def _heading(doc, text, *, level=1):
    sizes = {1: 17, 2: 13.5, 3: 11.5}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    _apply_font(run, size=sizes.get(level, 12), bold=True,
                color=NAVY if level <= 2 else ACCENT)
    return p


def _bullet(doc, text, *, level=0, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Pt(18 + level * 14)
    run = p.add_run(text)
    _apply_font(run, size=size)
    return p


def _table(doc, headers, rows, *, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(h)
        _apply_font(run, size=9.5, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            _apply_font(run, size=9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table


def _set_cell_shading(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.makeelement(qn("w:shd"), {})
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def _shade_header(table, hex_color="142A55"):
    for cell in table.rows[0].cells:
        _set_cell_shading(cell, hex_color)


def _code_block(doc, lines):
    for line in lines:
        _para(doc, line, size=9, mono=True, color=MUTED, space_after=0)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


# ---------------------------------------------------------------------------
# 섹션 빌더
# ---------------------------------------------------------------------------


def build_cover(doc):
    for _ in range(4):
        doc.add_paragraph()
    _para(doc, "프로젝트 진행 보고서", size=30, bold=True, color=NAVY,
          align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
    _para(doc, "AI 기반 CD-SEM / VeritySEM Recipe 자동 Setup PoC", size=15,
          color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    _para(doc, "VLM 배포·운영  ·  workflow_1 / workflow_2 / workflow_3", size=12,
          color=ACCENT, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
    _para(doc, "목적 · PoC 방향 · 성과 · 확장성", size=11, color=MUTED,
          align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()


def build_toc(doc):
    _heading(doc, "목차 (Contents)", level=1)
    items = [
        "1. 임원 요약 (Executive Summary)",
        "2. VLM 배포·운영 (deploy_vlms + Flask Proxy)",
        "3. workflow_1 — RCS GUI 자동화 + CCTV 캡처 PoC",
        "4. workflow_2 — 오프라인 CV 평가 벤치",
        "5. workflow_3 — 실시간 Align Fail 모니터링 루프 (현재 주력)",
        "6. 현황 & 로드맵 (Status & Roadmap)",
    ]
    for it in items:
        _para(doc, it, size=11, space_after=4)
    doc.add_page_break()


def build_executive_summary(doc):
    _heading(doc, "1. 임원 요약 (Executive Summary)", level=1)

    _heading(doc, "1.1 프로젝트 목적", level=2)
    _para(doc, "CD-SEM / VeritySEM 계측 장비의 recipe setup을 사람이 수동으로 만드는 과정을 "
               "AI로 자동화한다. 특히 계측 중 빈번한 Align Fail(정렬 실패, ALID=9006)을 무인으로 "
               "감지·보정하여 엔지니어 개입과 장비 idle 시간을 줄이는 것이 1차 목표다.")
    _para(doc, "문제의 난이도:", bold=True, space_after=2)
    _bullet(doc, "RCS(Recipe Control System)는 legacy GUI 클라이언트로 accessibility tree가 "
                 "부실해 일반 UI 자동화(UIA)로 안정 제어가 어렵다.")
    _bullet(doc, "Align key는 화면 contrast·밝기·반복 패턴이 매 측정마다 달라 단순 좌표 인식이 통하지 않는다.")
    _bullet(doc, "수동 대응은 야간/주말 무인 운영 불가, 실패 원인 분석용 화면 증거도 사후 소실.")

    _heading(doc, "1.2 PoC 핵심 방향 — VLM과 CV의 역할 분리", level=2)
    _para(doc, "일관된 설계 원칙(2026-05-25 확정): OpenCV(CV)는 정량 점수와 최종 좌표를 만들고, "
               "VLM은 영역 식별·모호성 설명·보정 가능성 판단을 한다. 낮은 CV 점수를 VLM 답변이 "
               "덮어쓰거나 반복 가능한 단계 전환을 VLM이 결정하게 하지 않는다.")
    t = _table(doc, ["구분", "담당", "이유"], [
        ["VLM", "\"어디를 볼까\" — UI 위치, align key 후보 영역, OM/SEM 모드 판독, 모호성 설명",
         "화면 의미 이해에 강함 (단, stateless·픽셀 한계)"],
        ["CV (OpenCV)", "\"얼마나 닮았나 / 정확히 어디\" — template matching 점수, 최종 클릭 좌표",
         "반복 가능·정량적, stateless VLM의 기억을 외부 상태로 대신"],
    ])
    _shade_header(t)

    _heading(doc, "1.3 단계별 진행", level=2)
    t = _table(doc, ["단계", "워크스트림", "한 일", "상태"], [
        ["0", "deploy_vlms", "오픈소스 VLM 5종 조사·선정 -> 사내 HCP(H200×2)에 vLLM 설치, Flask proxy 통합", "운영 중"],
        ["1", "workflow_1", "RCS GUI 자동화(2-stage VLM) + Align Fail 감지 + CCTV 캡처 PoC", "동결(검증 완료)"],
        ["2", "workflow_2", "오프라인 CV 평가 벤치 — golden set으로 matching/consensus A/B·튜닝", "활성"],
        ["3", "workflow_3", "통합 production 실시간 align-fail 모니터링 루프 (현재 주력)", "구현 완료, 활성화 중"],
    ])
    _shade_header(t)
    _para(doc, "설계 흐름: VLM 인프라 확보(0) -> GUI 자동화 가능성 증명(1) -> CV 정확도 확보(2) "
               "-> 실시간 통합 루프(3).", color=MUTED, size=10)

    _heading(doc, "1.4 핵심 성과 (Highlights)", level=2)
    _bullet(doc, "VLM 운영 효율: 5개 특화 모델을 H200 2장으로 운영. 단일 프런티어 모델(Kimi-K2, "
                 "1.04T MoE)은 동일 작업에 H200 약 8장 필요 -> 하드웨어 약 4배, 밀도·풋프린트 약 20배 절감.")
    _bullet(doc, "GUI 자동화 가능성 증명(wf1): coarse(UI-Venus)->fine(MAI-UI)->OCR 검증 파이프라인으로 "
                 "UIA 없이 RCS 화면 좌표를 안정적으로 클릭 가능함을 입증.")
    _bullet(doc, "정렬 정확도 도약(wf2): rcp 단독 대비 consensus template 라우팅으로 align key "
                 "탐색 recall(in_topk) 0.434 -> 0.876, rank1 0.318 -> 0.764 (golden set 벤치, min_s=3).")
    _bullet(doc, "통합 루프 구현(wf3): 알람->RCS 접속->CV 보정->실패 시 cube 알림->상시 녹화->자동 종료 "
                 "end-to-end 루프와 4-layer 모듈 아키텍처 구축.")

    _heading(doc, "1.5 확장성 (Scalability)", level=2)
    _bullet(doc, "모델 추가 용이: Flask proxy에 .env + 모듈 1개 등록으로 6번째 모델 추가(GPU 1에 ~50 GiB 헤드룸).")
    _bullet(doc, "장비·recipe 일반화: consensus 이력 풀이 <class>/<recipe> 키(장비 무관) -> 같은 recipe면 "
                 "여러 장비가 학습 데이터 공유, 신규 장비 onboarding 시 코드 변경 최소.")
    _bullet(doc, "데이터 자산화: 상시 녹화가 엔지니어 수동 보정 조작까지 프레임 보존 -> 모방학습/원인 분류 학습 데이터.")
    _bullet(doc, "벤치->프로덕션 파이프라인: workflow_2 golden 검증 후 bit-parity 포팅 -> 회귀 위험 통제.")
    doc.add_page_break()


def build_vlm_deployment(doc):
    _heading(doc, "2. VLM 배포·운영 (deploy_vlms + Flask Proxy)", level=1)
    _para(doc, "목적: 오픈소스 VLM을 조사·선정하고 사내 HCP(GPU 서버)에 설치·운영하는 기반을 구축한다. "
               "이후 모든 workflow가 이 인프라로 화면을 이해한다.", color=MUTED, size=10)

    _heading(doc, "2.1 왜 다중 특화 모델인가", level=2)
    _bullet(doc, "전체 화면 클릭 좌표를 한 번에 픽셀 정확도로 맞히기 어려움 -> coarse/fine 모델 분리.")
    _bullet(doc, "텍스트 검증·OCR은 전용 OCR 모델이 더 정확·경량.")
    _bullet(doc, "거대 단일 모델은 하드웨어 비용이 비현실적(아래 2.5).")

    _heading(doc, "2.2 선정 모델 5종과 역할", level=2)
    t = _table(doc, ["모델", "파라미터", "Port", "역할 (route_slug)"], [
        ["UI-Venus-1.5-8B", "8.3 B", "8001", "메인 GUI grounding — coarse bbox (ui-venus)"],
        ["MAI-UI-8B", "8 B", "8002", "정밀 클릭점 — crop 확대 후 (mai-ui)"],
        ["UI-TARS-1.5-7B", "7.6 B", "8003", "GUI agent 대안 (Qwen2.5-VL) (ui-tars)"],
        ["PaddleOCR-VL-1.5", "0.9 B", "8004", "OCR 보조 — Spotting/표 (paddleocr-vl-1.5)"],
        ["GOT-OCR-2.0-hf", "0.58 B", "8005", "하드 OCR fallback (transformers) (got-ocr)"],
    ])
    _shade_header(t)
    _para(doc, "대표 파이프라인: Capture -> UI-Venus(coarse bbox) -> Crop&Zoom -> MAI-UI(refined point) "
               "-> PaddleOCR-VL(입력 검증).", size=10, color=MUTED)

    _heading(doc, "2.3 사내 HCP 설치 구성", level=2)
    _bullet(doc, "하드웨어: H200 140 GiB × 2. 서빙: vLLM(BF16) 4종 + GOT-OCR(transformers).")
    _bullet(doc, "GPU 0: UI-Venus + UI-TARS(각 u≈0.44 auto-tune) -> ~123 GiB(88%).")
    _bullet(doc, "GPU 1: MAI-UI(u=0.45) + PaddleOCR-VL(u=0.10) + GOT-OCR(~4 GiB) -> ~81 GiB(58%).")
    _bullet(doc, "오프라인 정책: HF live pull 금지(사전 stage), telemetry 비활성, 절대 경로.")
    _bullet(doc, "vLLM 활용: PagedAttention, continuous batching, prefix caching(고정 prompt 재사용).")

    _heading(doc, "2.4 Flask Proxy — 통합 라우팅·헬스", level=2)
    _bullet(doc, "서비스 레지스트리: config.py의 ALL_VLM_SERVICES (route_slug/port/enabled).")
    _bullet(doc, "proxy URL: {flask_base}/api/vlm_serve/{slug}/v1/chat/completions.")
    _bullet(doc, "health: GET /api/vlm_serve/health — 각 upstream /v1/models probe.")
    _bullet(doc, "클라이언트는 모델명이 아니라 route_slug로 호출 -> 모델 교체에 무관.")

    _heading(doc, "2.5 단일 거대 모델 대비 효율 (Kimi-K2 비교)", level=2)
    t = _table(doc, ["지표", "본 스택 (특화 5종)", "Kimi-K2 (FP8)"], [
        ["필요 하드웨어", "H200 2장", "H200 약 8장"],
        ["GPU당 모델 밀도", "2.5 모델/H200", "0.125 모델/H200"],
        ["가중치 풋프린트", "~51 GiB", "~1,040 GiB"],
        ["레이턴시(짧은 JSON)", "~80–150 tok/s/req", "~30–60 tok/s/req"],
    ])
    _shade_header(t)
    _para(doc, "핵심: GUI grounding + OCR 작업 표면에 한정하면 하드웨어 약 4배, 밀도·풋프린트 약 20배 절감. "
               "포기하는 것은 범용 추론 능력 — 의도적 트레이드오프. (근거: docs/setup_vlms/05)",
          size=10, color=MUTED)
    doc.add_page_break()


def build_workflow_1(doc):
    _heading(doc, "3. workflow_1 — RCS GUI 자동화 + CCTV 캡처 PoC", level=1)
    _para(doc, "목적: VLM으로 RCS 화면을 이해해 GUI를 자동 조작할 수 있는지, Align Fail 시점 증거(CCTV)를 "
               "무인 보존할 수 있는지 증명. 현재 동결(frozen), production은 wf3로 이전.", color=MUTED, size=10)

    _heading(doc, "3.1 무엇을 만들었나 (5단계 파이프라인)", level=2)
    for i, txt in enumerate([
        "RCS 로그인 & Tool List 진입",
        "Align Fail 감지 — 알람 API 1~2분 폴링, ALID=9006만 필터",
        "CCTV/DVR 진입 — Tool DVR 자동 오픈, Channel 4 확대",
        "프레임 캡처 — 최대 8분(~4,800 프레임) 100ms 간격 저장",
        "알림 & 로그 — Windows 팝업 + 누적 로그, 중복은 edge-trigger",
    ], start=1):
        _bullet(doc, f"{i}. {txt}")

    _heading(doc, "3.2 핵심 기술 — 2-stage VLM 로케이터", level=2)
    _bullet(doc, "Coarse(UI-Venus): 전체 화면에서 타겟 대략 bbox(bbox_1000).")
    _bullet(doc, "Crop & Zoom: coarse bbox에 padding 더해 확대.")
    _bullet(doc, "Fine(MAI-UI): zoom crop에서 픽셀 클릭점 -> 원본 좌표 역변환.")
    _bullet(doc, "Verify(PaddleOCR-VL): 입력 후 OCR 재확인 -> closed-loop 자동화.")
    _bullet(doc, "DPI 좌표 변환(125/150% 대응), Tool 이름 OCR 정규화, edge-trigger 폴링.")

    _heading(doc, "3.3 데이터 자산 — align_images 파일시스템 계약", level=2)
    _code_block(doc, [
        "align_images/<eqp_id>/<class>/<recipe>/",
        "├─ align_img_from_rcp/   IMAP0001.*(OM) IMAP0002.*(SEM)  # 등록 align key",
        "├─ align_img_from_msr/   S*/E*                           # 측정 궤적 (E=fail)",
        "└─ captured_img_from_rcs/ <tag>/…                        # fail 캡처 + recording/",
    ])
    _para(doc, "OM(저배율·반복)과 SEM(고배율·sparse edge)은 별도 modality. cond.txt 사이드카에 "
               "crosshair·box 좌표 등 조건 메타데이터 저장.", size=10, color=MUTED)

    _heading(doc, "3.4 증명한 것과 한계", level=2)
    _para(doc, "증명한 것:", bold=True, space_after=2)
    _bullet(doc, "VLM coarse->fine 좌표 인식이 UIA 없이도 RCS UI를 안정적으로 클릭할 만큼 견고.")
    _bullet(doc, "Align Fail 시점 챔버 영상 100% 자동 보존 -> 무인 모니터링, 원인 분석 시간 단축.")
    _para(doc, "한계 / 배운 점:", bold=True, space_after=2)
    _bullet(doc, "GUI 화면 읽기 != align key 찾기 — 정밀 매칭은 CV 영역(-> wf2/wf3).")
    _bullet(doc, "전 단계 VLM 의존은 비용·지연·edge-case 부담 -> full 자동화 경제성 한계.")
    doc.add_page_break()


def build_workflow_2(doc):
    _heading(doc, "4. workflow_2 — 오프라인 CV 평가 벤치", level=1)
    _para(doc, "목적: align key matching 정확도 향상 CV 변경을 golden set으로 객관 검증·A/B·튜닝한 뒤 "
               "검증된 변경만 production으로 포팅. 동결 아님 — 활성 연구·검증 harness.", color=MUTED, size=10)

    _heading(doc, "4.1 역할과 원칙", level=2)
    _bullet(doc, "production 엔진 무수정 — poc.workflow_3.align을 import만(반대 금지), 실험은 ensemble_lab fork.")
    _bullet(doc, "\"가정 말고 측정\" — 정책 변경 전 golden set 수치 우선 확인.")
    _bullet(doc, "벤치->프로덕션 파이프라인으로 회귀 위험 통제.")

    _heading(doc, "4.2 3대 Golden Driver & CV 기법", level=2)
    _bullet(doc, "golden_localization / golden_consensus / golden_combined(라우팅 end-to-end + 3축 + OM/SEM 층화).")
    _bullet(doc, "Ensemble proposer 3채널(Canny/Scharr/orientation + Chamfer) -> RRF 융합 -> NCC rerank.")
    _bullet(doc, "Youden 임계: match=0.6053, adjust=0.4727.")
    _bullet(doc, "Consensus re-registration: 최근 성공 S crop을 crosshair 기준 co-register 후 median blend.")
    _bullet(doc, "평가 지표: in_topk(proposer recall), rank1(top 후보 정답). 방법론: cond.txt GT + LOO.")

    _heading(doc, "4.3 측정 결과 (golden set 벤치)", level=2)
    t = _table(doc, ["지표", "rcp 단독", "consensus 라우팅", "비고"], [
        ["in_topk (recall)", "0.434", "0.876", "+0.442 (약 +102%)"],
        ["rank1 (top 정답)", "0.318", "0.764", "+0.446"],
    ])
    _shade_header(t)
    _para(doc, "LOO A/B, min_s=3 기준. production 기본 min_s=4는 의도적 보수 정책.", size=10, color=MUTED)
    _para(doc, "주의: 위 수치는 golden set 벤치 기준이다. 오피스 실데이터의 라우팅 종합 정확도·OM/SEM 층화 "
               "판정은 office GOLDEN_ROOT/HISTORY_ROOT 데이터에서 golden_combined_eval_cond.py 실행 후 "
               "[DIGEST] 한 줄로 확정 예정.", size=10, color=ACCENT)

    _heading(doc, "4.4 의의", level=2)
    _bullet(doc, "consensus re-registration은 정렬 정확도의 천장(rcp 단독 ~0.43)을 뚫은 핵심 발견.")
    _bullet(doc, "bench/config 분리 + [DIGEST] 한 줄 회신으로 오피스<->개발 결과 전달 비용 최소화.")
    doc.add_page_break()


def build_workflow_3(doc):
    _heading(doc, "5. workflow_3 — 실시간 Align Fail 모니터링 루프 (현재 주력)", level=1)
    _para(doc, "목적: wf1(GUI 자동화) + wf2(CV 보정)의 production 경로를 하나의 end-to-end 실시간 루프로 통합.",
          color=MUTED, size=10)

    _heading(doc, "5.1 루프 개요", level=2)
    _code_block(doc, [
        "알람 감지(ALID=9006) → RCS 장비 접속 → CV align fail 보정",
        "→ 실패 시 cube rich notification → 상시 screenshot 녹화(수동 조작 포함)",
        "→ tool 닫기 → 다음 장비 대기",
    ])
    _bullet(doc, "popup 직후 daemon thread로 consensus gather가 겹쳐 실행(최근 S 이미지 stage).")
    _bullet(doc, "office 모듈 부재 시 자동 비활성 -> 기존 동작·루프 응답성 불변(회귀 위험 0).")

    _heading(doc, "5.2 4-Layer 모듈 아키텍처", level=2)
    t = _table(doc, ["서브패키지", "내용"], [
        ["monitor/", "루프 본체 — 폴링·사이클·상시 녹화·알림·엔지니어done·office adapter"],
        ["rcs/", "RCS GUI 자동화 — 실행/로그인/tool 선택·종료/캡처"],
        ["align/ (+matching/diagnostics)", "보정 도메인 — 자산·correction·live_search·consensus·matcher 엔진"],
        ["sem_monitor/", "SEM panel 위치 검출 + 실장비 controller adapter"],
        ["vlm/ · runner/ · util/", "VLM 클라이언트 / WorkflowRunner·journal / env·image·json 공통"],
    ])
    _shade_header(t)
    _para(doc, "의존 방향: monitor -> {rcs, align, sem_monitor, runner, vlm, util}. wf3는 wf1/2를 import 안 함.",
          size=10, color=MUTED)

    _heading(doc, "5.3 핵심 능력", level=2)
    _bullet(doc, "Per-alarm cycle + 보장 teardown — cleanup은 try/finally로 반드시 실행.")
    _bullet(doc, "Consensus 라우팅 보정 — modality별 consensus-or-rcp 폴백(회귀 0), CONSENSUS=0 킬스위치.")
    _bullet(doc, "상시 녹화 — 변화 감지 적응 캡처, 엔지니어 수동 조작까지 프레임 보존.")
    _bullet(doc, "Engineer-done 감지 — Recipe Monitor 카운터(N/M) hybrid 판독으로 watch 조기 종료.")
    _bullet(doc, "Feasibility 판정 — possible/not_visible/ambiguous, 모호 시 재등록 권고 audit.")
    _bullet(doc, "Zoom ladder / PM dropdown — 배율 바꿔가며 재매칭(어느 배율에서 key 보이는지).")
    _bullet(doc, "Check-only 변형 — 접속->1프레임->feasibility->닫기(진단·캘리브레이션용).")

    _heading(doc, "5.4 안전 장치 (Safe-mode gating)", level=2)
    t = _table(doc, ["env", "기본", "의미"], [
        ["SAFE_MODE", "0", "1이면 모든 마우스/키보드 차단(전역 dry-run)"],
        ["ALIGN_FAIL_CORRECTION", "1", "CV 보정 단계 수행 여부"],
        ["ALIGN_FAIL_CORRECTION_DRY_RUN", "1", "실클릭은 SAFE_MODE=0 그리고 이 값=0일 때만"],
        ["ALIGN_FAIL_CONSENSUS", "1", "consensus 라우팅 마스터 토글(킬스위치)"],
        ["ALIGN_FAIL_ENGINEER_DONE_DETECT", "0", "측정-시작 감지(캘리브레이션 후 1)"],
    ])
    _shade_header(t)
    _para(doc, "실보정은 두 단계 게이트(SAFE_MODE=0 + DRY_RUN=0)를 모두 통과해야만 작동.", size=10, color=MUTED)
    doc.add_page_break()


def build_status_roadmap(doc):
    _heading(doc, "6. 현황 & 로드맵 (Status & Roadmap)", level=1)

    _heading(doc, "6.1 현황 한눈에", level=2)
    t = _table(doc, ["영역", "상태", "비고"], [
        ["VLM 배포·운영", "완료", "H200×2, 5 모델 운영 중"],
        ["workflow_1 GUI 자동화 + CCTV", "완료", "동결, production은 wf3"],
        ["workflow_2 CV 평가 벤치", "활성", "오피스 golden 실측 대기"],
        ["wf3 루프·보정·녹화·알림(코드)", "완료", "dry-run 게이트"],
        ["consensus 라이브 보정 활성화", "대기", "office_success_downloader 구현 필요"],
        ["engineer-done / SEM-box 캘리브레이션", "진행 중", "모델별 landmark·배율 비율"],
        ["pilot 실보정(actuation)", "대기", "dry-run 후 단일 장비부터"],
    ])
    _shade_header(t)

    _heading(doc, "6.2 오피스 PC 이전 체크리스트 (요약)", level=2)
    for i, txt in enumerate([
        "office_* 복사 (office_align_fail_alarm, office_rich_notify -> monitor/)",
        "office_success_downloader.py 신규 작성 — consensus 라이브 보정 활성화 게이트",
        "office_rcp_msr_downloader.py (선택, MES 직접 적재 불가 환경만)",
        "import sweep / SAFE_MODE=1 dry-run / record-only 패리티 / 보정 dry-run",
        "캘리브레이션(SEM panel landmark, 더블클릭 recenter, wheel↔배율, read_mode)",
        "pilot actuation — 단일 장비 DRY_RUN=0",
    ], start=1):
        _bullet(doc, f"{i}. {txt}")

    _heading(doc, "6.3 확장성 (Scalability)", level=2)
    _bullet(doc, "모델 확장: Flask proxy에 .env+모듈 1개로 6번째 모델(GPU 1에 ~50 GiB 헤드룸), multi-GPU parallel.")
    _bullet(doc, "장비·recipe 일반화: consensus 이력 풀이 <class>/<recipe> 키(장비 무관) -> 학습 데이터 공유.")
    _bullet(doc, "데이터 자산화: 상시 녹화 + recording_filter로 interaction timeline 추출 -> 모방학습 데이터.")
    _bullet(doc, "VLM ROI prior: align key 영역을 VLM이 grounding해 CV 탐색 범위 축소(2-tier fallback) — 진행 중.")

    _heading(doc, "6.4 리스크 & 완화", level=2)
    t = _table(doc, ["리스크", "완화", "상태"], [
        ["office downloader 부재", "자동 비활성 -> 기존 rcp 경로 동일 동작", "OK"],
        ["cold consensus cache", "bounded sync 8s 후 rcp 폴백", "OK"],
        ["정확도 수치가 벤치 기준", "오피스 실데이터 [DIGEST]로 확정 예정", "진행 중"],
        ["SEM-box 미캘리브레이션", "dry-run 게이트(실클릭 차단)", "진행 중"],
        ["실보정 사고", "2단계 게이트 + pilot 단일 장비", "OK"],
    ])
    _shade_header(t)

    _heading(doc, "6.5 다음 주 우선순위", level=2)
    for i, txt in enumerate([
        "office_success_downloader 구현 -> consensus 라이브 보정 활성화",
        "오피스 캘리브레이션(zoom/click 좌표, engineer-done) 완료",
        "golden_combined_eval_cond.py 오피스 실행 -> OM/SEM 층화·라우팅 정확도 확정",
        "pilot 장비 1대 dry-run -> 실보정 단계적 전환",
    ], start=1):
        _bullet(doc, f"{i}. {txt}")


def main():
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = KR_FONT
    normal.font.size = Pt(10.5)
    normal_rpr = normal.element.get_or_add_rPr()
    rfonts = normal_rpr.makeelement(qn("w:rFonts"), {})
    rfonts.set(qn("w:eastAsia"), KR_FONT)
    normal_rpr.append(rfonts)

    build_cover(doc)
    build_toc(doc)
    build_executive_summary(doc)
    build_vlm_deployment(doc)
    build_workflow_1(doc)
    build_workflow_2(doc)
    build_workflow_3(doc)
    build_status_roadmap(doc)

    doc.save(OUTPUT_PATH)
    print(f"[INFO] DOCX saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
