# -*- coding: utf-8 -*-
"""Align Tuning Agent 경과 보고 - Bento 슬라이드(.bento.html) 생성 스크립트.

임원 보고용 **본편 4장** + 클릭으로만 열리는 **백업 state 2장**. 근거는 같은 폴더의
00~04 Markdown 보고서이며, 수치를 고칠 때는 .md 를 먼저 고치고 본 스크립트를 맞춘다.

Bento 덱은 런타임(앱)이 동봉된 단일 HTML 파일이고, 문서는 그 안의 id="bento-doc" script
블록에 든 JSON 하나가 전부다. 본 스크립트는 그 블록만 통째로 교체한다(런타임은 건드리지
않는다). 파일이 700KB 대이고 JSON 이 한 줄이라 git diff 로는 내용 변화를 읽을 수 없으므로,
수정은 반드시 이 스크립트로 한다.

대상 파일이 없으면 최신 릴리스를 먼저 내려받을 것:
  curl -fsSL https://bento.page/releases/slides/Bento_Slides.bento.html \
    -o docs/project_progress/Align_Tuning_Agent.bento.html

실행:
  uv run python docs/project_progress/build_slides_bento.py
출력:
  docs/project_progress/Align_Tuning_Agent.bento.html (in-place 갱신)

같은 내용의 PowerPoint 판은 짝 스크립트 build_slides_pptx.py 참조.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "Align_Tuning_Agent.bento.html"

BG      = "#0B1220"
PANEL   = "#111C2E"
TEXTC   = "#E8EDF5"
MUTED   = "#93A2B8"
DIM     = "#6F8098"
ACC     = "#4CC9F0"
ACC_DIM = "rgba(76,201,240,0.16)"
CARD    = "rgba(255,255,255,0.045)"
CARDB   = "rgba(76,201,240,0.22)"
LINE    = "rgba(255,255,255,0.10)"
NEUT    = "#3B4A63"
FONT    = "Pretendard, 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', system-ui, sans-serif"


def T(i, x, y, w, h, html, size=16, weight=600, color=TEXTC, align="left",
      valign="top", lh=1.45, ls=None, fx=None, link=None):
    e = {"id": i, "type": "text", "x": x, "y": y, "w": w, "h": h, "rotation": 0,
         "opacity": 1, "html": html, "fontSize": size, "fontFamily": FONT,
         "fontWeight": weight, "color": color, "align": align, "valign": valign,
         "lineHeight": lh}
    if ls is not None:
        e["letterSpacing"] = ls
    if fx:
        e["fx"] = fx
    if link:
        e["link"] = link
    return e


def R(i, x, y, w, h, fill, stroke="none", sw=0, radius=12, fx=None, link=None, grad=None):
    e = {"id": i, "type": "shape", "shape": "rect", "x": x, "y": y, "w": w, "h": h,
         "rotation": 0, "opacity": 1, "fill": fill, "stroke": stroke,
         "strokeWidth": sw, "radius": radius}
    if grad:
        e["fillGradient"] = grad
    if fx:
        e["fx"] = fx
    if link:
        e["link"] = link
    return e


def LN(i, x, y, w, color=ACC, sw=2, style="dashed", fx=None):
    return {"id": i, "type": "shape", "shape": "line", "x": x, "y": y, "w": w, "h": 2,
            "rotation": 0, "opacity": 1, "fill": "none", "stroke": color,
            "strokeWidth": sw, "strokeStyle": style, "radius": 0,
            **({"fx": fx} if fx else {})}


MARCH = {"loop": {"type": "dash-march", "distance": 16, "duration": 1.6}}

TBL_STYLE = {
    "headerBg": "rgba(76,201,240,0.14)", "headerColor": ACC,
    "zebra": "rgba(255,255,255,0.03)", "borderColor": "rgba(255,255,255,0.10)",
    "borderWidth": 1, "cellPadX": 14, "cellPadY": 9, "fontSize": 15,
    "color": TEXTC, "radius": 10,
}


def table(i, x, y, w, h, cols, rows, style=None, fs=15, padY=9):
    st = dict(TBL_STYLE)
    st["fontSize"] = fs
    st["cellPadY"] = padY
    if style:
        st.update(style)
    return {"id": i, "type": "table", "x": x, "y": y, "w": w, "h": h, "rotation": 0,
            "opacity": 1, "header": True, "columns": [{"w": c} for c in cols],
            "rows": [{"cells": [c if isinstance(c, dict) else {"html": c} for c in r]} for r in rows],
            "style": st}


slides = []

# ───────────────────────────────────────────────────────────── S1 · Head Message
s1 = []
s1.append(R("bg", 0, 0, 1280, 720, BG, radius=0,
            grad={"angle": 135, "stops": [{"at": 0, "color": "#0B1220"},
                                          {"at": 1, "color": "#132238"}]}))
s1.append({"id": "glow", "type": "shape", "shape": "ellipse", "x": 830, "y": -220,
           "w": 700, "h": 700, "rotation": 0, "opacity": 1,
           "fill": "rgba(76,201,240,0.10)", "stroke": "none", "strokeWidth": 0, "radius": 0})
s1.append(T("kick", 96, 88, 700, 32, "과제 종류 · 현업&nbsp;&nbsp;|&nbsp;&nbsp;경과 보고",
            20, 700, ACC, ls=3))
s1.append(T("title", 96, 128, 1000, 110, "Align Tuning Agent", 92, 900, TEXTC, lh=1.05))
s1.append(R("rule", 96, 250, 110, 6, ACC, radius=3))
s1.append(T("sub", 96, 282, 1010, 76,
            "Align Fail 무인 대응 AI Agent &nbsp;·&nbsp; 화면을 이해하는 VLM과 좌표를 결정하는 CV의 역할 분리 설계",
            22, 600, MUTED, lh=1.5))
s1.append(T("head", 96, 352, 1088, 60,
            "수동 <b>5분</b> 대응을 무인 <b>1분</b> 대응으로 — 회수한 장비 가동 시간으로 <b>주 WAFER 100장</b>을 더 측정합니다.",
            29, 800, TEXTC, lh=1.4, fx={"enter": "fade-up", "order": 1}))

kpis = [
    ("4분", "1건당 대응 지연 단축<br>감지~보정 5분 → 1분"),
    ("주 100장", "추가 측정 가능한 WAFER<br>400건 × 4분 ÷ 15분/장"),
    ("28대", "대상 계측 장비<br>R3 CD-SEM"),
]
for n, (v, lab) in enumerate(kpis):
    x = 96 + n * 374
    s1.append(R("k%dbox" % n, x, 434, 340, 132, CARD, CARDB, 1, 14,
                fx={"enter": "fade-up", "order": 2 + n}))
    s1.append(T("k%dv" % n, x + 22, 452, 296, 56, v, 46, 900, ACC,
                fx={"enter": "fade-up", "order": 2 + n, "countUp": True}))
    s1.append(T("k%dl" % n, x + 22, 512, 296, 46, lab, 14, 600, MUTED, lh=1.5))

s1.append(R("endbar", 96, 594, 1088, 78, ACC_DIM, "rgba(76,201,240,0.40)", 1, 12))
s1.append(T("endt", 120, 610, 380, 34, "예상 종료 · <b>2026년 9월 말</b>", 24, 800, ACC))
s1.append(T("endd", 500, 612, 660, 50,
            "현업 엔지니어 합동 실전 테스트와 단일 장비 Pilot 완료 시점<br>이후 4Q 대상 장비 확대 · 2027년 Recipe Tuning 확장",
            14, 600, MUTED, lh=1.5))
slides.append({"id": "s1", "background": BG, "transition": "none", "elements": s1, "notes":
    "표지 겸 Head Message. 핵심 한 줄: 주 400건의 Align Fail에 대해 1건당 대응 지연을 5분에서 1분으로 줄여 "
    "주 WAFER 약 100장의 측정 여력을 회수한다. 대상은 R3 CD-SEM 28대. 예상 종료는 2026년 9월 말 — "
    "현업 합동 실전 테스트와 단일 장비 Pilot 완료를 과제 종료 시점으로 본다. "
    "환산 근거를 물으면: 400건/주 × 4분 단축 ÷ 15분/장 ≈ 주 100장."})

# ──────────────────────────────────────────────── S2 · 선정 배경 / 목적
s2 = []
s2.append(T("t", 96, 52, 900, 46, "선정 배경과 목적", 38, 900, TEXTC))
s2.append(T("st", 96, 100, 1088, 30,
            "Align Fail이 나면 장비가 멈추고, Line 구성원이 직접 접속해 정렬을 다시 맞춰야 재개됩니다.",
            17, 600, MUTED))

s2.append(T("lh", 96, 152, 520, 28, "AS-IS · 현재의 문제", 15, 800, ACC, ls=2))
probs = [
    ("주 400건", "R3 CD-SEM 28대에서 발생하는 Align Fail 알람"),
    ("1건당 5분", "Line 구성원의 감지 → RCS 접속 → 수동 재정렬"),
    ("주 33시간", "누적 대응 지연 — 그만큼 장비가 멈춰 있습니다"),
]
for n, (v, lab) in enumerate(probs):
    y = 190 + n * 86
    s2.append(R("p%db" % n, 96, y, 520, 74, CARD, "rgba(255,255,255,0.10)", 1, 10))
    s2.append(T("p%dv" % n, 116, y + 14, 180, 40, v, 30, 900, TEXTC))
    s2.append(T("p%dl" % n, 300, y + 18, 300, 44, lab, 14, 600, MUTED, lh=1.45))

s2.append(T("wh", 96, 456, 520, 26, "왜 자동화가 어려웠는가", 15, 800, ACC, ls=2))
hard = [
    "<b>표준 컴포넌트가 아닌 계측 화면</b> — 일반 UI 자동화 도구가 버튼·입력창을 인식하지 못해, 화면을 이미지로 보고 좌표를 찾아야 합니다.",
    "<b>되풀이될 수밖에 없는 현상</b> — 등록 Align Key와 실제 Wafer 이미지가 공정 Variation으로 벌어집니다. Split Lot이 많은 R3에서는 상시 발생합니다.",
    "<b>의미 이해와 픽셀 정밀도를 동시에</b> — 매번 조금씩 달라지는 Align Key를 찾아 정확한 좌표로 재정렬해야 합니다.",
]
for n, txt in enumerate(hard):
    y = 486 + n * 62
    s2.append(T("h%dd" % n, 96, y + 6, 14, 20, "▪", 14, 900, ACC))
    s2.append(T("h%d" % n, 118, y, 498, 58, txt, 13.5, 600, MUTED, lh=1.5))

s2.append(T("rh", 656, 152, 528, 28, "TO-BE · 과제 목표", 15, 800, ACC, ls=2))
s2.append(table("goal", 656, 186, 528, 336,
                [0.72, 1.0, 1.35],
                [["구분", "As-Is (수동)", "To-Be (무인 Agent)"],
                 ["감지", "사람이 수시로 확인", "10초 주기 자동 확인 · 9006만 선별"],
                 ["접속·보정", "엔지니어가 RCS 접속 후 수동 재정렬", "RCS 자동 접속 → 성공 사례 기반 자동 보정"],
                 ["실패·오측", "사람이 발견할 때까지 지연", "담당자 Cube 알림 · 오측 시 측정 중단"],
                 ["기록", "별도 기록 없음", "상시 녹화 + 성공/실패 이미지 자동 저장"],
                 [{"html": "<b>1건당 시간</b>"}, {"html": "<b>평균 5분</b>"},
                  {"html": "<b>목표 1분</b>", "color": ACC, "bold": True}]],
                fs=13.5, padY=8))
s2.append(R("obj", 656, 540, 528, 132, ACC_DIM, "rgba(76,201,240,0.35)", 1, 12))
s2.append(T("objt", 680, 558, 480, 26, "과제 목적", 14, 800, ACC, ls=2))
s2.append(T("objd", 680, 588, 480, 72,
            "사람이 하던 <b>감지 → 장비 접속 → 정렬 재수정</b>을 AI가 대신 맡아 "
            "장비 idle 시간을 줄이고 계측 처리량을 회복합니다.", 16, 600, TEXTC, lh=1.55))
slides.append({"id": "s2", "background": BG, "transition": "none", "elements": s2, "notes":
    "선정 배경: 주 400건 × 1건당 5분 = 주 33시간의 장비 정지. 어려웠던 이유 3가지 — 비표준 UI, 공정 Variation으로 "
    "인한 이미지 괴리(R3는 Split Lot이 많아 상시), 의미 이해와 픽셀 정밀도의 동시 요구. "
    "목표는 감지부터 보정·기록까지 무인화하여 1건당 5분을 1분으로 단축하는 것."})

# ──────────────────────────────────── S3 · 추진 전략 & 성과 (노드는 S4로 morph)
STEPS = [
    ("01", "VLM 인프라 확보", "사내 HCP에 오픈소스 소형 모델 배포<br>GPU 1~2장 규모 · 통합 API 운영", "✅ 2종 상시 운영"),
    ("02", "GUI 자동화 증명", "비표준 계측 화면에서 좌표 인식<br>coarse → fine 2단계 + OCR 확인", "✅ 검증 완료 (90%+)"),
    ("03", "정렬 정확도 확보", "오프라인 벤치에서 A/B·튜닝<br>검증 통과한 방식만 운영 반영", "✅ 운영 반영"),
    ("04", "실시간 통합 루프", "감지 → 접속 → 보정 → 알림 → 녹화<br>end-to-end 무인 루프", "🟡 실운전 안정화"),
]
s3 = []
s3.append(T("t3", 96, 46, 900, 44, "추진 전략과 성과", 38, 900, TEXTC))
s3.append(T("st3", 96, 92, 1088, 26,
            "네 단계로 나눠 쌓았습니다. 앞 단계의 검증을 통과한 방식만 다음 단계에 반영했습니다.", 15, 600, MUTED))
s3.append(LN("flow", 96, 140, 1088, ACC, 2, "dashed", MARCH))
for n, (num, ttl, desc, chip) in enumerate(STEPS):
    x = 96 + n * 278
    s3.append(R("st%d_box" % n, x, 152, 254, 152, CARD, CARDB, 1, 12))
    s3.append(T("st%d_num" % n, x + 18, 166, 60, 34, num, 26, 900, ACC))
    s3.append(T("st%d_ttl" % n, x + 18, 202, 218, 28, ttl, 17, 800, TEXTC))
    s3.append(T("st%d_desc" % n, x + 18, 232, 218, 46, desc, 12.5, 600, MUTED, lh=1.5))
    s3.append(T("st%d_chip" % n, x + 18, 278, 218, 22, chip, 12, 700, ACC))

s3.append(T("ch3", 96, 328, 560, 26, "정량적 성과 · 정렬 위치 재조정 정확도", 15, 800, ACC, ls=1))
s3.append(T("ch3s", 96, 354, 560, 22,
            "Recipe 200개 오프라인 벤치 · 등록 이미지 1장 → 최근 성공 사례 종합(consensus)", 12, 600, DIM))
s3.append({
    "id": "chart", "type": "chart", "x": 88, "y": 378, "w": 576, "h": 286,
    "rotation": 0, "opacity": 1, "preset": "bar",
    "option": {
        "color": [NEUT, ACC],
        "textStyle": {"color": MUTED, "fontSize": 12},
        "legend": {"show": True, "top": 0, "textStyle": {"color": MUTED, "fontSize": 12}},
        "grid": {"left": 52, "right": 16, "top": 46, "bottom": 34},
        "xAxis": {"type": "category", "data": ["후보 포함률", "1순위 적중률"],
                   "axisLabel": {"color": MUTED, "fontSize": 13},
                   "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.14)"}}},
        "yAxis": {"type": "value", "min": 0, "max": 1,
                   "axisLabel": {"color": DIM, "fontSize": 11},
                   "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.07)"}}},
        "series": [
            {"type": "bar", "name": "등록 이미지 1장", "data": [0.434, 0.318],
             "itemStyle": {"color": NEUT, "borderRadius": [5, 5, 0, 0]}},
            {"type": "bar", "name": "성공 사례 종합(consensus)", "data": [0.876, 0.764],
             "itemStyle": {"color": ACC, "borderRadius": [5, 5, 0, 0]}},
        ],
        "tooltip": {"trigger": "item", "formatter": "{b}: {c}"},
    },
    "fx": {"enter": "fade-up"},
})
s3.append(T("dlt", 96, 664, 560, 22,
            "후보 포함률 <b>0.434 → 0.876</b> (약 2배) · 1순위 적중률 <b>0.318 → 0.764</b> (2배 이상)",
            13, 700, TEXTC))

s3.append(T("qh", 688, 328, 496, 26, "정성적 성과 · 구조적으로 확보한 것", 15, 800, ACC, ls=1))
QUAL = [
    ("VLM 인프라 자체 확보", "GPU 1~2장 규모 소형 모델을 사내 HCP에서 직접 구동. 벤치로 이긴 2종만 상시 운영해 모델 교체를 측정으로 결정합니다."),
    ("실장비 보정 반자동 개통 (2026-08)", "정렬 위치 재조정은 자동, 마지막 확정(OK)은 엔지니어. 개통 과정에서 보정이 한 번도 실행되지 않던 병목을 찾아 제거했습니다."),
    ("암묵지 자산화 경로 구축", "엔지니어 조작 녹화 → 조작 타임라인 → 한국어 절차서. 입력은 후킹하지 않고 화면만 관측해 동의 범위를 코드에 고정했습니다."),
]
for n, (ttl, desc) in enumerate(QUAL):
    y = 358 + n * 104
    s3.append(R("q%db" % n, 688, y, 496, 94, CARD, "rgba(255,255,255,0.10)", 1, 10))
    s3.append(T("q%dt" % n, 708, y + 12, 456, 24, ttl, 15, 800, TEXTC))
    s3.append(T("q%dd" % n, 708, y + 38, 456, 46, desc, 12.5, 600, MUTED, lh=1.5))
s3.append(R("more", 688, 668, 496, 26, "rgba(0,0,0,0)", radius=6, link="s3-detail"))
s3.append(T("moret", 688, 670, 496, 24,
            "▸ 성과 상세 (모드별 정확도 · 재순위 채택 · 잔여 개선 분해) — 클릭",
            12, 700, ACC, align="right", link="s3-detail"))
slides.append({"id": "s3", "background": BG, "transition": "none", "elements": s3, "notes":
    "추진 전략 4단계와 그 성과. 정량 핵심: 등록 이미지 1장만 쓸 때 대비 최근 성공 사례를 종합하면 후보 포함률 "
    "0.434→0.876, 1순위 적중률 0.318→0.764. 운영에서는 결국 한 곳으로 재조정하므로 1순위 적중률이 실제 성능에 "
    "가장 가깝다. 단, 오프라인 벤치 기준이며 오피스 실데이터 확정은 진행 중. "
    "상세 질문이 나오면 오른쪽 아래 링크를 클릭해 백업 슬라이드를 연다."})

# ─────────────────────────────────── S3-state · 성과 상세 (백업)
sd = []
sd.append(R("ov", 0, 0, 1280, 720, "#0A1120", radius=0, link="s3"))
sd.append(T("dt", 96, 60, 900, 44, "성과 상세 · 백업", 34, 900, TEXTC))
sd.append(T("dh", 96, 108, 1088, 26,
            "촬영 모드별 정확도와, 남은 개선 여지가 어디에 있는지에 대한 결론입니다.", 15, 600, MUTED))
sd.append(table("mtbl", 96, 156, 560, 176, [1.0, 0.9, 0.9, 0.6],
                [["촬영 모드 (성공 사례 종합 기준)", "후보 포함률", "1순위 적중률", "표본"],
                 ["OM · 저배율 (키가 화면의 10~20%)", "0.911", "0.852", "135"],
                 ["SEM · 고배율 (키가 화면의 80~100%)", "0.789", "0.665", "185"]], fs=13.5, padY=10))
sd.append(T("mn", 96, 342, 560, 44,
            "통념과 달리 고배율 SEM이 더 까다롭습니다. 화면 대부분이 비슷한 반복 패턴이라 진짜 정렬 위치와 헷갈리는 가짜 후보가 많습니다.",
            12.5, 600, MUTED, lh=1.5))
DSTAT = [("+0.073", "촬영 모드별 재순위 채택 시 SEM 1순위 적중률 상승 (OM +0.021, 무손실 확인)"),
         ("0.826", "재순위 반영 후 최종 채택 구성의 1순위 적중률"),
         ("1.000", "장비 창 버튼 판독 정확도 — 모델 벤치로 5종 평가 후 2종 상시 운영으로 정리")]
for n, (v, lab) in enumerate(DSTAT):
    y = 156 + n * 92
    sd.append(R("d%db" % n, 688, y, 496, 80, CARD, CARDB, 1, 10))
    sd.append(T("d%dv" % n, 708, y + 12, 150, 40, v, 28, 900, ACC))
    sd.append(T("d%dl" % n, 862, y + 14, 302, 56, lab, 12, 600, MUTED, lh=1.45))
sd.append(R("cb", 96, 420, 1088, 118, ACC_DIM, "rgba(76,201,240,0.35)", 1, 12))
sd.append(T("cbt", 120, 438, 1040, 26, "잔여 개선 여지의 분해 — 알고리즘 축은 소진됐습니다", 17, 800, ACC))
sd.append(T("cbd", 120, 468, 1040, 60,
            "재순위 반영 후 남은 실패의 <b>약 87%</b>가 '정답이 애초에 후보 목록에 없어 어떤 재순위로도 고칠 수 없는' 경우였습니다. "
            "남은 몫은 정렬 키를 더 잘 구별되는 위치로 <b>재등록</b>하는 일이며, 이는 현업 엔지니어와의 협업 과제입니다. "
            "이 결론에 따라 무게중심을 실시간 루프의 실운전 안정화로 옮겼습니다.", 14, 600, TEXTC, lh=1.6))
sd.append(T("cav", 96, 556, 1088, 50,
            "※ 위 정확도는 모두 오프라인 벤치(다운로드 샘플) 기준입니다. 오피스 실데이터 확정은 진행 중이며, 확정 전에는 성공 사례를 더 많이 요구하는 보수적 기준으로 운영합니다.",
            12.5, 600, DIM, lh=1.5))
sd.append(T("bk", 96, 640, 1088, 26, "클릭하면 본 슬라이드로 돌아갑니다", 12, 700, DIM, align="right", link="s3"))
slides.append({"id": "s3-detail", "stateOf": "s3", "background": "#0A1120",
               "transition": "morph", "name": "성과 상세", "elements": sd, "notes":
    "백업 슬라이드. 질문이 나올 때만 연다. 핵심 메시지: SEM이 구조적으로 더 어렵고, 남은 개선의 87%는 매칭 "
    "알고리즘이 아니라 정렬 키 재등록으로 풀어야 한다."})

# ─────────────────────────── S4 · 기대 효과 & 향후 일정 (S3 노드 morph)
s4 = []
s4.append(T("t4", 96, 46, 900, 44, "기대 효과와 향후 일정", 38, 900, TEXTC))
for n, (num, ttl, desc, chip) in enumerate(STEPS):
    x = 96 + n * 278
    s4.append(R("st%d_box" % n, x, 100, 254, 48, "rgba(255,255,255,0.03)", "rgba(255,255,255,0.10)", 1, 8))
    s4.append(T("st%d_num" % n, x + 14, 112, 30, 24, num, 14, 900, ACC))
    s4.append(T("st%d_ttl" % n, x + 46, 112, 196, 24, ttl, 13, 700, MUTED))

s4.append(T("eh", 96, 172, 560, 26, "기대 효과", 15, 800, ACC, ls=2))
s4.append(table("etbl", 96, 204, 560, 208, [1.35, 0.75],
                [["항목", "값"],
                 ["대상 장비 수", "28대"],
                 ["주 평균 Align Fail 발생", "약 400건"],
                 ["1건당 대응 지연", {"html": "<b>5분 → 1분</b>", "color": ACC}],
                 ["주간 절감 환산", {"html": "<b>WAFER 약 100장 추가 측정</b>", "color": ACC}]],
                fs=14, padY=9))
QEFF = ["야간·주말 무인 감시·보정", "실패 시점 화면 증거 자동 보존 → 원인 분석 시간 단축",
        "수집 데이터의 VLM 학습 자산화", "같은 Recipe면 장비가 달라도 성공 이력 공유 → 확대 비용 최소"]
for n, txt in enumerate(QEFF):
    x = 96 + (n % 2) * 288
    y = 428 + (n // 2) * 74
    s4.append(R("e%db" % n, x, y, 272, 64, CARD, "rgba(255,255,255,0.10)", 1, 8))
    s4.append(T("e%dt" % n, x + 14, y + 12, 244, 44, txt, 12.5, 650, MUTED, lh=1.45))

s4.append(T("sh", 688, 172, 496, 26, "향후 일정", 15, 800, ACC, ls=2))
SCHED = [
    ("8월 후반", "첫 실알람 완주 확인 (엔지니어 입회) · 엔지니어 작업 완료 감지 판독 확인 · 녹화 1건의 절차 추출 판정", True),
    ("9월", "MI 현업 엔지니어와 합동 실전 테스트 · 단일 장비 Pilot · 엔지니어 PC 배포 방식 확정", True),
    ("4Q", "Pilot 결과를 반영해 대상 장비 확대", False),
    ("2027년", "VLM이 Recipe Tuning을 수행하도록 workflow 구조 확장", False),
]
for n, (when, what, hot) in enumerate(SCHED):
    y = 204 + n * 74
    s4.append(R("sc%dd" % n, 692, y + 22, 12, 12, ACC if hot else NEUT, radius=6))
    if n < 3:
        s4.append(LN("sc%dl" % n, 697, y + 34, 2, "rgba(255,255,255,0.14)", 2, "solid"))
    s4.append(T("sc%dw" % n, 720, y + 12, 110, 26, when, 16, 800, ACC if hot else MUTED))
    s4.append(T("sc%dt" % n, 838, y + 12, 346, 56, what, 12.5, 600, MUTED, lh=1.45))
s4.append(R("risk", 688, 500, 496, 84, "rgba(255,255,255,0.03)", "rgba(255,255,255,0.12)", 1, 10,
            link="s4-detail"))
s4.append(T("riskt", 708, 512, 456, 22, "가장 가까운 관문 — 개통과 완주는 다릅니다", 13.5, 800, TEXTC))
s4.append(T("riskd", 708, 536, 456, 40,
            "실보정은 열렸지만 실알람 완주 사례가 아직 없습니다. ▸ 예상 허들과 진행 현황 — 클릭",
            12, 600, MUTED, lh=1.45, link="s4-detail"))

s4.append(R("end2", 96, 606, 1088, 74, ACC_DIM, "rgba(76,201,240,0.40)", 1, 12))
s4.append(T("end2t", 120, 622, 380, 32, "예상 종료 · <b>2026년 9월 말</b>", 22, 800, ACC))
s4.append(T("end2d", 500, 622, 660, 48,
            "현업 합동 실전 테스트와 단일 장비 Pilot 완료를 과제 종료 시점으로 봅니다.<br>이후 4Q 확대 적용은 후속 과제로 이어집니다.",
            13, 600, MUTED, lh=1.5))
slides.append({"id": "s4", "background": BG, "transition": "morph", "elements": s4, "notes":
    "기대 효과와 일정으로 마무리. 정량 효과 재확인(28대 / 주 400건 / 5분→1분 / 주 WAFER 100장)과 "
    "정성 효과 4가지. 일정은 8월 후반 첫 실알람 완주 확인, 9월 합동 실전 테스트와 단일 장비 Pilot, "
    "4Q 확대, 2027년 Recipe Tuning 확장. 예상 종료는 2026년 9월 말. "
    "허들 질문이 나오면 오른쪽 '가장 가까운 관문' 박스를 클릭한다."})

# ─────────────────────────── S4-state · 진행 현황 & 예상 허들 (백업)
se = []
se.append(R("ov2", 0, 0, 1280, 720, "#0A1120", radius=0, link="s4"))
se.append(T("et", 96, 52, 900, 42, "진행 현황과 예상 허들 · 백업", 32, 900, TEXTC))
se.append(table("ptbl", 96, 110, 560, 452, [1.0, 1.05],
                [["항목", "상태"],
                 ["VLM 배포·운영 (사내 HCP)", "✅ 2종 상시 운영"],
                 ["GUI 자동화 · 화면 증거 캡처", "✅ 오피스 검증 완료"],
                 ["CV 정렬 정확도 (consensus + 모드별 재순위)", "✅ 운영 반영"],
                 ["실시간 루프 골격·보정·녹화·알림", "✅ 개발 완료"],
                 ["실장비 보정 (반자동, 확정은 사람)", "✅ 개통 · 🟡 완주 대기"],
                 ["루프 실패 경로 하드닝", "✅ 완료"],
                 ["성공 사례 기반 보정 활성화", "🟡 이미지 공급 대기"],
                 ["지식 자산화 (녹화 → 절차서)", "✅ 녹화 확보 · 🟡 추출 미실행"],
                 ["엔지니어 PC 배포 방식", "🟡 방식 결정 대기"],
                 ["현업 합동 실전 테스트", "🟡 9월 예정"]], fs=13, padY=7))
se.append(T("eh2", 688, 110, 496, 26, "예상 허들과 대응", 15, 800, ACC, ls=2))
HURD = [
    ("개통과 완주는 다릅니다", "실알람 완주 사례가 없습니다. → replay 절차와 점검 모드로 단계별 사전 확인, 첫 실알람은 엔지니어 입회하에 관찰합니다."),
    ("오프라인 수치의 현장 검증 갭", "정확도는 벤치 기준입니다. → 오피스 실데이터로 확정하며, 그전까지는 보수적 기준으로 운영합니다."),
    ("성공 이미지 적재가 선행 조건", "적재 전에는 등록 이미지 폴백으로만 동작합니다. → 수집·적재 기능을 최우선 활성화 조건으로 진행합니다."),
    ("재등록은 협업 과제", "구별력이 약한 정렬 키는 알고리즘으로 풀리지 않습니다. → 우선순위 리스트로 순위화해 현업과 재등록을 협의합니다."),
    ("VLM 서버 가용성·배포 마찰", "재시작 시 환경 재설치에 3~4시간. → 실시간 루프는 호출이 적은 경로로 설계하고, 배포는 더블클릭 실행 방식을 사내 자원으로 고정합니다."),
]
for n, (ttl, desc) in enumerate(HURD):
    y = 142 + n * 86
    se.append(R("hr%db" % n, 688, y, 496, 76, CARD, "rgba(255,255,255,0.10)", 1, 9))
    se.append(T("hr%dt" % n, 706, y + 10, 460, 22, ttl, 13.5, 800, TEXTC))
    se.append(T("hr%dd" % n, 706, y + 32, 460, 40, desc, 11.5, 600, MUTED, lh=1.4))
se.append(T("stg", 96, 578, 560, 60,
            "실보정 전환은 <b>관찰 전용 → 반자동(현재) → 단일 장비 Pilot → 확대</b> 순서의 단계적 활성화로 리스크를 관리합니다.",
            13, 600, MUTED, lh=1.55))
se.append(T("bk2", 96, 656, 1088, 26, "클릭하면 본 슬라이드로 돌아갑니다", 12, 700, DIM, align="right", link="s4"))
slides.append({"id": "s4-detail", "stateOf": "s4", "background": "#0A1120",
               "transition": "morph", "name": "진행 현황", "elements": se, "notes":
    "백업 슬라이드. 진행 현황은 구현/오피스 검증/실알람 경험을 구분해 표기했다. "
    "가장 가까운 관문은 첫 실알람 완주 확인."})

doc = {
    "format": "bento/slides",
    "version": 1,
    "title": "Align Tuning Agent 경과 보고",
    "size": {"width": 1280, "height": 720},
    "meta": {"author": "기반기술센터", "company": "SK hynix",
             "subject": "Align Fail 무인 대응 AI Agent", "event": "과제 경과 보고"},
    "theme": {"background": BG, "color": TEXTC, "accent": ACC, "fontFamily": FONT},
    "slides": slides,
}

payload = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")

if not PATH.exists():
    print("[ERROR] 대상 파일이 없습니다:", PATH)
    print("[ERROR] 최신 Bento 릴리스를 먼저 내려받으십시오:")
    print("[ERROR]   curl -fsSL https://bento.page/releases/slides/Bento_Slides.bento.html -o", PATH)
    sys.exit(1)

html = PATH.read_text(encoding="utf-8")
i = html.index('id="bento-doc"')
s = html.index(">", i) + 1
e = html.index("</script>", s)
PATH.write_text(html[:s] + " " + payload + " " + html[e:], encoding="utf-8")

n_main = sum(1 for sl in slides if "stateOf" not in sl)
print("[INFO] 본편 슬라이드:", n_main, "| 백업(state):", len(slides) - n_main)
print("[INFO] elements:", sum(len(sl["elements"]) for sl in slides), "| payload KB:", round(len(payload) / 1024, 1))
