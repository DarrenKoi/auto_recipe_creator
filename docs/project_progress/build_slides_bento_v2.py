# -*- coding: utf-8 -*-
"""Align Tuning Agent 경과 보고 v2 - 동료 공유용 Bento 슬라이드(.bento.html) 생성 스크립트.

v1(build_slides_bento.py, 임원 보고용 본편 4장 + 백업 2장, 다크 테마)과 별도 파일이다.
v2는 **흰 배경 · 동료 공유용**으로 성과 요약 위에 workflow_3 최신 하드닝(Recovery Episode
수집·notify 재설계·grid_search 재설계·env 리팩터)과 workflow_4(상태 머신 프레임워크) 두
섹션을 더 얹어 상세도를 높인다. 근거는 00~04 Markdown(00_executive_summary.md,
04_workflow_3.md)이며, 수치를 고칠 때는 .md를 먼저 고치고 본 스크립트를 맞춘다.

대상 파일이 없으면 최신 릴리스를 먼저 내려받을 것:
  curl -fsSL https://bento.page/releases/slides/Bento_Slides.bento.html \
    -o docs/project_progress/Align_Tuning_Agent_v2.bento.html

실행:
  uv run python docs/project_progress/build_slides_bento_v2.py
출력:
  docs/project_progress/Align_Tuning_Agent_v2.bento.html (in-place 갱신)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "Align_Tuning_Agent_v2.bento.html"

# ── 흰 배경 팔레트 (build_slides_pptx.py 의 흰 배경 팔레트와 통일) ──────────
BG      = "#FFFFFF"
PANEL   = "#F6F9FC"
TEXTC   = "#10233A"
MUTED   = "#56677D"
DIM     = "#8A97A8"
ACC     = "#0E6FB8"
ACC_DK  = "#0A5389"
ACC_DIM = "#EAF3FB"
CARD    = "#F6F9FC"
CARDB   = "#B9D6EC"
LINE    = "#DCE5EE"
NEUT    = "#AEBECD"
GOOD    = "#1E9E6B"
WARN    = "#C9832A"
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
    "headerBg": ACC_DIM, "headerColor": ACC_DK,
    "zebra": "rgba(14,111,184,0.035)", "borderColor": LINE,
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
s1.append(R("bg", 0, 0, 1280, 720, BG, radius=0))
s1.append({"id": "glow", "type": "shape", "shape": "ellipse", "x": 830, "y": -220,
           "w": 700, "h": 700, "rotation": 0, "opacity": 1,
           "fill": "rgba(14,111,184,0.08)", "stroke": "none", "strokeWidth": 0, "radius": 0,
           "fx": {"ambient": "kenburns", "ken": {"dir": "drift", "scale": 1.06, "duration": 24}}})
s1.append(T("kick", 96, 88, 700, 32, "과제 종류 · 현업&nbsp;&nbsp;|&nbsp;&nbsp;경과 보고 (동료 공유판 v2)",
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

s1.append(R("endbar", 96, 594, 1088, 78, ACC_DIM, "rgba(14,111,184,0.28)", 1, 12))
s1.append(T("endt", 120, 610, 380, 34, "예상 종료 · <b>2026년 9월 말</b>", 24, 800, ACC_DK))
s1.append(T("endd", 500, 612, 660, 50,
            "현업 엔지니어 합동 실전 테스트와 단일 장비 Pilot 완료 시점<br>이후 4Q 대상 장비 확대 · 2027년 Recipe Tuning 확장",
            14, 600, MUTED, lh=1.5))
slides.append({"id": "s1", "background": BG, "transition": "none", "elements": s1, "notes":
    "표지 겸 Head Message (동료 공유판). 핵심 한 줄: 주 400건의 Align Fail에 대해 1건당 대응 지연을 "
    "5분에서 1분으로 줄여 주 WAFER 약 100장의 측정 여력을 회수한다. 이 v2는 임원 보고용 4장 구성에 "
    "workflow_3 최신 하드닝과 workflow_4 상태 머신 프레임워크 상세를 더 얹은 동료 공유판이다."})

# ──────────────────────────────────────────────── S2 · 선정 배경 / 목적
s2 = []
s2.append(R("bg2", 0, 0, 1280, 720, BG, radius=0))
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
    s2.append(R("p%db" % n, 96, y, 520, 74, CARD, LINE, 1, 10))
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
                 ["실패·오측", "사람이 발견할 때까지 지연", "담당자 Cube 알림(원인·요구 행동 포함) · 오측 시 측정 중단"],
                 ["기록", "별도 기록 없음", "상시 녹화 + 성공/실패 이미지 자동 저장 + Recovery Episode 기록"],
                 [{"html": "<b>1건당 시간</b>"}, {"html": "<b>평균 5분</b>"},
                  {"html": "<b>목표 1분</b>", "color": ACC, "bold": True}]],
                fs=13.5, padY=8))
s2.append(R("obj", 656, 540, 528, 132, ACC_DIM, "rgba(14,111,184,0.30)", 1, 12))
s2.append(T("objt", 680, 558, 480, 26, "과제 목적", 14, 800, ACC_DK, ls=2))
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
s3.append(R("bg3", 0, 0, 1280, 720, BG, radius=0))
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
                   "axisLine": {"lineStyle": {"color": LINE}}},
        "yAxis": {"type": "value", "min": 0, "max": 1,
                   "axisLabel": {"color": DIM, "fontSize": 11},
                   "splitLine": {"lineStyle": {"color": LINE}}},
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
    ("루프 하드닝 + Recovery Episode 수집 (~09월)", "실패 경로를 촘촘히 막고, 알람 1건을 사람이 읽을 기록(Episode)으로 표준화했습니다 — §5, §6."),
]
for n, (ttl, desc) in enumerate(QUAL):
    y = 358 + n * 104
    s3.append(R("q%db" % n, 688, y, 496, 94, CARD, LINE, 1, 10))
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
sd.append(R("ov", 0, 0, 1280, 720, "#FAFCFE", radius=0, link="s3"))
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
sd.append(R("cb", 96, 420, 1088, 118, ACC_DIM, "rgba(14,111,184,0.30)", 1, 12))
sd.append(T("cbt", 120, 438, 1040, 26, "잔여 개선 여지의 분해 — 알고리즘 축은 소진됐습니다", 17, 800, ACC_DK))
sd.append(T("cbd", 120, 468, 1040, 60,
            "재순위 반영 후 남은 실패의 <b>약 87%</b>가 '정답이 애초에 후보 목록에 없어 어떤 재순위로도 고칠 수 없는' 경우였습니다. "
            "남은 몫은 정렬 키를 더 잘 구별되는 위치로 <b>재등록</b>하는 일이며, 이는 현업 엔지니어와의 협업 과제입니다. "
            "이 결론에 따라 무게중심을 실시간 루프의 실운전 안정화로 옮겼습니다.", 14, 600, TEXTC, lh=1.6))
sd.append(T("cav", 96, 556, 1088, 50,
            "※ 위 정확도는 모두 오프라인 벤치(다운로드 샘플) 기준입니다. 오피스 실데이터 확정은 진행 중이며, 확정 전에는 성공 사례를 더 많이 요구하는 보수적 기준으로 운영합니다.",
            12.5, 600, DIM, lh=1.5))
sd.append(T("bk", 96, 640, 1088, 26, "클릭하면 본 슬라이드로 돌아갑니다", 12, 700, DIM, align="right", link="s3"))
slides.append({"id": "s3-detail", "stateOf": "s3", "background": "#FAFCFE",
               "transition": "morph", "name": "성과 상세", "elements": sd, "notes":
    "백업 슬라이드. 질문이 나올 때만 연다. 핵심 메시지: SEM이 구조적으로 더 어렵고, 남은 개선의 87%는 매칭 "
    "알고리즘이 아니라 정렬 키 재등록으로 풀어야 한다."})

# ─────────────────────────── S4 · 기대 효과 & 향후 일정 (S3 노드 morph)
s4 = []
s4.append(R("bg4", 0, 0, 1280, 720, BG, radius=0))
s4.append(T("t4", 96, 46, 900, 44, "기대 효과와 향후 일정", 38, 900, TEXTC))
for n, (num, ttl, desc, chip) in enumerate(STEPS):
    x = 96 + n * 278
    s4.append(R("st%d_box" % n, x, 100, 254, 48, "#F6F9FC", LINE, 1, 8))
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
    s4.append(R("e%db" % n, x, y, 272, 64, CARD, LINE, 1, 8))
    s4.append(T("e%dt" % n, x + 14, y + 12, 244, 44, txt, 12.5, 650, MUTED, lh=1.45))

s4.append(T("sh", 688, 172, 496, 26, "향후 일정", 15, 800, ACC, ls=2))
SCHED = [
    ("9월 초", "첫 실알람 완주 확인(진행 중) · 접속 게이트/알림 재설계 오피스 검증 · Recovery Episode 오피스 첫 1건 수집(티켓18)", True),
    ("9월", "MI 현업 엔지니어와 합동 실전 테스트 · 단일 장비 Pilot · 엔지니어 PC 배포 방식 확정", True),
    ("4Q", "Pilot 결과를 반영해 대상 장비 확대 · Episode 기록 기반 자동 판단 후속 티켓 착수", False),
    ("2027년", "VLM이 Recipe Tuning을 수행하도록 workflow 구조 확장", False),
]
for n, (when, what, hot) in enumerate(SCHED):
    y = 204 + n * 74
    s4.append(R("sc%dd" % n, 692, y + 22, 12, 12, ACC if hot else NEUT, radius=6))
    if n < 3:
        s4.append(LN("sc%dl" % n, 697, y + 34, 2, LINE, 2, "solid"))
    s4.append(T("sc%dw" % n, 720, y + 12, 110, 26, when, 16, 800, ACC if hot else MUTED))
    s4.append(T("sc%dt" % n, 838, y + 12, 346, 56, what, 12.5, 600, MUTED, lh=1.45))
s4.append(R("risk", 688, 500, 496, 84, "#F6F9FC", LINE, 1, 10,
            link="s4-detail"))
s4.append(T("riskt", 708, 512, 456, 22, "가장 가까운 관문 — 개통과 완주는 다릅니다", 13.5, 800, TEXTC))
s4.append(T("riskd", 708, 536, 456, 40,
            "실보정은 열렸지만 실알람 완주 사례가 아직 없습니다. ▸ 예상 허들과 진행 현황 — 클릭",
            12, 600, MUTED, lh=1.45, link="s4-detail"))

s4.append(R("end2", 96, 606, 1088, 74, ACC_DIM, "rgba(14,111,184,0.30)", 1, 12))
s4.append(T("end2t", 120, 622, 380, 32, "예상 종료 · <b>2026년 9월 말</b>", 22, 800, ACC_DK))
s4.append(T("end2d", 500, 622, 660, 48,
            "현업 합동 실전 테스트와 단일 장비 Pilot 완료를 과제 종료 시점으로 봅니다.<br>이후 4Q 확대 적용은 후속 과제로 이어집니다.",
            13, 600, MUTED, lh=1.5))
slides.append({"id": "s4", "background": BG, "transition": "morph", "elements": s4, "notes":
    "기대 효과와 일정으로 마무리. 정량 효과 재확인(28대 / 주 400건 / 5분→1분 / 주 WAFER 100장)과 "
    "정성 효과 4가지. 일정은 9월 초 첫 실알람 완주 확인과 Recovery Episode 오피스 첫 1건 수집, "
    "9월 합동 실전 테스트와 단일 장비 Pilot, 4Q 확대, 2027년 Recipe Tuning 확장. 예상 종료는 2026년 9월 말. "
    "허들 질문이 나오면 오른쪽 '가장 가까운 관문' 박스를 클릭한다. 이 뒤로 동료 공유판 전용 상세 섹션(S5 workflow_3 "
    "최신 하드닝, S6 workflow_4)이 이어진다."})

# ─────────────────────────── S4-state · 진행 현황 & 예상 허들 (백업)
se = []
se.append(R("ov2", 0, 0, 1280, 720, "#FAFCFE", radius=0, link="s4"))
se.append(T("et", 96, 52, 900, 42, "진행 현황과 예상 허들 · 백업", 32, 900, TEXTC))
se.append(table("ptbl", 96, 110, 560, 452, [1.0, 1.05],
                [["항목", "상태"],
                 ["VLM 배포·운영 (사내 HCP)", "✅ 2종 상시 운영"],
                 ["GUI 자동화 · 화면 증거 캡처", "✅ 오피스 검증 완료"],
                 ["CV 정렬 정확도 (consensus + 모드별 재순위)", "✅ 운영 반영"],
                 ["실시간 루프 골격·보정·녹화·알림", "✅ 개발 완료"],
                 ["실장비 보정 (반자동, 확정은 사람)", "✅ 개통 · 🟡 완주 대기"],
                 ["루프 실패 경로 하드닝 + env 리팩터(443→10)", "✅ 완료"],
                 ["점유 tool 화면공유 요청 + 접속 게이트 재설계", "✅ 구현 · 🟡 오피스 검증 대기"],
                 ["탐색 범위 확대(grid_search 격자 탐색)", "✅ 구현 · 🟡 오피스 미검증"],
                 ["Recovery Episode 수집(Guard·Verification·Outcome)", "✅ 구현 · 🟡 오피스 1건 대기(티켓18)"],
                 ["workflow_4 상태 머신 프레임워크", "✅ 골격 구현 · 데모 단계"],
                 ["지식 자산화 (녹화 → 절차서)", "✅ 녹화 확보 · 🟡 추출 미실행"],
                 ["엔지니어 PC 배포 방식", "🟡 방식 결정 대기"],
                 ["현업 합동 실전 테스트", "🟡 9월 예정"]], fs=12.5, padY=6))
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
    se.append(R("hr%db" % n, 688, y, 496, 76, CARD, LINE, 1, 9))
    se.append(T("hr%dt" % n, 706, y + 10, 460, 22, ttl, 13.5, 800, TEXTC))
    se.append(T("hr%dd" % n, 706, y + 32, 460, 40, desc, 11.5, 600, MUTED, lh=1.4))
se.append(T("stg", 96, 578, 560, 60,
            "실보정 전환은 <b>관찰 전용 → 반자동(현재) → 단일 장비 Pilot → 확대</b> 순서의 단계적 활성화로 리스크를 관리합니다.",
            13, 600, MUTED, lh=1.55))
se.append(T("bk2", 96, 656, 1088, 26, "클릭하면 본 슬라이드로 돌아갑니다", 12, 700, DIM, align="right", link="s4"))
slides.append({"id": "s4-detail", "stateOf": "s4", "background": "#FAFCFE",
               "transition": "morph", "name": "진행 현황", "elements": se, "notes":
    "백업 슬라이드. 진행 현황은 구현/오피스 검증/실알람 경험을 구분해 표기했다. "
    "가장 가까운 관문은 첫 실알람 완주 확인과 Recovery Episode 오피스 첫 1건 수집(티켓18)이다."})

# ═══════════════════════════ 동료 공유판 전용 섹션 ═══════════════════════════
s5div = []
s5div.append(R("bgdiv1", 0, 0, 1280, 720, ACC_DK, radius=0,
               grad={"angle": 120, "stops": [{"at": 0, "color": "#0A5389"}, {"at": 1, "color": "#0E6FB8"}]}))
s5div.append(T("divk", 96, 300, 1088, 28, "SECTION · 동료 공유 상세", 16, 700, "rgba(255,255,255,0.75)", ls=3))
s5div.append(T("divt", 96, 330, 1088, 90, "workflow_3 최신 하드닝<br>2026-08-16 이후 (80 commits)", 46, 900, "#FFFFFF", lh=1.2,
               fx={"enter": "fade-up", "order": 1}))
s5div.append(T("divs", 96, 430, 1000, 40,
               "실알람 완주 관문을 좁히기 위해 진단·알림, 설정 구조, 탐색 범위, 기록 표준을 함께 정리했습니다.",
               17, 600, "rgba(255,255,255,0.85)", lh=1.5))
slides.append({"id": "s5-div", "background": ACC_DK, "transition": "none", "elements": s5div, "notes":
    "구분 슬라이드. 여기부터는 임원 보고 4장 이후 동료 공유판에 추가한 상세 섹션이다. "
    "00~04 문서 최종 갱신(2026-08-16) 이후 workflow_3 에서 진행된 80개 커밋을 4개 축으로 정리한다."})

# ─────────────────────────── S5 · workflow_3 최신 하드닝 (4카드)
s5 = []
s5.append(R("bg5", 0, 0, 1280, 720, BG, radius=0))
s5.append(T("t5", 96, 46, 900, 44, "workflow_3 최신 하드닝", 36, 900, TEXTC))
s5.append(T("st5", 96, 92, 1088, 26,
            "실보정 개통(§3) 이후, 실알람 완주를 가로막는 사각지대를 하나씩 없앴습니다.", 15, 600, MUTED))

W3 = [
    ("진단·알림 재설계", "2026-09-01 · 구현완료",
     "보정이 안 될 때 상태 코드만 보내던 알림을 <b>원인 + 요구 행동 + 매칭 점수</b> 포함 9종 안내로 바꿨습니다. "
     "점유 tool 의 접속 요청 팝업이 원격 화면 <i>내부</i>에 그려져 로컬 창으로는 잡히지 않던 구조적 사각지대도 "
     "찾아 수정했습니다(탐지를 tool 창 crop 기준으로 전환). 엔지니어 도착 판정 후에야 대기 시간을 다시 세도록 "
     "고쳐, 도착 전에 장비가 먼저 닫히는 문제를 없앴습니다."),
    ("설정 구조 리팩터 (443→10)", "2026-08-31 · 구현완료",
     "산개해 있던 env 변수 443개를 <b>진입점 상단 상수 블록</b>으로 모았습니다. 계기는 오피스 사본이 추적 파일의 "
     "기본값을 조용히 덮어써 <code>SHARE_REQUEST</code>(공유 요청 알림)가 뜻하지 않게 켜졌던 실사고였습니다. "
     "지금은 무시된 상수를 콘솔에 자기고발하도록 만들어 같은 사고가 조용히 재발하지 않습니다."),
    ("탐색 범위 확대", "2026-08-28 · 기본값 채택 · 오피스 미검증",
     "휠 조작만으로는 배율을 크게 못 바꿔 탐색 반경이 좁았던 한계를 배율 드롭다운 절대값 zoom-out + 시야(FOV) "
     "격자 sweep 로 재설계했습니다. 픽셀이 아니라 <b>FOV 비율·배율 비율</b>로 계산해 배율이 달라져도 같은 로직이 "
     "적용됩니다. grid 탐색이 기본값이 됐고, 필요하면 <code>ALIGN_FAIL_SEARCH_MODE=legacy</code> 로 되돌릴 수 있습니다."),
    ("Recovery Episode 기록 표준화", "2026-08-30 · 구현완료 · 오피스 1건 대기",
     "알람 1건의 시작부터 종료까지를 <b>Episode</b> 하나로 묶어, 화면 관측 가능성·점유 상태·모드 판독이라는 "
     "Guard 3종과 측정 검증 기록, 최종 판정(Outcome)을 자동으로 남깁니다. 재시도는 같은 Episode 의 새 attempt 로 "
     "쌓여, 예전에 한 폴더에 두 시도가 섞이던 결함이 구조적으로 닫혔습니다."),
]
for n, (ttl, meta, desc) in enumerate(W3):
    x = 96 + (n % 2) * 560
    y = 132 + (n // 2) * 264
    s5.append(R("w3_%db" % n, x, y, 528, 244, CARD, CARDB, 1, 12))
    s5.append(R("w3_%dtag" % n, x + 20, y + 20, 20, 20, ACC_DIM, radius=6))
    s5.append(T("w3_%dn" % n, x + 20, y + 18, 20, 24, str(n + 1), 13, 900, ACC, align="center"))
    s5.append(T("w3_%dt" % n, x + 54, y + 16, 450, 26, ttl, 17, 800, TEXTC))
    s5.append(T("w3_%dm" % n, x + 54, y + 44, 450, 20, meta, 11.5, 700, ACC_DK, ls=1))
    s5.append(T("w3_%dd" % n, x + 20, y + 74, 488, 158, desc, 12.5, 600, MUTED, lh=1.55))

s5.append(R("more5", 96, 668, 1088, 26, "rgba(0,0,0,0)", radius=6, link="s5-detail"))
s5.append(T("moret5", 96, 670, 1088, 24,
            "▸ Recovery Episode 티켓 진행표 · env 리팩터 전/후 비교 — 클릭",
            12, 700, ACC, align="right", link="s5-detail"))
slides.append({"id": "s5", "background": BG, "transition": "none", "elements": s5, "notes":
    "workflow_3 최신 진전 4가지: (1) 자동 보정 불가 시 원인·요구 행동을 알리는 알림 재설계와 점유 접속 게이트의 "
    "구조적 사각지대 수정 — 오늘(9/1) 커밋. (2) env 443개를 진입점 상수 블록으로 모은 설정 리팩터 — 계기는 "
    "SHARE_REQUEST 오작동 실사고. (3) grid_search 탐색 재설계, 기본값 채택했으나 오피스 미검증. (4) Recovery "
    "Episode 기록 표준화, 티켓 10~17 구현완료·테스트 통과, 티켓18(오피스에서 알람 1건 수집)이 다음 관문. "
    "네 항목 모두 '실알람 완주'라는 하나의 목표로 수렴한다."})

# ─────────────────────────── S5-state · 티켓 진행표 + 리팩터 비교 (백업)
se5 = []
se5.append(R("ov5", 0, 0, 1280, 720, "#FAFCFE", radius=0, link="s5"))
se5.append(T("dt5", 96, 56, 900, 42, "Recovery Episode 티켓 진행 · 백업", 32, 900, TEXTC))
se5.append(T("dh5", 96, 100, 1088, 26,
            "알람 1건 = Episode 1건. 각 attempt(재시도)에 Guard·Verification·Outcome 기록이 자동으로 남습니다.",
            14, 600, MUTED))
se5.append(table("tk5", 96, 142, 620, 420, [0.3, 1.0, 0.55],
                [["#", "산출물", "상태"],
                 ["10", "Episode identity (alarm row 처리 시 생성)", "✅ 완료"],
                 ["11", "attempt-scoped 산출물 폴더", "✅ 완료"],
                 ["12", "자동 녹화 공유 frame metadata", "✅ 완료"],
                 ["13", "Guard 3종(관측·점유·모드) 기록", "✅ 완료"],
                 ["14", "재시작 재개 + orphan 스캔", "✅ 완료"],
                 ["15", "Measurement Verification 기록", "✅ 완료 (unknown-only)"],
                 ["16", "분자 per-read 판정 기록", "✅ 완료"],
                 ["17", "Outcome 판정 + [DIGEST] 한 줄", "✅ 완료"],
                 [{"html": "<b>18</b>"}, {"html": "<b>오피스 실알람 1건 수집</b>"},
                  {"html": "🟡 대기 · 코드 변경 없음", "color": ACC_DK}],
                 ["19~27", "행동 어휘·승인 기록·검토 패킷 등 후속", "⬜ 명세만, 착수 전(티켓18 대기)"]],
                fs=12.5, padY=6))
se5.append(T("eh5", 748, 142, 436, 26, "env 리팩터 · 전/후", 15, 800, ACC, ls=2))
BEFORE_AFTER = [
    ("이전", "env 변수 443개가 파일 곳곳에 산개 · 오피스 gitignored 사본이 조용히 우선순위를 가짐"),
    ("계기", "그 사본의 기본값이 실운전 SHARE_REQUEST(공유 요청 알림)를 뜻하지 않게 켬 — 당일 발견·수정"),
    ("이후", "진입점 상단 상수 블록 + setdefault 시딩. 셸 env > 파일 상수 > 코드 기본값. 무시된 상수는 콘솔에 출력"),
]
for n, (ttl, desc) in enumerate(BEFORE_AFTER):
    y = 176 + n * 96
    se5.append(R("ba%db" % n, 748, y, 436, 84, CARD, LINE, 1, 10))
    se5.append(T("ba%dt" % n, 766, y + 12, 400, 20, ttl, 12.5, 800, ACC_DK, ls=1))
    se5.append(T("ba%dd" % n, 766, y + 34, 400, 46, desc, 12, 600, MUTED, lh=1.45))
se5.append(R("cb5", 748, 470, 436, 96, ACC_DIM, "rgba(14,111,184,0.30)", 1, 10))
se5.append(T("cb5t", 766, 486, 400, 60,
            "위험도 높은 4개 토글(입력 차단·RCS 강제 종료·접속 승인·점유 중 보정)과 반자동 계약인 OK 클릭은 "
            "여전히 <b>기본 off</b>로 남겨 뒀습니다.", 12.5, 600, TEXTC, lh=1.55))
se5.append(T("bk5", 96, 656, 1088, 26, "클릭하면 본 슬라이드로 돌아갑니다", 12, 700, DIM, align="right", link="s5"))
slides.append({"id": "s5-detail", "stateOf": "s5", "background": "#FAFCFE",
               "transition": "morph", "name": "티켓 진행", "elements": se5, "notes":
    "백업 슬라이드. Recovery Episode 티켓 10~17은 전부 구현·테스트 완료. 티켓18은 코드가 아니라 오피스에서 "
    "실알람 1건을 수집·확인하는 순수 실행 게이트이며 결과(성공/미완)와 무관하게 파일이 남았는지만 확인하면 "
    "통과한다. 19~27은 18 통과 전까지 명세만 있고 착수하지 않는다. env 리팩터는 SHARE_REQUEST 오작동을 계기로 "
    "시작된 근본 원인 수정이다."})

# ─────────────────────────── S6 · workflow_4 상태 머신 프레임워크
s6 = []
s6.append(R("bg6", 0, 0, 1280, 720, BG, radius=0))
s6.append(T("t6", 96, 46, 900, 44, "실패 회복 과정을 눈에 보이게 · workflow_4", 32, 900, TEXTC))
s6.append(T("st6", 96, 92, 1088, 40,
            "workflow_3 대응 루프가 커지며 '지금 어느 단계인지, 실패하면 어디로 넘어가는지'를 코드 없이 "
            "확인할 별도 계층이 필요해졌습니다. 2026-08-28 착수.", 15, 600, MUTED, lh=1.5))

s6.append(T("wh6", 96, 148, 1088, 24, "세 가지를 분리해서 봐야 합니다", 15, 800, ACC, ls=1))
s6.append(table("w4tbl", 96, 180, 1088, 176, [0.62, 1.5, 1.0],
                [["구분", "무엇을 하나", "지금 production에 닿는가"],
                 [{"html": "<b>미러</b><br>(mirror)"},
                  "workflow_3 실행 기록을 읽기 전용으로 따라가며 진행 상황을 그림+표로 남긴다",
                  {"html": "✅ 켤 수 있음 (기본 꺼짐 · 켜도 기존 동작 동일)", "color": ACC_DK}],
                 [{"html": "<b>판정 계층</b><br>(Outcome)"},
                  "Recovery Episode(04문서 §9)의 최종 판정(회복/에스컬레이션 등)을 결정",
                  {"html": "✅ 실제로 쓰이는 중 (Outcome 판정의 유일한 소유자)", "color": ACC_DK}],
                 [{"html": "<b>엔진</b><br>(engine)"},
                  "흐름 자체를 상태 머신으로 실행하는 실행기",
                  {"html": "🔵 데모 전용 (가짜 시나리오 3종으로만 검증)", "color": DIM}]],
                fs=13, padY=10))

s6.append(T("wh7", 96, 380, 528, 24, "왜 만들어 쓰지 않고 새로 짰나", 15, 800, ACC, ls=1))
WHY4 = [
    "표준 상태 머신 라이브러리는 실패 종류별 대체 경로 · 노드별/전체 재시도 예산 · 중단 신호 폴링 · 그림+표 "
    "동시 출력 요구에 비해 무겁거나 맞지 않았습니다 — 신규 의존성 0개로 직접 구현.",
    "기존 실행기를 바로 이 엔진으로 갈아타는 안도 검토했지만, 위험의 시점(회귀가 오피스에서 뒤늦게 드러남) · "
    "기존 기록 형식 · 뒷정리 책임(장비 창 닫기 등은 실행기 소관 아님) 세 가지 이유로 당분간 데모 전용으로 "
    "묶어 두는 쪽을 택했습니다.",
]
for n, txt in enumerate(WHY4):
    y = 412 + n * 96
    s6.append(T("wh7_d%d" % n, 96, y + 4, 14, 20, "▪", 14, 900, ACC))
    s6.append(T("wh7_%d" % n, 118, y, 506, 86, txt, 12.5, 600, MUTED, lh=1.55))

s6.append(T("nx6", 656, 380, 528, 24, "다음 실제 활용처와 전제 조건", 15, 800, ACC, ls=1))
s6.append(R("nx6b", 656, 412, 528, 236, ACC_DIM, "rgba(14,111,184,0.30)", 1, 12))
s6.append(T("nx6t", 676, 428, 488, 40,
            "첫 실제 활용처는 정렬 보정 하위 흐름(재조정 → 탐색 → 재탐색)을 이 엔진으로 옮기는 것입니다.",
            13.5, 700, TEXTC, lh=1.5))
COND4 = [
    "중단 신호가 오래 걸리는 노드 안까지 관통해야 함 — 이번 착수 중 놓치는 결함을 실제로 잡아 고침",
    "예산은 시도 횟수가 아니라 실제 걸리는 시간 기준이어야 함",
    "미러가 여는 그림과 보정 흐름의 그림이 겹치지 않아야 함",
]
for n, txt in enumerate(COND4):
    y = 470 + n * 56
    s6.append(T("c4n%d" % n, 676, y, 24, 24, str(n + 1), 13, 900, ACC))
    s6.append(T("c4_%d" % n, 704, y, 460, 50, txt, 12, 600, MUTED, lh=1.4))

s6.append(R("more6", 96, 668, 1088, 26, "rgba(0,0,0,0)", radius=6, link="s6-detail"))
s6.append(T("moret6", 96, 670, 1088, 24,
            "▸ 진행 현황표 · ADR 설계 결정 요약 — 클릭",
            12, 700, ACC, align="right", link="s6-detail"))
slides.append({"id": "s6", "background": BG, "transition": "none", "elements": s6, "notes":
    "workflow_4 소개. 핵심은 3분리: 미러(읽기전용, production에 opt-in으로 이미 닿음) / 판정 계층(Outcome, "
    "Recovery Episode에 실제로 쓰임) / 엔진(상태 머신 실행기, 아직 데모 전용 — 가짜 시나리오 3종). "
    "표준 라이브러리 대신 직접 짠 이유와 기존 실행기를 바로 안 바꾼 이유(회귀 위험 시점, 기록 형식, 뒷정리 "
    "책임)를 설명한다. 다음 활용처는 정렬 보정 하위 흐름 이전이며 전제 조건 3가지를 먼저 정리해 뒀다."})

# ─────────────────────────── S6-state · 진행 현황 + ADR 요약 (백업)
se6 = []
se6.append(R("ov6", 0, 0, 1280, 720, "#FAFCFE", radius=0, link="s6"))
se6.append(T("dt6", 96, 56, 900, 42, "workflow_4 진행 현황 · 백업", 32, 900, TEXTC))
se6.append(table("pt6", 96, 108, 1088, 300, [1.6, 1.0],
                [["항목", "상태"],
                 ["상태 머신 프레임워크(그래프 정의·검증·유계 실행 루프)", "✅ 구현 완료"],
                 ["진행 상황 그림+표 실시간 뷰(오프라인, 아무 브라우저)", "✅ 구현 완료"],
                 ["workflow_3 대응 루프 읽기 전용 미러", "✅ 구현 완료 · 기본 꺼짐(옵트인)"],
                 ["Recovery Episode Outcome 판정 로직", "✅ 구현 완료 · 실제로 쓰이는 중"],
                 ["데모 시나리오(전 성공 / 대체 경로 / 에스컬레이션)", "✅ 검증 완료(모의 데이터)"],
                 ["정렬 보정 하위 흐름을 엔진으로 이전", "🔵 계획 단계 · 전제 조건 3가지 정리 완료"],
                 ["오피스 실장비에서의 미러 확인", "🟡 9월 초 예정"]], fs=13, padY=8))
se6.append(T("adrh", 96, 434, 1088, 24, "설계 결정 노트 (ADR)", 15, 800, ACC, ls=1))
ADRS = [
    ("0001 · 왜 직접 구현했나", "LangGraph 등 외부 상태 머신 라이브러리 대비, 이 요구사항(실패별 대체 경로·이중 재시도 예산·중단 폴링)엔 "
     "과중하거나 어긋나 신규 의존성 없이 직접 구현."),
    ("0002 · 읽기 전용 미러 + HTML 뷰", "cycle.py 의 실행 기록을 옆에서 읽기만 하는 미러로 한정. mermaid + 자체완결 HTML 뷰로 브라우저만 있으면 "
     "확인 가능, 기본 꺼짐."),
    ("0003 · 첫 실제 소비처는 정렬 보정 하위 흐름", "엔진에 실행기 라우팅을 아직 넘기지 않는다 — 첫 실제 적용은 run_correction 실행자 안에 중첩된 "
     "정렬 보정 하위 흐름부터."),
]
for n, (ttl, desc) in enumerate(ADRS):
    y = 464 + n * 68
    se6.append(T("adr%dt" % n, 96, y, 1088, 20, ttl, 13, 800, TEXTC))
    se6.append(T("adr%dd" % n, 96, y + 22, 1088, 40, desc, 12, 600, MUTED, lh=1.4))
se6.append(T("bk6", 96, 656, 1088, 26, "클릭하면 본 슬라이드로 돌아갑니다", 12, 700, DIM, align="right", link="s6"))
slides.append({"id": "s6-detail", "stateOf": "s6", "background": "#FAFCFE",
               "transition": "morph", "name": "workflow_4 상세", "elements": se6, "notes":
    "백업 슬라이드. workflow_4 진행 현황 전체 표와 ADR 3건 요약(직접 구현 이유·읽기전용 미러 범위·첫 실제 "
    "소비처가 엔진 전체 교체가 아니라 정렬 보정 하위 흐름이라는 점)."})

print("[INFO] S1~S6(+detail) 완료")

doc = {
    "format": "bento/slides",
    "version": 1,
    "title": "Align Tuning Agent 경과 보고 v2 (동료 공유판)",
    "size": {"width": 1280, "height": 720},
    "meta": {"author": "기반기술센터", "company": "SK hynix",
             "subject": "Align Fail 무인 대응 AI Agent", "event": "과제 경과 보고 (동료 공유판)"},
    "theme": {"background": BG, "color": TEXTC, "accent": ACC, "fontFamily": FONT},
    "slides": slides,
}

payload = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")

if not PATH.exists():
    print("[ERROR] 대상 파일이 없습니다:", PATH)
    sys.exit(1)

html = PATH.read_text(encoding="utf-8")
i = html.index('id="bento-doc"')
s = html.index(">", i) + 1
e = html.index("</script>", s)
PATH.write_text(html[:s] + " " + payload + " " + html[e:], encoding="utf-8")

n_main = sum(1 for sl in slides if "stateOf" not in sl)
print("[INFO] 본편 슬라이드:", n_main, "| 백업(state):", len(slides) - n_main)
print("[INFO] elements:", sum(len(sl["elements"]) for sl in slides), "| payload KB:", round(len(payload) / 1024, 1))
