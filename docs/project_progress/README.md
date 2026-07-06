# 프로젝트 진행 보고서 (Project Progress Report)

> VLM-GUI 기반 Auto Recipe Creation — 1차 PoC(Align Fail 대응 자동화). 그동안의 작업(VLM 배포·운영, GUI 자동화, CV 정확도 벤치, 실시간 통합 루프) 정리.

이 폴더는 사내 보고용 진행 보고서의 **원본(source of truth)** 입니다. 본문 Markdown을 근거로
Word(.docx) 보고서와 시각 부록(`_appendix.html`)을 생성합니다. (보고서 본문은 경어체로 작성합니다.)

## 문서 목록 (읽는 순서)

| # | 문서 | 한 줄 요약 |
|---|------|-----------|
| 00 | [00_executive_summary.md](00_executive_summary.md) | **요약** — 목적(정량 효과 포함), PoC 방향(VLM↔CV 역할 분리), 설계 흐름, 성과, 확장성·효과 |
| 01 | [01_vlm_deployment.md](01_vlm_deployment.md) | **VLM 인프라 확보** — 자사 HCP에 SOTA 오픈소스 소형 모델 5종(VLM 3 + OCR 2) 배포, 단일 모델 Action 한계 → coarse→fine 2단계 + OCR 검증, HCP 제약·확보 역량 |
| 02 | [02_workflow_1.md](02_workflow_1.md) | **GUI 자동화 검증 (workflow 1)** — RCS 자동 조작 + CCTV 캡처 PoC. As-Is→To-Be, 핵심 확보 기술(클릭·타이핑 90%+·재시도/fallback·rescale), 데이터 자산화, CCTV 한계, 동결 사유 |
| 03 | [03_workflow_2.md](03_workflow_2.md) | **정렬 위치 재조정 CV 정확도 평가 (workflow 2)** — Recipe 200개 오프라인 벤치. As-Is→To-Be(등록 1장 → consensus), in_topk·rank1·OM/SEM 결과, 아이디어별 실험과 결정/남은 평가 |
| 04 | [04_workflow_3.md](04_workflow_3.md) | **실시간 Align Fail 자동 대응 Agent (workflow 3)** — As-Is→To-Be(수동 5분 → 무인 1분), 10초 알람 감지(9006)→RCS 접속→자동 보정, consensus gather 병렬, 오측 감지 시 측정 중단·Cube 알람, 지식 자산화, **진행 현황·예상 허들과 대응·향후 일정**(현재 주력) |

## 산출물 생성

본문 5개 `.md`(00~04)가 **단일 source-of-truth**입니다. `build_report_docx.py`는 이 `.md`를 공용 파서(`_md_report.py`)로
직접 파싱해 `.docx`로 렌더링하므로 **내용 수정은 `.md`만 고치면 산출물에 반영**됩니다.

```bash
# 의존성 (python-docx 필요)
uv sync

# Word 보고서 생성 -> docs/project_progress/project_progress_report.docx
uv run python docs/project_progress/build_report_docx.py
```

- **시각 부록**: `_appendix.html` 은 docx와 짝을 이루는 시각 자료(역할 분리·설계 흐름·VLM 인프라·
  2-stage 로케이터·실시간 루프·정렬 정확도·진행 현황 등)입니다. 브라우저로 열거나 인쇄(PDF)하여 docx 보고서를 보충합니다.
- (이전 `build_report_pptx.py`/`.pptx` 산출물은 `_appendix.html` 시각 부록으로 대체되어 제거되었습니다.)

## 작성 원칙

- 본문은 한국어 **경어체**로 작성하고 모델명·CV 기법·env·경로 등 기술 용어는 영문을 병기합니다.
- **standalone 제출용**입니다: 본문에는 저장소 내부 근거 경로를 넣지 않습니다(수치는 임의 추정 금지,
  근거는 저장소 문서 `docs/setup_vlms/`, `poc/workflow_2/docs/`, `poc/workflow_3/` 에서 확인).
- 정확도 수치는 **벤치(golden set) 기준**임을 명시하고 오피스 실데이터 검증 대기 항목과 구분합니다.
