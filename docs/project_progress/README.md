# 프로젝트 진행 보고서 (Project Progress Report)

> AI 기반 CD-SEM/VeritySEM recipe 자동 setup PoC — 그동안의 작업(VLM 배포·운영, workflow_1/2/3 구현·테스트·운영) 정리.

이 폴더는 사내 보고용 진행 보고서의 **원본(source of truth)** 입니다. 본문 Markdown을 근거로
Word(.docx) 보고서와 시각 부록(`_appendix.html`)을 생성합니다. (보고서 본문은 경어체로 작성합니다.)

## 문서 목록 (읽는 순서)

| # | 문서 | 한 줄 요약 |
|---|------|-----------|
| 00 | [00_executive_summary.md](00_executive_summary.md) | **임원 요약** — 목적, PoC 방향(VLM↔CV 역할 분리), 단계별 성과, 확장성·효과 |
| 01 | [01_vlm_deployment.md](01_vlm_deployment.md) | **VLM 배포·운영** — 오픈소스 VLM 5종 조사·선정(운영 4종 + 대안 1종 비활성), 사내 HCP(H200×2) 설치, vLLM·Flask proxy 운영, Kimi-K2 대비 풋프린트·밀도 기준 ~20배 효율 |
| 02 | [02_workflow_1.md](02_workflow_1.md) | **workflow_1** — RCS GUI 자동화 + CCTV/DVR 캡처 PoC. 증명한 것과 한계, frozen 사유 |
| 03 | [03_workflow_2.md](03_workflow_2.md) | **workflow_2** — 오프라인 CV 평가 벤치(golden set / ensemble / consensus). 정확도 측정 결과 |
| 04 | [04_workflow_3.md](04_workflow_3.md) | **workflow_3** — production 실시간 align-fail 모니터링 루프 (현재 주력) |
| 05 | [05_status_roadmap.md](05_status_roadmap.md) | **현황 & 로드맵** — 완료/대기 항목, 오피스 이전 체크리스트, 확장 방향·리스크 |

## 산출물 생성

본문 6개 `.md`가 **단일 source-of-truth**입니다. `build_report_docx.py`는 이 `.md`를 공용 파서(`_md_report.py`)로
직접 파싱해 `.docx`로 렌더링하므로 **내용 수정은 `.md`만 고치면 산출물에 반영**됩니다.

```bash
# 의존성 (python-docx 필요)
uv sync

# Word 보고서 생성 -> docs/project_progress/project_progress_report.docx
uv run python docs/project_progress/build_report_docx.py
```

- **시각 부록**: `_appendix.html` 은 docx와 짝을 이루는 시각 자료(역할 분리·루프·아키텍처·정확도
  차트·로드맵 등)입니다. 브라우저로 열거나 인쇄(PDF)하여 docx 보고서를 보충합니다.
- (이전 `build_report_pptx.py`/`.pptx` 산출물은 `_appendix.html` 시각 부록으로 대체되어 제거되었습니다.)

## 작성 원칙

- 본문은 한국어 **경어체**로 작성하고 모델명·CV 기법·env·경로 등 기술 용어는 영문을 병기합니다.
- 모든 수치·경로는 저장소 내 근거 문서를 인용합니다(임의 추정 금지). 주요 근거:
  `docs/setup_vlms/`, `poc/workflow_2/docs/`, `poc/workflow_3/README.md` + `poc/workflow_3/docs/`.
- 정확도 수치는 **벤치(golden set) 기준**임을 명시하고 오피스 실데이터 검증 대기 항목과 구분합니다.
