# Workflow 2 Docs

`poc/workflow_2/docs` 는 세 가지 목적별 폴더로 관리한다.

## 폴더 역할

- `journals/`: 날짜별 작업 기록. 어떤 이슈가 있었는지, 무엇을 진행했는지, 어떤 해결 방향이나 남은 결정을 잡았는지 기록한다.
- `weekly_report/`: 주간/매니저 보고 자료. 진행 상황, 성과, 다음 작업을 보고서 형태로 정리한다.
- `study/`: 학습용 상세 문서. 알고리즘 원리, runbook, ADR, PaddleOCR/SEM box/CV 해설처럼 길고 자세한 자료를 둔다.

## 현재 구조

```text
docs/
├─ journals/
│  ├─ 260528/
│  └─ 260529/
├─ weekly_report/
│  ├─ generate_status_report.py
│  ├─ weekly_report_2026-05-28.html
│  └─ workflow_2_status_report.{html,pptx}
└─ study/
   ├─ adr/
   ├─ algorithms/
   ├─ cv/
   ├─ paddleOCR/
   ├─ runbooks/
   └─ sem_box/
```

## 중복 정리 기준

- 같은 내용을 Markdown 과 HTML 로 모두 가진 경우에는 원문 Markdown 을 남긴다.
- HTML 로만 작성된 학습/보고 자료는 유지한다.
- 폐기된 결정도 이력 가치가 있으면 `study/adr/` 에 보존하되, 본문에 `superseded` 상태를 유지한다.
