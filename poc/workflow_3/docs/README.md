# workflow_3 Docs

`poc/workflow_3/docs` 는 production 패키지(workflow_3)의 **사람이 작성한 문서** home 이다.
workflow_3 이 현재 build focus 이므로, workflow_3 관련 새 문서(설계 spec, ADR, 작업 저널,
runbook, 보고서)는 여기에 둔다. 예전처럼 `poc/workflow_2/docs` 에 쌓지 않는다.

> **생성물(generated)은 여기에 두지 않는다.** 스크립트를 돌려 만들어지는 산출물
> (debug overlay, `align_review` 의 `index.html`, `compare_report.json` 등)은
> `poc/workflow_3/debug_images/` 로 간다 (git 무시, `.gitkeep` 만 추적).
> `docs/` 는 git 으로 추적하는 authored 문서 전용이다.

## 폴더 역할 (필요할 때 생성)

- `journals/` - 날짜별 작업 기록 (`YYMMDD/`). 무엇을 했고 어떤 결정이 남았는지.
- `specs/` - 기능/설계 spec (예: serial multi-tool queue, matching roadmap).
- `study/` - 알고리즘 원리, runbook, ADR 등 길고 자세한 학습 자료.
- `weekly_report/` - 주간/매니저 보고 자료.

빈 폴더를 미리 만들어 두지 않는다. 첫 문서를 추가할 때 해당 폴더가 생긴다.

## workflow_2/docs 와의 관계

`poc/workflow_2/docs` 는 active offline CV bench 의 과거 연구 기록, ADR, runbook 을
계속 보관한다(CV 절차 히스토리의 authoritative 소스). `workflow_3/docs` 는 production
루프(monitor/cycle/recording/notify, RCS 자동화, office 이관) 문서의 새 home 이다.

- CV 매칭 실험/튜닝 문서 -> bench 쪽 `poc/workflow_2/docs`
- 루프/운영/이관 문서 -> 여기 `poc/workflow_3/docs`
