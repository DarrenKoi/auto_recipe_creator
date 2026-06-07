# Workflow 1 Docs

`poc/workflow_1/docs` 는 `poc/workflow_2/docs` 와 **같은 폴더 구조**를 따른다.
목적은 두 가지다. (1) workflow_1 이 쓰는 방법(method)을 본인 학습용으로 정리하고,
(2) 상사·동료의 질문에 바로 답할 수 있는 설명 자료를 모아 둔다.

> workflow_1 의 역할: **RCS 로그인 → Tool 선택 → Align Fail 알람 감지(ALID=9006) → Tool 화면 캡처.**
> workflow_2 가 CV 매칭(Chamfer/ORB)이 주제라면, workflow_1 은 **GUI 자동화**가 주제다 —
> "화면을 보고(VLM) 어디를 누를지 결정하고(좌표 변환) 실제로 클릭/입력한다(pynput)".

## 폴더 역할

- `journals/`: 날짜별 작업 기록 (현재 비어 있음; 작업 시 `journals/<YYMMDD>/` 로 추가).
- `weekly_report/`: 주간/매니저 보고 자료 (현재 비어 있음).
- `console_results/`: 실행 콘솔 로그 스냅샷 (현재 비어 있음).
- `study/`: **학습용 상세 문서.** 이 README 가 가리키는 교육 자료의 본체.

## study/ 안내 (읽는 순서)

처음이라면 **이 순서**로 읽기를 권한다.

1. `study/runbooks/workflow_1_procedure.md` — 전체 흐름(로그인→Tool 선택→알람→캡처)을 한눈에. **단일 권위 문서.**
2. `study/algorithms/automation_methods_intro.md` — 핵심 기법 5가지를 한 장에 요약 (마스터 개요).
3. 그다음 관심 주제별 deep-dive:
   - `study/algorithms/two_stage_vlm_locator.md` — 2단계 VLM 좌표 찾기 (coarse→fine→confirm)
   - `study/algorithms/dpi_coordinate_mapping.md` — 이미지 좌표 → 화면 좌표 (DPI 배율 보정)
   - `study/algorithms/tool_name_canonicalization.md` — OCR 혼동 글자 정규화로 Tool ID 매칭
   - `study/algorithms/alarm_polling_loop.md` — 알람 폴링 루프 (edge-triggered 중복 제거)
   - `study/cv/ocr_spotting_intro.md` — OCR Spotting(text+bbox) 파싱과 "확인 전용" 규칙
   - `study/cv/cursor_detection.md` — 프레임에서 커서 끝점 찾기 (2단계)
   - `study/cv/image_capture_pipeline.md` — 화면 캡처 → WebP → crop/zoom 파이프라인
4. 설계 결정의 "왜":
   - `study/adr/` — ADR(아키텍처 결정 기록). VLM 좌표 vs pywinauto, Flask proxy vs direct, 로깅 전략.
5. 도메인 참고:
   - `study/sem_box/rcs_list_tab_layout.md` — RCS List 탭 UI 구조 (workflow_2 의 `sem_box/` 에 대응하는 도메인 UI 레퍼런스)
   - `study/paddleOCR/README.md` — PaddleOCR-VL 을 "확인용"으로만 쓰는 이유와 방법

## 현재 구조

```text
docs/
├─ README.md                  # 이 파일 (folder roles + 읽는 순서)
├─ journals/                  # 날짜별 작업 기록 (비어 있음)
├─ weekly_report/             # 주간 보고 (비어 있음)
├─ console_results/           # 콘솔 로그 스냅샷 (비어 있음)
└─ study/
   ├─ adr/                    # 아키텍처 결정 기록
   ├─ algorithms/            # 핵심 자동화 기법 해설 (개요 + deep-dive)
   ├─ cv/                     # 영상/OCR 처리 해설
   ├─ paddleOCR/             # PaddleOCR-VL 사용 가이드
   ├─ runbooks/             # 실행 절차 (procedure / 모듈 실행법)
   └─ sem_box/              # 도메인 UI 구조 레퍼런스 (RCS List 탭)
```

## 작성 규칙

- 교육 문서는 **한국어 prose, 기술 용어만 영어** 로 유지한다 (workflow_2 docs 와 동일).
- 코드 상수·함수명은 실제 소스(`poc/workflow_1/`)와 일치시킨다. 값이 바뀌면 문서도 갱신한다.
- 같은 내용을 Markdown 과 HTML 둘 다 만들지 않는다. 원문은 Markdown.
