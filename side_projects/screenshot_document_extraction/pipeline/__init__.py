"""스크린샷 문서 추출 파이프라인 패키지.

세 개의 독립적인 단계로 구성된다:
1. capture — Office/PDF 파일을 열어 페이지별 JPEG 로 저장
2. extract — JPEG 를 paddleocr-vl-1.5, ui-venus 로 추출
3. organize — 페이지별 raw JSON 을 머지하여 Markdown + JSON sidecar 생성

각 단계는 `logs/pipeline_state.json` 원장(ledger)을 공유하므로
중간에 멈춰도 다음 실행에서 끝낸 페이지를 건너뛰고 이어서 처리한다.
"""
