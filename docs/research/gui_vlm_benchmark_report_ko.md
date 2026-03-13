# GUI VLM 벤치마크 기반 진행 보고서 (2026-03-13)

1. 현재 비교 세트는 회사 API baseline인 `Kimi-K2.5`와 self-hosted GUI 모델 `UI-Venus-1.5-8B`, `UI-TARS-1.5-7B`로 잡는 것이 맞다.
2. `deploy_vlms/config/` 기준 self-hosted 활성/준비 상태는 `UI-Venus(8001, GPU 0)`, `MAI-UI(8002, GPU 1)`, `UI-TARS(8003, GPU 0, env 준비)`, `PaddleOCR-VL-1.5(8004, GPU 1)`, `GOT-OCR-2.0-hf(8005, GPU 1)`다.
3. 다만 Flask proxy 기준 현재 활성 서비스는 `UI-Venus`, `MAI-UI`, `PaddleOCR-VL-1.5`, `GOT-OCR-2.0-hf`이고, `UI-TARS`는 `flask_api/vlm_serve/config.py`에서 아직 `enabled=False`다.
4. 따라서 1차 full-screen benchmark는 `Kimi-K2.5` vs `UI-Venus` vs `UI-TARS`로 수행하되, `UI-TARS`는 proxy 활성화 또는 direct `8003` 호출이 필요하다.
5. `MAI-UI-8B`는 primary 비교 대상보다 small target / crowded toolbar / ambiguous crop에 대한 **zoom-in sidecar**로 두는 편이 현재 GPU 배치에 더 잘 맞는다.
6. OCR 보강은 `PaddleOCR-VL-1.5`를 기본 sidecar로 두고, 아주 작은 숫자/코드/format-sensitive 텍스트는 `GOT-OCR-2.0-hf`로 fallback하는 구조가 적절하다.
7. 운영 평가 순서는 `Kimi/ Venus/ TARS` 1차 비교 후, self-hosted 승자 모델에 `MAI-UI`, `PaddleOCR-VL`, `GOT-OCR`를 순차적으로 붙여 gain을 측정하는 방식이 가장 해석하기 쉽다.
8. 측정 항목은 `element hit rate`, `click-point drift(px)`, `retry count`, `step completion rate`, `small-text OCR recall`, `latency`, `sidecar escalation rate`로 통일하는 편이 좋다.
