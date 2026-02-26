# GUI VLM 벤치마크 기반 진행 보고서 (2026-02)

1. 우리는 `ScreenSpot-Pro` 기준으로 현재 self-hosted 모델 `Kimi-K2.5`와 GUI 특화 VLM/UI Parsing 조합을 비교 분석했다.
2. 또한 `RCS/CD-SEM` 자동화 실험에서 narrow-range 영역의 click-point 안정성, icon/widget 인식, element grounding 품질을 점검했다.
3. 분석 결과 현재 운영 중인 `Kimi-K2.5`의 점수는 52.8%로, `UI-Venus-1.5-8B`(69.6%) 등 GUI 특화 모델 대비 격차가 확인되었다.
4. 특히 밀집 UI(요소 간격 <10~20px)에서 좌표 drift와 인접 요소 혼동이 반복되어 recipe editor 자동화 신뢰도를 떨어뜨리고 있다.
5. 현 파이프라인은 `UI parsing` 전처리 부재로 인해 복잡한 화면에서 deterministic한 element 분리가 부족한 상태다.
6. 따라서 `OmniParser V2 + GUI-specialized VLM` 구조로 전환해 parser가 bbox를 먼저 고정하고 VLM이 target selection을 수행하도록 개선이 필요하다.
7. 이 전환을 위해 self-hosted 환경에서 신규 VLM(예: `UI-TARS-1.5-7B`, `UI-Venus-1.5-8B`)을 설치·서빙하고 성능을 재검증할 계획이다.
8. 실행을 위해 GPU 리소스 요청(우선 2x H200 기준)을 진행하고, 설치 후 benchmark와 실제 RCS workflow A/B test를 수행해 최종 모델을 선정하겠다.
