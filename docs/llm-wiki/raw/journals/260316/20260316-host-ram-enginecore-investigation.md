# 2026-03-16 host RAM 부족 가능성과 vLLM EngineCore 종료 이슈 정리

## 1. 진행 사항

- `deploy_vlms/config/models/mai-ui.env`, `deploy_vlms/config/models/paddleocr-vl-1.5.env`, `deploy_vlms/config/common.env`를 확인해 현재 배포 설정이 `H200 140GB x 2` 기준으로 작성되어 있음을 다시 정리했다.
- `deploy_vlms/scripts/start_model.py`, `deploy_vlms/scripts/serve_vlm.py`를 확인해 launcher가 다른 모델을 강제로 내리는 구조는 아니며, `EngineCore_DP0 died unexpectedly` 이후 `AsyncLLM output_handler failed`가 따라오는 흐름일 가능성이 높다고 정리했다.
- 사용자 환경 설명을 반영해 `280GB GPU VRAM(H200 2장)`과 `8GB system RAM`을 구분했다.
- 이번 이슈의 1차 진단 포인트를 `GPU VRAM 부족`보다 `host RAM 부족` 또는 `MAI-UI-8B runtime compatibility`로 재정리했다.
- 후속 확인용 운영 명령으로 `free -h`, `dmesg -T | tail -n 100`, `tail -n 200 deploy_vlms/runtime/logs/mai-ui.log`를 제안했다.
- 사용자가 추후 심층 분석을 진행하기 전에 사내 클라우드 서비스 제공자에게 system RAM 증설 요청을 우선 진행하기로 결정한 내용을 반영했다.

## 2. 수정 내용

- 신규 문서 추가: `docs/setup_vlms/10-system-ram-and-vllm.md`
- 문서 링크 갱신: `docs/setup_vlms/README.md`
- 신규 저널 추가: `docs/journals/deploy-vlms/20260316-host-ram-enginecore-investigation.md`

## 3. 다음 단계

- 클라우드 서비스 제공자에게 host/system RAM 증설 가능 여부와 목표 용량을 문의한다.
- RAM 증설 후 `MAI-UI-8B` 단독 기동 상태에서 `EngineCore_DP0 died unexpectedly`가 재현되는지 다시 확인한다.
- 재현 시 `deploy_vlms/runtime/logs/mai-ui.log`, `dmesg -T`, `free -h` 결과를 같이 수집해 host OOM인지 runtime mismatch인지 분리한다.
- 필요하면 `MAI-UI-8B`와 `UI-Venus-1.5-8B`의 기동 성공 여부를 같은 서버에서 비교해 모델별 runtime 민감도를 확인한다.

## 4. 메모리 업데이트

- 변경 없음
