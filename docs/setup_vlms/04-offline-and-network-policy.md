# 간단한 오프라인 설정

회사에서 outbound 데이터 전송이 이미 차단되어 있다면, PoC 문서는 아래 정도만 지키면 충분하다.

## 최소 기준

- 모델 경로는 로컬 절대경로만 사용: `/project/.../data/models/...`
- Hugging Face repo id를 `MODEL_ID`로 직접 쓰지 않음
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `HF_HUB_DISABLE_TELEMETRY=1`
- `DO_NOT_TRACK=1`
- `VLLM_DO_NOT_TRACK=1`
- `VLLM_NO_USAGE_STATS=1`
- `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` 같은 토큰은 unset

위 값들은 [serve_vlm.py](../../deploy_vlms/scripts/serve_vlm.py)에 이미 들어 있다.

## 실무 해석

- 모델이 이미 `data/models/` 아래에 있고
- 회사 outbound 차단이 이미 적용되어 있고
- 위 스크립트로 실행하면

초기 PoC에서는 Hugging Face로 다시 나갈 것을 크게 걱정하지 않아도 된다.

## 참고

필요하면 아래만 참고하면 된다.

- Hugging Face environment variables: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables
- Transformers offline mode: https://huggingface.co/docs/transformers/en/installation
- vLLM usage stats controls: https://docs.vllm.ai/en/latest/api/vllm/usage/usage_lib/
