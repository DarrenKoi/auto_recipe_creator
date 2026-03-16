# system RAM 이 vLLM / VLM 서빙에 중요한 이유

이 문서는 `GPU VRAM`과 `host system RAM`을 분리해서 이해하기 위한 운영 메모다.

핵심은 단순하다.

- `GPU VRAM`이 충분해도 `system RAM`이 너무 작으면 `vLLM` 엔진이 시작 중 죽을 수 있다.
- `AsyncLLM output_handler failed` 같은 에러는 종종 원인이 아니라, 이미 `EngineCore` 프로세스가 죽은 뒤에 따라오는 2차 증상이다.
- 따라서 `GPU 메모리가 크다 = 무조건 안전하다`로 보면 안 된다.

## 1. GPU VRAM 과 system RAM 의 역할 차이

- `GPU VRAM`은 모델 weight, KV cache, 일부 multimodal encoder 연산, attention 관련 텐서를 올리는 주 공간이다.
- `system RAM`은 Python 프로세스, `vLLM` runtime, tokenizer/processor, 모델 설정 로딩, shard index 처리, request buffer, CUDA host-side buffer, IPC, web server(예: Flask) 같은 CPU 측 작업 공간이다.
- 즉, `vLLM`은 GPU만 쓰는 프로그램이 아니라 `CPU process + system RAM + GPU VRAM`을 함께 쓰는 구조다.

## 2. VLM 실행 중 system RAM 이 쓰이는 위치

VLM 서버를 띄울 때 system RAM 은 아래 단계에서 관여한다.

### 2.1 Python / vLLM 프로세스 자체

- `python -m vllm.entrypoints.openai.api_server`로 뜬 API 서버 프로세스
- 내부 engine worker / scheduler / tokenizer 관련 프로세스 또는 스레드
- OpenAI-compatible request/response 객체와 queue

### 2.2 모델 메타데이터와 로더

- `config.json`, tokenizer 설정, processor 설정, `chat_template.json`
- safetensors index 읽기와 shard 경로 해석
- `trust_remote_code` 사용 시 custom Python module import

### 2.3 CPU 측 버퍼와 전송 경로

- 이미지 전처리 중간 결과
- CPU pinned memory
- GPU와 주고받는 host-side staging buffer
- 프로세스 간 shared memory / IPC buffer

### 2.4 web server / proxy / 부가 서비스

- Flask API
- reverse proxy
- health check / logging / request serialization
- 같은 머신에 같이 떠 있는 OCR sidecar 또는 추가 worker

즉, GPU가 크더라도 host RAM이 너무 작으면 engine init 단계나 첫 요청 전후에 죽을 수 있다.

## 3. system RAM 부족 시 보일 수 있는 증상

대표적인 증상은 아래와 같다.

- `EngineCore_DP0 died unexpectedly`
- `AsyncLLM output_handler failed`
- request 처리 전에 worker 프로세스가 비정상 종료
- `Killed` 또는 `Out of memory`가 `dmesg`에 기록
- GPU 메모리는 남아 있는데 API 서버가 계속 재시작되거나 종료

중요한 점:

- `AsyncLLM output_handler failed`는 원인 메시지보다 결과 메시지일 때가 많다.
- 실제 1차 원인은 그 직전에 있는 `EngineCore` 종료 로그, Python traceback, 또는 kernel OOM log에서 확인해야 한다.

## 4. 왜 큰 GPU 가 있어도 host RAM 이 부족할 수 있나

예를 들어 `H200 140GB x 2` 같은 환경에서도 host RAM 이 `8GB` 정도면 빡빡할 수 있다.

이유는 아래와 같다.

- 큰 모델일수록 tokenizer / processor / runtime import 크기도 커진다.
- 멀티모달 모델은 text-only 모델보다 processor, vision path, image buffer 때문에 CPU 메모리 관여가 더 크다.
- `vLLM` 서버 하나만 있는 것이 아니라 Flask, proxy, logging, OCR sidecar, shell session, system daemon이 같이 RAM을 사용한다.
- Linux page cache와 파일 IO cache도 RAM을 먹는다.

즉, `GPU VRAM 280GB`와 `system RAM 8GB`는 서로 대체 관계가 아니다.

## 5. 운영 중 확인해야 할 명령

host RAM 관점에서 가장 먼저 볼 명령은 아래 3개다.

```bash
free -h
dmesg -T | tail -n 100
tail -n 200 deploy_vlms/runtime/logs/mai-ui.log
```

해석 기준:

- `free -h`
  현재 RAM / swap 사용량과 `available` 메모리를 본다.
- `dmesg -T`
  kernel OOM killer가 특정 Python/vLLM 프로세스를 죽였는지 본다.
- `mai-ui.log`
  engine init 직전 traceback, import error, CUDA error, worker crash 징후를 본다.

추가로 보면 좋은 명령:

```bash
ps aux --sort=-%mem | head
```

이 명령은 어떤 프로세스가 RAM을 많이 쓰는지 빠르게 확인할 때 유용하다.

## 6. 이번 저장소 기준 해석 포인트

현재 `deploy_vlms` 문서는 기본적으로 `H200 140GB x 2` 전제를 둔다.

- `docs/setup_vlms/README.md`
- `docs/setup_vlms/01-layout-and-settings.md`
- `deploy_vlms/config/common.env`

하지만 여기서 중요한 것은 문서가 `GPU sizing`을 주로 다룬다는 점이다.

- `GPU_MEMORY_UTILIZATION`
- `MAX_MODEL_LEN`
- `MAX_NUM_SEQS`
- `COLOCATED_MODELS_PER_GPU`

이 값들은 주로 `GPU VRAM` budgeting에 가깝다.
반면 host RAM 부족은 이 계산만으로 잡히지 않는다.

즉, `GPU_MEMORY_UTILIZATION`을 맞췄더라도 host RAM 이 너무 작으면 별도 원인으로 죽을 수 있다.

## 7. 실무 가이드

- `EngineCore`가 죽으면 먼저 GPU OOM로 단정하지 않는다.
- `GPU VRAM`, `system RAM`, `runtime compatibility`를 분리해서 본다.
- 문제 재현 시 sidecar를 모두 내리고 모델 하나만 단독 기동해 본다.
- host RAM이 매우 작다면 먼저 증설 가능 여부를 확인한다.
- RAM 증설 전에는 같은 머신에 여러 VLM/OCR 서버를 동시에 올리는 구성을 보수적으로 본다.

## 8. 한 줄 결론

`vLLM`은 GPU만 쓰는 서버가 아니다.
`system RAM`은 engine process, tokenizer/processor, host buffer, web server, logging, IPC에 계속 관여하므로, GPU가 커도 host RAM 이 작으면 VLM 서버는 충분히 죽을 수 있다.
