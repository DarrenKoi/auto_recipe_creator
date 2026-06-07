---
status: accepted
---

# 런타임 로그는 print 기반, 감사 추적만 RotatingFileHandler 로 분리한다

## 결정

- **런타임 로그** 는 `print()` 기반 `[INFO]` / `[ERROR]` / `[WARNING]` 접두사를 쓴다 (프로젝트 전역
  컨벤션). `logging` 모듈을 쓰지 않는다.
- **예외 — 파일 감사 추적(audit trail)** 만 `logger.py` 에서 Python `logging` +
  `RotatingFileHandler` 로 남긴다:
  - `logs/vlm_calls.log` — VLM 호출(서비스/모델/지연/토큰/status/endpoint).
  - `logs/work2.log` — 일반 이벤트(component/message + 임의 필드).

## 맥락 / 이유

- 대부분의 자동화는 **콘솔에서 흐름을 즉시 읽는 것** 이 중요하다. `[INFO]`/`[ERROR]` print 는
  단순하고, 오피스 디버깅(콘솔 출력을 사용자에게 보고)에도 잘 맞는다.
- 반면 VLM 호출은 **나중에 추적·집계** 해야 하는 감사 대상이다(비용·지연·실패율). 콘솔에만 남기면
  사라지므로, 이것만 파일로 남기되 **무한정 늘어나지 않게 회전(rotation)** 시킨다.

## 설정

```python
_MAX_BYTES     = 10 * 1024 * 1024    # 파일당 10MB
_BACKUP_COUNT  = 5                    # .log.1 ~ .log.5
formatter      = "%(asctime)s [%(levelname)s] %(message)s"
propagate      = False                # root 로 전파 안 함
```

- `_get_logger(name)` 는 name 별 싱글턴, 핸들러에 `_HANDLER_MARKER` 로 **중복 부착 방지(idempotent)**.
- `log_vlm_call(...)` 성공: `service=.. model=.. status=ok latency_ms=.. *_tokens=.. endpoint=.. response_chars=..`
- `log_work2_event(component, message, level, **fields)`: `component=.. message=.. <필드>=..`

## 결과 (Consequences)

- 콘솔과 파일의 역할이 나뉜다: 콘솔은 실시간 흐름, 파일은 감사·집계.
- 로그 파일이 디스크를 잠식하지 않는다(10MB×5 상한).
- VLM 클라이언트는 성공·에러를 가리지 않고 매 호출 끝에 `log_vlm_call()` 을 부른다.

## 주의

- `logging` 모듈은 **`logger.py` 한 곳에서만 쓴다**. 다른 모듈에서 `logging` 을 새로 끌어다 쓰지
  말 것(컨벤션 위반). 일반 메시지는 print 나 `log_work2_event` 로 남긴다.
