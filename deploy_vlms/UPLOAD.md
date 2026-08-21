# 모델 가중치 업로드 (로컬 PC -> 사내 Flask 서버)

code-server 웹 드래그앤드롭이 1GB 근처에서 깨지는 것을 대체한다.
요청 하나에 파일 하나를 통째로 싣지 않고 **청크로 쪼개** 올린다.

| 요구 | 구현 |
|---|---|
| 스트리밍 | 서버는 `request.stream` 을 1MB 블록으로 흘려 `.part` 에 바로 쓴다. 파일이 메모리에 올라오지 않는다 |
| 이어받기 | 서버가 커밋된 offset 을 디스크에 기록. 끊기면 **같은 명령을 다시 실행**하면 그 지점부터 |
| 무결성 | 청크마다 sha256 + 완료 시 파일 전체 sha256 재검증. 통과해야만 목적지로 원자적 이동 |

## 서버 (사내 private cloud)

`flask_api` 에 이미 배선돼 있다. 앱만 띄우면 `/api/model_upload/*` 가 열린다.

```bash
MODEL_UPLOAD_ROOT=/project/day/workSpace/.../data/models \
MODEL_UPLOAD_TOKEN=<공유비밀>                              \
uv run python index.py     # 또는 기존 WSGI 기동 방식 그대로
```

| env | 기본값 | 뜻 |
|---|---|---|
| `MODEL_UPLOAD_ROOT` | `ALLOWED_MODEL_ROOT` 와 같은 경로 | 업로드 목적지 루트. **이 밖으로는 절대 쓰지 않는다** |
| `MODEL_UPLOAD_TOKEN` | (빈값 = 인증 없음) | 설정하면 모든 업로드 요청에 `X-Upload-Token` 필요 |
| `MODEL_UPLOAD_MAX_CHUNK_MB` | 64 | 청크 하나의 상한 |
| `MODEL_UPLOAD_STAGING_DIR` | `<root>/.upload_staging` | 받는 중인 `.part` 위치. **반드시 root 와 같은 파일시스템** |
| `MODEL_UPLOAD_ENABLED` | 1 | 0 이면 엔드포인트를 아예 등록하지 않는다 |

도달성 확인:

```bash
curl -s http://<서버>:<포트>/api/model_upload/health
```

## 클라이언트 (로컬 PC)

```bash
MODEL_UPLOAD_URL=http://<서버>:<포트>      \
MODEL_UPLOAD_SRC=C:/models/MAI-UI-8B       \
MODEL_UPLOAD_TOKEN=<공유비밀>              \
uv run python deploy_vlms/scripts/upload_model.py
```

폴더면 재귀로 전부, 단일 파일이면 그것만 올린다. 숨김 파일/폴더(`.git`, `.cache`)는 건너뛴다.

| env | 기본값 | 뜻 |
|---|---|---|
| `MODEL_UPLOAD_URL` | (필수) | Flask 서버 base URL |
| `MODEL_UPLOAD_SRC` | (필수) | 올릴 로컬 폴더 또는 파일 |
| `MODEL_UPLOAD_DEST` | 소스 폴더명 | 서버 루트 아래 목적지 경로 |
| `MODEL_UPLOAD_TOKEN` | (빈값) | 서버가 요구하면 필수 |
| `MODEL_UPLOAD_CHUNK_MB` | 32 | 서버 상한보다 크면 자동으로 줄인다 |
| `MODEL_UPLOAD_MAX_RETRIES` | 5 | 청크 하나당 재시도 한도 (지수 백오프, 최대 30s) |

### 끊겼을 때

**같은 명령을 다시 실행하면 된다.** 이미 올라간 파일은 건너뛰고, 진행 중이던 파일은
서버가 확실히 받은 지점부터 이어간다. 실측: 1.2GB 업로드를 중간에 kill -9 한 뒤
재실행 -> 남은 352MB 만 전송, 최종 sha256 원본 일치.

### 자주 나오는 실패

| 증상 | 원인 / 조치 |
|---|---|
| `HTTP 413` | 앞단 프록시의 `client_max_body_size` 가 청크보다 작다. `MODEL_UPLOAD_CHUNK_MB` 를 낮춘다 |
| `HTTP 401` | `MODEL_UPLOAD_TOKEN` 불일치 |
| `HTTP 400 PathNotAllowed` | `MODEL_UPLOAD_DEST` 가 루트를 벗어난다 |
| `HTTP 422` 반복 | 청크가 계속 손상돼 도착한다. 네트워크 경로를 의심 |
| 완료 시 `ChecksumMismatch` | 청크는 다 통과했는데 조립 결과가 다르다 = 디스크 의심. 서버가 `.part` 를 버리고 0 부터 다시 받는다 |

### 잘못 올린 세션 정리

`.upload_staging` 에 `.part` 가 남아 디스크를 먹으면:

```bash
curl -X DELETE -H "X-Upload-Token: <비밀>" \
  http://<서버>:<포트>/api/model_upload/sessions/<upload_id>
```

## 테스트

전부 Mac 에서 실서버 없이 돈다 (마지막 것만 로컬에 임시 서버를 띄운다).

```bash
uv run pytest flask_api/model_upload              # 33 (store / routes / 배선)
uv run pytest deploy_vlms/scripts                 # 14 (클라이언트 루프 + 실제 HTTP 왕복)
```
