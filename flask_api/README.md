# flask_api integration template

`flask_api` 는 `web_main.py` 에서 다른 blueprint 와 같이 등록해서 쓰는 용도의 패키지다.
외부로 노출되는 기본 API prefix 는 `/api` 이다.
지금 구조는 `vlm_serve` 기준으로 VLM proxy route 를 확장할 수 있게 잡아둔 상태다.

## 권장 등록 형태

`gpu_dashboard` 와 함께 붙일 때는 URL prefix 를 분리해서 충돌을 피한다.

```python
from flask import Flask

from flask_api import register_flask_api
from gpu_dashboard import gpu_dashboard_dp


app = Flask(__name__)
app.register_blueprint(gpu_dashboard_dp, url_prefix="/gpu-dashboard")
register_flask_api(app, url_prefix="/api")
```

## 직접 blueprint 등록도 가능

```python
from flask_api import api_blueprint

app.register_blueprint(api_blueprint, url_prefix="/api")
```

## flask_api 안에 코드 추가하는 위치

- `flask_api/router.py`: root blueprint 생성 + 하위 feature router 등록
- `flask_api/vlm_serve/router.py`: `/api/vlm_serve` 아래 VLM route 등록
- `flask_api/vlm_serve/<service>.py`: 모델별 service config
- `flask_api/vlm_serve/service_template.py`: 공통 proxy blueprint template
- `flask_api/__init__.py`: blueprint 등록 helper 유지

예시 구조:

```text
flask_api/
  __init__.py
  router.py
  vlm_serve/
    __init__.py
    router.py
    service_template.py
    ui_venus.py
    mai_ui.py
    ui_tars.py
```

각 VLM 모듈은 service config 를 노출하고, 공통 template 이 upstream vLLM 으로 프록시한다.

## 현재 기본 엔드포인트

- `/api/vlm_serve/`
- `/api/vlm_serve/health`
- `/api/vlm_serve/ui-venus/health`
- `/api/vlm_serve/mai-ui/health`
- `/api/vlm_serve/ui-tars/health`
- `/api/vlm_serve/ui-venus/v1/models`
- `/api/vlm_serve/ui-venus/v1/chat/completions`

현재 upstream 기본 포트:
- `ui-venus -> 8001`
- `mai-ui -> 8002`
- `ui-tars -> 8003`
