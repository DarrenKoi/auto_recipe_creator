# flask_api integration template

`flask_api` 는 `web_main.py` 에서 다른 blueprint 와 같이 등록해서 쓰는 용도의 패키지다.
외부로 노출되는 기본 API prefix 는 `/api` 이다.
기본 구조는 topic 기준 하위 폴더까지 확장할 수 있게 잡아둔 상태다.

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

- `flask_api/routes.py`: root blueprint 생성 + topic route 자동 등록
- `flask_api/topics/<topic>/...`: topic별 route 모듈 추가
- `flask_api/__init__.py`: blueprint 등록 helper 유지

예시 구조:

```text
flask_api/
  __init__.py
  routes.py
  topics/
    __init__.py
    system/
      __init__.py
      status.py
    recipes/
      __init__.py
      list.py
      detail.py
```

각 topic 모듈은 `register_routes(api_blueprint)` 함수를 노출하면 자동으로 잡힌다.
하위 폴더를 더 중첩해도 `flask_api/topics/` 아래에 있기만 하면 같은 방식으로 등록된다.

## 현재 기본 엔드포인트

- `/api/`
- `/api/health`
- `/api/example`
