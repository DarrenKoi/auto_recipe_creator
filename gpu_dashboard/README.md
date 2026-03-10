# gpu_dashboard template

`gpu_dashboard` 는 `web_main.py` 에서 함께 등록하는 placeholder blueprint 패키지다.
실제 대시보드 구현이 들어오면 이 템플릿을 교체하거나 확장하면 된다.

## 현재 export

- `gpu_dashboard_dp`

## 기본 등록 예시

```python
from gpu_dashboard import gpu_dashboard_dp

app.register_blueprint(gpu_dashboard_dp, url_prefix="/gpu-dashboard")
```

## 현재 기본 엔드포인트

- `/gpu-dashboard/`
- `/gpu-dashboard/health`
