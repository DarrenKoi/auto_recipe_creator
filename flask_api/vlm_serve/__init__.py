"""VLM serve 라우트 패키지."""

from .router import register_vlm_serve_routes, vlm_serve_blueprint

__all__ = ["register_vlm_serve_routes", "vlm_serve_blueprint"]
