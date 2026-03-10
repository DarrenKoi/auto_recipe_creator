"""UI-Venus VLM route template."""

from .service_template import VLMServiceConfig, create_vlm_service_blueprint

SERVICE_CONFIG = VLMServiceConfig(
    blueprint_name="ui_venus_vlm",
    route_slug="ui-venus",
    display_name="UI-Venus-1.5-8B",
    upstream_port=8001,
)
service_blueprint = create_vlm_service_blueprint(SERVICE_CONFIG)

__all__ = ["SERVICE_CONFIG", "service_blueprint"]
