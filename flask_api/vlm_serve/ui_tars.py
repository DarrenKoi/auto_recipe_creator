"""UI-TARS VLM route template."""

from .service_template import VLMServiceConfig, create_vlm_service_blueprint

SERVICE_CONFIG = VLMServiceConfig(
    blueprint_name="ui_tars_vlm",
    route_slug="ui-tars",
    display_name="UI-TARS-1.5-7B",
    upstream_port=8003,
)
service_blueprint = create_vlm_service_blueprint(SERVICE_CONFIG)

__all__ = ["SERVICE_CONFIG", "service_blueprint"]
