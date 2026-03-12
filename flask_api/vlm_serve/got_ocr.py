"""GOT-OCR-2.0-hf route template."""

from .service_template import VLMServiceConfig, create_vlm_service_blueprint

SERVICE_CONFIG = VLMServiceConfig(
    blueprint_name="got_ocr",
    route_slug="got-ocr",
    display_name="GOT-OCR-2.0-hf",
    upstream_port=8005,
)
service_blueprint = create_vlm_service_blueprint(SERVICE_CONFIG)

__all__ = ["SERVICE_CONFIG", "service_blueprint"]
