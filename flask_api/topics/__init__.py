"""flask_api topic 라우트 등록."""

from importlib import import_module
from pkgutil import walk_packages

from flask import Blueprint


def _iter_topic_module_names() -> list[str]:
    """topic 하위의 route 모듈 이름을 정렬해서 반환한다."""
    prefix = f"{__name__}."
    module_names = [
        module_info.name
        for module_info in walk_packages(__path__, prefix)
        if not module_info.ispkg
    ]
    return sorted(module_names)


def register_topic_routes(api_blueprint: Blueprint) -> None:
    """topic 하위 모듈의 register_routes()를 자동 호출한다."""
    for module_name in _iter_topic_module_names():
        module = import_module(module_name)
        register_routes = getattr(module, "register_routes", None)
        if callable(register_routes):
            register_routes(api_blueprint)


__all__ = ["register_topic_routes"]
