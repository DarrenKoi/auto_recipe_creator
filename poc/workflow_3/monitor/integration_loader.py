"""office 전용 adapter 로딩 helper.

`office_*` 파일은 gitignore 대상이라 오피스 PC 에만 존재한다. monitor 쪽에서는
정위치(workflow_3.monitor)와 legacy 위치(workflow_1)를 순서대로 시도하되, 어떤
경로가 실제로 연결됐는지 콘솔과 workflow_3 로그에 남긴다.
"""

import importlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OfficeIntegration:
    """office adapter 로딩 결과."""

    name: str
    module_path: str
    module: Any | None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "missing"
    is_legacy: bool = False
    error: str = ""

    @property
    def available(self) -> bool:
        """호출 가능한 office adapter 를 찾았는지."""
        return self.module is not None and self.status in {"loaded", "legacy_loaded"}


def _log_office_status(
    *,
    name: str,
    status: str,
    module_path: str = "",
    is_legacy: bool = False,
    error: str = "",
    missing_attrs: list[str] | None = None,
) -> None:
    """office adapter 로딩 상태를 콘솔과 workflow_3 파일 로그에 남긴다."""
    level = "info" if status in {"loaded", "factory_loaded"} else "warning"
    prefix = "[WARNING]" if level == "warning" else "[INFO]"
    details = [
        f"name={name}",
        f"status={status}",
    ]
    if module_path:
        details.append(f"module_path={module_path}")
    if is_legacy:
        details.append("legacy=true")
        details.append("copy_to=poc/workflow_3/monitor")
    if missing_attrs:
        details.append(f"missing_attrs={','.join(missing_attrs)}")
    if error:
        details.append(f"error={error}")

    print(f"{prefix} office integration: " + " ".join(details))

    try:
        from poc.workflow_3.logger import log_work2_event

        log_work2_event(
            component="office_integrations",
            message=status,
            level=level,
            name=name,
            module_path=module_path,
            legacy=str(is_legacy),
            missing_attrs=",".join(missing_attrs or []),
            error=error,
        )
    except Exception:
        # 로깅 실패가 monitor import 자체를 깨면 안 된다.
        pass


def _is_missing_target_module(exc: ModuleNotFoundError, module_path: str) -> bool:
    """대상 office 모듈이 없는 경우와 내부 import 실패를 구분한다."""
    missing_name = getattr(exc, "name", "")
    return missing_name == module_path


def load_office_integration(
    name: str,
    module_paths: tuple[tuple[str, bool], ...],
    *,
    required_attrs: tuple[str, ...] = (),
) -> OfficeIntegration:
    """office adapter 모듈을 정위치 → legacy 순서로 찾는다."""
    attempted = [path for path, _is_legacy in module_paths]
    for module_path, is_legacy in module_paths:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if _is_missing_target_module(exc, module_path):
                continue
            _log_office_status(
                name=name,
                status="import_error",
                module_path=module_path,
                is_legacy=is_legacy,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        except Exception as exc:
            _log_office_status(
                name=name,
                status="import_error",
                module_path=module_path,
                is_legacy=is_legacy,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue

        attrs = {attr: getattr(module, attr, None) for attr in required_attrs}
        missing_attrs = [attr for attr, value in attrs.items() if value is None]
        if missing_attrs:
            _log_office_status(
                name=name,
                status="missing_attr",
                module_path=module_path,
                is_legacy=is_legacy,
                missing_attrs=missing_attrs,
            )
            continue

        status = "legacy_loaded" if is_legacy else "loaded"
        _log_office_status(
            name=name,
            status=status,
            module_path=module_path,
            is_legacy=is_legacy,
        )
        return OfficeIntegration(
            name=name,
            module_path=module_path,
            module=module,
            attrs=attrs,
            status=status,
            is_legacy=is_legacy,
        )

    _log_office_status(name=name, status="missing", module_path=",".join(attempted))
    return OfficeIntegration(name=name, module_path="", module=None)


def log_office_factory_error(name: str, module_path: str, exc: Exception) -> None:
    """office factory 실행 실패를 표준 형식으로 기록한다."""
    _log_office_status(
        name=name,
        status="factory_error",
        module_path=module_path,
        error=f"{type(exc).__name__}: {exc}",
    )


def log_office_factory_loaded(name: str, module_path: str) -> None:
    """office factory 가 adapter 객체를 정상 생성했음을 기록한다."""
    _log_office_status(name=name, status="factory_loaded", module_path=module_path)


__all__ = [
    "OfficeIntegration",
    "load_office_integration",
    "log_office_factory_error",
    "log_office_factory_loaded",
]
