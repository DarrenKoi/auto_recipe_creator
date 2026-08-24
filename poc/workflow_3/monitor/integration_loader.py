"""office 전용 adapter 로딩 helper.

`office_*` 파일은 gitignore 대상이라 오피스 PC 에만 존재한다. monitor 쪽에서는
정위치(workflow_3.monitor)에서 모듈을 찾고, 실제로 연결됐는지 콘솔과 workflow_3
로그에 남긴다.
"""

import importlib
import traceback
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
    error: str = ""

    @property
    def available(self) -> bool:
        """호출 가능한 office adapter 를 찾았는지."""
        return self.module is not None and self.status == "loaded"


def _exc_location(exc: BaseException) -> str:
    """예외가 실제로 난 마지막 프레임을 ``file:line`` 으로 요약한다.

    office_* 는 gitignore 라 오피스 PC 에만 존재하고 개발 PC 에서 열어볼 수 없다.
    타입과 메시지만으로는(예: "TypeError: 'NoneType' object is not subscriptable")
    그 파일의 어느 줄인지 특정할 수 없어 진단이 통째로 막힌다 - 실제로 이 형태의
    한 줄만 보고는 원인을 좁힐 수 없었다. 마지막 프레임을 함께 남겨 바로 짚게 한다.
    """
    location = ""
    for frame, lineno in traceback.walk_tb(exc.__traceback__):
        location = f"{frame.f_code.co_filename}:{lineno}"
    return location


def _print_exc_traceback(exc: BaseException) -> None:
    """office adapter import/factory 실패의 전체 traceback 을 들여쓰기해 찍는다.

    콘솔 한 줄 요약(status/error)과 별개로 전체를 남긴다. import 실패는 드물지만
    그 기능 전체를 무력화하므로(consensus gather 영구 비활성 등) 첫 발생에서
    원인을 확정하는 편이 재현을 기다리는 것보다 싸다.
    """
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print("[WARNING] office adapter traceback (맨 아래 프레임이 실제 원인 줄):")
    for line in text.rstrip().splitlines():
        print(f"    {line}")


def _log_office_status(
    *,
    name: str,
    status: str,
    module_path: str = "",
    error: str = "",
    missing_attrs: list[str] | None = None,
    location: str = "",
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
    if missing_attrs:
        details.append(f"missing_attrs={','.join(missing_attrs)}")
    if error:
        details.append(f"error={error}")
    if location:
        details.append(f"at={location}")

    print(f"{prefix} office integration: " + " ".join(details))

    try:
        from poc.workflow_3.logger import log_work2_event

        log_work2_event(
            component="office_integrations",
            message=status,
            level=level,
            name=name,
            module_path=module_path,
            missing_attrs=",".join(missing_attrs or []),
            error=error,
            at=location,
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
    module_path: str,
    *,
    required_attrs: tuple[str, ...] = (),
) -> OfficeIntegration:
    """office adapter 모듈을 정위치(workflow_3)에서 찾는다.

    모듈이 아예 없으면(개발 PC 의 정상 케이스) 조용히 missing 을, 모듈은 있으나 내부
    import 가 깨졌거나 required attr 가 빠졌으면 그 사유를 로그로 남기고 빈
    OfficeIntegration 을 돌려준다(어느 경우든 monitor import 자체는 죽지 않는다).
    """
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if _is_missing_target_module(exc, module_path):
            _log_office_status(name=name, status="missing", module_path=module_path)
        else:
            _log_office_status(
                name=name,
                status="import_error",
                module_path=module_path,
                error=f"{type(exc).__name__}: {exc}",
                location=_exc_location(exc),
            )
            _print_exc_traceback(exc)
        return OfficeIntegration(name=name, module_path="", module=None)
    except Exception as exc:
        _log_office_status(
            name=name,
            status="import_error",
            module_path=module_path,
            error=f"{type(exc).__name__}: {exc}",
            location=_exc_location(exc),
        )
        _print_exc_traceback(exc)
        return OfficeIntegration(name=name, module_path="", module=None)

    attrs = {attr: getattr(module, attr, None) for attr in required_attrs}
    missing_attrs = [attr for attr, value in attrs.items() if value is None]
    if missing_attrs:
        _log_office_status(
            name=name,
            status="missing_attr",
            module_path=module_path,
            missing_attrs=missing_attrs,
        )
        return OfficeIntegration(name=name, module_path="", module=None)

    _log_office_status(name=name, status="loaded", module_path=module_path)
    return OfficeIntegration(
        name=name,
        module_path=module_path,
        module=module,
        attrs=attrs,
        status="loaded",
    )


def log_office_factory_error(name: str, module_path: str, exc: Exception) -> None:
    """office factory 실행 실패를 표준 형식으로 기록한다."""
    _log_office_status(
        name=name,
        status="factory_error",
        module_path=module_path,
        error=f"{type(exc).__name__}: {exc}",
        location=_exc_location(exc),
    )
    _print_exc_traceback(exc)


def log_office_factory_loaded(name: str, module_path: str) -> None:
    """office factory 가 adapter 객체를 정상 생성했음을 기록한다."""
    _log_office_status(name=name, status="factory_loaded", module_path=module_path)


__all__ = [
    "OfficeIntegration",
    "load_office_integration",
    "log_office_factory_error",
    "log_office_factory_loaded",
]
