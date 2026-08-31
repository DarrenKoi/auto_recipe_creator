"""환경변수 파싱 유틸리티."""

import os


def env_flag(name: str, default: bool = False) -> bool:
    """bool 환경변수를 파싱한다."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int) -> int:
    """int 환경변수를 읽고 잘못된 값이면 default 를 사용한다."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError:
        print(f"[WARNING] {name} 값이 잘못되었습니다. default={default} 사용: {raw_value!r}")
        return default


def env_float(name: str, default: float) -> float:
    """float 환경변수를 읽고 잘못된 값이면 default 를 사용한다."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return float(raw_value)
    except ValueError:
        print(f"[WARNING] {name} 값이 잘못되었습니다. default={default} 사용: {raw_value!r}")
        return default


def as_env_value(value) -> str:
    """상수 하나를 env 문자열로 바꾼다 (미설정이면 "").

    리스트를 받는 이유는 py 파일에서 경로/라벨을 콤마 문자열로 적는 것이 실수하기
    쉽기 때문이다(따옴표 안에서 콤마를 빠뜨리면 두 항목이 한 덩어리가 된다).
    bool 은 "True"/"False" 가 아니라 "1"/"0" 으로 내린다 - 같은 값을 숫자 파서가
    읽을 수도 있는 자리라 문자열 표기를 하나로 맞춘다.

    None 과 빈 문자열은 "미설정"이라 "" 를 준다. 숫자 0 은 **유효한 값**이므로
    "0" 이 된다(명시적 off 를 미설정과 구분해야 한다).
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def seed_env_from_constants(namespace, pairs, label="파일 상수") -> None:
    """진입점 파일 상단 상수를 env 로 setdefault 한다 (셸 env 가 항상 이긴다).

    이 프로젝트는 CLI 인자를 쓰지 않는다. "인자"는 실행하는 파일 맨 위의 상수이고,
    우선순위는 **실제 셸 env > 파일 상수 > 코드 기본값** 이다. 흐름은 한 방향뿐이다:

        상수 -> os.environ -> 기존 reader

    그래서 이 함수를 빼도 env 로 돌던 사용법이 그대로 살아 있고, 읽는 쪽 코드는
    상수의 존재를 몰라도 된다. 새 설정 체계가 아니라 기존 env 로 가는 다리다.

    셸 env 때문에 무시된 상수는 **반드시 콘솔에 남긴다** - 파일을 고쳤는데 예전 env 가
    export 된 채 남아 있어 다른 동작이 나오는 사고가 이 규약에서 제일 흔한 실수다.
    사본 설정 파일에는 없는 자기고발 장치이며, 이것 때문에 파일 상수가 폴더 사본보다
    안전하다.

    Args:
        namespace: 상수가 사는 모듈의 globals() dict.
        pairs: (상수명, env 이름) 순서쌍. 같은 뜻이면 새 env 이름을 만들지 말 것.
        label: 콘솔 출력에 쓰는 이름(진입점이 여럿일 때 어느 블록인지 구분).
    """
    applied = []
    ignored = []
    for const_name, env_name in pairs:
        value = as_env_value(namespace.get(const_name))
        if not value:
            continue
        if os.environ.get(env_name, "").strip():
            ignored.append(f"{const_name}(env {env_name}={os.environ[env_name]} 우선)")
            continue
        os.environ[env_name] = value
        applied.append(f"{const_name}={value}")

    if applied:
        print(f"[INFO] {label} 적용: {', '.join(applied)}")
    if ignored:
        print(f"[INFO] {label} 무시(셸 env 우선): {', '.join(ignored)}")
