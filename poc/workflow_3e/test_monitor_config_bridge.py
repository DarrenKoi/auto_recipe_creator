"""모니터 entry point 가 workflow_3_config.py 브리지를 실제로 태우는지 검증한다.

`workflow_3_config.py` 의 토글(SAFE_MODE / ZOOM_PROBE / CORRECTION_DRY_RUN ...)은
`seed_env()` 가 env 로 옮겨줘야 적용된다. 그래서 **모든** 모니터 진입점의 `__main__` 은
`load_*_settings()` 가 돌기 전에 seed_env() 를 불러야 한다.

빠뜨리면 조용히 실패한다 - 파일에 안전 토글을 적어두고 모니터를 띄웠는데 그 진입점만
브리지를 안 태우면, 사용자는 안전하다고 믿지만 코드 기본값(SAFE_MODE=0 = 클릭 허용)으로
돈다. 로그에도 티가 안 난다. 그래서 소스 수준으로 못박는다.

`__main__` 블록은 import 로 실행되지 않으므로 AST 로 확인한다(문자열 매칭 아님 - 주석이나
docstring 에 seed_env 가 적혀 있어도 통과하지 않는다).

    uv run python poc/workflow_3e/test_monitor_config_bridge.py
"""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# 브리지를 태워야 하는 진입점 전부. 새 모니터를 추가하면 여기에도 넣는다.
ENTRY_POINTS = (
    "poc/workflow_3/monitor/align_fail_monitor.py",
    "poc/workflow_3/monitor/align_fail_monitor_only_check.py",
    "poc/workflow_3e/monitor.py",
)


def _is_main_guard(node) -> bool:
    """`if __name__ == "__main__":` 노드인지."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return False
    left, right = test.left, test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def _calls_seed_env(node) -> bool:
    """블록 안에서 seed_env() 를 '호출' 하는지 (언급이 아니라 호출)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and func.id == "seed_env":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "seed_env":
                return True
    return False


def _check(rel_path: str) -> bool:
    path = _REPO_ROOT / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guards = [n for n in tree.body if _is_main_guard(n)]
    if not guards:
        print(f"[FAIL] {rel_path}: __main__ 가드 없음")
        return False
    ok = any(_calls_seed_env(g) for g in guards)
    print(f"[{'PASS' if ok else 'FAIL'}] {rel_path}: seed_env() 호출 {'있음' if ok else '없음'}")
    return ok


def test_all_monitor_entry_points_seed_config():
    return all([_check(p) for p in ENTRY_POINTS])


def main():
    print("[INFO] 모니터 config 브리지 배선 검사")
    ok = test_all_monitor_entry_points_seed_config()
    print(f"[INFO] {'전부 통과' if ok else '누락 있음'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
