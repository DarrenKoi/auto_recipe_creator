"""Search movement primitives shared by align live search and diagnostics."""


def square_spiral_step(idx: int, step: int) -> tuple[int, int]:
    """Return the idx-th square-spiral delta from the previous position."""
    if idx <= 0:
        return 0, 0
    leg = 0
    cum = 0
    while cum < idx:
        leg += 1
        leg_length = (leg + 1) // 2
        cum += leg_length
    direction = (leg - 1) % 4
    if direction == 0:
        return step, 0
    if direction == 1:
        return 0, -step
    if direction == 2:
        return -step, 0
    return 0, step


# Backward-compatible internal name for moved diagnostics that still use it locally.
_square_spiral_step = square_spiral_step

__all__ = ["square_spiral_step"]
