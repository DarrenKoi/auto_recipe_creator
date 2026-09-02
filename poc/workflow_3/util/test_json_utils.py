"""bbox 거부 사유 분류 - "미검출" 로그가 실제로 무엇을 말하는지 고정한다.

이 분류가 흔들리면 오피스 콘솔이 다시 "못 찾았다" 한 덩어리가 된다. absent(모델이
없다고 답함) 와 degenerate/bad_shape(좌표는 왔는데 우리가 버림)는 고칠 곳이 정반대라
둘을 섞으면 안 된다.

    uv run pytest poc/workflow_3/util/test_json_utils.py
"""

from poc.workflow_3.util.json_utils import describe_bbox_reject, normalize_bbox_1000


def test_모델이_없다고_답하면_absent():
    assert describe_bbox_reject(None) == "absent"


def test_음수_좌표는_명시적_거부():
    assert describe_bbox_reject([-1, -1, -1, -1]) == "refusal"
    assert describe_bbox_reject({"left": -1, "top": -1, "right": -1, "bottom": -1}) == "refusal"


def test_폭이_0_이면_degenerate():
    assert describe_bbox_reject([500, 500, 500, 500]) == "degenerate"


def test_규약과_다른_모양은_bad_shape():
    assert describe_bbox_reject({"box": [1, 2, 3, 4]}) == "bad_shape"
    assert describe_bbox_reject([1, 2, 3]) == "bad_shape"


def test_숫자가_아니면_non_numeric():
    assert describe_bbox_reject(["a", "b", "c", "d"]) == "non_numeric"


def test_정상_bbox_는_ok_이고_normalize_와_판정이_일치한다():
    """ok 를 말했는데 normalize 가 None 이면 로그가 거짓말을 하게 된다."""
    for raw in ([10, 10, 900, 900], {"x": 10, "y": 10, "w": 100, "h": 100}):
        assert describe_bbox_reject(raw) == "ok"
        assert normalize_bbox_1000(raw) is not None
    for raw in (None, [-1, -1, -1, -1], [5, 5, 5, 5], {"box": 1}, ["a", "b", "c", "d"]):
        assert describe_bbox_reject(raw) != "ok"
        assert normalize_bbox_1000(raw) is None
