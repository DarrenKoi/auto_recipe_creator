"""tool 창 버튼 레지스트리 + 공통 프롬프트 템플릿 + 레지스트리 기반 화면 inventory.

버튼마다 프롬프트를 손으로 고치던 것을(File Manager 한 번 누르는 데 오피스 6회) 데이터
한 줄로 바꾼다. 호출부는 이름 하나만 넘기고, 버튼별 차이(라벨, 창 안 위치, OCR needle,
확인 crop 크기, 클릭 뒤 열릴 창)는 `ButtonSpec` 에 산다. 프롬프트 문장은 `describe` 한
곳에서만 만든다.

식별은 라벨이 아니라 **(window, label)** 이다. OK 처럼 여러 창에 있는 라벨은 이름만으로
고르지 않고 거부한다(`resolve`). 단, 지금 클릭 경로는 tool 본 화면(`window="tool"`) 버튼만
지원한다 - 팝업 안 버튼은 탐색·OCR 을 **그 팝업 내부로 제한**해야 다른 창의 OK 를 누르지
않는데, 원격 뷰의 팝업은 로컬 창이 아니라 그 영역을 아직 검증할 수단이 없다(Codex 검토
2026-09-19).

이 모듈은 VLM/OCR/마우스를 직접 부르지 않는다 - 전부 주입받아 Mac 에서 시험된다.
"""

from dataclasses import dataclass

TOOL_WINDOW = "tool"


@dataclass(frozen=True)
class ButtonSpec:
    """버튼 하나. 좌표 비율은 Remote Monitoring(tool) 창 이미지 기준이다."""

    key: str
    label: str                     # 프롬프트에 그대로 들어가는 전체 라벨
    required: tuple                # OCR 확인 needle 묶음(묶음 하나를 통째로 만족)
    window: str = TOOL_WINDOW      # 'tool' = 본 화면, 그 외 = 팝업 이름(클릭 미지원)
    hint: str = ""                 # 창 안 위치 서술(전체 화면 탐색 때만 쓴다)
    center: tuple | None = None    # (x, y) 창 비율 - 모르면 None(첫 실행이 출력한다)
    search_half: tuple = (0.25, 0.08)    # 탐색 crop 반폭/반높이(창 비율)
    confirm_half: tuple = (0.10, 0.015)  # 라벨 확인/inventory OCR crop - 버튼 한 개 크기
    reveal: bool = False           # 가림 해제 Alt+click 허용(오피스 검증된 버튼만)
    whole_word: bool = False       # needle 을 단어 전체로만 인정('amp' 가 'Sample' 에 걸리지 않게)
    opens_title: tuple = ()        # 클릭 뒤 열릴 창 제목 needle 묶음 - 비면 효과 미검증
    opens_description: str = ""    # 그 제목줄을 VLM 에 설명하는 문장

    def __post_init__(self):
        # 빈 required 는 `_confirm_point` 가 strict 에서도 확인을 건너뛴다(forbidden 만 본다).
        # 레지스트리 버튼은 반드시 라벨로 확인되어야 하므로 등록 단계에서 막는다.
        if not self.required or not all(self.required):
            raise ValueError(f"{self.key}: required needle 이 비었습니다")


def resolve(name: str, specs) -> ButtonSpec:
    """이름으로 spec 을 고른다: key > 'window/label' > label(대소문자 무시).

    라벨이 여러 창에 걸리면 임의로 고르지 않고 후보를 나열해 거부한다.
    """
    wanted = (name or "").strip().lower()
    for spec in specs:
        if spec.key.lower() == wanted:
            return spec
    for spec in specs:
        if f"{spec.window}/{spec.label}".lower() == wanted:
            return spec
    hits = [spec for spec in specs if spec.label.lower() == wanted]
    if len(hits) == 1:
        return hits[0]
    if hits:
        keys = ", ".join(f"{s.key}({s.window}/{s.label})" for s in hits)
        raise ValueError(f"'{name}' 은 여러 창에 있습니다 - key 나 'window/label' 로 지정: {keys}")
    raise ValueError(f"등록되지 않은 버튼 '{name}'. 등록된 key: "
                     + ", ".join(spec.key for spec in specs))


def _word_tokens(spec: ButtonSpec, tokens) -> list:
    """whole_word 버튼은 needle 과 **정확히 같은 단어**만 남기고 나머지 토큰을 가린다.

    확인 판정(`classify_label`)은 토큰 부분 일치라 짧은 라벨이 이웃 단어에 걸린다(AMP ->
    'Sample'). 판정 함수를 바꾸면 다른 확인 경로가 전부 흔들리므로 여기서 토큰만 거른다.
    """
    tokens = list(tokens)
    if not spec.whole_word:
        return tokens
    needles = {needle for group in spec.required for needle in group}
    words = ("".join(ch for ch in t.lower() if ch.isalnum()) for t in tokens)
    return [t if w in needles else "" for t, w in zip(tokens, words)]


def describe(spec: ButtonSpec, *, with_hint: bool = True) -> str:
    """모든 버튼이 공유하는 VLM 설명문. 버튼별 차이는 spec 필드로만 들어온다.

    첫 글자 anchor 는 쓰지 않는다 - 같은 글자로 시작하는 라벨이 많아 이웃을 짚었다
    (File Manager, 2026-09-18). crop 탐색에서는 hint 를 뺀다: '화면 아래쪽 오른쪽' 같은
    전체 화면 서술이 crop 좌표계에서는 틀린 말이 된다. 창 소속은 crop 에서도 남긴다.
    """
    text = f"the button labeled exactly '{spec.label}'"
    if spec.window != TOOL_WINDOW:
        text += f" in the '{spec.window}' window"
    if with_hint and spec.hint:
        text += f". It is {spec.hint}"
    return (text + ". Ignore buttons whose label only shares a word or the first letter. "
            "Click the center of that button.")


def ratio_box(spec: ButtonSpec, width: int, height: int, half: tuple) -> dict | None:
    """spec.center 를 중심으로 창 비율 반폭/반높이 box(px, 경계 clamp). center 없으면 None."""
    if spec.center is None:
        return None
    from poc.workflow_3.vlm.label_verify import crop_box_around_point

    point = {"x": int(width * spec.center[0]), "y": int(height * spec.center[1])}
    return crop_box_around_point(
        point, width, height,
        left_ratio=half[0], right_ratio=half[0], half_height_ratio=half[1],
    )


def find_button(image, spec: ButtonSpec, *, locate_fn, read_tokens_fn, policy: str = "strict"):
    """버튼을 찾아 OCR 로 확인된 점만 돌려준다: `(point, reason, source)`.

    등록 위치가 있으면 그 영역 crop 에서 먼저 찾는다(후보가 줄고 확대된다). crop 결과가
    **미검출이든 라벨 불일치든** 전체 화면(hint 포함)으로 한 번 더 찾는다. 좌표는 VLM,
    확인은 전체 이미지 위 좁은 OCR crop(`_confirm_point` - 확인 규약을 포크하지 않는다).
    source 는 'region' | 'full'.
    """
    from poc.workflow_3.monitor.demonstration_rcs_control import FlowStep, _confirm_point
    from poc.workflow_3.vlm.ui_venus_mai_locator import TargetConfig

    def _read_tokens(image_, point, key):
        return _word_tokens(spec, read_tokens_fn(image_, point, key))

    box = ratio_box(spec, image.width, image.height, spec.search_half)
    if box is not None:
        crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))

        def _locate_in_crop(_image, target):
            found = locate_fn(crop, target)
            if found is None:
                return None
            return {"x": int(found["x"]) + box["left"], "y": int(found["y"]) + box["top"]}

        step = FlowStep(TargetConfig(key=spec.key, description=describe(spec, with_hint=False)),
                        required=spec.required)
        point, reason = _confirm_point(image, step, locate_fn=_locate_in_crop,
                                       read_tokens_fn=_read_tokens, policy=policy)
        if point is not None:
            return point, reason, "region"
        print(f"[INFO] {spec.key}: 등록 영역에서 확인 실패({reason}) - 전체 화면에서 다시 찾습니다")

    step = FlowStep(TargetConfig(key=spec.key, description=describe(spec)), required=spec.required)
    point, reason = _confirm_point(image, step, locate_fn=locate_fn,
                                   read_tokens_fn=_read_tokens, policy=policy)
    return point, reason, "full"


def inventory(image, specs, *, read_fn) -> list:
    """등록 버튼마다 **버튼 한 개 크기** 라벨 영역을 OCR 로 읽어 상태 표를 만든다.

    `read_fn(image, box, key) -> tokens` (예외 = 판독 실패). 상태:
      label_seen : 그 자리에 라벨 글자가 읽혔다(클릭 가능하다는 뜻은 아니다)
      not_seen   : 다른 글자/빈칸 - 가림·이동·커서 겹침·OCR 누락이 섞여 있다
      read_error : OCR 호출 실패
      no_region  : 위치 미등록(첫 클릭 실행이 비율을 출력한다)
    **클릭 승인에 쓰지 않는다** - 넓은 탐색 영역이 아니라 좁은 라벨 영역만 읽는 것도
    여러 줄 crop 에서 OCR 이 위 줄만 읽던 실패(2026-09-18)를 되풀이하지 않으려는 것이다.
    """
    rows = []
    for spec in specs:
        row = {"key": spec.key, "label": spec.label, "window": spec.window, "tokens": []}
        box = ratio_box(spec, image.width, image.height, spec.confirm_half)
        if box is None:
            row["status"] = "no_region"
        else:
            try:
                tokens = list(read_fn(image, box, spec.key))
            except Exception as exc:
                row["status"], row["error"] = "read_error", f"{type(exc).__name__}: {exc}"
            else:
                row["tokens"] = tokens
                seen = label_in_tokens(_word_tokens(spec, tokens), spec.required)
                row["status"] = "label_seen" if seen else "not_seen"
        rows.append(row)
    return rows


def label_in_tokens(tokens, required) -> bool:
    """required 묶음 하나의 needle 이 전부 읽혔는가('FileManager' 로 붙어도 통과)."""
    text = " ".join(tokens).lower()
    return any(all(needle in text for needle in group) for group in required)


def poll_until(check_fn, *, timeout_sec: float, interval_sec: float, clock, sleep):
    """효과가 확인될 때까지 반복 확인한다: `(found, elapsed_sec, checks)`.

    elapsed 는 '클릭 후 처음 확인된 시각' 이지 창이 실제로 뜬 시각이 아니다 - 확인 한 번의
    VLM+OCR 지연이 포함된다. 다음 확인이 timeout 을 넘길 차례면 더 보지 않는다. 확인이
    안 돼도 호출부는 다시 누르지 않는다(떴는데 확인만 놓쳤을 수 있다).
    """
    start = clock()
    checks = 0
    while True:
        checks += 1
        if check_fn():
            return True, round(clock() - start, 3), checks
        elapsed = clock() - start
        if elapsed + interval_sec > timeout_sec:
            return False, round(elapsed, 3), checks
        sleep(interval_sec)


# ===========================================================================
# 등록 버튼 - 새 버튼은 여기 한 줄. center 를 모르면 None 으로 두고 첫 클릭 실행이
# 출력하는 `center=(x, y)` 를 옮겨 적는다(그 뒤로는 등록 영역 crop 에서 먼저 찾는다).
# reveal=True 는 가림 해제 Alt+click 이 오피스에서 확인된 버튼만(Codex 검토: 덮은 창
# 제목을 긍정 확인하지 않는 Alt+click 을 새 버튼으로 일반화하지 않는다).
# ===========================================================================

_BOTTOM_GROUP = "in the group of buttons along the BOTTOM of the Remote Monitoring screen"

BUTTONS = (
    ButtonSpec(
        key="file_manager",
        label="File Manager",
        hint=f"{_BOTTOM_GROUP}, on the RIGHT side of that group",
        # 'manag': 가림 해제 뒤 장비 커서가 끝 글자를 가려 'Manage' 로 읽혔다(2026-09-18).
        required=(("file", "manag"),),
        center=(0.80, 0.90),
        reveal=True,
        # 제목 'File Manager( Class, IDW, IDP, Recipe )' - 'File Manager' 만으로는 방금
        # 누른 버튼도 통과하므로 제목에만 있는 IDW/Recipe 를 함께 요구한다.
        opens_title=(("manag", "idw"), ("manag", "recipe")),
        opens_description=(
            "the title bar text of the large 'File Manager' window that just opened, which "
            "reads 'File Manager( Class, IDW, IDP, Recipe )'. Point at the middle of that "
            "title text, not at the 'File Manager' button in the bottom button group."
        ),
    ),
    ButtonSpec(
        key="amp",
        label="AMP",
        hint=f"{_BOTTOM_GROUP}, directly BELOW the 'File Manager' button",
        required=(("amp",),),
        whole_word=True,
    ),
    ButtonSpec(
        key="rotation",
        label="Rot.",
        hint="near the 'PM' button, which sits next to the live SEM image",
        required=(("rot",),),
    ),
)
