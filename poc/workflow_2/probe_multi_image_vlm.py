"""회사 direct 게이트웨이(Qwen3-VL, OpenAI-compatible)의 *멀티이미지 능력*을 찍는 capability probe.

목적
----
large VLM 으로 (align point 마커가 찍힌) rcp + msr 두 장을 함께 보내 region 을 묻는 전략을
실행하기 전에, **게이트웨이가 두 이미지를 한 요청에 받아 둘 다 추론에 쓰는지**를 먼저 검증한다.
OpenAI-compatible 이라도 다음에서 막힐 수 있다:

  1. vLLM ``--limit-mm-per-prompt image=1`` — prompt 당 이미지 상한 1 → 2장 요청이 4xx.
  2. proxy/게이트웨이 truncation — 첫/마지막 이미지만 포워딩하고 **200 OK 로 조용히 1장만** 처리.
  3. 포맷 — direct 게이트웨이가 WebP 를 거부하고 JPEG/PNG 만 받을 수 있다.
  4. payload 크기 — 이미지가 1MB 이상이면 처리 실패(현장 확인됨).

판별 방식 (discriminative)
--------------------------
"정확도"가 아니라 "무엇이 통과되는가"를 격리하기 위해 **정답이 결정적인 합산**을 쓴다.
  - reference 이미지: 정수 N1 + 빨간 align-point 마커(이미지 *중심*).
  - scene 이미지: 정수 N2.
  - 단일 이미지 테스트: scene 만 보내 "이 정수?" → GT=N2 (단일+포맷 검증).
  - 멀티이미지 테스트: 둘 다 보내 "두 정수의 합?" → GT=N1+N2. *둘 다 봐야만* 풀린다.
  - composite 테스트: 둘을 한 캔버스로 합쳐 1장으로 보내 "합?" → fallback 경로 검증.
합(N1+N2)이 나오면 둘 다 본 것, N1/N2 한쪽만 나오면 truncation 신호(PARTIAL).

실행
----
    uv run python poc/workflow_2/probe_multi_image_vlm.py

오피스(회사망)에서 실행해야 게이트웨이에 도달한다. Mac 에서는 네트워크 ERROR 가 정상이며,
생성된 테스트 이미지와 payload 크기 점검 결과는 그대로 확인할 수 있다. 결과는 콘솔 digest +
JSON 으로 저장되어 그대로 피드백에 붙여넣을 수 있다.
"""

import os

# OpenBLAS/OMP 스레드 수 제한 — 반드시 numpy/cv2 import *이전*에 설정해야 한다.
# Windows 다중코어 환경에서 OpenBLAS 가 스레드별 스크래치 버퍼 할당에 실패하며 내는
# "Memory allocation failed after 10 retries, giving up" 를 막는다. numpy 가 한 번
# import 된 뒤에는 효과가 없다. setdefault 라 외부에서 이미 지정했으면 존중한다.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests

from poc.workflow_2 import DEBUG_IMAGE_DIR

# ====================================================================
# 설정 (CLI 인자 없음 — 상수로만).
# ====================================================================

# large VLM 직접 연결 — 이 파일 안에서 standalone 으로 정의한다(flask_vlm 의존 제거).
# api_base 는 /v1 까지 포함한다고 가정한다 → 엔드포인트는 {api_base}/chat/completions.
# 오피스에서 실행 전 아래 두 값을 실제 게이트웨이 값으로 교체할 것.
LARGE_VLM_API_BASE = "http://workplace-litellm.aipp02.skhynix.com/v1"   # /v1 포함.
LARGE_VLM_API_KEY = ""                                                   # TODO: 새 api_key 로 교체.

# LiteLLM 게이트웨이에 등록된 large VLM 들 — 각 모델에 대해 동일한 매트릭스를 돈다.
# 모델명은 LiteLLM 의 alias 와 정확히 일치해야 한다(불일치 시 400 model_not_found).
MODELS = [
    "Qwen3-VL-30B-A3B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
]

# discriminative 합산용 두 정수. 한 자리 echo 와 헷갈리지 않게 두 자리 + 합도 두 자리.
REF_NUMBER = 17
SCENE_NUMBER = 25
EXPECTED_SUM = REF_NUMBER + SCENE_NUMBER  # 42

# 1MB 제약 — 이미지가 1MB 이상이면 처리 실패(현장 확인). base64 문자열 길이 기준으로 강제한다
# (data: URL 에 그대로 들어가는 실제 크기). 안전 마진 두고 0.95MB.
MAX_B64_CHARS = 950_000

TIMEOUT_SEC = 90.0
MAX_TOKENS = 64

OUTPUT_DIR = DEBUG_IMAGE_DIR / "probe_multi_image"

SYSTEM_MESSAGE = (
    "너는 이미지 속 숫자를 읽는 도구다. 요청한 정수 값만 아라비아 숫자로 답하라. "
    "설명·단위·문장 금지. 예: 42"
)


# ====================================================================
# 테스트 이미지 생성.
# ====================================================================


def _put_centered_number(canvas: np.ndarray, number: int, label: str,
                         cy_frac: float = 0.5) -> None:
    """캔버스에 큰 정수를 (가로 중앙, 세로 cy_frac 위치) 그리고 좌상단에 라벨을 그린다.

    cy_frac 으로 세로 위치를 올릴 수 있다 — reference 패널에서 중심 마커가 숫자를 가리지
    않도록 숫자를 위쪽에 둔다.
    """
    h, w = canvas.shape[:2]
    text = str(number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 4.0
    thick = 8
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    org = ((w - tw) // 2, int(h * cy_frac) + th // 2)
    cv2.putText(canvas, text, org, font, scale, (30, 30, 30), thick, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 34), font, 0.9, (90, 90, 90), 2, cv2.LINE_AA)


def make_reference_panel(number: int = REF_NUMBER, size: int = 360) -> np.ndarray:
    """reference 패널: 정수 + 이미지 *중심* 빨간 align-point 마커.

    마커 기준은 align point = 이미지 중심 규약을 따른다 (box 중심이 아님).
    """
    canvas = np.full((size, size, 3), 235, np.uint8)
    # 숫자는 위쪽(0.30)에, align-point 마커는 이미지 *중심*에 — 겹쳐 가리지 않도록 분리.
    _put_centered_number(canvas, number, "IMAGE-1 (reference)", cy_frac=0.30)
    cx, cy = size // 2, size // 2
    cv2.drawMarker(canvas, (cx, cy), (0, 0, 220), cv2.MARKER_CROSS, 34, 3, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 22, (0, 0, 220), 2, cv2.LINE_AA)
    return canvas


def make_scene_panel(number: int = SCENE_NUMBER, size: int = 360) -> np.ndarray:
    """scene 패널: 정수만 (마커 없음)."""
    canvas = np.full((size, size, 3), 220, np.uint8)
    _put_centered_number(canvas, number, "IMAGE-2 (scene)")
    return canvas


def make_composite(reference: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """reference + scene 을 구분선·라벨과 함께 좌우로 한 캔버스에 합친다 (fallback 경로)."""
    h = max(reference.shape[0], scene.shape[0])
    divider = 6
    left = cv2.copyMakeBorder(reference, 0, h - reference.shape[0], 0, 0,
                              cv2.BORDER_CONSTANT, value=(235, 235, 235))
    right = cv2.copyMakeBorder(scene, 0, h - scene.shape[0], 0, 0,
                               cv2.BORDER_CONSTANT, value=(220, 220, 220))
    sep = np.full((h, divider, 3), 40, np.uint8)
    composite = np.hstack([left, sep, right])
    cv2.putText(composite, "LEFT=image1  RIGHT=image2", (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2, cv2.LINE_AA)
    return composite


# ====================================================================
# 인코딩 — 1MB(base64) 제약 강제.
# ====================================================================


def encode_under_limit(
    bgr: np.ndarray, fmt: str, max_b64_chars: int = MAX_B64_CHARS
) -> tuple[str, str, int] | None:
    """이미지를 fmt(jpeg/webp)로 인코딩하되 base64 길이가 한도 이하가 되도록 품질/크기를 낮춘다.

    반환: (base64_str, mime, b64_chars) 또는 인코딩 불가 시 None.
    """
    import base64

    ext = ".jpg" if fmt == "jpeg" else ".webp"
    mime = "image/jpeg" if fmt == "jpeg" else "image/webp"
    qual_flag = cv2.IMWRITE_JPEG_QUALITY if fmt == "jpeg" else cv2.IMWRITE_WEBP_QUALITY

    img = bgr
    for scale in (1.0, 0.8, 0.6, 0.45):
        if scale != 1.0:
            img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for q in (92, 85, 75, 60, 45):
            ok, buf = cv2.imencode(ext, img, [int(qual_flag), int(q)])
            if not ok:
                return None  # 이 빌드가 해당 포맷 인코딩을 지원하지 않음.
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            if len(b64) <= max_b64_chars:
                return b64, mime, len(b64)
    return None  # 최대 축소에도 한도 초과.


# ====================================================================
# 요청 / 결과.
# ====================================================================


@dataclass
class ProbeResult:
    """단일 probe 요청의 결과 레코드."""

    name: str
    model: str
    n_images: int
    fmt: str
    expected: int
    ok_http: bool = False
    status_code: int | None = None
    latency_s: float | None = None
    payload_kb: float | None = None
    img_b64_kb: list[float] = field(default_factory=list)
    response_text: str = ""
    parsed_ints: list[int] = field(default_factory=list)
    verdict: str = "ERROR"   # PASS | PARTIAL | FAIL | ERROR | SKIP
    error: str = ""

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def _endpoint_and_headers() -> tuple[str, dict]:
    """large VLM 게이트웨이 endpoint URL 과 auth 헤더를 구성한다 (standalone 상수 기반)."""
    base = LARGE_VLM_API_BASE.rstrip("/")
    # api_base 가 /v1 포함 가정 — 혹시 빠졌으면 보정.
    endpoint = (
        f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    )
    headers = {"Content-Type": "application/json"}
    key = LARGE_VLM_API_KEY.strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    else:
        print("[WARNING] LARGE_VLM_API_KEY 가 비어 있음 — 401 가능 (파일 상단 상수 미설정).")
    return endpoint, headers


def _extract_text(data: object, raw_text: str) -> str:
    """OpenAI 형식 응답 body 에서 assistant 텍스트를 뽑는다 (실패 시 raw)."""
    try:
        msg = data["choices"][0]["message"]["content"]  # type: ignore[index]
        if isinstance(msg, list):
            msg = " ".join(p.get("text", "") for p in msg if isinstance(p, dict))
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    except (KeyError, IndexError, TypeError):
        pass
    return (raw_text or "").strip()


def _verdict(parsed: list[int], expected: int, name: str) -> str:
    """파싱된 정수와 기대값으로 판정한다."""
    if not parsed:
        return "FAIL"
    if expected in parsed:
        return "PASS"
    # 합산 테스트에서 operand 만 나오면 truncation/미합산 신호.
    if name.startswith(("multi", "composite")) and (REF_NUMBER in parsed or SCENE_NUMBER in parsed):
        return "PARTIAL"
    return "FAIL"


def run_probe(name: str, model: str, content_blocks: list[dict], n_images: int, fmt: str,
              expected: int, img_b64_kb: list[float]) -> ProbeResult:
    """content_blocks 를 large VLM 게이트웨이에 보내고 결과를 ProbeResult 로 반환한다."""
    res = ProbeResult(name=name, model=model, n_images=n_images, fmt=fmt, expected=expected,
                      img_b64_kb=img_b64_kb)
    endpoint, headers = _endpoint_and_headers()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content_blocks},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    res.payload_kb = round(len(json.dumps(payload)) / 1024, 1)

    t0 = time.time()
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=TIMEOUT_SEC)
        res.latency_s = round(time.time() - t0, 2)
        res.status_code = resp.status_code
        resp.raise_for_status()
        res.ok_http = True
        try:
            data = resp.json()
        except ValueError:
            data = None
        res.response_text = _extract_text(data, resp.text)[:300]
        res.parsed_ints = [int(x) for x in re.findall(r"-?\d+", res.response_text)]
        res.verdict = _verdict(res.parsed_ints, expected, name)
    except requests.RequestException as exc:
        res.latency_s = round(time.time() - t0, 2)
        res.error = str(exc)[:300]
        status = getattr(getattr(exc, "response", None), "status_code", None)
        res.status_code = status
        res.verdict = "ERROR"
    print(
        f"[INFO] {model:<26s} {name:<14s} fmt={fmt:<4s} imgs={n_images} "
        f"status={res.status_code} verdict={res.verdict} "
        f"resp={res.response_text!r} err={res.error[:60]}"
    )
    return res


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _image_block(b64: str, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


# ====================================================================
# 엔트리.
# ====================================================================


def _run_model_matrix(model: str, ref: np.ndarray, scene: np.ndarray,
                      composite: np.ndarray) -> list[ProbeResult]:
    """한 모델에 대해 (jpeg/webp) × (single/multi/composite) 매트릭스를 돈다."""
    out: list[ProbeResult] = []
    for fmt in ("jpeg", "webp"):
        ref_enc = encode_under_limit(ref, fmt)
        scene_enc = encode_under_limit(scene, fmt)
        comp_enc = encode_under_limit(composite, fmt)
        if ref_enc is None or scene_enc is None or comp_enc is None:
            print(f"[WARNING] fmt={fmt}: 인코딩 실패(1MB 초과 또는 미지원) — 해당 포맷 SKIP")
            for nm, ni in (("single", 1), ("multi_two", 2), ("composite", 1)):
                out.append(ProbeResult(name=f"{nm}_{fmt}", model=model, n_images=ni, fmt=fmt,
                                       expected=SCENE_NUMBER if nm == "single" else EXPECTED_SUM,
                                       verdict="SKIP", error="encode_failed"))
            continue

        ref_b64, mime, ref_kb = ref_enc
        scene_b64, _, scene_kb = scene_enc
        comp_b64, _, comp_kb = comp_enc
        ref_kb, scene_kb, comp_kb = round(ref_kb / 1024, 1), round(scene_kb / 1024, 1), round(comp_kb / 1024, 1)

        # 1) 단일 이미지 (scene) — baseline: 단일 + 포맷.
        out.append(run_probe(
            f"single_{fmt}", model,
            [_text_block("이 이미지에 적힌 정수 하나를 숫자로만 답하라."),
             _image_block(scene_b64, mime)],
            n_images=1, fmt=fmt, expected=SCENE_NUMBER, img_b64_kb=[scene_kb],
        ))

        # 2) 네이티브 멀티이미지 — discriminative 합산.
        out.append(run_probe(
            f"multi_two_{fmt}", model,
            [_text_block("첫 번째 이미지(IMAGE-1)의 정수와 두 번째 이미지(IMAGE-2)의 정수를 "
                         "더한 값을 숫자로만 답하라."),
             _image_block(ref_b64, mime),
             _image_block(scene_b64, mime)],
            n_images=2, fmt=fmt, expected=EXPECTED_SUM, img_b64_kb=[ref_kb, scene_kb],
        ))

        # 3) composite fallback — 한 장으로 합산.
        out.append(run_probe(
            f"composite_{fmt}", model,
            [_text_block("이 이미지의 왼쪽(image1) 정수와 오른쪽(image2) 정수를 더한 값을 "
                         "숫자로만 답하라."),
             _image_block(comp_b64, mime)],
            n_images=1, fmt=fmt, expected=EXPECTED_SUM, img_b64_kb=[comp_kb],
        ))
    return out


def main() -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = make_reference_panel()
    scene = make_scene_panel()
    composite = make_composite(ref, scene)

    # 생성 이미지 저장 — Mac 에서도 눈으로 확인 가능 (JPEG, 코드 컨벤션).
    cv2.imwrite(str(OUTPUT_DIR / "panel_reference.jpg"), ref)
    cv2.imwrite(str(OUTPUT_DIR / "panel_scene.jpg"), scene)
    cv2.imwrite(str(OUTPUT_DIR / "panel_composite.jpg"), composite)
    print(f"[INFO] 테스트 이미지 저장: {OUTPUT_DIR}")
    print(f"[INFO] 기대값: 단일 scene={SCENE_NUMBER}, 합산={EXPECTED_SUM}")
    print(f"[INFO] 대상 모델: {', '.join(MODELS)}")

    results: list[ProbeResult] = []
    for model in MODELS:
        print(f"\n[INFO] ===== model={model} =====")
        results.extend(_run_model_matrix(model, ref, scene, composite))

    digest = _build_digest(results)
    print(digest)

    recommendation = {model: _recommend([r for r in results if r.model == model])
                      for model in MODELS}
    out_json = OUTPUT_DIR / "probe_result.json"
    out_json.write_text(
        json.dumps({"results": [r.to_dict() for r in results],
                    "recommendation": recommendation},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "probe_digest.txt").write_text(digest, encoding="utf-8")
    print(f"[INFO] 저장: {out_json}")
    return digest


def _by_name(results: list[ProbeResult]) -> dict[str, ProbeResult]:
    return {r.name: r for r in results}


def _recommend(results: list[ProbeResult]) -> str:
    """결과로부터 라우팅 권고를 도출한다."""
    m = _by_name(results)

    def passed(name: str) -> bool:
        return name in m and m[name].verdict == "PASS"

    native_ok = passed("multi_two_jpeg") or passed("multi_two_webp")
    composite_ok = passed("composite_jpeg") or passed("composite_webp")
    single_ok = passed("single_jpeg") or passed("single_webp")
    webp_ok = any(m.get(n) and m[n].verdict == "PASS" for n in
                  ("single_webp", "multi_two_webp", "composite_webp"))
    fmt_note = "WebP OK" if webp_ok else "WebP 거부/실패 → JPEG 사용"

    if native_ok:
        return (f"NATIVE 멀티이미지 지원 → chat_with_images() 경로 사용. ({fmt_note})")
    if composite_ok:
        return ("멀티이미지 미지원/truncation → COMPOSITE fallback 사용 "
                f"(단일 이미지라 서버 변경 불필요). ({fmt_note})")
    if single_ok:
        return ("단일 이미지만 동작, composite 추론도 실패 → describe→search(2-call) 검토. "
                f"({fmt_note})")
    return "단일 이미지조차 실패 — 게이트웨이 연결/키/포맷부터 점검 (오피스망에서 실행했는지 확인)."


def _build_digest(results: list[ProbeResult]) -> str:
    lines = ["", "=" * 72, "멀티이미지 capability probe 결과", "=" * 72]
    models = list(dict.fromkeys(r.model for r in results))  # 입력 순서 유지.
    for model in models:
        model_results = [r for r in results if r.model == model]
        lines.append(f"■ model={model}")
        lines.append(f"{'test':<20s}{'fmt':<6s}{'imgs':<5s}{'status':<8s}"
                     f"{'verdict':<9s}{'payloadKB':<11s}resp")
        lines.append("-" * 72)
        for r in model_results:
            lines.append(
                f"{r.name:<20s}{r.fmt:<6s}{r.n_images:<5d}"
                f"{str(r.status_code):<8s}{r.verdict:<9s}"
                f"{str(r.payload_kb):<11s}{r.response_text or r.error[:40]!r}"
            )
        lines.append("-" * 72)
        lines.append(f"권고: {_recommend(model_results)}")
        lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
