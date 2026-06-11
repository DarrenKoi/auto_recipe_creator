"""엔지니어 수동 align 보정 완료 감지 — Recipe Monitor 측정 카운터 기반.

미보정 engineer watch 동안 "측정이 시작됐다"(= align 완료)를 감지해 녹화를 조기
종료한다. 신호는 tool 창 Recipe Monitor 의 측정 점 카운터 분자(N/M 의 N)가
증가하는 것 (1/350 -> 2/350 -> ...).

hybrid 파이프라인 (grounding 은 성공 시 1회 캐시):
  1. grounding(성공 시 캐시): VLM(ui-venus)으로 분자 위치를 찾아 tool-window
     상대비율 ROI 로 캐시한다 (tool 마다/드래그로 위치가 달라 고정 ROI 불가).
     **오피스 관찰(2026-06-11): re-align 진행 중에는 카운터(N/M)가 빈칸**이라
     grounding 거부([-1,-1])가 정상 상태다. 따라서 거부는 영구 포기가 아니라
     `reground_sec` 간격 재시도 — 측정이 시작되면 숫자가 나타나 성공한다.
  2. CV gate(매 호출): ROI crop 변화감지 — align-fix 중엔 카운터가 정적이라
     OCR 호출이 0회로 유지된다 (recording.py 의 다운샘플+delta 로직 재사용).
  3. OCR confirm(변화 시에만): paddleocr 로 분자 N 을 읽어
     N >= min_count 이고 직전 읽기 대비 비감소(연속 2회 확인)면 done.
     연속 2회 확인은 단일 프레임 OCR 오독(분모 등)으로 끊기는 것을 막는다.
     OCR 연속 미검출(숫자 -> blank 전환 = 새 재정렬 시작 가능)은 ROI 재grounding.

실패는 전부 graceful: grounding 거부/OCR 실패/예외 -> False -> watch 의
engineer_watch_sec cap 이 안전망. (CLAUDE.md 규칙: VLM 은 위치만, 전이 판정의
정량 근거는 CV 변화 + OCR 숫자.)

오피스 캘리브레이션 (단독 실행):
  지금 측정 중인 tool 의 Remote Monitoring 창을 열어 두고 실행하면 — 측정
  중에는 분자가 실제로 증가하므로 — grounding/gate/OCR 전 체인을 align fail
  없이 즉시 검증할 수 있다.

  uv run python poc/workflow_3/monitor/engineer_done.py
"""

import os
import re
import time
from dataclasses import replace

from poc.workflow_3.debug_artifacts import save_debug_jpeg
from poc.workflow_3.monitor.recording import _frame_changed, _to_diff_gray
from poc.workflow_3.util import capture_window

_POINT_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_INT_RE = re.compile(r"\d+")


def parse_point_1000(text: str) -> tuple[int, int] | None:
    """ui-venus 응답에서 첫 [x,y](0-1000)를 파싱한다.

    거부([-1,-1])·범위 밖·미검출은 None.
    """
    match = _POINT_RE.search(text or "")
    if not match:
        return None
    x, y = int(match.group(1)), int(match.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


def point_to_roi_ratios(
    x_1000: int, y_1000: int, pad_x: float, pad_y: float
) -> tuple[float, float, float, float] | None:
    """grounding 점(0-1000)을 분자 crop 용 상대비율 ROI (l,t,r,b) 로 확장한다."""
    cx, cy = x_1000 / 1000.0, y_1000 / 1000.0
    left = min(max(cx - pad_x, 0.0), 1.0)
    right = min(max(cx + pad_x, 0.0), 1.0)
    top = min(max(cy - pad_y, 0.0), 1.0)
    bottom = min(max(cy + pad_y, 0.0), 1.0)
    if right - left <= 0.0 or bottom - top <= 0.0:
        return None
    return left, top, right, bottom


def extract_numerator(text: str) -> int | None:
    """OCR 텍스트에서 분자 정수를 뽑는다 ('2/350' -> 2, 첫 연속 숫자열)."""
    match = _INT_RE.search(text or "")
    if match is None:
        return None
    return int(match.group(0))


class EngineerDoneDetector:
    """Recipe Monitor 분자 기반 측정-시작 감지기 (watch iteration 마다 호출).

    capture_fn/ground_fn/ocr_fn 은 테스트 주입점 (RecordingSession 의 capture_fn
    패턴). 실배선은 build_engineer_done_detector 가 담당한다.

      capture_fn() -> PIL.Image          (기본: util.capture_window(tool_window))
      ground_fn(image) -> (x,y) 0-1000 | None   (VLM grounding, 거부 시 None)
      ocr_fn(crop_image) -> str          (분자 crop OCR 텍스트)
    """

    def __init__(
        self,
        tool_window,
        settings,
        *,
        capture_fn=None,
        ground_fn=None,
        ocr_fn=None,
        debug_dir=None,
    ):
        self.tool_window = tool_window
        self.s = settings
        self._capture_fn = capture_fn or (lambda: capture_window(self.tool_window))
        self._ground_fn = ground_fn
        self._ocr_fn = ocr_fn
        self.debug_dir = debug_dir
        self._roi_ratios: tuple[float, float, float, float] | None = None
        self._next_localize_at = 0.0  # 거부(blank) 후 재시도 가능 시각 (throttle).
        self._prev_gray = None
        self._last_n: int | None = None
        self._ocr_miss_streak = 0
        self._debug_seq = 0
        self.last_debug: dict = {}

    def __call__(self) -> bool:
        """측정 시작이 확인되면 True. 모든 실패/미확정은 False (cap 이 안전망)."""
        self.last_debug = {}
        if self._roi_ratios is None:
            now = time.time()
            if now < self._next_localize_at:
                return False
            self._roi_ratios = self._localize()
            if self._roi_ratios is None:
                # 재정렬 중 카운터 blank 면 거부가 정상 — throttle 후 재시도.
                self._next_localize_at = now + max(self.s.engineer_done_reground_sec, 0.0)
                return False

        crop = self._crop_numerator()
        if crop is None:
            return False

        gray = _to_diff_gray(crop)
        first_sample = self._prev_gray is None
        changed = (not first_sample) and _frame_changed(
            self._prev_gray, gray, self.s.engineer_done_change_min_px
        )
        self._prev_gray = gray
        self.last_debug.update({"changed": changed, "first_sample": first_sample})
        if not changed:
            return False

        self._save_debug_crop(crop)
        n = self._read_numerator(crop)
        if n is None:
            self._ocr_miss_streak += 1
            self.last_debug["ocr_miss_streak"] = self._ocr_miss_streak
            if self._ocr_miss_streak >= self.s.engineer_done_relocalize_after_miss:
                print("[INFO] OCR 연속 미검출 - ROI 재grounding 예약(패널 이동/카운터 blank 가능성)")
                self._roi_ratios = None
                self._next_localize_at = 0.0  # 즉시 재grounding 허용.
                self._ocr_miss_streak = 0
                self._prev_gray = None
            return False

        self._ocr_miss_streak = 0
        # 연속 2회 확인: 직전 읽기가 있어야 하고 비감소 + min_count 이상.
        is_done = (
            n >= self.s.engineer_done_min_count
            and self._last_n is not None
            and n >= self._last_n
        )
        self.last_debug["n"] = n
        self._last_n = n
        if is_done:
            print(
                f"[INFO] 측정 카운터 확인: N={n} "
                f"(>= {self.s.engineer_done_min_count}, 연속 2회) - 측정 시작 판정"
            )
        return is_done

    # ---- 내부 ----

    def _localize(self):
        """VLM grounding - 분자 위치를 상대비율 ROI 로. 실패/거부 시 None(재시도 가능)."""
        if self._ground_fn is None:
            print("[WARNING] engineer-done grounding fn 없음 - 감지 비활성(cap 대기)")
            return None
        try:
            image = self._capture_fn()
            point = self._ground_fn(image)
        except Exception as exc:
            print(f"[WARNING] engineer-done grounding 실패(재시도 예정): {exc}")
            return None
        if point is None:
            print(
                "[INFO] engineer-done grounding 거부/미발견 - 카운터 blank(재정렬 중) "
                f"가능성, {self.s.engineer_done_reground_sec:.0f}s 후 재시도"
            )
            return None
        ratios = point_to_roi_ratios(
            point[0], point[1],
            self.s.engineer_done_roi_pad_x, self.s.engineer_done_roi_pad_y,
        )
        if ratios is None:
            print(f"[WARNING] engineer-done ROI 생성 실패: point={point}")
            return None
        print(
            f"[INFO] engineer-done ROI 캐시: point={point}, "
            f"ratios=({ratios[0]:.3f},{ratios[1]:.3f},{ratios[2]:.3f},{ratios[3]:.3f})"
        )
        return ratios

    def _crop_numerator(self):
        """tool 창 재캡처 후 캐시된 상대비율 ROI 로 분자 셀을 crop 한다."""
        try:
            image = self._capture_fn()
        except Exception as exc:
            print(f"[WARNING] engineer-done 캡처 실패(회차 skip): {exc}")
            return None
        left, top, right, bottom = self._roi_ratios
        width, height = image.size
        box = (
            int(left * width),
            int(top * height),
            max(int(left * width) + 1, int(right * width)),
            max(int(top * height) + 1, int(bottom * height)),
        )
        return image.crop(box)

    def _read_numerator(self, crop) -> int | None:
        """분자 crop 을 OCR 해 정수 N 을 얻는다. 실패는 None."""
        if self._ocr_fn is None:
            return None
        try:
            text = self._ocr_fn(crop)
        except Exception as exc:
            print(f"[WARNING] engineer-done OCR 실패(회차 미판정): {exc}")
            return None
        return extract_numerator(text)

    def _save_debug_crop(self, crop) -> None:
        """debug_dir 설정 시 변화-발화 crop 을 저장한다 (실패 무시)."""
        if self.debug_dir is None:
            return
        try:
            self._debug_seq += 1
            save_debug_jpeg(crop, self.debug_dir / f"numerator_{self._debug_seq:03d}.jpg")
        except Exception as exc:
            print(f"[WARNING] engineer-done debug crop 저장 실패: {exc}")


# ------------------------------------------------------------------
# 실배선 builder.
# ------------------------------------------------------------------


def _make_ground_fn(settings, vlm_client=None):
    """grounding closure - ui-venus 단일요소 프롬프트 + [x,y] 파싱."""
    from poc.workflow_3.vlm.prompts.prompt_recipe_monitor_counter import (
        build_recipe_monitor_counter_prompt,
    )
    from poc.workflow_3.util import encode_image_webp

    client = vlm_client
    if client is None or getattr(client, "service_slug", "") != settings.engineer_done_vlm_service:
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        client = Workflow1VLMClient(settings.engineer_done_vlm_service)

    def ground(image):
        system_message, user_text = build_recipe_monitor_counter_prompt()
        image_b64, _, _ = encode_image_webp(image)
        response = client.chat_with_image_b64(
            image_b64=image_b64, system_message=system_message, user_text=user_text
        )
        return parse_point_1000(response.text)

    return ground


def _make_ocr_fn(settings):
    """분자 crop OCR closure - paddleocr `OCR:` 태스크 (tight crop 만, 환각 회피)."""
    from poc.workflow_3.vlm.prompts.prompt_ocr_assist import build_ocr_assist_prompt
    from poc.workflow_3.util import encode_image_webp
    from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

    client = Workflow1VLMClient(settings.engineer_done_ocr_service)

    def ocr(crop):
        system_message, user_text = build_ocr_assist_prompt(crop.size[0], crop.size[1])
        image_b64, _, _ = encode_image_webp(crop)
        response = client.chat_with_image_b64(
            image_b64=image_b64, system_message=system_message, user_text=user_text
        )
        return response.text

    return ocr


def build_engineer_done_detector(tool_window, settings, *, vlm_client=None, debug_dir=None):
    """설정 게이트 확인 후 실 VLM/OCR 배선된 detector 를 만든다.

    비활성/창 없음/클라이언트 생성 실패 -> None (호출부는 고정 timeout 폴백).
    cycle 의 OK-버튼용 vlm_client 가 같은 서비스면 재사용한다.
    """
    if not settings.engineer_done_detect_enabled:
        return None
    if tool_window is None:
        return None
    try:
        ground_fn = _make_ground_fn(settings, vlm_client=vlm_client)
        ocr_fn = _make_ocr_fn(settings)
    except Exception as exc:
        print(f"[WARNING] engineer-done 클라이언트 생성 실패(고정 timeout 폴백): {exc}")
        return None
    return EngineerDoneDetector(
        tool_window, settings, ground_fn=ground_fn, ocr_fn=ocr_fn, debug_dir=debug_dir
    )


# ------------------------------------------------------------------
# 오피스 캘리브레이션 단독 실행.
# ------------------------------------------------------------------

# 비우면 env ALIGN_DONE_CALIB_EQP_ID 폴백 (그것도 비면 아무 Remote Monitoring 창).
EQP_ID_OVERRIDE = ""

_REMOTE_MONITORING_TITLE_PREFIX = "Remote Monitoring System -"


def _tool_label_from_title(title: str) -> str:
    """창 제목에서 tool id 부분을 debug 폴더명용으로 추출/정제한다.

    'Remote Monitoring System - MCD630' -> 'MCD630'. 영숫자/'-'/'_' 외는 '_' 로
    치환해 Windows 경로에 안전하게 만든다.
    """
    text = title or ""
    if text.lower().startswith(_REMOTE_MONITORING_TITLE_PREFIX.lower()):
        text = text[len(_REMOTE_MONITORING_TITLE_PREFIX):]
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_" for ch in text.strip()
    )
    return cleaned.strip("_")


def run_calibration() -> bool:
    """지금 측정 중인 tool 창에 대해 grounding/gate/OCR 전 체인을 즉시 검증한다.

    측정 중에는 분자가 실제로 증가하므로, align fail 을 기다리지 않고 done 감지
    성공까지의 전 경로(위치 grounding 정확성 포함)를 확인할 수 있다.
    Remote Monitoring 창은 미리 열어 둔다 (직접 또는 workflow_select_tool).
    """
    try:
        from poc.workflow_3.rcs.login_rcs_common import find_remote_monitoring_window
    except Exception as exc:
        print(f"[ERROR] RCS 모듈 로드 실패 - 캘리브레이션은 office Windows 전용: {exc}")
        return False

    from poc.workflow_3 import DEBUG_IMAGE_DIR
    from poc.workflow_3.config import load_workflow3_settings

    eqp_id = (EQP_ID_OVERRIDE or "").strip() or os.getenv("ALIGN_DONE_CALIB_EQP_ID", "").strip()
    duration_sec = float(os.getenv("ALIGN_DONE_CALIB_SEC", "120"))

    settings = load_workflow3_settings()
    if not settings.engineer_done_detect_enabled:
        # 캘리브레이션은 게이트를 무시하고 강제 활성(검증이 목적이므로).
        settings = replace(settings, engineer_done_detect_enabled=True)

    window, title, _backend = find_remote_monitoring_window(eqp_id)
    if window is None:
        print(f"[ERROR] Remote Monitoring 창 없음 (eqp_id={eqp_id!r}) - 먼저 tool 을 여세요.")
        return False
    print(f"[INFO] 캘리브레이션 대상 창: {title!r}")

    # run 별 폴더: <tool>_<timestamp> — 여러 tool/회차의 debug crop 이 안 섞이고 보존된다.
    from poc.workflow_3.util import make_timestamp_tag

    tool_label = eqp_id or _tool_label_from_title(title) or "any"
    debug_dir = DEBUG_IMAGE_DIR / "engineer_done_calib" / f"{tool_label}_{make_timestamp_tag()}"
    detector = build_engineer_done_detector(window, settings, debug_dir=debug_dir)
    if detector is None:
        print("[ERROR] detector 생성 실패 - VLM 서비스 설정을 확인하세요.")
        return False

    print(
        f"[INFO] 캘리브레이션 시작: 최대 {duration_sec:.0f}s, "
        f"poll={settings.engineer_done_poll_sec}s, min_count={settings.engineer_done_min_count}, "
        f"debug={debug_dir}"
    )
    deadline = time.time() + duration_sec
    tick = 0
    while time.time() < deadline:
        tick += 1
        done = detector()
        dbg = detector.last_debug
        print(
            f"[INFO] tick {tick}: changed={dbg.get('changed')}, "
            f"n={dbg.get('n')}, miss={dbg.get('ocr_miss_streak', 0)}, done={done}"
        )
        if done:
            print("[INFO] 캘리브레이션 성공: 측정 중 tool 에서 done 감지 체인 검증 완료")
            print("[INFO] 운영 활성화: ALIGN_FAIL_ENGINEER_DONE_DETECT=1")
            return True
        time.sleep(settings.engineer_done_poll_sec)

    print(
        "[WARNING] duration 내 done 미감지 - debug crop 으로 ROI 를 확인하고 "
        "grounding 문구(RECIPE_MONITOR_NUMERATOR_INSTRUCTION)/ROI pad 를 조정하세요."
    )
    return False


if __name__ == "__main__":
    raise SystemExit(0 if run_calibration() else 1)


__all__ = [
    "EngineerDoneDetector",
    "build_engineer_done_detector",
    "extract_numerator",
    "parse_point_1000",
    "point_to_roi_ratios",
]
