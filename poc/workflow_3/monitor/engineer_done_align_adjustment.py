"""엔지니어 수동 align 보정 완료 감지 — Assist 우선, 측정 카운터 fallback.

**align fail 관리 전용** 모듈이다. 미보정 engineer watch 동안 "align 보정이 끝나고
본 측정이 정상적으로 진행 중이다"를 감지해 녹화를 조기 종료하고, 그 결과 cycle
teardown 이 tool 창을 자동으로 닫는다. done 판정 우선순위는 다음과 같다:

  1. Assist primary: watch 시작 뒤 Measurement 패널 fingerprint 가 새로 바뀌고,
     Recipe Monitor Assist Window 가 연속 정상(검정) 측정을
     `engineer_done_ok_streak`(기본 6) 회 이상 보여주면 완료다.
  2. 분자 fallback: Assist 가 연속 3회 unusable 이고 이 watch 에서 Assist fail 을
     한 번도 보지 않았을 때만, 분자 N 이 3회 엄격히 증가하면 완료다.

Assist 는 분자보다 먼저 같은 캡처 프레임에서 평가한다. 분자는 Assist 판독 자체가
불가능할 때만 쓰는 보수적 fallback 이며, usable Assist 에서 한 번이라도 fail 을
보면 해당 watch 동안 fallback 을 영구 차단한다.

done 판정 시 `cycle._engineer_watch` 가 조기 종료하고, `run_alarm_cycle` 의
teardown(finally)이 `close_tool(eqp_id)` 로 tool 창을 닫는다 (별도 닫기 호출
없이 기존 teardown 경로 재사용 — close 동작은 workflow_2 에서 확립됨).

파이프라인 (grounding 은 성공 시 1회 캐시):
  1. grounding(성공 시 캐시): VLM(mai-ui)으로 분자 위치를 찾아 tool-window
     상대비율 ROI 로 캐시한다 (tool 마다/드래그로 위치가 달라 고정 ROI 불가).
     **오피스 관찰(2026-06-11): re-align 진행 중에는 카운터(N/M)가 빈칸**이라
     grounding 거부([-1,-1])가 정상 상태다. 따라서 거부는 영구 포기가 아니라
     `reground_sec` 간격 재시도 — 측정이 시작되면 숫자가 나타나 성공한다.
  2. CV gate(매 호출): ROI crop 변화감지 — align-fix 중엔 카운터가 정적이라
     OCR 호출이 0회로 유지된다 (recording.py 의 다운샘플+delta 로직 재사용).
  3. OCR confirm(변화 시에만): paddleocr 로 분자 N 을 읽고 엄격히 증가하는 최근
     3개 표본만 유지한다. 같은 값/감소는 새 시퀀스를 시작하고 OCR miss 는 지운다.
     OCR 연속 미검출(숫자 -> blank 전환 = 새 재정렬 시작 가능)은 ROI 재grounding.
  4. Assist primary(매 호출, 분자보다 먼저): `assist_score.locate_assist_layout` 으로
     score 격자를 1회 잡아 캐시하고, 이후 `read_row_states` + `ok_streak` 로
     연속 정상 횟수를 구한다. 격자 로케이트 자체도 재시도가 실패할 때마다
     `reground_sec` 로 throttle 한다(안 그러면 VLM+OCR 왕복이 매 폴링마다 반복돼
     watch 루프가 막힌다).

실패는 전부 graceful: grounding 거부/OCR 실패/예외 -> False -> watch 의
engineer_watch_sec cap 이 안전망. (CLAUDE.md 규칙: VLM 은 위치만, 전이 판정의
정량 근거는 CV 변화 + OCR 숫자 + Assist 색.)

오피스 캘리브레이션 (단독 실행):
  지금 측정 중인 tool 의 Remote Monitoring 창을 열어 두고 실행하면 — 측정
  중에는 분자가 실제로 증가하므로 — grounding/gate/OCR/Assist 전 체인을 align
  fail 없이 즉시 검증할 수 있다.

  uv run python poc/workflow_3/monitor/engineer_done_align_adjustment.py
"""

import os
import re
import time
from dataclasses import dataclass, replace

from poc.workflow_3.config import validate_engineer_done_priority_settings
from poc.workflow_3.debug_artifacts import save_debug_jpeg, save_debug_json
from poc.workflow_3.monitor.recording import _frame_changed, _to_diff_gray
from poc.workflow_3.sem_monitor.assist_score import AssistObservation, ok_streak
from poc.workflow_3.util import capture_window

_POINT_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_INT_RE = re.compile(r"\d+")

# Assist 전 행이 연속으로 이만큼 빈칸이면 패널이 이동한 것으로 보고 격자를 다시 잡는다.
ALL_BLANK_RELOCATE_AFTER = 3

# 전 행 빈칸으로 격자를 다시 잡는 횟수 상한. 표가 원래 비어 있는 tool 에서 watch 내내
# VLM 을 반복 호출하지 않게 한다.
MAX_BLANK_RELOCATES = 2


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


@dataclass(frozen=True)
class NumeratorObservation:
    """Recipe Monitor 카운터 N 을 한 회차 관측한 결과.

    sampled=False 는 이번 회차에 읽기를 시도하지 않았다는 뜻이고(throttle/우선순위상
    Assist 로 충분했던 경우), sampled=True 인데 value 가 None 이면 읽으려 했으나
    실패한 것이다 - 이 둘을 섞으면 "못 읽었다" 와 "안 읽었다" 가 구분되지 않아
    fallback 판정이 무너진다. reason 은 그 사유, reset_reason 은 이 관측이
    누적 상태를 되돌리게 만든 사유다.
    """

    sampled: bool
    value: int | None = None
    reason: str = ""
    reset_reason: str = ""


class EngineerDoneDetector:
    """Assist 우선 측정-시작 감지기 (watch iteration 마다 호출).

    capture_fn/ground_fn/ocr_fn/assist_fn/numerator_fn 은 테스트 주입점
    (RecordingSession 의 capture_fn 패턴). 실배선은 build_engineer_done_detector 가
    담당한다.

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
        assist_fn=None,
        numerator_fn=None,
        debug_dir=None,
    ):
        self.tool_window = tool_window
        self.s = settings
        self._capture_fn = capture_fn or (lambda: capture_window(self.tool_window))
        self._ground_fn = ground_fn
        self._ocr_fn = ocr_fn
        self._assist_fn = assist_fn
        self._numerator_fn = numerator_fn
        self.debug_dir = debug_dir
        self._roi_ratios: tuple[float, float, float, float] | None = None
        self._next_localize_at = 0.0  # 거부(blank) 후 재시도 가능 시각 (throttle).
        self._prev_gray = None
        self._ocr_miss_streak = 0
        self._debug_seq = 0
        self._assist_unusable_streak = 0
        self._assist_baseline_fingerprint = None
        self._assist_changed_since_start = False
        self._assist_failure_seen = False
        self._numerator_sequence: list[int] = []
        self._numerator_decision_seq = 0
        self.last_debug: dict = {}
        try:
            validate_engineer_done_priority_settings(settings)
        except ValueError as exc:
            self._configuration_error = str(exc)
            print(f"[ERROR] engineer-done 설정 오류 - detector fail-closed: {exc}")
        else:
            self._configuration_error = ""

    def __call__(self) -> bool:
        """측정 시작이 확인되면 True. 모든 실패/미확정은 False (cap 이 안전망)."""
        self.last_debug = {}
        if self._configuration_error:
            self.last_debug["configuration_error"] = self._configuration_error
            return False
        try:
            image = self._capture_fn()
        except Exception as exc:
            print(f"[WARNING] engineer-done 캡처 실패(회차 skip): {exc}")
            return False

        assist = (
            self._assist_fn(image)
            if self._assist_fn is not None
            else AssistObservation(status="unusable", reason="assist_fn_missing")
        )
        if assist.status == "usable":
            # fail 은 '판독 가능한' 관측에서만 인정한다. unusable(예: Measurement 열이
            # 전부 blank) 회차의 verdict 는 판독 실패의 부산물이라, 그것까지 세면
            # Addressing1 이 붉은 프레임 한 장만으로 numerator fallback 이 영구히
            # 닫힌다(design 3절 "실제로 관측하면").
            if "fail" in [row.verdict for row in assist.rows]:
                self._assist_failure_seen = True
            self._assist_unusable_streak = 0
            if self._assist_baseline_fingerprint is None:
                self._assist_baseline_fingerprint = assist.panel_fingerprint
            elif assist.panel_fingerprint != self._assist_baseline_fingerprint:
                self._assist_changed_since_start = True
            streak = ok_streak(assist.rows)
            self.last_debug.update({
                "assist_status": assist.status,
                "assist_changed": self._assist_changed_since_start,
                "assist_failure_seen": self._assist_failure_seen,
                "streak": streak,
            })
            if (
                self._assist_changed_since_start
                and streak >= self.s.engineer_done_ok_streak
            ):
                print(
                    f"[INFO] Assist 새 측정 진행 + 연속 정상 {streak}회 "
                    f"(>= {self.s.engineer_done_ok_streak}) - align 완료 판정, "
                    "watch 조기 종료 후 tool 창 닫기 진행"
                )
                return True
        else:
            self._assist_unusable_streak += 1
            self.last_debug.update({
                "assist_status": assist.status,
                "assist_unusable_streak": self._assist_unusable_streak,
                "assist_failure_seen": self._assist_failure_seen,
            })

        numerator = (
            self._numerator_fn(image)
            if self._numerator_fn is not None
            else self._observe_numerator(image)
        )
        reset_reason = self._update_numerator_sequence(numerator)
        self.last_debug.update({
            "numerator_sampled": numerator.sampled,
            "n": numerator.value,
            "numerator_reason": numerator.reason,
            "numerator_sequence": list(self._numerator_sequence),
            "numerator_reset_reason": reset_reason,
        })
        fallback_open = (
            self._assist_unusable_streak
            >= self.s.engineer_done_assist_unusable_after
            and not self._assist_failure_seen
        )
        done = (
            fallback_open
            and len(self._numerator_sequence)
            >= self.s.engineer_done_numerator_increase_reads
        )
        self._save_numerator_decision(
            numerator,
            reset_reason=reset_reason,
            fallback_open=fallback_open,
            done=done,
        )
        return done

    # ---- 내부 ----

    def _localize(self, image):
        """VLM grounding - 분자 위치를 상대비율 ROI 로. 실패/거부 시 None(재시도 가능)."""
        if self._ground_fn is None:
            print("[WARNING] engineer-done grounding fn 없음 - 감지 비활성(cap 대기)")
            return None
        try:
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

    def _crop_numerator(self, image):
        """이미 캡처한 tool 창에서 캐시된 상대비율 ROI 로 분자 셀을 crop 한다."""
        left, top, right, bottom = self._roi_ratios
        width, height = image.size
        box = (
            int(left * width),
            int(top * height),
            max(int(left * width) + 1, int(right * width)),
            max(int(top * height) + 1, int(bottom * height)),
        )
        return image.crop(box)

    def _observe_numerator(self, image) -> NumeratorObservation:
        """같은 poll 프레임에서 분자 변화/OCR 표본을 만든다."""
        regrounded = False
        if self._roi_ratios is None:
            now = time.time()
            if now < self._next_localize_at:
                return NumeratorObservation(False, reason="localize_throttled")
            self._roi_ratios = self._localize(image)
            if self._roi_ratios is None:
                self._next_localize_at = now + max(
                    self.s.engineer_done_reground_sec, 0.0
                )
                return NumeratorObservation(False, reason="roi_unavailable")
            regrounded = True

        crop = self._crop_numerator(image)
        gray = _to_diff_gray(crop)
        first_sample = self._prev_gray is None
        changed = (not first_sample) and _frame_changed(
            self._prev_gray, gray, self.s.engineer_done_change_min_px
        )
        self._prev_gray = gray
        self.last_debug.update({"changed": changed, "first_sample": first_sample})
        if not changed:
            return NumeratorObservation(
                False,
                reason="no_change",
                reset_reason="reground" if regrounded else "",
            )

        self._save_debug_crop(crop)
        n = self._read_numerator(crop)
        if n is None:
            self._ocr_miss_streak += 1
            self.last_debug["ocr_miss_streak"] = self._ocr_miss_streak
            if self._ocr_miss_streak >= self.s.engineer_done_relocalize_after_miss:
                print("[INFO] OCR 연속 미검출 - ROI 재grounding 예약(패널 이동/카운터 blank 가능성)")
                self._roi_ratios = None
                self._next_localize_at = 0.0
                self._ocr_miss_streak = 0
                self._prev_gray = None
                return NumeratorObservation(
                    True,
                    None,
                    "ocr_miss",
                    reset_reason="reground",
                )
            return NumeratorObservation(True, None, "ocr_miss")

        self._ocr_miss_streak = 0
        return NumeratorObservation(True, n, "read")

    def _update_numerator_sequence(
        self,
        observation: NumeratorObservation,
    ) -> str | None:
        if observation.reset_reason:
            self._numerator_sequence.clear()
            reset_reason = observation.reset_reason
        else:
            reset_reason = None
        if not observation.sampled:
            return reset_reason
        n = observation.value
        if n is None:
            self._numerator_sequence.clear()
            return reset_reason or "ocr_miss"
        if self._numerator_sequence and n > self._numerator_sequence[-1]:
            self._numerator_sequence.append(n)
        elif self._numerator_sequence:
            self._numerator_sequence = [n]
            reset_reason = "equal_or_decrease"
        else:
            self._numerator_sequence = [n]
        keep = max(1, self.s.engineer_done_numerator_increase_reads)
        self._numerator_sequence = self._numerator_sequence[-keep:]
        return reset_reason

    def _save_numerator_decision(
        self,
        observation: NumeratorObservation,
        *,
        reset_reason: str | None,
        fallback_open: bool,
        done: bool,
    ) -> None:
        """평가한 numerator 표본과 sequence 결정을 run 폴더에 poll별로 저장한다."""
        if self.debug_dir is None:
            return
        try:
            self._numerator_decision_seq += 1
            save_debug_json(
                self.debug_dir
                / f"numerator_decision_{self._numerator_decision_seq:03d}.json",
                {
                    "poll": self._numerator_decision_seq,
                    "reading": observation.reason,
                    "sampled": observation.sampled,
                    "value": observation.value,
                    "sequence": list(self._numerator_sequence),
                    "reset_reason": reset_reason,
                    "assist_unusable_streak": self._assist_unusable_streak,
                    "assist_failure_seen": self._assist_failure_seen,
                    "fallback_open": fallback_open,
                    "done": done,
                },
            )
        except Exception as exc:
            print(f"[WARNING] numerator decision 저장 실패: {exc}")

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


def _make_assist_fn(tool_window, settings, *, debug_dir=None):
    """캡처 프레임 하나에서 Assist 관측값을 만드는 클로저.

    로케이트에 실패하면 None 을 캐시하지 않고 재시도하되, 실패 로그는 1회만 낸다
    (watch 내내 같은 경고가 반복되면 콘솔이 쓸모없어진다). 재시도 자체도
    `settings.engineer_done_reground_sec` 로 throttle 한다 - 카운터 grounding
    (`EngineerDoneDetector._localize`)이 이미 같은 간격으로 재시도를 절제하는 것과
    같은 이유다: 로케이트 실패마다 2단계 VLM(15s timeout) + PaddleOCR(30s timeout)
    왕복이 매 결정 폴링마다 반복되면 watch 루프 전체가 그만큼 막힌다.
    """
    from poc.workflow_3.sem_monitor.assist_score import (
        AssistObservation,
        locate_assist_layout,
        measurement_fingerprint,
        read_row_states,
        save_assist_overlay,
    )
    from poc.workflow_3.util import crop_image

    state = {"panel_box": None, "layout": None, "warned": False, "last_verdicts": None,
             "seq": 0, "all_blank_streak": 0, "blank_relocates": 0,
             "blank_relocate_limit_logged": False, "next_locate_at": 0.0}

    def assist_fn(image):
        # detector 밖에서 직접 호출돼도 이 클로저 자체가 안전해야 한다.
        try:
            if state["layout"] is None and time.time() < state["next_locate_at"]:
                return AssistObservation(status="unusable", reason="locate_throttled")

            if state["layout"] is None:
                # window_title/backend 는 빈 문자열로 넘긴다. image 를 함께 주면
                # analyze_window_target 이 창 활성화/재캡처를 건너뛰므로 쓰이지 않는다.
                located = locate_assist_layout(tool_window, "", "", image, debug_dir=debug_dir)
                if located is None:
                    state["next_locate_at"] = time.time() + max(
                        settings.engineer_done_reground_sec, 0.0
                    )
                    if not state["warned"]:
                        print("[WARNING] Assist 격자 확보 실패 - 분자 fallback 판정 대기")
                        state["warned"] = True
                    return AssistObservation(status="unusable", reason="layout_unavailable")
                state["panel_box"], state["layout"] = located
                state["all_blank_streak"] = 0

            panel = crop_image(image, state["panel_box"])
            rows = read_row_states(panel, state["layout"])
            if not rows:
                return AssistObservation(status="unusable", reason="rows_empty")
            fingerprint = measurement_fingerprint(panel, state["layout"])

            # 패널이 이동/리사이즈되면 빈 영역을 샘플링해 모든 행이 pending 으로 나온다.
            # 실제로 전 행이 비는 일은 거의 없으므로, 연속으로 그러면 격자를 버리고 다시 잡는다.
            # 단, 표가 원래 비어 있는 tool 이면 이 재확보가 watch 내내 반복돼 VLM 을
            # 계속 호출하므로 MAX_BLANK_RELOCATES 회로 상한을 둔다.
            if rows and all(row.verdict == "pending" for row in rows):
                state["all_blank_streak"] += 1
                if state["all_blank_streak"] >= ALL_BLANK_RELOCATE_AFTER:
                    if state["blank_relocates"] < MAX_BLANK_RELOCATES:
                        state["blank_relocates"] += 1
                        print(
                            "[INFO] Assist 전 행이 계속 빈칸 - 패널 이동 가능성, 격자 재확보 "
                            f"({state['blank_relocates']}/{MAX_BLANK_RELOCATES})"
                        )
                        state["layout"] = None
                        state["panel_box"] = None
                        state["all_blank_streak"] = 0
                    else:
                        if not state["blank_relocate_limit_logged"]:
                            print(
                                f"[INFO] Assist 재확보 상한({MAX_BLANK_RELOCATES}) 도달 - 표가 원래 "
                                "비어 있는 tool 로 보고 더는 재확보하지 않음"
                            )
                            state["blank_relocate_limit_logged"] = True
                    state["all_blank_streak"] = 0
            else:
                state["all_blank_streak"] = 0

            verdicts = [row.verdict for row in rows]
            if debug_dir is not None and verdicts != state["last_verdicts"]:
                state["seq"] += 1
                save_assist_overlay(
                    panel, state["layout"], rows,
                    debug_dir / f"assist_grid_{state['seq']:03d}.jpg",
                )
                state["last_verdicts"] = verdicts
            measurement_states = [
                row.cells.get("Measurement", "blank") for row in rows
            ]
            if all(cell in {"blank", "unknown"} for cell in measurement_states):
                return AssistObservation(
                    status="unusable",
                    rows=rows,
                    panel_fingerprint=fingerprint,
                    reason="measurement_unreadable",
                )
            return AssistObservation(
                status="usable",
                rows=rows,
                panel_fingerprint=fingerprint,
                reason="ok",
            )
        except Exception as exc:
            print(f"[WARNING] Assist 행 판독 클로저 예외(이번 회차 미판정): {exc}")
            return AssistObservation(status="unusable", reason="exception")

    return assist_fn


def build_engineer_done_detector(tool_window, settings, *, vlm_client=None, debug_dir=None):
    """설정 게이트 확인 후 실 VLM/OCR 배선된 detector 를 만든다.

    비활성/창 없음 -> None (호출부는 고정 timeout 폴백). 분자 fallback 클라이언트
    생성이 실패해도 Assist primary 는 유지한다. cycle 의 OK-버튼용 vlm_client 가
    같은 서비스면 재사용한다.
    """
    if not settings.engineer_done_detect_enabled:
        return None
    if tool_window is None:
        return None
    assist_fn = _make_assist_fn(tool_window, settings, debug_dir=debug_dir)
    try:
        ground_fn = _make_ground_fn(settings, vlm_client=vlm_client)
        ocr_fn = _make_ocr_fn(settings)
    except Exception as exc:
        print(f"[WARNING] numerator fallback client 생성 실패(Assist primary 유지): {exc}")
        ground_fn = None
        ocr_fn = None

    return EngineerDoneDetector(
        tool_window,
        settings,
        ground_fn=ground_fn,
        ocr_fn=ocr_fn,
        assist_fn=assist_fn,
        debug_dir=debug_dir,
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
        f"poll={settings.engineer_done_poll_sec}s, "
        f"ok_streak={settings.engineer_done_ok_streak}, "
        f"assist_unusable_after={settings.engineer_done_assist_unusable_after}, "
        f"numerator_reads={settings.engineer_done_numerator_increase_reads}, "
        f"debug={debug_dir}"
    )
    deadline = time.time() + duration_sec
    tick = 0
    while time.time() < deadline:
        tick += 1
        done = detector()
        dbg = detector.last_debug
        print(
            f"[INFO] tick {tick}: assist={dbg.get('assist_status')}, "
            f"fresh={dbg.get('assist_changed')}, streak={dbg.get('streak')}, "
            f"changed={dbg.get('changed')}, n={dbg.get('n')}, "
            f"sequence={dbg.get('numerator_sequence')}, "
            f"miss={dbg.get('ocr_miss_streak', 0)}, done={done}"
        )
        if done:
            print("[INFO] 캘리브레이션 성공: 측정 중 tool 에서 done 감지 체인 검증 완료")
            print("[INFO] 운영 활성화: ALIGN_FAIL_ENGINEER_DONE_DETECT=1")
            return True
        time.sleep(settings.engineer_done_poll_sec)

    print(
        "[WARNING] duration 내 done 미감지. 원인 구분:\n"
        f"  - Assist 는 watch 시작 뒤 fingerprint 변화와 ok_streak="
        f"{settings.engineer_done_ok_streak} 충족이 모두 필요하다. Assist fail 을 한 번이라도 "
        "보면 분자 fallback 은 이 watch 동안 차단된다.\n"
        f"  - Assist 가 연속 {settings.engineer_done_assist_unusable_after}회 unusable 일 때만 "
        f"분자 {settings.engineer_done_numerator_increase_reads}회 엄격 증가 fallback 을 쓴다.\n"
        "  - Assist 격자/색 문제는 debug_images/engineer_done_calib/.../assist_*.jpg "
        "오버레이로 열 매핑/색 임계를 확인한다.\n"
        "  - n 이 계속 None/blank 면 카운터 grounding/OCR 문제 - debug crop 으로 ROI 확인 후 "
        "grounding 문구(RECIPE_MONITOR_NUMERATOR_INSTRUCTION)/ROI pad 조정."
    )
    return False


if __name__ == "__main__":
    raise SystemExit(0 if run_calibration() else 1)


__all__ = [
    "EngineerDoneDetector",
    "NumeratorObservation",
    "build_engineer_done_detector",
    "extract_numerator",
    "parse_point_1000",
    "point_to_roi_ratios",
]
