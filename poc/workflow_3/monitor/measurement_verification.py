"""Measurement Verification - primary Recovery Verification 의 3상태 decision record.

복구가 실제로 됐는지는 "측정이 정상으로 돌아왔는가" 로만 말할 수 있다. 클릭했다는 기록,
알람이 사라졌다는 사실, `corrected` status 는 전부 그 증거가 아니다. 그래서 이 판정은
따로 record 로 남고, 값은 `success`/`failure`/`unknown` 셋이다.

record 를 채우는 source 는 둘이다.

  * `reader`     - 자동 판독기. **지금은 unknown 만 내는 stub 이다**(아래 참조).
  * `annotation` - 행동 수행자가 프레임을 가리키며 남긴 `verification_reading`.
                   첫 자격 Episode 들에서는 이쪽이 **정식 primary** 이며 열등한 tier 가
                   아니다. 채우는 쪽은 티켓 21 이다.

왜 stub 인가
-----------
현재 Assist CV(`sem_monitor/assist_score.py`)는 패널 전체의 **행 band** 만 센다.
Measurement 열을 Addressing 열과 분리하지 못하므로 "측정이 정상화됐다" 를 말할 수 없다.
열 분리 reader 는 오피스 프레임 없이는 만들 수도, 검증할 수도 없다 - 그것은 오피스
캘리브레이션 gate 이지 집에서 쓸 로직이 아니다. 그래서 여기서는 **판독을 시도조차 하지
않고** 패널 crop 만 근거로 남긴 뒤 `unknown(reader_not_calibrated)` 를 낸다.

캘리브레이션 이후의 reader 도 **같은 record** 를 채운다 - 값과 reason 만 달라지고
스키마·저장 위치·source 규약은 그대로다. 그때 바꿀 것은 `read_measurement_stub` 하나다.
"""

import json
import os
from pathlib import Path

VERIFICATION_FILENAME = "measurement_verification.json"
VERIFICATION_SCHEMA_VERSION = "measurement_verification.v1"

SUCCESS = "success"
FAILURE = "failure"
UNKNOWN = "unknown"
VERIFICATION_VALUES = (SUCCESS, FAILURE, UNKNOWN)

SOURCE_READER = "reader"
SOURCE_ANNOTATION = "annotation"
VERIFICATION_SOURCES = (SOURCE_READER, SOURCE_ANNOTATION)

# 판독기가 아직 열 분리를 못 한다는 사실 자체를 값으로 남긴다(빈 값이나 false 가 아니다).
REASON_NOT_CALIBRATED = "reader_not_calibrated"

# crop 저장 파일명 - attempt 폴더 안이라 Episode-relative 참조가 된다.
CROP_FILENAME = "measurement_panel.jpg"


def verification_record(
    *,
    value: str,
    reason: str,
    source: str,
    baseline_ref: str = "",
    post_action_ref: str = "",
    evidence: str = "",
    **detail,
) -> dict:
    """Measurement Verification decision record 1건.

    `baseline_ref`/`post_action_ref` 는 "무엇에 견주어 무엇을 봤는가" 다. 둘 다
    Episode-relative 이며, 없으면 빈 문자열이다(밖을 가리키는 절대 경로를 적지 않는다).
    """
    if value not in VERIFICATION_VALUES:
        raise ValueError(f"unsupported verification value: {value!r}")
    if source not in VERIFICATION_SOURCES:
        raise ValueError(f"unsupported verification source: {source!r}")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "value": value,
        "reason": reason,
        "source": source,
        "baseline_ref": str(baseline_ref or ""),
        "post_action_ref": str(post_action_ref or ""),
        "evidence": str(evidence or ""),
        "detail": detail,
    }


def read_measurement_stub(
    image,
    *,
    locate_fn,
    save_crop_fn,
    attempt_dir,
    crop_ref_prefix: str = "",
    baseline_ref: str = "",
    post_action_ref: str = "",
) -> dict:
    """자동 판독 stub - **어떤 입력에도 `unknown`** 이고 패널 crop 만 근거로 남긴다.

    `locate_fn(image) -> panel_box | None` 과 `save_crop_fn(crop, path)` 를 주입받아
    실장비/VLM 없이도 시험된다. crop 을 남기지 못하면 값은 그대로 `unknown` 이되
    **사유가 다르다** - "판독기가 아직 없다" 와 "근거조차 못 남겼다" 는 사후에 구분되어야
    한다(후자는 수집이 깨진 것이다).

    이 함수는 crop 의 **픽셀을 보지 않는다**. 열 분리 판독을 집에서 쓰지 않는다는 계약이
    코드로 성립해야 하기 때문이다.
    """
    try:
        panel_box = locate_fn(image)
    except Exception as exc:
        return verification_record(
            value=UNKNOWN, reason=f"crop_failed:locate:{type(exc).__name__}: {exc}",
            source=SOURCE_READER, baseline_ref=baseline_ref,
            post_action_ref=post_action_ref,
        )
    if not panel_box:
        return verification_record(
            value=UNKNOWN, reason="crop_failed:panel_not_located",
            source=SOURCE_READER, baseline_ref=baseline_ref,
            post_action_ref=post_action_ref,
        )
    try:
        crop = image.crop((
            int(panel_box["left"]), int(panel_box["top"]),
            int(panel_box["right"]), int(panel_box["bottom"]),
        ))
        out_path = Path(attempt_dir) / CROP_FILENAME
        save_crop_fn(crop, out_path)
    except Exception as exc:
        return verification_record(
            value=UNKNOWN, reason=f"crop_failed:save:{type(exc).__name__}: {exc}",
            source=SOURCE_READER, baseline_ref=baseline_ref,
            post_action_ref=post_action_ref, panel_box=panel_box,
        )
    prefix = f"{crop_ref_prefix.rstrip('/')}/" if crop_ref_prefix else ""
    return verification_record(
        value=UNKNOWN, reason=REASON_NOT_CALIBRATED, source=SOURCE_READER,
        baseline_ref=baseline_ref, post_action_ref=post_action_ref,
        evidence=f"{prefix}{CROP_FILENAME}", panel_box=panel_box,
    )


def write_verification_record(attempt_dir, record: dict) -> Path:
    """attempt 폴더에 `measurement_verification.json` 을 원자적으로 쓴다."""
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    path = attempt_dir / VERIFICATION_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_verification_record(path) -> dict:
    """record 를 읽고 값/source 가 규약 안인지 확인한다(어긋나면 ValueError)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("value") not in VERIFICATION_VALUES:
        raise ValueError(f"unsupported verification value: {data.get('value')!r}")
    if data.get("source") not in VERIFICATION_SOURCES:
        raise ValueError(f"unsupported verification source: {data.get('source')!r}")
    return data


__all__ = [
    "CROP_FILENAME",
    "FAILURE",
    "REASON_NOT_CALIBRATED",
    "SOURCE_ANNOTATION",
    "SOURCE_READER",
    "SUCCESS",
    "UNKNOWN",
    "VERIFICATION_FILENAME",
    "VERIFICATION_SCHEMA_VERSION",
    "VERIFICATION_SOURCES",
    "VERIFICATION_VALUES",
    "load_verification_record",
    "read_measurement_stub",
    "verification_record",
    "write_verification_record",
]
