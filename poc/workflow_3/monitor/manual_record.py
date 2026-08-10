"""엔지니어 수동 조작 녹화 런처 - 알람 없이 이미 열린 tool 창을 녹화한다.

알람 사이클(`monitor/cycle.py`)의 녹화는 align fail 이 떠야만 시작된다. 이 모듈은
엔지니어와 "지금부터 녹화하겠다"고 약속한 뒤, 이미 열려 있는 Remote Monitoring 창을
그 자리에서 녹화하기 위한 독립 진입점이다. 접속(tool 더블클릭)은 하지 않는다.

수집한 프레임은 모방 학습/절차 분석의 원천 데이터가 되며, 분석은 별도 실행이다
(`recording_filter/filter_recording.py`).

실행:
    uv run python poc/workflow_3/monitor/manual_record.py
"""

import re

from poc.workflow_3 import ALIGN_IMAGES_DIR
from poc.workflow_3.rcs.login_rcs_common import REMOTE_MONITORING_WINDOW_TITLE_PREFIX

# 폴더명으로 쓸 수 없는 문자(Windows 예약 문자 + 공백/괄호)를 밑줄로 바꾼다.
# \w 는 유니코드 단어 문자(한글 포함)까지 허용한다 - ASCII 로 한정하면 "장비1" 같은
# 한글 EQP 명이 전부 "1" 처럼 깎여나가 서로 다른 장비가 같은 폴더로 충돌한다
# (2026-08-10 코디네이터 리뷰 FINDING 2). "." 는 여기서 허용 문자라 정규식만으로는
# ".."(부모 디렉터리 이동) 를 걸러내지 못한다 - 그건 sanitize_eqp_for_path 의
# 후처리(양끝 "._- " 트림 + 빈 결과 폴백)가 담당한다 (FINDING 1).
_PATH_HOSTILE_RE = re.compile(r"[^\w.-]+", re.UNICODE)
# EQP 를 못 읽었을 때의 대체 폴더명 - 프레임을 잃는 것보다 낫다.
UNKNOWN_EQP = "unknown_eqp"
# 수동 세션 전용 하위 폴더명 (알람 캡처의 captured_img_from_rcs 와 구분).
MANUAL_DIRNAME = "_manual"


def parse_eqp_from_title(title) -> str:
    """창 제목에서 EQP 문자열을 추출한다(접두어 제거). 실패하면 빈 문자열.

    제목은 "Remote Monitoring System - <EQP>" 형태다. 접두어 매칭은 대소문자를
    무시하고, EQP 뒤에 부가 정보가 붙어 있으면 통째로 보존한다(폴더명 정규화는
    sanitize_eqp_for_path 의 몫이라 여기서는 자르지 않는다).
    """
    normalized = (title or "").strip()
    prefix = REMOTE_MONITORING_WINDOW_TITLE_PREFIX
    if len(normalized) < len(prefix):
        return ""
    if normalized[: len(prefix)].lower() != prefix.lower():
        return ""
    return normalized[len(prefix):].strip()


def sanitize_eqp_for_path(eqp) -> str:
    """EQP 문자열을 폴더명으로 안전한 형태로 바꾼다. 비면 UNKNOWN_EQP.

    양끝의 "." / "-" / "_" / 공백은 잘라낸다 - Windows 는 이름 끝의 "." 를
    잘못 처리하고("MCD916." 같은 폴더), 입력이 온통 "."/".."로만 되어 있으면
    (".", "..", "...") 트림 후 빈 문자열이 되어 자동으로 UNKNOWN_EQP 로 폴백한다.
    이 폴백이 없으면 manual_recording_dir 가 ALIGN_IMAGES_DIR / ".." 를 만들어
    의도한 루트 밖에 쓰게 된다.
    """
    cleaned = _PATH_HOSTILE_RE.sub("_", (eqp or "").strip())
    cleaned = cleaned.strip("._- \t")
    return cleaned or UNKNOWN_EQP


def manual_recording_dir(eqp_id, tag):
    """수동 세션 프레임 저장 폴더 - <root>/<eqp>/_manual/<tag>/recording."""
    return ALIGN_IMAGES_DIR / sanitize_eqp_for_path(eqp_id) / MANUAL_DIRNAME / str(tag) / "recording"
