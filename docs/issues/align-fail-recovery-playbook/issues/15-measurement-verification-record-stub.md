# 15 — Measurement Verification record and unknown-only reader stub

Type: task
Status: resolved
Blocked by: 11
Spec: [spec.md](../spec.md) (Structured Verification and Outcome)

## What to build

primary Verification 인 Measurement 판독을 3상태 decision record 로 정의한다: `success`/`failure`/
`unknown`, reason, baseline 참조, post-Action frame 참조, source ∈ {`reader`, `annotation`}. record 는
attempt 폴더에 저장된다.

자동 reader 는 **`unknown`-only stub** 으로 출하한다. 기존 Assist panel locator/crop 을 재사용해 crop 을
evidence 로 남기고 항상 `unknown(reason=reader_not_calibrated)` 을 낸다. 현재 Assist CV 는 패널 전체의
행 band 만 세고 Measurement 열을 Addressing 열과 분리하지 못하므로, 열 분리 reader 는 오피스 프레임
없이 지을 수 없다 - 그것은 오피스 캘리브레이션 gate 이지 이 티켓이 아니다. `annotation` source 를
채우는 쪽은 21 번이다.

## Acceptance criteria

- [x] 세 값이 구분되는 record 가 직렬화/역직렬화되고 source 필드가 reader/annotation 을 가른다(합성 record 로 검증).
- [x] stub 은 어떤 입력에도 `unknown` 이고 crop 을 attempt 폴더에 남긴다; crop 실패도 `unknown` 이되 reason 이 다르다.
- [x] 열 분리 판독 로직이 집에서 작성되지 않았다(향후 reader 는 같은 record 를 채운다는 주석/문서만).
- [x] spec 테스트 7(reader 부분)을 덮는다.

## Comments

구현 완료 (2026-08-30).

**무엇을 어디에**
- 새 모듈 `monitor/measurement_verification.py` - `verification_record()`(3상태 +
  reason + baseline/post-action 참조 + source), `read_measurement_stub()`(unknown-only
  판독기), `write_verification_record()` / `load_verification_record()`(값·source 검증).
  값/source 밖의 문자열은 만들 수도 읽을 수도 없다(ValueError).
- `monitor/cycle.py` - `_locate_measurement_panel()` 이 기존 `locate_assist_panel` 을
  **그대로** 쓰고, `write_attempt_verification()` 이 stub 을 돌려 record + crop 을
  attempt 폴더에 남긴다. 호출은 Guard 기록 바로 뒤, **teardown 앞**이다(창이 닫히면
  패널 crop 을 못 남긴다). 화면 캡처가 실패해도 record 는 남긴다 - "판독을 못 했다"
  자체가 관측이다.
- `monitor/recovery_episode.py` - `_ATTEMPT_RECORD_FILES` 에 한 줄 추가해 Episode 의
  attempt artifacts 가 이 record 를 Episode-relative 로 가리킨다.

**stub 의 계약** - 픽셀을 **보지 않는다**. locate -> crop -> 저장만 하고 값은 항상
`unknown` 이다. 사유는 두 갈래로 갈린다: `reader_not_calibrated`(근거는 남겼고 판독기가
아직 없다) vs `crop_failed:*`(근거조차 못 남겼다 = 수집이 깨졌다). 테스트가 "정반대 화면
두 장이 같은 판정" 을 assert 해 열 분리 로직이 집에서 작성되지 않았음을 코드로 고정한다.
캘리브레이션 후의 reader 는 **같은 record** 를 채운다 - 스키마/저장 위치/source 규약은
그대로이고 바뀌는 것은 `read_measurement_stub` 하나다(모듈 docstring 에 명시).

**테스트** `monitor/test_measurement_verification.py` 7개. spec 테스트 7(reader 부분) 커버.
회귀: monitor 543 / recording_filter+workflow_extract+workflow_4 269 통과.

**판단**
- stub 호출은 attempt 당 VLM grounding 1회를 쓴다. 값을 얻으려는 것이 아니라 **crop 을
  근거로 남기려는** 호출이고 spec 이 그 crop 보존을 명시하므로 수용했다. 수집 플래그
  뒤에 있어 기본 운전에는 영향이 없다.
- `annotation` source 를 채우는 쪽은 티켓 21 이다. 여기서는 record 가 그 source 를
  1급으로 받아들인다는 것만 스키마와 테스트로 고정했다(열등한 tier 가 아니다).
