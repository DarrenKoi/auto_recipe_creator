# 15 — Measurement Verification record and unknown-only reader stub

Type: task
Status: ready-for-agent
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

- [ ] 세 값이 구분되는 record 가 직렬화/역직렬화되고 source 필드가 reader/annotation 을 가른다(합성 record 로 검증).
- [ ] stub 은 어떤 입력에도 `unknown` 이고 crop 을 attempt 폴더에 남긴다; crop 실패도 `unknown` 이되 reason 이 다르다.
- [ ] 열 분리 판독 로직이 집에서 작성되지 않았다(향후 reader 는 같은 record 를 채운다는 주석/문서만).
- [ ] spec 테스트 7(reader 부분)을 덮는다.
