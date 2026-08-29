# 16 — Numerator decision records into the attempt directory

Type: task
Status: resolved
Blocked by: 11
Spec: [spec.md](../spec.md) (Structured Verification and Outcome)

## What to build

engineer-done detector 는 Recipe Monitor 분자(N/M)의 per-read 판독(값, 시각, 읽힘/미읽힘/relocalization,
단조 판정)을 지금 debug 폴더에만 남긴다. Episode 수집이 켜져 있으면 같은 판독을 attempt 폴더에
JSONL 로 쓴다. fallback Verification 이 이 기록을 읽기 때문이다. detector 의 boolean 반환은 녹화 조기
종료용으로만 남고 Verification 입력이 아니다 - `false` 와 `unknown` 을 구분하지 못하기 때문이다.

## Acceptance criteria

- [x] 알람 사이클의 engineer watch 가 attempt 폴더에 numerator 기록을 남긴다(수집 off 면 종전 debug 폴더만).
- [x] 기록에서 OCR miss / 같음·감소 / reground reset / strictly increasing 이 구분된다.
- [x] detector 반환 계약과 기존 detector 테스트는 무변경이다.
- [x] spec 테스트 8(기록 부분)을 덮는다.

## Comments

구현 완료 (2026-08-30).

**무엇을 어디에**
- `monitor/engineer_done_align_adjustment.py`
  - `classify_numerator_decision()` (순수 함수) + 닫힌 판정 집합 `NUMERATOR_DECISIONS`
    = `not_sampled` / `ocr_miss` / `equal_or_decrease` / `reground_reset` /
    `first_sample` / `strictly_increasing`.
  - `EngineerDoneDetector(..., record_dir=None)` - 지정되면 회차마다 `numerator_reads.jsonl`
    에 한 줄 append 한다. 종전 debug 폴더 JSON 은 **그대로** 쓴다(둘 다 같은 dict).
  - `build_engineer_done_detector(..., record_dir=None)` 로 배선을 노출.
- `monitor/cycle.py` - engineer watch 가 수집 on 일 때만 attempt 폴더를 `record_dir` 로
  넘긴다. off 면 종전대로 debug 폴더만.
- `monitor/recovery_episode.py` - `_ATTEMPT_RECORD_FILES` 에 한 줄 추가.

**판정 우선순위(계약)**
- **reground 는 값이 읽혔어도 이긴다.** 재grounding 은 누적 sequence 를 되돌린 사건이라,
  그 회차 값을 '증가' 근거로 쓰면 서로 다른 ROI 에서 읽은 숫자를 한 줄로 이어 붙이게 된다.
- 첫 표본은 `strictly_increasing` 이 아니라 `first_sample` 이다. 길이 1 수열을 증가라고
  부르면 판독 한 번으로 "증가를 봤다" 가 성립한다.
- detector 의 boolean 반환·기존 테스트(53/53)는 무변경이다. 기록은 보조물이므로 기록
  쓰기가 실패해도 감지는 계속된다(테스트로 고정).

**테스트** `monitor/test_numerator_records.py` 6개. spec 테스트 8(기록 부분) 커버.
회귀: monitor 549 / engineer_done 53 / recording_filter+workflow_extract+workflow_4 269 통과.
