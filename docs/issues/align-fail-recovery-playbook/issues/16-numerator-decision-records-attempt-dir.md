# 16 — Numerator decision records into the attempt directory

Type: task
Status: ready-for-agent
Blocked by: 11
Spec: [spec.md](../spec.md) (Structured Verification and Outcome)

## What to build

engineer-done detector 는 Recipe Monitor 분자(N/M)의 per-read 판독(값, 시각, 읽힘/미읽힘/relocalization,
단조 판정)을 지금 debug 폴더에만 남긴다. Episode 수집이 켜져 있으면 같은 판독을 attempt 폴더에
JSONL 로 쓴다. fallback Verification 이 이 기록을 읽기 때문이다. detector 의 boolean 반환은 녹화 조기
종료용으로만 남고 Verification 입력이 아니다 - `false` 와 `unknown` 을 구분하지 못하기 때문이다.

## Acceptance criteria

- [ ] 알람 사이클의 engineer watch 가 attempt 폴더에 numerator 기록을 남긴다(수집 off 면 종전 debug 폴더만).
- [ ] 기록에서 OCR miss / 같음·감소 / reground reset / strictly increasing 이 구분된다.
- [ ] detector 반환 계약과 기존 detector 테스트는 무변경이다.
- [ ] spec 테스트 8(기록 부분)을 덮는다.
