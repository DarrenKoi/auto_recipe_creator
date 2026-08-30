# 첫 Recovery Episode 수집 (티켓 18 오피스 게이트)

목적: 진짜 Align Fail 알람 1건을 Episode 수집 on 으로 기록한다.
결과는 무관하다. **파일이 생기면 통과다.** `outcome=unknown` 도 `incomplete` 도 통과다.

가져올 것은 **텍스트뿐**이다. 이미지와 원본 프레임은 반출하지 않는다.

---

## 0. 실행 전 확인

```
git status
```

`align_images/` 나 `align_consensus_cache/` 아래 파일이 staged 로 보이면 **커밋하지 말 것**.
`087c915` 전까지 이 두 경로가 실제로 무시되지 않았다. 보이면 `git restore --staged <경로>` 로 내린다.

```
git pull            # 2b2e3db 이상이어야 한다
git log --oneline -1
```

## 1. 수집 켜기

`poc/workflow_3/workflow_3_config.py` (오피스 PC 에만 있는 파일) 에서 한 줄만 바꾼다:

```python
EPISODE_COLLECT = 1          # None -> 1
```

이 파일이 없으면 `workflow_3_config.example.py` 를 복사해서 만든다.
셸 env 로도 되지만(`ALIGN_FAIL_EPISODE_COLLECT=1`) 파일 상수가 기본 규약이다.

## 2. 실행

```
uv run python poc/workflow_3/monitor/align_fail_monitor.py
```

이 진입점은 **실클릭이 기본**이다(`SAFE_MODE=0`). 알람이 올 때까지 폴링한다.

### 리허설만 먼저 하고 싶으면

```
SAFE_MODE=1 uv run python poc/workflow_3/monitor/align_fail_monitor.py
```

셸 env 가 항상 이긴다. 이건 **폴더가 생기는지만** 증명한다.
클릭이 막히므로 장비에 붙지 못하고, `guards.json` / Verification / 프레임은 비거나 안 생긴다.
게이트를 진짜로 통과하려면 실클릭 실행이 필요하다.

## 3. 무엇을 볼 것인가

알람 1건이 지나가면 콘솔에 이 한 줄이 찍힌다:

```
[DIGEST] episode id=xxxxxxxx eqp=... recipe=... attempts=N outcome=... guards=... verify=... complete=...
```

**이 줄을 그대로 복사한다.** 이미지 없이 이 줄만으로 상태가 읽히도록 만든 줄이다.

예상되는 정상 모습 (고장이 아니다):
- `outcome=unknown` 또는 `escalated`
- `guards=` 안에 `align:unknown` — matcher 가 `second_ratio` 를 안 채우면 unknown 이다
- `verify=` 가 fallback 또는 unknown — 자동 판독기는 일부러 unknown-only stub 이다

## 4. 폴더 확인

```
find <align_images 경로>/<EQP>/<class>/<recipe>/captured_img_from_rcs/<tag>/
```

있어야 하는 것:

```
<tag>/
  recovery_episode.json          <- Episode 정본 (첫 GUI step 전에 쓰인다)
  attempt_1/
    guards.json
    measurement_verification.json
    numerator_reads.jsonl
    frame_meta.jsonl
    recording/
```

`recovery_episode.json` 이 아예 없으면 게이트 실패다. 티켓 10 으로 돌아간다.

## 5. 집으로 가져올 것

| 가져온다 | 가져오지 않는다 |
|---|---|
| `[DIGEST] episode …` 한 줄 | 프레임 이미지 (`recording/*.jpg`) |
| `find <tag>/` 출력 전체 | 캡처 원본 |
| `recovery_episode.json` | 디버그 스크린샷 |
| capture manifest | |
| `guards.json` | |
| `measurement_verification.json` | |
| `numerator_reads.jsonl` | |
| `frame_meta.jsonl` **앞부분 20줄 정도** | |

`frame_meta.jsonl` 은 프레임당 1줄이라 길다. 스키마 확인이 목적이므로 앞부분이면 된다.

EQP/recipe 식별자 마스킹 여부는 사용자가 정한다. 마스킹한다면 **모든 파일에서 같은 문자열로**
바꾼다 - 파일끼리 이어붙는 키다.

## 6. 어디에 넣을 것인가

```
poc/workflow_3/monitor/episode_fixture/
```

티켓 19~27 이 여기서 읽는다. 이 경로는 gitignore 되지 않으므로 커밋된다.

## 7. 어긋난 것 기록

스펙과 다른 점(경로, 필드 이름, 예외 메시지)을 보면
`docs/issues/align-fail-recovery-playbook/issues/18-office-gate-first-real-episode.md`
의 Comments 에 텍스트로 남긴다. 이게 19~27 의 설계 입력이다.

## 되돌리기

수집이 시끄럽거나 느리면 `EPISODE_COLLECT = 0`.
새 동작이 전부 사라진다 - 폴더, 사이드카, manifest 필드, Guard/Verification 파일.
켰을 때 추가되는 비용은 프레임 사이드카(20fps Win32 호출)와 Assist 패널 VLM grounding **1콜**이다.
