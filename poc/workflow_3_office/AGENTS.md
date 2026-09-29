# AGENTS.md - 오피스 LLM 작업 규칙 (`poc/workflow_3_office/`)

이 파일은 **오피스 PC 에서 도는 LLM 에이전트**용이다. 저장소 루트의 `AGENTS.md` / `CLAUDE.md`
와 어긋나면 **파일을 어디에 쓰는지에 한해서는 이 파일이 이긴다**. 코드 규약과 도메인 지식은
여전히 루트 `CLAUDE.md` 가 정본이다 - 작업 전에 읽을 것.

## 왜 이 폴더인가

- 오피스 PC 는 GitHub 에 **pull 만 된다(push 불가)**. core(`poc/workflow_3/` 등)는 Mac 에서
  고쳐 push 되고 오피스가 pull 한다.
- 오피스에서 추적 파일을 고치면 다음 `git pull` 이 "would be overwritten" 으로 멈추고, 누군가
  `git checkout .` 을 치는 순간 그 작업은 사라진다.
- 이 폴더는 `__init__.py` 와 이 `AGENTS.md` 를 빼고 gitignore 다. 여기 쓴 코드는 pull 과
  절대 충돌하지 않는다.

## 규칙

1. **쓰기는 이 폴더 안에서만.** `poc/workflow_3/`, `poc/workflow_3e/`, `poc/workflow_4/`,
   루트 파일, 다른 모든 추적 파일은 읽기 전용이다. 이 폴더의 `__init__.py` 와 `AGENTS.md`
   도 추적 파일이라 고치지 않는다.
2. **workflow_3 는 import 해서 쓴다** - `from poc.workflow_3.xxx import ...` (절대 import).
   역방향(workflow_3 가 이 폴더를 import)은 불가능하다 - core 는 이 폴더를 모른다.
3. **core 동작을 바꿔야 할 때**, 빠른 순서대로:
   1. env / 진입점 상단 상수로 되는지 먼저 본다(`load_workflow3_settings`, `ALIGN_FAIL_*`).
   2. 안 되면 필요한 함수를 **이 폴더의 모듈로 복사**해 고치고, 자기 진입점에서 그 사본을 쓴다.
   3. 깊은 곳이라 복사로 안 닿으면 **자기 진입점 맨 위에서 monkeypatch** 해도 된다
      (`import poc.workflow_3.align.correction as c; c.함수 = 내_함수`). core 파일 자체는
      건드리지 않는다.
   4. 2·3 을 했으면 **반드시 `CORE_REQUESTS.md` 에 한 건 적는다**(아래 형식). 사람이 그
      텍스트를 Mac 으로 옮기면 core 에 정식 반영되고, 그 뒤 사본/patch 를 지운다.
4. **`git pull` 뒤엔 반드시** `uv run pytest poc/workflow_3_office` 를 돌린다. core 함수의
   이름·인자가 바뀌면 여기 코드가 조용히 깨진다(이 폴더는 Mac 에서 안 보이므로 Mac 쪽은
   깨는 줄 모른다). 그래서 **모듈마다 작은 `test_*.py` 하나**를 둔다 - 최소한 import 와
   core 호출 모양이 맞는지.
5. **안전.** 새 진입점은 `SAFE_MODE=1` 리허설부터. `SAFE_MODE=0`(실클릭)은 사람이 명시적으로
   시킬 때만 돌린다. `align_fail_monitor` 의 `[위험]` 상수 넷(`BLOCK_INPUT`/`RCS_KILL_STALE`/
   `ACCESS_GRANT`/`CORRECT_WHEN_OCCUPIED`)은 켜지 않는다. OK/Print 를 누르는 새 흐름을
   만들지 않는다(기존 보정 사이클의 OK 는 예외).
6. **fab 데이터는 텍스트로만 밖에 나간다.** `CORE_REQUESTS.md` 에 이미지·웨이퍼 사진·원본
   로그 덤프를 붙이지 않는다 - 숫자/`[DIGEST]` 줄/짧은 콘솔 발췌만.

## 저장소 규약 (루트 `CLAUDE.md` 요약)

- 실행: 저장소 루트에서 `uv run python poc/workflow_3_office/<파일>.py`. **CLI 인자 금지**
  (argparse 없음) - knob 은 그 파일 맨 위 상수 블록. `.env` 는 비밀값 전용.
- 한국어 docstring, `[INFO]`/`[WARNING]`/`[ERROR]` print 로깅(`logging` 모듈 금지),
  `from __future__` 금지, `print()` 문자열에 em-dash(—) 금지(오피스 콘솔 cp949).
- debug 이미지는 JPEG. 새 debug 저장은 `poc.workflow_3.util.event_dir` 의 `debug_root()` /
  `runs_root()` 를 호출 시점에 쓴다.

## `CORE_REQUESTS.md` 형식

이 폴더에 만든다(gitignore). 한 건당:

```
## <YYYY-MM-DD> <한 줄 제목>
- 대상: poc/workflow_3/<파일>.py : <함수/상수>
- 왜: <현장에서 본 증상 - 숫자/digest 로>
- 바꿀 것: <diff 또는 정확한 서술>
- 여기서 임시로: <사본 모듈 이름 또는 monkeypatch 위치>
- 상태: open | Mac 반영됨(<커밋>) -> 사본 삭제함
```
