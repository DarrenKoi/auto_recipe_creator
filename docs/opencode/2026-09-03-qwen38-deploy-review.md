# oc-review: Qwen3.8-27B 배포 + 16GB 호스트 재배치 (2026-09-03)

- 대상: `77df01e...HEAD` (`84ccd7f` feat, `07b81f4` BF16 수정) - 13 files, +603/-25
- tier: `heavy` (glm-5.3 high). 근거: 13 files, 서빙/배포 표면, 신규 /proc 파서.
- Standards 근거 문서: 루트 `CLAUDE.md`(Code Conventions), `poc/workflow_3/README.md`,
  `deploy_vlms/UPLOAD.md`
- Spec: 저장소에 스펙 문서가 없어 **사용자가 세션 중 말한 요구사항을 원문 그대로**
  `_workspace/oc-review/spec.md`(gitignored)에 옮겨 그것을 기준으로 채점했다.
  diff 에서 역추론한 스펙이 아니다.
- 실행: Standards 1회 성공. Spec 은 **1차 실패**(glm-5.3 "Unexpected error" ->
  kimi-k3 fallback "Invalid API key"), gpt-5.6-luna 재시도는 서론만 내고 빈 응답,
  glm-5.3 재시도에서 성공(53s). 즉 Spec 축은 3번째 시도에서만 결과가 나왔다.

## Standards (glm-5.3, 원문 요지)

- **HARD(단, 선례 있음)**: `flask_api/vlm_serve/mai_ui_2b.py:1`,
  `qwen3_8_27b.py:1` 의 docstring 이 영어. CLAUDE.md 는 "Korean docstrings
  throughout all modules". 다만 기존 형제 파일(`mai_ui.py`, `got_ocr.py`)이 전부
  같은 영어 한 줄이라 패키지 관행과는 일치. 어느 쪽이 이기는지 판정 필요.
- **JUDGEMENT / Duplicated Code**: lazy-load 근거 주석이 3개 env 에 그대로 복붙.
  env 포맷에 include 가 없어 불가피 - 모델도 suppressed 로 분류.
- **JUDGEMENT / Mysterious mismatch**: `check_host_ram.py:97-99` 의
  `roll.get("Rss_Anon", roll.get("RssAnon", 0.0))` - smaps_rollup 이 내지 않는 키.
- 확인 후 clean: argparse 없음, logging 모듈 없음, `__future__` 없음,
  print 내 em-dash 없음(env 의 `─` 는 U+2500 주석), sibling import 는 규칙 밖.

## Spec (glm-5.3, 원문 요지)

- (a) 미흡: req 1(업로드 경로 확인)·req 6(PaddleOCR 대체 조사) 산출물이 diff 에
  없음 - 대화로만 전달됨. req 10(push)은 diff 로 검증 불가.
- (b) scope creep: mai-ui-2b 서빙 배선(단, `enabled=False` 기본 off + bench 전용
  명시라 "defensible"), bench_tool_locator 조합 수정(req 8 기록에 필요).
- (c) 구현이 틀려 보임:
  1. `start_all.py` 헤더가 아직 "27B **FP8**" - 07b81f4 의 BF16 전환과 모순.
  2. runbook 이 "호스트를 32GB 로 올리는 것이 정답"을 2곳에서 권함 - req 9
     ("increase to 32 불가, 16GB 고정")가 무효화한 조언.
  3. `--kv-cache-dtype fp8` 이 req 4(긴 컨텍스트)를 일부 잠식. env 가 스스로
     "장거리 recall 을 건드린다"고 적어둠. spec-pure 대안은 bf16 KV + SEQS=4.

## Claude 판정 / 조치

**동의하고 수정함:**

- Spec (c)1 `start_all.py` "27B FP8" -> BF16. **진짜 결함**. BF16 커밋에서 헤더를
  같이 안 고쳤다. `READY_TIMEOUT_SEC` 주석의 "27B FP8" 도 같이 발견해 수정.
- Spec (c)2 runbook 의 32GB 권고 2곳 삭제. 사용자가 증설 불가를 확정한 것이
  runbook 작성 **이후**라 문서가 낡았다. 올바른 레버(인스턴스 제거)로 교체.
- Standards `check_host_ram.py` 키 오류 - **진짜 결함이고 내가 쓴 코드다.**
  다만 모델의 설명은 틀렸다: "smaps_rollup only ever emits `Rss_Anon`/`Rss_File`"
  이라 했지만 커널 문서상 rollup 은 **그 둘 다 내지 않는다**
  (`RssAnon`/`RssFile` 은 `/proc/<pid>/status` 쪽, rollup 은 `Pss_Anon`/
  `Pss_File`/`Pss_Shmem`/`Anonymous`). 즉 fallback 이 dead 인 게 아니라 **주 키까지
  전부 miss** 라, anon 열과 "회수 불가" 합계가 **항상 0** 으로 찍혔다. 판정의
  근거가 되는 숫자가 조용히 거짓이 되는 결함이라 심각도는 모델이 매긴 것보다 높다.
  `Pss_Anon`/`Pss_File`+`Pss_Shmem` 으로 고치고, 파서를 순수 함수로 분리해
  /proc 없는 Mac 에서 도는 self-check(표본 파싱 + "RssAnon 은 없다" 회귀 단언)를 붙였다.

**동의하지 않거나 조정한 것:**

- Standards docstring 영어: 고치지 **않는다**. 형제 7개 중 5개가 영어 한 줄인데
  새로 넣은 2개만 한국어로 바꾸면 패키지 일관성이 더 나빠진다. CLAUDE.md 규칙과
  패키지 관행이 충돌하는 사안이라 사용자 판정 대상으로 남긴다.
- Spec (b)1 mai-ui-2b 배선 = scope creep: **내 spec 파일의 누락**이다. 사용자가
  조사 결과를 듣고 "I have to test by myself then" 이라고 말한 줄을 spec.md 에
  안 옮겼다. 그 발언이 harness 를 정당화하므로 creep 이 아니다. 모델은 주어진
  스펙 기준으로 정확히 판단했고, 틀린 것은 스펙을 옮긴 쪽이다.
- Spec (c)3 fp8 KV: 결함 아님. BF16 가중치(~48GiB)로 KV 풀이 72GiB 라
  bf16 KV 는 262k 를 4-way 로 제한한다. "긴 컨텍스트 제공"이 요구사항이므로
  동시성을 절반으로 깎는 쪽이 오히려 요구를 덜 만족한다. 되돌리는 절차는 env 에
  이미 적혀 있다. 모델도 "documented, reasoned tradeoff" 로 인정했다.
- Spec (a)2 PaddleOCR 조사 미기록: 타당하다. 결정("유지")의 근거가 저장소에 없다.
  절차가 아니라 판단이므로 runbook 대신 memory 에 기록했다.

## 축별 결론

- Standards: 3건 (HARD 1 = docstring 언어, 판정 보류 / JUDGEMENT 2). 최악:
  `check_host_ram.py` 키 오류 - **수정 완료**.
- Spec: 6건 (미흡 3 = 대부분 diff 밖 / creep 2 / 오구현 3). 최악:
  `start_all.py` FP8 표기와 runbook 의 32GB 권고 - **둘 다 수정 완료**.
