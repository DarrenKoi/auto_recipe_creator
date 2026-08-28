---
status: accepted
---

# cycle3(workflow_3 알람 사이클)은 읽기 전용 mirror 로 live graph view 를 얻는다

## 결정

workflow_3 의 알람 사이클(`run_alarm_cycle` / `run_check_only_cycle`)은 **그대로
WorkflowRunner 로 실행**한다. 실행을 workflow_4 엔진으로 옮기지 않는다. 대신
옵트인 어댑터 `CycleGraphMirror`(`poc/workflow_4/adapters/workflow3_cycle.py`)가
runner 가 남기는 **step 저널(run_dir/run_state.json + step_<id>.json)을 읽기
전용**으로 폴링해 workflow_4 `RunState` + 그래프 스냅샷으로 미러링한다.

- Windows 에서 볼 live view 는 **self-contained HTML**(`framework/assets/mermaid.min.js`
  를 인라인 임베드 + `<meta http-equiv="refresh">` 자동 새로고침)이다. IDE/VS Code
  없이 Edge/Chrome 등 아무 브라우저로 열린다.
- `ALIGN_FAIL_GRAPH_VIEW`(기본 0) 로 켠다. 켜지 않으면 workflow_3 동작은
  byte-identical. workflow_3 은 workflow_4 에 하드 의존하지 않는다(import 실패 시
  경고 1회 후 자동 비활성).
- import 경계: `poc/workflow_4/adapters/` 만이 workflow_4 → poc.workflow_3 import
  가 허용되는 유일한 지점이다(현재는 저널 JSON 을 직접 읽어 실제 wf3 import 없음).

## 맥락 / 이유

- workflow_3 사이클은 이미 성숙한 실행기(WorkflowRunner) + 보장된 teardown +
  결과-후-알림 규약을 가진다. 실행을 다른 엔진으로 옮기면 회귀 위험(teardown 순서,
  알림 게이트, cooldown)이 전부 다시 생긴다.
- 실요구는 "지금 사이클이 어느 step 에 있고, 지금까지 뭐가 지나갔나" 를 **눈으로
  보는 것**이다. runner 는 이미 전이마다 `run_state.json` + `step_<id>.json` 을
  남기므로, 읽기만 하면 실행에 손대지 않고 같은 정보를 얻을 수 있다.
- 오피스 PC 는 Windows + Edge 가 있고, 엔지니어는 IDE 를 열지 않는다. mermaid.md 를
  IDE 미리보기로 보라는 접근은 오피스에서 쓸 수 없으므로, 브라우저에서 바로 뜨는
  self-contained HTML 이 필요하다. 인라인 임베드는 파일을 복사/이동해도 깨지지
  않는다(오프라인 공장망에서 CDN 을 못 쓴다는 가정까지 해결).

## 고려한 대안

| 대안 | 검토 결과 |
|---|---|
| workflow_3 실행을 WF4 엔진으로 이관 | 실행기 교체 = teardown/알림/cooldown 회귀 위험 + 실행기 코드 중복. 행동 변화가 없는 게 최우선이라 기각. |
| workflow_3 에서 직접 HTML 생성 | 시각화 로직이 프레임워크와 분리돼 중복 발생. WF4 `graph_view.py` 를 재사용하는 편이 일관적. |
| 저널을 복사/이동해 다른 곳에서 렌더 | 읽기 전용 유지는 좋지만 파일 복제가 늘고, run_dir 안에 두는 게 엔지니어가 찾기 쉽다(저널 옆). |
| IDE mermaid 미리보기(.md) | Windows 오피스 PC 는 IDE 가 아니라 브라우저가 실사용 경로. HTML 이 정본이고 .md 는 부가 산출물. |
| CDN 참조 HTML | 공장망은 오프라인일 수 있어 실패 리스크. vendored asset 인라인 임베드로 해결(3.6MB). |

## 결과 (Consequences)

- workflow_3 쪽 변경은 최소: `config.py` 에 `graph_view_enabled`/`graph_view_autoopen`
  두 필드, `cycle.py` 에 가드된 `_maybe_start_graph_mirror` 훅 2곳(run 전 시작,
  finally 종료). 기본 off 로 기존 동작 불변.
- `run_dir` 은 runner 가 run() 시작 시 만들어 `context["run_dir"]` 에 넣는다. mirror
  는 그 값을 `run_dir_fn` 으로 폴링마다 읽는다. (처음엔 경로를 예측하고 glob 으로
  "가장 새 폴더" 를 채택했는데, 첫 폴이 runner 의 mkdir 보다 빠르면 **직전 run 의
  폴더**에 영구히 붙는 경쟁이 있어 2026-08-28 에 바꿨다.)
- 그래프는 어댑터 안의 step/실패 class 표가 아니라 호출부가 runner 에 넘기는 실제
  step 목록으로 만든다. 복사한 표는 첫날부터 어긋나 있었다(`rcs_recovery_failed`
  는 존재하지 않는 class). 실패 간선은 runner 의 실제 동작대로 step 마다
  `failed -> aborted` 하나다.
- 저널 타이밍 특성상 "in-flight 노드" 는 정확히 알 수 없다(저널은 step **끝**에
  쓰인다). 따라서 in-flight = step_results 개수 번째 step 으로 근사하고, aborted
  종료 시 마지막 실패 step → aborted 로 가는 **합성 레코드** 1개를 history 에
  닫아준다(모듈 docstring 에 명시).
- HTML 은 폴링 변경마다 3.6MB(인라인 mermaid 포함)를 덮어쓴다. overwrite-only 라
  디스크 누적은 없고, live view = 최신 상태 하나다. 쓰기는 tmp + `os.replace` 라
  1s 마다 새로고침하는 브라우저가 반쯤 쓰인 파일을 읽지 않는다.
- 미해결: 실시간(초 단위) step 내부 진행(예: VLM 왕복)은 저널에 없어 그래프에
  안 보인다 — step 단위 granularity 만 약속한다.