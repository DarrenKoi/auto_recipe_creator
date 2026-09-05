# VLM 서비스 변경 후 오피스 체크리스트 (2026-09-05)

목적: 2026-09-03 (ui-venus / ui-tars / got-ocr 가중치 삭제) 과 2026-09-05 (mai-ui-2b 폐기)
이후 `poc/workflow_3` 가 오피스에서 **첫 사이클을 돌기 전에** 확인할 것을 모은다.

git 에 올라간 코드는 손댈 것이 없다. grounding 기본값 6종은 전부 `mai-ui`, OCR 은
`paddleocr-vl-1.5`, `service_fallback_order` 는 `("mai-ui",)` 하나다. 남은 위험은 **git 이
안 건드리는 자리** - 셸 env, gitignored 설정 사본, `.env`, 서버 프로세스 - 다.

가져올 것은 **텍스트뿐**이다. 이미지는 반출하지 않는다.

---

## 0. 코드 동기화

```
git pull
git log --oneline -1        # cc2972a 이상
```

## 1. 서빙 프로세스 3개 살아 있는지

```
curl http://itc-1stop-solution-gpu-image-webapp.aipp02.skhynix.com/api/vlm_serve/health
```

통과 기준: `mai-ui`(8002) / `paddleocr-vl-1.5`(8004) / `qwen3.8-27b`(8006) 셋이 up.
workflow_3 진입점은 **VLM preflight 를 하지 않는다** - 죽은 포트는 첫 사이클 안에서
404/connection error 로만 드러난다. 여기서 먼저 잡는다.

`poc/work2/connection_check.py` 는 쓰지 않는다. 옛 work2 registry 를 import 해서
ui-venus/ui-tars/got-ocr 를 여전히 대상으로 잡고 실패로 찍는다 - 회귀가 아니다.
굳이 쓰려면:

```
$env:CONNECTION_CHECK_SERVICES = "mai-ui,paddleocr-vl-1.5,qwen3.8-27b"
```

## 2. 셸 env 에 죽은 slug 가 박혀 있지 않은지

```
Get-ChildItem env: | Where-Object { $_.Name -match "VLM|_SERVICE|LOCATOR" }
```

아래 이름 중 값이 `ui-venus` / `ui-tars` / `got-ocr` / `mai-ui-2b` 인 것이 있으면 지운다.
셸 env 는 파일 상수보다 **항상 이기므로** 여기 남은 값 하나가 전체를 막는다.

| env 이름 | 살아 있는 값 |
|---|---|
| `VLM_LOCATOR_COMBO` | 비움 (기본 `mai-ui>mai-ui`) |
| `ALIGN_FAIL_SEM_BOX_SERVICE` | `mai-ui` |
| `ALIGN_FAIL_OCCUPIED_POPUP_SERVICE` | `mai-ui` |
| `ALIGN_FAIL_ENGINEER_DONE_VLM_SERVICE` | `mai-ui` |
| `ALIGN_OK_BUTTON_VLM_SERVICE` | `mai-ui` |
| `MEAS_FAIL_ABORT_BUTTON_SERVICE` | `mai-ui` |
| `ALIGN_FAIL_PM_OCR_SERVICE` | `paddleocr-vl-1.5` |
| `ALIGN_FAIL_ENGINEER_DONE_OCR_SERVICE` | `paddleocr-vl-1.5` |

```
Remove-Item env:VLM_LOCATOR_COMBO      # 예시. 시스템 env 에 박혀 있으면 설정 앱에서 지운다
```

죽은 slug 는 프록시가 아니라 **`get_service_by_slug` 가 `None`** 을 내는 자리에서 실패한다.
에러 문구에 slug 이름이 그대로 나오니 그걸로 어느 env 인지 역추적한다.

## 3. gitignored `workflow_3_config.py` 사본

오피스 사본은 Mac 사본과 다르다. 열어서 확인:

```
Select-String -Path poc\workflow_3\workflow_3_config.py -Pattern "SERVICE|COMBO|VLM"
```

통과 기준: `LOCATOR_COMBO = None`, 서비스 상수에 죽은 slug 없음.
진입점 시작 시 loader 가 "env 에 밀려 무시된 상수" 를 찍는다 - **그 블록도 읽는다.**
무시된 상수가 있으면 2번의 셸 env 가 아직 남아 있다는 뜻이다.

## 4. `.env`

비밀값 전용이어야 한다.

```
Select-String -Path .env -Pattern "VLM|_SERVICE|LOCATOR"
```

한 줄이라도 나오면 지운다. `.env` 도 파일 상수보다 앞선다.

## 5. 직결 게이트웨이 모델명

클라이언트 registry 의 direct 항목은 `Kimi-K2.6` / `GLM-5.2` 다.

```
curl http://common.llm.skhynix.com/v1/models
```

통과 기준: 두 이름이 목록에 있다. `prompt_align_compare` 가 Kimi-K2.6 을 쓰고 사내
Qwen3-VL 은 deprecated 라 이름이 바뀌면 그 경로만 조용히 죽는다.

## 6. 첫 사이클

check-only 로 먼저 돈다 (접속 -> 1 캡처 -> 닫기, 보정/녹화 없음):

```
uv run python poc/workflow_3/monitor/align_fail_monitor_only_check.py
```

시작 배너에서 볼 것:

- 로케이터 조합 줄이 `mai-ui>mai-ui`
- loader 의 무시된 상수 블록이 비어 있음
- `_report_data_paths` 의 경로 건강 보고

한 사이클 뒤 `debug_images/mai-ui/` 아래에 좌표/오버레이가 쌓이면 통과.
`debug_images/ui-venus/` 가 **새로** 생기면 어딘가 옛 slug 가 살아 있다 - 2~4번으로 돌아간다.

## 7. 아직 mai-ui 로 미검증인 경로 (서비스 변경과 무관, 상황이 와야 함)

- OK 버튼 (`align/ok_button.py`)
- 점유 팝업 (`monitor/occupied_popup.py`)
- engineer-done 카운터 (`monitor/engineer_done_align_adjustment.py`)
- 3e abort 버튼 (`poc/workflow_3e/abort_button.py`)

오피스 `workflow_3_config.py` 의 프로파일 [B] 관측 SAFE 가 이 넷을 보려고 만든 설정이다.
결과는 `[DIGEST]` 줄과 `debug_images/mai-ui/<tag>/` 의 json/txt 만 가져온다.

## 낡아서 실패하는 것 (고칠 대상 아님)

- `test/flask_api/tests/test_vlm_serve.py` - 삭제된 slug 를 기대. 고친 판은 `../llm_serving/tests/`.
- `poc/work2/connection_check.py` - 1번 참조.
