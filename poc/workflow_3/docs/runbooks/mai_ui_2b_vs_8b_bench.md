# MAI-UI 2B vs 8B A/B (오피스 절차)

목적: grounding 모델을 8B -> 2B 로 내려도 되는지 **우리 화면에서** 재는 것.
공개 수치만으로는 결론이 안 난다(아래 §1). 실측이 유일한 근거다.

> **먼저 읽을 것 - 이 A/B 는 상방이 없다.**
> production 은 `mai-ui > mai-ui` (8B 양단)이고 **현재 성공률 100%** 다 (사용자 확인
> 2026-09-03). 즉 2B 가 낼 수 있는 최선은 '똑같이 100%' 이고, 그 대가로 얻는 것은
> 수백 MB~수 GB 의 호스트 RAM 추정치이며 **프로세스는 하나도 줄지 않는다**(§2).
> 기댓값이 '무승부 아니면 손해'인 교체다.
> 그러므로 이 문서는 "2B 로 갈 수 있나" 가 아니라 **"굳이 갈 이유가 있나"** 를 재는
> 절차이고, 기본 답은 **아니오** 다.
> **호스트 RAM 증설은 불가능하다**(사용자 확인 2026-09-03: 16GB 고정, 다른 선택지 없음).
> 그래도 RAM 이 병목이면 올바른 레버는 모델 축소가 아니라 **인스턴스 제거**다 - 2B 로
> 바꿔봐야 프로세스 수는 그대로이기 때문이다(§2). 그래도 재보고 싶다면 아래를 따른다.

작성 2026-09-03. Claude 없는 오피스에서 그대로 따라 하도록 쓴 문서다.

---

## 1. 왜 실측이 필요한가 - 공개 수치가 말해주는 것과 아닌 것

MAI-UI 논문/프로젝트 표 (저자 발표값):

| 벤치 | 2B | 8B | 8B 우위 |
|---|---|---|---|
| ScreenSpot-Pro | 57.4 | 65.8 | +8.4 |
| **ScreenSpot-Pro + Zoom** | **62.8** | **70.9** | **+8.1** |
| UI-Vision | 30.3 | 40.7 | +10.4 |
| MMBench-GUI L2 | 82.6 | 88.8 | +6.2 |
| OSWorld-G | 52.0 | 60.1 | +8.1 |
| ScreenSpot-v2 | 92.5 | 95.2 | +2.7 |

**Zoom 이 격차를 좁혀주지 않는다.** 우리 로케이터가 2단 zoom 구조라 2B 에 유리할
거라 기대하기 쉬운데(2B 는 zoom 으로 57.4 -> 62.8 로 오른다), 8B 도 65.8 -> 70.9 로
같이 올라 격차는 +8.4 -> +8.1 로 그대로다. 구조가 작은 모델을 구해주지 않는다.

ScreenSpot-Pro 는 **고해상도·조밀·전문 소프트웨어** 벤치라 우리 RCS List 화면과
가장 가깝다. 그 축에서 8점 차다.

무엇이 먼저 무너지는가 (논문 세부 표):
- 아이콘 -12.6점 / 텍스트 -5.9점 -> 아이콘이 먼저 무너진다.
- fine-grained 조작 51.0 -> 40.3.
- 전체 화면 텍스트 매칭 72.0 -> 62.8.

우리 대상은 아이콘이 아니라 텍스트 행이라 유리해 보이지만, **진짜 어려움은 ID 를
읽는 게 아니라 비슷하게 생긴 얇은 행들 중 맞는 행을 고르는 것**이다
(MCDC12/MCDC22, MCDN01/MCDN02). 이건 위 어느 벤치도 직접 재지 않는다.

**실사용 후기는 찾지 못했다.** GitHub issue / HF discussion / Reddit 를 뒤졌지만
2B 와 8B 를 실제 데스크톱 자동화에 붙여 비교한 field report 가 없다. 제3자 실험이
하나 있으나(슬라이드 요소 13만개, model-as-judge) 저자 스스로 최종 벤치가 아니라고
못박았고, 거기서도 **thinking 설정이 모델 크기보다 결과를 더 흔들었다**. 그래서
우리 프롬프트/파서 조합으로 직접 재는 것 말고는 방법이 없다.

## 2. 기대 이익을 먼저 낮춰 잡을 것 (중요)

이 A/B 의 동기는 호스트 RAM 16GB 압박이었다. 그런데 **절감은 크지 않다**:

- 체크포인트 차이는 17.6GB(8B) vs 4.27GB(2B) = 13.3GB 지만 그건 **GPU/디스크** 쪽이다.
- 우리는 이미 `--load-format safetensors --safetensors-load-strategy lazy` 와
  `--mm-processor-cache-gb 0` 을 쓴다. lazy 는 mmap 이라 가중치가 익명 호스트 메모리로
  복사되지 않는다. 즉 **가장 큰 절감은 이미 받아 놓은 상태**다.
- 남는 호스트 절감은 수백 MB ~ 수 GB 수준 추정이고, **프로세스 수는 그대로다**
  (Python/PyTorch/CUDA/토크나이저/이미지 프로세서 고정 비용은 두 모델이 거의 같다).

그러므로 판단 기준은 "RAM 이 얼마나 주나" 가 아니라 **"정확도를 얼마나 잃나"** 다.
잃는 게 있으면 하지 않는 편이 낫다. 그리고 **증설은 선택지가 아니다**(16GB 고정 확정).
RAM 이 진짜 병목이면 vLLM 인스턴스를 하나 **없애야** 한다 - 예를 들어 PaddleOCR-VL 을
vLLM 대신 PP-OCRv6 ONNX 로 돌리면 API/EngineCore 프로세스 한 쌍이 통째로 사라진다.
모델을 작게 바꾸는 것은 프로세스를 안 줄이므로 이 제약에 대한 답이 아니다.

## 3. 실행 절차

### 3-1. 준비 (한 번만)

`MAI-UI-2B` 를 서버 모델 루트에 올린다 (오프라인이라 청크 업로드, `deploy_vlms/UPLOAD.md`).
경로는 `.../data/models/MAI-UI-2B` (`config/models/mai-ui-2b.env` 의 `MODEL_ID`).

`flask_api/vlm_serve/config.py` 에서 `mai-ui-2b` 를 `enabled=True` 로 바꾸고 Flask 재기동.
(기본 off 다 - 상시 기동이 아니라 벤치용이므로.)

### 3-2. 벤치용 GPU/RAM 재배치

**4번째 vLLM 인스턴스를 올리지 말 것.** 호스트 RAM 16GB 에 3개가 이미 한계다.
벤치 동안 제일 큰 것을 내린다:

```bash
uv run python deploy_vlms/scripts/stop_model.py qwen3.8-27b
uv run python deploy_vlms/scripts/start_model.py mai-ui-2b
uv run python deploy_vlms/scripts/check_vlm.py     # mai-ui, mai-ui-2b, paddleocr 3개 alive 확인
```

RAM 을 눈으로 확인할 것 (ps 의 RSS 는 mmap 때문에 과장된다):

```bash
free -h
for p in $(pgrep -f 'vllm.entrypoints'); do echo "== $p"; grep -E '^(Rss|Pss|Private)' /proc/$p/smaps_rollup; done
```

### 3-3. 측정

RCS 로그인 + List 탭이 보이는 상태여야 한다. **클릭하지 않으므로 장비에 영향이 없고
알람도 필요 없다.**

```bash
# 1) 스모크 - 경로가 도는지, tool 이 화면에 보이는지 (조합당 1회)
BENCH_REPEATS=1 uv run python poc/workflow_3/rcs/bench_tool_locator.py

# 2) 본 측정 - 기본 조합 4개가 이미 이 A/B 다
uv run python poc/workflow_3/rcs/bench_tool_locator.py
```

기본 조합이 4개인 이유는 **어느 단이 나빠졌는지 갈라내기 위해서**다:

| 조합 | 뜻 |
|---|---|
| `mai-ui > mai-ui` | production baseline |
| `mai-ui-2b > mai-ui` | coarse(전체 화면에서 얇은 행 찾기)만 2B |
| `mai-ui > mai-ui-2b` | fine(확대 crop 에서 점 찍기)만 2B |
| `mai-ui-2b > mai-ui-2b` | 실제 교체안 |

**스크롤 위치를 바꿔 최소 3회 반복할 것.** 벤치는 프레임을 한 번만 캡처하므로
`BENCH_REPEATS=3` 은 '같은 화면에서의 흔들림'만 재지 화면 다양성을 재지 않는다.

### 3-4. 되돌리기 (반드시)

```bash
uv run python deploy_vlms/scripts/stop_model.py mai-ui-2b
uv run python deploy_vlms/scripts/start_model.py qwen3.8-27b
uv run python deploy_vlms/scripts/check_vlm.py
```

`config.py` 의 `mai-ui-2b` 도 `enabled=False` 로 되돌린다.

## 4. 합격 기준

**평균 정확도로 판단하지 말 것.** 하류 OCR 확인 게이트가 lenient 라서
'못 읽음'은 통과하고 '다른 ID 를 읽음'만 거부한다(`tool_row_verify.py`). 즉 위험은
평균이 아니라 **wrong_row 가 늘어나는 것**에 있다.

교체 승인 조건 (baseline 이 100% 이므로 사실상 만점 요구다):

- [ ] `mai-ui-2b>mai-ui-2b` 도 **100%**. baseline 이 100% 라 'wrong_row 0 증가' 는
      곧 '한 건도 틀리지 않기' 와 같은 말이다. 한 건이라도 틀리면 기각.
- [ ] 헷갈리는 인접 ID 쌍(MCDC12/MCDC22 등)에서 계통적 열화가 없을 것.
- [ ] **2B 가 baseline 과 다르게 답한 모든 케이스의 overlay 를 눈으로 확인**할 것.
      채점 오라클이 PaddleOCR 이라 오라클도 틀릴 수 있다.
- [ ] 스크롤 위치 3곳 이상에서 재현될 것.
- [ ] 8B-only / 2B-only 각각에서 `MemAvailable` 과 프로세스 PSS 를 기록해
      **실제 절감이 얼마인지 수치로 남길 것** (§2 의 기대치가 맞는지 확인).

교차 조합 해석:
- coarse 만 2B 일 때 나빠지면 -> 전체 화면 탐색이 병목. 이때 8B 를 coarse 로 남기면
  품질은 지키지만 **인스턴스가 그대로라 RAM 목적은 달성되지 않는다.**
- fine 만 2B 일 때 멀쩡하면 -> 확대된 crop 에서는 2B 로 충분하다는 뜻.

## 5. 결론이 '기각'일 때

그게 기본값이고, 지금 가진 근거로는 그쪽이 유력하다:

- 공개 수치는 전 벤치에서 8B 가 5~10점 앞서고, 우리 화면과 가장 가까운 축
  (ScreenSpot-Pro)에서 8점 차이며 zoom 을 써도 격차가 줄지 않는다(§1).
- 실사용 후기가 존재하지 않아 남의 경험으로 위험을 낮출 수 없다(§1).
- RAM 절감은 수백 MB~수 GB 추정이고 **프로세스는 안 준다**(§2).
- **baseline 이 이미 100% 라 이길 수가 없다.**

**잃는 것이 확실하고 얻는 것이 불확실하면 8B 를 유지한다.** 이 문서는 나중에
"2B 는 검토했었나?" 라는 질문에 답하기 위한 기록으로도 쓴다 - 검토했고, 위 근거로
하지 않기로 했다.
