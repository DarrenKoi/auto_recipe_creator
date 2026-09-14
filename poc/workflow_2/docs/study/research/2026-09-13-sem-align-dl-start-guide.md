# SEM align DL 실험 시작 가이드 — 수집·저장·첫 비교

> **2026-09-13 · 오프라인 실험 준비 가이드**
> [실험 기획서](2026-09-13-sem-align-dl-experiment-plan.md)의 실행 순서를 구체화한다.
> 아래 수량·학습 설정은 출발점 제안이며 확보된 데이터량이나 성능 보장이 아니다. 이 문서 작성으로 수집기·학습기가 구현되거나 장비 실험이 수행된 것은 아니다.

## 1. 가장 먼저 할 일

**첫 목표는 모델 학습이 아니라, 10개 recipe에서 약 100개의 올바른 recipe–SEM 쌍을 만드는 것이다.** 여기서 “올바른 쌍”은 같은 recipe revision·align step의 등록 영상과 target이며, 성공 근거와 좌표 의미를 확인할 수 있는 것을 뜻한다. 이미지 100장과 학습 쌍 100개는 다르다.

| 순서 | 바로 할 작업 | 완료 산출물 |
|---|---|---|
| 1 | 사내 원본 저장 위치와 이미지·cond·성공 기록의 연결 키 확인 | `source_inventory.csv` |
| 2 | 10개 recipe에서 정상 S 약 10장씩, 가능하면 서로 다른 측정 건에서 수집. 관련 E도 별도 수집 | 원본 snapshot + 수집 기록 |
| 3 | 먼저 20쌍을 사람이 확인: SEM인가, 같은 step인가, 좌표가 맞는가 | 검토 기록 + 좌표 표시 그림 |
| 4 | 나머지까지 분류해 유효 쌍과 제외 사유 확정 | `pairs.jsonl`, `audit.md` |
| 5 | 좌표 왕복 점검 → CV 기준선 → D1 연결성 확인 | 샘플별 예측과 픽셀 오차 |
| 6 | 위 과정이 성립하면 50~100개 recipe/패턴 계열, 500~2,000쌍으로 확대 | 고정 개발 split + 별도 신규 holdout |

20쌍은 좌표 계약 점검용이지 정확도 측정에 충분한 규모가 아니다. 처음 본 100쌍은 이후 개발 데이터로 취급한다. 이미 눈으로 보고 튜닝에 쓴 사례를 최종 test로 다시 이름 붙이지 않는다.

**오늘 확보할 정보:** 원본 루트, recipe revision/step 식별 방법, 정상 성공 기록의 위치, 실제 측정 시각의 출처, 검토 담당자, 사내 GPU 환경. 모르는 항목은 `unknown`으로 기록하고, 필요한 근거가 없는 샘플만 해당 실험에서 제외한다. GPU가 없어도 수집·라벨 감사는 시작할 수 있다.

## 2. 무엇을 어떻게 수집하나

### 2.1 최소 수집 묶음

한 측정 이벤트에서 다음을 묶어 복사한다. 원본은 읽기만 하고 사내 연구 저장소에 snapshot을 만든다. 덮어쓰기가 가능한 운영 폴더를 학습 중 직접 읽지 않는다.

| 항목 | 수집 내용 | 빠졌을 때 |
|---|---|---|
| Recipe reference | 등록 SEM 이미지, 숨김 cond 폴더, recipe revision 또는 당시 reference hash | target 시점의 등록 영상인지 모르면 pair 보류 |
| 정상 S | 원본 이미지, 해당 cond, 측정/step ID, 성공 상태 근거 | 파일명 S만으로 검증 라벨 승격 금지 |
| 실패 E | 원본 이미지, 해당 cond, 실패/step 기록 | 독립 정답 전에는 좌표 없는 실패 자료 |
| 이벤트 메타 | 장비, class, recipe, lot/wafer/측정 ID(확보 가능 범위), 실제 측정 시각·출처 | 시간 미상은 시간 평가·과거 consensus 재료 선정에서 제외 |
| 수집 메타 | 원본 경로, 수집 시각, 파일 크기, SHA-256, 복사 누락/오류 | 복사 검증 전 snapshot 확정 금지 |
| 선택: live | 복구 중 캡처한 원본 화면, SEM ROI, 표시 크기·배율, 캡처 시각/step | 저장 MSR과 별도 도메인으로 유지 |

`align_img_from_msr`의 S/E는 저장된 측정 자료다. **복구 도중 현재 SEM은 live 캡처로 얻으며 MES 다운로드로 대체하지 않는다.** 첫 실험은 저장 recipe–MSR에 한정하고 live 수집은 후속 관측 단계로 둔다.

수집은 현재 사내 파일 공유/export 경로로 작은 묶음부터 한다. MES 전체를 가져오는 신규 다운로드 도구가 이미 있다고 가정하지 않는다. 기존 consensus 캐시는 최근 S의 일부만 담을 수 있어 모집단을 대표하는 학습 저장소로 취급하지 않는다.

### 2.2 표본 선정표를 먼저 만든다

`source_inventory.csv`는 원본 폴더/recipe revision별 한 행으로 시작한다. 권장 열은 `source_root, eqp_id, class_name, recipe_name, recipe_revision, n_S, n_E, n_cond, modality_evidence, time_evidence, export_status`다. 미집계 수량은 0 대신 빈 값으로 둔다.

처음 10개 recipe에는 반복 패턴과 구별되는 표식, 낮은/높은 SEM 배율, 서로 다른 box 크기·중심 offset, 기존 CV 성공·실패를 섞는다. 가능하면 장비도 둘 이상 포함하되 확보하지 못하면 단일 장비 탐색으로 명시한다. 쉬운 S만 모으지 말고 E도 접근 가능한 만큼 보존한다. E 20~50장은 검토 작업량을 가늠하기 위한 추가 표본 제안이며, 개수를 채우려고 정답을 만들어내지 않는다.

같은 측정 건의 연속 프레임은 서로 독립된 성공 사례가 아니다. recipe당 10장이 모두 한 건에서 나왔다면 측정 건수도 1이라고 보고한다. 파일명·다운로드 순서·mtime만으로 step 또는 실제 촬영 순서를 추정하지 않는다.

### 2.3 복사 완료 확인

1. 이미지와 `.<이미지 파일명>/cond.txt`를 **함께** 복사한다. 숨김 폴더 제외 옵션을 해제한다.
2. 원본과 복사본의 파일 수·크기·해시를 비교한다. 변경 중인 파일은 다시 snapshot을 받아 확인한다.
3. 이미지 decode, cond 파싱, SEM 판정, 원본 해상도, step 연결을 검사한다.
4. 원본이 이미 JPEG라면 그대로 보존한다. 정제·crop 결과와 mask는 PNG 등 무손실 파생본으로 저장한다. 눈검토 그림은 JPEG여도 된다.
5. 누락도 기록에 남긴다. 다시 받은 파일은 새 snapshot/version으로 저장한다.

## 3. 권장 폴더 구조

사내 저장소에 **코드 checkout 밖의** `sem_align_dl/` 루트를 둔다. 예시 위치는 Windows `D:\sem_align_dl`, Linux `/data/sem_align_dl`이며 실제 존재하는 경로라는 뜻은 아니다. 이미지뿐 아니라 cond, manifest, 검토 그림, 학습 가중치도 사내에 둔다. Git에는 이 가이드와 추후 코드·비식별 설정 예제만 넣는다.

```text
sem_align_dl/
├── raw/
│   └── snap_20260913_001/
│       └── EQP_A/CLASS_A/RECIPE_A/
│           └── events/evt_000001/
│               ├── event.json
│               ├── align_img_from_rcp/
│               │   ├── IMAP0002.jpeg
│               │   └── .IMAP0002.jpeg/cond.txt
│               └── align_img_from_msr/
│                   ├── S0001.jpeg
│                   ├── .S0001.jpeg/cond.txt
│                   ├── E0001.jpeg
│                   └── .E0001.jpeg/cond.txt
├── manifests/
│   └── ds_v001/
│       ├── source_inventory.csv
│       ├── files.jsonl
│       ├── pairs.jsonl
│       ├── annotations.jsonl
│       ├── exclusions.csv
│       ├── dataset.json
│       └── splits/
│           ├── family_v1/{train,val,test}.txt
│           ├── time_v1/{train,val,test}.txt
│           └── equipment_v1/{train,val,test}.txt
├── derived/
│   └── ds_v001/prep_v001/<pair_id>/
│       ├── recipe_clean.png
│       ├── target_clean.png
│       ├── recipe_mask.png
│       ├── target_mask.png
│       └── transforms.json
├── review/
│   └── ds_v001/{audit.md,overlays/}
├── adapters/
│   └── ds_v001/                    # evaluator/YOLO 형식의 파생 export
├── models/
│   └── <model_id>/{weights,provenance.json,LICENSE}
└── runs/
    └── 20260913_001_B0/
        ├── config.json
        ├── predictions.jsonl
        ├── metrics.json
        ├── environment.txt
        ├── run.log
        └── cases/
```

위 `{train,val,test}.txt` 등은 여러 파일을 줄여 쓴 표기다. **실제 파일명은 `train.txt`, `val.txt`, `test.txt`로 각각 만든다.**

이 `events/<event_id>`는 **연구 snapshot에서 동명 파일 충돌을 막기 위한 추가 계층**이다. 운영의 `align_images/<eqp>/<class>/<recipe>/align_img_from_*` 레이아웃은 변경하지 않는다. 기존 loader/evaluator가 이 연구 트리를 자동으로 읽는다고 가정하지 않는다.

동일 recipe에 새 측정 결과가 도착해 `S0001.jpeg`가 다시 생겨도 다른 event 아래 보존한다. 같은 recipe 이름이 재등록되면 revision 또는 reference content hash로 구분한다. source event ID가 없으면 연구용 고유 ID를 발급하고 `source_event_id=null`을 남긴다. 연구 ID의 숫자는 측정 순서의 증거가 아니다.

처음에는 `raw`, `manifests`, `review`만 만들면 된다. `derived`, `models`, `runs`, `adapters`는 해당 작업을 시작할 때 만든다. TB 단위 파일을 처음부터 복제하거나 DB·object storage를 신규 도입할 필요는 없다.

## 4. manifest는 무엇을 기록하나

### 4.1 파일 목록과 학습 쌍을 분리한다

- `files.jsonl`: 이미지·cond·성공 증거 파일당 한 행. `file_id, relative_path, sha256, size_bytes, source_path, collected_at`를 기록한다. 경로는 연구 루트 기준 상대 경로다.
- `event.json`: `event_id, source_event_id, eqp_id, class_name, recipe_name, recipe_revision, measurement_id, wafer_id, measured_at, time_source, collected_at`를 기록한다. 모르는 값은 `null`이다.
- `pairs.jsonl`: **reference 1개 + target 1개 + step 대응 + 라벨 1개**당 한 행. 실제 파일 경로는 `file_id`로 찾는다. 한 reference를 여러 S와 연결해도 각각 다른 pair다.
- `annotations.jsonl`: 검토자, 검토 시각, 증거 file ID, 판단, 정정 대상 ID. 정정은 새 행으로 남기고 dataset 확정 시 적용 버전을 고정한다.
- `exclusions.csv`: `pair_id, reason, detail`. 제외한 pair도 원본과 연결을 유지한다.

### 4.2 pair 한 행의 예시

다음은 **가상 데이터의 읽기용 예시**다. 실제 JSONL에는 객체 하나를 한 줄로 저장한다. 숫자는 장비 기준을 뜻하지 않는다.

```json
{
  "schema_version": 1,
  "pair_id": "pair_000001",
  "event_id": "evt_000001",
  "eqp_id": "EQP_A",
  "recipe_key": "CLASS_A/RECIPE_A/rev_03",
  "pattern_group_id": "family_007",
  "measurement_id": "measurement_001",
  "step_id": "sem_align_02",
  "pairing_evidence_file_id": "f_step_record",
  "reference_file_id": "f_rcp_001",
  "reference_cond_file_id": "f_rcp_cond_001",
  "target_file_id": "f_s_001",
  "target_cond_file_id": "f_s_cond_001",
  "target_kind": "S",
  "modality": "sem",
  "modality_evidence": "accelerating_voltage",
  "reference_size_wh": [512, 512],
  "target_size_wh": [512, 512],
  "reference_cond_pixel_wh": [512, 512],
  "target_cond_pixel_wh": [512, 512],
  "reference_magnification": 10000,
  "target_magnification": 10000,
  "reference_anchor_xy": [256.0, 256.0],
  "reference_box_xyxy": [180.0, 170.0, 340.0, 330.0],
  "target_point_xy": [243.6, 271.2],
  "target_box_xyxy": null,
  "coordinate_space": "loaded_original_image_xy",
  "label_source": "cond_crosshair",
  "label_grade": "S_verified",
  "success_evidence_file_id": "f_success_record",
  "annotation_id": "annotation_001",
  "measured_at": "2026-09-10T09:15:00+09:00",
  "time_source": "measurement_record",
  "consensus_source_pair_ids": []
}
```

`pattern_group_id`는 이름이 달라도 같은 패턴·recipe 파생본이면 묶는 식별자다. 원본 해시가 같은 샘플과 유사 중복도 함께 묶는다. `consensus_source_pair_ids=[]`는 현재 미지정이라는 뜻이며, B1 실행 시 실제 사용 재료와 제외 사유를 별도로 저장한다.

`dataset.json`에는 snapshot/manifest 해시, 코드 commit과 미커밋 변경 유무, annotation 버전, 표본 선정 규칙, split seed, 제외 규칙, 확정 시각을 남긴다. 나중에 라벨을 고치면 `ds_v002`로 만들고 이전 실험 결과를 덮어쓰지 않는다.

### 4.3 라벨 등급

| 등급 | 의미 | 사용 |
|---|---|---|
| `S_candidate` | cond crosshair는 읽었지만 성공/step/기하 확인 전 | 감사 대기, point 학습·채점 제외 |
| `S_verified` | 성공 근거와 step 연결을 확인했고 좌표 계약 감사 통과 | 정상 S 감독 학습·평가 |
| `E_independent` | E crosshair와 CV 예측에 의존하지 않는 별도 근거로 정답 확인 | 실패 좌표 평가; 학습 사용 시 split 준수 |
| `ambiguous` | 반복 주기 등으로 단일 정답을 확정할 수 없음 | 보류/실패 분석, 단일 point 채점 제외 |
| `unlabeled` | 좌표가 없거나 품질 미확인 | train에 속한 원본만 합성/무라벨 학습에 사용 |

E 정답 검토에서는 원본 recipe와 E를 먼저 보고, 가능하면 두 사람이 모델 예측을 보지 않고 독립 지정한다. 의견 차이가 요구 정밀도에 비해 크면 조정 근거를 남기거나 `ambiguous`로 둔다. 사람 판단 자체가 몇 픽셀 정확하다고 자동 보장하지 않는다. 사후 성공 화면의 점을 E에 복사하려면 두 화면의 기하 관계를 독립적으로 입증해야 한다.

## 5. 첫 20쌍으로 좌표 계약을 확인한다

### 5.1 재사용할 코드와 중요한 불일치

현재 [cond 파서](../../../../workflow_3/align/cond_file.py)의 `load_cond`, `cond_for_image`, `msr_modality`와 [좌표 변환 함수](../../../../workflow_3/align/clean_align_image.py)의 `cursor_to_image`를 재사용한다. MSR은 `Scope`가 없을 수 있어 파일명이나 recipe 모드만으로 SEM이라고 단정하지 않는다. modality가 불명확하면 별도 검토한다.

**현재 문서 간 의미 불일치가 있다.** [CONTEXT.md](../../../../../CONTEXT.md)는 target point를 recipe-box center로 설명하지만, [현재 기하 코드](../../../../workflow_3/align/cond_template.py)는 image center를 align anchor로 두고 box center와의 offset을 분리한다. 기획서도 후자의 계약을 따른다. 이 가이드는 코드에 맞춰 두 좌표를 모두 보존하며, 양산 의미가 확인되기 전에는 전체 데이터에 정답을 자동 확정하지 않는다. 기준점이 box 밖인 사례는 무조건 오류로 버리지 말고 recipe 기하와 기존 guard를 점검한다.

좌표는 `(x,y)`, 크기는 `(width,height)`, OpenCV shape는 `(height,width)`다. 왼쪽 위 원점을 쓰며 유효 point는 `0 <= x < width`, `0 <= y < height`다. Recipe anchor는 현재 코드의 `w/2, h/2` 계약을 기록하고 `(w-1)/2`와 섞지 않는다.

### 5.2 좌표 후보 추출 예시

아래는 추후 수집/감사 코드 안에서 사용할 부분 예시다. 전체 수집기나 성공 검증기는 아니다.

```python
from poc.workflow_3.align.cond_file import cond_for_image, load_cond, msr_modality
from poc.workflow_3.align.clean_align_image import cursor_to_image

# target_path와 target_gray는 실제 읽은 target 이미지 경로와 grayscale 배열이다.
cond_raw = load_cond(target_path)
if cond_raw is None or cond_raw.pixel is None or min(cond_raw.pixel) <= 0:
    raise ValueError("유효한 cond.Pixel이 없어 좌표를 확정할 수 없음")
if msr_modality(cond_raw) != "sem":
    raise ValueError("SEM 확인 필요")
cond = cond_for_image(cond_raw, target_gray.shape)
if cond.crosshair_xy is None:
    raise ValueError("crosshair 좌표 없음")
x, y = cursor_to_image(cond.crosshair_xy)
h, w = target_gray.shape[:2]
if not (0 <= x < w and 0 <= y < h):
    raise ValueError("좌표가 이미지 밖에 있음")
# 여기서는 S_candidate일 뿐이다. S 성공/step 검증 후 S_verified로 승격한다.
```

`cond_for_image`는 해상도에 맞게 raw cursor를 보정하며 내부에서 정수 반올림한다. 원래 cond 값과 보정 결과를 모두 기록한다. 이후 point를 다시 정수 픽셀로 반올림하지 않는다. **이 함수는 resize 보정이지 crop 위치 복원기가 아니다.** 이미 잘려 저장된 이미지라면 crop 원점 증거가 필요하다. cond.Pixel 불일치를 무조건 “resize였음”으로 처리하지 않는다.

### 5.3 검토 그림과 왕복 변환

한 pair당 recipe 원본+box+anchor, target 원본+좌표 후보, 정제본+제거 mask, 좌표 확대 보기를 만든다. Mask는 선 제거로 손상된 범위를 나타내며 정답 자체의 신뢰도를 나타내지 않는다. [기존 정제 함수](../../../../workflow_3/align/clean_align_image.py)를 사용하되 recipe의 실제 패턴을 십자선으로 오인해 제거하지 않도록 [recipe crop 처리](../../../../workflow_3/align/cond_template.py)와 구분한다.

전처리마다 crop 원점, 실제 resize 계수, padding을 `transforms.json`에 저장한다. Pixel-center 보정이 필요한 resize API는 그 convention까지 명시하고 forward/inverse를 같은 구현으로 사용한다. 단순 좌표 affine의 예는 다음과 같다.

```text
x_model = (x_original - crop_left) * scale_x + pad_left
x_original = (x_model - pad_left) / scale_x + crop_left
(y도 동일)
```

알려진 임의 점을 변환했다 되돌렸을 때 0.01px 이내인지 수치 점검한다(좌표 계산 점검 기준, 장비 허용 오차가 아님). 화면에는 변환한 패턴과 점이 함께 맞는지도 확인한다. Crop 밖 anchor는 음수가 될 수 있으므로 0으로 clamp해서 가짜 라벨을 만들지 않는다. Target 중심 crop이나 GT 중심 crop을 사용해 어려운 탐색 문제를 제거하지 않는다. GT ROI는 별도 oracle 진단에서만 사용한다.

## 6. 학습·검증·시험 데이터를 나누는 방법

**원본 그룹을 먼저 나누고, 정제·패치 합성·증강·consensus를 그 뒤에 만든다.** 동일 원본의 crop을 train과 test에 각각 넣으면 안 된다. Label이 없는 test/E도 사전학습에서 제외한다.

| 평가 트랙 | 나누는 기준 | 답할 질문 |
|---|---|---|
| `family_v1` — 초기 주 평가 | recipe revision/패턴 계열·동일/유사 원본을 하나의 그룹으로 묶어 대략 70/15/15% | 보지 못한 패턴 계열에 일반화하는가 |
| `time_v1` — 별도 평가 | 확인된 측정 시각으로 과거 train → 중간 val → 미래 test, 측정 건 경계 유지 | 같은 recipe의 미래 외관 변화에 견디는가 |
| `equipment_v1` — 데이터 확보 후 | 특정 장비 전체를 test로 보류 | 다른 장비에 전이되는가 |

세 트랙은 서로 다른 실험이다. 소량 데이터에 세 조건을 한꺼번에 강제해 빈 split을 만들지 않는다. 각 트랙의 학습과 튜닝을 분리하며, 이미 다른 트랙의 test를 보고 선택한 모델이면 독립 test 주장을 하지 않는다. 같은 recipe가 장비만 바뀐 경우 equipment 평가에서 그 사실도 기록한다.

70/15/15는 **그룹 기준 출발 비율**이다. 이미지 단위 무작위 분할은 사용하지 않는다. 고유 계열이 적으면 개발용 group 교차 검증을 하고 최종 일반화 결론을 보류한다. 신규 holdout은 이후 확보해 잠그며 그 전에는 `test.txt`를 비워 두고 보고서에 `test_not_ready`를 기록해도 된다.

Split 파일에는 한 줄에 `pair_id` 하나를 쓴다. 동일 pair·원본 hash·근접 중복·측정 건·그룹이 경계를 넘지 않는지 검사하고 건수/그룹 수/장비/S/E 분포를 `audit.md`에 적는다. 모든 arm은 같은 평가 pair 목록을 사용하며 예측 불가도 행을 남긴다.

**Consensus 재료는 별도 통제한다.** Target 자신, 같은 이벤트의 사후 결과, 미래 S를 제외하고 reference revision·modality·배율/크기 호환성을 검사한다. 주 family 일반화 비교에서는 holdout 계열의 S를 별도 support로 추가하지 않는다. 시간 평가에서 사전 존재한 동일 recipe S를 허용하는 경우, 이를 모델 학습과 구분한 support 정책으로 명시하고 B1/D2에 동일하게 제공한다. 측정 순서를 증명하지 못하면 B1 시간 평가는 실행하지 않고 B0를 진행한다. LOO 결과는 탐색용으로 따로 남긴다.

## 7. 첫 실험의 실행 순서

### 7.1 E00 — 환경·좌표 점검

사내 실행 머신에서 코드 revision과 `uv.lock`을 고정하고 Python/PyTorch/CUDA/GPU/호스트 RAM을 기록한다. H200 등 과거 문서의 자원을 현재 가용 자원으로 가정하지 않는다. 이미지 전체를 RAM에 올리지 말고 파일 단위로 읽으며, 처음에는 batch 1·data loader worker 0으로 메모리/입출력을 측정한다.

아래는 **저장소 루트에서 현재 실행 가능한 기존 좌표 관련 테스트**다.

```bash
uv run pytest poc/workflow_3/align/test_cond_file.py poc/workflow_3/align/test_cond_template.py poc/workflow_3/align/test_clean_align_image.py
```

이 테스트와 20쌍 실물 overlay 검토를 둘 다 통과해야 한다. 합성 테스트 통과만으로 실제 cond 라벨을 검증했다고 하지 않는다. DL 모델 환경은 기존 GUI/VLM 서비스 환경에 의존성을 무작정 추가하지 말고 별도 uv 프로젝트로 고정한다. 모델/weight는 사내 반입 절차에 맞게 준비하고 artifact hash·사용 조건을 기획서 기준으로 기록한다.

### 7.2 E01 — B0/B1 기준선

기존 동작을 먼저 확인하려면 [golden 실행 가이드](../runbooks/golden_combined_office_runbook.md)를 사용한다. `poc/workflow_2/golden_eval_config.example.py`를 **실편집 파일이 없을 때만** `golden_eval_config.py`로 복사하고 `GOLDEN_ROOT`, `HISTORY_ROOT`, scale/consensus 설정 및 env 우선순위를 점검한다.

```bash
uv run python poc/workflow_2/golden_combined_eval_cond.py
```

기존 입력은 `GOLDEN_ROOT/<eqp>/<class>/<recipe>/align_img_from_*`다. 연구용 `raw/.../events/...`나 `pairs.jsonl`을 직접 지정하지 않는다. 기존 결과는 `poc/workflow_2/debug_images/golden_combined_eval_cond/<timestamp>/`에 저장된다. `no_data`는 성공 평가가 아니라 입력 부재다.

**이 명령은 기존 golden 경로 확인용이다.** 새 split·pair별 원본 좌표·시간 필터·B0/B1 공통 모집단 평가를 자동 보장하지 않는다. 특히 기존 `rcp_only` 출력이 모든 pair의 B0 비교를 뜻한다고 해석하지 않는다. 새 비교에는 manifest를 읽는 작은 오프라인 adapter가 필요하다. 기존 evaluator 형식으로 export하더라도 동일 recipe의 여러 event를 같은 파일명으로 덮어쓰지 않아야 하며, 시간별 support는 target마다 선정해야 한다.

새 adapter에서 B0(recipe+CV), B1(현재 consensus 선택/recipe 폴백)을 동일 pair 전체에 각각 실행한다. `best_xy`가 template 중심인 반환 경로에서는 `align_offset_xy * best_scale`을 한 번 적용한다. 이미 최종 anchor를 반환하는 경로에 다시 더하지 않는다. 현재 [표시 scale 계약](../../../../workflow_3/docs/study/align_display_scale_contract_260913.md)에 따라 원본 FOV·배율 정보를 보존한다. 정답 위치로 후보를 골라주는 코드는 채점 쪽에만 둔다.

### 7.3 E02 — D1 사전학습 연결성 확인

[EfficientLoFTR 공식 예제](https://github.com/zju3dv/EfficientLoFTR)는 두 영상의 대응점과 점수를 출력한다. 이는 SEM align point를 직접 출력하는 완성품이 아니다. 먼저 개발 20쌍에서 전처리·대응 좌표 환산·anchor 변환을 연결한다. Zero-shot 실패만으로 fine-tuning의 가능성을 판정하지 않는다.

권장 첫 adapter는 다음 순서다.

1. Recipe와 target의 전체 FOV 및 전처리 변환을 보존한다. 입력 크기 요구사항은 선택한 구현에 맞추고 실제 resize를 기록한다.
2. 대응점을 각 영상의 원본 픽셀계로 돌린다. Recipe box 주변의 정당한 입력 영역을 쓸 수 있으나 target GT 주변만 선택하지 않는다.
3. 먼저 이동+등방 scale의 제한된 기하 모델로 robust fitting한다. 회전은 실제 필요성이 확인되면 허용한다. 변환 행렬·inlier 수·잔차·공간 분포·허용 scale 범위를 기록한다.
4. Recipe anchor에 그 변환을 적용해 target 예측을 얻는다. 한 점의 대응이나 우연한 반복 격자 일치만으로 채택하지 않는다. 대응 부족, 공간 편중, 범위 밖, 복수 변환 경쟁이면 보류한다.
5. 신뢰도/보류 기준은 val에서만 조정한다. 기하 잔차가 작더라도 잘못된 반복 주기일 수 있어 실제 point error로 채점한다.

Target cond는 GT 추출과 오버레이 정제에 쓰일 수 있지만 **target point, GT 중심 ROI, target mask를 모델 feature나 후보 선택 힌트로 전달하지 않는다.** 정제 자체의 잔상 누설은 남으므로 §8 대조시험이 필요하다. 개발 완료 후 동일 고정 val에서 B0/B1과 비교한다.

### 7.4 E03 — Y1 소규모 감독 학습

사용 가능한 YOLO 구현·weight가 확정된 경우 진행한다. 먼저 “표식이 같으면 기준점 의미도 같은가”를 확인하고 고정 의미의 패턴 계열부터 시작한다. Reference 없이 구별할 수 없는 recipe별 기준점을 하나의 class/keypoint로 억지로 합치지 않는다.

[Ultralytics pose 데이터 형식](https://docs.ultralytics.com/datasets/pose/)은 객체 class·box·keypoint를 사용한다. 해당 구현을 선택했다면 정본 `pairs.jsonl`에서 다음 파생본을 export한다. 다른 구현이면 그 형식에 맞춰 export하고 정본을 바꾸지 않는다.

```text
adapters/ds_v001/yolo_family_v1/
├── images/train/<pair_id>.png
├── images/val/<pair_id>.png
├── labels/train/<pair_id>.txt
├── labels/val/<pair_id>.txt
└── data.yaml
```

한 keypoint·visibility 포함 형식의 가상 라벨 예:

```text
0 0.5 0.5 0.25 0.25 0.4765625 0.529296875 2
```

순서는 class, 정규화 box 중심 x/y·폭/높이, 정규화 point x/y, visibility다. 위 point는 512×512에서 `(244,271)`이다. Box는 target에서 직접 검증한 대상 영역이어야 하며 recipe box를 복사하거나 point 주변에 임의 box를 만들어 실제 object box라고 부르지 않는다. `2` 역시 예시이며, 패턴/기준점의 실제 가시성 규칙에 따라 부여한다. 미라벨 E를 빈 label 파일로 만들어 “객체 없음” negative로 학습하지 않는다.

먼저 train의 검증된 16~32쌍으로 과적합 점검을 한다. 학습 loss가 내려가고 같은 train 영상의 point를 맞히는지 확인해 label export와 학습 경로를 검증한다. 이는 일반화 성능이 아니다. 다음에는 전체 train에서 학습하고 val point error로 checkpoint를 선택한다. 처음에는 증강을 끄고, 이후 실제 가능한 작은 이동/scale·밝기 변화만 추가한다. Flip·큰 회전·mosaic를 라이브 조건 확인 없이 기본 활성화하지 않는다.

학습량 출발안은 최대 30 epoch, 개선 없는 val 5회 평가 시 중단, seed 1개로 연결 확인 후 유망 설정만 3개 seed 반복이다. 이는 제안값이며 dataset 크기·GPU 시간에 맞춰 사전에 고정한다. Learning rate는 선택한 구현의 권장 fine-tuning 설정에서 시작해 값과 근거를 기록한다. 동일 비교 예산은 GPU 시간·학습 샘플 수·튜닝 시도 수로도 보고한다.

### 7.5 E04 — 유망군만 도메인 적응

D1의 대응이 일부 성립하면 train 원본에서 오버레이/제거 영향 영역을 피한 패치를 골라 두 view와 알려진 기하 정답을 만든다. Transform·random seed·source pair·유효 영역을 저장하고, 중심 구조가 빠지는 편향과 사용 가능한 패치 비율도 집계한다. 이어 실제 S 한 점 감독을 추가하는 것은 **별도 loss/학습 adapter 개발 작업**이다. 공개 trainer에 S crosshair 하나만 넣어 dense GT처럼 학습하지 않는다.

Y1이 대략 영역을 찾지만 정밀도가 부족하면 Y2를, D1/Y2가 개선되면 동일 모델의 recipe 대 consensus인 D2를 실행한다. D1·Y1·Y2·D2를 처음부터 모두 구현하지 않는다. E02/E03의 실제 결과에 따라 후속 투자를 선택한다.

## 8. 반드시 함께 돌릴 잔상·편향 대조시험

| 대조 | 첫 설정 제안 | 저장할 결과 |
|---|---|---|
| Raw vs clean | 같은 val 쌍, 동일 모델, 두 전처리 | 좌표 오차·예측 이동량; 학습 입력과 다르면 그 사실 표시 |
| Clean 민감도 | 기본 mask와 dilate를 1px 늘린 설정 | 동일 GT에서 예측 이동량, 채택/보류 변화 |
| 합성 선 paired 시험 | 같은 무영향 패치에 선 없음/임의 선 추가/추가 선 제거 | 대응 오차와 선 위치를 따라가는 정도 |
| 표시/흔적 위주 진단 | 패턴 정보를 약화한 별도 입력 | 높은 적중이 남는지; 실성능 표와 분리 |
| 독립 E 평가 | `E_independent`만 채점, 나머지 보류 분포 | S와 E의 성능 차이·정답 없는 E 개수 |
| 쉬운 편향 기준 | target 영상 중심을 항상 출력 | S GT가 중앙에 몰려 모델 점수가 부풀었는지 |

원본과 clean의 결과 차이만으로 누설 여부를 확정하지 않는다. Target mask는 분석용 저장물이며 모델 입력이 아니다. 정답 없는 E에 대한 “정밀 적중률”도 계산하지 않는다. 잔상 변화에 예측이 크게 끌리면 데이터를 늘리기 전에 전처리·감독 설계를 다시 검토한다.

## 9. 실험 결과를 남기는 방법

`runs/<run_id>/config.json`에는 dataset/split/prep hash, arm, 코드 revision, weight hash, seed, 입력 크기, 실제 학습 설정, consensus source 정책, 선택 임계값, 환경을 저장한다. 비밀이 섞인 전체 env dump는 사용하지 않고 실험에 필요한 설정만 명시한다.

`predictions.jsonl`은 **평가 pair당 반드시 한 행**이다. 권장 필드는 `pair_id, arm, pred_xy_original, score_raw, accepted, abstain_reason, error_px, latency_ms, candidate_count, consensus_source_pair_ids`다. 예측이 없으면 `pred_xy_original=null`, 라벨이 없으면 `error_px=null`이다. 실패·보류·입력 오류를 삭제하지 않는다. 학습이 읽는 입력 manifest와 evaluator가 읽는 GT를 구분한다.

`metrics.json`과 읽기용 보고서에는 다음을 남긴다.

- 전체/유효 라벨/예측 성공/채택/보류/입력 오류 건수, 고유 recipe·패턴·측정 건수.
- 원본 target 좌표의 1·2·3·5px 적중률. 기본 적중률은 유효 라벨 전체를 분모로 하며 보류/실패도 미적중이다.
- 채택률 = 채택 수/유효 라벨 수, 채택 오답률 = 허용 오차 밖 채택 수/채택 수. 채택 0건이면 오답률은 `null`이다.
- 예측이 있는 사례의 median/p90/p95 오차와 그 분모. 보류를 뺀 오차 통계만 전체 성과처럼 제시하지 않는다.
- B1 실패→신규 적중, B1 적중→회귀를 각각 같은 pair 기준으로 집계. 같은 오답 허용 수준에서 coverage-risk 비교.
- S/E·패턴 계열·장비·배율별 결과, 큰 오답 사례, preprocessing 포함 end-to-end 지연과 모델 추론 지연.

허용 오차 `t`, 큰 오답 기준 `E_large`는 장비 요구로 정하기 전 `null`로 둔다. 탐색 때 1/2/3/5px 곡선을 보고하되 임의로 3px를 장비 합격 기준으로 확정하지 않는다. 최종 신뢰구간은 이미지 독립 가정 대신 recipe/패턴 그룹 단위 bootstrap을 사용하고 그룹 수가 적으면 한계를 적는다. 최종 test는 모델·전처리·임계값을 잠근 뒤 한 번 평가한다.

첫 결과 보고서의 결론은 “S_verified 몇 쌍에서 기준선 대비 신규 적중 몇 건, 회귀 몇 건, E 검증은 몇 건, 잔상 민감도는 어느 정도”로 쓴다. 오프라인 좌표 적중률을 장비 복구율로 바꾸어 표현하지 않는다.

## 10. 지금 있는 것과 다음에 구현할 것

| 구분 | 현재 상태 | 다음 작업 |
|---|---|---|
| cond 파싱·크기 보정·정제 | 기존 함수/테스트 있음 | 수집 감사에서 재사용 |
| golden CV 평가 | 기존 명령/런북 있음 | 기존 데이터 입력·scale 계약 확인 |
| 사내 원본 일괄 수집 | 실제 접근 경로·구현 확인 필요 | 작은 export부터 이미지/cond/증거 묶음 확보 |
| 위 연구 폴더·manifest | 이 가이드의 제안 규격 | inventory + pair builder + 검토 export |
| 누설 없는 고정 split·시간 support | 실험 요구사항 | 중복/그룹 검사와 target별 과거 필터 구현 |
| B0/B1/D1 공통 원본 좌표 평가 | 신규 adapter 필요 | 같은 pair 목록 → 예측 JSONL → 공통 채점 |
| YOLO export/training | 구현·weight 선택 후 | target box 감사 → 과적합 점검 → val 학습 |
| D1 합성/실제 point fine-tuning | 별도 구현 필요 | E02 결과가 유망할 때 loss 연결 |

구현 순서는 **① inventory/pair 감사 → ② split·공통 채점과 B0/B1 → ③ D1 adapter → ④ 조건이 갖춰진 Y1**을 권장한다. 새 실험 코드는 `poc/workflow_2/`의 연구 경계 안에서 만들고 운영 workflow를 수정하거나 click/move를 연결하지 않는다. 아직 없는 `train_sem.py` 같은 명령을 실행 가능한 것으로 제시하지 않는다.

## 11. 첫 묶음 완료 체크리스트

- [ ] 사내 원본과 연구 저장소 경로를 정했고 운영 파일을 덮어쓰지 않는다.
- [ ] 약 10개 recipe·100쌍 후보의 이미지와 숨김 cond·step/성공 근거를 함께 보존했다.
- [ ] 이미지 중심과 box 중심의 의미 불일치를 실물·담당자 근거로 확인했다.
- [ ] 20쌍의 좌표/정제 overlay를 확인하고 나머지 라벨 등급·제외 사유를 기록했다.
- [ ] 원본·파생본·검토 기록을 분리하고 snapshot/해시를 고정했다.
- [ ] 개발 split과 최종 holdout의 역할을 정했다. 중복·시간 누설을 검사했다.
- [ ] B0/B1과 D1을 같은 pair·좌표·채점 기준으로 비교할 adapter 범위를 확정했다.
- [ ] 모델 비교와 함께 잔상 대조시험·영상 중심 baseline을 준비했다.
- [ ] 첫 보고서에 데이터 수·좌표 오차·보류·회귀·미검증 항목을 남겼다.

이 묶음이 끝나면 데이터 확대나 학습 투자 여부를 실제 실패 사례를 보며 결정할 수 있다. 샘플 이미지·cond·검토 그림은 사내에 두고 외부 공유가 필요하면 내부 기준에 맞춰 집계 결과만 사용한다.
