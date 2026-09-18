# Consensus 방식 쉽게 이해하기 — 성공 이미지를 모아서 align point 를 찾는 방법

작성일: 2026-07-07 · 작성자: study note (Claude) · 상태: 교육용 설명 문서 (코드 기준: `poc/workflow_3/align/consensus_*.py`)

수정: 2026-09-18 — 기본 consensus도 등록 box 내부만 합성·매칭하고 align offset을 보존한다.

> **질문**: consensus 방식은 성공(S) 이미지에서 어떤 정보를 어떻게 모으는가?
> 여러 장의 이미지를 어떻게 하나로 합치고, 거기서 align 정답 위치를 어떻게 뽑아내는가?
>
> 이 문서는 수식·코드 없이 동작 원리를 쉬운 말로 설명하고, 각 단계가 코드 어디에
> 있는지만 파일 경로로 연결한다.

---

## 0. 한 줄 요약

**"최근 성공 사진마다 crosshair를 기준으로 등록 box 영역을 찾아 잘라낸 뒤 median으로
합친다. 라이브 화면에서 그 무늬를 찾고, box 중심에서 align point까지의 offset을 더한다."**

이전에는 등록 box 설정이 켜져 있어도 consensus만 큰 crosshair-중심 crop으로 바뀌었다.
key가 화면 가장자리에 보이지만 그 주변 문맥은 화면 밖인 경우에는 전체 template이 들어갈
자리가 없어 놓칠 수 있었다. 현재 기본 경로는 등록 box 크기를 유지한다. 아래 변경은
합성 위치 이동 회귀로 확인했으며, 실제 Windows/SEM 데이터에서의 성능은 별도 검증 대상이다.

---

## 1. 왜 이런 방식이 필요한가

기본 방식은 recipe 에 등록된 align 이미지(rcp)를 template 으로 써서 라이브 화면과
매칭한다. 문제는 rcp 이미지가 **등록 시점의 모습**이라는 데 있다. 공정이 흐르면서
웨이퍼 패턴의 밝기·대비·미세 형상이 조금씩 달라지기 때문에, 옛날 사진 한 장으로는
매칭 점수가 낮아지거나 엉뚱한 곳(decoy)에 걸리는 일이 생긴다.

반면 **최근에 측정이 성공했던 이미지(S 이미지)** 는 두 가지를 동시에 알려준다:

1. **지금 시점의 실제 모습** — 공정 변화가 이미 반영된 최신 외형.
2. **정답 위치** — 성공한 측정에서 장비가 실제로 정렬한 지점이 crosshair 로 찍혀 있다.

즉 S 이미지 한 장 한 장이 "정답 좌표가 표시된 최신 사진"이다. consensus 방식은 이걸
여러 장 모아서 한 장짜리 '대표 사진'을 만든다.

과거 center-crop bench(workflow_2)의 결과는 in_topk 0.434 → 0.876,
rank-1 0.318 → 0.764였다. 새 box consensus의 성능 수치로 인용하면 안 된다.

---

## 2. 1단계 — 모으기 (gather)

**무엇을**: recipe 별로 최근 성공 측정 이벤트 몇 건(기본 4건, 오피스 수집 정책은
modality 별 최근 8~10장 rolling)의 S 이미지와 그 짝꿍 cond.txt 파일.

**어디에**: eqp(장비) 와 무관하게 recipe 단위로 한 폴더에 모은다. 같은 recipe 는 어느
장비에서 측정했든 같은 모습이므로 pool 을 공유하는 게 유리하다.

```
<ALIGN_CONSENSUS_CACHE_DIR>/<class>/<recipe>/events/<event_id>/
├─ S0001.jpeg                 # 성공 측정 사진
└─ .S0001.jpeg/cond.txt       # 숨김 폴더 안의 측정 조건 파일
```

cond.txt에서 사용하는 주요 정보:

- **crosshair 좌표** (`!Cursor_info`) — 이 사진에서 장비가 정렬한 지점. 좌표는 이미지
  픽셀의 10배 단위(cursor frame)로 적혀 있어서 10으로 나눠 픽셀 좌표로 바꾼다.
- **modality (OM / SEM)** — 광학 이미지와 전자현미경 이미지는 생김새가 완전히 달라서
  절대 섞으면 안 되므로, 어느 쪽 사진인지 구분한다.
- **Pixel / Magnification** — 좌표 정규화 및 원본 해상도·배율 호환성 검사에 쓴다.
- **등록 box와 align point** — RCP에서 box 크기와 `align_offset_xy`를 정한다.

다운로드는 임시 폴더에 받아서 성공했을 때만 기존 폴더와 통째로 교체한다. 도중에
실패해도 이전 캐시가 깨지지 않는다.

- 코드: `align/consensus_gather.py` (`gather_success_images`), 오피스 전용 다운로더는
  `office_success_downloader`(gitignored).

---

## 3. 2단계 — 한 장씩 손질하기 (crop)

모은 S 이미지를 바로 겹칠 수는 없다. 장마다 세 가지 손질을 한다:

1. **crosshair 지우기** — crosshair 는 화면에 그려진 표시선이지 웨이퍼의 실제 무늬가
   아니다. 그대로 두면 matcher 가 무늬 대신 십자선을 찾아가므로 inpaint 로 지운다.
2. **등록 box 영역 자르기** — `crop 중심 = S crosshair - RCP align_offset_xy`.
   크기는 RCP template의 box 내부 크기로 통일한다. box가 없거나 검증에서 거부되면
   RCP의 center-area fallback 크기를 쓴다. `ALIGN_FAIL_COND_BOX_CROP=0`일 때만
   종전 consensus center-area 방식을 유지한다.
3. **불량 프레임 버리기** — cond 가 없거나, crosshair 좌표가 없거나, modality 를 알 수
   없거나, 크기 기준이 될 template 이 없는 사진은 합성에 넣지 않고 버린다(사유별 집계
   로그 출력). 해상도·배율이 맞지 않거나 crop이 프레임 밖으로 잘리는 사진도 제외한다.
   잘린 crop을 옮기거나 늘려 채우면 형상과 좌표가 틀어지므로 사용하지 않는다.

- 코드: `align/consensus_crops.py` (`load_coregistered_crops`), crosshair 제거는
  `align/clean_align_image.py`.

---

## 4. 3단계 — 겹쳐서 합치기 (merge)

crosshair 기준으로 box를 잘랐어도 측정마다 위치가 몇 픽셀씩 어긋날 수 있다.
그냥 평균을 내면 어긋난 만큼 사진이 번진다(blur). 그래서 두 단계로 합친다:

1. **미세 정렬 (co-registration)** — 사진들을 겹치기 전에 서로 살짝 밀어서 무늬가
   정확히 포개지게 맞춘다. phase correlation 이라는 방법으로 "이 사진을 몇 픽셀
   옮기면 기준 사진과 가장 잘 겹치는가"를 소수점 픽셀 단위까지 계산한다.
   기준 사진은 현재 세트의 median 으로 잡고, 이 과정을 2회 반복해 기준을 다듬는다.
   단, 이동은 항상 **원본에서 한 번만** 적용해서 보간 blur 가 누적되지 않게 한다.
   계산된 이동량이 비정상적으로 크면(변의 30% 초과) 잘못 잡힌 것으로 보고 건너뛴다.

2. **픽셀별 다수결 (median)** — 정렬된 사진들을 쌓아 놓고, 같은 위치 픽셀들의
   **중앙값**을 취해 한 장을 만든다. 평균이 아니라 median 을 쓴다는 것이 곧
   "consensus(합의)"라는 이름의 유래다: 대부분의 성공 사진에 공통으로 있는 무늬만
   살아남고, 한두 장에만 있던 노이즈·이물·특이 측정은 다수결에서 밀려 사라진다.

이렇게 만든 한 장이 **consensus 이미지** = "요즘 이 align key 가 실제로 생긴 모습의
대표 사진"이다.

- 코드: `align/consensus_cv.py` (`coregister_crops`, `_consensus`) — workflow_2 bench 에서
  검증된 로직의 bit-parity 포팅이라 재구현 금지.

---

## 5. 4단계 — 믿어도 되는지 검사하기 (gate)

합성이 항상 성공하는 건 아니다. template 으로 쓰기 전에 세 가지를 검사한다:

- **장수**: 같은 modality 의 crop 이 최소 3장 이상인가 (`insufficient_s`).
- **선명도 1**: consensus 의 edge 밀도가 개별 crop 들의 중앙값 대비 70% 이상인가.
- **선명도 2**: Laplacian 분산(초점 지표)이 개별 crop 대비 50% 이상인가 (`blurry`).

선명도가 떨어졌다는 건 3단계의 미세 정렬이 잘 안 됐다는 신호다. 검사에 하나라도
걸리면 consensus template 을 **만들지 않고 None 을 반환**하고, 호출부는 그 modality 를
기존 RCP template으로 되돌린다(폴백). 이 검사는 입력 품질을 거르는 것이며,
선명한 template이 항상 올바른 위치를 찾는다는 보장은 아니다.

- 코드: `align/consensus_template.py` (`build_consensus_template`, `ConsensusPolicy`),
  라우팅 조립은 `align/consensus_resolve.py` (`resolve_templates` — consensus 우선,
  rcp 폴백; killswitch = `ALIGN_FAIL_CONSENSUS`).

---

## 6. 5단계 — align 정답 위치 뽑기 (extract)

합성 이미지의 중심은 **box 중심**이며, align point와 같다고 가정하지 않는다.
등록 template의 `align_offset_xy`, `source_wh`, `source_magnification`을 보존한다.

실전 보정(correction) 시점의 흐름:

1. 멈춘 라이브 화면을 캡처하고 OM/SEM 모드를 읽어 해당 modality 의 template
   (consensus 가 gate 를 통과했으면 consensus, 아니면 rcp)을 고른다.
2. ensemble matcher (3-채널 proposer + RRF + NCC rerank)가 라이브 화면에서 template 과
   가장 닮은 위치 `best_xy` 를 찾는다.
3. RCP와 consensus 모두 `best_xy + align_offset_xy * best_scale`이 align 좌표다.
4. 그 좌표를 FOV 안으로 clamp 한 뒤 더블클릭(recenter) → OK 클릭으로 실제 보정한다.

- 코드: `align/correction.py` (`correct_align_fail_auto`, offset 적용은 323–326행 부근),
  matcher 는 `align/matching/engine.py`.

---

## 7. 전체 그림 (요약 파이프라인)

```
[오피스 MES]                          [workflow_3 align]
성공 측정 S 이미지 + cond.txt
        │ office_success_downloader
        ▼
recipe 별 events/ 캐시 (eqp 무관)      ← 1. gather   (consensus_gather.py)
        │
        ▼
crosshair 지우고, 등록 box 영역 crop   ← 2. crop     (consensus_crops.py)
        │
        ▼
미세 정렬(phase corr.) + median 합성   ← 3. merge    (consensus_cv.py)
        │
        ▼
3장 이상? 선명한가? → 아니면 rcp 폴백   ← 4. gate     (consensus_template.py / consensus_resolve.py)
        │
        ▼
라이브 매칭 → best_xy + scaled offset  ← 5. extract  (correction.py + matching/engine.py)
```

## 8. 기억할 세 가지

1. **box 중심과 align point를 구분한다** — crop은 key만 담고, 정답 좌표는 등록 offset으로 복원한다.
2. **co-registration 은 위치를, median 은 외형을 담당한다** — 위치 어긋남은 정렬로,
   외형 변동(공정 drift·노이즈)은 다수결로 각각 해결한다. 역할이 분리돼 있어서
   "median 이 흐리다 = 정렬 실패"라는 진단(gate)이 가능하다.
3. **입력 부족·blur는 RCP로 폴백한다** — 별도로 실제 이미지의 오탐·누락 검증이 필요하다.
