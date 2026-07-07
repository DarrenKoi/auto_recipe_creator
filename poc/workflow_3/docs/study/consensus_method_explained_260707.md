# Consensus 방식 쉽게 이해하기 — 성공 이미지를 모아서 align point 를 찾는 방법

작성일: 2026-07-07 · 작성자: study note (Claude) · 상태: 교육용 설명 문서 (코드 기준: `poc/workflow_3/align/consensus_*.py`)

> **질문**: consensus 방식은 성공(S) 이미지에서 어떤 정보를 어떻게 모으는가?
> 여러 장의 이미지를 어떻게 하나로 합치고, 거기서 align 정답 위치를 어떻게 뽑아내는가?
>
> 이 문서는 수식·코드 없이 동작 원리를 쉬운 말로 설명하고, 각 단계가 코드 어디에
> 있는지만 파일 경로로 연결한다.

---

## 0. 한 줄 요약

**"최근에 정렬에 성공했던 사진 여러 장을 성공 지점(crosshair)이 정중앙에 오도록 잘라서
겹친 뒤, 픽셀마다 다수결(median)을 내서 '지금 이 recipe 의 align key 가 실제로 생긴 모습'
을 만든다. 이 합성 사진의 중앙이 곧 align point 이므로, 라이브 화면에서 이 사진과 가장
닮은 곳을 찾으면 그 위치가 바로 정답이다."**

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

bench(workflow_2)에서 검증된 효과: 후보 안에 정답이 들어오는 비율(in_topk) 0.434 → 0.876,
1등으로 맞추는 비율(rank-1) 0.318 → 0.764.

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

cond.txt 에서 꺼내 쓰는 정보는 딱 두 가지다:

- **crosshair 좌표** (`!Cursor_info`) — 이 사진에서 장비가 정렬한 지점. 좌표는 이미지
  픽셀의 10배 단위(cursor frame)로 적혀 있어서 10으로 나눠 픽셀 좌표로 바꾼다.
- **modality (OM / SEM)** — 광학 이미지와 전자현미경 이미지는 생김새가 완전히 달라서
  절대 섞으면 안 되므로, 어느 쪽 사진인지 구분한다.

다운로드는 임시 폴더에 받아서 성공했을 때만 기존 폴더와 통째로 교체한다. 도중에
실패해도 이전 캐시가 깨지지 않는다.

- 코드: `align/consensus_gather.py` (`gather_success_images`), 오피스 전용 다운로더는
  `office_success_downloader`(gitignored).

---

## 3. 2단계 — 한 장씩 손질하기 (crop)

모은 S 이미지를 바로 겹칠 수는 없다. 장마다 세 가지 손질을 한다:

1. **crosshair 지우기** — crosshair 는 화면에 그려진 표시선이지 웨이퍼의 실제 무늬가
   아니다. 그대로 두면 matcher 가 무늬 대신 십자선을 찾아가므로 inpaint 로 지운다.
2. **crosshair 를 정중앙에 놓고 일정한 크기로 자르기** — 여기가 핵심 트릭이다.
   모든 crop 의 **중앙 = 그 사진의 정답 지점**이 되도록 자른다. crop 크기는 rcp
   template 과 같은 크기로 통일한다.
3. **불량 프레임 버리기** — cond 가 없거나, crosshair 좌표가 없거나, modality 를 알 수
   없거나, 크기 기준이 될 template 이 없는 사진은 합성에 넣지 않고 버린다(사유별 집계
   로그 출력). 오염된 한 장이 전체 합성을 망치는 것보다 버리는 게 안전하다.

- 코드: `align/consensus_crops.py` (`load_coregistered_crops`), crosshair 제거는
  `align/clean_align_image.py`.

---

## 4. 3단계 — 겹쳐서 합치기 (merge)

crosshair 중심으로 잘랐어도 측정마다 crosshair 가 찍힌 위치가 몇 픽셀씩 어긋난다.
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
기존 rcp template 으로 되돌린다(폴백). 부족/blur/캐시 없음/예외 등 **어떤 실패든 최악의
경우가 "검증된 기존 방식(rcp)"이라 회귀 위험이 0**. 이 설계 덕에 실전 루프에 바로
넣을 수 있었다.

- 코드: `align/consensus_template.py` (`build_consensus_template`, `ConsensusPolicy`),
  라우팅 조립은 `align/consensus_resolve.py` (`resolve_templates` — consensus 우선,
  rcp 폴백; killswitch = `ALIGN_FAIL_CONSENSUS`).

---

## 6. 5단계 — align 정답 위치 뽑기 (extract)

여기서 흥미로운 점: **좌표를 "계산"하는 단계가 따로 없다.**

2단계에서 모든 crop 을 crosshair(=정답 지점) 중심으로 잘랐기 때문에, 합성된 consensus
이미지의 **정중앙이 곧 align point** 다. 정답 위치가 이미지의 기하 구조 안에 미리
새겨져 있는 셈이다.

실전 보정(correction) 시점의 흐름:

1. 멈춘 라이브 화면을 캡처하고 OM/SEM 모드를 읽어 해당 modality 의 template
   (consensus 가 gate 를 통과했으면 consensus, 아니면 rcp)을 고른다.
2. ensemble matcher (3-채널 proposer + RRF + NCC rerank)가 라이브 화면에서 template 과
   가장 닮은 위치 `best_xy` 를 찾는다.
3. template 중앙 = align point 이므로 offset 보정 없이 (`align_offset_xy = (0,0)`)
   `best_xy` 가 그대로 정답 좌표다. (rcp box-crop template 만 매칭 중심과 align point
   가 달라서 offset 을 더한다.)
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
crosshair 지우고, crosshair 중심 crop  ← 2. crop     (consensus_crops.py)
        │
        ▼
미세 정렬(phase corr.) + median 합성   ← 3. merge    (consensus_cv.py)
        │
        ▼
3장 이상? 선명한가? → 아니면 rcp 폴백   ← 4. gate     (consensus_template.py / consensus_resolve.py)
        │
        ▼
라이브 화면 매칭 → best_xy = 정답 좌표  ← 5. extract  (correction.py + matching/engine.py)
```

## 8. 기억할 세 가지

1. **정답 좌표는 공짜로 얻는다** — crosshair 중심 crop 이라는 기하학적 트릭 덕에
   좌표 회귀나 투표 없이 "매칭 중심 = align point" 가 성립한다.
2. **co-registration 은 위치를, median 은 외형을 담당한다** — 위치 어긋남은 정렬로,
   외형 변동(공정 drift·노이즈)은 다수결로 각각 해결한다. 역할이 분리돼 있어서
   "median 이 흐리다 = 정렬 실패"라는 진단(gate)이 가능하다.
3. **실패해도 손해가 없다** — 모든 실패 경로가 rcp 폴백으로 수렴하는 pure-upgrade
   설계라서, 실전 루프에 회귀 위험 없이 투입됐다.
