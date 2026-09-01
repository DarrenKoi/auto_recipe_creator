---
status: research
date: 2026-09-02
scope: 데이터/라벨 타당성 감사 (Brief C) — 코드/문서 근거만, 오피스 실데이터 미접근
---

# [DIGEST] cond.txt 는 코드가 이미 파싱하는 무료 좌표 라벨이지만(box=주석, crosshair=GT, align point=이미지 중심+offset), golden 실측(298 recipe)이 보여준 분포는 "장수는 수백만이어도 recipe당 성공 커버리지는 얕다"는 형태다 — dominant-modality S ≥4장 recipe 는 298개 중 1개뿐이었다. E 이미지는 crosshair 가 없어(실측 0/182) 좌표 지도학습 라벨이 못 되고, 남은 실패 87%가 recall-miss(정답이 top-K 에 아예 없음)라는 2026-08 판정과 겹치면 pair-ranker(순위 재조정)보다 self-supervised pretraining(P3-I, unlabeled S+E 전부 사용 가능) 또는 proposer 자체를 학습하는 경로가 이 데이터 분포와 더 맞는다. 2026-07 문서의 우선순위(P0-E pair-ranker 먼저)는 그 시점에 "recall-miss 비율"이 지금처럼 87%로 확정되지 않았던 것과 골든 몇백 장을 전제했던 것이 배경이며, 지금은 그 전제 둘 다 바뀌었다.

---

## 0. 선행 연구 문서와의 관계 (코디네이터 델타 지시 반영)

`align_fail_vlm_deep_learning_addendum_ko.md`(2026-07-10)와 `align_fail_cv_methods_research_ko.md`(2026-07-10)를 모두 읽었다. 두 문서를 반복하지 않고, 이번 브리프(데이터/라벨 타당성)가 그 문서들의 전제를 어떻게 바꾸는지만 적는다.

- **7월 문서의 전제 A**: "golden data의 독립 target-point 라벨"(`align_fail_vlm_deep_learning_addendum_ko.md` §P0-E, 라벨 생성 절)로 positive/hard-negative를 만든다. 이때 golden 은 사람이 의도적으로 모은 항상 성공하는 recipe 소량 세트였다(`align_success_dataset_plan.md` §3, 부트스트랩 ~10 → 30 → 50 recipe 단계 계획). 이번 감사가 확인한 실측(§4)은 다르다. 그 계획대로 의도 수집이 아직 안 됐고, 대신 fail-알람 트리거로 저절로 모인 298 recipe 골든 세트가 극도로 sparse 하다(298개 중 dominant-modality S ≥4 장 recipe 단 1개). 7월 문서가 가정한 golden 수백 장의 실제 모양은 계획한 균형 잡힌 세트가 아니라 우연히 fail 근처에서 딸려온 few-shot 잔여물이었다.
- **7월 문서의 전제 B**: recall-miss(proposer wall)와 rank error(reranker 영역)의 비중이 아직 실험 중이었다(`align_fail_cv_methods_research_ko.md` §1.2 는 SEM `rank1≈0.665`까지만 확인했고, "proposer miss" 열은 방법 후보만 나열한다). 코디네이터가 준 델타 (1)은 2026-08 벤치에서 잔여 실패의 87%가 recall-miss로 확정됐다는 것이다(`common.md` 핵심 수치). `docs/study/adr/0006-*.md` 의 template-bank 결과도 같은 결론이다. in_topk 는 이겨도 rank1 이 동전던지기라 후보군엔 있는데 순서가 틀린 쪽이 아니라 애초에 후보군에 없는 쪽이 지배적이었다.
- **이 감사의 결론이 7월 문서와 갈리는 지점**: 7월 `align_fail_vlm_deep_learning_addendum_ko.md` §3 는 실행 순서를 R1(pair-ranker) → R3(frozen DINOv2) → P1-F(retrieval) → P3-I(self-supervised pretraining) 순으로 두었다(§6 "다음 실험 순서"). pair-ranker(P0-E)는 정의상 top-K 재순위 방법이고(§P0-E "성공 판정": `gt_in_topk=true` subset 의 rank-1만 본다), recall-miss 87% 를 못 건드린다고 그 문서 스스로 명시한다(§P0-E 마지막 줄: "top-K 에 없는 정답은 ranker의 실패가 아니라 proposer 한계로 별도 표에 둡니다"). 델타 (1)이 그 별도 표가 결과의 대부분임을 확정했으므로 pair-ranker를 첫 실험으로 두는 순서는 더 이상 맞지 않는다. 이 판단은 방법론(에이전트 B 영역)이지만, 데이터 쪽에서 이를 뒷받침하는 사실은 §3(라벨 품질)과 §4(분포)에 정리한다.
- **델타 (2)와의 연결**: 7월 문서 §P3-I 는 "unlabeled S/E frame 이 충분하지만 target label 이 적으면"이라는 조건부로 self-supervised pretraining 을 P3(최후순위)에 뒀다. 그 조건은 당시 미충족이었지만(golden 몇백 장 규모), cond.txt 가 붙은 수백만 장이 실제로 접근 가능하다면(`common.md` 전제) unlabeled pool 조건이 바뀐다. 다만 수백만 장이 실제로 어떤 분포인지는 이 저장소의 코드/문서로 검증된 적이 없다. §4 에서 이 간극을 명시한다. E 프레임은 좌표 라벨이 없지만(§3) self-supervised pretraining 의 unlabeled 입력으로는 그대로 쓸 수 있다. 그래서 7월 문서의 unlabeled 충분이라는 전제 자체는 이번 델타로 더 그럴듯해졌다.

---

## 1. cond.txt 스키마 확정

파서는 `poc/workflow_3/align/cond_file.py` 하나뿐이다(다른 곳에서 별도 파싱 없음).

### 1.1 파일 위치와 형식

- 이미지 `<image>.jpeg` 의 조건 파일은 `<image의 부모 폴더>/.<image 파일명>/cond.txt` 다(파일명 앞에 점을 붙인 숨김 폴더). 경로는 `cond_path_for()` 가 만든다(`cond_file.py:139-142`).
- 한 줄 형식은 `key<공백/탭>값1,값2,...` 이고 `parse_cond()` 가 읽는다(`cond_file.py:66-79`). key 는 첫 토큰, 값은 콤마 분해. 빈 줄은 무시.
- key 정규화는 앞의 `!` 제거 + 소문자다(`_norm_key`, `cond_file.py:46-48`). `Scope`/`!Cursor_info`/`!Cursor_inf` 가 모두 `raw` dict 에서 `scope`/`cursor_info`/`cursor_inf` 키로 들어간다.

### 1.2 키 목록

| key(정규화) | 의미 | 파서 처리 | 근거 |
|---|---|---|---|
| `scope` | `OM`/`SEM`(+`OMDF` 등 변형) — **rcp cond 에만 있음** | `CondInfo.scope`, `is_om`/`is_sem` 프로퍼티는 대소문자 무관 부분일치(`"SEM" in scope.upper()`) | `cond_file.py:38-43,81` |
| `pixel` | 이미지 크기 `(width, height)` | 토큰 2개 모두 int 파싱 성공해야 채워짐, 아니면 `None` | `cond_file.py:83-86` |
| `!cursor_info`/`!cursor_inf`(접두 매칭) | crosshair + white box 좌표 한 줄 | `elements[4],[5]` = crosshair(cx,cy), `elements[6],[7],[8],[9]` = box(left,top,right,bottom). 둘 다 `-1` 이면 "없음"으로 처리 | `cond_file.py:22-24,88-95` |
| `magnification` | 배율 | msr cond 의 modality 추론 보조 신호로만 소비(§1.4) | `cond_file.py:178-179` |
| `!om_brightness`, `accelerating_voltage` | msr cond 의 modality 확정 키 | §1.4 | `cond_file.py:174-177` |

실데이터에서 키 철자가 흔들린다. 실제 오피스 파일은 `!Cursor_inf`(끝의 `o` 없음)를 쓴다(2026-06-08 확인, `cond_file.py:88` 주석 + `test_cond_file.py:52-57` 회귀 테스트). 파서는 `startswith("cursor_inf")` 접두 매칭으로 두 철자를 모두 수용한다.

`Scope` 는 rcp cond 에만 있다. msr(S/E) cond.txt 에는 `Scope` 키가 없다(2026-06-08 사용자 확인, `cond_file.py:154` 주석). 실측이 초기 가정을 뒤집은 자리다. `align_similarity.py` 등 msr 쪽 `Scope` 읽기 코드는 사후에 죽은 코드로 삭제됐다(`journals/260608/..._consensus-validated-and-productization-handoff.md:24`: "msr 경로의 죽은 Scope 코드 제거").

### 1.3 좌표계 (px@native → 이미지 px)

`!Cursor_info` 의 숫자는 `Pixel` 의 10배(OVERSAMPLE=10) oversample 프레임이다. 이미지 px 는 cursor 값 / 10 이다(`clean_align_image.py:28-29,40-43`, `cursor_to_image()`). 예를 들어 `test_cond_file.py:31` 의 `RCP_BOX` 는 `Pixel 512,512` + box 좌표 `1770,1770,3380,3330` 이므로 이미지 px 로는 `177,177,338,333` 이 된다. 그림 검증은 `poc/workflow_2/draw_white_box_from_numbers.py`(9-해석 grid search 스크립트)로 별도 재확인한 흔적이 있다.

해상도 불일치는 자동 보정된다. `cond.Pixel` 이 실제 로드한 이미지 해상도와 다르면(리사이즈 저장 등) `cond_for_image()` 가 축별 스케일(`sx=w/pw, sy=h/ph`)로 box/crosshair 를 보정하고 `pixel` 을 로드 크기로 갱신한다. 멱등이라 여러 레이어가 겹쳐 호출해도 이중 보정되지 않는다(`cond_file.py:106-136`). 이 보정을 거치지 않으면 모든 프레임에 동일하게 걸리는 계통 오차가 생긴다고 주석이 명시한다(`cond_file.py:112-114`). **학습 데이터 추출 시 반드시 이 함수를 거쳐야 한다**(직접 `/10` 계산 금지).

### 1.4 msr modality 추론 (Scope 없음 → 대체 신호)

`msr_modality()` (`cond_file.py:164-185`)는 두 단계로 판정한다. 1순위는 키 존재(`accelerating_voltage`→sem, `om_brightness`→om, 확정), 2순위는 `Magnification`(< 200 → om, > 500 → sem, 그 사이/미상은 `None`=모호). 키가 배율보다 우선한다. `test_cond_file.py:180-183` 이 "om_brightness + Magnification 9000 도 om" 을 회귀 고정한다.

### 1.5 파서 예외 처리

- `_to_int()` 가 파싱에 실패하면 `None` 을 반환하고 그 값이 포함된 좌표 그룹은 통째로 없음이 된다(`_present()` 가 전부 `not in (None, -1)` 을 요구, `cond_file.py:59-63`).
- cond.txt 파일 자체가 없으면 `load_cond()` 이 `None` 을 반환하며 예외를 던지지 않는다(`cond_file.py:145-150`).
- 값 토큰 인덱스가 부족하면(`len(tokens) <= max(idx)`) `_present()` 가 `False` 다. 마찬가지로 조용히 없음 처리.

---

## 2. rcp box 중심 vs msr crosshair — align point 공식

메모리 `project_rcp_white_box_unique_area`("align point 는 박스 중심이 아니라 이미지 중심")는 `poc/workflow_3/align/cond_template.py` 에 그대로 구현되어 있다.

### 2.1 핵심 분리(decoupled offset)

```
box_center  = _cond_box_center(box_ltrb)                      # cond_template.py:40-44
              = ((l+r)/2, (t+b)/2), 단 l,t,r,b 는 cursor_to_image() 로 이미 px 변환됨
align_offset_xy = cond_align_offset(box_ltrb, shape_hw)        # cond_template.py:47-55
              = (round(w/2 - box_cx), round(h/2 - box_cy))
```

align point 는 이미지 중심이고 `align_offset_xy` 는 그 recipe 이미지에서 box 중심이 이미지 중심으로부터 얼마나 떨어져 있는지를 담는다. 이 오프셋은 crop 방식과 완전히 분리해서 cond 기하만으로 계산한다(주석 `cond_template.py:8-11,48-52`: "crop 을 어떻게 잡든 align point 의 기하는 안 변한다"). 검출(inner-crop)이 off-center 로 튀어도 offset 값 자체는 오염되지 않는다. 이것이 이 설계의 핵심이다. 과거 버전은 검출된 inner-crop 중심을 오프셋 기준으로 썼다가 여기서 오염이 생겼다는 것이 주석의 배경 설명이다.

### 2.2 template crop과 offset의 일관성

`cond_template_crop()` (`cond_template.py:91-114`)은 box stroke 를 inpaint 로 지운 뒤, box 내부를 대칭(symmetric) inset(기본 2px)해서 crop 한다. 대칭이라 crop 중심 == box 중심이 유지되고, 그래서 `align_offset_xy` 와 정확히 맞아떨어진다(주석 `cond_template.py:92`: "대칭 inset → crop 중심 == box 중심 → cond_align_offset 과 정확히 일관").

### 2.3 매칭 시 실제 좌표 계산 (production 소비 지점)

`templates.py:load_template()` 이 이 둘을 묶어 `AlignKeyTemplate(align_offset_xy=offset)` 를 만든다(`templates.py:36,43-49`). 그다음 실제 매칭 엔진(`matching/engine.py`, 이 브리프에서 미독, 경로만 확인)이 template match 중심 좌표에 `align_offset_xy` 를 더해 최종 `best_xy` 를 낸다. 이것이 `align_fail_cv_methods_research_ko.md:28` 의 계약("template-center / live-frame pixel" 의미와 `align_offset_xy` 보정 계약)이다.

### 2.4 (rcp crop, msr image, target xy) 학습 라벨 구성 공식

브리프가 요구한 정확한 공식을 코드 그대로 정리하면:

```
1. rcp 쪽 (positive template + offset):
   gray_rcp        = load_gray(rcp_path)                          # assets.py:283-288
   cond_rcp        = cond_for_image(load_cond(rcp_path), gray_rcp.shape)   # 해상도 보정 필수
   status, reason, onorm = check_cond_box(cond_rcp.box_ltrb, gray_rcp.shape)  # cond_template.py:66-88
   if status == "skip": box 라벨 없음 → center-area crop(§2.5) 폴백 또는 이 샘플 제외
   crop_rcp, _bbox = cond_template_crop(gray_rcp, cond_rcp)        # inpaint+대칭inset
   offset_xy       = cond_align_offset(cond_rcp.box_ltrb, gray_rcp.shape)   # (dx, dy)

2. msr 쪽 (GT target — S 라벨만):
   gray_msr        = load_gray(msr_path)
   cond_msr        = cond_for_image(load_cond(msr_path), gray_msr.shape)
   if cond_msr is None or cond_msr.crosshair_xy is None:
       # E 이미지 또는 crosshair 미검출 msr → 좌표 GT 없음 (§3)
   gx, gy          = cursor_to_image(cond_msr.crosshair_xy, OVERSAMPLE)     # golden_localization_eval_cond.py:413-415 와 동일 패턴
   target_xy       = (round(gx), round(gy))     # = 이미지 px 상의 crosshair, 곧 GT align point

3. 학습 튜플:
   (crop_rcp, offset_xy, gray_msr_또는_crop_msr, target_xy)
   target_xy 는 msr 프레임 좌표계이며 rcp box center 가 아니라 crosshair 그 자체다.
   "이미지 중심 + offset" 공식은 rcp 쪽 template 을 만들 때만 쓰이고(template 의 매칭 중심을
   보정하는 값), msr 쪽 GT 는 crosshair_xy 를 직접 읽으면 된다 — 둘을 혼동해 rcp offset 을
   msr GT 에 다시 더하면 이중 보정이 된다(현재 어떤 production 코드도 이렇게 하지 않는다).
```

이 패턴은 `golden_localization_eval_cond.py:413-419`(msr GT 추출)와 `cond_template.py` 전체(rcp offset 추출)가 프로덕션과 벤치 양쪽에서 이미 쓰고 있다. 그러니 학습 데이터 추출 스크립트는 새 로직을 짜지 말고 이 두 함수(`cond_for_image`+`cursor_to_image`, `cond_template_crop`+`cond_align_offset`)를 그대로 재사용해야 한다. 중복 구현하면 좌표계 버그가 재발할 위험이 있다(`cond_file.py:112-114` 의 계통 오차 경고 참고).

### 2.5 box 라벨이 없거나 부실한 경우

`check_cond_box()` (`cond_template.py:66-88`)가 4단계로 거른다: `degenerate`(w/h ≤0) → `out_of_bounds` → `too_small`(대칭 inset 후 16px 미만) → `offset_too_far`(대각선 정규화 offset > 0.38). 이 넷은 `skip` 이고, box 라벨을 신뢰하지 않고 `centered_area_crop`(면적비 15%, `templates.py:40-42`) 로 폴백한다. `warn` 등급(offset 0.25~0.38, 또는 inner 16~24px)은 쓰되 경고를 남긴다. 학습 데이터 큐레이션에서 `skip` 케이스는 별도로 세거나 제외해야 한다. box 라벨 품질이 낮은 recipe 를 그대로 positive 로 쓰면 offset 오염이 들어간다.

---

## 3. 라벨 품질

### 3.1 S 라벨(도구 self-report) false positive 가능성

메모리 `feedback_doubt_s_labels` 에 "도구의 self-reported success 도 false-positive 가능; tool_label 은 metadata, CV 입력 금지"가 이미 프로젝트 원칙으로 박혀 있다. 코드에서 이 원칙을 구현하는 지점은 `assets.py:iter_msr_images()` 다. docstring 이 "파일명 접두 (S/E) 는 도구가 self-reported 한 라벨이라 항상 신뢰할 수 없다"(`assets.py:138-139`)고 명시하고, 그래서 이 함수는 라벨을 걸러내지 않고 경로를 그대로 흘려보내 호출자가 직접 검사하게 한다. `probe_recipe_s_counts.py:65` 도 `_tool_label(p.name) != "S"` 로 파일명 접두만 보고 카운트하며 별도 신뢰도 필터는 없다. S 라벨을 곧이곧대로 카운트하는 것이 현재 코드베이스의 실제 동작이고, 신뢰도 검증은 아직 구현되어 있지 않다. 학습 라벨로 쓸 때는 최소한 crosshair 존재 여부를 1차 필터로 걸어야 한다(S인데 crosshair 없음 = 의심 신호, `golden_localization_eval_cond.py:430`의 `label != "S" or crosshair_xy is None` 게이트가 이 최소 방어선). 이보다 강한 검증(예: matcher score 로 S 를 다시 검증)은 이 저장소에 구현이 없다. 추측 표시.

### 3.2 crosshair 는 CV matcher 의 distractor (inpaint 필요)

메모리 `project_msr_crosshair_cv_distractor` + 코드: `clean_align_image.py` 의 전체 목적이 이것이다. crosshair 는 "FOV 전체를 가로지르는 두 선"이라 매칭 신호를 오염시키므로 매칭 직전에 `cv2.inpaint(TELEA)` 로 지운다(`clean_align_image.py:71-82, 85-111`). 중요한 것은 rcp box 와 msr crosshair 를 다르게 다룬다는 점이다. box 는 테두리만 지우고 내부는 실제 내용이라 보존하지만(`cond_template.py:98-99` 주석), msr crosshair 는 선 자체가 실제 측정 결과이므로 전체를 distractor 로 지운다. 학습 데이터 파이프라인도 같은 구분을 지켜야 한다. msr 이미지를 모델 입력으로 쓸 때 crosshair 를 지울지 남길지는 태스크에 따라 다르다. 좌표 회귀/매칭 학습이면 지워야 crosshair 좌표 자체를 모델이 치팅으로 배우는 것을 막는다(현재 production 관행, `golden_localization_eval_cond.py:441-444` 가 cond 있으면 `clean_image` 로 지우고 없으면 검출+inpaint 폴백). 단, `align_fail_cv_methods_research_ko.md:53`("crosshair inpaint/removal: 원본보다 약 -2%p")은 CV matcher 재순위 실험에서 crosshair 제거가 오히려 손해였다는 결과이므로, 학습 기반 방법에 그대로 이식할 근거는 아니다(방법이 다르면 이 실험 결과가 적용되지 않을 수 있음, 추측 표시, 재검증 필요).

### 3.3 E 이미지엔 crosshair 없음 → 좌표 학습에 쓸 수 없음

실측(코드 밖 문서, CV 검출 기반, cond.txt 정착 이전): `journals/260529/..._matcher-reliability-and-diagnostics.md:29` 가 "E(fail) 이미지는 crosshair 가 거의 없음(with_crosshair=0/182)"이라고 적었다. 이후 `journals/260602/..._mi-reranker-ruled-out-contour-next.md:76-77` 는 다른(더 큰) 표본에서 `no_crosshair=127, with_crosshair=71`(검출 기반, 오검출 포함 가능)을 보고한다. 두 수치가 정확히 일치하진 않지만(검출 알고리즘/표본 차이), E는 대부분 crosshair 가 없다는 방향은 일관된다. `align_similarity.py:7-10` 은 애초 두 가설(E-type1=틀린 위치 crosshair 있음 vs E-type2=crosshair 미검출)을 세웠는데, 실측이 E-type2 압도적 우세를 보여 E-type1 코드 경로(crosshair 있는 E)는 사후 죽은 코드로 정리됐다(§1.4 근거와 동일 커밋대).

주의: 위 수치는 CV 검출기 기반이며 cond.txt 필드 기반 집계는 이 저장소 어디에도 없다(cond.txt 는 2026-06-08 이후 도입, 위 저널은 그 이전 데이터 기준일 가능성이 있음, 정확한 날짜 대조는 못 함, 표시). `cond_file.py` 자체는 E/S 를 구분하지 않고 그냥 파싱한다. 그러니 cond.txt 로도 같은 결론(E 의 `crosshair_xy` 필드가 대부분/전부 `(-1,-1)`)이 재현되는지는 오피스에서 재확인이 필요하다.

**학습에 어떻게 쓰나 (이 저장소에 구현/결정 없음, 이 브리프의 판단)**:
- **좌표 지도학습(회귀/매칭) positive 라벨**: E 는 못 쓴다. 좌표 GT 가 없다.
- **P0-E pair-ranker 의 hard negative**(`align_fail_vlm_deep_learning_addendum_ko.md:133`): "unknown: independent target 이 없는 E frame 은 supervised train/metric 계산에서 제외하거나 separate unlabeled pool 로만 씁니다" 라고 이미 그 문서가 답을 내려놨다. E 를 candidate-vs-target pair 라벨로 강제로 쓰지 말라는 뜻이다.
- **self-supervised pretraining(P3-I) 의 unlabeled pool**: 좌표가 없어도 이미지 자체(텍스처/구조)는 유효하므로 그대로 쓸 수 있다. 델타(2)가 이 옵션의 실현 가능성(unlabeled 물량)을 키운 지점이다.
- **"다음 S 와 짝짓기"(브리프가 제시한 대안) 는 이 저장소에 근거가 없다**: E 다음 순번 S 가 그 E 가 결국 어디로 정렬됐는지를 알려준다는 가정은 그럴듯하다. 하지만 `align/assets.py` 의 `iter_msr_images()`/`_pick_current_sem()` 문서 어디에도 E→다음 S 를 같은 시도의 정답으로 잇는 로직이나 그렇게 잇는 것이 타당하다는 근거가 없다. 오히려 반증 신호가 있다. `_pick_current_sem()` 은 "E* 파일 중 visit-order(A000X)가 가장 큰 것"을 현재 실패로 고르는데(`assets.py:144-157`), 여기엔 한 recipe 의 `from_msr` 궤적이 여러 시도(성공/실패 섞임)를 담을 수 있다는 전제가 깔려 있다. E 다음 S 가 그 E 를 고친 결과인지 다음 별개 측정인지는 알람 재시도 구조(`monitor/recovery_episode.py`, cooldown 재시도)를 모르면 코드만으로 확정할 수 없다. 추측 표시, 방법론 에이전트(B)나 오피스 확인 필요.

### 3.4 측정당 OM 2장 / SEM 3장, 다른 stage 위치

`golden_eval_config.example.py:35`: "수집 단위 = 측정 건수: 1건 = OM 2장 / SEM 3장(같은 마크·다른 stage 위치)". `align_similarity.py:877-878` 도 같은 사실을 재확인한다("같은 마크·다른 stage 위치라 modality 내 풀링은 그대로 타당"). 이 사실은 학습 데이터의 그룹핑 단위에 영향을 준다. 같은 측정(measurement) 이벤트 안의 여러 msr 이미지는 서로 다른 stage 위치(=다른 실제 좌표)를 찍은 것이라, 이미지 단위가 아니라 measurement 이벤트 단위로 recipe/시간을 묶어 split 해야 누수를 막는다(§4.2). 단 메모리(`project_om_sem_positions_per_measurement`)가 언급하는 "dual recipe 서 SEM(3)>OM(2)라 OM consensus 가 영영 측정 안 됐다"는 소비자 버그는 이미 per-modality 평가로 수정됐다는 이력만 있고(`20efe49` 커밋, 메모리 근거) 이 브리프가 직접 코드로 재확인하지는 않았다(범위 밖).

---

## 4. 데이터셋 구성안

### 4.1 실측된 분포 (probe 스크립트 기준, 오피스 golden 298 recipe)

출처: `probe_recipe_s_counts.py`(로직) 실행 결과가 기록된 `journals/260608/..._consensus-validated-and-productization-handoff.md:29-31,60-64`.

| 지표 | 값 | 근거 |
|---|---|---|
| recipe 디렉토리 수 | 298 | journal:27 (glob/rglob/`_collect_recipes` 일치, collector 버그 아님) |
| 고유 recipe_id(leaf 이름만) | 276 | journal:31 — **leaf 이름 충돌로 22개 유실**(다른 eqp/class 인데 같은 recipe 이름) |
| dominant-modality S ≥4장 recipe | **1개** | journal:30 |
| dominant-modality S = 정확히 3장 | 135개 | journal:30 |
| dominant-modality S = 0장(fail-only) | 151개 | journal:30 |
| recipe당 평균 S 장수 | ~2.6장 | journal:60 |

**해석**: 298개 중 절반(151개)은 성공 이미지가 아예 없는 fail-only recipe 다. 남은 절반의 대부분(135/147)이 정확히 3장에 몰려 있다. §3.4 의 "측정 1건 = SEM 3장(또는 OM 2장)"과 정확히 일치하는 숫자다. 이 golden set 은 recipe 대부분에서 성공 측정 1건만 우연히 잡혀 있다. consensus 재등록 레버가 `min_s=3`(=측정 1건, LOO 시 실질 2장)에서만 겨우 통계를 냈다는 것(§4.4.1, journal:32-34,60-62)이 이 분포의 직접 귀결이다.

**"수백만 장" 규모에서 기대할 수 있는 분포 (추정, 표시)**: golden 298개는 알람이 트리거된 recipe 주변에서 우연히 모인 표본이라 편향돼 있다(`align_success_dataset_plan.md:52-53`: "현재 align_images/ 는 align-fail 알람으로 트리거되어 수집된다. 항상 성공하는 recipe 는 알람을 울리지 않으므로 이 데이터셋에 없다"). MES 전체를 대상으로 하면 이 편향이 반전될 가능성이 높다. 정상 recipe 는 매 측정마다 계속 성공 이미지를 쌓으므로, recipe 당 S 장수 분포는 golden(대부분 0~3장)보다 훨씬 두터운 꼬리(수백~수천 장)를 가질 것으로 예상된다. 다만 이 예상을 뒷받침할 실측이나 확률모형은 이 저장소에 없다. MES 원본 규모, recipe 개수, recipe당 평균 측정 빈도 어느 것도 코드나 문서로 확인되지 않는다. 오피스에서 `probe_recipe_s_counts.py` 와 동형인 스크립트를 MES 전체 접근 경로에 대해 돌려야 실측 가능(§4.3 스펙 참고).

### 4.2 recipe 단위 split 원칙

두 문서가 독립적으로 같은 원칙을 요구한다:

- `align_fail_cv_methods_research_ko.md:190-191`("§4.3 데이터 누수와 결과 기록 방지"): "평가 대상 S frame 이 template 에 들어가면 안 됩니다"(LOO), "recipe 단위와 OM/SEM 단위로 분리한 결과를... equipment/time bucket 도 보조 표로".
- `align_fail_vlm_deep_learning_addendum_ko.md:135`("split"): "recipe 단위로 train/validation/test 를 완전히 나누고, 가능하면 equipment 와 time bucket holdout 도... 같은 recipe 의 success frame 이 train 과 test 에 섞이면 consensus 와 appearance 가 누수됩니다."

이를 이 저장소의 실제 recipe_id 문제에 맞추면 split 키는 `AlignFailAssets.recipe_id`(leaf 이름)가 아니라 `(eqp_id, class_name, recipe_name)` 3단 전체여야 한다. leaf 이름만 쓰면 §4.1 의 298→276 충돌과 같은 유실이 split 경계에서도 재현된다(`golden_consensus_eval_cond.py:335` 주석이 이미 이 실수를 명시한다: "달라도 leaf 이름이 같으면 dict 에서 덮어써져 데이터가 유실된다"). 또한 §3.4 의 measurement-event 단위(OM 2장/SEM 3장 묶음)를 split 이전에 유지해야 같은 측정의 다른 stage 위치 이미지가 train/test 로 쪼개지는 것을 막을 수 있다(같은 측정 내 이미지들은 매우 유사한 appearance 를 공유하므로 쪼개면 누수).

**consensus 오염과의 관계**: production 의 consensus 캐시(`consensus_gather.py`)는 eqp 무관, `<class>/<recipe>` 키로 최근 성공 S 를 모은다(§1.5 근거는 `consensus_gather.py:72-81`, 문서에 이미 서술). 학습 데이터를 뽑을 때 consensus 캐시가 이미 채워진 recipe 라면 consensus 캐시에 들어간 이미지와 학습·평가에 쓰는 이미지가 겹치지 않게 해야 한다(같은 LOO 원칙의 학습판).

### 4.3 오피스에서 돌릴 추출 스크립트 스펙 (제안 — 이 저장소엔 아직 없음)

기존 스크립트(`dump_cond_samples.py`, `probe_recipe_s_counts.py`, `golden_localization_eval_cond.py`)가 정확히 이런 형태의 순회+파싱을 이미 수행하므로 신규 스크립트는 그 조합이다. 이 저장소의 "인자는 파일 상단 상수" 규약(CLAUDE.md "진입점 상단 상수 블록이 인자다")을 따른다.

```
입력 루트: ALIGN_IMAGES_ROOT 계열 경로 하나(MES 전체 접근 시 별도 루트일 수 있음 — 오피스에서
           경로 확정 필요, §5). 순회는 assets.py:iter_recipe_dirs() 패턴 재사용
           (root.glob("*/*/*/align_img_from_rcp")).

한 레코드 = 한 msr 이미지(S/E 라벨 무관하게 전부 순회):
  {
    "eqp_id": str, "class_name": str, "recipe_name": str,          # split 키(§4.2)
    "measurement_event_id": str,        # A000X visit-order 묶음(§3.4) — 같은 값이면 같은 stage 세트
    "modality": "om" | "sem" | null,    # msr_modality(cond) 결과, null=모호
    "label": "S" | "E" | 기타,          # _tool_label(파일명) — 신뢰 낮음(§3.1), metadata 로만
    "msr_image_path": str,              # 이미지 파일 경로만(반출 금지 — §5)
    "target_xy": [x, y] | null,         # S+crosshair_xy 있을 때만. cursor_to_image(cond.crosshair_xy) (§2.4)
    "frame_hw": [h, w],
    "rcp_om_path": str | null, "rcp_sem_path": str | null,   # recipe 의 registered key(경로만)
    "rcp_box_status": "ok" | "warn" | "skip" | "absent",     # check_cond_box() 결과(§2.5)
    "rcp_align_offset_xy": [dx, dy] | null,                  # cond_align_offset() 결과(box status != skip 일 때만)
    "cond_pixel_mismatch": bool         # cond_for_image() 가 보정을 실행했는지(계통오차 감시용, §1.3)
  }

출력 형식: 한 줄 = 한 레코드의 jsonl (parquet 도 가능하나 이 저장소 기존 산출물은 전부
          jsonl+json — golden_localization_eval_cond.py:22 의 rows.jsonl/summary.json 패턴을
          따르는 것이 일관적). 이미지 자체는 절대 싣지 않고 **경로만** 적는다(§5 반출 금지).

집계(요약): recipe 수, eqp/class 별 recipe 수, measurement_event 당 이미지 수 히스토그램,
           dominant-modality S 장수 히스토그램(probe_recipe_s_counts.py:75-80 과 동일 형태),
           rcp_box_status 분포(skip 비율이 높으면 box 라벨 신뢰도 문제로 별도 보고),
           modality=null(모호) 비율, target_xy=null(E 또는 crosshair 미검출) 비율.

CLI 인자 없음, 오피스 실행, uv run python <script>.py.
```

이 스펙은 제안이며 이 저장소에 코드로 존재하지 않는다. probe/golden 스크립트들의 로직을 재사용해 조립할 수 있다는 것만 확인했다.

### 4.4 OM/SEM 비율과 최소 수량

- **OM/SEM 비율**: §3.4 의 측정 단위(OM 2장, SEM 3장)를 보면 recipe 가 dual-modality(둘 다 등록)일 때 원본 비율은 대략 OM:SEM = 2:3 이다. 다만 §4.1 의 SEM-dominant 편향(메모리 `project_om_sem_positions_per_measurement`: "SEM(3)>OM(2)라 OM consensus 가 영영 측정 안 됐음")이 실제 수집 파이프라인에서 SEM 쪽을 더 많이 흘렸던 이력이 있다(수정됐다고 메모리에 기록되어 있으나 이 브리프가 재확인은 못 함). 학습 데이터 추출 시 modality 별 카운트를 반드시 별도 집계해야 한다(§4.3 스펙에 포함).
- **필요 최소 수량**: 이 저장소의 유일한 정량 가이드는 CV consensus(비-학습) 맥락의 `align_success_dataset_plan.md:60-73` 하나뿐이다. 여기서 "recipe당 ≥8~10 S", "recipe 개수는 부트스트랩 10 → 사용가능 30 → 견고 50+" 를 제시한다. percentile 임계 calibration 목적의 수량이지 딥러닝 학습 목적이 아니다. 딥러닝 최소 수량에 대한 코드나 문서 근거는 이 저장소에 없다(추측 표시). 다만 §4.1 의 실측 분포(298개 중 대부분이 0~3장)를 그대로 학습 가능 recipe 수로 읽으면 골든 트리만으로는 지도학습에 부족하다는 결론은 코드·문서로 뒷받침된다. 브리프 전제("수백만 장")가 골든 트리가 아니라 MES 전체를 가리키는 것이 맞다면 그 규모 확인이 §5 의 병목이다.

### 4.5 델타 반영 — proposer 학습 vs pair-ranker 관점에서 본 데이터 요구

- **pair-ranker(P0-E, 재순위)**: recipe 당 최소 1개의 독립 target 과 그 top-K 후보(현재 consensus matcher 출력)가 필요하다. 후보(hard negative)는 CV 파이프라인을 먼저 돌려야 나온다. §4.1 의 sparse 골든만으로는 관측치가 절대적으로 적다(§0 의 델타 (1)이 지적하듯 애초에 이 방법이 겨냥하는 실패가 소수다).
- **self-supervised pretraining(P3-I) 또는 proposer 자체 학습**: `target_xy` 라벨이 전혀 필요 없는 사전학습 단계는 E 이미지를 포함한 모든 msr 이미지(§3.3)를 unlabeled pool 로 쓸 수 있다. "수백만 장" 전제가 실제로 값을 내는 지점이 여기다. §2.4 의 좌표 튜플은 이후 fine-tune 단계에서만 필요하다. 그 물량은 §4.1 실측 분포가 보여주듯 golden 트리 기준으로는 제한적이다. fine-tune 용 라벨 물량은 MES 전체 S 커버리지에 의존하므로 §5 확인 없이는 추정할 수 없다.
- 이 판단(어느 방법을 먼저 시도할지)은 방법론(에이전트 B)의 범위다. 이 문단은 §0 델타 지시에 따라 "데이터 쪽 사실이 어느 방법에 유리한 조건을 만드는가"만 짚었다.

---

## 5. 오피스 제약 (코드/문서 근거만)

- **이미지 반출 불가**: 메모리 `feedback_no_office_data_to_mac` 의 원문은 "fab 이미지 반출 불가; blind 작성 + 오피스 실행 후 텍스트(digest)로 피드백. 샘플 이미지 요청 금지" 다. 이 브리프도 그 원칙을 지켜 이미지를 요구하지 않았다. §4.3 스펙이 경로만 싣도록 한 것도 이 제약을 그대로 반영한 결과다.
- **학습은 오피스 H200 서버**: 메모리 `project_gpu_server_host_ram_16gb` 에 따르면 H200 140GB×2 를 쓰면서도 호스트 RAM 은 16GB 다. 병목은 GPU 메모리가 아니라 프로세스 수다. 대규모 이미지셋을 로드하는 학습 파이프라인이라면 이 RAM 제약이 데이터 로더 설계(스트리밍 vs 전체 적재)에 영향을 줄 수 있다. 다만 구체적 딥러닝 학습 파이프라인의 RAM 요구는 이 저장소에 근거가 없다(추측 표시, 방법론 확정 후 재검토 필요).
- **align_images 경로가 MES 출력 위치와 어긋날 수 있음**: 메모리 `project_align_images_path_mismatch_mes` 와 CLAUDE.md 의 "Root constant" 절을 보면, `ALIGN_IMAGES_DIR` 기본값은 2026-06-11 에 `poc/workflow_3/align_images` 로 옮겨 갔지만 오피스 MES 는 역사적으로 `poc/workflow_1/align_images` 에 쓴다. 이 어긋남을 고치지 않으면 코드가 읽는 루트가 비어 있어 수백만 장에 전혀 닿지 못한다. MES 에서 서버로 데이터가 실제로 어떤 경로를 타는지(직접 파일시스템 공유인지, 별도 export/copy 파이프라인인지)도 이 저장소의 코드/문서로는 확인되지 않는다. `office_success_downloader`(consensus 캐시에 최근 S 를 채우는 역할)는 `consensus_gather.py:43-68` 에 Protocol(인터페이스)만 정의되어 있고 실제 구현체는 이 저장소에 없다(gitignored, 오피스 전용. CLAUDE.md "office_* 모듈" 절, 메모리 `project_office_module_signature_skew`). **MES 원본 수백만 장을 학습 파이프라인이 쓸 수 있는 형태로 옮기는 일은 현재 이 프로젝트의 어떤 코드도 하지 않는다.** 코드로 확인되는 통로는 align-fail 알람 트리거 저장(`align_images/`)과 consensus 캐시(최근 S 몇 장만, `ALIGN_CONSENSUS_CACHE_DIR`) 둘뿐이다. 그나마도 수백만 장 전체가 아니라 알람이 울린 근처나 최근 몇 건만 다룬다.
- **결론**: 수백만 장 + cond.txt 라는 전제 자체가 이 저장소 코드 기준으로는 미검증이다. cond.txt 스키마와 좌표 공식(§1-2)까지는 코드로 확정할 수 있었다. 그러나 그 스키마가 적용되는 모집단의 규모, 접근 경로, 서버로 옮기는 방법은 오피스에서만 확인할 수 있고 현재 어떤 스크립트도 그 경로를 구현하지 않았다. 학습 착수 전 가장 먼저 필요한 오피스 작업은 학습 파이프라인 자체가 아니라 §4.3 스펙에 따른 탐색적 규모 probe(recipe 수, 접근 가능한 최대 트리 루트, 이미지 총량)다.

RESEARCH_DONE
