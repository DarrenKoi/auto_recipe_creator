# 저장 이미지와 live SEM box의 scale 계약

2026-09-13. 공용 구현은 `align/matching/engine.py`의 `template_frame_scale`과
`frame_scales`이며, workflow_2 golden과 workflow_3 보정이 함께 사용한다.

## 원본 FOV와 template crop

`AlignKeyTemplate.source_wh`는 **crop 전 전체 FOV 이미지의 실제 로드 크기**다.
`raw_image.shape`는 매칭할 crop 크기이고, `cond.Pixel`은 cond 좌표의 기준 크기다.
세 값은 같다고 가정하지 않는다. `source_magnification`은 원본 cond의 배율이며,
consensus 재료의 호환성 검사에 사용한다.

같은 배율에서 저장 FOV 전체와 live 영상 FOV 전체가 대응한다는 조건 아래:

```text
base_scale = live_FOV_width / source_image_width
pixel_scale = base_scale × relative_scale
align_point = matched_crop_center + round(align_offset × pixel_scale)
```

예: 512px 원본에서 128px key를 잘랐고 live FOV가 1024px이면, key의 예상 폭은
256px이다. crop 128px을 live 전체 1024px으로 확대하면 안 된다.
물리 FOV 상수나 nm 단위를 이 비율 계산에 도입하지 않는다.

## 각 경로의 처리

- **PRIMARY**: 기존 `(0.7, 0.85, 1.0, 1.2, 1.4)`를 잔여 scale band로 사용한다.
  실제 matcher에는 `base × band`를 전달한다. `best_scale`과 offset은 원본 template
  픽셀 기준 그대로다. tiny-scale 가시성 게이트는 `best_scale / base`를 검사한다.
  따라서 단순히 화면이 0.5배 작다는 이유로 zoom-out 상태로 오판하지 않는다.
- **grid**: template과 offset을 `base`로 한 번 resize한 뒤 실제 판독 배율비
  `cur_mag / reg_mag`로 탐색·confirm한다. 최소 key 크기는 정규화된 **crop의 짧은 변**을
  기준으로 계산한다. sweep/confirm 좌표에 offset을 반영하며, 두 번 곱하지 않는다.
  zoom-out을 생략해 등록 배율 최근접 단을 선택할 때도 판독값으로 scale과 FOV를 계산한다.
  판독 실패 시 matcher/pan 전에 `mag_unreadable`로 degrade한다.
- **legacy live search**: 동일 표시 비율을 wide band에 적용한다. 기존 pan/zoom 예산과
  ORB 확인 게이트는 유지한다.
- **feasibility 진단**: 검출된 live box ROI에만 표시 비율을 적용한다. box가 없는 전체
  창 캡처를 FOV로 간주하지 않는다.
- **golden localization/consensus**: 같은 helper로 기존 비교 band를 표시 픽셀로 환산한다.
  offset과 위치 오차 허용폭도 표시 픽셀에 맞춘다. 기존 엔진 API의 `scales` 자체는
  여전히 절대 pixel scale이므로 후보 scale을 재채점할 때 다시 환산하면 안 된다.
  `align_similarity`의 ROI는 호환되는 modality의 race winner 크기를 사용한다.
  truth sweep도 geometry가 불일치하는 modality만 제외하여 다른 modality의 평가를 보존한다.

## consensus와 metadata 부재

workflow_3의 sizing은 workflow_2와 같은 중앙 면적 15% crop이다. 과거에는
`cond_box_crop=False`가 원본 전체를 반환하는데 이를 center crop으로 오해해
전체 크기로 S를 잘랐다. 지금은 전체 FOV metadata를 유지하면서 별도로 center crop한다.

고정-px median pool에는 기준 recipe와 **실제 이미지 크기가 같은 S**만 넣는다.
기준 배율을 알 때는 S의 배율도 알려져 있고 같아야 한다. 불일치/미상은
`source_size_mismatch`, `magnification_mismatch`, `missing_magnification`으로 기록한다.
다른 해상도나 배율의 S를 이번 변경에서 resize/warp하지 않는다. 재료 부족/blur는
기존대로 rcp로 폴백한다. 기준 배율 자체가 없는 기존 자료는 물리 배율 일치를 보장하지 못한다.

원본 크기 metadata가 없는 수동/과거 template은 PRIMARY와 정적 평가에서 기존 pixel band를
유지한다. grid는 crop 폭을 원본 FOV 폭으로 추정하지 않고 **장비 조작 전에**
`missing_source_geometry`로 degrade하여 기존 검색 경로로 넘긴다.
가로·세로 표시 비율 차이가 10%를 넘으면 등방 scale 가정을 거부한다. 이 한계는
letterbox나 잘못 잘린 ROI를 임의 stretch해서 맞추지 않기 위한 것이며, 장비 보정값은 아니다.
PRIMARY는 `escalated_invalid_geometry`로 무조작 보류하며 엔지니어 확인을 요청한다.
legacy 검색도 첫 zoom 이전에 검사하고 `invalid_source_geometry` 사유로 중단한다.
feasibility는 `invalid_geometry`와 `geometry_errors`를 남기며 좌표를 제시하지 않는다.

## 검증과 오피스 확인

회귀 테스트: `align/test_display_scale.py`, `../workflow_2/test_display_scale.py`.
0.5배/2배 PRIMARY 위치·offset, 여러 표시 크기의 crop 정규화, 실제 matcher의
grid sweep → 등록 배율 confirm, crop 기준 zoom-out 선택, metadata 부재 시 무조작,
consensus sizing/호환성, golden 후보·offset 환산을 검증한다.

오피스에서는 다음을 확인해야 한다.

1. 검출 ROI가 테두리·라벨·letterbox를 제외한 전체 영상 FOV인지.
2. 원본 FOV와 live FOV가 같은 종횡비인지, paused 배율이 등록 배율과 같은지.
3. `paused_match`의 `source_wh`, `frame_wh`, `base_scale`, `best_scale`,
   `relative_scale`, `scale_pinned`와 debug overlay가 실제 key를 설명하는지.
4. grid의 source/frame 크기, base, 탐색·복귀 배율, offset 적용 좌표가 맞는지.
5. consensus drop 사유와 rcp 폴백 비율이 과도하게 늘지 않는지.

기존 golden 수치와 이번 결과는 scale/consensus 입력 계약이 달라 구분해 기록한다.
**rank1은 해당 오프라인 입력에서의 위치 적중률이다.** live 검출 정확도나 장비 복구율,
실전 성능의 상한을 보장하지 않는다. 장비 복구 성공은 별도 Recovery Verification이 필요하다.
