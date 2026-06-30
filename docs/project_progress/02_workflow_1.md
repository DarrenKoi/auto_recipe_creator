# 02. workflow_1 — RCS GUI 자동화 + CCTV 캡처 PoC

> 목적: VLM으로 RCS 화면을 이해하고 GUI를 자동 조작하는 것이 가능한지, 그리고 Align Fail 발생
> 시점의 증거(CCTV 영상)를 무인으로 보존할 수 있는지를 증명합니다. **현재는 동결(frozen)** 이며
> production 경로는 workflow_3로 이전되었습니다.

근거: `docs/codes/workflow_1/`, `poc/workflow_1/`, `poc/workflow_1/build_report_pptx.py`.

## 1. 무엇을 만들었나

5단계 GUI 자동화 파이프라인입니다.

1. **RCS 로그인 & 탐색** — RCS 실행 → 로그인 → Tool List 창 진입.
2. **Align Fail 감지** — CD-SEM 알람 API를 1~2분 주기로 폴링, **ALID=9006(Align Fail)** 만 필터링.
3. **CCTV/DVR 진입** — 감지된 EQP_ID의 Tool DVR(CCTV) 창을 자동 오픈, Channel 4 확대.
4. **프레임 캡처** — 최대 8분(약 4,800 프레임)간 100ms 간격으로 챔버 내부 영상을 JPEG/WebP로 저장.
5. **알림 & 로그** — Windows MessageBox 팝업 + 누적 텍스트 로그. 동일 EQP_ID 중복 알람은
   **edge-trigger**(해제 후 재발 시에만 재알림)로 처리합니다.

## 2. 핵심 기술 — 2-stage VLM 로케이터

RCS는 legacy 클라이언트라 UIA/Win32 컨트롤 노출이 부실하여 **VLM 화면 좌표 인식이 유일하게 안정적**입니다.

- **Coarse (UI-Venus-1.5-8B)**: 전체 화면에서 타겟 UI의 대략 bbox(`bbox_1000` 좌표계)를 추출합니다.
- **Crop & Zoom**: coarse bbox에 padding을 더해 잘라내고 확대하여 디테일을 보존합니다.
- **Fine (MAI-UI-8B)**: zoom crop에서 픽셀 단위 refined 클릭점을 결정하고, crop offset을 더해 원본 좌표로 역변환합니다.
- **Verify (PaddleOCR-VL-1.5)**: 클릭·입력 후 화면을 다시 OCR하여 의도한 값이 들어갔는지 확인합니다 →
  closed-loop 자동화로 무인 신뢰도를 확보합니다.
- 구현: `vlm/ui_venus_mai_locator.analyze_window_target()`가 2단계를 오케스트레이션합니다.

추가 처리:

- **DPI 좌표 변환**: VLM 출력(이미지 좌표) → DPI 보정(125/150% 대응) → 화면 절대 좌표 → `pynput` 입력.
- **Tool 이름 정규화(OCR)**: OCR 혼동("0"↔"O" 등)을 정규화 규칙으로 보정하여 EQP_ID와 매칭합니다.
- **알람 폴링 루프**: edge-trigger 중복 제거 + look-back 윈도우로 지연 보고 알람을 포착합니다.

## 3. 데이터 자산 — align_images 파일시스템 계약

workflow_1은 align key 데이터 루트(`align_images/`)의 원조입니다. office MES가 쓰고 align 코드가 읽는 계약은 다음과 같습니다.

```
align_images/<eqp_id>/<class>/<recipe>/
├─ align_img_from_rcp/   IMAP0001.*(OM)  IMAP0002.*(SEM)   # recipe 등록 align key (office MES)
├─ align_img_from_msr/   S*/E*                              # 측정 궤적 (E=fail) (office MES)
└─ captured_img_from_rcs/ <tag>/…                           # fail 시점 캡처 + recording/ (workflow가 씀)
```

- **OM**(Optical Microscopy, 저배율·반복 패턴)과 **SEM**(고배율·sparse edge)은 별도 modality로 다룹니다.
- `cond.txt` 사이드카에 crosshair·white box 좌표 등 조건 메타데이터가 함께 저장됩니다.

## 4. 증명한 것과 한계

**증명한 것:**

- VLM 기반 coarse→fine 좌표 인식이 UIA 없이도 RCS UI를 안정적으로 클릭할 만큼 견고합니다.
- Align Fail 시점 챔버 영상을 100% 자동 보존할 수 있어 야간/주말 무인 모니터링, 원인 분석 시간 단축이 가능합니다.
- 수집된 프레임은 후속 VLM 기반 자동 진단의 학습/검증 데이터로 재활용할 수 있습니다.

**한계 / 배운 점:**

- GUI "화면 읽기"가 곧 align key "찾기"는 아닙니다 — align key 정밀 매칭은 CV의 영역입니다(→ workflow_2/3).
- 전 단계를 VLM 호출에 의존하면 비용·지연·edge-case 처리 부담이 커져 full 자동화의 경제성이 떨어집니다.

## 5. 동결(frozen) 사유와 잔류물

- production RCS 자동화는 유지보수 일원화를 위해 **workflow_3로 이전**했습니다.
- workflow_1에는 **CCTV/DVR 경로와 초기 실험 스크립트만 잔류**하며, 한동안 `align_images` 데이터 루트
  역할도 겸하고 있습니다(→ workflow_3로 이전 진행 중, [05_status_roadmap.md](05_status_roadmap.md) 참조).
