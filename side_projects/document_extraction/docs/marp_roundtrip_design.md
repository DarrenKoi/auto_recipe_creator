# 슬라이드 → 이미지 → 구조 추출 → Marp → 재렌더 라운드트립 설계

> 목표: PowerPoint 슬라이드를 슬라이드쇼 캡처 이미지로 만든 뒤, VLM으로 텍스트·표·차트·도형·슬라이드 구조를 추출하고, 이를 Marp(Markdown 슬라이드)로 변환해 다시 슬라이드로 렌더링했을 때 **원본과 최대한 유사**하게 복원한다. 우선순위는 **구조 충실도 우선 + 시각 보정 균형**.
>
> 제약: 외부 클라우드 VLM(Gemini/GPT/Claude) 사용 불가(사내 데이터 반출 금지). 사용 가능 모델 = 사내 자체호스팅(UI-Venus-1.5-8B, MAI-UI-8B, PaddleOCR-VL-1.5, GOT-OCR-2.0, UI-TARS-1.5-7B[비활성]) + API는 Kimi-K2.6 하나.
>
> 단계 (1) 캡처는 이미 `document_extraction/ppt_handler.py`(슬라이드쇼 모드 화면 캡처 → WebP)가 해결. 본 문서는 (2)~(4)를 설계한다.

---

## 1. 연구 결론 (deep-research, 검증됨 / 미검증 구분)

### 검증된 사실 (3-0 또는 2-1 합의)

| # | 결론 | 근거 |
|---|------|------|
| C1 | **Kimi-K2.6은 네이티브 멀티모달** — 이미지 입력 가능(MoonViT 400M 비전 인코더). 텍스트 전용이 아님. K2/K2-Thinking은 텍스트 전용이었으나 K2.5부터 비전 추가, K2.6에서 비디오까지 확장. | HF moonshotai/Kimi-K2.6, platform.kimi.ai docs |
| C2 | Kimi-K2.6 비전 API 제약: **base64만 허용**(URL 이미지 불가), png/jpeg/webp/gif, 요청 본문 100MB, 권장 해상도 ≤4K(4096×2160). **비전은 Moonshot/Kimi *Platform* 엔드포인트에서만 동작 — Kimi *Code*(coding) 엔드포인트는 이미지 입력 미지원.** | platform.kimi.ai docs |
| C3 | **PaddleOCR-VL은 반드시 2단계 `layout→crop→recognize`로 써야 한다.** PP-DocLayoutV3(레이아웃 검출 + bbox + reading order) → 0.9B VLM이 crop된 영역별 인식. **full 스크린샷에 VLM 단독 = 공식 문서가 명시한 "과도한 환각" 실패모드** (multi-column·text+graphic 혼합 레이아웃에서 특히 심함 = 슬라이드가 바로 이 케이스). 사내 관측과 일치. | PaddleOCR 공식 docs, arXiv 2510.14528 / 2601.21957 |
| C4 | **PaddleOCR-VL-1.5는 구조를 복원한다(평문 OCR 아님)**: 표→OTSL(셀/행/열, HTML·xlsx 내보내기), 차트→마크다운 표(축/값), 수식→LaTeX(inline/display 구분). 6개 태스크(OCR/수식/표/차트/Seal/Text Spotting). OmniDocBench: Table TEDS 92.76, Formula CDM 94.21. **이 셋에서 표·차트·수식 구조 복원이 명시적으로 검증된 유일 모델.** | arXiv 2510.14528 / 2601.21957, 공식 docs |
| C5 | GOT-OCR-2.0도 구조 OCR 가능(표→Mathpix-md, 수식→LaTeX, 차트→표/dict, 도형→TikZ). 단 **기하/다이어그램은 저자 스스로 "basic shapes만" 한계 인정** → 임의 슬라이드 도형을 벡터로 복원 불가. | arXiv 2409.01704 |

### 미검증 (검색에서 확인됐으나 적대적 검증 단계 미통과 — 참고용)

- **Marp의 구조적 천장**: `pptx2marp`(오픈소스 PPTX→Marp 변환기) 문서 — "이미지·도형·SmartArt·차트·애니메이션은 평문 Markdown으로 표현 불가하여 skip된다. PowerPoint 그래프는 자동으로 이미지화되지 않는다." → **시각 보정의 실체 = 복원 불가 요소를 원본에서 crop해 이미지로 재삽입**.
- **선행 연구 패턴**: SliDer + Slide2SVG (arXiv 2511.13478, "Semantic Document Derendering") — 래스터 슬라이드 이미지 → 편집 가능한 구조 표현. 핵심 패턴: **VLM은 high-level 구조를 복원하고, deterministic renderer가 세부를 채운다.** 본 설계의 근간 패턴.
- **GUI grounding의 슬라이드 일반화 의문**: GUI-Actor(MS Research) — "real-world image와 GUI image 사이 semantic gap" 때문에 GUI 학습 모델은 도메인 특화. → 슬라이드(문서 도메인) 레이아웃엔 목적 빌트인 **PP-DocLayoutV3가 우월**, UI-Venus/MAI-UI는 보조에 한정.

### 공통 리스크(검증자 주의사항)

모든 능력 수치는 **문서 스캔 도메인** 벤치마크다. **슬라이드쇼 캡처**(안티앨리어싱 렌더 텍스트, 장식 배경, 겹친 도형)에 대한 robustness는 사내 검증 필요. PaddleOCR은 crop 안 하면 환각 영역으로 진입함이 이미 확인됨.

---

## 2. 핵심 설계 결정 — "하이브리드 복원"

"구조 우선 + 시각 보정"을 한 문장으로: **마크다운으로 충실히 복원 가능한 것은 Marp 네이티브로, 불가능한 것은 원본 crop을 이미지로 재삽입한다.**

각 영역을 두 부류로 분기한다.

- **구조 복원 (Marp 네이티브 텍스트)**: 제목/부제, 본문·불릿, 번호목록, 표, 수식, 코드, 단순 2~N단 텍스트 레이아웃. → 편집 가능, 재테마링 가능, 검색 가능.
- **시각 보정 (래스터 재삽입)**: 차트 그래픽, 도형/화살표/다이어그램/SmartArt, 로고·그림, 정밀 절대 위치가 핵심인 장식. → 원본 캡처에서 bbox crop → 이미지로 Marp에 배치. 픽셀 동일.

> 차트는 **이중 트랙**: (a) PaddleOCR-VL이 추출한 데이터 표를 화자 노트/부록에 보존(검색·재활용용), (b) 화면 충실도는 원본 crop 이미지로. 데이터로의 완전 재렌더(Marp+chart.js 등)는 선택적 후속 단계.

---

## 3. 모델 역할 배치 (고정 셋)

| 단계 | 모델 | 역할 | 비고 |
|------|------|------|------|
| 레이아웃 검출 | **PP-DocLayoutV3** (PaddleOCR-VL 1단계) | 영역 분류 + bbox + **reading order**. 슬라이드 구조의 1차 권위. | 슬라이드 도메인엔 GUI 모델보다 목적 적합 |
| 영역별 인식 | **PaddleOCR-VL-1.5** (0.9B, 2단계) | crop별 텍스트/표(OTSL→HTML)/차트(→표)/수식(→LaTeX) **구조 복원**. | **반드시 crop 입력.** full 슬라이드 단독 금지(환각) |
| OCR 폴백 | **GOT-OCR-2.0** | PaddleOCR가 특정 영역 오인식 시 대체 OCR. | 도형/다이어그램 벡터 복원 기대 금지(basic만) |
| 구조 합성 + 코드 생성 | **Kimi-K2.6** (비전) | 영역별 evidence + 원본 캡처(저해상)를 함께 보고 **슬라이드 논리 구조 판단 → Marp Markdown 생성**. 모호 영역(어느 게 제목/본문/장식인지) tie-break. | base64·Platform 엔드포인트·≤4K. evidence 위에서 합성, 없는 값 생성 금지 프롬프트 |
| 영역 bbox 폴백(선택) | UI-Venus-1.5-8B / MAI-UI-8B | PP-DocLayout가 놓친 영역 재검출 보조. | semantic gap 있음 → 1순위 아님, 폴백만 |
| (미사용) | UI-TARS-1.5-7B | 비활성 유지 | — |

설계 원칙(사내 메모와 일치): **OpenCV/검출기가 정량 좌표·구조를, VLM은 영역 식별·모호성 해소·합성만.** Kimi가 낮은 신뢰 OCR 점수를 덮어쓰지 않게 한다.

---

## 4. 단계별 파이프라인

```
Stage 0  캡처            ppt_handler.py (기존) → page_NNN.webp  + (가능시) 슬라이드 원본 px 크기 기록
Stage 1  전처리          원본 JPEG 보관 / VLM 전송용 WebP(q90, ≤4K) / 원본은 crop 소스로 무손실 보관
Stage 2  레이아웃 검출    PP-DocLayoutV3 → regions[{type,bbox,reading_order}]
Stage 3  영역 분기        각 region을 "텍스트류" vs "래스터류"로 라우팅(아래 규칙)
Stage 4  영역별 인식      텍스트류 → PaddleOCR-VL-1.5(crop) → 구조 텍스트/표/수식
                         래스터류 → 원본에서 bbox crop 저장(차트는 추가로 PaddleOCR 차트→표 보존)
Stage 5  구조 합성        Kimi-K2.6: evidence(JSON) + 저해상 원본 슬라이드 → 슬라이드 논리 구조 + Marp Markdown
Stage 6  Marp 렌더        marp-cli → PPTX/PDF/HTML. 테마/CSS로 폰트·색·여백 시각 보정
Stage 7  검증 루프        재렌더 이미지 vs 원본 캡처 SSIM/diff → 임계 미달 영역은 crop 재삽입으로 강등 후 재렌더
```

### Stage 3 분기 규칙 (region.type → 트랙)

- 텍스트류(Marp 네이티브): `title, text/paragraph, list, table, formula, code, header/footer, page-number`
- 래스터류(crop 재삽입): `chart, figure/image, shape/diagram/SmartArt, seal/logo`, 그리고 **bbox 겹침·회전·자유배치가 큰 영역**(절대 위치 보존 필요)

### Stage 5 Kimi 입력 계약 (요지)

- 입력: ① Stage 4 evidence JSON(영역별 text/표/수식 + bbox + reading_order + 신뢰도), ② 저해상 원본 슬라이드 1장(전체 맥락), ③ 래스터 crop들의 자리표시자 경로.
- 출력: Marp Markdown 1슬라이드. 표/수식은 evidence 그대로 사용(재생성 금지). 래스터 영역은 `![](crop_path)` + 위치 디렉티브. **읽을 수 없거나 불확실하면 unknown 표기, 값 창작 금지.**

---

## 5. Marp 변환 · 시각 보정 전략

1. **테마/CSS 우선**: 폰트 패밀리·크기·색·슬라이드 배경·여백을 커스텀 Marp 테마 CSS로 원본에 맞춤(전역 시각 보정). `theme-css` 디렉티브.
2. **레이아웃**: 다단/박스 배치는 Marpit의 CSS(그리드/flex)와 `<!-- _class: -->`로 근사. 절대 위치가 본질이면 트랙을 래스터로 강등.
3. **차트/도형/다이어그램**: 원본 crop을 `![bg]`/인라인 이미지로 재삽입(픽셀 동일). 차트 데이터 표는 화자 노트/부록에 동봉.
4. **배경 일치**: 슬라이드 전체 배경이 복잡하면 원본 배경을 이미지로 깔고(`![bg](...)`) 그 위에 텍스트를 얹는 "텍스트만 재타이핑" 모드(최대 시각 충실, 일부 편집성 희생).
5. **수식**: Marp의 수식(KaTeX) 사용, PaddleOCR LaTeX 그대로.
6. **재렌더 검증(Stage 7)**: 슬라이드별 원본 캡처 vs Marp 렌더 이미지의 구조 diff. 텍스트 영역 mismatch → OCR/합성 재시도, 시각 영역 mismatch → 해당 영역을 래스터 crop으로 강등 후 재렌더. **자동 강등이 충실도의 안전망.**

> 참고 구현 출발점: 오픈소스 `pptx2marp`(단, 이건 PPTX 직접 파싱 — 본 과제는 DRM/캡처 전제라 그대로는 못 쓰고 변환·테마 로직만 차용). 아키텍처 패턴은 SliDer/Slide2SVG(VLM=구조, 렌더러=세부) 차용.

---

## 6. 한계 · 리스크

- **임의 도형/다이어그램의 벡터 복원은 이 모델 셋으로 불가**(GOT 기하 basic-only). → 래스터 재삽입이 유일 현실해. 완전 편집 가능 도형 복원은 범위 밖.
- **슬라이드 캡처 robustness 미검증**: 모든 벤치는 문서 스캔. 안티앨리어싱·장식 배경에서 OCR 저하 가능 → 사내 9장 벤치(아래)로 먼저 측정.
- **PaddleOCR 환각**: crop 누락 시 재발. Stage 2→4 crop 강제가 필수 가드.
- **Kimi 엔드포인트 함정**: coding 엔드포인트엔 이미지 입력 없음 → Platform 엔드포인트 + base64 확인. 미설정 시 Kimi는 텍스트 합성으로 자동 강등(evidence-only)해도 동작은 함.
- **절대 위치/겹침 레이아웃**: Marp 구조로는 한계 → 래스터 트랙 비중↑(시각 충실 ↔ 편집성 trade-off, "균형" 정책에 부합).
- **reading order 오류**가 합성 품질을 좌우 → PP-DocLayout reading order를 신뢰하되 Stage 7 diff로 교정.

---

## 7. 사내 컨벤션 · 운영 제약 반영

- 데이터 반출 금지: 전 단계 사내 PC에서 실행. Mac은 코드만 작성/push(샘플 이미지 반입 금지).
- print 기반 로깅(`[INFO]/[WARNING]/[ERROR]`), em-dash 금지(cp949 콘솔). `logging` 모듈 미사용.
- CLI 인자 없음: `extract.py` 상단 상수/.env로 설정(기존 document_extraction 스타일 유지).
- 이미지: 로컬 JPEG 보관, VLM 전송은 WebP(q90). Kimi는 base64 인코딩 추가.
- 기존 자산 재사용: `ppt_handler.py`(캡처), `util/screen_capture.save_webp_capped`(1MB 캡), `poc.workflow_3.vlm` 클라이언트 패턴.

---

> **구현 상태(2026-06-26): Stage 5+6+7 구현됨.** `side_projects/document_extraction/marp/`
> - **Stage 5(생성)** `generate.py`: evidence -> Marp Markdown(텍스트류 네이티브 +
>   래스터류 crop 재삽입/데이터표 대체) + `build_marp.py`(raw_evidence -> deck.md).
> - **Stage 6(렌더)** `render.py`: `build_render_args`(순수, png/pptx/pdf/html) +
>   `render_deck`(marp-cli 호출, 부재 시 graceful). marp 우선 PATH, 없으면 `npx
>   @marp-team/marp-cli`. Mac e2e 검증: 2-슬라이드 deck -> deck.001/002.png 렌더 성공.
> - **Stage 7(검증/강등)** `verify.py`: `ssim`(numpy, skimage 비의존) + `slide_fidelity`
>   (해상도 보정) + `flag_low_fidelity`(floor=0.90) + `plan_downgrade`(차트 영역 ->
>   래스터, 최후엔 슬라이드 전체 `![bg]` 래스터 = 안전망) + `apply_downgrade_plans`
>   (보정 deck 재작성) + `verify_and_downgrade`(render->score->강등 루프, I/O).
>
> 순수 결정 로직은 스모크 테스트로 검증(`test_render_smoke` 6, `test_verify_smoke` 11,
> `test_marp_smoke` 8 = 25 통과). 실제 render/score 루프는 원본 캡처가 있어야 의미가
> 있어 office 에서 돈다(없으면 graceful degrade). 남은 office 작업: 9장 미니 벤치로
> floor/crop 분기 임계 보정 + Marp 커스텀 테마(아래 §8).

## 8. 다음 단계 (검증 먼저)

1. **9장 미니 벤치**(benchmark_plan.md 재활용): 슬라이드 캡처에 대해 PaddleOCR-VL(crop) 텍스트/표/수식 recall, Kimi 비전 합성 품질, 재렌더 SSIM 측정.
2. crop 분기 규칙(Stage 3) 임계 튜닝 — 어느 type을 래스터로 강등할지 결정.
3. Kimi Platform 엔드포인트 + base64 이미지 입력 1콜 스모크.
4. Marp 커스텀 테마 1종(시각 보정). Stage 7 자동 강등 루프는 구현 완료 — office 에서
   원본 캡처로 `verify_and_downgrade` 돌려 floor=0.90 보정 + 강등 빈도 측정.
