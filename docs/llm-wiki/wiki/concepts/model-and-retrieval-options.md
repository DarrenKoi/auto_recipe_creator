---
tags: [concept, model-evaluation, retrieval, vlm]
level: intermediate
last_updated: 2026-05-02
status: in-progress
owner: 대영
sources: [
  raw/journals/260407/260407_075054-model-potential-research.md,
  raw/journals/260407/260407_081348-model-option-decision-report.md
]
---

# Model and Retrieval Options

> InternVL3-8B, DINOv2, NV-DINOv2, Qdrant를 같은 층위의 대체재가 아니라 reasoning, visual embedding, retrieval service 계층별 후보로 나누어 보는 판단이다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)

## 왜 필요한가? (Why)

- 현재 GUI grounding/OCR stack과 embedding/retrieval stack은 역할이 다르므로, 후보 모델을 단순 spec 비교가 아니라 시스템 계층별 적합성으로 평가해야 한다. (source: raw/journals/260407/260407_075054-model-potential-research.md)
- H200 2장 기준으로 대부분 후보의 적재성 자체보다 planner/verifier 품질, retrieval 품질, 운영 복잡도 같은 적용 관점이 더 중요하다고 정리되었다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)

## 핵심 개념 (What)

### 정의

- model option evaluation은 후보 기술을 기존 `UI-Venus`, `UI-TARS`, `CLIP`, `FAISS`의 직접 대체 여부와 보강 계층으로 분리해 판단하는 절차다. (source: raw/journals/260407/260407_075054-model-potential-research.md)

### 관련 용어

- `InternVL3-8B`: GUI click grounding 직접 대체보다 screen summarization, next-step reasoning, verification 보강 후보로 평가되었다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)
- `DINOv2`: 현재 `CLIP` 기반 image embedding 경로와 병렬 비교하기 쉬운 image-to-image retrieval 후보로 평가되었다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)
- `NV-DINOv2`: NVIDIA TAO/DeepStream/TensorRT 운영 체인과 domain adaptation 전략을 감수할 때 의미가 큰 장기 후보로 정리되었다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)
- `Qdrant`: `MongoDB + FAISS`를 즉시 전면 교체하기보다 공유형 retrieval service와 hybrid search 수요가 생길 때 PoC할 후보로 정리되었다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)

### 시각화 / 모델

```text
GUI grounding: UI-Venus / UI-TARS
Reasoning and verification: InternVL3-8B candidate
Image embedding: CLIP baseline -> DINOv2 candidate
Retrieval store: MongoDB + FAISS baseline -> Qdrant candidate
Enterprise domain adaptation: NV-DINOv2 candidate
```

## 어떻게 사용하는가? (How)

### 최소 예제

```text
1. DINOv2 small PoC next to CLIP embedding path
2. InternVL3-8B planner/verifier benchmark on RCS screenshots
3. Qdrant small collection schema PoC when retrieval service demand appears
4. NV-DINOv2 only after NVIDIA operations strategy is accepted
```

이 우선순위는 저널의 최종 권고 순서를 요약한 것이다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)

### 실무 패턴

- `DINOv2`는 유사 화면 검색 정확도, 상태 클러스터링 품질, 처리 속도를 `CLIP` 경로와 비교한다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)
- `InternVL3-8B`는 click coordinate grounding보다 화면 요약, 다음 단계 reasoning, verification prompt로 평가한다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)
- `Qdrant`는 `frame_id`, `video_id`, `step_name`, `window_title`, `ocr_text`, `image_dense`, `ocr_sparse`를 포함한 작은 collection schema부터 시작한다. (source: raw/journals/260407/260407_081348-model-option-decision-report.md)

### 주의사항 / 함정

- `DINOv2`는 CLIP처럼 텍스트-이미지 정렬을 전제로 한 모델이 아니므로 text query 기반 image search를 단독 대체한다고 보면 안 된다. (source: raw/journals/260407/260407_075054-model-potential-research.md)
- `Qdrant`는 GPU에 적재하는 모델이 아니라 vector database이므로 H200 VRAM보다 CPU, RAM, NVMe, 운영 방식이 더 중요한 판단 축이다. (source: raw/journals/260407/260407_075054-model-potential-research.md)

## 참고 자료 (References)

- 원본 메모: [260407_075054-model-potential-research.md](../../raw/journals/260407/260407_075054-model-potential-research.md)
- 원본 메모: [260407_081348-model-option-decision-report.md](../../raw/journals/260407/260407_081348-model-option-decision-report.md)
- 관련 컴포넌트: [work2-vlm-routing.md](../components/work2-vlm-routing.md)
