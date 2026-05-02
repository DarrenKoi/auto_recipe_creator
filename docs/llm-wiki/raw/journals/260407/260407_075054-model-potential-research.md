# 모델 및 벡터 검색 기술 도입 가능성 조사 보고서

## 1. 문서 목적

본 문서는 현재 repo에서 사용 중인 VLM/OCR 및 임베딩·검색 스택을 기준으로, 신규 후보 기술의 도입 가능성과 역할 적합성을 검토하기 위해 작성되었다. 이번 조사 대상은 `InternVL3-8B`, `DINOv2`, `NV-DINOv2`, `Qdrant`이며, 각 후보를 단순 사양 비교가 아니라 현재 프로젝트 구조에 실제로 어떤 방식으로 연결할 수 있는지까지 포함해 평가했다.

조사의 관점은 크게 두 가지다. 첫째, 현재 주력 GUI grounding 및 OCR 흐름을 직접 대체할 수 있는지 여부다. 둘째, 기존 파이프라인을 유지한 상태에서 상위 판단 계층 또는 검색 계층을 강화하는 보강 수단으로서 가치가 있는지 여부다. 추가로, 사무실 환경에서 검토 중인 `H200` 2장 구성에서 각 후보의 적재 및 운영 현실성도 함께 정리했다.

본 문서의 핵심 사실 관계는 2026-04-07 기준 공식 공개 자료를 다시 확인해 반영했다. 적용 해석과 우선순위 판단은 공식 문서 내용과 현재 repo 구조를 함께 놓고 내린 실무적 해석이다.

## 2. 현행 기준 스택

현재 비교의 기준이 되는 문서는 [`docs/setup_vlms/README.md`](/Users/daeyoung/Codes/auto_recipe_creator/docs/setup_vlms/README.md), [`docs/setup_vlms/02-model-bringup-and-special-settings.md`](/Users/daeyoung/Codes/auto_recipe_creator/docs/setup_vlms/02-model-bringup-and-special-settings.md), [`docs/setup_vlms/04-operations-integration-and-benchmarking.md`](/Users/daeyoung/Codes/auto_recipe_creator/docs/setup_vlms/04-operations-integration-and-benchmarking.md)이다. 이 기준 문서와 현재 코드 구조를 함께 보면, 본 repo의 주력 VLM/OCR 스택은 `UI-Venus`를 primary full-screen grounding 모델로, `MAI-UI`를 crop retry sidecar로, `UI-TARS`를 alternate primary로 운용하는 구조이며, OCR 계층에는 `PaddleOCR-VL-1.5`와 `GOT-OCR`이 배치되어 있다.

임베딩 및 검색 경로는 별도의 축으로 존재한다. [`test/video_frame_parser/analyzer.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/analyzer.py)는 `CLIP` 기반 이미지 임베딩을 생성하고, [`test/video_frame_parser/db_handler.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/db_handler.py)는 `MongoDB + FAISS` 조합으로 메타데이터와 벡터 검색을 처리한다. 따라서 이번 후보군은 하나의 풀로 묶어 비교하기보다, "기존 grounding VLM 대체 또는 보강"과 "기존 embedding/retrieval 강화"로 구분해 보는 것이 타당하다.

## 3. 평가 관점

이번 평가는 다음 네 가지 기준으로 진행했다.

| 평가 항목 | 판단 기준 |
| --- | --- |
| 역할 적합성 | 현재 GUI grounding, OCR, planner/verifier, retrieval 중 어느 계층에 가장 자연스럽게 들어가는가 |
| 대체 가능성 | 기존 `UI-Venus`, `UI-TARS`, `CLIP`, `FAISS`를 직접 교체할 수 있는가 |
| 운영 현실성 | 현재 repo의 Python 스크립트, Flask proxy, 로컬 실험 중심 구조에 무리 없이 들어오는가 |
| 인프라 적합성 | 사무실 `H200` 2장 구성에서 추론 또는 실험 운영이 현실적인가 |

## 4. 후보별 조사 결과

### 4.1 InternVL3-8B

OpenGVLab의 Hugging Face 모델 카드 기준으로 `InternVL3-8B-hf`는 8B 파라미터급 BF16 멀티모달 모델이다. 이미지, 비디오, 텍스트를 함께 다룰 수 있으며, 공식 설명 범위는 시각 이해와 추론, tool usage, GUI agents, industrial image analysis까지 포함한다.

추가로, 공식 Hugging Face 구현 문서에서는 이 모델이 Transformers 네이티브 구현으로 제공되며, interleaved image, video, text 입력과 batched inference를 지원한다고 명시하고 있다. 이는 단일 스크린샷 분석뿐 아니라 다중 화면 비교, 전후 상태 대조, 짧은 작업 맥락을 포함한 추론 실험을 설계하기에 유리한 특성이다.

다만 현재 프로젝트에 바로 연결하는 관점에서는, 이 모델을 `UI-Venus`나 `UI-TARS`와 동일한 의미의 1차 클릭 좌표 grounding 모델로 보는 것은 맞지 않다. 현재 스택의 GUI 모델들은 직접적인 좌표 grounding 계약이 강한 반면, `InternVL3-8B`는 보다 범용적인 멀티모달 reasoning 성격이 강하다. 따라서 실무 적용 측면에서는 click grounding의 직접 대체재보다는 다음과 같은 상위 판단 계층에서 잠재력이 크다.

- 화면 상태 요약
- 현재 단계의 맥락 해석
- 다음 액션 후보 도출
- step-level verification
- 실패 원인 설명
- action 후보 우선순위화

즉, `InternVL3-8B`의 가장 현실적인 포지션은 "주 클릭 모델 교체"가 아니라 "planner/verifier 역할 보강"이다. 이 판단은 OpenGVLab 모델 카드의 설명과 현재 repo 문서에서 정의된 `UI-Venus`/`UI-TARS`의 역할을 함께 놓고 해석한 적용 평가다.

### 4.2 DINOv2

Meta의 공식 DINOv2 README 기준으로, DINOv2는 self-supervised vision foundation model이며 범용적인 visual feature를 다양한 이미지 분포와 다운스트림 작업에 재사용하는 데 초점이 맞춰져 있다. 공식 README는 이 모델이 1억 4200만 장의 무라벨 이미지로 사전학습되었고, fine-tuning 없이도 다양한 도메인에서 강한 visual feature를 제공한다고 설명한다. 이 계열은 텍스트-이미지 정렬보다는 이미지 자체의 표현력과 분포 일반화 능력에 강점이 있다.

현재 repo 구조와 연결하면, `DINOv2`는 [`test/video_frame_parser/analyzer.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/analyzer.py)의 `CLIP` 임베딩 자리에 가장 손쉽게 A/B 테스트할 수 있는 후보다. 특히 다음과 같은 image-to-image retrieval 성격의 작업에서 잠재력이 있다.

- 화면-화면 유사도 검색
- 상태 클러스터링
- 중복 프레임 제거
- 에러 화면 retrieval
- 유사 레시피 화면 검색

반면 텍스트 질의를 바로 이미지 검색으로 연결해야 하는 경우에는 주의가 필요하다. `DINOv2`는 CLIP처럼 텍스트-이미지 정렬을 전제로 한 모델이 아니므로, 텍스트 기반 검색을 유지하려면 기존 `CLIP` 또는 별도 text embedding 경로를 함께 유지하는 편이 안전하다. 따라서 `DINOv2`는 "검색 품질 향상을 위한 image embedding 강화"에는 유력하지만, "텍스트-이미지 통합 검색의 단독 대체"로 보기에는 제한이 있다.

### 4.3 NV-DINOv2

최신 NVIDIA TAO 문서 기준으로 NV-DINOv2는 self-supervised learning 계열 비전 모델로 설명되며, 라벨 없는 대규모 이미지 데이터에서 학습할 수 있고 classification, detection, segmentation 같은 다운스트림 작업에 활용 가능한 semantic-rich representation을 제공하는 방향으로 소개된다. NVIDIA 블로그 사례에서는 일반 이미지 약 7억 장 수준으로 사전 학습된 기반 모델을 두고, 이후 약 70만 장의 unlabeled PCB 이미지로 domain adaptation을 수행하는 흐름을 제시한다.

프로젝트 적용 관점에서 NV-DINOv2의 가장 큰 강점은 모델 자체보다도 배포 체인과 생태계 결합성에 있다. 사무실 환경이 NVIDIA GPU 서버 중심이고, 장기적으로 반도체 장비 화면, SEM 이미지, 공정 이미지와 같은 도메인 이미지에 대해 unlabeled adaptation까지 고려한다면, 오픈 DINOv2보다 NVIDIA 운영 체인과의 친화성이 더 높은 선택지가 될 수 있다.

그러나 현재 repo는 Python 스크립트, Flask proxy, 로컬 실험 중심 구조를 기반으로 움직이고 있다. 이 상태에서 NV-DINOv2를 도입하면 TAO, DeepStream, TensorRT 계열 운영 복잡도가 함께 따라온다. 따라서 기술 잠재력은 분명하지만, 현재 개발 방식과의 간극 및 lock-in 비용을 함께 보아야 한다. 현재 시점에서 NV-DINOv2는 "강한 장기 후보"이지 "지금 바로 가볍게 붙여볼 기본 후보"라고 보기는 어렵다.

### 4.4 Qdrant

Qdrant 공식 문서 기준으로 Qdrant는 client-server 형태의 vector database이며, dense vector뿐 아니라 sparse vector, named vector, payload filtering, HNSW 기반 검색, sharding/replication, hybrid retrieval을 지원한다. 공식 문서는 semantic search와 lexical search를 결합한 hybrid retrieval을 지원한다고 설명하며, dense + sparse + multivector reranking을 단일 collection 안에서 구성하는 가이드도 제공한다.

현재 [`test/video_frame_parser/db_handler.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/db_handler.py)는 `MongoDB + FAISS`를 분리해 사용하는 구조다. 이 구조는 단순하고 빠르게 실험하기에는 적합하지만, 검색 조건이 복합화되거나 공유형 retrieval service로 발전할 경우 데이터와 검색 레이어가 분리되어 있다는 점이 운영 부담으로 바뀔 수 있다.

Qdrant를 도입하면 화면 임베딩, OCR 키워드 sparse vector, 액션 단계 메타데이터를 하나의 검색 계층에 통합할 수 있다. 예를 들어 `frame_id`, `video_id`, `step_name`, `window_title`, `ocr_text` 같은 payload와 `image_dense`, `ocr_sparse`를 함께 저장하고, dense image retrieval과 payload filter, sparse OCR keyword search를 조합한 검색 구성이 가능해진다. 따라서 Qdrant는 현재 `FAISS`의 단순 대체재라기보다, 개인용 실험 단계 이후 coworker 공유형 retrieval service가 필요해질 때 가치가 커지는 저장소 후보로 보는 편이 맞다.

## 5. 현재 스택 대비 역할 정리

각 후보를 현재 사용 중인 구성과 직접 비교하면 다음과 같이 정리할 수 있다.

| 후보 | 현재 스택에서의 대응 위치 | 현실적인 역할 |
| --- | --- | --- |
| `InternVL3-8B` | `UI-Venus`/`UI-TARS`와 일부 겹쳐 보일 수 있음 | 직접 grounding 교체보다는 planner/verifier 보강 |
| `DINOv2` | `CLIP` 임베딩 경로와 직접 비교 가능 | image-to-image retrieval 품질 강화 또는 병행 실험 |
| `NV-DINOv2` | `DINOv2`의 enterprise/NVIDIA stack 특화 대안 | 도메인 적응과 NVIDIA 운영 체인까지 고려한 장기 후보 |
| `Qdrant` | `FAISS` 기반 검색 저장소 확장 | 공유형 retrieval service 및 hybrid search 인프라 후보 |

이 비교를 기준으로 보면, 이번 후보군은 서로 같은 층위의 경쟁 상대가 아니다. `InternVL3-8B`는 reasoning 계층, `DINOv2`와 `NV-DINOv2`는 visual embedding 계층, `Qdrant`는 저장 및 검색 계층에 해당한다. 따라서 "무엇이 가장 좋으냐"보다 "현재 시스템에서 어느 병목을 먼저 풀고 싶은가"가 우선순위를 결정한다.

## 6. H200 2장 기준 적재 및 운영 가능성

NVIDIA 공식 사양 기준으로 H200은 GPU당 `141GB HBM3e`, `4.8TB/s` 메모리 대역폭을 제공한다. 2장 구성으로 보면 총 `282GB` VRAM 풀이 확보된다. 이 기준에서 각 후보의 적재성은 다음과 같이 해석할 수 있다.

### 6.1 InternVL3-8B

공개 모델 카드 기준 `InternVL3-8B`는 `8B params`, `BF16` 모델이다. 단순 계산으로 BF16 가중치만 약 `16GB` 수준이며, 실제 추론에서는 vision tower, KV cache, activation, runtime overhead가 추가된다. 그럼에도 H200 1장에 충분히 수용될 가능성이 높고, H200 2장 구성에서는 적재 자체가 문제 될 가능성은 낮다.

운영상 실제 변수는 메모리 적재 여부보다도 긴 컨텍스트, 다중 이미지 배치, 동시성 증가에 따라 커지는 KV cache다. 따라서 H200 2장 환경에서는 "적재 가능 여부"보다 "단일 GPU 고정 운용이 나은지, tensor parallel 분산이 필요한지"를 운영 정책으로 정하면 된다.

### 6.2 DINOv2

Meta 공식 저장소 기준으로 DINOv2는 `ViT-B/14` 86M, `ViT-L/14` 300M, `ViT-g/14` 1.1B 파라미터 variant까지 공개되어 있다. BF16 추론 가중치 기준으로 대략 `0.2GB`, `0.6GB`, `2.2GB` 수준이므로, 어떤 공개 variant를 선택하더라도 H200 1장에 매우 여유 있게 들어간다.

따라서 H200 2장 구성에서 DINOv2의 핵심 이슈는 적재 가능성이 아니라 처리량이다. 대량 배치 임베딩, 병렬 실험, 혹은 파인튜닝 계열 작업에서 처리 속도나 실험 반복 속도를 얼마나 끌어올릴 수 있는지가 더 실질적인 운영 포인트다.

### 6.3 NV-DINOv2

NVIDIA 공개 페이지에서는 NV-DINOv2의 고정 파라미터 수나 메모리 프로파일을 일괄 표로 명시하지 않는다. 다만 공식 설명상 localization/classification backbone과 domain adaptation 흐름에 초점이 있으므로, 일반적인 backbone 추론은 H200 2장에 충분히 들어갈 가능성이 높다.

문제는 학습 및 적응 단계다. self-supervised 재학습이나 대규모 domain adaptation은 입력 해상도, batch size, adapter/head 구성에 따라 메모리 사용량이 크게 달라질 수 있다. 따라서 추론 관점에서는 현실성이 높지만, 적응 학습 관점에서는 recipe 세부 조건 확인 없이 "무조건 충분하다"고 단정할 수 없다.

### 6.4 Qdrant

Qdrant는 모델이 아니라 vector database이므로 H200 2장에 "적재"하는 대상이라고 보기는 어렵다. 공식 FAQ 기준으로 Qdrant는 기본적으로 CPU 중심으로 동작하며, 최신 문서 기준 v1.13 이상에서는 GPU-accelerated indexing을 추가로 활용할 수 있다. 따라서 Qdrant 도입 여부는 GPU VRAM보다 CPU, RAM, NVMe, 서비스 운영 편의성의 문제에 더 가깝고, GPU를 쓰더라도 주 용도는 검색 serving 자체보다 indexing 가속에 가깝다.

즉, H200은 Qdrant 자체보다 Qdrant에 적재할 임베딩을 생성하는 `InternVL3-8B`, `DINOv2`, `CLIP` 같은 모델 쪽에서 의미가 크다.

## 7. 종합 판단

이번 조사 결과를 실무 관점에서 압축하면 다음과 같다.

첫째, `InternVL3-8B`는 현재 GUI grounding 모델을 직접 교체하는 1차 후보로 보기 어렵다. 대신 화면 해석과 절차 판단, verification을 담당하는 상위 reasoning 계층에서 실험 가치가 높다.

둘째, `DINOv2`는 현재 `CLIP` 임베딩을 가장 현실적으로 보강하거나 대체해 볼 수 있는 후보다. 특히 화면-화면 유사도, 상태 클러스터링, 중복 제거, 에러 화면 retrieval 같은 image-to-image retrieval 성격에서 유의미한 차이를 만들 가능성이 있다.

셋째, `NV-DINOv2`는 단순 실험 편의성보다 NVIDIA 운영 체인과 도메인 적응 전략을 함께 가져갈 의사가 있을 때 의미가 커진다. 따라서 당장 1순위 실험 후보라기보다, 장기적 enterprise 방향과 맞물려 검토할 항목이다.

넷째, `Qdrant`는 현재 `MongoDB + FAISS`를 즉시 전면 교체하는 용도보다, 검색 계층을 service화하고 hybrid retrieval을 구성해야 할 시점에 가장 큰 가치가 있다.

다섯째, `H200` 2장 기준으로 이번 후보군에서 적재성 자체가 주요 리스크가 되는 항목은 사실상 크지 않다. `InternVL3-8B`는 추론 기준으로 충분히 현실적이고, `DINOv2`는 더 가볍다. `Qdrant`는 GPU 적재 대상이 아니며, `NV-DINOv2`만 적응 학습 조건에 따라 별도 확인이 필요하다.

## 8. 권고 우선순위

현재 repo 구조와 실험 효율을 기준으로 하면 다음 순서가 가장 현실적이다.

| 우선순위 | 권고 항목 | 권고 이유 |
| --- | --- | --- |
| 1 | `DINOv2` 소규모 PoC | 현재 `CLIP` 경로 옆에 가장 낮은 비용으로 붙일 수 있음 |
| 2 | `InternVL3-8B` planner/verifier 벤치마크 | GUI grounding 교체가 아니라 상위 판단 품질 검증에 적합 |
| 3 | `Qdrant` 검색 계층 PoC | 검색 복합화 및 공유형 서비스 수요가 생길 때 효과 큼 |
| 4 | `NV-DINOv2` 장기 검토 | 운영 복잡도와 lock-in을 감수할 전략이 있을 때 의미가 큼 |

## 9. 후속 실행 제안

`InternVL3-8B`에 대해서는 직접 click grounding 교체 후보라는 가정을 두지 말고, 실제 RCS 스크린샷 세트를 기반으로 `screen summarization`, `next-step reasoning`, `verification` 프롬프트를 별도로 구성해 벤치마크하는 것이 적절하다.

`DINOv2`에 대해서는 [`test/video_frame_parser/analyzer.py`](/Users/daeyoung/Codes/auto_recipe_creator/test/video_frame_parser/analyzer.py)의 `CLIP` 임베딩 경로와 병렬로 붙일 수 있는 최소 PoC를 만들고, 유사 화면 검색 정확도, 상태 클러스터링 품질, 처리 속도를 비교하는 방식이 가장 효율적이다.

`Qdrant`는 기존 `MongoDB + FAISS`를 즉시 전면 교체하기보다, `frame_id`, `video_id`, `step_name`, `window_title` payload와 `image_dense`, `ocr_sparse`를 함께 넣는 작은 collection schema PoC부터 시작하는 것이 바람직하다.

`NV-DINOv2`는 NVIDIA TAO 및 DeepStream 연동까지 운영할 의사가 있는지 먼저 판단한 뒤 착수하는 편이 맞다. 운영 복잡도를 감수할 계획이 없다면 우선순위는 open `DINOv2`보다 낮게 두는 것이 합리적이다.

GPU 적재성에 대한 실제 검증이 필요할 경우에는 사무실 `H200` 2장 서버에서 각 후보에 대해 `단일 이미지 추론`, `배치 추론`, `최대 동시성`, `OOM 발생 시점`을 분리해 측정하는 방식이 적절하다. 특히 `InternVL3-8B`는 KV cache 증가량을, `NV-DINOv2`는 학습 해상도와 batch size에 따른 메모리 곡선을 별도로 기록하는 것이 필요하다.

## 10. 참고 자료

- OpenGVLab, `InternVL3-8B-hf` 모델 카드: `https://huggingface.co/OpenGVLab/InternVL3-8B-hf`
- Meta, DINOv2 공식 README: `https://github.com/facebookresearch/dinov2/blob/main/README.md`
- NVIDIA TAO, NV-DINOv2 최신 문서: `https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/self_supervised_learning/nvdinov2.html`
- NVIDIA 기술 블로그, NV-DINOv2 활용 예시: `https://developer.nvidia.com/blog/build-a-real-time-visual-inspection-pipeline-with-nvidia-tao-6-and-nvidia-deepstream-8/`
- Qdrant Overview: `https://qdrant.tech/documentation/overview/`
- Qdrant Hybrid Search 가이드: `https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/`
- Qdrant Fundamentals FAQ: `https://qdrant.tech/documentation/faq/qdrant-fundamentals/`
- Qdrant GPU indexing 가이드: `https://qdrant.tech/documentation/guides/running-with-gpu/`
- NVIDIA H200 공식 페이지: `https://www.nvidia.com/en-us/data-center/h200/`
