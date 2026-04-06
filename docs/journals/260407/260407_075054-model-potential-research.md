### 1. 진행 사항
- 현재 기준 비교 대상을 정리했다. `docs/setup_vlms/README.md`, `docs/setup_vlms/02-model-bringup-and-special-settings.md`, `docs/setup_vlms/04-operations-integration-and-benchmarking.md`를 기준으로 이 repo의 주력 VLM/OCR 스택은 `UI-Venus`(primary full-screen grounding), `MAI-UI`(crop retry sidecar), `UI-TARS`(alternate primary), `PaddleOCR-VL-1.5`, `GOT-OCR`임을 다시 확인했다.
- 현재 임베딩/검색 경로도 함께 확인했다. `test/video_frame_parser/analyzer.py`는 `CLIP` 기반 이미지 임베딩을 만들고, `test/video_frame_parser/db_handler.py`는 `MongoDB + FAISS` 조합으로 메타데이터와 벡터 검색을 처리한다. 따라서 이번 후보군은 "기존 grounding VLM 대체"와 "기존 embedding/retrieval 강화"로 나눠서 보는 것이 맞다.
- `InternVL3-8B`의 특성을 조사했다. OpenGVLab의 Hugging Face 모델 카드 기준으로 `InternVL3-8B-hf`는 8B 파라미터 BF16 멀티모달 모델이며, 이미지/비디오/텍스트 입력을 함께 다룰 수 있고, 시각 이해와 추론 능력, tool usage, GUI agents, industrial image analysis까지 범위를 넓힌 계열로 설명된다. 소스: `https://huggingface.co/OpenGVLab/InternVL3-8B-hf`
- `InternVL3-8B`로 이 프로젝트에서 할 수 있는 일도 정리했다. 현재 `UI-Venus`와 `UI-TARS`는 직접적인 GUI grounding 계약이 강한 반면, `InternVL3-8B`는 공식 설명상 더 범용적인 멀티모달 reasoning 성격이 강하다. 따라서 현재 스택을 바로 대체하는 1차 클릭 좌표 모델보다는, 화면 상태 요약, 절차 판단, step-level verification, 실패 원인 설명, action 후보 우선순위화 같은 상위 판단 계층에서 잠재력이 크다. 이 평가는 OpenGVLab 모델 카드와 현재 repo의 모델 역할 문서를 함께 보고 내린 적용 해석이다.
- `DINOv2`의 특성을 조사했다. Meta의 공식 DINOv2 README와 관련 모델 카드 기준으로 DINOv2는 self-supervised vision foundation model이며, 범용적인 visual feature를 다양한 이미지 분포와 다운스트림 작업에 맞게 재사용하는 데 초점을 둔다. 소스: `https://github.com/facebookresearch/dinov2/blob/main/README.md`, `https://huggingface.co/facebook/dinov2-base-imagenet1k-1-layer`
- `DINOv2`로 이 프로젝트에서 할 수 있는 일도 정리했다. 현재 `test/video_frame_parser/analyzer.py`의 `CLIP` 임베딩 자리에 바로 A/B 테스트할 수 있고, 텍스트-이미지 정렬보다 화면-화면 유사도, 상태 클러스터링, 중복 프레임 제거, 에러 화면 retrieval, 유사 레시피 화면 검색 같은 image-to-image retrieval 쪽에서 잠재력이 있다. 반대로 텍스트 질의에서 바로 이미지 검색을 걸고 싶다면 현재 `CLIP` 또는 별도 text embedding 경로를 유지하는 편이 안전하다. 이 부분은 DINOv2의 성격과 현재 코드 구조를 연결한 적용 해석이다.
- `NV-DINOv2`의 특성을 조사했다. NVIDIA TAO 문서 기준으로 NV-DINOv2는 NVIDIA proprietary large-scale dataset으로 학습된 visual foundation model이고, self-supervised 기반의 robust fine-grained representation을 제공하며 localization/classification 같은 작업의 backbone으로 사용할 수 있다. NVIDIA 블로그 예시에서는 NV-DINOv2를 일반 이미지 약 7억 장 수준으로 학습된 기반 모델로 설명하고, 이후 약 70만 장의 unlabeled PCB 이미지로 domain adaptation을 수행하는 흐름을 제시한다. 소스: `https://docs.nvidia.com/tao/tao-toolkit-archive/5.2.0/text/foundation_models/overview.html`, `https://developer.nvidia.com/blog/build-a-real-time-visual-inspection-pipeline-with-nvidia-tao-6-and-nvidia-deepstream-8/`
- `NV-DINOv2`로 이 프로젝트에서 할 수 있는 일도 정리했다. 사무실 환경이 NVIDIA GPU 서버 중심이고, 향후 반도체 장비 화면/SEM 이미지/공정 이미지 같은 도메인 이미지에 대한 unlabeled adaptation까지 염두에 둔다면, open DINOv2보다 "NVIDIA 배포 체인과의 결합" 측면에서 더 강한 후보가 될 수 있다. 다만 현재 repo는 Python 스크립트 + vLLM/Flask proxy + 로컬 실험 중심이므로, TAO/DeepStream/TensorRT 쪽 운영 부담과 lock-in이 생긴다는 점은 비용으로 봐야 한다. 이 평가는 NVIDIA 공식 문서와 현재 repo 구조를 같이 본 적용 해석이다.
- `Qdrant`의 특성을 조사했다. Qdrant 공식 문서 기준으로 Qdrant는 client-server 형태의 vector database이며, dense vector뿐 아니라 sparse vector, named vector, payload filtering, HNSW 기반 검색, sharding/replication, hybrid retrieval을 지원한다. 공식 hybrid search 문서는 dense + sparse + multivector reranking까지 한 collection에 구성하는 예시를 제공한다. 소스: `https://qdrant.tech/documentation/overview/`, `https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/`
- `Qdrant`로 이 프로젝트에서 할 수 있는 일도 정리했다. 현재 `test/video_frame_parser/db_handler.py`는 `MongoDB + FAISS`를 나눠 쓰고 있는데, Qdrant를 도입하면 화면 임베딩, OCR 키워드 sparse vector, 액션 단계 메타데이터를 한 검색 계층으로 묶을 수 있다. 예를 들어 `frame_id`, `video_id`, `step_name`, `window_title`, `ocr_text` payload를 함께 저장하고, dense image retrieval + payload filter + sparse OCR keyword search를 조합한 검색이 가능하다. 따라서 개인용 실험을 넘어 coworker 공유형 retrieval service가 필요해질 때 잠재력이 크다.
- 후보군을 현재 사용 중인 것과 비교해 우선순위도 정리했다. `InternVL3-8B`는 현재 `UI-Venus`를 바로 대체하는 모델이라기보다 planner/verifier 성격의 보강 후보이고, `DINOv2`는 현재 `CLIP` 대체 또는 보강 후보이며, `NV-DINOv2`는 DINOv2의 enterprise/NVIDIA stack 특화 대안이고, `Qdrant`는 현재 `FAISS`를 확장 가능한 service 형태로 바꾸는 저장소 후보라고 보는 것이 가장 현실적이다.

### 2. 수정 내용
- 신규 파일 추가: `docs/journals/260407/260407_075054-model-potential-research.md`
- 코드 파일 수정 없음. 이번 세션은 모델/벡터DB 조사와 현재 스택 대비 적용 가능성 정리에 집중했다.

### 3. 다음 단계
- `InternVL3-8B`는 직접 click grounding 교체 후보가 아니라는 가정하에, 실제 RCS 스크린샷 세트로 `screen summarization`, `next-step reasoning`, `verification` 프롬프트를 따로 벤치마크한다.
- `DINOv2`는 `test/video_frame_parser/analyzer.py`의 `CLIP` 임베딩 경로와 나란히 붙일 수 있는 최소 PoC를 만들고, `유사 화면 검색 정확도`, `상태 클러스터링 품질`, `처리 속도`를 비교한다.
- `Qdrant`는 `MongoDB + FAISS`를 즉시 전면 교체하기보다, `frame_id/video_id/step_name/window_title` payload와 `image_dense`, `ocr_sparse`를 함께 넣는 작은 collection schema PoC부터 만든다.
- `NV-DINOv2`는 NVIDIA TAO/DeepStream 연동까지 운영할 의사가 있는지 먼저 판단한 뒤 진행한다. 운영 복잡도를 감수할 계획이 없다면 우선순위는 open `DINOv2`보다 낮게 두는 편이 맞다.

### 4. 메모리 업데이트
- 변경 없음
