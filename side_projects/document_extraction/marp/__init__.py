"""슬라이드 -> 이미지 -> 구조 추출 -> Marp 라운드트립 (marp_roundtrip_design.md).

extraction/ 이 만든 evidence(ExtractionResult)를 Marp(Markdown 슬라이드)로 변환한다.
"하이브리드 복원": 마크다운으로 충실히 복원 가능한 것(제목/본문/표/수식)은 Marp
네이티브 텍스트로, 불가능한 것(차트/도형/그림)은 원본 crop 이미지로 재삽입한다.

이 패키지는 Stage 5(구조 -> Marp 생성)만 구현한다. Stage 6 렌더(marp-cli) +
Stage 7 SSIM 검증/자동 강등 루프는 marp-cli/이미지가 필요해 office TODO 로 둔다.
생성 자체는 순수 함수라 집에서 검증된다.
"""

from side_projects.document_extraction.marp.generate import (
    evidence_to_marp,
    results_to_deck,
)

__all__ = ["evidence_to_marp", "results_to_deck"]
