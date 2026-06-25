"""슬라이드 -> 이미지 -> 구조 추출 -> Marp 라운드트립 (marp_roundtrip_design.md).

extraction/ 이 만든 evidence(ExtractionResult)를 Marp(Markdown 슬라이드)로 변환한다.
"하이브리드 복원": 마크다운으로 충실히 복원 가능한 것(제목/본문/표/수식)은 Marp
네이티브 텍스트로, 불가능한 것(차트/도형/그림)은 원본 crop 이미지로 재삽입한다.

이 패키지는 Stage 5(구조 -> Marp 생성) + Stage 6(marp-cli 렌더) + Stage 7(SSIM
검증 / 자동 강등)을 구현한다. 순수 결정 로직(생성/SSIM/강등 계획)은 집에서 검증되고,
실제 렌더/검증 루프(render_deck / verify_and_downgrade)는 marp-cli + 원본 캡처가
있으면 office 에서 돈다(없으면 graceful degrade).
"""

from side_projects.document_extraction.marp.generate import (
    evidence_to_marp,
    results_to_deck,
)
from side_projects.document_extraction.marp.render import (
    RenderResult,
    build_render_args,
    render_deck,
    resolve_marp_command,
)
from side_projects.document_extraction.marp.verify import (
    DEFAULT_SSIM_FLOOR,
    DowngradePlan,
    apply_downgrade_plans,
    flag_low_fidelity,
    plan_downgrade,
    slide_fidelity,
    ssim,
    verify_and_downgrade,
    whole_slide_marp,
)

__all__ = [
    "DEFAULT_SSIM_FLOOR",
    "DowngradePlan",
    "RenderResult",
    "apply_downgrade_plans",
    "build_render_args",
    "evidence_to_marp",
    "flag_low_fidelity",
    "plan_downgrade",
    "render_deck",
    "resolve_marp_command",
    "results_to_deck",
    "slide_fidelity",
    "ssim",
    "verify_and_downgrade",
    "whole_slide_marp",
]
