"""
Home Study Environment for GUI Automation

회사 API 없이 집에서 GUI 자동화를 학습하기 위한 환경.
Hugging Face 무료 Inference API를 활용합니다.

Requirements:
- Hugging Face 계정 (무료): https://huggingface.co/join
- HF Token 발급: https://huggingface.co/settings/tokens

Usage:
    # 1. HF 토큰 설정
    export HF_TOKEN="hf_xxxx"

    # 2. 테스트 실행
    uv run python -m poc.home.test_setup

    # 3. 데모 실행
    uv run python -m poc.home.demo
"""

from .hf_vlm import HuggingFaceVLM, HFModel, VLMResponse
from .demo import HomeAutomationDemo

__all__ = [
    "HuggingFaceVLM",
    "HFModel",
    "VLMResponse",
    "HomeAutomationDemo",
]
