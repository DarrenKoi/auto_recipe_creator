"""호환성 유지용 wrapper.

기존 `prepare_research_envs.py` 호출은 유지하되, 실제 구현은
`prepare_variant_envs.py`를 사용한다.
"""

from prepare_variant_envs import main


if __name__ == "__main__":
    main()
