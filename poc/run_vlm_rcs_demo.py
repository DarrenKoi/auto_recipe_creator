"""
VLM RCS 자동화 POC 데모 실행 스크립트

RCS 로그인 → 도구 선택까지 VLM 에이전트가 자동 수행.
설정은 poc/.env 파일에서 관리합니다.

Usage:
    python -m poc.run_vlm_rcs_demo
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc.vlm_rcs_agent import VLMRCSAgent, AgentConfig


def print_banner(config: AgentConfig, rcs_server: str, rcs_username: str):
    """데모 시작 배너 출력"""
    mode = "SAFE (분석만)" if config.safe_mode else "LIVE (실제 입력)"
    print()
    print("=" * 60)
    print("  VLM RCS 자동화 POC 데모")
    print(f"  모델: {config.model_name} | 모드: {mode}")
    print("=" * 60)
    print()
    print(f"  API: {config.api_url}")
    print(f"  이미지: {'WebP' if config.use_webp else 'PNG'}, 최대 {config.max_image_size}px")
    print(f"  서버: {rcs_server}")
    print(f"  사용자: {rcs_username}")
    print()


def print_phase_header(phase_num: int, title: str):
    """Phase 헤더 출력"""
    print()
    print("=" * 60)
    print(f"  Phase {phase_num}: {title}")
    print("=" * 60)


def print_result(result: dict, phase_name: str):
    """Phase 결과 출력"""
    status = "성공" if result["completed"] else "미완료"
    print()
    print("-" * 60)
    print(f"  {phase_name} 결과: {status}")
    print(f"  총 스텝: {result['total_steps']}")
    print(f"  소요 시간: {result['total_time_sec']:.1f}초")
    print(f"  성공률: {result['success_rate']:.0f}%")
    print("-" * 60)


def print_final_summary(results: list):
    """최종 결과 요약"""
    total_steps = sum(r["total_steps"] for r in results)
    total_time = sum(r["total_time_sec"] for r in results)
    all_completed = all(r["completed"] for r in results)

    print()
    print("=" * 60)
    print(f"  최종 결과: {'성공' if all_completed else '일부 미완료'}")
    print(f"  총 {total_steps}단계 | 소요 {total_time:.1f}초")
    print("=" * 60)
    print()


def main():
    """데모 실행"""
    from poc.config import PocConfig

    poc_config = PocConfig.load()
    poc_config.print_summary()

    # AgentConfig 생성
    config = AgentConfig(
        api_url=poc_config.vlm.api_url,
        api_key=poc_config.vlm.api_key,
        model_name=poc_config.vlm.model_name,
        safe_mode=poc_config.operation.safe_mode,
        use_webp=poc_config.operation.use_webp,
        max_image_size=poc_config.operation.max_image_size,
        action_delay=poc_config.operation.action_delay,
    )

    rcs_server = poc_config.rcs.server
    rcs_username = poc_config.rcs.username
    rcs_password = poc_config.rcs.password
    tool_name = poc_config.operation.rcs_tool_name
    max_steps_login = poc_config.operation.max_steps_login
    max_steps_tool = poc_config.operation.max_steps_tool

    # 배너 출력
    print_banner(config, rcs_server, rcs_username)

    # 에이전트 생성
    agent = VLMRCSAgent(config)

    results = []

    # --- Phase 1: RCS 로그인 ---
    print_phase_header(1, "RCS 로그인")

    login_task = (
        f"RCS 로그인 화면에서 서버 '{rcs_server}' 선택, "
        f"사용자명 '{rcs_username}' 입력, "
        f"비밀번호 입력 후 로그인 버튼 클릭"
    )

    login_result = agent.run(login_task, max_steps=max_steps_login)
    print_result(login_result, "Phase 1 (로그인)")
    results.append(login_result)

    # --- Phase 2: 도구 선택 (로그인 성공 시) ---
    if login_result["completed"]:
        print_phase_header(2, "도구 선택")

        tool_task = f"메인 화면에서 '{tool_name}' 도구를 찾아 선택(더블클릭)"

        tool_result = agent.run(tool_task, max_steps=max_steps_tool)
        print_result(tool_result, "Phase 2 (도구 선택)")
        results.append(tool_result)
    else:
        print("\n[INFO] Phase 1 미완료로 Phase 2 건너뜀")

    # --- 최종 요약 ---
    print_final_summary(results)


if __name__ == "__main__":
    main()
