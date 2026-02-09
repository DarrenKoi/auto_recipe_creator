"""
VLM RCS 자동화 POC 데모 실행 스크립트

RCS 로그인 → 도구 선택까지 VLM 에이전트가 자동 수행.
모든 설정은 아래 CONFIG 섹션에서 직접 변경하세요.

Usage:
    python -m poc.run_vlm_rcs_demo
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc.vlm_rcs_agent import VLMRCSAgent, AgentConfig

# ============================================================
# CONFIG - 여기에서 설정을 변경하세요
# ============================================================

# VLM API 설정
API_URL = "http://internal-api:8080"           # VLM API 엔드포인트
API_KEY = ""                                    # API 인증 키 (없으면 빈 문자열)
MODEL_NAME = "Qwen3-VL-30B-Instruct"           # 모델 이름

# 실행 모드
SAFE_MODE = True                               # True: 실제 입력 없음 (분석만), False: 실제 입력

# RCS 로그인 정보
RCS_SERVER = "SEM_SERVER_01"                    # RCS 서버
RCS_USERNAME = "admin"                          # 사용자명
RCS_PASSWORD = "password"                       # 비밀번호

# Phase 2 도구 선택 (로그인 성공 후)
TOOL_NAME = "CD-SEM Recipe Editor"             # 선택할 도구 이름

# 이미지 설정
USE_WEBP = True                                # WebP 변환 (False면 PNG)
MAX_IMAGE_SIZE = 1280                          # 리사이즈 최대 픽셀

# 에이전트 설정
MAX_STEPS_LOGIN = 15                           # Phase 1 로그인 최대 스텝
MAX_STEPS_TOOL = 10                            # Phase 2 도구 선택 최대 스텝
ACTION_DELAY = 0.5                             # 액션 후 대기 시간 (초)

# ============================================================


def print_banner(config: AgentConfig):
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
    print(f"  서버: {RCS_SERVER}")
    print(f"  사용자: {RCS_USERNAME}")
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
    # 설정 생성
    config = AgentConfig(
        api_url=API_URL,
        api_key=API_KEY,
        model_name=MODEL_NAME,
        safe_mode=SAFE_MODE,
        use_webp=USE_WEBP,
        max_image_size=MAX_IMAGE_SIZE,
        action_delay=ACTION_DELAY,
    )

    # 배너 출력
    print_banner(config)

    # 에이전트 생성
    agent = VLMRCSAgent(config)

    results = []

    # --- Phase 1: RCS 로그인 ---
    print_phase_header(1, "RCS 로그인")

    login_task = (
        f"RCS 로그인 화면에서 서버 '{RCS_SERVER}' 선택, "
        f"사용자명 '{RCS_USERNAME}' 입력, "
        f"비밀번호 입력 후 로그인 버튼 클릭"
    )

    login_result = agent.run(login_task, max_steps=MAX_STEPS_LOGIN)
    print_result(login_result, "Phase 1 (로그인)")
    results.append(login_result)

    # --- Phase 2: 도구 선택 (로그인 성공 시) ---
    if login_result["completed"]:
        print_phase_header(2, "도구 선택")

        tool_task = f"메인 화면에서 '{TOOL_NAME}' 도구를 찾아 선택(더블클릭)"

        tool_result = agent.run(tool_task, max_steps=MAX_STEPS_TOOL)
        print_result(tool_result, "Phase 2 (도구 선택)")
        results.append(tool_result)
    else:
        print("\n[INFO] Phase 1 미완료로 Phase 2 건너뜀")

    # --- 최종 요약 ---
    print_final_summary(results)


if __name__ == "__main__":
    main()
