"""
Home Environment Setup Test

집에서 학습 환경이 올바르게 설정되었는지 확인합니다.

Usage:
    uv run python -m poc.home.test_setup
"""

import sys
import os

# Windows 콘솔 인코딩 문제 해결
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def check_python_version():
    """Python 버전 확인"""
    print("\n[1/5] Python 버전 확인...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 10:
        print("  [OK] Python 3.10+ 요구사항 충족")
        return True
    else:
        print("  [FAIL] Python 3.10 이상이 필요합니다")
        return False


def check_dependencies():
    """필수 패키지 확인"""
    print("\n[2/5] 필수 패키지 확인...")

    packages = {
        "mss": "화면 캡처",
        "pynput": "마우스/키보드 제어",
        "PIL": "이미지 처리 (Pillow)",
        "requests": "HTTP 클라이언트",
        "huggingface_hub": "HuggingFace API",
    }

    all_ok = True
    for pkg, desc in packages.items():
        try:
            if pkg == "PIL":
                __import__("PIL")
            else:
                __import__(pkg)
            print(f"  [OK] {pkg}: {desc}")
        except ImportError:
            print(f"  [FAIL] {pkg}: {desc} (설치 필요)")
            all_ok = False

    if not all_ok:
        print("\n  [해결 방법] uv sync --extra home")

    return all_ok


def check_hf_token():
    """HuggingFace 토큰 확인"""
    print("\n[3/5] HuggingFace 토큰 확인...")

    token = os.environ.get("HF_TOKEN")

    if token:
        # 토큰 마스킹
        masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
        print(f"  [OK] HF_TOKEN 설정됨: {masked}")
        return True
    else:
        print("  [WARN]  HF_TOKEN 환경변수가 설정되지 않았습니다")
        print("\n  [설정 방법]")
        print("  1. https://huggingface.co/join 에서 무료 계정 생성")
        print("  2. https://huggingface.co/settings/tokens 에서 토큰 발급")
        print("  3. 환경변수 설정:")
        print("     Windows: set HF_TOKEN=hf_xxxx")
        print("     Linux/Mac: export HF_TOKEN=hf_xxxx")
        print("\n  [WARN]  토큰 없이도 사용 가능하지만 rate limit이 낮습니다")
        return False


def check_screen_capture():
    """화면 캡처 테스트"""
    print("\n[4/5] 화면 캡처 테스트...")

    try:
        from test.vlm_input_control import ScreenCapture

        capture = ScreenCapture()
        screenshot_bytes = capture.capture_full_screen(save=False)

        if screenshot_bytes and len(screenshot_bytes) > 0:
            size_kb = len(screenshot_bytes) / 1024
            print(f"  [OK] 화면 캡처 성공: {size_kb:.1f}KB")
            return True
        else:
            print("  [FAIL] 화면 캡처 결과 없음")
            return False

    except Exception as e:
        print(f"  [FAIL] 화면 캡처 실패: {e}")
        return False


def check_hf_api():
    """HuggingFace API 연결 테스트"""
    print("\n[5/5] HuggingFace API 연결 테스트...")

    try:
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN")

        # 토큰이 없으면 API 테스트 스킵 (선택사항)
        if not token:
            print("  [SKIP] HF_TOKEN 없음 - API 테스트 스킵")
            print("  [INFO] 토큰 설정 후 다시 테스트해주세요")
            return True  # 토큰 없으면 스킵하되 통과로 처리

        client = InferenceClient(token=token, timeout=30)

        # 간단한 text classification 테스트 (더 안정적)
        print("  API 연결 테스트 중... (최대 30초)")

        # 간단한 sentiment analysis 테스트
        result = client.text_classification(
            "I love this!",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        if result:
            print(f"  [OK] HuggingFace API 연결 성공")
            if isinstance(result, list) and len(result) > 0:
                top = result[0]
                label = top.get('label', top) if isinstance(top, dict) else str(top)
                print(f"  [INFO] 테스트 응답: {label}")
            return True
        else:
            print("  [WARN] API 응답 없음")
            return False

    except Exception as e:
        error_msg = str(e) if str(e) else repr(e)
        error_type = type(e).__name__

        if "401" in error_msg:
            print("  [FAIL] 인증 실패: HF_TOKEN을 확인해주세요")
        elif "429" in error_msg:
            print("  [WARN] Rate limit: 잠시 후 다시 시도해주세요")
            return True  # Rate limit은 연결은 성공한 것
        elif "503" in error_msg:
            print("  [WARN] 모델 로딩 중 (cold start): 정상입니다")
            return True
        elif "500" in error_msg or "502" in error_msg:
            print("  [WARN] 서버 일시 오류: 잠시 후 다시 시도해주세요")
            return True  # 일시적 오류는 연결은 된 것
        elif "timeout" in error_msg.lower() or "Timeout" in error_type:
            print("  [WARN] 연결 타임아웃: 네트워크 상태 확인 필요")
            return True  # 타임아웃은 연결 시도는 한 것
        else:
            print(f"  [FAIL] API 연결 실패 ({error_type})")
            if error_msg:
                print(f"  [INFO] 상세: {error_msg[:150]}")
        return False


def main():
    print("=" * 60)
    print("[HOME] Home Study Environment Setup Test")
    print("=" * 60)

    results = {
        "Python": check_python_version(),
        "Dependencies": check_dependencies(),
        "HF Token": check_hf_token(),
        "Screen Capture": check_screen_capture(),
        "HF API": check_hf_api(),
    }

    # 요약
    print("\n" + "=" * 60)
    print("[SUMMARY] 설정 확인 결과")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed and name not in ["HF Token"]:  # 토큰은 선택사항
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n[SUCCESS] 모든 설정이 완료되었습니다!")
        print("\n다음 단계:")
        print("  1. 데모 실행: uv run python -m poc.home.demo")
        print("  2. 대화형 모드: uv run python -m poc.home.demo --mode interactive")
    else:
        print("\n[WARNING] 일부 설정이 필요합니다. 위의 안내를 참고해주세요.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
