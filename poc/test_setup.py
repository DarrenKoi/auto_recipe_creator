"""
Setup verification script

PoC 데모 실행 전 환경 확인
"""

import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """필수 모듈 import 테스트"""
    print("="*60)
    print("Setup Verification")
    print("="*60 + "\n")

    # 1. VLM Screen Analysis
    print("[1/5] Testing VLM Screen Analysis import...")
    try:
        from test.vlm_input_control.vlm_screen_analysis import VLMProvider, VLMScreenAnalyzer
        providers = [p.value for p in VLMProvider]
        print(f"  ✅ VLMProvider loaded: {providers}")
        print(f"  ✅ New providers: kimi_2, qwen3_vl")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

    # 2. Screen Capture
    print("\n[2/5] Testing Screen Capture import...")
    try:
        from test.vlm_input_control import ScreenCapture
        print("  ✅ ScreenCapture loaded")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

    # 3. Mouse Control
    print("\n[3/5] Testing Mouse Control import...")
    try:
        from test.vlm_input_control import MouseController
        print("  ✅ MouseController loaded")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

    # 4. Keyboard Control
    print("\n[4/5] Testing Keyboard Control import...")
    try:
        from test.vlm_input_control import KeyboardController
        print("  ✅ KeyboardController loaded")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

    # 5. Dependencies
    print("\n[5/5] Testing dependencies...")
    deps = {
        "mss": False,
        "pynput": False,
        "PIL": False,
        "requests": False
    }

    try:
        import mss
        deps["mss"] = True
        print("  ✅ mss installed")
    except ImportError:
        print("  ❌ mss not installed (pip install mss)")

    try:
        import pynput
        deps["pynput"] = True
        print("  ✅ pynput installed")
    except ImportError:
        print("  ❌ pynput not installed (pip install pynput)")

    try:
        from PIL import Image
        deps["PIL"] = True
        print("  ✅ Pillow installed")
    except ImportError:
        print("  ❌ Pillow not installed (pip install Pillow)")

    try:
        import requests
        deps["requests"] = True
        print("  ✅ requests installed")
    except ImportError:
        print("  ❌ requests not installed (pip install requests)")

    print("\n" + "="*60)
    if all(deps.values()):
        print("✅ All checks passed! Ready to run PoC demo.")
    else:
        print("⚠️  Some dependencies missing. Install them first:")
        print("   pip install -r test/vlm_input_control/requirements.txt")
    print("="*60)

    return all(deps.values())


if __name__ == "__main__":
    test_imports()
