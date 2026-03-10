"""UI-TARS 인스턴스 시작 (serve_vlm.py 래퍼)."""
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.exit(subprocess.call([sys.executable, str(script_dir / "serve_vlm.py"), "ui-tars"]))
