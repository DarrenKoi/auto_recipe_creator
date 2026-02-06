# CLAUDE.md — auto_recipe_creator (ARC)

## Project Purpose

AI-powered automation system for CD-SEM/VeritySEM recipe setup. Uses VLM (Vision Language Models) for screen analysis and GUI automation to replace manual recipe creation workflows in semiconductor metrology equipment.

## Repository Structure

```
automation/rcs/       # RCS GUI automation (Windows-only, pywinauto)
test/vlm_input_control/  # Screen capture + VLM analysis + mouse/keyboard control
test/video_frame_parser/ # CLIP-based video frame extraction & analysis (GPU)
```

## Setup & Dependencies

No root-level requirements.txt. Each module has its own:

```bash
# RCS automation (Windows only)
pip install -r requirements-automation.txt   # pywinauto

# VLM input control
pip install -r test/vlm_input_control/requirements.txt   # mss, pynput, Pillow, requests

# Video frame parser (needs GPU for CLIP)
pip install -r test/video_frame_parser/requirements.txt  # opencv, torch, pymongo, faiss-gpu
```

## Running Modules

```bash
# RCS auto-login (Windows only)
python -m automation.rcs.run_login --server SERVER --username USER --password PASS
python -m automation.rcs.run_login --debug   # dump UI control tree

# Video frame parser example
python -m test.video_frame_parser.example_usage
```

## Testing

```bash
# Video frame parser tests
pytest test/video_frame_parser/tests/

# VLM input control integration test (has safe mode / live mode)
python -m test.vlm_input_control.integration_test
```

## Code Conventions

- **Korean docstrings** throughout all modules
- **Print-based logging**: `[INFO]`, `[ERROR]`, `[WARNING]` prefixes (not the `logging` module)
- **Import guards** for optional dependencies:
  ```python
  try:
      import pywinauto
      PYWINAUTO_AVAILABLE = True
  except ImportError:
      PYWINAUTO_AVAILABLE = False
  ```
- **`@dataclass` config classes** with Korean field comments
- **`__all__` exports** in every `__init__.py` with relative imports
- **Enums** for categorical values (`FrameType`, `AnalysisStatus`, `VLMProvider`, `MouseButton`)
- **`to_dict()` / `from_dict()`** on data models for MongoDB serialization

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `RCSConfig` | `automation/rcs/rcs_config.py` | RCS connection/login settings |
| `RCSLauncher` | `automation/rcs/rcs_launcher.py` | Orchestrates full RCS login sequence |
| `ScreenCapture` | `test/vlm_input_control/screen_capture.py` | Screen/region capture via mss |
| `MouseController` | `test/vlm_input_control/mouse_control.py` | Mouse input via pynput |
| `KeyboardController` | `test/vlm_input_control/keyboard_control.py` | Keyboard input via pynput |
| `VLMScreenAnalyzer` | `test/vlm_input_control/vlm_screen_analysis.py` | VLM API integration (Qwen3-VL, GPT-4V, Claude) |
| `VideoFrameParser` | `test/video_frame_parser/parser.py` | Main video processing pipeline |
| `FrameAnalyzer` | `test/video_frame_parser/analyzer.py` | CLIP-based frame embeddings |
| `BatchProcessor` | `test/video_frame_parser/batch_processor.py` | Multi-GPU batch processing |
| `DatabaseHandler` | `test/video_frame_parser/db_handler.py` | MongoDB + FAISS vector storage |

## Architecture Notes

- **automation/rcs/**: Requires Windows + pywinauto. Uses `uia` or `win32` backends for UI element interaction. Entry point is `RCSLauncher.run()` with retry logic.
- **vlm_input_control**: Cross-platform (macOS/Linux/Windows). Captures screen → sends to VLM API → parses response → executes mouse/keyboard actions. Supports multiple VLM providers via `VLMProvider` enum.
- **video_frame_parser**: Designed for H200 GPU cluster. Extracts frames from video → generates CLIP embeddings → stores in MongoDB + FAISS index. Factory: `create_h200_optimized_parser(num_gpus=8)`.
