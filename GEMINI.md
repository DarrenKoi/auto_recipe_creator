# Gemini Context: Auto Recipe Creator (ARC)

## Project Overview
**Auto Recipe Creator (ARC)** is an automation solution designed to streamline the Recipe Setup process for **CD-SEM** and **VeritySEM** semiconductor metrology equipment.

The system utilizes **Vision Language Models (VLM)** (Qwen3-VL, Kimi 2, etc.) to analyze the Remote Control System (RCS) interface and performs GUI automation to control the equipment. It aims to reduce manual repetitive tasks while supporting a **Supervised Autonomy** model where engineers can intervene when necessary.

### Key Features
*   **VLM-driven UI Analysis:** Uses Vision Language Models to understand screen states and determine actions.
*   **GUI Automation:** Controls mouse and keyboard inputs via `pynput` and `pywinauto`.
*   **RCS Integration:** Automates launching, logging in, and navigating the Remote Control System.
*   **Video Analysis:** Includes tools for parsing and analyzing video feeds (likely for workflow extraction or training).

## Architecture & Directory Structure
*   `automation/`: Core automation logic for the RCS application.
    *   `rcs/`: RCS-specific controllers, launchers, and configuration.
*   `poc/`: Proof of Concept for CPU-based automation and GPU ROI analysis.
*   `test/`: Prototyping and testing modules.
    *   `vlm_input_control/`: Integration of Screen Capture (`mss`), VLM analysis, and Input Control.
    *   `video_frame_parser/`: Tools for extracting and analyzing video frames (supports H200 cluster optimization).
*   `docs/`: Project documentation (setup guides, requirements).
*   `requirements-automation.txt`: Python dependencies for the automation components.

## Getting Started

### Prerequisites
*   Python 3.x
*   Windows (for RCS automation) or macOS/Linux (for development/testing components)
*   **Note:** The core RCS automation relies on `pywinauto`, which is Windows-specific. `mss` and `pynput` are cross-platform but may require specific permissions on macOS.

### Installation
Install the required Python packages:
```bash
pip install -r requirements-automation.txt
```
*Note: Additional requirements may exist in `test/video_frame_parser/requirements.txt`, `test/vlm_input_control/requirements.txt`, and `poc/README.md`.*

## Key Commands & Usage

### 1. RCS Automation
To run the RCS login automation sequence:
```bash
# Execute from the project root
python -m automation.rcs.run_login
```
*   **Configuration:** Settings are managed in `automation/rcs/rcs_config.py`.
*   **Launcher:** `automation/rcs/rcs_launcher.py` orchestrates the process.

### 2. CPU-Based Automation PoC
To demonstrate automation feasibility using internal VLM APIs (CPU-only):
```bash
# Verify setup
python -m poc.test_setup

# Run screen analysis demo
python -m poc.cpu_automation_demo --provider qwen3_vl --api-url YOUR_API_URL --demo-type screen_analysis
```
*   **Providers:** Supports `qwen3_vl` and `kimi_2`.
*   **Optimization:** Includes automatic WebP conversion and rate limiting.

### 3. VLM & Input Control Integration Test
To test the screen capture, VLM analysis, and input simulation pipeline:
```bash
# Run in Safe Mode (no actual input)
python test/vlm_input_control/integration_test.py

# Run in Live Mode (PERFORMS ACTUAL INPUTS - USE CAUTION)
python test/vlm_input_control/integration_test.py --live
```

### 4. Video Frame Parser
Refer to `test/video_frame_parser/example_usage.py` for detailed usage examples of the video analysis tools.

## Development Conventions
*   **Code Style:** Follow standard Python PEP 8 conventions.
*   **Documentation:** Maintain documentation in `docs/` and update `User_Requirements_Documents.md` as features evolve.
*   **Testing:** Use the scripts in `test/` to validate components before integrating them into the core automation package.
