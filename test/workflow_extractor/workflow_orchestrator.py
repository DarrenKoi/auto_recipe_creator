"""
Workflow Orchestrator - RAG 기반 멀티스텝 자동화 루프

화면 캡처 → RAG 검색 → VLM 분석 → 작업 실행 → 검증의 순환 루프를 구현합니다.
전문가 워크플로우 데이터를 RAG로 활용하여 VLM의 판단을 보강합니다.

루프:
    1. 현재 화면 캡처
    2. RAGContextManager로 유사 과거 화면 검색
    3. RAGPromptBuilder로 프롬프트 보강
    4. VLMScreenAnalyzer로 다음 작업 결정
    5. 작업 실행 (마우스/키보드)
    6. 결과 검증 (화면 재캡처 + 비교)
    7. 결과 DB 기록 (피드백 루프)
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

try:
    from video_frame_parser.models import ActionSequence
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("[WARNING] video_frame_parser 모델을 불러올 수 없습니다.")

try:
    from vlm_input_control.screen_capture import ScreenCapture
    SCREEN_CAPTURE_AVAILABLE = True
except ImportError:
    SCREEN_CAPTURE_AVAILABLE = False
    print("[WARNING] ScreenCapture를 불러올 수 없습니다.")

try:
    from vlm_input_control.mouse_control import MouseController
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False
    print("[WARNING] MouseController를 불러올 수 없습니다.")

try:
    from vlm_input_control.keyboard_control import KeyboardController
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("[WARNING] KeyboardController를 불러올 수 없습니다.")

try:
    from vlm_input_control.vlm_screen_analysis import VLMScreenAnalyzer
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    print("[WARNING] VLMScreenAnalyzer를 불러올 수 없습니다.")

try:
    from vlm_input_control.rag_context_manager import RAGContextManager
    RAG_MANAGER_AVAILABLE = True
except ImportError:
    RAG_MANAGER_AVAILABLE = False
    print("[WARNING] RAGContextManager를 불러올 수 없습니다.")

try:
    from vlm_input_control.rag_prompt_builder import RAGPromptBuilder
    RAG_BUILDER_AVAILABLE = True
except ImportError:
    RAG_BUILDER_AVAILABLE = False
    print("[WARNING] RAGPromptBuilder를 불러올 수 없습니다.")

try:
    from video_frame_parser.db_handler import DatabaseHandler
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("[WARNING] DatabaseHandler를 불러올 수 없습니다.")


class OrchestratorState(Enum):
    """오케스트레이터 상태"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StepResult:
    """단일 단계 실행 결과"""

    step_number: int
    action_type: str  # 실행된 작업 타입
    target: str  # 대상 요소
    success: bool  # 성공 여부
    confidence: float  # VLM 판단 신뢰도
    execution_time_ms: float  # 실행 시간 (밀리초)
    rag_context_used: bool  # RAG 컨텍스트 사용 여부
    error_message: Optional[str] = None
    screenshot_before: Optional[bytes] = None
    screenshot_after: Optional[bytes] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환 (스크린샷 제외)"""
        return {
            "step_number": self.step_number,
            "action_type": self.action_type,
            "target": self.target,
            "success": self.success,
            "confidence": self.confidence,
            "execution_time_ms": self.execution_time_ms,
            "rag_context_used": self.rag_context_used,
            "error_message": self.error_message,
        }


@dataclass
class OrchestratorConfig:
    """오케스트레이터 설정"""

    # 최대 실행 단계 수 (무한 루프 방지)
    max_steps: int = 50

    # 단계 간 대기 시간 (초)
    step_delay: float = 1.0

    # 작업 실행 후 검증 대기 시간 (초)
    verify_delay: float = 0.5

    # VLM 신뢰도 임계값 (이 값 미만이면 사용자 확인 요청)
    confidence_threshold: float = 0.7

    # 연속 실패 허용 횟수
    max_consecutive_failures: int = 3

    # 안전 모드 (True: 실제 작업 미실행, 로그만)
    safe_mode: bool = True

    # RAG 검색 설정
    rag_top_k: int = 3
    use_rag: bool = True

    # 피드백 루프 (결과를 DB에 저장)
    enable_feedback: bool = True


class WorkflowOrchestrator:
    """
    RAG 기반 멀티스텝 자동화 오케스트레이터.

    전문가 워크플로우 데이터를 활용하여 VLM 판단을 보강하고,
    마우스/키보드 작업을 실행합니다.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        screen_capture: Optional["ScreenCapture"] = None,
        mouse: Optional["MouseController"] = None,
        keyboard: Optional["KeyboardController"] = None,
        vlm_analyzer: Optional["VLMScreenAnalyzer"] = None,
        rag_manager: Optional["RAGContextManager"] = None,
        rag_builder: Optional["RAGPromptBuilder"] = None,
        db_handler: Optional["DatabaseHandler"] = None,
        on_step_complete: Optional[Callable[[StepResult], None]] = None,
    ):
        """
        Args:
            config: 오케스트레이터 설정
            screen_capture: 화면 캡처 모듈
            mouse: 마우스 컨트롤러
            keyboard: 키보드 컨트롤러
            vlm_analyzer: VLM 화면 분석기
            rag_manager: RAG 컨텍스트 매니저
            rag_builder: RAG 프롬프트 빌더
            db_handler: 데이터베이스 핸들러
            on_step_complete: 단계 완료 콜백
        """
        self.config = config or OrchestratorConfig()
        self._screen = screen_capture
        self._mouse = mouse
        self._keyboard = keyboard
        self._vlm = vlm_analyzer
        self._rag_manager = rag_manager
        self._rag_builder = rag_builder
        self._db = db_handler
        self._on_step_complete = on_step_complete

        self._state = OrchestratorState.IDLE
        self._step_results: List[StepResult] = []
        self._consecutive_failures = 0

    @property
    def state(self) -> OrchestratorState:
        """현재 상태"""
        return self._state

    @property
    def step_results(self) -> List[StepResult]:
        """실행된 단계 결과 목록"""
        return list(self._step_results)

    def run(
        self,
        task_description: str,
        stop_condition: Optional[Callable[[bytes], bool]] = None,
    ) -> List[StepResult]:
        """
        자동화 루프를 실행합니다.

        Args:
            task_description: 수행할 작업 설명 (예: "RCS 서버에 로그인")
            stop_condition: 종료 조건 함수 (스크린샷 → bool)

        Returns:
            각 단계의 실행 결과 리스트
        """
        self._validate_components()
        self._state = OrchestratorState.RUNNING
        self._step_results = []
        self._consecutive_failures = 0

        print(f"[INFO] 오케스트레이터 시작: {task_description}")
        if self.config.safe_mode:
            print("[INFO] 안전 모드 활성화 — 실제 작업을 실행하지 않습니다.")

        step_number = 0

        while self._state == OrchestratorState.RUNNING:
            step_number += 1

            # 최대 단계 체크
            if step_number > self.config.max_steps:
                print(f"[WARNING] 최대 단계({self.config.max_steps})에 도달. 종료합니다.")
                self._state = OrchestratorState.COMPLETED
                break

            # 연속 실패 체크
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                print(f"[ERROR] 연속 {self._consecutive_failures}회 실패. 종료합니다.")
                self._state = OrchestratorState.FAILED
                break

            print(f"\n--- 단계 {step_number} ---")
            result = self._execute_step(step_number, task_description)
            self._step_results.append(result)

            # 콜백 호출
            if self._on_step_complete:
                self._on_step_complete(result)

            # 연속 실패 카운터 업데이트
            if result.success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1

            # 피드백 루프
            if self.config.enable_feedback and self._db is not None:
                self._record_feedback(result, task_description)

            # 종료 조건 체크
            if stop_condition:
                screen = self._capture_screen()
                if screen and stop_condition(screen):
                    print("[INFO] 종료 조건 충족. 완료합니다.")
                    self._state = OrchestratorState.COMPLETED
                    break

            # 단계 간 대기
            time.sleep(self.config.step_delay)

        success_count = sum(1 for r in self._step_results if r.success)
        print(f"\n[INFO] 오케스트레이터 종료: {self._state.value}")
        print(f"[INFO] 총 {len(self._step_results)}단계, "
              f"성공 {success_count}개")

        return self._step_results

    def pause(self):
        """실행을 일시 정지합니다."""
        if self._state == OrchestratorState.RUNNING:
            self._state = OrchestratorState.PAUSED
            print("[INFO] 오케스트레이터 일시 정지")

    def resume(self):
        """일시 정지된 실행을 재개합니다."""
        if self._state == OrchestratorState.PAUSED:
            self._state = OrchestratorState.RUNNING
            print("[INFO] 오케스트레이터 재개")

    def stop(self):
        """실행을 중지합니다."""
        self._state = OrchestratorState.COMPLETED
        print("[INFO] 오케스트레이터 중지")

    def _validate_components(self):
        """필수 컴포넌트 확인"""
        if self._screen is None and SCREEN_CAPTURE_AVAILABLE:
            self._screen = ScreenCapture()
            print("[INFO] ScreenCapture 자동 초기화")

        if self._mouse is None and MOUSE_AVAILABLE:
            self._mouse = MouseController()
            print("[INFO] MouseController 자동 초기화")

        if self._keyboard is None and KEYBOARD_AVAILABLE:
            self._keyboard = KeyboardController()
            print("[INFO] KeyboardController 자동 초기화")

        if self._vlm is None:
            print("[WARNING] VLMScreenAnalyzer 미설정 — VLM 분석 불가")

        if self.config.use_rag and self._rag_manager is None:
            print("[WARNING] RAGContextManager 미설정 — RAG 기능 비활성화")

    def _capture_screen(self) -> Optional[bytes]:
        """현재 화면을 캡처합니다."""
        if self._screen is None:
            print("[ERROR] ScreenCapture를 사용할 수 없습니다.")
            return None

        screen_data = self._screen.capture_full_screen(save=False)
        return screen_data

    def _execute_step(self, step_number: int, task_description: str) -> StepResult:
        """단일 단계를 실행합니다."""
        start_time = time.time()

        # 1. 화면 캡처
        screen_before = self._capture_screen()
        if screen_before is None:
            return StepResult(
                step_number=step_number,
                action_type="capture_failed",
                target="screen",
                success=False,
                confidence=0.0,
                execution_time_ms=0.0,
                rag_context_used=False,
                error_message="화면 캡처 실패",
            )

        # 2. RAG 컨텍스트 검색
        rag_context = None
        if self.config.use_rag and self._rag_manager is not None:
            try:
                rag_context = self._rag_manager.retrieve_context(
                    current_screen=screen_before,
                    query_text=task_description,
                    top_k=self.config.rag_top_k,
                    include_actions=True,
                    include_errors=True,
                )
                print(f"[INFO] RAG 컨텍스트 검색 완료 "
                      f"(유사 프레임: {len(rag_context.similar_frames)}개, "
                      f"작업 시퀀스: {len(rag_context.action_sequences)}개)")
            except Exception as e:
                print(f"[WARNING] RAG 검색 실패: {e}")

        # 3. VLM으로 다음 작업 결정
        action_info = self._decide_next_action(
            screen_before, task_description, rag_context
        )

        if action_info is None:
            elapsed = (time.time() - start_time) * 1000
            return StepResult(
                step_number=step_number,
                action_type="analysis_failed",
                target="screen",
                success=False,
                confidence=0.0,
                execution_time_ms=elapsed,
                rag_context_used=rag_context is not None,
                error_message="VLM 분석 실패",
            )

        action_type = action_info.get("action_type", "unknown")
        target = action_info.get("target", "unknown")
        confidence = action_info.get("confidence", 0.0)

        print(f"[INFO] VLM 판단: {action_type} → {target} (신뢰도: {confidence:.2f})")

        # 신뢰도 체크
        if confidence < self.config.confidence_threshold:
            print(f"[WARNING] 신뢰도({confidence:.2f})가 임계값"
                  f"({self.config.confidence_threshold})보다 낮습니다.")

        # 완료 감지
        if action_type == "done" or action_type == "verify":
            if action_info.get("task_complete", False):
                elapsed = (time.time() - start_time) * 1000
                print("[INFO] VLM이 작업 완료를 판단했습니다.")
                self._state = OrchestratorState.COMPLETED
                return StepResult(
                    step_number=step_number,
                    action_type="done",
                    target=target,
                    success=True,
                    confidence=confidence,
                    execution_time_ms=elapsed,
                    rag_context_used=rag_context is not None,
                    screenshot_before=screen_before,
                )

        # 4. 작업 실행
        exec_success = self._execute_action(action_info)

        # 5. 검증
        time.sleep(self.config.verify_delay)
        screen_after = self._capture_screen()

        elapsed = (time.time() - start_time) * 1000

        return StepResult(
            step_number=step_number,
            action_type=action_type,
            target=target,
            success=exec_success,
            confidence=confidence,
            execution_time_ms=elapsed,
            rag_context_used=rag_context is not None,
            screenshot_before=screen_before,
            screenshot_after=screen_after,
        )

    def _decide_next_action(
        self,
        screen_data: bytes,
        task_description: str,
        rag_context: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """VLM으로 다음 작업을 결정합니다."""
        if self._vlm is None:
            print("[ERROR] VLMScreenAnalyzer를 사용할 수 없습니다.")
            return None

        try:
            # VLM 분석 (RAG 통합이 내장된 analyze_screen 사용)
            result = self._vlm.analyze_screen(
                image_data=screen_data,
                task="state_recognition",
                query_text=task_description,
            )

            if result is None:
                return None

            # suggested_actions에서 첫 번째 작업 추출
            if result.suggested_actions:
                first_action = result.suggested_actions[0]
                return self._parse_suggested_action(first_action, result.confidence)

            return {
                "action_type": "wait",
                "target": result.state_name,
                "confidence": result.confidence,
                "description": result.description,
            }
        except Exception as e:
            print(f"[ERROR] VLM 분석 실패: {e}")
            return None

    def _parse_suggested_action(
        self, action_str: str, confidence: float
    ) -> Dict[str, Any]:
        """VLM의 suggested_action 문자열을 파싱합니다."""
        # JSON 형태인 경우
        try:
            action = json.loads(action_str)
            action.setdefault("confidence", confidence)
            return action
        except (json.JSONDecodeError, TypeError):
            pass

        # 단순 문자열인 경우
        action_str_lower = action_str.lower()

        if "click" in action_str_lower:
            action_type = "click"
        elif "type" in action_str_lower or "input" in action_str_lower:
            action_type = "type"
        elif "scroll" in action_str_lower:
            action_type = "scroll"
        elif "wait" in action_str_lower:
            action_type = "wait"
        elif "done" in action_str_lower or "complete" in action_str_lower:
            action_type = "done"
        else:
            action_type = "click"

        return {
            "action_type": action_type,
            "target": action_str,
            "confidence": confidence,
        }

    def _execute_action(self, action_info: Dict[str, Any]) -> bool:
        """실제 마우스/키보드 작업을 실행합니다."""
        action_type = action_info.get("action_type", "")
        x = action_info.get("x") or action_info.get("coordinates", [None, None])[0] if action_info.get("coordinates") else None
        y = action_info.get("y") or action_info.get("coordinates", [None, None])[1] if action_info.get("coordinates") else None
        text = action_info.get("text") or action_info.get("input_text")

        if self.config.safe_mode:
            print(f"[SAFE MODE] 실행 건너뜀: {action_type} "
                  f"({x}, {y}) text={text}")
            return True

        try:
            if action_type in ("click", "double_click", "right_click"):
                if self._mouse is None:
                    print("[ERROR] MouseController 없음")
                    return False
                if x is not None and y is not None:
                    if action_type == "click":
                        self._mouse.click_at(int(x), int(y))
                    elif action_type == "double_click":
                        self._mouse.double_click(int(x), int(y))
                    elif action_type == "right_click":
                        self._mouse.right_click(int(x), int(y))
                else:
                    print("[WARNING] 좌표 없음 — 클릭 건너뜀")
                    return False

            elif action_type == "type":
                if self._keyboard is None:
                    print("[ERROR] KeyboardController 없음")
                    return False
                if text:
                    self._keyboard.type_text(text)
                else:
                    print("[WARNING] 입력 텍스트 없음")
                    return False

            elif action_type == "scroll":
                if self._mouse is None:
                    return False
                dy = action_info.get("scroll_amount", -3)
                self._mouse.scroll(dy=dy)

            elif action_type == "hotkey":
                if self._keyboard is None:
                    return False
                keys = action_info.get("keys", [])
                for key in keys:
                    self._keyboard.press_key(key)

            elif action_type in ("wait", "verify", "done"):
                wait_time = action_info.get("wait_time", 1.0)
                time.sleep(wait_time)

            else:
                print(f"[WARNING] 알 수 없는 작업 타입: {action_type}")
                return False

            return True
        except Exception as e:
            print(f"[ERROR] 작업 실행 실패: {e}")
            return False

    def _record_feedback(self, result: StepResult, task_description: str):
        """실행 결과를 DB에 피드백으로 기록합니다."""
        if self._db is None or not MODELS_AVAILABLE:
            return

        try:
            feedback_id = hashlib.md5(
                f"feedback_{result.step_number}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

            action_sequence = ActionSequence(
                action_id=feedback_id,
                video_id="live_session",
                start_frame_id=f"step_{result.step_number}_before",
                end_frame_id=f"step_{result.step_number}_after",
                start_time=0.0,
                end_time=result.execution_time_ms / 1000.0,
                actions=[{
                    "type": result.action_type,
                    "target": result.target,
                    "success": result.success,
                    "confidence": result.confidence,
                }],
                description=f"[{task_description}] 단계 {result.step_number}: {result.action_type}",
                success=result.success,
                extraction_method="live",
                confidence=result.confidence,
            )

            self._db.save_action_sequence(action_sequence)
        except Exception as e:
            print(f"[WARNING] 피드백 저장 실패: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """실행 요약을 반환합니다."""
        total = len(self._step_results)
        success = sum(1 for r in self._step_results if r.success)
        total_time = sum(r.execution_time_ms for r in self._step_results)
        rag_used = sum(1 for r in self._step_results if r.rag_context_used)

        return {
            "state": self._state.value,
            "total_steps": total,
            "successful_steps": success,
            "failed_steps": total - success,
            "success_rate": success / total if total > 0 else 0.0,
            "total_time_ms": total_time,
            "rag_context_used_count": rag_used,
            "steps": [r.to_dict() for r in self._step_results],
        }
