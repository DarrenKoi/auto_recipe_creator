"""
VLM 기반 RCS GUI 자동화 에이전트

Observe-Think-Act 루프를 통한 에이전틱 VLM 자동화.
화면 캡처 → VLM 분석 → 액션 실행 → 검증 → 반복.

기존 cpu_automation_demo.py와의 차이:
- 한 번에 모든 UI 요소를 찾는 대신, 매 스텝마다 하나의 액션만 결정
- VLM이 reasoning + bbox 좌표를 포함한 JSON 반환
- 실행 후 화면 변화 검증, 실패 시 재시도 히스토리 전달
"""

import time
import json
import base64
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from io import BytesIO

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARNING] Pillow 라이브러리가 설치되지 않았습니다. pip install Pillow")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARNING] requests 라이브러리가 설치되지 않았습니다. pip install requests")

from .screen_capture import ScreenCapture
from .mouse_control import MouseController
from .keyboard_control import KeyboardController


@dataclass
class AgentConfig:
    """에이전트 설정"""
    # VLM API 설정
    api_url: str = ""                          # VLM API 엔드포인트 (예: http://internal-api:8080)
    api_key: str = ""                          # API 인증 키
    model_name: str = "Qwen3-VL-30B-Instruct" # 모델 이름

    # 실행 모드
    safe_mode: bool = True                     # True면 실제 입력 안 함

    # 이미지 최적화
    max_image_size: int = 1280                 # 리사이즈 최대 픽셀 (긴 쪽)
    use_webp: bool = True                      # WebP 변환 사용
    webp_quality: int = 85                     # WebP 품질 (0-100)

    # 에이전트 루프 설정
    max_steps: int = 30                        # 최대 스텝 수
    action_delay: float = 0.5                  # 액션 실행 후 대기 (초)
    history_length: int = 3                    # VLM에 전달할 히스토리 수

    # Rate limiting (5 req / 5 sec)
    rate_limit_requests: int = 5               # 허용 요청 수
    rate_limit_window: float = 5.0             # 윈도우 (초)


@dataclass
class ActionResult:
    """VLM이 반환한 액션 정보"""
    reasoning: str = ""
    action_type: str = "wait"     # click|double_click|type|scroll|hotkey|wait|done
    target_name: str = ""
    bbox: List[int] = field(default_factory=list)   # [x1, y1, x2, y2]
    text: str = ""                # type 액션 시 입력할 텍스트
    confidence: float = 0.0
    task_complete: bool = False

    @property
    def click_point(self) -> Optional[tuple]:
        """bbox 중심점 계산"""
        if len(self.bbox) == 4:
            return ((self.bbox[0] + self.bbox[2]) // 2,
                    (self.bbox[1] + self.bbox[3]) // 2)
        return None


class VLMRCSAgent:
    """VLM 기반 RCS GUI 자동화 에이전트"""

    def __init__(self, config: AgentConfig):
        """
        Args:
            config: 에이전트 설정
        """
        self.config = config

        # 모듈 초기화
        self.screen = ScreenCapture(output_dir="./captures")
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

        # Rate limiting: 타임스탬프 리스트
        self._request_timestamps: List[float] = []

        # 스텝 히스토리
        self.history: List[Dict] = []

        # 화면 크기 (첫 캡처 시 설정)
        self.screen_width = 0
        self.screen_height = 0
        self.scale_factor = 1.0

        # DPI 스케일 (macOS Retina: 2.0, 일반: 1.0)
        # mss는 물리 픽셀로 캡처하고 pynput은 논리 좌표로 동작하므로 보정 필요
        from .vlm_click_demo import get_dpi_scale
        self.dpi_scale = get_dpi_scale()

        # 설정 검증
        if not config.api_url:
            print("[WARNING] API URL이 설정되지 않았습니다. AgentConfig.api_url을 설정하세요.")

        print(f"[INFO] VLM RCS Agent 초기화 완료")
        print(f"[INFO] DPI 스케일: {self.dpi_scale:.1f}x")
        print(f"[INFO] 모델: {config.model_name}")
        print(f"[INFO] 모드: {'SAFE (실제 입력 없음)' if config.safe_mode else 'LIVE (실제 입력)'}")
        print(f"[INFO] 이미지: {'WebP' if config.use_webp else 'PNG'}, 최대 {config.max_image_size}px")

    def run(self, task_description: str, max_steps: Optional[int] = None) -> Dict:
        """
        에이전틱 루프 실행: 캡처 → VLM 분석 → 실행 → 검증 → 반복

        Args:
            task_description: 수행할 작업 설명 (한국어)
            max_steps: 최대 스텝 수 (None이면 config 값 사용)

        Returns:
            실행 결과 딕셔너리
        """
        max_steps = max_steps or self.config.max_steps
        self.history = []

        print(f"\n[INFO] 태스크 시작: {task_description}")
        print(f"[INFO] 최대 스텝: {max_steps}")
        print("-" * 60)

        total_start = time.time()
        completed = False
        step_results = []

        for step_num in range(1, max_steps + 1):
            step_start = time.time()

            # 1. 화면 캡처
            print(f"\n[Step {step_num}] 화면 캡처 ", end="")
            capture_start = time.time()
            screenshot_bytes, resized_w, resized_h = self._capture_screen()
            capture_ms = (time.time() - capture_start) * 1000

            if screenshot_bytes is None:
                print("... FAIL")
                print("[ERROR] 화면 캡처 실패")
                step_results.append({"step": step_num, "status": "capture_fail"})
                continue

            print(f".............. OK ({capture_ms:.0f}ms)")

            # 2. VLM 분석
            print(f"[Step {step_num}] VLM 분석 중 ...........")
            self._rate_limit_wait()

            vlm_start = time.time()
            action = self._ask_vlm(
                screenshot_bytes, task_description,
                step_num, resized_w, resized_h
            )
            vlm_ms = (time.time() - vlm_start) * 1000

            if action is None:
                print(f"  VLM 응답 실패 ({vlm_ms:.0f}ms)")
                step_results.append({"step": step_num, "status": "vlm_fail"})
                continue

            # VLM 추론 출력
            print(f"  VLM 추론: \"{action.reasoning}\"")
            if action.click_point:
                # 스케일 복원 좌표
                scaled_x = int(action.click_point[0] / self.scale_factor / self.dpi_scale)
                scaled_y = int(action.click_point[1] / self.scale_factor / self.dpi_scale)
                print(f"  다음 액션: {action.action_type} ({scaled_x}, {scaled_y}) -> \"{action.target_name}\"")
            else:
                print(f"  다음 액션: {action.action_type} -> \"{action.target_name}\"")
            print(f"  신뢰도: {action.confidence:.2f} ({vlm_ms:.0f}ms)")

            # 태스크 완료 체크
            if action.task_complete or action.action_type == "done":
                print(f"\n[Step {step_num}] 태스크 완료!")
                completed = True
                step_results.append({
                    "step": step_num, "status": "done",
                    "reasoning": action.reasoning
                })
                break

            # 3. 액션 실행
            print(f"[Step {step_num}] 실행: {action.action_type}", end="")
            if action.click_point:
                scaled_x = int(action.click_point[0] / self.scale_factor / self.dpi_scale)
                scaled_y = int(action.click_point[1] / self.scale_factor / self.dpi_scale)
                print(f" ({scaled_x}, {scaled_y})", end="")
            print(" ... ", end="")

            success = self._execute_action(action)
            print("OK" if success else "FAIL")

            # 히스토리에 추가
            self.history.append({
                "step": step_num,
                "action_type": action.action_type,
                "target": action.target_name,
                "success": success,
                "reasoning": action.reasoning
            })

            step_results.append({
                "step": step_num, "status": "ok" if success else "fail",
                "action": action.action_type, "target": action.target_name
            })

            # 액션 후 대기 (화면 변화 반영)
            time.sleep(self.config.action_delay)

            print("-" * 60)

        total_time = time.time() - total_start
        success_count = sum(1 for r in step_results if r["status"] == "ok")
        total_steps = len(step_results)

        result = {
            "completed": completed,
            "total_steps": total_steps,
            "successful_steps": success_count,
            "total_time_sec": total_time,
            "success_rate": (success_count / total_steps * 100) if total_steps > 0 else 0,
            "steps": step_results
        }

        return result

    def _capture_screen(self) -> tuple:
        """
        화면 캡처 + 리사이즈 + 바이트 변환

        Returns:
            (image_bytes, resized_width, resized_height) 또는 (None, 0, 0)
        """
        if not PIL_AVAILABLE:
            print("[ERROR] Pillow 라이브러리 필요")
            return None, 0, 0

        # 전체 화면 캡처 (PNG bytes)
        png_data = self.screen.capture_full_screen(save=False)
        if png_data is None:
            return None, 0, 0

        # PIL Image로 변환
        image = Image.open(BytesIO(png_data))
        self.screen_width, self.screen_height = image.size

        # 리사이즈
        max_dim = max(image.size)
        if max_dim > self.config.max_image_size:
            self.scale_factor = self.config.max_image_size / max_dim
            new_w = int(image.size[0] * self.scale_factor)
            new_h = int(image.size[1] * self.scale_factor)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            self.scale_factor = 1.0

        resized_w, resized_h = image.size

        # 바이트 변환
        buffer = BytesIO()
        if self.config.use_webp:
            image.save(buffer, format="WEBP", quality=self.config.webp_quality, method=4)
        else:
            image.save(buffer, format="PNG", optimize=True)

        return buffer.getvalue(), resized_w, resized_h

    def _ask_vlm(
        self, screenshot_bytes: bytes, task: str,
        step_num: int, screen_w: int, screen_h: int
    ) -> Optional[ActionResult]:
        """
        VLM API 호출하여 다음 액션 결정

        Returns:
            ActionResult 또는 None
        """
        if not REQUESTS_AVAILABLE:
            print("[ERROR] requests 라이브러리 필요")
            return None

        if not self.config.api_url:
            print("[ERROR] API URL이 설정되지 않음")
            return None

        # 프롬프트 구성
        prompt = self._build_prompt(task, step_num, screen_w, screen_h)

        # 이미지 base64 인코딩
        image_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        # 이미지 포맷 감지
        image_format = "png"
        if screenshot_bytes[:4] == b'RIFF' and screenshot_bytes[8:12] == b'WEBP':
            image_format = "webp"

        # API 호출
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "당신은 GUI 자동화 에이전트입니다. "
                        "화면 스크린샷을 분석하여 다음에 수행할 단일 액션을 결정합니다. "
                        f"이 이미지의 해상도는 {screen_w}x{screen_h} 픽셀입니다. "
                        f"좌표는 반드시 0~{screen_w}(x), 0~{screen_h}(y) 범위의 픽셀 값으로 반환하세요. "
                        "반드시 JSON 형식으로만 응답하세요."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1
        }

        try:
            response = requests.post(
                f"{self.config.api_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]

            return self._parse_vlm_response(response_text, screen_w, screen_h)

        except requests.exceptions.ConnectionError:
            print(f"[ERROR] API 서버 연결 실패: {self.config.api_url}")
            return None
        except requests.exceptions.Timeout:
            print("[ERROR] API 요청 타임아웃 (60초)")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] API HTTP 오류: {e.response.status_code} {e.response.reason}")
            return None
        except (KeyError, IndexError) as e:
            print(f"[ERROR] API 응답 형식 오류: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] VLM API 호출 실패: {e}")
            return None

    def _build_prompt(self, task: str, step_num: int, screen_w: int, screen_h: int) -> str:
        """VLM에 보낼 프롬프트 구성"""
        # 최근 히스토리
        recent_history = self.history[-self.config.history_length:]
        history_text = ""
        if recent_history:
            history_lines = []
            for h in recent_history:
                status = "성공" if h["success"] else "실패 (화면 변화 없음)"
                history_lines.append(
                    f"  Step {h['step']}: {h['action_type']} \"{h['target']}\" -> {status}"
                )
            history_text = "이전 액션 히스토리:\n" + "\n".join(history_lines) + "\n\n"

        # 실패 감지 알림
        failure_hint = ""
        if recent_history and not recent_history[-1]["success"]:
            failure_hint = (
                "주의: 이전 클릭이 실패한 것 같습니다. "
                "다른 좌표나 다른 접근 방식을 시도하세요.\n\n"
            )

        return f"""현재 작업: {task}

현재 스텝: {step_num}

{history_text}{failure_hint}화면 해상도: {screen_w}x{screen_h} 픽셀

다음에 수행할 하나의 액션을 결정하세요.
bbox는 대상 UI 요소의 바운딩 박스 [x1, y1, x2, y2] (픽셀 좌표)입니다.

다음 JSON 형식으로만 응답하세요:
{{
  "reasoning": "현재 화면 상태와 다음 액션을 선택한 이유",
  "action_type": "click|double_click|type|scroll|hotkey|wait|done",
  "target_name": "대상 UI 요소 설명",
  "bbox": [x1, y1, x2, y2],
  "text": "type 액션인 경우 입력할 텍스트 (아닌 경우 빈 문자열)",
  "confidence": 0.0,
  "task_complete": false
}}

action_type 설명:
- click: bbox 중심점 좌클릭
- double_click: bbox 중심점 더블클릭
- type: bbox 중심점 클릭 후 text 입력
- scroll: 스크롤 (bbox 위치에서 아래로)
- hotkey: 키보드 단축키 (text에 "ctrl+a" 형식으로 지정)
- wait: 대기 (로딩 등)
- done: 작업 완료"""

    def _parse_vlm_response(self, response_text: str, screen_w: int, screen_h: int) -> Optional[ActionResult]:
        """VLM 응답에서 ActionResult 파싱"""
        try:
            json_str = self._extract_json_from_response(response_text)
            data = json.loads(json_str)

            action = ActionResult(
                reasoning=data.get("reasoning", ""),
                action_type=data.get("action_type", "wait"),
                target_name=data.get("target_name", ""),
                bbox=data.get("bbox", []),
                text=data.get("text", ""),
                confidence=float(data.get("confidence", 0.0)),
                task_complete=bool(data.get("task_complete", False))
            )

            # 정규화 좌표 감지 (모든 값이 0~1 사이면 픽셀로 변환)
            if action.bbox and all(0 <= v <= 1.0 for v in action.bbox):
                action.bbox = [
                    int(action.bbox[0] * screen_w),
                    int(action.bbox[1] * screen_h),
                    int(action.bbox[2] * screen_w),
                    int(action.bbox[3] * screen_h)
                ]

            # 화면 경계 클램핑
            if action.bbox and len(action.bbox) == 4:
                action.bbox[0] = max(0, min(action.bbox[0], screen_w))
                action.bbox[1] = max(0, min(action.bbox[1], screen_h))
                action.bbox[2] = max(0, min(action.bbox[2], screen_w))
                action.bbox[3] = max(0, min(action.bbox[3], screen_h))

            return action

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[ERROR] VLM 응답 파싱 실패: {e}")
            print(f"[DEBUG] 응답: {response_text[:300]}...")
            return None

    def _execute_action(self, action: ActionResult) -> bool:
        """
        VLM이 결정한 액션 실행

        Returns:
            성공 여부
        """
        try:
            # 스케일 복원: VLM 좌표 → 물리 픽셀 → 논리 좌표 (마우스)
            if action.click_point:
                # VLM 좌표 → 물리 픽셀 (리사이즈 역변환)
                phys_x = int(action.click_point[0] / self.scale_factor)
                phys_y = int(action.click_point[1] / self.scale_factor)

                # 물리 픽셀 → 논리 좌표 (DPI 보정, Retina 대응)
                actual_x = int(phys_x / self.dpi_scale)
                actual_y = int(phys_y / self.dpi_scale)

                # 논리 좌표 경계 클램핑
                logical_w = int(self.screen_width / self.dpi_scale)
                logical_h = int(self.screen_height / self.dpi_scale)
                actual_x = max(0, min(actual_x, logical_w))
                actual_y = max(0, min(actual_y, logical_h))
            else:
                actual_x, actual_y = 0, 0

            if action.action_type == "click":
                if self.config.safe_mode:
                    print(f" [SAFE] click ({actual_x}, {actual_y})", end="")
                else:
                    self.mouse.click_at(actual_x, actual_y)

            elif action.action_type == "double_click":
                if self.config.safe_mode:
                    print(f" [SAFE] double_click ({actual_x}, {actual_y})", end="")
                else:
                    self.mouse.double_click(actual_x, actual_y)

            elif action.action_type == "type":
                if self.config.safe_mode:
                    print(f" [SAFE] click ({actual_x}, {actual_y}) + type \"{action.text}\"", end="")
                else:
                    self.mouse.click_at(actual_x, actual_y)
                    time.sleep(0.2)
                    # 기존 내용 클리어
                    self.keyboard.hotkey('ctrl', 'a')
                    time.sleep(0.05)
                    self.keyboard.type_text(action.text)

            elif action.action_type == "scroll":
                # text에서 스크롤 양 파싱 (기본값: -3)
                scroll_amount = -3
                if action.text:
                    try:
                        scroll_amount = int(action.text)
                    except ValueError:
                        pass
                if self.config.safe_mode:
                    print(f" [SAFE] scroll({scroll_amount}) at ({actual_x}, {actual_y})", end="")
                else:
                    self.mouse.move_to(actual_x, actual_y)
                    self.mouse.scroll(dy=scroll_amount)

            elif action.action_type == "hotkey":
                keys = [k.strip() for k in action.text.split("+")]
                if self.config.safe_mode:
                    print(f" [SAFE] hotkey {'+'.join(keys)}", end="")
                else:
                    self.keyboard.hotkey(*keys)

            elif action.action_type == "wait":
                wait_time = 1.0
                if self.config.safe_mode:
                    print(f" [SAFE] wait {wait_time}s", end="")
                else:
                    time.sleep(wait_time)

            elif action.action_type == "done":
                pass

            else:
                print(f" [WARN] 알 수 없는 액션: {action.action_type}", end="")
                return False

            return True

        except Exception as e:
            print(f" [ERROR] 액션 실행 실패: {e}", end="")
            return False

    def _rate_limit_wait(self):
        """토큰 버킷 Rate Limiter (5 req / 5 sec)"""
        now = time.time()
        window = self.config.rate_limit_window

        # 윈도우 밖의 타임스탬프 제거
        self._request_timestamps = [
            ts for ts in self._request_timestamps if now - ts < window
        ]

        # 윈도우 내 요청 수 확인
        if len(self._request_timestamps) >= self.config.rate_limit_requests:
            # 가장 오래된 요청 이후 윈도우가 지나야 함
            oldest = self._request_timestamps[0]
            wait_time = window - (now - oldest)
            if wait_time > 0:
                print(f"  [Rate Limit] {wait_time:.1f}초 대기...")
                time.sleep(wait_time)

        self._request_timestamps.append(time.time())

    def _extract_json_from_response(self, response: str) -> str:
        """응답에서 JSON 블록 추출"""
        # ```json 블록 찾기
        if '```json' in response:
            start_idx = response.find('```json') + 7
            end_idx = response.find('```', start_idx)
            if end_idx != -1:
                return response[start_idx:end_idx].strip()

        # 중괄호로 시작하는 JSON 찾기
        if '{' in response:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if end_idx > start_idx:
                return response[start_idx:end_idx + 1]

        return response
