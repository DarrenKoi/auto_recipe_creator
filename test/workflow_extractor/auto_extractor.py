"""
Auto Extractor - VLM 기반 자동 워크플로우 추출

화면 녹화에서 자동으로 워크플로우를 추출합니다.
프레임 차이 분석으로 이벤트를 감지하고, VLM으로 작업을 추론합니다.

파이프라인:
1. 프레임 추출 (2fps)
2. 이벤트 감지 (프레임 차이 > 임계값)
3. 핵심 프레임 선택 (클러스터링)
4. VLM 작업 추론 (before/after 프레임 쌍)
5. 시퀀스 조립 (ActionSequence)
"""

import json
import hashlib
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python 또는 numpy가 설치되지 않았습니다.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARNING] requests가 설치되지 않았습니다.")

from .models import (
    WorkflowAnnotation,
    WorkflowStep,
    InferredAction,
    KeyFrame,
    ActionType,
    RecipeType,
    ExtractionMethod,
)


@dataclass
class AutoExtractorConfig:
    """자동 추출기 설정"""

    # 프레임 추출 간격 (초)
    frame_interval: float = 0.5  # 2fps

    # 이벤트 감지 임계값 (0~1, 프레임 간 변화량)
    change_threshold: float = 0.02

    # 클러스터링: 연속 변화 프레임 병합 시간 (초)
    cluster_gap: float = 1.0

    # VLM 설정
    vlm_api_url: Optional[str] = None  # VLM API URL
    vlm_api_key: Optional[str] = None  # API 키
    vlm_model: str = "qwen-vl-max"  # 모델 이름
    vlm_provider: str = "qwen_vl"  # "qwen_vl", "openai_gpt4v", "anthropic_claude"

    # 최소 신뢰도 (이 값 미만의 추론 결과는 제외)
    min_confidence: float = 0.3

    # 출력 설정
    save_keyframes: bool = True  # 핵심 프레임 이미지 저장
    output_dir: str = "./auto_extracted"


class AutoExtractor:
    """
    VLM 기반 자동 워크플로우 추출기.

    화면 녹화를 분석하여 자동으로 작업 단계를 추출합니다.
    """

    # VLM 프롬프트 (한국어)
    VLM_ACTION_PROMPT = """화면 녹화에서 연속된 두 프레임입니다.
Frame A (이전 상태)와 Frame B (이후 상태)를 비교하여,
두 프레임 사이에 수행된 작업을 분석하세요.

다음 JSON 형식으로 응답하세요:
{
  "action_type": "click|double_click|right_click|type|select|scroll|drag|wait|verify|hotkey",
  "target": "버튼/필드 이름 또는 설명",
  "coordinates": [x, y],
  "input_text": "입력된 텍스트 (해당 시)",
  "confidence": 0.0~1.0,
  "description": "수행된 작업에 대한 설명"
}

변화가 감지되지 않으면 confidence를 0으로 설정하세요.
JSON만 응답하세요."""

    def __init__(self, config: Optional[AutoExtractorConfig] = None):
        """
        Args:
            config: 추출기 설정
        """
        self.config = config or AutoExtractorConfig()
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def extract_workflow(
        self,
        video_path: str,
        recipe_type: str = "OTHER",
        description: str = "",
    ) -> Optional[WorkflowAnnotation]:
        """
        영상에서 워크플로우를 자동 추출합니다.

        Args:
            video_path: 영상 파일 경로
            recipe_type: 레시피 타입
            description: 워크플로우 설명

        Returns:
            WorkflowAnnotation 또는 None (실패 시)
        """
        if not CV2_AVAILABLE:
            print("[ERROR] opencv-python이 필요합니다.")
            return None

        print(f"[INFO] 자동 추출 시작: {Path(video_path).name}")

        # 1. 프레임 추출
        all_frames = self._extract_frames(video_path)
        if not all_frames:
            print("[ERROR] 프레임 추출 실패")
            return None
        print(f"[INFO] {len(all_frames)}개 프레임 추출 완료")

        # 2. 이벤트 감지
        change_frames = self._detect_events(all_frames)
        print(f"[INFO] {len(change_frames)}개 변화 프레임 감지")

        if not change_frames:
            print("[WARNING] 변화 감지된 프레임이 없습니다.")
            return None

        # 3. 핵심 프레임 선택 (클러스터링)
        key_frames = self._select_key_frames(change_frames)
        print(f"[INFO] {len(key_frames)}개 핵심 프레임 선택")

        # 4. VLM 작업 추론
        inferred_actions = self._infer_actions(all_frames, key_frames)
        print(f"[INFO] {len(inferred_actions)}개 작업 추론 완료")

        if not inferred_actions:
            print("[WARNING] 추론된 작업이 없습니다.")
            return None

        # 5. WorkflowAnnotation 조립
        annotation = self._assemble_workflow(
            video_path=video_path,
            inferred_actions=inferred_actions,
            recipe_type=recipe_type,
            description=description,
        )

        # JSON 저장
        output_path = self._save_annotation(annotation)
        print(f"[INFO] 자동 추출 완료: {output_path}")
        print(f"[INFO] 총 {len(annotation.steps)}개 단계 추출")

        return annotation

    def _extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """
        영상에서 일정 간격으로 프레임을 추출합니다.

        Returns:
            [{"frame_number": int, "timestamp": float, "image": np.ndarray}, ...]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] 영상 열기 실패: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(self.config.frame_interval * fps))

        frames = []
        frame_num = 0

        while frame_num < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break

            frames.append({
                "frame_number": frame_num,
                "timestamp": frame_num / fps,
                "image": frame,
            })
            frame_num += frame_step

        cap.release()
        return frames

    def _detect_events(
        self, frames: List[Dict[str, Any]]
    ) -> List[KeyFrame]:
        """
        프레임 차이 분석으로 이벤트를 감지합니다.

        연속 프레임 간 픽셀 변화량이 임계값을 초과하면 이벤트로 판단합니다.
        """
        change_frames = []

        for i in range(1, len(frames)):
            prev = frames[i - 1]["image"]
            curr = frames[i]["image"]

            # 그레이스케일 변환 후 차이 계산
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            change_score = float(np.mean(diff) / 255.0)

            if change_score > self.config.change_threshold:
                kf = KeyFrame(
                    frame_number=frames[i]["frame_number"],
                    timestamp=frames[i]["timestamp"],
                    change_score=change_score,
                )
                change_frames.append(kf)

        return change_frames

    def _select_key_frames(self, change_frames: List[KeyFrame]) -> List[KeyFrame]:
        """
        연속된 변화 프레임을 클러스터링하고 대표 프레임을 선택합니다.

        시간 간격이 cluster_gap 이내인 프레임을 동일 클러스터로 묶고,
        각 클러스터에서 변화량이 가장 큰 프레임을 대표로 선택합니다.
        """
        if not change_frames:
            return []

        clusters: List[List[KeyFrame]] = [[change_frames[0]]]
        change_frames[0].cluster_id = 0

        for kf in change_frames[1:]:
            last_in_cluster = clusters[-1][-1]
            if kf.timestamp - last_in_cluster.timestamp <= self.config.cluster_gap:
                kf.cluster_id = len(clusters) - 1
                clusters[-1].append(kf)
            else:
                kf.cluster_id = len(clusters)
                clusters.append([kf])

        # 각 클러스터에서 대표 프레임 선택 (변화량 최대)
        representatives = []
        for cluster in clusters:
            best = max(cluster, key=lambda kf: kf.change_score)
            best.is_representative = True
            representatives.append(best)

        return representatives

    def _infer_actions(
        self,
        all_frames: List[Dict[str, Any]],
        key_frames: List[KeyFrame],
    ) -> List[InferredAction]:
        """
        VLM을 사용하여 핵심 프레임 쌍에서 작업을 추론합니다.

        각 핵심 프레임의 직전 프레임(before)과 해당 프레임(after)을
        VLM에 전달하여 작업을 추론합니다.
        """
        # 프레임 번호 → 인덱스 매핑
        frame_index = {f["frame_number"]: i for i, f in enumerate(all_frames)}

        inferred = []

        for kf in key_frames:
            # before 프레임 찾기 (핵심 프레임 직전)
            idx = frame_index.get(kf.frame_number)
            if idx is None or idx == 0:
                continue

            before_frame = all_frames[idx - 1]
            after_frame = all_frames[idx]

            # VLM 추론
            action = self._call_vlm_for_action(
                before_image=before_frame["image"],
                after_image=after_frame["image"],
                before_frame_num=before_frame["frame_number"],
                after_frame_num=after_frame["frame_number"],
                timestamp=kf.timestamp,
            )

            if action and action.confidence >= self.config.min_confidence:
                inferred.append(action)

            # 핵심 프레임 이미지 저장 (선택)
            if self.config.save_keyframes:
                self._save_keyframe_image(kf, after_frame["image"])

        return inferred

    def _call_vlm_for_action(
        self,
        before_image: "np.ndarray",
        after_image: "np.ndarray",
        before_frame_num: int,
        after_frame_num: int,
        timestamp: float,
    ) -> Optional[InferredAction]:
        """
        VLM API를 호출하여 두 프레임 간 작업을 추론합니다.

        Args:
            before_image: 이전 프레임 이미지
            after_image: 이후 프레임 이미지
            before_frame_num: 이전 프레임 번호
            after_frame_num: 이후 프레임 번호
            timestamp: 이벤트 시간

        Returns:
            InferredAction 또는 None
        """
        if not REQUESTS_AVAILABLE:
            print("[ERROR] requests 라이브러리가 필요합니다.")
            return None

        if not self.config.vlm_api_url:
            print("[WARNING] VLM API URL 미설정 — 목업 응답 사용")
            return self._mock_inference(before_frame_num, after_frame_num, timestamp)

        # 이미지 인코딩
        before_b64 = self._encode_image(before_image)
        after_b64 = self._encode_image(after_image)

        # API 호출
        try:
            response = self._send_vlm_request(before_b64, after_b64)
            if response is None:
                return None

            return self._parse_vlm_response(
                response, before_frame_num, after_frame_num, timestamp
            )
        except Exception as e:
            print(f"[ERROR] VLM API 호출 실패: {e}")
            return None

    def _encode_image(self, image: "np.ndarray") -> str:
        """이미지를 base64로 인코딩합니다."""
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def _send_vlm_request(self, before_b64: str, after_b64: str) -> Optional[str]:
        """
        VLM API에 요청을 보냅니다.

        Provider별 API 형식을 지원합니다.
        """
        provider = self.config.vlm_provider
        headers = {"Content-Type": "application/json"}

        if self.config.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vlm_api_key}"

        if provider in ("openai_gpt4v", "qwen_vl", "qwen3_vl"):
            # OpenAI 호환 API 형식
            payload = {
                "model": self.config.vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.VLM_ACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{before_b64}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{after_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 500,
            }

            resp = requests.post(
                self.config.vlm_api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        elif provider == "anthropic_claude":
            payload = {
                "model": self.config.vlm_model,
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.VLM_ACTION_PROMPT},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": before_b64,
                                },
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": after_b64,
                                },
                            },
                        ],
                    }
                ],
            }

            if self.config.vlm_api_key:
                headers["x-api-key"] = self.config.vlm_api_key
                headers["anthropic-version"] = "2023-06-01"

            resp = requests.post(
                self.config.vlm_api_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

        else:
            print(f"[ERROR] 지원하지 않는 VLM 프로바이더: {provider}")
            return None

    def _parse_vlm_response(
        self,
        response: str,
        before_frame: int,
        after_frame: int,
        timestamp: float,
    ) -> Optional[InferredAction]:
        """VLM 응답을 InferredAction으로 파싱합니다."""
        try:
            # JSON 추출 (응답에 추가 텍스트가 포함될 수 있음)
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            coords = data.get("coordinates")
            if coords and isinstance(coords, list) and len(coords) == 2:
                coords = (int(coords[0]), int(coords[1]))
            else:
                coords = None

            return InferredAction(
                action_type=data.get("action_type", "click"),
                target=data.get("target", "unknown"),
                coordinates=coords,
                input_text=data.get("input_text"),
                confidence=float(data.get("confidence", 0.5)),
                description=data.get("description", ""),
                before_frame=before_frame,
                after_frame=after_frame,
                timestamp=timestamp,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[WARNING] VLM 응답 파싱 실패: {e}")
            return None

    def _extract_json(self, text: str) -> str:
        """텍스트에서 JSON 부분을 추출합니다."""
        # ```json ... ``` 블록 찾기
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()

        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()

        # { ... } 찾기
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]

        return text

    def _mock_inference(
        self, before_frame: int, after_frame: int, timestamp: float
    ) -> InferredAction:
        """VLM 미설정 시 목업 추론 결과를 반환합니다."""
        return InferredAction(
            action_type="click",
            target="unknown_element",
            confidence=0.5,
            description=f"프레임 {before_frame}→{after_frame} 사이 변화 감지 (목업)",
            before_frame=before_frame,
            after_frame=after_frame,
            timestamp=timestamp,
        )

    def _save_keyframe_image(self, kf: KeyFrame, image: "np.ndarray") -> None:
        """핵심 프레임 이미지를 저장합니다."""
        frames_dir = self._output_dir / "keyframes"
        frames_dir.mkdir(parents=True, exist_ok=True)
        filename = f"kf_{kf.frame_number:08d}_t{kf.timestamp:.2f}s.jpg"
        path = frames_dir / filename
        cv2.imwrite(str(path), image)
        kf.image_path = str(path)

    def _assemble_workflow(
        self,
        video_path: str,
        inferred_actions: List[InferredAction],
        recipe_type: str,
        description: str,
    ) -> WorkflowAnnotation:
        """추론된 작업을 WorkflowAnnotation으로 조립합니다."""
        # 타임스탬프 순 정렬
        inferred_actions.sort(key=lambda a: a.timestamp)

        steps = []
        for i, action in enumerate(inferred_actions, start=1):
            step = WorkflowStep(
                step_number=i,
                action_type=action.action_type,
                target_description=action.target,
                timestamp=action.timestamp,
                screenshot_frame=action.after_frame or 0,
                coordinates=action.coordinates,
                input_text=action.input_text,
                notes=action.description,
                confidence=action.confidence,
            )
            steps.append(step)

        # 워크플로우 ID 생성
        vid_name = Path(video_path).stem
        workflow_id = hashlib.md5(
            f"{vid_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # 영상 메타데이터
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return WorkflowAnnotation(
            workflow_id=workflow_id,
            video_path=video_path,
            recipe_type=recipe_type,
            description=description or f"{vid_name} 자동 추출",
            steps=steps,
            total_duration=duration,
            success=True,
            annotated_by="auto_extractor",
            extraction_method=ExtractionMethod.AUTOMATED.value,
            video_resolution=(width, height),
            tags=["auto_extracted"],
        )

    def _save_annotation(self, annotation: WorkflowAnnotation) -> str:
        """어노테이션을 JSON으로 저장합니다."""
        filename = f"workflow_{annotation.workflow_id}.json"
        output_path = self._output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation.to_dict(), f, ensure_ascii=False, indent=2)

        return str(output_path)


def main():
    """CLI 진입점"""
    import argparse

    parser = argparse.ArgumentParser(
        description="VLM 기반 자동 워크플로우 추출 도구"
    )
    parser.add_argument("video", help="영상 파일 경로")
    parser.add_argument(
        "--recipe-type", default="OTHER",
        help=f"레시피 타입 (기본: OTHER)"
    )
    parser.add_argument("--description", default="", help="워크플로우 설명")
    parser.add_argument("--output-dir", default="./auto_extracted", help="출력 디렉토리")
    parser.add_argument("--vlm-url", help="VLM API URL")
    parser.add_argument("--vlm-key", help="VLM API 키")
    parser.add_argument("--vlm-model", default="qwen-vl-max", help="VLM 모델")
    parser.add_argument(
        "--vlm-provider", default="qwen_vl",
        choices=["qwen_vl", "openai_gpt4v", "anthropic_claude", "qwen3_vl"],
        help="VLM 프로바이더"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="변화 감지 임계값 (기본: 0.02)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="프레임 추출 간격 초 (기본: 0.5)"
    )

    args = parser.parse_args()

    config = AutoExtractorConfig(
        frame_interval=args.interval,
        change_threshold=args.threshold,
        vlm_api_url=args.vlm_url,
        vlm_api_key=args.vlm_key,
        vlm_model=args.vlm_model,
        vlm_provider=args.vlm_provider,
        output_dir=args.output_dir,
    )

    extractor = AutoExtractor(config)
    result = extractor.extract_workflow(
        video_path=args.video,
        recipe_type=args.recipe_type,
        description=args.description,
    )

    if result:
        print(f"\n추출 완료: {len(result.steps)}개 단계")
        for step in result.steps:
            print(f"  {step.step_number}. [{step.timestamp:.2f}s] "
                  f"{step.action_type}: {step.target_description} "
                  f"(신뢰도: {step.confidence:.2f})")
    else:
        print("\n추출 실패")


if __name__ == "__main__":
    main()
