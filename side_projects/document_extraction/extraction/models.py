"""스테이지별 VLM 호출 래퍼 (offline/dry-run 폴백 포함).

각 스테이지는 service slug 를 통해 poc.workflow_3.vlm 의 Workflow1VLMClient 로
이미지를 보낸다. 모델 서버가 없을 때(사외 dev PC)는 OFFLINE 경로로 결정론적
stub JSON 을 돌려주어, 파이프라인 골격/merge/chunk 로직을 서버 없이 검증한다.

OFFLINE 토글:
    - 환경변수 DOC_EXTRACT_OFFLINE=1  -> 강제 offline
    - 또는 StageRunner(offline=True)
    - 그 외에는 실제 VLM 호출을 시도하고, 연결 실패 시 자동 offline 폴백.

서비스 slug (poc/workflow_3/vlm/flask_vlm.py):
    Stage 2 OCR        -> "paddleocr-vl-1.5"
    Stage 3 layout     -> "ui-venus"
    Stage 4 crop       -> "mai-ui" (또는 "paddleocr-vl-1.5")
    Stage 6 synthesis  -> "kimi-k2.6"  (docs 의 kimi-k2.5 는 stale)
                          폴백 "glm-5.2" (텍스트 전용, evidence-only)

Stage 6 폴백 체인: kimi(비전, 이미지+evidence) 실패 시 glm(텍스트, evidence-only)
로 1회 재시도하고, 둘 다 실패해야 offline stub 으로 강등한다. OCR/layout 은 기존
동작 유지(1회 실패 시 runner 전체 offline 강등 = 서버 다운 가정).
"""

import json
import os
import time

from side_projects.document_extraction.extraction import prompts


# 스테이지 -> 기본 service slug
OCR_SERVICE = "paddleocr-vl-1.5"
LAYOUT_SERVICE = "ui-venus"
CROP_SERVICE = "mai-ui"
SYNTHESIS_SERVICE = "kimi-k2.6"
# 합성 폴백/대체: 사내 로컬 API 의 텍스트 LLM(GLM-5.2). 이미지 없이 evidence 만 합성.
SYNTHESIS_FALLBACK_SERVICE = "glm-5.2"


def _offline_env() -> bool:
    return os.getenv("DOC_EXTRACT_OFFLINE", "").strip() in {"1", "true", "True"}


def _parse_json_loose(text: str) -> dict:
    """모델 응답에서 첫 JSON object 를 관대하게 파싱한다.

    markdown fence 나 앞뒤 prose 가 섞여 와도 첫 '{' ~ 마지막 '}' 구간을 시도한다.
    실패하면 빈 dict 를 반환(파이프라인이 죽지 않게).
    """
    if not text:
        return {}
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except Exception:
            return {}
    return {}


class StageRunner:
    """스테이지별 VLM 호출 + offline 폴백을 담당."""

    def __init__(self, offline: bool | None = None):
        # offline=None 이면 env 로 결정. 실제 호출 실패 시에도 offline 으로 강등.
        self.offline = _offline_env() if offline is None else bool(offline)
        self._client_cache: dict = {}
        self.stage_log: list[dict] = []

    # --- 내부 헬퍼 ----------------------------------------------------------

    def _client(self, service_slug: str):
        """service slug 별 Workflow1VLMClient 를 lazy 생성/캐시한다."""
        if service_slug in self._client_cache:
            return self._client_cache[service_slug]
        # import 는 lazy: offline 모드에서는 requests 경로를 아예 안 탄다.
        from poc.workflow_3.vlm.vlm_client import Workflow1VLMClient

        client = Workflow1VLMClient(service_slug)
        self._client_cache[service_slug] = client
        return client

    def _call(
        self,
        *,
        stage: str,
        service_slug: str,
        image_path: str,
        system_message: str,
        user_text: str,
        offline_stub: dict,
    ) -> dict:
        """한 스테이지 VLM 호출. offline 이거나 실패하면 stub 반환."""
        if self.offline:
            self.stage_log.append(
                {"stage": stage, "service": service_slug, "mode": "offline", "ok": True}
            )
            return dict(offline_stub)

        started = time.time()
        try:
            client = self._client(service_slug)
            resp = client.chat_with_image_path(
                image_path=image_path,
                system_message=system_message,
                user_text=user_text,
            )
            parsed = _parse_json_loose(resp.text)
            self.stage_log.append(
                {
                    "stage": stage,
                    "service": service_slug,
                    "mode": "online",
                    "ok": True,
                    "latency_ms": (time.time() - started) * 1000,
                    "token_usage": resp.token_usage,
                }
            )
            return parsed
        except Exception as exc:
            # 연결 실패 등 -> offline 폴백(파이프라인 골격은 계속 검증 가능).
            print(f"[WARNING] stage={stage} VLM 호출 실패 -> offline 폴백: {exc}")
            self.offline = True
            self.stage_log.append(
                {
                    "stage": stage,
                    "service": service_slug,
                    "mode": "offline-fallback",
                    "ok": False,
                    "error": str(exc),
                    "latency_ms": (time.time() - started) * 1000,
                }
            )
            return dict(offline_stub)

    # --- 스테이지 진입점 ----------------------------------------------------

    def run_ocr(self, image_path: str, width: int, height: int) -> dict:
        """Stage 2: first-pass OCR/document parsing."""
        system, user = prompts.prompt_first_pass_ocr(width, height)
        stub = {
            "raw_text": "[offline-stub] OCR text not available without VLM server",
            "reading_order": [],
            "tables": [],
            "charts": [],
            "formulas": [],
        }
        return self._call(
            stage="ocr",
            service_slug=OCR_SERVICE,
            image_path=image_path,
            system_message=system,
            user_text=user,
            offline_stub=stub,
        )

    def run_layout(self, image_path: str, width: int, height: int) -> dict:
        """Stage 3: layout/region detection."""
        system, user = prompts.prompt_layout_regions(width, height)
        # offline stub: 전체 이미지를 하나의 body region 으로 둔다.
        stub = {
            "source_type": "unknown",
            "regions": [
                {
                    "type": "body",
                    "bbox": {"left": 0, "top": 0, "right": width, "bottom": height},
                }
            ],
        }
        return self._call(
            stage="layout",
            service_slug=LAYOUT_SERVICE,
            image_path=image_path,
            system_message=system,
            user_text=user,
            offline_stub=stub,
        )

    def run_crop_refine(
        self, crop_path: str, width: int, height: int, region_type: str
    ) -> dict:
        """Stage 4: crop refinement (작은/dense 영역 재인식)."""
        system, user = prompts.prompt_crop_refine(width, height, region_type)
        stub = {"text": "", "header": [], "rows": [], "labels": []}
        return self._call(
            stage="crop",
            service_slug=CROP_SERVICE,
            image_path=crop_path,
            system_message=system,
            user_text=user,
            offline_stub=stub,
        )

    def _synthesis_stub(self) -> dict:
        return {
            "summary_markdown": "[offline-stub] synthesis skipped (no VLM server)",
            "overall_confidence": 0.0,
            "unresolved": [],
            "synthesis_service": "offline",
        }

    def _text_synthesis_once(
        self, service_slug: str, source_type: str, evidence_json: str
    ) -> dict:
        """텍스트 전용 합성 1회 시도(이미지 없음). 실패는 호출부로 raise."""
        system, user = prompts.prompt_synthesis_text_only(source_type)
        started = time.time()
        client = self._client(service_slug)
        resp = client.chat_text(
            system_message=system,
            user_text=user + evidence_json,
        )
        parsed = _parse_json_loose(resp.text)
        if not parsed:
            raise ValueError(f"synthesis JSON 파싱 실패(service={service_slug})")
        self.stage_log.append(
            {
                "stage": "synthesis",
                "service": service_slug,
                "mode": "online-text",
                "ok": True,
                "latency_ms": (time.time() - started) * 1000,
                "token_usage": resp.token_usage,
            }
        )
        parsed["synthesis_service"] = service_slug
        return parsed

    def run_synthesis(
        self, image_path: str, source_type: str, evidence_json: str
    ) -> dict:
        """Stage 6: large-VLM synthesis (요약 + 충돌 해소).

        폴백 체인: kimi(비전) -> glm(텍스트, evidence-only) -> offline stub.
        중간 실패는 runner 를 offline 으로 강등하지 않는다(체인이 끝까지
        실패했을 때만 강등) — OCR/layout 의 즉시 강등 정책과 다른 점.
        """
        if self.offline:
            self.stage_log.append(
                {"stage": "synthesis", "service": SYNTHESIS_SERVICE,
                 "mode": "offline", "ok": True}
            )
            return self._synthesis_stub()

        # 1차: kimi-k2.6 비전 합성(이미지 + evidence)
        system, user = prompts.prompt_synthesis(source_type)
        started = time.time()
        try:
            client = self._client(SYNTHESIS_SERVICE)
            resp = client.chat_with_image_path(
                image_path=image_path,
                system_message=system,
                user_text=user + evidence_json,
            )
            parsed = _parse_json_loose(resp.text)
            if not parsed:
                raise ValueError("synthesis JSON 파싱 실패(kimi)")
            self.stage_log.append(
                {
                    "stage": "synthesis",
                    "service": SYNTHESIS_SERVICE,
                    "mode": "online",
                    "ok": True,
                    "latency_ms": (time.time() - started) * 1000,
                    "token_usage": resp.token_usage,
                }
            )
            parsed["synthesis_service"] = SYNTHESIS_SERVICE
            return parsed
        except Exception as exc:
            print(
                f"[WARNING] synthesis 1차({SYNTHESIS_SERVICE}) 실패 -> "
                f"{SYNTHESIS_FALLBACK_SERVICE} 텍스트 폴백: {exc}"
            )
            self.stage_log.append(
                {
                    "stage": "synthesis",
                    "service": SYNTHESIS_SERVICE,
                    "mode": "online",
                    "ok": False,
                    "error": str(exc),
                    "latency_ms": (time.time() - started) * 1000,
                }
            )

        # 2차: glm-5.2 텍스트 합성(evidence-only)
        return self.run_synthesis_text(
            source_type, evidence_json, service_slug=SYNTHESIS_FALLBACK_SERVICE
        )

    def run_synthesis_text(
        self, source_type: str, evidence_json: str, *, service_slug: str | None = None
    ) -> dict:
        """Stage 6 텍스트 전용 합성(이미지 없음). 기본 서비스 = glm-5.2.

        SYNTHESIS_MODE="glm" 일 때의 직접 진입점이자 run_synthesis 의 2차 폴백.
        실패하면 offline stub 으로 강등한다(기존 _call 정책과 동일).
        """
        slug = service_slug or SYNTHESIS_FALLBACK_SERVICE
        if self.offline:
            self.stage_log.append(
                {"stage": "synthesis", "service": slug, "mode": "offline", "ok": True}
            )
            return self._synthesis_stub()

        started = time.time()
        try:
            return self._text_synthesis_once(slug, source_type, evidence_json)
        except Exception as exc:
            print(f"[WARNING] synthesis 텍스트({slug}) 실패 -> offline 폴백: {exc}")
            self.offline = True
            self.stage_log.append(
                {
                    "stage": "synthesis",
                    "service": slug,
                    "mode": "offline-fallback",
                    "ok": False,
                    "error": str(exc),
                    "latency_ms": (time.time() - started) * 1000,
                }
            )
            return self._synthesis_stub()


__all__ = [
    "CROP_SERVICE",
    "LAYOUT_SERVICE",
    "OCR_SERVICE",
    "SYNTHESIS_FALLBACK_SERVICE",
    "SYNTHESIS_SERVICE",
    "StageRunner",
]
