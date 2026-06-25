"""Ground-truth 스키마 + 로더 + 예시 템플릿 writer.

각 스크린샷마다 사람이 만든 작은 정답 파일(JSON)을 둔다(benchmark_plan.md).
스크린샷에 *보이는* 정보만 담는다. 의도적으로 읽을 수 없게 둔 영역은 `unreadable`
에 적어, 누락돼도 penalize 하지 않게 한다.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GTTable:
    """정답 표 한 개 (header + 선택된 row)."""

    title: str = ""
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class GTChart:
    """정답 차트 한 개의 보이는 라벨."""

    title: str = ""
    axis_labels: list[str] = field(default_factory=list)
    legend_labels: list[str] = field(default_factory=list)
    visible_values: list[str] = field(default_factory=list)
    trend: str = ""


@dataclass
class GroundTruth:
    """스크린샷 1장의 정답."""

    screenshot_id: str
    source_type: str = "unknown"
    title: str = ""
    # 반드시 추출돼야 할 중요한 visible text 조각들(구/문장)
    important_texts: list[str] = field(default_factory=list)
    tables: list[GTTable] = field(default_factory=list)
    charts: list[GTChart] = field(default_factory=list)
    # 기대 region type (multiset). 첫 벤치는 approximate box 허용 -> type 존재만 본다.
    region_types: list[str] = field(default_factory=list)
    # 최종 요약에 들어가야 할 키워드(요약 품질 가늠)
    expected_summary_keywords: list[str] = field(default_factory=list)
    # 의도적으로 읽을 수 없는 영역(누락돼도 감점 안 함)
    unreadable: list[str] = field(default_factory=list)
    # hallucination 검사용 추가 허용 토큰(보이는 것). 비우면 위 필드들에서 유도.
    visible_tokens: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        return cls(
            screenshot_id=str(data.get("screenshot_id", "")),
            source_type=str(data.get("source_type", "unknown")),
            title=str(data.get("title", "")),
            important_texts=[str(t) for t in (data.get("important_texts") or [])],
            tables=[
                GTTable(
                    title=str(t.get("title", "")),
                    header=[str(h) for h in (t.get("header") or [])],
                    rows=[[str(c) for c in row] for row in (t.get("rows") or [])],
                )
                for t in (data.get("tables") or [])
                if isinstance(t, dict)
            ],
            charts=[
                GTChart(
                    title=str(c.get("title", "")),
                    axis_labels=[str(a) for a in (c.get("axis_labels") or [])],
                    legend_labels=[str(l) for l in (c.get("legend_labels") or [])],
                    visible_values=[str(v) for v in (c.get("visible_values") or [])],
                    trend=str(c.get("trend", "")),
                )
                for c in (data.get("charts") or [])
                if isinstance(c, dict)
            ],
            region_types=[str(r) for r in (data.get("region_types") or [])],
            expected_summary_keywords=[
                str(k) for k in (data.get("expected_summary_keywords") or [])
            ],
            unreadable=[str(u) for u in (data.get("unreadable") or [])],
            visible_tokens=[str(v) for v in (data.get("visible_tokens") or [])],
        )

    @classmethod
    def load(cls, path: Path) -> "GroundTruth":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_template(path: Path, screenshot_id: str = "ppt_001") -> None:
    """정답 파일 작성용 빈 템플릿을 저장한다(사람이 채워 넣음)."""
    template = {
        "screenshot_id": screenshot_id,
        "source_type": "powerpoint",
        "title": "",
        "important_texts": [""],
        "tables": [{"title": "", "header": [], "rows": [[]]}],
        "charts": [
            {
                "title": "",
                "axis_labels": [],
                "legend_labels": [],
                "visible_values": [],
                "trend": "",
            }
        ],
        "region_types": ["title", "body"],
        "expected_summary_keywords": [],
        "unreadable": [],
        "visible_tokens": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["GTChart", "GTTable", "GroundTruth", "write_template"]
