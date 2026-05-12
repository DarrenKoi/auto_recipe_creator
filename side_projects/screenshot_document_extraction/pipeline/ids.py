"""소스 파일에서 안정적인 doc_id 를 만든다.

원본 절대 경로를 sha1 해시한 후 앞 12 자를 doc_id 로 쓴다.
같은 파일을 다른 폴더로 옮기면 새 doc 이 되지만, 같은 자리에 두면 항상 같은 id 가 나온다.
"""

import hashlib
from pathlib import Path

from pipeline.settings import EXTENSION_TO_SOURCE_TYPE


def compute_doc_id(source_path: Path) -> str:
    """절대 경로 기준 sha1 해시 앞 12자를 doc_id 로 반환한다."""
    abs_path = str(source_path.resolve())
    digest = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()
    return digest[:12]


def infer_source_type(source_path: Path) -> str:
    """확장자로 source_type 을 추정한다. 알 수 없으면 'unknown'."""
    suffix = source_path.suffix.lower()
    return EXTENSION_TO_SOURCE_TYPE.get(suffix, "unknown")
