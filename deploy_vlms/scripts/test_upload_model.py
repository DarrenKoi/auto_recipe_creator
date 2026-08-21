"""upload_model 클라이언트 테스트.

transport 뒤에 진짜 UploadStore 를 꽂아 HTTP 만 걷어낸 상태로 돌린다.
    uv run pytest deploy_vlms/scripts/test_upload_model.py
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

from flask_api.model_upload.store import ChecksumMismatch, OffsetMismatch, UploadStore

_SPEC = importlib.util.spec_from_file_location(
    "upload_model", Path(__file__).with_name("upload_model.py")
)
upload_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(upload_model)


class _StoreTransport:
    """UploadStore 를 직접 호출하는 transport (HTTP 없음)."""

    def __init__(self, store: UploadStore):
        self.store = store
        self.sent_chunks: list[tuple[str, int]] = []
        self.fail_once_at: set[int] = set()
        self.fatal_at: set[int] = set()

    def open_session(self, rel_path, size, sha256, chunk_size):
        """세션을 연다."""
        session = self.store.begin(
            rel_path=rel_path, size=size, sha256=sha256, chunk_size=chunk_size
        )
        return {
            "upload_id": session.upload_id,
            "committed_offset": session.committed_offset,
            "completed": session.completed,
        }

    def put_chunk(self, upload_id, offset, data, chunk_sha256):
        """청크를 보낸다(주입된 실패 지점을 흉내낸다)."""
        if offset in self.fatal_at:
            raise ConnectionResetError("simulated hard failure")
        if offset in self.fail_once_at:
            self.fail_once_at.discard(offset)
            raise upload_model.TransientError("simulated flaky network")

        self.sent_chunks.append((upload_id, offset))
        import io

        try:
            session = self.store.append_chunk(
                upload_id=upload_id,
                offset=offset,
                stream=io.BytesIO(data),
                length=len(data),
                chunk_sha256=chunk_sha256,
            )
        except OffsetMismatch as error:
            raise upload_model.RemoteOffsetMismatch(error.expected_offset) from error
        except ChecksumMismatch as error:
            raise upload_model.RemoteChecksumMismatch(str(error)) from error
        return {"committed_offset": session.committed_offset}

    def complete(self, upload_id):
        """업로드를 마무리한다."""
        session = self.store.finish(upload_id)
        return {"completed": session.completed}


@pytest.fixture()
def env(tmp_path):
    """로컬 소스 트리 + 서버 store 를 갖춘 테스트 환경."""
    source = tmp_path / "src" / "MAI-UI-8B"
    source.mkdir(parents=True)
    (source / "config.json").write_bytes(b'{"model": "mai-ui"}')
    (source / "model-00001.safetensors").write_bytes(bytes(range(256)) * 7)
    (source / "nested").mkdir()
    (source / "nested" / "tokenizer.json").write_bytes(b"tok" * 40)

    store = UploadStore(
        dest_root=tmp_path / "models", staging_root=tmp_path / "staging"
    )
    return {
        "source": source,
        "store": store,
        "transport": _StoreTransport(store),
        "dest": tmp_path / "models",
    }


def _run(env, chunk_size=64, **kwargs):
    """계획된 모든 파일을 올린다."""
    jobs = upload_model.plan_jobs(env["source"], dest_prefix="MAI-UI-8B")
    return [
        upload_model.upload_file(
            env["transport"], job, chunk_size=chunk_size, sleep_fn=lambda _s: None, **kwargs
        )
        for job in jobs
    ]


def test_uploads_whole_directory_byte_exact(env):
    """폴더 전체를 올리면 서버 쪽 트리가 원본과 바이트 동일하다."""
    _run(env)

    for local in sorted(env["source"].rglob("*")):
        if local.is_file():
            rel = local.relative_to(env["source"]).as_posix()
            assert (env["dest"] / "MAI-UI-8B" / rel).read_bytes() == local.read_bytes()


def _big_job(env):
    """청크가 여러 개 나오는 파일 하나의 job 을 고른다."""
    jobs = upload_model.plan_jobs(env["source"], dest_prefix="MAI-UI-8B")
    return next(j for j in jobs if j.local_path.name == "model-00001.safetensors")


def _upload(env, job, chunk_size=64, **kwargs):
    """job 하나를 올린다."""
    return upload_model.upload_file(
        env["transport"], job, chunk_size=chunk_size, sleep_fn=lambda _s: None, **kwargs
    )


def test_rerun_resumes_and_does_not_resend_committed_chunks(env):
    """중단 후 다시 실행하면 이미 올라간 청크는 다시 보내지 않는다."""
    job = _big_job(env)
    transport = env["transport"]
    transport.fatal_at = {128}

    with pytest.raises(ConnectionResetError):
        _upload(env, job)
    first_pass = list(transport.sent_chunks)
    assert [offset for _id, offset in first_pass] == [0, 64]

    transport.fatal_at = set()
    transport.sent_chunks.clear()
    _upload(env, job)

    resent = [offset for _id, offset in transport.sent_chunks]
    assert min(resent) == 128
    assert (env["dest"] / "MAI-UI-8B" / "model-00001.safetensors").read_bytes() == job.local_path.read_bytes()


def test_transient_error_is_retried(env):
    """일시적 네트워크 오류는 같은 청크를 재시도해 넘어간다."""
    job = _big_job(env)
    env["transport"].fail_once_at = {64, 192}

    result = _upload(env, job)

    assert result.skipped is False
    assert (env["dest"] / "MAI-UI-8B" / "model-00001.safetensors").read_bytes() == job.local_path.read_bytes()


def test_transient_errors_give_up_after_max_retries(env):
    """끝없이 재시도하지 않는다."""

    class _AlwaysFlaky(_StoreTransport):
        def put_chunk(self, upload_id, offset, data, chunk_sha256):
            """항상 실패한다."""
            raise upload_model.TransientError("always down")

    env["transport"] = _AlwaysFlaky(env["store"])
    job = _big_job(env)

    with pytest.raises(upload_model.TransientError):
        _upload(env, job, max_retries=3)


def test_lost_response_resyncs_to_server_offset(env):
    """응답만 유실돼 서버가 이미 그 청크를 가진 경우, 서버 offset 으로 맞춘다."""
    job = _big_job(env)
    transport = env["transport"]

    class _LosesFirstResponse(_StoreTransport):
        def __init__(self, store):
            super().__init__(store)
            self.dropped = False

        def put_chunk(self, upload_id, offset, data, chunk_sha256):
            """첫 청크는 서버에 반영하고 응답만 잃어버린다."""
            result = super().put_chunk(upload_id, offset, data, chunk_sha256)
            if not self.dropped:
                self.dropped = True
                raise upload_model.TransientError("response lost in flight")
            return result

    env["transport"] = _LosesFirstResponse(env["store"])

    result = _upload(env, job)

    assert result.skipped is False
    assert (env["dest"] / "MAI-UI-8B" / "model-00001.safetensors").read_bytes() == job.local_path.read_bytes()


def test_second_run_skips_completed_files(env):
    """같은 폴더를 다시 올리면 이미 끝난 파일은 한 청크도 보내지 않는다."""
    _run(env)
    env["transport"].sent_chunks.clear()

    results = _run(env)

    assert env["transport"].sent_chunks == []
    assert all(r.skipped for r in results)


def test_changed_file_is_uploaded_again(env):
    """내용이 바뀐 파일은 완료 기록과 무관하게 다시 올라간다."""
    _run(env)
    changed = env["source"] / "config.json"
    changed.write_bytes(b'{"model": "mai-ui", "revision": 2}')
    env["transport"].sent_chunks.clear()

    _run(env)

    assert env["transport"].sent_chunks != []
    assert (env["dest"] / "MAI-UI-8B" / "config.json").read_bytes() == changed.read_bytes()


@pytest.mark.parametrize(
    "requested, server_limit, expected",
    [
        (32 * 1024 * 1024, 64 * 1024 * 1024, 32 * 1024 * 1024),
        (64 * 1024 * 1024, 8 * 1024 * 1024, 8 * 1024 * 1024),
        (32 * 1024 * 1024, 0, 32 * 1024 * 1024),
        (32 * 1024 * 1024, None, 32 * 1024 * 1024),
    ],
)
def test_chunk_size_is_clamped_to_server_limit(requested, server_limit, expected):
    """서버가 알려준 상한보다 큰 청크는 자동으로 줄인다(프록시 413 회피)."""
    assert upload_model.resolve_chunk_size(requested, server_limit) == expected


def test_zero_byte_file_uploads_without_chunks(env):
    """0바이트 파일은 청크 없이 완료된다."""
    empty = env["source"] / "empty.txt"
    empty.write_bytes(b"")

    _run(env)

    assert (env["dest"] / "MAI-UI-8B" / "empty.txt").read_bytes() == b""


def test_huggingface_local_dir_layout_is_handled(tmp_path):
    """huggingface-cli download --local-dir 산출물을 그대로 올릴 수 있다.

    - .cache/huggingface 메타데이터는 올리지 않는다
    - blobs 를 가리키는 심볼릭 링크는 실제 내용으로 올린다
    """
    local_dir = tmp_path / "MAI-UI-8B"
    (local_dir / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (local_dir / ".cache" / "huggingface" / "download" / "meta.lock").write_text("x")
    (local_dir / "config.json").write_text('{"model_type": "mai-ui"}')
    (local_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-one" * 100)
    (local_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard-two" * 100)
    (local_dir / ".gitattributes").write_text("*.safetensors filter=lfs")

    blob = tmp_path / "blobs" / "abc123"
    blob.parent.mkdir()
    blob.write_bytes(b"tokenizer-payload" * 50)
    (local_dir / "tokenizer.json").symlink_to(blob)

    jobs = upload_model.plan_jobs(local_dir, dest_prefix="MAI-UI-8B")
    rel_paths = sorted(job.rel_path for job in jobs)

    assert rel_paths == [
        "MAI-UI-8B/config.json",
        "MAI-UI-8B/model-00001-of-00002.safetensors",
        "MAI-UI-8B/model-00002-of-00002.safetensors",
        "MAI-UI-8B/tokenizer.json",
    ]
    symlinked = next(j for j in jobs if j.rel_path.endswith("tokenizer.json"))
    assert symlinked.size == blob.stat().st_size
    assert upload_model.sha256_of_file(symlinked.local_path) == hashlib.sha256(
        blob.read_bytes()
    ).hexdigest()
