# oc-review: 모델 업로드 경로 (2026-09-03)

- 기준점: `c606711` (= `57e5c7e^`, 업로드 기능 직전)
- 범위: `flask_api/model_upload/`, `deploy_vlms/scripts/upload_model.py`, 테스트, `UPLOAD.md`, `nginx/model_upload.conf`
- 티어: heavy (glm-5.3 high). Standards 1회 성공, Spec 은 3회차에 성공
  (1회차 glm-5.3 'Unexpected error' + Zen fallback 'Invalid API key',
   2회차 deepseek-v4-pro 'China-hosted opt-in required'. Zen provider 자체가 죽어 있다)
- 표준 근거 문서: 루트 `CLAUDE.md` (Code Conventions + model_upload 계약 셋 + 상수 블록 규약), `deploy_vlms/UPLOAD.md`
- 다운로드 경로: **없음** (health / session open·status·delete / chunk PUT / complete 뿐)

## Standards (exit 0)

No HARD (documented-standard) findings. Verified against CLAUDE.md "Code Conventions" + UPLOAD.md: Korean docstrings throughout (incl. tests); `[INFO]/[WARNING]/[ERROR]` print logging only; zero em-dash/U+2014 anywhere in the diff (grep-verified); no argparse; no `__future__`; constant-block precedence (shell env > file constant > default) implemented in `upload_model.py:_env` with the mandated "ignored constant" self-report print — and tested (`test_env_precedence_shell_beats_constant_beats_default`).

Security surfaces — the documented contracts hold: `_safe_rel_path` (store.py:80) resolve-based, rejects absolute/drive-qualified/root/escaping paths and symlink escapes; `append_chunk` truncates to `committed_offset` before writing and truncates back on short/corrupt body, so offsets never advance on bad chunks; `finish` re-hashes the whole `.part` before `os.replace`, and staging lives inside the root (same filesystem) — a partial file cannot reach the destination root. Token check uses `compare_digest` with Bearer fallback.

JUDGEMENT findings:

1. **Unvalidated `upload_id` reaches path construction** — `store.py:_state_path/_part_path` do `self.staging_root / f"{upload_id}{SUFFIX}"` with the raw URL segment. Flask's default converter permits `..` and (crucially) `\`. On a Windows-hosted Flask, `/sessions/..%5C..%5Cx/chunk` escapes staging via `_write_state`/`unlink`. Linux (the actual deploy) is safe since bare `..` without `/` is inert. Fix: reject anything not `^[0-9a-f]{32}$` (what `_make_upload_id` emits).
2. **Unhandled `ValueError` → 500** — routes.py `open_session` (`int(body.get("size", 0))`, `chunk_size`) and `put_chunk` (`int(request.headers.get("X-Upload-Offset", "-1"))`) crash on junk instead of a 4xx `UploadError`.
3. **Duplicated Code** — `routes.py` default `max_chunk_bytes: int = 64 * 1024 * 1024` restates `config.py DEFAULT_MAX_CHUNK_MB = 64`; `routes.health` and `config.build_model_upload_health_payload` near-duplicate the same field set.
4. **Speculative Generality** — `chunk_size` is accepted by `begin()`, persisted in state JSON, echoed in every payload, but never enforced or consumed server-side.
5. **Minor**: `compare_digest` on a non-ASCII token raises `TypeError` (500); `sha256_of_file` (client) duplicates `_hash_file` (store) but the repo explicitly documents the client must not import server code — suppressed per repo-override.

## Spec (exit 0, 3회차)

Now let me inspect the diff.
## Spec review: model upload (c606711...HEAD)

**(a) Missing / partly done**

1. **`MODEL_UPLOAD_ROOT` default doesn't follow `ALLOWED_MODEL_ROOT`.** Spec: "`MODEL_UPLOAD_ROOT` | `ALLOWED_MODEL_ROOT` 와 같은 경로 | 업로드 목적지 루트". `flask_api/model_upload/config.py:17-20` hardcodes `DEFAULT_DEST_ROOT`; it never reads `ALLOWED_MODEL_ROOT`. Values coincide today, but if `ALLOWED_MODEL_ROOT` changes, the upload root silently diverges from the model root.
2. **Spec test counts are stale.** Spec: "`uv run pytest flask_api/model_upload` # 33", "`deploy_vlms/scripts` # 14". Actual collects: **34** and **21**. Numbers in the same diff disagree with the code they document.

**(b) Scope creep (nobody asked)**

1. `routes.py:46-48` accepts `Authorization: Bearer` as token fallback; spec documents only `X-Upload-Token` ("설정하면 모든 업로드 요청에 `X-Upload-Token` 필요").
2. `nginx/model_upload.conf` adds `proxy_send_timeout`/`proxy_connect_timeout`/`client_body_timeout` and a commented IP allowlist beyond the three spec-mandated lines (`client_max_body_size 128m`, `proxy_read_timeout 900s`, `proxy_request_buffering off`). Benign.
3. `_make_progress` ETA readout — cosmetic, undocumented.

**(c) Implemented but looks wrong**

- **504 re-call race**: spec warns "무작정 `/complete` 를 다시 부르면 같은 재해싱을 반복하게 된다". `_complete_with_proxy_tolerance` polls first, but while the server is *still hashing* (status not yet completed) the loop re-POSTs `/complete` → a second concurrent `finish()` on the same `.part`; the loser's `os.replace`/`_hash_file` hits a missing file → spurious 500 churn. Tolerable but real.

**Q1 — un-configurable nginx, 51GB:** Survives. 413 path: `upload_model.py:168-179` halves on 413 (floor 256KB, no retry-count burn); 32MB→…→1MB passes nginx's 1m default, and `main()` propagates the learned size session-wide ("알아낸 상한은 세션 전체로 전파"). Gives up only if 413 persists at ≤256KB — impossible with a 1m limit. 504 path: 504∈`RETRYABLE_STATUS` → sleep → poll `get_session`; completed ⇒ success. Caveat: default `MAX_RETRIES=5` gives a ~31s poll window; a 51GB rehash can outlast it, so **one run may exit 4** — but re-run hits `open_session` → `completed` → skip, so end-to-end it finishes (spec's documented remedy: raise `MAX_RETRIES`).

**Q2 — download:** There is **no** download/retrieval capability (endpoints are health, session open/status/delete, chunk PUT, complete; the GET returns metadata only). The spec does not ask for one — upload path only.

**Q3 — Argument convention:** Honoured. Constants at file top (`upload_model.py:19-24`), `_env` (line 342) implements shell env > constant > default with "0" valid, and prints ignored constants (line 349); covered by `test_env_precedence_shell_beats_constant_beats_default`.
