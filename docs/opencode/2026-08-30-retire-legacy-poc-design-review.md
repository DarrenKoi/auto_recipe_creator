# Legacy POC 제거 설계 — opencode oc-review 기록

- 실행일: 2026-08-30
- 스킬: oc-review
- 모델: opencode-go/kimi-k3 (tier=heavy, 실패), opencode-go/gpt-5.6-luna (fallback)
- 대상: `061ede7`의 `docs/superpowers/specs/2026-08-30-retire-legacy-poc-packages-design.md`
- 소요: Kimi 호출 실패 후 Luna 대화형 검토 약 6분 · spec 세션: `ses_fb0c336a9ffe015sNxTzfhLrXs`

## 모델이 지적한 것

Kimi standards 축은 다음 오류로 실패했습니다.

```text
[standards] empty final message from opencode-go/kimi-k3; retrying once
[standards] opencode-go failed: empty final message twice
[standards] retrying on Zen
[standards] FAILED on both providers for model 'kimi-k3'.
[standards]   opencode-go: empty final message twice
[standards]   opencode:    Error: Invalid API key.
```

Kimi spec 축도 다음 오류로 실패했습니다.

```text
[spec] opencode-go failed: Error: Unexpected error
[spec] retrying on Zen
[spec] FAILED on both providers for model 'kimi-k3'.
[spec]   opencode-go: Error: Unexpected error
[spec]   opencode:    Error: Invalid API key.
```

Luna standards 축은 저장소 검토를 시작했지만 내부 검토 작업이 완료되지 않아 중단했습니다. 따라서 standards 축에는 사용할 수 있는 최종 판정이 없습니다.

Luna spec 축의 최종 응답은 다음과 같습니다.

```text
Spec Findings

- High: Archive is not guaranteed to remain ignored. Requirement handoff.md:5; design 13-16. .gitignore:102-103 ignores .superpowers/ but not .scratch/; git check-ignore confirms the archive path is not ignored. Add the ignore rule and verify both archive and checksum paths.

- High: Asset-root guidance contradicts the design. Requirement handoff.md:9; design 27-31. poc/workflow_3/__init__.py:68-73 defaults to workflow_3/align_images, but poc/workflow_3/README.md:345-347 still declares workflow_1/align_images current and README.md:353-386 instructs a future migration using robocopy. Deleting workflow_1 without resolving this breaks the documented office contract.

- High: Live contract scope is under-specified. Requirement handoff.md:9; design 29-35. AGENTS.md:6-8,35-45 still lists workflow1/2 as current surfaces and commands; CLAUDE.md:125-127 lists a workflow1 entrypoint; requirements.txt:4,13 describes current work2 flows; docs/setup_vlms/README.md:46,56-58 still points users to work2. Enumerate each file as update, historical retention, or removal.

- Medium: Recovery omits deleted external tests. Requirement handoff.md:6-7; design 23-24,48-50. The restore command covers only the three package directories, not the eight tracked tests under test/work2/ and test/flask_api/test_work2_*.py.

- Medium: Active workflow_2 bench has no disposition. CLAUDE.md:41-44 calls it an active, non-frozen evaluation harness. Design 23-26 deletes it while only addressing runtime modules. State whether the benchmark is intentionally abandoned or must be relocated.

Clean point: repository inspection found no Python runtime imports from workflow_3/3e/4 into the retired packages, supporting the no-port claim in design 25-26.

Implementation status: No implementation exists yet; HEAD remains the design-only commit 061ede7 (handoff.md:2,14-16).
```

## 판단

Spec 축의 지적을 채택합니다. `.scratch/`는 현재 ignore 대상이 아니므로 비밀값이 포함될 수 있는 archive와 checksum을 만들기 전에 ignore 규칙과 `git check-ignore` 검증이 필요합니다.

또한 로컬 `office_align_fail_alarm.py`와 `office_rich_notify.py`는 `poc/workflow_1/`에만 ignored 파일로 존재하지만 workflow_3는 `poc.workflow_3.monitor.office_*`만 동적 로드합니다. 삭제 전에 두 로컬 adapter를 workflow_3 정위치로 옮기고 office 환경에서 별도 확인해야 합니다.

workflow_2 bench는 사용자의 현재 결정으로 명시적으로 폐기하는 것이므로 보존 요구는 반려하지만, 이 결정과 기존 `CLAUDE.md` 계약의 폐기를 설계에 명시해야 한다는 지적은 채택합니다. 외부 work2 테스트 8개와 현재 운영 문서의 복구·정리 범위도 설계에 열거해야 합니다.

Standards 축은 완결된 응답이 없으므로 clean으로 간주하지 않습니다.

## 후속

구현 전에 설계를 수정하여 archive ignore, office adapter 이동, 의도적인 workflow_2 bench 폐기, 현재 문서 목록, 외부 테스트 복구 경로를 명시해야 합니다. 현재 결과 상태는 design-only이며 implementation은 시작하지 않았습니다.
