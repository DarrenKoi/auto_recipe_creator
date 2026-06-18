---
status: accepted
---

# routed combined eval(consensus 우선·rcp 폴백)와 office S 수집 정책

## 맥락

production(`correct_align_fail_auto` + `consensus_resolve.resolve_templates`, 2026-06-12 landed)은
recipe·modality 별로 consensus 신뢰 가능(같은 modality S >= min_s + blur 가드)하면 consensus,
아니면 rcp 로 *라우팅*한다. 그런데 오프라인 벤치는 둘을 따로 측정해왔다:

- `golden_localization_eval_cond.py` — rcp 단일 키 localization(production 의 *폴백 arm* 만)
- `golden_consensus_eval_cond.py` — consensus vs rcp A/B(comparative lift 만)

둘 다 production 이 실제로 내보내는 숫자가 아니다(라우팅된 pick 의 정확도가 빠짐).

## 결정 1 — routed combined eval 드라이버

`poc/workflow_2/golden_combined_eval_cond.py` 신규. 기존 두 드라이버를 *그대로* 재사용한다
(LOO consensus 수학 = `align_similarity._consensus_template_ab` 그대로, bit-drift 0). 라우팅:
eligible(dominant modality S >= min_s) → consensus arm, 나머지 → rcp box localization arm.

지표는 양 arm 동일 정의로 통일: `in_topk`(진실이 후보 pool 에 듦), `rank1`(topk_rank==1).
세 축을 낸다:

- **(A) consensus scaling** — eligible recipe 를 n_S(LOO 점 수)별 bin(`S=3 / 4 / 5-6 / 7-9 / >=10`)으로
  층화해 "consensus 많을수록 rank1/topk 가 오르나"를 본다. S-희박이면 high bin 의 `n_recipes` 가 작아
  신뢰 못 하므로 rate 와 recipe 수를 함께 출력한다.
- **(B) rcp-only arm** — consensus 불가 recipe 의 rcp box localization. consensus 가 못 돕는 영역이라
  matching-engine 레버(`ALIGN_ENSEMBLE_LAB_MODE=edge_ncc` 등)의 testbed. 이 arm 은 `gle._matcher_for_eval()`
  를 타므로 lab mode 가 자동 적용된다(consensus arm 은 production ensemble 고정).
- **(C) routed overall** — eligible→consensus, rest→rcp 로 frame-weighted 종합.

주의(문서화된 wart): consensus arm 의 rcp counterfactual 은 center 템플릿(`_consensus_template_ab` 내부),
rcp-only arm 은 box 템플릿(포팅 경로). routed *pick* 은 일관(eligible=consensus, rest=rcp box)이나
arm 간 rcp 정의가 달라 lift 비교는 arm 별로만 해석한다. 순수 헬퍼(`_bin_by_s_count`/`_pick_cell`/
`_arm_rates`/`_routed_overall`)는 `test_golden_combined_eval_cond.py` 로 합성 row 검증(golden 불요).

## 결정 2 — consensus 과거 S 풀: 별도 root, class/recipe 키(eqp 무관)

consensus 이력(과거 성공 S)은 align_images 안의 per-recipe 하위폴더가 **아니라 별도 root** 에서
운영한다. 키는 **`<class>/<recipe>` 만 — eqp_id 무관**(같은 recipe 면 장비 달라도 공유). 레이아웃은
production consensus 캐시와 동일:

```
<HISTORY_ROOT>/<class>/<recipe>/events/<event_id>/S*.jpeg   (+ .<img>/cond.txt 숨김 sidecar)
```

`office_success_downloader` 출력 포맷 그대로라 무변환 적재 가능. 설정은 `golden_eval_config.HISTORY_ROOT`
(→ `seed_env()` → env `ALIGN_MSR_HISTORY_ROOT`). production `align/assets.py` 는 **건드리지 않는다**
(history 는 벤치/골든 전용 개념) — `gce._history_images(assets)` 가 `<root>/<class>/<recipe>` 를
`_list_images`(rglob, 숨김 cond 폴더 제외)로 평면화해 읽는다.

**history-first + LOO 폴백**: history 풀이 >= min_s 면 `_consensus_template_ab` 가 그 disjoint 풀로
consensus 를 빌드(eval=from_msr S, 누설 0, LOO 불필요)하고, 없으면 기존 from_msr leave-one-out 으로
폴백한다(LOO 경로는 byte-identical 보존). per_recipe row 에 `cons_pool_n`(consensus 풀 크기 = scaling 축)
+ `mode`(history|loo) 추가. combined 드라이버는 `cons_pool_n` 으로 층화(가중치는 eval frame 수).
digest 에 `consMode=hist:X/loo:Y` 노출.

### office 수집 정책
- **class·recipe·modality 당 최근 S 8~10장(rolling)** 을 `<HISTORY_ROOT>/<class>/<recipe>/events/` 에 적재.
  8 median 이면 noise+드리프트 평탄화 + scaling 곡선(n_S=2..9) 확보. 10+ 는 한계효용 급감.
- 카운트는 **modality(OM/SEM)별**(매칭은 fail 난 modality 키로만). dual-modality 는 양쪽 다.
- **최신순 rolling** — consensus 는 "현재 외형 추종"이라 오래된 S 는 drop(stale 방지).
- **S only.** E(fail)는 crosshair 가 없어 consensus 정렬 anchor 도, 자동 채점 GT 도 못 됨.
- **eqp 무관 공유로 확정** — 같은 class/recipe 면 tool 무관하게 한 풀로 합친다(tool-to-tool 외형차는 고려 안 함).

## 검증

Mac dev(golden 없음): `py_compile` OK, `uv run pytest test_golden_combined_eval_cond.py
test_ensemble_lab.py test_golden_localization_eval_cond.py -q` → 73 passed,
`uv run python poc/workflow_2/golden_combined_eval_cond.py` → `[WARNING] no_data` + exit 0.
accuracy 숫자는 office `ALIGN_GOLDEN_ROOT` 에서만.

## office 실행

```text
# routed 종합 + consensus scaling + rcp-only baseline
uv run python poc/workflow_2/golden_combined_eval_cond.py

# rcp-only arm 에 edge_ncc 레버 적용 — (B) 숫자가 오르나
ALIGN_ENSEMBLE_LAB_MODE=edge_ncc uv run python poc/workflow_2/golden_combined_eval_cond.py
```

판정: (A) cons r1/topk 가 n_S bin 따라 단조 증가하면 "많을수록 좋음" 확인(단 high bin n_recipes 확인).
(B) edge_ncc 가 rcp-only rank1 을 올리고 회귀 없으면 production rcp 경로 포팅 후보.
(C) routed rank1 이 production 예상 정확도.
