"""serve_vlm 메모리 sizing 계산 검증."""

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "deploy_vlms" / "scripts" / "serve_vlm.py"
)
SPEC = importlib.util.spec_from_file_location("deploy_vlms_serve_vlm", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
serve_vlm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = serve_vlm
SPEC.loader.exec_module(serve_vlm)


def test_estimate_model_weight_bytes_uses_unique_shards_from_index(tmp_path):
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    shard2 = tmp_path / "model-00002-of-00002.safetensors"
    shard1.write_bytes(b"a" * 11)
    shard2.write_bytes(b"b" * 7)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a.weight": "model-00001-of-00002.safetensors", '
        '"b.weight": "model-00001-of-00002.safetensors", '
        '"c.weight": "model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )

    assert serve_vlm.estimate_model_weight_bytes(tmp_path) == 18


def test_estimate_kv_cache_bytes_per_token_supports_gqa():
    model_config = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }

    expected = 2 * 32 * 8 * 128 * 2
    assert serve_vlm.estimate_kv_cache_bytes_per_token(model_config, "bfloat16") == expected


def test_calculate_memory_sizing_for_two_colocated_8b_models():
    model_config = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    }

    sizing = serve_vlm.calculate_memory_sizing(
        weight_bytes=int(16 * serve_vlm.GIB),
        model_config=model_config,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=8,
        tensor_parallel_size=1,
        gpu_total_memory_gib=140.0,
        colocated_models_per_gpu=2,
        gpu_shared_reserve_gib=8.0,
        gpu_process_reserve_gib=4.0,
    )

    assert sizing.recommended_utilization == pytest.approx(((140.0 - 8.0) / 2.0 - 4.0) / 140.0)
    assert sizing.kv_cache_per_gpu_gib == pytest.approx(32.0)
    assert sizing.min_required_total_gib == pytest.approx(52.0)
    assert sizing.min_required_total_gib < sizing.per_process_share_gib


def test_calculate_memory_sizing_for_three_colocated_8b_models_suggests_lower_seqs():
    model_config = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    }

    sizing = serve_vlm.calculate_memory_sizing(
        weight_bytes=int(16 * serve_vlm.GIB),
        model_config=model_config,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=8,
        tensor_parallel_size=1,
        gpu_total_memory_gib=140.0,
        colocated_models_per_gpu=3,
        gpu_shared_reserve_gib=8.0,
        gpu_process_reserve_gib=4.0,
    )

    assert sizing.min_required_total_gib > sizing.per_process_share_gib
    assert sizing.suggested_max_num_seqs == 6
