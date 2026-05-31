"""cache_source populates exactly the (layer, token, component) entries asked for."""
from __future__ import annotations

import torch

from actpatch import ActivationPatcher, CacheSpec, Component


def test_cache_only_requested_entries(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens(
        layers=[0, 2],
        tokens=[1, 3],
        components=[Component.RESID_IN, Component.K, Component.V],
    )
    cache = patcher.cache_source(sample_inputs, spec)

    expected_keys = {(0, 1), (0, 3), (2, 1), (2, 3)}
    assert set(cache.resid_in.keys()) == expected_keys
    assert set(cache.k_proj.keys()) == expected_keys
    assert set(cache.v_proj.keys()) == expected_keys

    # Tensor shapes — resid is [B, hidden]; K/V are [B, num_kv_heads*head_dim].
    hidden = tiny_model.config.hidden_size
    kv_dim = tiny_model.config.num_key_value_heads * tiny_model.config.head_dim
    for v in cache.resid_in.values():
        assert v.shape == (1, hidden)
    for store in (cache.k_proj, cache.v_proj):
        for v in store.values():
            assert v.shape == (1, kv_dim)

    # Captures default to CPU.
    assert cache.resid_in[(0, 1)].device.type == "cpu"


def test_cache_subset_components(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens(
        layers=[1], tokens=[2], components=[Component.K]
    )
    cache = patcher.cache_source(sample_inputs, spec)
    assert (1, 2) in cache.k_proj
    assert cache.v_proj == {}
    assert cache.resid_in == {}


def test_cache_includes_kv_cache(tiny_model, tiny_adapter, sample_inputs):
    """`SourceCache.kv_cache` should hold the source prefill DynamicCache."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens(layers=[0], tokens=[0], components=[Component.K])
    cache = patcher.cache_source(sample_inputs, spec)

    assert cache.kv_cache is not None
    assert len(cache.kv_cache.key_cache) == tiny_model.config.num_hidden_layers
    T = sample_inputs["input_ids"].shape[-1]
    assert cache.kv_cache.key_cache[0].shape[-2] == T
