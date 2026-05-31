"""Unit tests for the version-robust DynamicCache helpers."""
from __future__ import annotations

import pytest
import torch

from actpatch import Component, PatchSpec
from actpatch.cache_ops import (
    apply_kv_patches_to_cache,
    cache_num_layers,
    clone_dynamic_cache,
    crop_dynamic_cache,
    read_cache_kv,
)


def _prefill_cache(tiny_model, inputs):
    """Run a forward with use_cache to get a populated DynamicCache."""
    with torch.no_grad():
        out = tiny_model(**inputs, use_cache=True)
    return out.past_key_values


def test_cache_num_layers_and_read(tiny_model, sample_inputs):
    cache = _prefill_cache(tiny_model, sample_inputs)
    assert cache_num_layers(cache) == tiny_model.config.num_hidden_layers

    T = sample_inputs["input_ids"].shape[-1]
    keys, values = read_cache_kv(cache, 0)
    # [B, num_kv_heads, T, head_dim]
    assert keys.shape == (1, tiny_model.config.num_key_value_heads, T, tiny_model.config.head_dim)
    assert values.shape == keys.shape


def test_read_cache_kv_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported cache type"):
        read_cache_kv(object(), 0)


def test_crop_dynamic_cache_truncates(tiny_model, sample_inputs):
    cache = _prefill_cache(tiny_model, sample_inputs)
    T = sample_inputs["input_ids"].shape[-1]
    crop_dynamic_cache(cache, T - 3)
    keys, values = read_cache_kv(cache, 0)
    assert keys.shape[-2] == T - 3
    assert values.shape[-2] == T - 3


def test_clone_dynamic_cache_is_independent(tiny_model, sample_inputs):
    cache = _prefill_cache(tiny_model, sample_inputs)
    clone = clone_dynamic_cache(cache)

    # Mutating the clone must not touch the original.
    ck, _ = read_cache_kv(clone, 0)
    ok, _ = read_cache_kv(cache, 0)
    before = ok[:, :, 0, :].clone()
    ck[:, :, 0, :] = 123.0
    after = read_cache_kv(cache, 0)[0][:, :, 0, :]
    assert torch.equal(before, after)


def test_apply_kv_patches_noop_when_start_index_zero(tiny_model, sample_inputs):
    target = _prefill_cache(tiny_model, sample_inputs)
    source = _prefill_cache(tiny_model, sample_inputs)
    before = read_cache_kv(target, 1)[0].clone()
    patch = PatchSpec.for_layers_tokens([1], [0], [Component.K])
    apply_kv_patches_to_cache(target, patch, source, start_index=0, layers=range(4))
    assert torch.equal(read_cache_kv(target, 1)[0], before)


def test_apply_kv_patches_overwrites_only_requested_slot(tiny_model, sample_inputs, other_inputs):
    target = _prefill_cache(tiny_model, other_inputs)
    source = _prefill_cache(tiny_model, sample_inputs)
    L, t = 1, 2

    patch = PatchSpec.for_layers_tokens([L], [t], [Component.K, Component.V])
    apply_kv_patches_to_cache(target, patch, source, start_index=5, layers=range(4))

    # Patched slot now equals source; a neighbouring slot is untouched.
    tgt_k = read_cache_kv(target, L)[0]
    src_k = read_cache_kv(source, L)[0]
    assert torch.equal(tgt_k[:, :, t, :], src_k[:, :, t, :])
    assert not torch.equal(tgt_k[:, :, t + 1, :], src_k[:, :, t + 1, :])


def test_apply_kv_patches_skips_tokens_at_or_after_start(tiny_model, sample_inputs, other_inputs):
    target = _prefill_cache(tiny_model, other_inputs)
    source = _prefill_cache(tiny_model, sample_inputs)
    L, t = 0, 6  # t >= start_index -> must NOT be touched in-cache

    before = read_cache_kv(target, L)[0][:, :, t, :].clone()
    patch = PatchSpec.for_layers_tokens([L], [t], [Component.K])
    apply_kv_patches_to_cache(target, patch, source, start_index=5, layers=range(4))
    assert torch.equal(read_cache_kv(target, L)[0][:, :, t, :], before)


def test_apply_kv_patches_unknown_layer_raises(tiny_model, sample_inputs):
    target = _prefill_cache(tiny_model, sample_inputs)
    source = _prefill_cache(tiny_model, sample_inputs)
    patch = PatchSpec.for_layers_tokens([99], [0], [Component.K])
    with pytest.raises(IndexError, match="layer 99"):
        apply_kv_patches_to_cache(target, patch, source, start_index=5, layers=range(4))


def test_apply_kv_patches_token_out_of_bounds_raises(tiny_model, sample_inputs):
    # Crop the source so a low token index is out of bounds for it.
    target = _prefill_cache(tiny_model, sample_inputs)
    source = _prefill_cache(tiny_model, sample_inputs)
    crop_dynamic_cache(source, 2)  # source now only has tokens [0, 2)
    patch = PatchSpec.for_layers_tokens([0], [3], [Component.K])
    with pytest.raises(IndexError, match="out of bounds"):
        apply_kv_patches_to_cache(target, patch, source, start_index=5, layers=range(4))
