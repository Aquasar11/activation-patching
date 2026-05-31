"""K/V patches: the live K/V cache reflects the patched source values."""
from __future__ import annotations

import torch

from actpatch import ActivationPatcher, CacheSpec, Component, PatchSpec
from actpatch.cache_ops import read_cache_kv


def test_k_patch_writes_source_k_into_cache(tiny_model, tiny_adapter, sample_inputs, other_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    L, t = 1, 2

    # Source K at (L, t).
    spec = CacheSpec.for_layers_tokens(layers=[L], tokens=[t], components=[Component.K])
    src_cache = patcher.cache_source(sample_inputs, spec)

    # Reference: get the SOURCE's post-update K at (L, t).
    source_run = tiny_model(**sample_inputs, use_cache=True)
    src_k_cache = read_cache_kv(source_run.past_key_values, L)[0][:, :, t, :]

    # Patched target run: should write source's k_proj output at (L, t) into target cache.
    patch = PatchSpec.for_layers_tokens(layers=[L], tokens=[t], components=[Component.K])
    target_with_cache = dict(other_inputs)
    target_with_cache["use_cache"] = True
    out = patcher.patched_forward(target_with_cache, src_cache, patch, mode="online")
    patched_k_cache = read_cache_kv(out.past_key_values, L)[0][:, :, t, :]

    assert torch.allclose(patched_k_cache, src_k_cache, atol=1e-6)


def test_v_patch_writes_source_v_into_cache(tiny_model, tiny_adapter, sample_inputs, other_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    L, t = 1, 2
    spec = CacheSpec.for_layers_tokens(layers=[L], tokens=[t], components=[Component.V])
    src_cache = patcher.cache_source(sample_inputs, spec)
    source_run = tiny_model(**sample_inputs, use_cache=True)
    src_v_cache = read_cache_kv(source_run.past_key_values, L)[1][:, :, t, :]

    patch = PatchSpec.for_layers_tokens(layers=[L], tokens=[t], components=[Component.V])
    target_with_cache = dict(other_inputs)
    target_with_cache["use_cache"] = True
    out = patcher.patched_forward(target_with_cache, src_cache, patch, mode="online")
    patched_v_cache = read_cache_kv(out.past_key_values, L)[1][:, :, t, :]

    assert torch.allclose(patched_v_cache, src_v_cache, atol=1e-6)


def test_kv_patch_changes_logits(tiny_model, tiny_adapter, sample_inputs, other_inputs):
    """Sanity-check: patching K and V at an earlier token shifts logits at later positions."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    L, t = 0, 1
    spec = CacheSpec.for_layers_tokens(
        layers=[L], tokens=[t], components=[Component.K, Component.V]
    )
    src_cache = patcher.cache_source(sample_inputs, spec)

    patch = PatchSpec.for_layers_tokens(
        layers=[L], tokens=[t], components=[Component.K, Component.V]
    )
    with torch.no_grad():
        unpatched = tiny_model(**other_inputs).logits
    patched = patcher.patched_forward(other_inputs, src_cache, patch, mode="online").logits

    # K/V at position t affect attention output at positions >= t in layer L,
    # which propagates to all subsequent positions and layers.
    assert not torch.allclose(patched[:, t:, :], unpatched[:, t:, :], atol=1e-4)
