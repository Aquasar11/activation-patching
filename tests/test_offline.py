"""Offline mode: KV-cache reuse + cache-mutation for pre-`start_index` patches."""
from __future__ import annotations

import torch

from actpatch import ActivationPatcher, CacheSpec, Component, PatchSpec
from actpatch.cache_ops import crop_dynamic_cache, read_cache_kv


def test_offline_no_patch_equals_unpatched(tiny_model, tiny_adapter, sample_inputs):
    """When patch_spec is empty, offline output should equal a normal forward."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    # Cache something arbitrary just to have a source.
    spec = CacheSpec.for_layers_tokens(layers=[0], tokens=[0], components=[Component.K])
    src_cache = patcher.cache_source(sample_inputs, spec)

    empty_patch = PatchSpec(patches={})
    with torch.no_grad():
        full = tiny_model(**sample_inputs).logits
    out = patcher.patched_forward(
        sample_inputs, src_cache, empty_patch, mode="offline", start_index=5
    )
    # Offline returns logits only for the live suffix [5, T).
    assert torch.allclose(out.logits, full[:, 5:, :], atol=1e-5)


def test_offline_online_agree_for_patches_in_suffix(
    tiny_model, tiny_adapter, sample_inputs, other_inputs
):
    """Patches at tok >= start_index should produce the same result in either mode."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    L, t = 2, 6  # t > start_index
    spec = CacheSpec.for_layers_tokens(layers=[L], tokens=[t], components=[Component.RESID_IN])
    src_cache = patcher.cache_source(sample_inputs, spec)
    patch = PatchSpec.for_layers_tokens(layers=[L], tokens=[t], components=[Component.RESID_IN])

    online_logits = patcher.patched_forward(
        other_inputs, src_cache, patch, mode="online"
    ).logits
    offline_logits = patcher.patched_forward(
        other_inputs, src_cache, patch, mode="offline", start_index=5
    ).logits

    # Compare on the overlapping range.
    assert torch.allclose(online_logits[:, 5:, :], offline_logits, atol=1e-4)


def test_offline_kv_cache_patch_matches_manual(
    tiny_model, tiny_adapter, sample_inputs, other_inputs
):
    """Patching K at a token < start_index should overwrite the target cache."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    L, t, start = 1, 2, 5

    spec = CacheSpec.for_layers_tokens(
        layers=[L], tokens=[t], components=[Component.K, Component.V]
    )
    src_cache = patcher.cache_source(sample_inputs, spec)
    patch = PatchSpec.for_layers_tokens(
        layers=[L], tokens=[t], components=[Component.K, Component.V]
    )

    # Reference: build target cache, overwrite slot manually, run live forward.
    with torch.no_grad():
        tgt_run = tiny_model(**other_inputs, use_cache=True)
        src_run = tiny_model(**sample_inputs, use_cache=True)
    ref_cache = tgt_run.past_key_values
    # Crop to start_index, then overwrite the K/V slot at (L, t) by hand.
    crop_dynamic_cache(ref_cache, start)
    ref_k, ref_v = read_cache_kv(ref_cache, L)
    src_k, src_v = read_cache_kv(src_run.past_key_values, L)
    ref_k[:, :, t, :] = src_k[:, :, t, :]
    ref_v[:, :, t, :] = src_v[:, :, t, :]

    # Run the live forward by hand on the suffix.
    suffix_ids = other_inputs["input_ids"][:, start:]
    cache_position = torch.arange(start, other_inputs["input_ids"].shape[-1])
    position_ids = cache_position.unsqueeze(0)
    attention_mask = torch.ones((1, other_inputs["input_ids"].shape[-1]), dtype=torch.long)
    with torch.no_grad():
        ref_out = tiny_model(
            input_ids=suffix_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=ref_cache,
            use_cache=True,
        )

    out = patcher.patched_forward(
        other_inputs, src_cache, patch, mode="offline", start_index=start
    )
    assert torch.allclose(out.logits, ref_out.logits, atol=1e-5)


def test_offline_requires_start_index(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens(layers=[0], tokens=[0], components=[Component.K])
    src_cache = patcher.cache_source(sample_inputs, spec)
    import pytest

    with pytest.raises(ValueError, match="start_index"):
        patcher.patched_forward(
            sample_inputs, src_cache, PatchSpec(patches={}), mode="offline"
        )
