"""Scenario and error-path tests for ActivationPatcher."""
from __future__ import annotations

import pytest
import torch

from actpatch import ActivationPatcher, CacheSpec, Component, PatchSpec

# --- error paths ----------------------------------------------------------

def test_unknown_mode_raises(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    with pytest.raises(ValueError, match="Unknown mode"):
        patcher.patched_forward(sample_inputs, src, PatchSpec(patches={}), mode="sideways")


def test_online_missing_input_ids_raises(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    with pytest.raises(KeyError, match="input_ids"):
        patcher.patched_forward({"attention_mask": torch.ones(1, 4)}, src, PatchSpec(patches={}))


def test_offline_start_index_out_of_range_raises(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    T = sample_inputs["input_ids"].shape[-1]
    empty = PatchSpec(patches={})
    # start_index == T leaves no live positions.
    with pytest.raises(ValueError, match="start_index"):
        patcher.patched_forward(sample_inputs, src, empty, mode="offline", start_index=T)
    with pytest.raises(ValueError, match="start_index"):
        patcher.patched_forward(sample_inputs, src, empty, mode="offline", start_index=-1)


def test_offline_missing_input_ids_raises(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    with pytest.raises(KeyError, match="input_ids"):
        patcher.patched_forward(
            {"attention_mask": torch.ones(1, 4)},
            src,
            PatchSpec(patches={}),
            mode="offline",
            start_index=2,
        )


def test_offline_start_index_zero_equals_plain_forward(tiny_model, tiny_adapter, sample_inputs):
    """start_index=0 builds an empty cache and runs the whole sequence live."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    with torch.no_grad():
        plain = tiny_model(**sample_inputs).logits
    out = patcher.patched_forward(
        sample_inputs, src, PatchSpec(patches={}), mode="offline", start_index=0
    )
    assert torch.allclose(out.logits, plain, atol=1e-5)


def test_patch_spec_out_of_range_layer_raises(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    bad = PatchSpec.for_layers_tokens([99], [0], [Component.RESID_IN])
    with pytest.raises(IndexError, match="layer 99"):
        patcher.patched_forward(sample_inputs, src, bad, mode="online")


def test_cache_spec_out_of_range_layer_raises(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    with pytest.raises(IndexError, match="layer 99"):
        patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([99], [0], [Component.K]))


# --- cache_source options -------------------------------------------------

def test_cache_source_metadata(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    cache = patcher.cache_source(
        sample_inputs, CacheSpec.for_layers_tokens([0], [1], [Component.RESID_IN])
    )
    assert cache.seq_len == sample_inputs["input_ids"].shape[-1]
    assert cache.dtype == cache.resid_in[(0, 1)].dtype
    assert cache.device == sample_inputs["input_ids"].device


def test_cache_source_without_kv_cache(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    cache = patcher.cache_source(
        sample_inputs,
        CacheSpec.for_layers_tokens([0], [0], [Component.K]),
        keep_kv_cache=False,
    )
    assert cache.kv_cache is None


def test_cache_source_keep_on_device(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    cache = patcher.cache_source(
        sample_inputs,
        CacheSpec.for_layers_tokens([0], [0], [Component.RESID_IN]),
        keep_on_device=True,
    )
    # On CPU this matches the model device; the point is no forced .cpu() copy error.
    assert cache.resid_in[(0, 0)].device == sample_inputs["input_ids"].device


# --- combined components --------------------------------------------------

def test_combined_resid_kv_multi_layer_changes_output(
    tiny_model, tiny_adapter, sample_inputs, other_inputs
):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    layers = [0, 1, 2]
    tokens = [1, 3]
    comps = [Component.RESID_IN, Component.K, Component.V]
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens(layers, tokens, comps))
    patch = PatchSpec.for_layers_tokens(layers, tokens, comps)

    with torch.no_grad():
        unpatched = tiny_model(**other_inputs).logits
    patched = patcher.patched_forward(other_inputs, src, patch, mode="online").logits
    assert not torch.allclose(patched, unpatched, atol=1e-4)


# --- offline residual semantics ------------------------------------------

def test_offline_residual_before_start_is_skipped(
    tiny_model, tiny_adapter, sample_inputs, other_inputs
):
    """Residual patches at tok < start_index must have no effect (cache-only region)."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    L, t, start = 1, 2, 5
    src = patcher.cache_source(
        sample_inputs, CacheSpec.for_layers_tokens([L], [t], [Component.RESID_IN])
    )
    resid_patch = PatchSpec.for_layers_tokens([L], [t], [Component.RESID_IN])

    skipped = patcher.patched_forward(
        other_inputs, src, resid_patch, mode="offline", start_index=start
    ).logits
    unpatched = patcher.patched_forward(
        other_inputs, src, PatchSpec(patches={}), mode="offline", start_index=start
    ).logits
    assert torch.allclose(skipped, unpatched, atol=1e-6)


# --- patching() context manager ------------------------------------------

def test_patching_context_matches_online_forward(
    tiny_model, tiny_adapter, sample_inputs, other_inputs
):
    """A full-sequence forward inside `patching()` equals an online patched_forward."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens([2], [3], [Component.RESID_IN])
    src = patcher.cache_source(sample_inputs, spec)
    patch = PatchSpec.for_layers_tokens([2], [3], [Component.RESID_IN])

    online = patcher.patched_forward(other_inputs, src, patch, mode="online").logits
    with patcher.patching(src, patch):
        with torch.no_grad():
            ctx_logits = tiny_model(**other_inputs).logits
    assert torch.allclose(online, ctx_logits, atol=1e-5)


def test_patching_context_removes_hooks_on_exit(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [1], [Component.K]))
    patch = PatchSpec.for_layers_tokens([0], [1], [Component.K])
    with patcher.patching(src, patch):
        pass
    # No patch hooks should linger on the decoder layers / their projections.
    layer = tiny_adapter.get_decoder_layers(tiny_model)[0]
    assert len(layer._forward_pre_hooks) == 0
    k_proj, _ = tiny_adapter.get_attn_kv_projs(layer)
    assert len(k_proj._forward_hooks) == 0


def test_patching_context_survives_incremental_decode(tiny_model, tiny_adapter, sample_inputs):
    """Within the context, a 1-token decode step must skip out-of-range patches.

    The patched position (token 3) is only present during the prefill; the
    follow-up single-token forward must not raise and must leave that step
    unpatched (its value is already in the KV cache).
    """
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([1], [3], [Component.K]))
    patch = PatchSpec.for_layers_tokens([1], [3], [Component.K])

    T = sample_inputs["input_ids"].shape[-1]
    with patcher.patching(src, patch):
        with torch.no_grad():
            out = tiny_model(**sample_inputs, use_cache=True)
            next_tok = out.logits[:, -1:].argmax(-1)
            step = tiny_model(
                input_ids=next_tok,
                past_key_values=out.past_key_values,
                use_cache=True,
                cache_position=torch.tensor([T]),
            )
    assert step.logits.shape[1] == 1  # decoded one token without error


# --- generation -----------------------------------------------------------

def test_patched_generate_single_token_shape(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    out = patcher.patched_generate(sample_inputs, src, PatchSpec(patches={}), max_new_tokens=1)
    assert out.shape == (1, 1)


def test_patched_generate_rejects_zero_tokens(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    with pytest.raises(ValueError, match="max_new_tokens"):
        patcher.patched_generate(sample_inputs, src, PatchSpec(patches={}), max_new_tokens=0)


def test_patched_generate_offline_single_token(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    src = patcher.cache_source(sample_inputs, CacheSpec.for_layers_tokens([0], [0], [Component.K]))
    out = patcher.patched_generate(
        sample_inputs, src, PatchSpec(patches={}), mode="offline", start_index=4, max_new_tokens=2
    )
    assert out.shape == (1, 2)


# --- online subset position_ids ------------------------------------------

def test_online_subset_sets_position_ids(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    sliced = patcher._slice_inputs_to_indices(sample_inputs, [2, 3, 4])
    assert sliced["input_ids"].shape[-1] == 3
    assert sliced["position_ids"].tolist() == [[2, 3, 4]]
