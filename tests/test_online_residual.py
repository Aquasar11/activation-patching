"""Online residual-stream patches: results match a hand-rolled forward."""
from __future__ import annotations

import copy

import torch

from actpatch import ActivationPatcher, CacheSpec, Component, PatchSpec


def _manual_forward_with_resid_patch(model, input_ids, layer_idx, token_idx, value):
    """Reference: run the tiny model manually, overwriting hidden state at the
    input to `layer_idx` at `token_idx` with `value`."""
    with torch.no_grad():
        h = model.embed_tokens(input_ids)
        for i, layer in enumerate(model.layers):
            if i == layer_idx:
                h = h.clone()
                h[:, token_idx, :] = value.to(h.device, dtype=h.dtype)
            h = layer(h)[0]
        h = model.final_norm(h)
        return model.lm_head(h)


def test_resid_patch_matches_manual_forward(tiny_model, tiny_adapter, sample_inputs, other_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    L, t = 2, 3
    spec = CacheSpec.for_layers_tokens(layers=[L], tokens=[t], components=[Component.RESID_IN])
    src_cache = patcher.cache_source(sample_inputs, spec)
    cached_value = src_cache.resid_in[(L, t)].clone()

    patch = PatchSpec.for_layers_tokens(
        layers=[L], tokens=[t], components=[Component.RESID_IN]
    )
    out = patcher.patched_forward(other_inputs, src_cache, patch, mode="online")
    patched_logits = out.logits

    ref_logits = _manual_forward_with_resid_patch(
        tiny_model, other_inputs["input_ids"], L, t, cached_value
    )
    assert torch.allclose(patched_logits, ref_logits, atol=1e-5)


def test_resid_patch_differs_from_unpatched(tiny_model, tiny_adapter, sample_inputs, other_inputs):
    """Sanity-check: the patch actually changes the output (not silently a no-op)."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens(layers=[0], tokens=[2], components=[Component.RESID_IN])
    src_cache = patcher.cache_source(sample_inputs, spec)

    patch = PatchSpec.for_layers_tokens(layers=[0], tokens=[2], components=[Component.RESID_IN])
    with torch.no_grad():
        unpatched = tiny_model(**other_inputs).logits
    patched = patcher.patched_forward(other_inputs, src_cache, patch, mode="online").logits

    assert not torch.allclose(patched, unpatched, atol=1e-4)
