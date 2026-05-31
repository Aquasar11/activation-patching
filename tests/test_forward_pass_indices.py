"""Validate `forward_pass_indices` behaviour in both modes."""
from __future__ import annotations

import pytest
import torch

from actpatch import ActivationPatcher, CacheSpec, Component, PatchSpec


def test_online_subset_indices_runs(tiny_model, tiny_adapter, sample_inputs):
    """Passing an explicit (contiguous) subset in online mode should run."""
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens(layers=[0], tokens=[0], components=[Component.K])
    src_cache = patcher.cache_source(sample_inputs, spec)
    out = patcher.patched_forward(
        sample_inputs,
        src_cache,
        PatchSpec(patches={}),
        mode="online",
        forward_pass_indices=[2, 3, 4, 5],
    )
    assert out.logits.shape[1] == 4


def test_online_out_of_range_raises(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens(layers=[0], tokens=[0], components=[Component.K])
    src_cache = patcher.cache_source(sample_inputs, spec)
    with pytest.raises(IndexError):
        patcher.patched_forward(
            sample_inputs,
            src_cache,
            PatchSpec(patches={}),
            mode="online",
            forward_pass_indices=[10],  # T=8 in fixture
        )


def test_offline_non_contiguous_raises(tiny_model, tiny_adapter, sample_inputs):
    patcher = ActivationPatcher(tiny_model, tiny_adapter)
    spec = CacheSpec.for_layers_tokens(layers=[0], tokens=[0], components=[Component.K])
    src_cache = patcher.cache_source(sample_inputs, spec)
    with pytest.raises(ValueError, match="contiguous"):
        patcher.patched_forward(
            sample_inputs,
            src_cache,
            PatchSpec(patches={}),
            mode="offline",
            start_index=4,
            forward_pass_indices=[4, 6, 7],
        )
