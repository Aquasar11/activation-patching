"""Unit tests for the spec dataclasses and the Component enum."""
from __future__ import annotations

import dataclasses

import pytest
import torch

from actpatch import CacheSpec, Component, PatchSpec, SourceCache


def test_component_values():
    assert Component.RESID_IN.value == "resid_in"
    assert Component.K.value == "k"
    assert Component.V.value == "v"
    # Component is a str-enum, so equality with the raw string holds.
    assert Component.K == "k"


def test_patchspec_for_layers_tokens_shape():
    spec = PatchSpec.for_layers_tokens([0, 1], [3, 5], [Component.RESID_IN, Component.K])
    assert set(spec.patches) == {0, 1}
    assert set(spec.patches[0]) == {3, 5}
    assert spec.patches[0][3] == frozenset({Component.RESID_IN, Component.K})
    # Same component set is shared at every (layer, token).
    assert spec.patches[1][5] == spec.patches[0][3]


def test_patchspec_layers_and_flags():
    resid_only = PatchSpec.for_layers_tokens([2], [0], [Component.RESID_IN])
    assert resid_only.layers() == frozenset({2})
    assert resid_only.has_resid() is True
    assert resid_only.has_kv() is False

    kv_only = PatchSpec.for_layers_tokens([0], [1], [Component.K, Component.V])
    assert kv_only.has_resid() is False
    assert kv_only.has_kv() is True

    empty = PatchSpec(patches={})
    assert empty.has_resid() is False
    assert empty.has_kv() is False
    assert empty.layers() == frozenset()


def test_cachespec_from_patch_spec_roundtrips_coordinates():
    patch = PatchSpec.for_layers_tokens([0, 3], [2, 4], [Component.V])
    cache_spec = CacheSpec.from_patch_spec(patch)
    assert cache_spec.captures.keys() == patch.patches.keys()
    for layer in patch.patches:
        assert cache_spec.captures[layer] == patch.patches[layer]


def test_cachespec_layers():
    spec = CacheSpec.for_layers_tokens([1, 4], [0], [Component.K])
    assert spec.layers() == frozenset({1, 4})


def test_specs_are_frozen():
    spec = PatchSpec.for_layers_tokens([0], [0], [Component.K])
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.patches = {}


def test_sourcecache_get_success_and_missing():
    cache = SourceCache()
    t = torch.zeros(1, 4)
    cache.resid_in[(0, 1)] = t
    cache.k_proj[(2, 3)] = t * 2

    assert cache.get(Component.RESID_IN, 0, 1) is t
    assert torch.equal(cache.get(Component.K, 2, 3), t * 2)

    with pytest.raises(KeyError, match="missing"):
        cache.get(Component.V, 0, 0)
    with pytest.raises(KeyError, match="missing"):
        cache.get(Component.RESID_IN, 9, 9)
