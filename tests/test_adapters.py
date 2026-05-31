"""Unit tests for the adapter protocol, registry, and layer resolution."""
from __future__ import annotations

import pytest
from torch import nn

import actpatch.adapters as adapters_mod
from actpatch import (
    InternVLAdapter,
    Qwen2_5_VLAdapter,
    get_adapter,
    register_adapter,
)
from actpatch.adapters.base import _attr_path, resolve_decoder_layers
from tiny_model import TinyDecoderLayer


@pytest.fixture
def clean_registry():
    """Snapshot and restore the global adapter registry around a test."""
    snapshot = dict(adapters_mod._REGISTRY)
    yield
    adapters_mod._REGISTRY.clear()
    adapters_mod._REGISTRY.update(snapshot)


# --- dispatch -------------------------------------------------------------

class _FakeQwen2_5_VLForConditionalGeneration:
    pass


class _FakeInternVLForConditionalGeneration:
    pass


class _FakeUnknownModel:
    pass


def test_get_adapter_dispatches_qwen():
    assert isinstance(get_adapter(_FakeQwen2_5_VLForConditionalGeneration()), Qwen2_5_VLAdapter)


def test_get_adapter_dispatches_internvl():
    assert isinstance(get_adapter(_FakeInternVLForConditionalGeneration()), InternVLAdapter)


def test_get_adapter_unknown_raises():
    with pytest.raises(KeyError, match="No adapter registered"):
        get_adapter(_FakeUnknownModel())


def test_register_adapter(clean_registry):
    class MyAdapter:
        pass

    class _FakeMyVLMModel:
        pass

    register_adapter("MyVLM", MyAdapter)
    assert isinstance(get_adapter(_FakeMyVLMModel()), MyAdapter)


# --- _attr_path -----------------------------------------------------------

def test_attr_path_resolves_and_misses():
    class A:
        pass

    root = A()
    root.b = A()
    root.b.c = 42
    assert _attr_path(root, "b.c") == 42
    assert _attr_path(root, "b.missing") is None
    assert _attr_path(root, "nope.at.all") is None


# --- resolve_decoder_layers ----------------------------------------------

def _layer():
    return TinyDecoderLayer(0, hidden=8, num_heads=2, num_kv_heads=1, head_dim=4)


def test_resolve_decoder_layers_finds_valid_path():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([_layer()])

    model = Model()
    layers, path = resolve_decoder_layers(model, ("language_model.layers",))
    assert path == "language_model.layers"
    assert len(layers) == 1


def test_resolve_decoder_layers_skips_layers_without_kv_proj():
    class VisionLayer(nn.Module):
        # No self_attn.k_proj/v_proj -> must be rejected.
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(4, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.Module()
            self.vision_tower.layers = nn.ModuleList([VisionLayer()])
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([_layer()])

    model = Model()
    # The vision path is tried first but rejected; the LLM path wins.
    layers, path = resolve_decoder_layers(
        model, ("vision_tower.layers", "language_model.layers")
    )
    assert path == "language_model.layers"


def test_resolve_decoder_layers_raises_when_nothing_matches():
    class Model(nn.Module):
        pass

    with pytest.raises(AttributeError, match="Could not locate decoder layers"):
        resolve_decoder_layers(Model(), ("a.b", "c.d"))


# --- TinyAdapter sanity ---------------------------------------------------

def test_tiny_adapter_surface(tiny_model, tiny_adapter):
    layers = tiny_adapter.get_decoder_layers(tiny_model)
    assert len(layers) == tiny_model.config.num_hidden_layers
    k_proj, v_proj = tiny_adapter.get_attn_kv_projs(layers[0])
    assert isinstance(k_proj, nn.Linear) and isinstance(v_proj, nn.Linear)
    assert tiny_adapter.get_image_token_id(tiny_model) == tiny_model.config.image_token_id
    assert tiny_adapter.num_kv_heads(tiny_model) == tiny_model.config.num_key_value_heads
    assert tiny_adapter.head_dim(tiny_model) == tiny_model.config.head_dim
