"""Unit tests for the Qwen2.5-VL and InternVL adapter methods.

These exercise the pure-Python layout/grid logic with lightweight fake models,
so the bug-prone grid-shape math is covered without downloading real weights.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from actpatch import InternVLAdapter, Qwen2_5_VLAdapter
from tiny_model import TinyDecoderLayer


def _decoder_layer():
    return TinyDecoderLayer(0, hidden=8, num_heads=2, num_kv_heads=1, head_dim=4)


def _llm(path_attr: str):
    """Build a fake model exposing decoder layers at the given nested path."""
    model = nn.Module()
    if path_attr == "language_model.layers":
        model.language_model = nn.Module()
        model.language_model.layers = nn.ModuleList([_decoder_layer()])
    elif path_attr == "language_model.model.layers":
        model.language_model = nn.Module()
        model.language_model.model = nn.Module()
        model.language_model.model.layers = nn.ModuleList([_decoder_layer()])
    return model


# --- Qwen2.5-VL -----------------------------------------------------------

def _qwen_config(with_head_dim=True):
    text = SimpleNamespace(num_key_value_heads=2, num_attention_heads=4, hidden_size=32)
    if with_head_dim:
        text.head_dim = 8
    return SimpleNamespace(
        image_token_id=151655,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        text_config=text,
    )


def test_qwen_layers_and_projs():
    adapter = Qwen2_5_VLAdapter()
    model = _llm("language_model.layers")
    layers = adapter.get_decoder_layers(model)
    assert len(layers) == 1
    k_proj, v_proj = adapter.get_attn_kv_projs(layers[0])
    assert isinstance(k_proj, nn.Linear) and isinstance(v_proj, nn.Linear)


def test_qwen_token_id_and_head_dims():
    adapter = Qwen2_5_VLAdapter()
    model = _llm("language_model.layers")
    model.config = _qwen_config(with_head_dim=True)
    assert adapter.get_image_token_id(model) == 151655
    assert adapter.num_kv_heads(model) == 2
    assert adapter.head_dim(model) == 8


def test_qwen_head_dim_computed_when_absent():
    adapter = Qwen2_5_VLAdapter()
    model = _llm("language_model.layers")
    model.config = _qwen_config(with_head_dim=False)
    # hidden_size / num_attention_heads = 32 / 4
    assert adapter.head_dim(model) == 8


def test_qwen_image_grid_shape_applies_spatial_merge():
    adapter = Qwen2_5_VLAdapter()
    model = _llm("language_model.layers")
    model.config = _qwen_config()
    inputs = {"image_grid_thw": torch.tensor([[1, 8, 12]])}
    assert adapter.image_grid_shape(inputs, model) == (4, 6)  # //2 each


def test_qwen_image_grid_shape_errors():
    adapter = Qwen2_5_VLAdapter()
    model = _llm("language_model.layers")
    model.config = _qwen_config()

    with pytest.raises(KeyError, match="image_grid_thw"):
        adapter.image_grid_shape({}, model)
    with pytest.raises(NotImplementedError, match="single image"):
        adapter.image_grid_shape({"image_grid_thw": torch.tensor([[1, 8, 12], [1, 8, 12]])}, model)
    with pytest.raises(NotImplementedError, match="Video"):
        adapter.image_grid_shape({"image_grid_thw": torch.tensor([[2, 8, 12]])}, model)


# --- InternVL -------------------------------------------------------------

def _internvl_config(image_seq_length=None, num_image_token=None, with_head_dim=True):
    text = SimpleNamespace(num_key_value_heads=4, num_attention_heads=8, hidden_size=64)
    if with_head_dim:
        text.head_dim = 8
    cfg = SimpleNamespace(
        image_token_id=92546,
        text_config=text,
        vision_config=SimpleNamespace(),
    )
    if image_seq_length is not None:
        cfg.image_seq_length = image_seq_length
    if num_image_token is not None:
        cfg.vision_config.num_image_token = num_image_token
    return cfg


def test_internvl_layers_resolved_under_model_path():
    adapter = InternVLAdapter()
    model = _llm("language_model.model.layers")
    layers = adapter.get_decoder_layers(model)
    assert len(layers) == 1
    assert adapter._layer_path == "language_model.model.layers"


def test_internvl_token_id_and_head_dims():
    adapter = InternVLAdapter()
    model = _llm("language_model.model.layers")
    model.config = _internvl_config(image_seq_length=256)
    assert adapter.get_image_token_id(model) == 92546
    assert adapter.num_kv_heads(model) == 4
    assert adapter.head_dim(model) == 8


def test_internvl_get_attn_kv_projs():
    adapter = InternVLAdapter()
    model = _llm("language_model.model.layers")
    layer = adapter.get_decoder_layers(model)[0]
    k_proj, v_proj = adapter.get_attn_kv_projs(layer)
    assert isinstance(k_proj, nn.Linear) and isinstance(v_proj, nn.Linear)


def test_internvl_head_dim_computed_when_absent():
    adapter = InternVLAdapter()
    model = _llm("language_model.model.layers")
    model.config = _internvl_config(image_seq_length=256, with_head_dim=False)
    # hidden_size / num_attention_heads = 64 / 8
    assert adapter.head_dim(model) == 8


def test_internvl_image_grid_shape_square_root():
    adapter = InternVLAdapter()
    model = _llm("language_model.model.layers")
    model.config = _internvl_config(image_seq_length=256)
    assert adapter.image_grid_shape({}, model) == (16, 16)


def test_internvl_image_grid_shape_from_vision_config():
    adapter = InternVLAdapter()
    model = _llm("language_model.model.layers")
    model.config = _internvl_config(num_image_token=64)
    assert adapter.image_grid_shape({}, model) == (8, 8)


def test_internvl_image_grid_shape_non_square_raises():
    adapter = InternVLAdapter()
    model = _llm("language_model.model.layers")
    model.config = _internvl_config(image_seq_length=10)
    with pytest.raises(NotImplementedError, match="Non-square"):
        adapter.image_grid_shape({}, model)


def test_internvl_image_grid_shape_missing_raises():
    adapter = InternVLAdapter()
    model = _llm("language_model.model.layers")
    model.config = _internvl_config()  # neither image_seq_length nor num_image_token
    with pytest.raises(KeyError, match="num_image_token"):
        adapter.image_grid_shape({}, model)
