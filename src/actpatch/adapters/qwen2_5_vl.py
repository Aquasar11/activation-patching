"""Adapter for HuggingFace `Qwen2_5_VLForConditionalGeneration`.

Verified layer paths (transformers main branch, May 2026):

    model.language_model.layers[i]            # Qwen2_5_VLDecoderLayer
        .self_attn.k_proj  / .v_proj          # nn.Linear
    model.config.image_token_id
    model.config.vision_config.spatial_merge_size
"""
from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .base import resolve_decoder_layers

# Ordered by likelihood across transformers revisions.
_CANDIDATE_LAYER_PATHS = (
    "language_model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.layers",
)


class Qwen2_5_VLAdapter:
    name = "qwen2_5_vl"

    def __init__(self) -> None:
        self._layer_path: str | None = None

    def get_decoder_layers(self, model: nn.Module) -> list[nn.Module]:
        layers, path = resolve_decoder_layers(model, _CANDIDATE_LAYER_PATHS)
        self._layer_path = path
        return layers

    def get_attn_kv_projs(self, layer: nn.Module) -> tuple[nn.Module, nn.Module]:
        attn = layer.self_attn
        return attn.k_proj, attn.v_proj

    def get_image_token_id(self, model: nn.Module) -> int:
        return int(model.config.image_token_id)

    def num_kv_heads(self, model: nn.Module) -> int:
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        return int(text_cfg.num_key_value_heads)

    def head_dim(self, model: nn.Module) -> int:
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        # Qwen2/Qwen2.5 text configs expose head_dim either directly or
        # implicitly as hidden_size // num_attention_heads.
        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
        return int(head_dim)

    def image_grid_shape(
        self, inputs: Mapping[str, object], model: nn.Module
    ) -> tuple[int, int]:
        # Qwen2.5-VL emits `image_grid_thw` of shape [num_images, 3] = (t, h, w).
        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is None:
            raise KeyError(
                "Qwen2.5-VL inputs must contain `image_grid_thw` "
                "(returned by Qwen2_5_VLProcessor) to compute the image grid."
            )
        if grid_thw.shape[0] != 1:
            raise NotImplementedError(
                "image_grid_shape currently supports a single image per sample."
            )
        merge = int(model.config.vision_config.spatial_merge_size)
        t, h, w = grid_thw[0].tolist()
        if t != 1:
            raise NotImplementedError("Video inputs (t>1) not supported in image_grid_shape.")
        return int(h // merge), int(w // merge)
