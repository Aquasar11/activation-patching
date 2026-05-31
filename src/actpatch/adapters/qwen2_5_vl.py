"""Adapter for HuggingFace `Qwen2_5_VLForConditionalGeneration`.

Verified layer paths (transformers main branch, May 2026):

    model.language_model.layers[i]            # Qwen2_5_VLDecoderLayer
        .self_attn.k_proj  / .v_proj          # nn.Linear
    model.config.image_token_id
    model.config.vision_config.spatial_merge_size
"""
from __future__ import annotations

from typing import List, Mapping, Tuple

from torch import nn

from .base import _attr_path


class Qwen2_5_VLAdapter:
    name = "qwen2_5_vl"

    def get_decoder_layers(self, model: nn.Module) -> List[nn.Module]:
        layers = _attr_path(model, "language_model.layers")
        if layers is None:
            # Older transformers revisions nest under `.model`.
            layers = _attr_path(model, "language_model.model.layers")
        if layers is None:
            raise AttributeError(
                "Could not locate Qwen2.5-VL decoder layers at "
                "`model.language_model.layers` or `model.language_model.model.layers`."
            )
        return list(layers)

    def get_attn_kv_projs(self, layer: nn.Module) -> Tuple[nn.Module, nn.Module]:
        attn = getattr(layer, "self_attn")
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
    ) -> Tuple[int, int]:
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
