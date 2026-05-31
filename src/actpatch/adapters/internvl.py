"""Adapter for HuggingFace `InternVLForConditionalGeneration` (InternVL 3.5).

The LLM decoder path varies with the text backbone:
    model.language_model.model.layers   # most Llama/Qwen2 text backbones
    model.language_model.layers         # some fused-API revisions

The adapter introspects on construction and caches the chosen attribute path.
"""
from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from .base import resolve_decoder_layers

# The text-backbone path varies (Qwen2 vs InternLM2); the resolver introspects.
_CANDIDATE_LAYER_PATHS = (
    "language_model.model.layers",
    "language_model.layers",
    "model.language_model.layers",
    "model.language_model.model.layers",
)


class InternVLAdapter:
    name = "internvl"

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

    def _text_cfg(self, model: nn.Module):
        cfg = model.config
        return getattr(cfg, "text_config", cfg)

    def num_kv_heads(self, model: nn.Module) -> int:
        return int(self._text_cfg(model).num_key_value_heads)

    def head_dim(self, model: nn.Module) -> int:
        text_cfg = self._text_cfg(model)
        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
        return int(head_dim)

    def image_grid_shape(
        self, inputs: Mapping[str, object], model: nn.Module
    ) -> tuple[int, int]:
        # InternVL emits a fixed-count `num_image_token` per tile after the
        # pixel-shuffle downsample. For the common single-tile path, the grid
        # is the square root of `num_image_token`.
        vis_cfg = getattr(model.config, "vision_config", model.config)
        num_image_token = getattr(model.config, "image_seq_length", None) or getattr(
            vis_cfg, "num_image_token", None
        )
        if num_image_token is None:
            raise KeyError(
                "InternVL config missing `image_seq_length` / `num_image_token`; "
                "cannot derive image grid shape."
            )
        side = int(num_image_token**0.5)
        if side * side != int(num_image_token):
            raise NotImplementedError(
                f"Non-square image grids not supported (num_image_token={num_image_token})."
            )
        return side, side
