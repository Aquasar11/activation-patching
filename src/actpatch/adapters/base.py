"""Model-adapter protocol.

An adapter is the small surface that lets the model-agnostic patcher reach
into a specific HF model: where the decoder layers live, where K/V projections
live within a layer, the image-token id, and how to derive the post-merge
image-token grid shape.

New models need only a ~30-line adapter.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Protocol, Tuple, runtime_checkable

from torch import nn


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol implemented by model-specific adapters."""

    def get_decoder_layers(self, model: nn.Module) -> List[nn.Module]:
        """Return the list of LLM decoder-block modules in order."""

    def get_attn_kv_projs(self, layer: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Return `(k_proj, v_proj)` for the layer's self-attention module."""

    def get_image_token_id(self, model: nn.Module) -> int:
        """Return the integer id used to mark image-token positions in `input_ids`."""

    def num_kv_heads(self, model: nn.Module) -> int: ...

    def head_dim(self, model: nn.Module) -> int: ...

    def image_grid_shape(self, inputs: Mapping[str, object], model: nn.Module) -> Tuple[int, int]:
        """Return the post-merge `(H_grid, W_grid)` for a single-image input.

        Used by `mask_to_token_indices` to validate user-supplied 2D masks.
        """


def _attr_path(obj, path: str):
    """Resolve a dotted attribute path, returning the final object or None if missing."""
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur
