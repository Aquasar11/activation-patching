"""Dataclasses and the Component enum that drive activation patching.

A `PatchSpec` (or `CacheSpec`) is a nested mapping
    {layer_idx: {token_idx: frozenset(components)}}
describing which residual / K / V slots to patch (or capture) at which
(layer, token) coordinates. `SourceCache` is the populated container produced
by `ActivationPatcher.cache_source`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

import torch


class Component(str, Enum):
    """Which sub-component of a decoder block to capture / patch."""

    # Input to the decoder block (the residual stream just before pre-norm).
    RESID_IN = "resid_in"
    # Output of self_attn.k_proj. Pre-RoPE — standard mech-interp convention.
    K = "k"
    # Output of self_attn.v_proj.
    V = "v"


# layer_idx -> token_idx -> components to act on
_PatchMap = Dict[int, Dict[int, FrozenSet[Component]]]


def _normalize(
    layers: Iterable[int],
    tokens: Iterable[int],
    components: Iterable[Component],
) -> _PatchMap:
    comps = frozenset(components)
    token_list = list(tokens)
    return {int(L): {int(t): comps for t in token_list} for L in layers}


@dataclass(frozen=True)
class PatchSpec:
    """Where to apply patches during `patched_forward`."""

    patches: _PatchMap = field(default_factory=dict)

    @classmethod
    def for_layers_tokens(
        cls,
        layers: Iterable[int],
        tokens: Iterable[int],
        components: Iterable[Component],
    ) -> "PatchSpec":
        """Convenience constructor: same component set at every (layer, token)."""
        return cls(patches=_normalize(layers, tokens, components))

    def layers(self) -> FrozenSet[int]:
        return frozenset(self.patches.keys())

    def has_resid(self) -> bool:
        return any(
            Component.RESID_IN in comps
            for layer_map in self.patches.values()
            for comps in layer_map.values()
        )

    def has_kv(self) -> bool:
        return any(
            Component.K in comps or Component.V in comps
            for layer_map in self.patches.values()
            for comps in layer_map.values()
        )


@dataclass(frozen=True)
class CacheSpec:
    """What to record from the source forward pass.

    Mirrors `PatchSpec` so the same coordinates can be cached and later patched.
    """

    captures: _PatchMap = field(default_factory=dict)

    @classmethod
    def for_layers_tokens(
        cls,
        layers: Iterable[int],
        tokens: Iterable[int],
        components: Iterable[Component],
    ) -> "CacheSpec":
        return cls(captures=_normalize(layers, tokens, components))

    @classmethod
    def from_patch_spec(cls, spec: PatchSpec) -> "CacheSpec":
        """Build a CacheSpec that records exactly what `spec` will patch."""
        return cls(captures={L: dict(toks) for L, toks in spec.patches.items()})

    def layers(self) -> FrozenSet[int]:
        return frozenset(self.captures.keys())


@dataclass
class SourceCache:
    """Populated activations from a source forward pass.

    Tensors are stored per (layer_idx, token_idx). Residuals are single
    hidden-dim vectors; K/V are flat [num_kv_heads * head_dim] vectors as
    emitted by `k_proj` / `v_proj`.

    `kv_cache` is the full HF `DynamicCache` from the source prefill — used
    by offline mode to reuse source K/V before `start_index`.
    """

    resid_in: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)
    k_proj: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)
    v_proj: Dict[Tuple[int, int], torch.Tensor] = field(default_factory=dict)
    kv_cache: Optional[object] = None  # transformers.cache_utils.DynamicCache
    seq_len: int = 0
    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None

    def get(self, comp: Component, layer: int, token: int) -> torch.Tensor:
        store = {
            Component.RESID_IN: self.resid_in,
            Component.K: self.k_proj,
            Component.V: self.v_proj,
        }[comp]
        try:
            return store[(layer, token)]
        except KeyError as e:
            raise KeyError(
                f"SourceCache missing {comp.value} at (layer={layer}, token={token}). "
                f"Did you include it in the CacheSpec?"
            ) from e
