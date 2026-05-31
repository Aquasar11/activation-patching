"""PyTorch forward hooks for capturing and patching activations.

Two flavours of hooks are produced here:

* **Capture hooks** record activations into a `SourceCache` during the source
  forward pass.
* **Patch hooks** overwrite activations from a `SourceCache` during the
  target forward pass.

All hooks are owned by a `HookHandle` context manager that registers them on
`__enter__` and removes them on `__exit__` (even when an exception escapes the
forward pass). No global state, no monkey-patching of `forward()` methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence

import torch
from torch import nn

from ._logging import get_logger
from .specs import CacheSpec, Component, PatchSpec, SourceCache

logger = get_logger(__name__)


@dataclass
class ForwardContext:
    """Translates target-sequence token indices to live-tensor row indices.

    The live forward may operate on a strict subset of the target sequence
    (offline mode), so a patch coordinate `tok` in the *target* seq may map to
    a different row index in the hidden_states tensor — or to no row at all
    (when the position is served from the KV cache instead and patches at
    that position belong on the cache, not on the live forward).
    """

    forward_pass_indices: Optional[Sequence[int]]
    # When set, only these target-seq positions are valid for the live forward.
    # In offline mode this is the contiguous suffix [start_index, T).

    def __post_init__(self) -> None:
        if self.forward_pass_indices is None:
            self._map: Optional[Dict[int, int]] = None
        else:
            self._map = {int(t): i for i, t in enumerate(self.forward_pass_indices)}

    def target_to_local(self, target_idx: int) -> Optional[int]:
        if self._map is None:
            return int(target_idx)
        return self._map.get(int(target_idx))


# ---------------------------------------------------------------------------
# Capture hooks
# ---------------------------------------------------------------------------

def _capture_resid_pre_hook(
    layer_idx: int,
    tokens_to_components: Dict[int, FrozenSet[Component]],
    cache: SourceCache,
    keep_on_device: bool,
):
    """Pre-hook on a decoder layer that records its residual input."""

    def hook(module: nn.Module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None:
            return
        for tok, comps in tokens_to_components.items():
            if Component.RESID_IN not in comps:
                continue
            row = hs[:, tok, :].detach()
            cache.resid_in[(layer_idx, tok)] = row if keep_on_device else row.cpu()
            logger.debug(
                "capture resid_in: layer=%d token=%d shape=%s", layer_idx, tok, tuple(row.shape)
            )
        return None  # do not modify inputs

    return hook


def _capture_kv_hook(
    layer_idx: int,
    component: Component,
    tokens_to_components: Dict[int, FrozenSet[Component]],
    cache: SourceCache,
    keep_on_device: bool,
):
    """Forward hook on `k_proj` / `v_proj` that records selected rows."""

    store = cache.k_proj if component is Component.K else cache.v_proj

    def hook(module: nn.Module, inputs, output: torch.Tensor):
        # output: [B, T, kv_dim]
        for tok, comps in tokens_to_components.items():
            if component not in comps:
                continue
            row = output[:, tok, :].detach()
            store[(layer_idx, tok)] = row if keep_on_device else row.cpu()
            logger.debug(
                "capture %s: layer=%d token=%d shape=%s",
                component.value, layer_idx, tok, tuple(row.shape),
            )
        return None

    return hook


# ---------------------------------------------------------------------------
# Patch hooks
# ---------------------------------------------------------------------------

def _patch_resid_pre_hook(
    layer_idx: int,
    tokens_to_components: Dict[int, FrozenSet[Component]],
    cache: SourceCache,
    ctx: ForwardContext,
):
    """Pre-hook on a decoder layer that overwrites the residual input."""

    def hook(module: nn.Module, args, kwargs):
        has_kw = "hidden_states" in kwargs
        hs = kwargs["hidden_states"] if has_kw else args[0]
        modified = False
        new_hs = hs
        for tok, comps in tokens_to_components.items():
            if Component.RESID_IN not in comps:
                continue
            local = ctx.target_to_local(tok)
            if local is None:
                continue
            if not modified:
                new_hs = hs.clone()
                modified = True
            src = cache.resid_in[(layer_idx, tok)].to(new_hs.device, dtype=new_hs.dtype)
            new_hs[:, local, :] = src
            logger.debug(
                "patch resid_in: layer=%d token=%d -> local_row=%d", layer_idx, tok, local
            )
        if not modified:
            return None
        if has_kw:
            kwargs["hidden_states"] = new_hs
            return args, kwargs
        return (new_hs,) + tuple(args[1:]), kwargs

    return hook


def _patch_kv_hook(
    layer_idx: int,
    component: Component,
    tokens_to_components: Dict[int, FrozenSet[Component]],
    cache: SourceCache,
    ctx: ForwardContext,
):
    """Forward hook on `k_proj` / `v_proj` overwriting selected rows."""

    store = cache.k_proj if component is Component.K else cache.v_proj

    def hook(module: nn.Module, inputs, output: torch.Tensor):
        modified = False
        new_out = output
        for tok, comps in tokens_to_components.items():
            if component not in comps:
                continue
            local = ctx.target_to_local(tok)
            if local is None:
                continue
            if not modified:
                new_out = output.clone()
                modified = True
            src = store[(layer_idx, tok)].to(new_out.device, dtype=new_out.dtype)
            new_out[:, local, :] = src
            logger.debug(
                "patch %s: layer=%d token=%d -> local_row=%d",
                component.value, layer_idx, tok, local,
            )
        return new_out if modified else None

    return hook


# ---------------------------------------------------------------------------
# Handle (context manager)
# ---------------------------------------------------------------------------

class HookHandle:
    """Registers a batch of PyTorch hooks on `__enter__`; removes on `__exit__`."""

    def __init__(self) -> None:
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def add_pre_hook(self, module: nn.Module, hook: Callable) -> None:
        self._handles.append(
            module.register_forward_pre_hook(hook, with_kwargs=True)
        )

    def add_forward_hook(self, module: nn.Module, hook: Callable) -> None:
        self._handles.append(module.register_forward_hook(hook))

    def __enter__(self) -> "HookHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
        self._handles.clear()


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def register_capture_hooks(
    handle: HookHandle,
    layers: Sequence[nn.Module],
    get_kv_projs: Callable[[nn.Module], "tuple[nn.Module, nn.Module]"],
    cache_spec: CacheSpec,
    cache: SourceCache,
    keep_on_device: bool,
) -> None:
    """Attach capture hooks for every (layer, token, component) in `cache_spec`."""
    n_before = len(handle._handles)
    for layer_idx, token_map in cache_spec.captures.items():
        if layer_idx >= len(layers):
            raise IndexError(
                f"CacheSpec references layer {layer_idx} but model has {len(layers)} layers."
            )
        layer = layers[layer_idx]
        wants_resid = any(Component.RESID_IN in c for c in token_map.values())
        wants_k = any(Component.K in c for c in token_map.values())
        wants_v = any(Component.V in c for c in token_map.values())

        if wants_resid:
            handle.add_pre_hook(
                layer,
                _capture_resid_pre_hook(layer_idx, token_map, cache, keep_on_device),
            )
        if wants_k or wants_v:
            k_proj, v_proj = get_kv_projs(layer)
            if wants_k:
                handle.add_forward_hook(
                    k_proj,
                    _capture_kv_hook(layer_idx, Component.K, token_map, cache, keep_on_device),
                )
            if wants_v:
                handle.add_forward_hook(
                    v_proj,
                    _capture_kv_hook(layer_idx, Component.V, token_map, cache, keep_on_device),
                )
    logger.debug(
        "registered %d capture hooks across %d layers (keep_on_device=%s)",
        len(handle._handles) - n_before, len(cache_spec.captures), keep_on_device,
    )


def register_patch_hooks(
    handle: HookHandle,
    layers: Sequence[nn.Module],
    get_kv_projs: Callable[[nn.Module], "tuple[nn.Module, nn.Module]"],
    patch_spec: PatchSpec,
    cache: SourceCache,
    ctx: ForwardContext,
    skip_tokens_below: Optional[int] = None,
) -> None:
    """Attach patch hooks.

    `skip_tokens_below` is used in offline mode: token positions strictly less
    than this index are NOT patched via hooks (they are patched into the
    KV cache directly by `cache_ops`); residual patches at those positions are
    silently skipped (no residual is reconstructible from a KV cache alone).
    """
    n_before = len(handle._handles)
    for layer_idx, token_map in patch_spec.patches.items():
        if layer_idx >= len(layers):
            raise IndexError(
                f"PatchSpec references layer {layer_idx} but model has {len(layers)} layers."
            )
        if skip_tokens_below is not None:
            token_map = {
                t: c for t, c in token_map.items() if t >= skip_tokens_below
            }
            if not token_map:
                continue

        layer = layers[layer_idx]
        wants_resid = any(Component.RESID_IN in c for c in token_map.values())
        wants_k = any(Component.K in c for c in token_map.values())
        wants_v = any(Component.V in c for c in token_map.values())

        if wants_resid:
            handle.add_pre_hook(
                layer,
                _patch_resid_pre_hook(layer_idx, token_map, cache, ctx),
            )
        if wants_k or wants_v:
            k_proj, v_proj = get_kv_projs(layer)
            if wants_k:
                handle.add_forward_hook(
                    k_proj,
                    _patch_kv_hook(layer_idx, Component.K, token_map, cache, ctx),
                )
            if wants_v:
                handle.add_forward_hook(
                    v_proj,
                    _patch_kv_hook(layer_idx, Component.V, token_map, cache, ctx),
                )
    logger.debug(
        "registered %d patch hooks across %d layers (skip_tokens_below=%s)",
        len(handle._handles) - n_before, len(patch_spec.patches), skip_tokens_below,
    )
