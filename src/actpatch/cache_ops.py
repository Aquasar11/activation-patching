"""DynamicCache access + mutation for offline activation patching.

In offline mode, the live forward only processes positions `[start_index, T)`.
Patches at positions `< start_index` cannot be applied by hooks (no live row
exists for them) — instead they must be written into the target's KV cache
before the live forward begins.

The values come from `SourceCache.kv_cache`, the `DynamicCache` produced by
the source prefill. Those values are *post-RoPE* K and V, which is exactly
what the cache stores, so the copy is a straight slice-overwrite.

Residual-stream patches at `tok < start_index` cannot be applied at all —
a residual stream value at layer L is not reconstructible from the K/V cache.
We silently skip them (and `register_patch_hooks` skips them too). Callers
who need such patches should use online mode.

DynamicCache has had two storage layouts across transformers versions:

* **legacy**: `cache.key_cache[i]` / `cache.value_cache[i]` are tensors.
* **current** (post ~4.54 refactor): `cache.layers[i].keys` / `.values`.

All access here goes through `read_cache_kv` / `cache_num_layers` so the rest
of the module is agnostic to which layout is installed.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable

import torch

from ._logging import get_logger
from .specs import Component, PatchSpec

logger = get_logger(__name__)


def cache_num_layers(cache) -> int:
    """Number of layers stored in a DynamicCache, across API versions."""
    if hasattr(cache, "layers"):
        return len(cache.layers)
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(cache)


def read_cache_kv(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the `(keys, values)` tensors for a layer.

    The returned tensors are the *live* stored tensors (not copies), so
    in-place edits — e.g. `keys[:, :, tok, :] = ...` — persist into the cache.
    Shape is `[B, num_kv_heads, T, head_dim]`.
    """
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    raise TypeError(
        f"Unsupported cache type {type(cache).__name__!r}: expected a DynamicCache "
        f"with either `.layers` or `.key_cache`/`.value_cache`."
    )


def crop_dynamic_cache(cache, length: int):
    """Crop every layer's K and V to the first `length` positions (in place)."""
    logger.debug("crop cache to length=%d (%d layers)", length, cache_num_layers(cache))
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    # Legacy fallback.
    for i in range(cache_num_layers(cache)):
        k, v = read_cache_kv(cache, i)
        cache.key_cache[i] = k[..., :length, :]
        cache.value_cache[i] = v[..., :length, :]
    for attr in ("_seen_tokens", "seen_tokens"):
        if hasattr(cache, attr):
            setattr(cache, attr, length)
    return cache


def clone_dynamic_cache(cache):
    """Deep-copy a DynamicCache so multiple patch runs can share one source."""
    return copy.deepcopy(cache)


def apply_kv_patches_to_cache(
    target_cache,
    patch_spec: PatchSpec,
    source_kv_cache,
    start_index: int,
    layers: Iterable[int],
) -> None:
    """Overwrite K/V slots in `target_cache` with values from `source_kv_cache`.

    Only positions strictly less than `start_index` are touched — patches at
    `tok >= start_index` are handled by live hooks during the forward.

    Args:
        target_cache: the `DynamicCache` that will be consumed by the live
            forward. Must already contain entries for positions in
            `[0, start_index)`.
        patch_spec: source of (layer, token, component) patch coordinates.
        source_kv_cache: the source prefill cache, populated for the full
            source sequence; must hold the patched (layer, token) slots.
        start_index: positions in `[0, start_index)` are patched in-cache;
            positions `>= start_index` are skipped here.
        layers: layer indices that exist in the target cache (sanity-check).
    """
    if start_index <= 0:
        logger.debug("apply_kv_patches_to_cache: start_index=%d, nothing to patch", start_index)
        return  # Nothing to patch in-cache.

    n_patched = 0
    layer_set = {int(x) for x in layers}
    for layer_idx, token_map in patch_spec.patches.items():
        if layer_idx not in layer_set:
            raise IndexError(
                f"PatchSpec references layer {layer_idx} but target cache has "
                f"layers {sorted(layer_set)}."
            )
        tgt_k, tgt_v = read_cache_kv(target_cache, layer_idx)
        src_k, src_v = read_cache_kv(source_kv_cache, layer_idx)

        for tok, comps in token_map.items():
            if tok >= start_index:
                continue
            if tok >= tgt_k.shape[-2] or tok >= src_k.shape[-2]:
                raise IndexError(
                    f"Token {tok} out of bounds for cache at layer {layer_idx}: "
                    f"target T={tgt_k.shape[-2]}, source T={src_k.shape[-2]}."
                )
            if Component.K in comps:
                tgt_k[:, :, tok, :] = src_k[:, :, tok, :].to(
                    tgt_k.device, dtype=tgt_k.dtype
                )
                n_patched += 1
                logger.debug("cache patch K: layer=%d token=%d", layer_idx, tok)
            if Component.V in comps:
                tgt_v[:, :, tok, :] = src_v[:, :, tok, :].to(
                    tgt_v.device, dtype=tgt_v.dtype
                )
                n_patched += 1
                logger.debug("cache patch V: layer=%d token=%d", layer_idx, tok)
    logger.debug("apply_kv_patches_to_cache: patched %d cache slots", n_patched)
