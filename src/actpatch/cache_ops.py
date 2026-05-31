"""DynamicCache mutation for offline activation patching.

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
"""
from __future__ import annotations

from typing import Iterable

from .specs import Component, PatchSpec


def _layer_caches(cache):
    """Return (key_cache_list, value_cache_list) for either old-style
    DynamicCache (`.key_cache` / `.value_cache` lists) or newer revisions."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache, cache.value_cache
    raise TypeError(
        f"Unsupported cache type {type(cache).__name__!r}: expected DynamicCache "
        f"with `.key_cache` / `.value_cache` list attributes."
    )


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
        layers: layer indices that exist in the target cache (used for
            sanity-checks).
    """
    if start_index <= 0:
        return  # Nothing to patch in-cache.

    tgt_k, tgt_v = _layer_caches(target_cache)
    src_k, src_v = _layer_caches(source_kv_cache)

    for layer_idx, token_map in patch_spec.patches.items():
        if layer_idx not in layers:
            raise IndexError(
                f"PatchSpec references layer {layer_idx} but target cache has "
                f"layers {sorted(layers)}."
            )
        # Cache shape: [B, num_kv_heads, T, head_dim]
        tgt_k_layer = tgt_k[layer_idx]
        tgt_v_layer = tgt_v[layer_idx]
        src_k_layer = src_k[layer_idx]
        src_v_layer = src_v[layer_idx]

        for tok, comps in token_map.items():
            if tok >= start_index:
                continue
            if tok >= tgt_k_layer.shape[-2] or tok >= src_k_layer.shape[-2]:
                raise IndexError(
                    f"Token {tok} out of bounds for cache at layer {layer_idx}: "
                    f"target T={tgt_k_layer.shape[-2]}, source T={src_k_layer.shape[-2]}."
                )
            if Component.K in comps:
                tgt_k_layer[:, :, tok, :] = src_k_layer[:, :, tok, :].to(
                    tgt_k_layer.device, dtype=tgt_k_layer.dtype
                )
            if Component.V in comps:
                tgt_v_layer[:, :, tok, :] = src_v_layer[:, :, tok, :].to(
                    tgt_v_layer.device, dtype=tgt_v_layer.dtype
                )


def clone_dynamic_cache(cache):
    """Return a deep-copied `DynamicCache` so multiple patch runs can share a source."""
    try:
        from transformers.cache_utils import DynamicCache  # noqa: F401
    except Exception:  # pragma: no cover — only matters with transformers installed
        DynamicCache = None  # type: ignore

    src_k, src_v = _layer_caches(cache)
    cloned = type(cache)()
    cloned.key_cache = [k.clone() for k in src_k]
    cloned.value_cache = [v.clone() for v in src_v]
    # Some DynamicCache revisions also track seen tokens — copy if present.
    for attr in ("_seen_tokens", "seen_tokens"):
        if hasattr(cache, attr):
            setattr(cloned, attr, getattr(cache, attr))
    return cloned
