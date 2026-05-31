"""ActivationPatcher — the public orchestration class.

A two-step API:

    patcher = ActivationPatcher(model, adapter)
    cache  = patcher.cache_source(source_inputs, cache_spec)
    out    = patcher.patched_forward(target_inputs, cache, patch_spec, mode=...)

`mode="online"` does a full target re-prefill with hooks fired live.
`mode="offline"` builds (or reuses) a target KV cache up to `start_index`,
writes K/V patches into it, then runs only the suffix `[start_index, T)`
through the model with live hooks for any patches in that range.
"""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import torch
from torch import nn

from .adapters.base import ModelAdapter
from .cache_ops import apply_kv_patches_to_cache, clone_dynamic_cache
from .hooks import (
    ForwardContext,
    HookHandle,
    register_capture_hooks,
    register_patch_hooks,
)
from .specs import CacheSpec, Component, PatchSpec, SourceCache


class ActivationPatcher:
    """Runs source-cached and target-patched forward passes on a VLM."""

    def __init__(self, model: nn.Module, adapter: ModelAdapter) -> None:
        self.model = model
        self.adapter = adapter
        # Cached so we don't re-introspect the model on every call.
        self._layers: List[nn.Module] = adapter.get_decoder_layers(model)
        self._kv_proj_lookup = adapter.get_attn_kv_projs

    # ------------------------------------------------------------------ #
    # Source caching                                                     #
    # ------------------------------------------------------------------ #
    def cache_source(
        self,
        inputs: Mapping[str, torch.Tensor],
        cache_spec: CacheSpec,
        *,
        keep_on_device: bool = False,
        keep_kv_cache: bool = True,
    ) -> SourceCache:
        """Run the source forward, capture activations into a `SourceCache`."""
        cache = SourceCache()

        with HookHandle() as handle, torch.no_grad():
            register_capture_hooks(
                handle,
                self._layers,
                self._kv_proj_lookup,
                cache_spec,
                cache,
                keep_on_device=keep_on_device,
            )
            forward_kwargs = dict(inputs)
            if keep_kv_cache:
                forward_kwargs["use_cache"] = True
            outputs = self.model(**forward_kwargs)

        # Populate metadata from the input we just ran.
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            cache.seq_len = int(input_ids.shape[-1])
            cache.device = input_ids.device
        if keep_kv_cache:
            cache.kv_cache = getattr(outputs, "past_key_values", None)
        # Recover dtype from a captured tensor if any.
        for store in (cache.resid_in, cache.k_proj, cache.v_proj):
            if store:
                cache.dtype = next(iter(store.values())).dtype
                break
        return cache

    # ------------------------------------------------------------------ #
    # Patched forward                                                    #
    # ------------------------------------------------------------------ #
    def patched_forward(
        self,
        target_inputs: Mapping[str, torch.Tensor],
        source_cache: SourceCache,
        patch_spec: PatchSpec,
        mode: str = "online",
        start_index: Optional[int] = None,
        forward_pass_indices: Optional[Sequence[int]] = None,
    ):
        """Run the target forward with activations from `source_cache` patched in.

        Args:
            target_inputs: model kwargs for the target forward (input_ids and
                anything else the model needs).
            source_cache: the result of a prior `cache_source` call.
            patch_spec: which (layer, token, component) slots to patch.
            mode: 'online' or 'offline'.
            start_index: required for offline mode. Positions `< start_index`
                are served from the (mutated) target KV cache; positions
                `>= start_index` are processed in the live forward.
            forward_pass_indices: explicit subset of target-seq positions to
                feed into the live forward. Defaults to the full sequence
                (online) or `[start_index, T)` (offline). In offline mode the
                indices must be a contiguous suffix.
        """
        if mode == "online":
            return self._online_forward(
                target_inputs, source_cache, patch_spec, forward_pass_indices
            )
        if mode == "offline":
            if start_index is None:
                raise ValueError("offline mode requires `start_index`.")
            return self._offline_forward(
                target_inputs, source_cache, patch_spec, start_index, forward_pass_indices
            )
        raise ValueError(f"Unknown mode {mode!r}; expected 'online' or 'offline'.")

    # ------------------------------------------------------------------ #
    # Online                                                             #
    # ------------------------------------------------------------------ #
    def _online_forward(
        self,
        target_inputs: Mapping[str, torch.Tensor],
        source_cache: SourceCache,
        patch_spec: PatchSpec,
        forward_pass_indices: Optional[Sequence[int]],
    ):
        input_ids = target_inputs.get("input_ids")
        if input_ids is None:
            raise KeyError("target_inputs must contain `input_ids` for online mode.")
        T = int(input_ids.shape[-1])

        if forward_pass_indices is None:
            ctx = ForwardContext(forward_pass_indices=None)
            live_inputs = dict(target_inputs)
        else:
            indices = list(forward_pass_indices)
            self._validate_indices(indices, T)
            ctx = ForwardContext(forward_pass_indices=indices)
            live_inputs = self._slice_inputs_to_indices(target_inputs, indices)

        with HookHandle() as handle, torch.no_grad():
            register_patch_hooks(
                handle,
                self._layers,
                self._kv_proj_lookup,
                patch_spec,
                source_cache,
                ctx,
                skip_tokens_below=None,
            )
            return self.model(**live_inputs)

    # ------------------------------------------------------------------ #
    # Offline                                                            #
    # ------------------------------------------------------------------ #
    def _offline_forward(
        self,
        target_inputs: Mapping[str, torch.Tensor],
        source_cache: SourceCache,
        patch_spec: PatchSpec,
        start_index: int,
        forward_pass_indices: Optional[Sequence[int]],
    ):
        input_ids = target_inputs.get("input_ids")
        if input_ids is None:
            raise KeyError("target_inputs must contain `input_ids` for offline mode.")
        T = int(input_ids.shape[-1])
        if not (0 <= start_index <= T):
            raise ValueError(f"start_index must be in [0, {T}], got {start_index}.")

        # 1. Build the target KV cache for positions [0, start_index).
        target_cache = self._build_target_prefill_cache(target_inputs, start_index)

        # 2. Apply K/V patches at tok < start_index into that cache.
        if patch_spec.has_kv():
            if source_cache.kv_cache is None:
                raise ValueError(
                    "offline K/V patches require source_cache.kv_cache; "
                    "re-run cache_source with keep_kv_cache=True."
                )
            apply_kv_patches_to_cache(
                target_cache,
                patch_spec,
                source_cache.kv_cache,
                start_index=start_index,
                layers=range(len(self._layers)),
            )

        # 3. Determine live `forward_pass_indices`.
        if forward_pass_indices is None:
            indices = list(range(start_index, T))
        else:
            indices = list(forward_pass_indices)
            self._validate_indices(indices, T)
            expected = list(range(start_index, T))
            if indices != expected:
                raise ValueError(
                    "offline mode requires forward_pass_indices to be the contiguous "
                    f"suffix {expected}, got {indices}."
                )
        ctx = ForwardContext(forward_pass_indices=indices)

        # 4. Build live inputs (text-only slice; pixels already represented in cache).
        live_inputs = self._build_offline_live_inputs(target_inputs, indices, T)
        live_inputs["past_key_values"] = target_cache
        live_inputs["use_cache"] = True

        with HookHandle() as handle, torch.no_grad():
            register_patch_hooks(
                handle,
                self._layers,
                self._kv_proj_lookup,
                patch_spec,
                source_cache,
                ctx,
                skip_tokens_below=start_index,
            )
            return self.model(**live_inputs)

    # ------------------------------------------------------------------ #
    # Convenience: greedy generation                                     #
    # ------------------------------------------------------------------ #
    def patched_generate(
        self,
        target_inputs: Mapping[str, torch.Tensor],
        source_cache: SourceCache,
        patch_spec: PatchSpec,
        *,
        mode: str = "online",
        start_index: Optional[int] = None,
        max_new_tokens: int = 1,
    ) -> torch.Tensor:
        """Run `patched_forward` once and greedily decode `max_new_tokens` tokens.

        Patches are applied only on the first forward; subsequent decoding
        steps reuse the KV cache produced by that call (so K/V patches before
        `start_index` persist via the cache for the rest of generation).
        Returns the generated token ids `[B, max_new_tokens]`.
        """
        outputs = self.patched_forward(
            target_inputs, source_cache, patch_spec, mode=mode, start_index=start_index
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]
        cache = getattr(outputs, "past_key_values", None)

        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = getattr(outputs, "past_key_values", cache)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
        return torch.cat(generated, dim=-1)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _validate_indices(self, indices: Sequence[int], T: int) -> None:
        if any(i < 0 or i >= T for i in indices):
            raise IndexError(
                f"forward_pass_indices out of range for sequence length {T}: {indices}."
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"forward_pass_indices contains duplicates: {indices}.")

    def _slice_inputs_to_indices(
        self,
        target_inputs: Mapping[str, torch.Tensor],
        indices: Sequence[int],
    ) -> dict:
        """Slice input_ids / attention_mask / position_ids to the given indices."""
        idx = torch.as_tensor(indices, dtype=torch.long)
        out = dict(target_inputs)
        input_ids = target_inputs["input_ids"]
        idx_dev = idx.to(input_ids.device)
        out["input_ids"] = input_ids.index_select(-1, idx_dev)
        if "attention_mask" in out:
            out["attention_mask"] = out["attention_mask"].index_select(-1, idx_dev)
        # Provide explicit position_ids so RoPE picks the right rotations.
        out["position_ids"] = idx_dev.unsqueeze(0).expand(input_ids.shape[0], -1)
        return out

    def _build_target_prefill_cache(
        self,
        target_inputs: Mapping[str, torch.Tensor],
        start_index: int,
    ):
        """Run target through the model to obtain a fresh KV cache, cropped to start_index."""
        if start_index == 0:
            # Empty cache: build a fresh DynamicCache via a zero-length convention.
            # The cleanest portable way is to import DynamicCache; we do it lazily.
            from transformers.cache_utils import DynamicCache  # noqa: WPS433
            return DynamicCache()

        with torch.no_grad():
            outputs = self.model(**dict(target_inputs), use_cache=True)
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError(
                "Target model did not return a `past_key_values` cache; "
                "ensure the model supports KV caching."
            )
        cache = clone_dynamic_cache(cache)
        return self._crop_dynamic_cache(cache, length=start_index)

    @staticmethod
    def _crop_dynamic_cache(cache, length: int):
        """Crop every layer's K and V to the first `length` positions in-place."""
        for i, k in enumerate(cache.key_cache):
            cache.key_cache[i] = k[..., :length, :]
        for i, v in enumerate(cache.value_cache):
            cache.value_cache[i] = v[..., :length, :]
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = length
        elif hasattr(cache, "seen_tokens"):
            cache.seen_tokens = length
        return cache

    def _build_offline_live_inputs(
        self,
        target_inputs: Mapping[str, torch.Tensor],
        indices: Sequence[int],
        T: int,
    ) -> dict:
        """Build the kwargs for the live forward in offline mode.

        Pixel / vision kwargs are stripped — the image-token rows are already
        represented in the KV cache, and re-passing pixel inputs would either
        be ignored or trigger a re-merge against a sliced input_ids.
        """
        idx = torch.as_tensor(indices, dtype=torch.long)
        input_ids = target_inputs["input_ids"]
        idx_dev = idx.to(input_ids.device)

        live = {
            "input_ids": input_ids.index_select(-1, idx_dev),
            "use_cache": True,
        }
        # Provide a full-length attention mask so the model attends to cached
        # positions and the live ones.
        if "attention_mask" in target_inputs:
            live["attention_mask"] = target_inputs["attention_mask"]
        else:
            live["attention_mask"] = torch.ones(
                (input_ids.shape[0], T), dtype=torch.long, device=input_ids.device
            )

        # Only inject explicit position_ids if the caller provided one for the
        # full sequence (we slice it). Some VLMs (e.g. Qwen2.5-VL with M-RoPE)
        # use 3D position_ids and would break under a naive 2D override; in
        # those cases we leave position_ids unset and let the model derive
        # them from `cache_position` + cache length.
        provided_pos = target_inputs.get("position_ids")
        if provided_pos is not None and provided_pos.dim() == 2 and provided_pos.shape[-1] == T:
            live["position_ids"] = provided_pos.index_select(-1, idx_dev)
        live["cache_position"] = idx_dev
        return live
