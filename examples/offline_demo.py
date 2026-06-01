"""Offline mode: patch image K/V into the KV cache, then decode from after it.

Offline mode prefills the target once, writes K/V patches into the cache for
positions *before* `start_index`, and only recomputes the suffix. This mirrors
real decoding: you patch the image region into the cache and continue from a
later position.

Note: residual-stream patches before `start_index` cannot be applied offline (a
residual is not reconstructible from a KV cache), so this demo patches K and V
only. That is a more surgical intervention than the online full-residual swap in
quickstart.py, so the effect may be smaller — the point here is the mechanism.

    python examples/offline_demo.py
"""
from __future__ import annotations

import argparse

import torch
from common import (
    DATA,
    build_inputs,
    cache_image_tokens,
    image_positions,
    load,
    top_k,
)

from actpatch import ActivationPatcher, Component, PatchSpec, get_adapter

KV_COMPONENTS = [Component.K, Component.V]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()

    model, processor, device = load(args.model)
    adapter = get_adapter(model)
    patcher = ActivationPatcher(model, adapter)

    proc_kwargs = {"crop_to_patches": False} if "InternVL" in type(model).__name__ else {}
    src = build_inputs(processor, DATA / "cat.jpeg", device, **proc_kwargs)
    tgt = build_inputs(processor, DATA / "red_apple.jpeg", device, **proc_kwargs)

    with torch.no_grad():
        print("apple (baseline):", top_k(processor, model(**tgt).logits)[:5])

    src_pos = image_positions(adapter, model, src)
    tgt_pos = image_positions(adapter, model, tgt)
    if len(src_pos) != len(tgt_pos):
        raise SystemExit("image-token counts differ; resize images / disable tiling.")

    cache, layers = cache_image_tokens(
        patcher, adapter, model, src, src_pos, tgt_pos, KV_COMPONENTS
    )

    # Decode the next token starting just after the (patched) image block.
    start_index = max(tgt_pos) + 1
    patch = PatchSpec.for_layers_tokens(layers, tgt_pos, KV_COMPONENTS)
    out = patcher.patched_forward(tgt, cache, patch, mode="offline", start_index=start_index)

    print(f"\napple PATCHED with cat (offline, start_index={start_index}):")
    print(top_k(processor, out.logits)[:8])


if __name__ == "__main__":
    main()
