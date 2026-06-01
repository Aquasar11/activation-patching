"""Quick-start: online full image swap (apple -> cat).

Runs the model on the apple image (baseline), then transplants every cat
image-token activation into the apple run and shows the prediction flip.

Usage (needs a GPU and model weights):

    python examples/quickstart.py
    python examples/quickstart.py --model OpenGVLab/InternVL3_5-1B-hf
"""
from __future__ import annotations

import argparse

import torch
from common import (
    DATA,
    FULL_COMPONENTS,
    build_inputs,
    cache_image_tokens,
    image_positions,
    load,
    top_k,
)

from actpatch import ActivationPatcher, PatchSpec, get_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()

    model, processor, device = load(args.model)
    adapter = get_adapter(model)
    patcher = ActivationPatcher(model, adapter)

    # InternVL needs dynamic tiling off so every image is a single fixed tile.
    proc_kwargs = {"crop_to_patches": False} if "InternVL" in type(model).__name__ else {}
    src = build_inputs(processor, DATA / "cat.jpeg", device, **proc_kwargs)        # donor
    tgt = build_inputs(processor, DATA / "red_apple.jpeg", device, **proc_kwargs)  # modified

    with torch.no_grad():
        print("apple (baseline):", top_k(processor, model(**tgt).logits)[:5])
        print("cat   (baseline):", top_k(processor, model(**src).logits)[:5])

    src_pos = image_positions(adapter, model, src)
    tgt_pos = image_positions(adapter, model, tgt)
    if len(src_pos) != len(tgt_pos):
        raise SystemExit(
            f"image-token counts differ ({len(src_pos)} vs {len(tgt_pos)}); "
            "resize both images to the same square and disable tiling."
        )

    cache, layers = cache_image_tokens(
        patcher, adapter, model, src, src_pos, tgt_pos, FULL_COMPONENTS
    )
    patch = PatchSpec.for_layers_tokens(layers, tgt_pos, FULL_COMPONENTS)
    out = patcher.patched_forward(tgt, cache, patch, mode="online")

    print("\napple PATCHED with cat:", top_k(processor, out.logits)[:8])


if __name__ == "__main__":
    main()
