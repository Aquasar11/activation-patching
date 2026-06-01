"""Shared helpers for the actpatch example scripts.

Kept deliberately small and dependency-light so the example scripts read like a
recipe. Everything here is built from the public `actpatch` API.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from actpatch import (
    CacheSpec,
    Component,
    image_token_positions,
)

DATA = Path(__file__).resolve().parent.parent / "data"
PROMPT = "What is the main object in this image? Answer with one word:"

# RESID_IN + K + V across all layers = a full transplant of the chosen tokens.
FULL_COMPONENTS = [Component.RESID_IN, Component.K, Component.V]


def load(model_id: str, device: str | None = None):
    """Load a VLM + processor. Works for both Qwen2.5-VL and InternVL."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, device_map=device
    ).eval()
    return model, processor, device


def build_inputs(processor, image_path, device, *, size: int = 448, **proc_kwargs):
    """Build single-image chat inputs, resized to a fixed square.

    The square resize guarantees that any two images share the same image-token
    grid, which image patching requires. For InternVL also pass
    `crop_to_patches=False` to disable dynamic tiling.
    """
    image = Image.open(image_path).convert("RGB").resize((size, size))
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt", **proc_kwargs).to(device)


def top_k(processor, logits, k: int = 8):
    """Return [(token_text, logit)] for the top-k next-token candidates."""
    tok = processor.tokenizer
    top = logits[0, -1].float().topk(k)
    return [(tok.decode([int(i)]).strip(), round(float(v), 2))
            for v, i in zip(top.values, top.indices)]


def image_positions(adapter, model, inputs):
    img_id = adapter.get_image_token_id(model)
    return image_token_positions(inputs["input_ids"][0], img_id).tolist()


def cache_image_tokens(patcher, adapter, model, src_inputs, src_pos, tgt_pos, components):
    """Cache the source image-token activations and re-key them to target positions."""
    layers = list(range(len(adapter.get_decoder_layers(model))))
    cache = patcher.cache_source(
        src_inputs, CacheSpec.for_layers_tokens(layers, src_pos, components)
    )
    src_to_tgt = dict(zip(src_pos, tgt_pos))
    for store in (cache.resid_in, cache.k_proj, cache.v_proj):
        remapped = {(L, src_to_tgt[t]): v for (L, t), v in store.items() if t in src_to_tgt}
        store.clear()
        store.update(remapped)
    return cache, layers
