"""Shared helpers for the apple/cat end-to-end experiment.

Both VLM e2e tests follow the same recipe:

    1. Load the model + processor.
    2. Build inputs for the cat image (source) and apple image (target).
    3. Find image-token positions in both inputs.
    4. Build a centred 2D foreground mask covering the central area of the
       image grid (object, not background).
    5. Cache cat activations at the foreground-image-token positions across
       all decoder layers (RESID_IN + K + V).
    6. Run patched_forward on the apple inputs with those patches.
    7. Decode the next token and verify it shifts toward "cat".

A subtlety that bites real VLMs: both Qwen2.5-VL and InternVL use *dynamic*
image tokenisation, so two differently-shaped photos produce different numbers
of image tokens (and different grids). Activation patching needs a 1:1 mapping
between source and target image tokens, so we force an identical grid for both
images:

* **Qwen2.5-VL** keeps aspect ratio in `smart_resize`, so we pre-resize each
  image to the same square whose side is a multiple of patch_size*merge_size
  (28). A 448x448 image -> 32x32 patches -> 16x16 merged tokens (256).
* **InternVL** tiles images dynamically (`crop_to_patches=True`); we pass
  `crop_to_patches=False` so every image becomes a single tile (256 tokens).

Run with:

    ACTPATCH_RUN_E2E=1 pytest tests/e2e -m slow -s
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from PIL import Image

from actpatch import (
    CacheSpec,
    Component,
    PatchSpec,
    enable_debug_logging,
    image_token_positions,
    mask_to_token_indices,
)

DATA = Path(__file__).resolve().parents[2] / "data"
APPLE = DATA / "red_apple.jpeg"
CAT = DATA / "cat.jpeg"

# Square side fed to the models. Must be a multiple of 28 (patch 14 * merge 2)
# for Qwen, and equal to InternVL's tile size (448) so a single tile is exact.
IMAGE_SIZE = 448


def load_square_image(path: str, size: int = IMAGE_SIZE) -> Image.Image:
    """Open an image and resize it to `size`x`size` so every input shares one grid."""
    return Image.open(path).convert("RGB").resize((size, size))

# A one-word-answer prompt so the *next* token is the object name itself,
# which makes the apple->cat flip observable in a single decoding step.
PROMPT = "What is the main object in this image? Answer with one word:"

CAT_WORDS = ("cat", "kitten", "kitty", "feline")
APPLE_WORDS = ("apple", "fruit")


def require_torchvision() -> None:
    """Skip (don't error) if torchvision is missing — the HF video processor needs it."""
    try:
        import torchvision  # noqa: F401
    except Exception:
        pytest.skip(
            "torchvision is required to load the VLM processor "
            "(pip install torchvision) — install it to run this end-to-end test."
        )


def maybe_enable_debug() -> None:
    """Turn on actpatch debug tracing when ACTPATCH_DEBUG=1.

    Handy on the GPU box: it prints every capture/patch (with the layer, token,
    and local-row mapping), so you can confirm the hooks actually fire and how
    many slots were patched — the first thing to check if a swap looks weak.
    """
    if os.environ.get("ACTPATCH_DEBUG") == "1":
        enable_debug_logging()


def centered_grid_mask(grid_shape: tuple[int, int], pad_fraction: float = 0.2) -> torch.Tensor:
    """Return a 2D bool mask covering the centre of the grid.

    `pad_fraction` is the fraction of cells trimmed from each border. With the
    default 0.2 a 20x20 grid keeps the central 12x12 cells — usually a
    reasonable approximation of "object, not background" for natural photos.
    """
    H, W = grid_shape
    top = max(int(H * pad_fraction), 1) if H > 2 else 0
    left = max(int(W * pad_fraction), 1) if W > 2 else 0
    bottom = H - top
    right = W - left
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[top:bottom, left:right] = True
    return mask


def top_k_tokens(processor_or_tokenizer, logits_row: torch.Tensor, k: int = 8) -> list:
    tok = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    top = torch.topk(logits_row.float(), k=k)
    return [(tok.decode([int(i)]).strip(), float(v)) for v, i in zip(top.values, top.indices)]


def contains_target_word(text: str, candidates: Iterable[str]) -> bool:
    low = text.strip().lower()
    return any(c in low for c in candidates)


def first_match_rank(pairs: list[tuple[str, float]], words: Iterable[str]) -> int | None:
    """Return the 0-based rank of the first token whose text contains any of `words`."""
    words = tuple(w.lower() for w in words)
    for rank, (text, _) in enumerate(pairs):
        low = text.lower()
        if any(w in low for w in words):
            return rank
    return None


def run_image_swap(patcher, adapter, model, src_inputs, tgt_inputs, *, mask=None):
    """Transplant the source image's tokens into the target run and return the output.

    Caches the source (cat) activations at the image-token positions across all
    decoder layers (RESID_IN + K + V), then patches them onto the matching
    target (apple) positions and runs an online patched forward.

    Args:
        mask: if None, swap *all* image tokens (a full image transplant — the
            most robust experiment). Otherwise a 2D bool grid selecting a
            subset of the grid (e.g. a foreground region). Note that a subset
            mask assumes row-major token order, which holds for Qwen2.5-VL but
            not necessarily for models that reorder tokens (e.g. InternVL's
            pixel-shuffle) — prefer a full swap there.

    Returns the patched model output (has `.logits`).
    """
    img_id = adapter.get_image_token_id(model)
    src_img = image_token_positions(src_inputs["input_ids"][0], img_id)
    tgt_img = image_token_positions(tgt_inputs["input_ids"][0], img_id)
    if src_img.numel() != tgt_img.numel():
        raise AssertionError(
            f"Source/target image-token counts differ: {src_img.numel()} vs "
            f"{tgt_img.numel()}. Image patching needs aligned grids — resize both "
            f"images to the same square (and disable dynamic tiling)."
        )

    if mask is None:
        src_pos = src_img.tolist()
        tgt_pos = tgt_img.tolist()
    else:
        grid = adapter.image_grid_shape(tgt_inputs, model)
        src_pos = mask_to_token_indices(mask, src_img, grid)
        tgt_pos = mask_to_token_indices(mask, tgt_img, grid)

    layers = list(range(len(adapter.get_decoder_layers(model))))
    comps = [Component.RESID_IN, Component.K, Component.V]

    src_cache = patcher.cache_source(
        src_inputs, CacheSpec.for_layers_tokens(layers, src_pos, comps)
    )
    # Re-key cached tensors from source positions to the matching target ones.
    src_to_tgt = dict(zip(src_pos, tgt_pos))
    for store in (src_cache.resid_in, src_cache.k_proj, src_cache.v_proj):
        remapped = {(L, src_to_tgt[t]): v for (L, t), v in store.items() if t in src_to_tgt}
        store.clear()
        store.update(remapped)

    patch = PatchSpec.for_layers_tokens(layers, tgt_pos, comps)
    return patcher.patched_forward(dict(tgt_inputs), src_cache, patch, mode="online")
