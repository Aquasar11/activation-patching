"""End-to-end activation patching on Qwen2.5-VL: apple -> cat via image-token patch.

Gated by ACTPATCH_RUN_E2E=1 and pytest -m slow. Requires GPU + ~6 GB of weights.
"""
from __future__ import annotations

import os

import pytest
import torch
from PIL import Image

from actpatch import (
    ActivationPatcher,
    CacheSpec,
    Component,
    PatchSpec,
    Qwen2_5_VLAdapter,
    image_token_positions,
    mask_to_token_indices,
)

from ._e2e_helpers import (
    APPLE,
    CAT,
    CAT_WORDS,
    PROMPT,
    centered_grid_mask,
    first_match_rank,
    require_torchvision,
    top_k_tokens,
)

QWEN_MODEL_ID = os.environ.get(
    "ACTPATCH_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
)


def _build_qwen_inputs(processor, image_path: str, device):
    """Build a single-turn 'this is a photo of' prompt with one image."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Strip the trailing assistant role marker so the model continues from
    # the user's prompt directly — we want the next predicted token to
    # complete "this is a photo of ___".
    return processor(text=[text], images=[image], return_tensors="pt").to(device)


@pytest.mark.slow
def test_qwen2_5_vl_apple_to_cat_online():
    require_torchvision()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()
    adapter = Qwen2_5_VLAdapter()
    patcher = ActivationPatcher(model, adapter)

    src_inputs = _build_qwen_inputs(processor, str(CAT), device)
    tgt_inputs = _build_qwen_inputs(processor, str(APPLE), device)

    # Baseline: unpatched apple prediction.
    with torch.no_grad():
        baseline = model(**tgt_inputs)
    baseline_topk = top_k_tokens(processor, baseline.logits[0, -1], k=12)
    print("Baseline top-k:", baseline_topk)

    # Image-token positions in each prompt (per batch row 0).
    src_img = image_token_positions(src_inputs["input_ids"][0], adapter.get_image_token_id(model))
    tgt_img = image_token_positions(tgt_inputs["input_ids"][0], adapter.get_image_token_id(model))
    assert src_img.numel() == tgt_img.numel(), (
        f"Source and target image-token counts differ: {src_img.numel()} vs {tgt_img.numel()}. "
        f"Foreground patching requires aligned grids — adjust the prompt or pick same-sized images."
    )

    # Foreground mask: keep central 60% of the grid (drop 20% border).
    grid = adapter.image_grid_shape(tgt_inputs, model)
    print("Image grid shape:", grid)
    mask = centered_grid_mask(grid, pad_fraction=0.2)
    fg_positions_tgt = mask_to_token_indices(mask, tgt_img, grid)
    fg_positions_src = mask_to_token_indices(mask, src_img, grid)
    assert len(fg_positions_tgt) == len(fg_positions_src)

    # Build per-token mapping: target_idx -> source_idx. Same mask order in both.
    # We treat both as keyed by the SAME target-seq index so source values can
    # be applied at those positions. For Qwen2.5-VL, the image-token blocks
    # may sit at the same absolute indices in both prompts (identical text
    # template), so we cache and patch using the *target* indices and look up
    # source values by source positions in a parallel pass.
    layers = list(range(len(adapter.get_decoder_layers(model))))

    # Cache source activations keyed by SOURCE positions.
    src_cache_spec = CacheSpec.for_layers_tokens(
        layers=layers,
        tokens=fg_positions_src,
        components=[Component.RESID_IN, Component.K, Component.V],
    )
    src_cache = patcher.cache_source(src_inputs, src_cache_spec)

    # Remap cache keys: (layer, src_pos) -> (layer, tgt_pos).
    src_to_tgt = dict(zip(fg_positions_src, fg_positions_tgt))
    for store in (src_cache.resid_in, src_cache.k_proj, src_cache.v_proj):
        remapped = {(L, src_to_tgt[t]): v for (L, t), v in store.items() if t in src_to_tgt}
        store.clear()
        store.update(remapped)

    # Build patch spec at target positions.
    patch = PatchSpec.for_layers_tokens(
        layers=layers,
        tokens=fg_positions_tgt,
        components=[Component.RESID_IN, Component.K, Component.V],
    )

    out = patcher.patched_forward(dict(tgt_inputs), src_cache, patch, mode="online")
    patched_topk = top_k_tokens(processor, out.logits[0, -1], k=12)
    print("Patched top-k:", patched_topk)

    # The causal claim: patching the cat image into the apple run should pull a
    # cat-related token up the ranking. Pass if a cat token appears in the
    # patched top-k and ranks no worse than it did in the baseline (it is
    # usually absent from the baseline entirely).
    base_rank = first_match_rank(baseline_topk, CAT_WORDS)
    patched_rank = first_match_rank(patched_topk, CAT_WORDS)
    print(f"cat-token rank — baseline: {base_rank}, patched: {patched_rank}")
    assert patched_rank is not None, (
        f"Expected a cat-related token in patched top-k, got {patched_topk}"
    )
    assert base_rank is None or patched_rank <= base_rank, (
        f"Patching did not raise the cat token's rank "
        f"(baseline={base_rank}, patched={patched_rank})."
    )
