"""End-to-end activation patching on InternVL 3.5: apple -> cat via image-token patch.

Gated by ACTPATCH_RUN_E2E=1 and pytest -m slow.
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
    InternVLAdapter,
    PatchSpec,
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

INTERN_MODEL_ID = os.environ.get(
    "ACTPATCH_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-1B-hf"
)


def _build_internvl_inputs(processor, image_path: str, device):
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
    return processor(text=[text], images=[image], return_tensors="pt").to(device)


@pytest.mark.slow
def test_internvl_apple_to_cat_online():
    require_torchvision()
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(INTERN_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        INTERN_MODEL_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()
    adapter = InternVLAdapter()
    patcher = ActivationPatcher(model, adapter)

    # Sanity: adapter resolved a non-empty layer list with k_proj/v_proj.
    assert len(adapter.get_decoder_layers(model)) > 0
    first_layer = adapter.get_decoder_layers(model)[0]
    k_proj, v_proj = adapter.get_attn_kv_projs(first_layer)
    assert hasattr(k_proj, "weight") and hasattr(v_proj, "weight")

    src_inputs = _build_internvl_inputs(processor, str(CAT), device)
    tgt_inputs = _build_internvl_inputs(processor, str(APPLE), device)

    with torch.no_grad():
        baseline = model(**tgt_inputs)
    baseline_topk = top_k_tokens(processor, baseline.logits[0, -1], k=12)
    print("Baseline top-k:", baseline_topk)

    img_id = adapter.get_image_token_id(model)
    src_img = image_token_positions(src_inputs["input_ids"][0], img_id)
    tgt_img = image_token_positions(tgt_inputs["input_ids"][0], img_id)
    assert src_img.numel() == tgt_img.numel()

    grid = adapter.image_grid_shape(tgt_inputs, model)
    print("Image grid shape:", grid)
    mask = centered_grid_mask(grid, pad_fraction=0.2)
    fg_positions_tgt = mask_to_token_indices(mask, tgt_img, grid)
    fg_positions_src = mask_to_token_indices(mask, src_img, grid)

    layers = list(range(len(adapter.get_decoder_layers(model))))

    src_cache_spec = CacheSpec.for_layers_tokens(
        layers=layers,
        tokens=fg_positions_src,
        components=[Component.RESID_IN, Component.K, Component.V],
    )
    src_cache = patcher.cache_source(src_inputs, src_cache_spec)

    src_to_tgt = dict(zip(fg_positions_src, fg_positions_tgt))
    for store in (src_cache.resid_in, src_cache.k_proj, src_cache.v_proj):
        remapped = {(L, src_to_tgt[t]): v for (L, t), v in store.items() if t in src_to_tgt}
        store.clear()
        store.update(remapped)

    patch = PatchSpec.for_layers_tokens(
        layers=layers,
        tokens=fg_positions_tgt,
        components=[Component.RESID_IN, Component.K, Component.V],
    )

    out = patcher.patched_forward(dict(tgt_inputs), src_cache, patch, mode="online")
    patched_topk = top_k_tokens(processor, out.logits[0, -1], k=12)
    print("Patched top-k:", patched_topk)

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
