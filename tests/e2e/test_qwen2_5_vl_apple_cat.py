"""End-to-end activation patching on Qwen2.5-VL: apple -> cat via image-token patch.

Gated by ACTPATCH_RUN_E2E=1 and pytest -m slow. Requires GPU + ~6 GB of weights.
"""
from __future__ import annotations

import os

import pytest
import torch

from actpatch import ActivationPatcher, Qwen2_5_VLAdapter

from ._e2e_helpers import (
    APPLE,
    CAT,
    CAT_WORDS,
    PROMPT,
    first_match_rank,
    load_square_image,
    maybe_enable_debug,
    require_torchvision,
    run_image_swap,
    top_k_tokens,
)

QWEN_MODEL_ID = os.environ.get(
    "ACTPATCH_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
)


def _build_qwen_inputs(processor, image_path: str, device):
    """Build a one-image prompt. The image is squared so both runs share a grid."""
    image = load_square_image(image_path)
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
def test_qwen2_5_vl_apple_to_cat_online():
    require_torchvision()
    maybe_enable_debug()
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

    # Full image swap: transplant every cat image token into the apple run.
    out = run_image_swap(patcher, adapter, model, src_inputs, tgt_inputs)
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
