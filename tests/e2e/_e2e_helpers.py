"""Shared helpers for the apple/cat end-to-end experiment.

Both VLM e2e tests follow the same recipe:

    1. Load the model + processor.
    2. Build inputs for the cat image (source) and apple image (target),
       with prompt "this is a photo of".
    3. Find image-token positions in both inputs.
    4. Build a centred 2D foreground mask covering the central area of the
       image grid (object, not background).
    5. Cache cat activations at the foreground-image-token positions across
       all decoder layers (RESID_IN + K + V).
    6. Run patched_forward on the apple inputs with those patches.
    7. Decode the next token and verify it shifts toward "cat".

Run with:

    ACTPATCH_RUN_E2E=1 pytest tests/e2e -m slow -s
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import torch

DATA = Path(__file__).resolve().parents[2] / "data"
APPLE = DATA / "red_apple.jpeg"
CAT = DATA / "cat.jpeg"
PROMPT = "this is a photo of"


def centered_grid_mask(grid_shape: Tuple[int, int], pad_fraction: float = 0.2) -> torch.Tensor:
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
