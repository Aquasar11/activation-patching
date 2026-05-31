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

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch

DATA = Path(__file__).resolve().parents[2] / "data"
APPLE = DATA / "red_apple.jpeg"
CAT = DATA / "cat.jpeg"

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
