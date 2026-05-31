"""Helpers for working with image tokens in a VLM input sequence.

Two operations:

* `image_token_positions(input_ids, image_token_id)` returns the absolute
  sequence positions where image tokens live.
* `mask_to_token_indices(mask_2d, image_token_positions, grid_shape)` maps a
  2D boolean grid (e.g. foreground vs background) onto the flat list of
  image-token sequence positions, so callers can patch only the foreground.

Adapters supply the post-merge `grid_shape`; this module does not need to
know about model-specific spatial-merge factors.
"""
from __future__ import annotations

import torch


def image_token_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Return a 1D LongTensor of positions in `input_ids` equal to the image-token id.

    Works for both shape `[T]` and `[B, T]` (in the batched case we require
    the same positions for every row — VLMs normally pad image tokens to the
    same length per batch).
    """
    if input_ids.dim() == 1:
        return (input_ids == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 1D or 2D, got shape {tuple(input_ids.shape)}")

    masks = input_ids == image_token_id
    # Require identical positions across every row of the batch.
    if masks.shape[0] > 1 and not bool((masks == masks[0]).all()):
        raise ValueError(
            "image_token_positions requires the same image-token positions across "
            "the batch dimension. Got differing rows in input_ids."
        )
    return masks[0].nonzero(as_tuple=False).squeeze(-1)


def mask_to_token_indices(
    mask_2d: torch.Tensor,
    positions: torch.Tensor,
    grid_shape: tuple[int, int],
    order: str = "row_major",
) -> list[int]:
    """Map a 2D bool grid onto absolute sequence positions of image tokens.

    Args:
        mask_2d: bool tensor `[H_grid, W_grid]` — True at cells to patch.
        positions: 1D LongTensor of absolute sequence positions for the image
            tokens, in the order they appear in `input_ids`. Length must be
            `H_grid * W_grid`.
        grid_shape: `(H_grid, W_grid)` after any model-specific spatial merge.
        order: layout of `positions` over the grid. Only `"row_major"` is
            supported in v1.

    Returns:
        Sorted list of absolute sequence positions for the True cells.
    """
    if order != "row_major":
        raise ValueError(f"order must be 'row_major', got {order!r}")

    H, W = grid_shape
    if mask_2d.shape != (H, W):
        raise ValueError(
            f"mask shape {tuple(mask_2d.shape)} does not match grid_shape {(H, W)}"
        )
    if positions.numel() != H * W:
        raise ValueError(
            f"positions has {positions.numel()} entries but grid_shape implies {H * W}. "
            f"Adapter `image_grid_shape` may be wrong, or the input has multiple images."
        )

    flat = mask_2d.reshape(-1).to(torch.bool)
    selected = positions[flat]
    return sorted(int(x) for x in selected.tolist())


def rect_mask(
    grid_shape: tuple[int, int], top: int, left: int, bottom: int, right: int
) -> torch.Tensor:
    """Convenience: build a 2D bool mask with a rectangular True region.

    Half-open on the right/bottom edges, like Python slicing.
    """
    H, W = grid_shape
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[top:bottom, left:right] = True
    return mask
