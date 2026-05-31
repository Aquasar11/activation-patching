"""image_utils: find image-token positions and map a 2D mask onto them."""
from __future__ import annotations

import pytest
import torch

from actpatch import image_token_positions, mask_to_token_indices, rect_mask


def test_image_token_positions_1d():
    ids = torch.tensor([5, 7, 7, 7, 7, 8, 9])
    pos = image_token_positions(ids, image_token_id=7)
    assert pos.tolist() == [1, 2, 3, 4]


def test_image_token_positions_2d_consistent_batch():
    ids = torch.tensor([[1, 7, 7, 2], [1, 7, 7, 2]])
    pos = image_token_positions(ids, image_token_id=7)
    assert pos.tolist() == [1, 2]


def test_image_token_positions_2d_inconsistent_batch_raises():
    ids = torch.tensor([[1, 7, 7, 2], [7, 7, 1, 2]])
    with pytest.raises(ValueError):
        image_token_positions(ids, image_token_id=7)


def test_mask_to_token_indices_rectangular_region():
    # 4x4 image grid, image tokens occupy positions [3..18] in the sequence.
    positions = torch.arange(3, 3 + 16)
    mask = rect_mask((4, 4), top=1, left=1, bottom=3, right=3)  # 2x2 center
    idx = mask_to_token_indices(mask, positions, grid_shape=(4, 4))
    # Center 2x2 at rows 1,2 and cols 1,2 corresponds to grid offsets
    # row 1: cols 1,2 -> 1*4+1=5, 1*4+2=6
    # row 2: cols 1,2 -> 2*4+1=9, 2*4+2=10
    # Add the base offset 3.
    assert idx == [3 + 5, 3 + 6, 3 + 9, 3 + 10]


def test_mask_shape_mismatch_raises():
    positions = torch.arange(0, 16)
    bad_mask = torch.zeros((3, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="mask shape"):
        mask_to_token_indices(bad_mask, positions, grid_shape=(4, 4))


def test_positions_count_mismatch_raises():
    positions = torch.arange(0, 9)  # 9 positions but grid wants 16
    mask = torch.zeros((4, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="positions"):
        mask_to_token_indices(mask, positions, grid_shape=(4, 4))


def test_image_token_positions_3d_raises():
    ids = torch.zeros((1, 2, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="1D or 2D"):
        image_token_positions(ids, image_token_id=0)


def test_image_token_positions_batch_gt2_all_rows_checked():
    # Third row differs -> must be caught even though rows 0 and 1 agree.
    ids = torch.tensor([[1, 7, 7, 2], [1, 7, 7, 2], [7, 1, 7, 2]])
    with pytest.raises(ValueError):
        image_token_positions(ids, image_token_id=7)


def test_rect_mask_region_and_dtype():
    mask = rect_mask((4, 5), top=1, left=2, bottom=3, right=4)
    assert mask.dtype == torch.bool
    assert mask.shape == (4, 5)
    assert int(mask.sum()) == (3 - 1) * (4 - 2)  # 2 rows x 2 cols
    assert mask[1, 2] and mask[2, 3]
    assert not mask[0, 0] and not mask[3, 4]


def test_mask_to_token_indices_rejects_unknown_order():
    positions = torch.arange(0, 16)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="row_major"):
        mask_to_token_indices(mask, positions, grid_shape=(4, 4), order="col_major")


def test_mask_to_token_indices_empty_mask_returns_empty():
    positions = torch.arange(0, 16)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    assert mask_to_token_indices(mask, positions, grid_shape=(4, 4)) == []
