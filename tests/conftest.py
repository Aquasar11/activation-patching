"""Shared pytest fixtures for the actpatch test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Allow `import actpatch` without installation when running locally.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiny_model import TinyAdapter, TinyVLM  # noqa: E402  — after sys.path tweak


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.slow tests unless ACTPATCH_RUN_E2E=1 is set."""
    if os.environ.get("ACTPATCH_RUN_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="set ACTPATCH_RUN_E2E=1 to run end-to-end VLM tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model = TinyVLM(vocab=64, hidden=32, num_layers=4, num_heads=4, num_kv_heads=2, head_dim=8)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_adapter():
    return TinyAdapter()


@pytest.fixture()
def sample_inputs():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 64, (1, 8))
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


@pytest.fixture()
def other_inputs():
    torch.manual_seed(2)
    input_ids = torch.randint(0, 64, (1, 8))
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
