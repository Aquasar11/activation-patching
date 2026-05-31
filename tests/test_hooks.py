"""Unit tests for hook plumbing: HookHandle lifecycle and ForwardContext."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from actpatch.hooks import ForwardContext, HookHandle


def test_forward_context_identity_when_no_indices():
    ctx = ForwardContext(forward_pass_indices=None)
    # With no slicing, target index maps to itself.
    assert ctx.target_to_local(0) == 0
    assert ctx.target_to_local(7) == 7


def test_forward_context_maps_subset_and_returns_none_for_absent():
    ctx = ForwardContext(forward_pass_indices=[5, 6, 7])
    assert ctx.target_to_local(5) == 0
    assert ctx.target_to_local(6) == 1
    assert ctx.target_to_local(7) == 2
    # Positions outside the live window have no local row.
    assert ctx.target_to_local(4) is None
    assert ctx.target_to_local(8) is None


def test_hookhandle_registers_and_removes():
    module = nn.Linear(4, 4)
    calls = []

    def hook(mod, inp, out):
        calls.append(1)
        return None

    with HookHandle() as handle:
        handle.add_forward_hook(module, hook)
        assert len(handle._handles) == 1
        module(torch.zeros(1, 4))
        assert len(calls) == 1
    # After the context exits the hook is gone — no further calls recorded.
    module(torch.zeros(1, 4))
    assert len(calls) == 1
    assert handle._handles == []


def test_hookhandle_removes_on_exception():
    module = nn.Linear(4, 4)

    def hook(mod, inp, out):
        return None

    with pytest.raises(RuntimeError):
        with HookHandle() as handle:
            handle.add_forward_hook(module, hook)
            raise RuntimeError("boom")
    # Even though the body raised, hooks were cleaned up.
    assert handle._handles == []
    assert module._forward_hooks == {} or len(module._forward_hooks) == 0


def test_hookhandle_pre_hook_with_kwargs():
    module = nn.Linear(4, 4)
    seen = {}

    def pre_hook(mod, args, kwargs):
        seen["args_len"] = len(args)
        return None

    with HookHandle() as handle:
        handle.add_pre_hook(module, pre_hook)
        module(torch.zeros(1, 4))
    assert seen["args_len"] == 1
