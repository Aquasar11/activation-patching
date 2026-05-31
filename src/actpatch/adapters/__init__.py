"""Adapter registry — dispatches on model class name."""
from __future__ import annotations

from .base import ModelAdapter
from .internvl import InternVLAdapter
from .qwen2_5_vl import Qwen2_5_VLAdapter

__all__ = ["ModelAdapter", "Qwen2_5_VLAdapter", "InternVLAdapter", "get_adapter"]


# substring of model class name -> adapter factory
_REGISTRY = {
    "Qwen2_5_VL": Qwen2_5_VLAdapter,
    "InternVL": InternVLAdapter,
}


def get_adapter(model) -> ModelAdapter:
    """Return an adapter matching the model class name.

    Dispatches on substring of `type(model).__name__`. Raises with a clear
    message if no adapter matches.
    """
    cls_name = type(model).__name__
    for key, factory in _REGISTRY.items():
        if key in cls_name:
            return factory()
    raise KeyError(
        f"No adapter registered for model class {cls_name!r}. "
        f"Known prefixes: {list(_REGISTRY)}."
    )
