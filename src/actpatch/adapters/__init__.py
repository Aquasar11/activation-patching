"""Adapter registry — dispatches on model class name."""
from __future__ import annotations

from typing import Callable

from .base import ModelAdapter
from .internvl import InternVLAdapter
from .qwen2_5_vl import Qwen2_5_VLAdapter

__all__ = [
    "ModelAdapter",
    "Qwen2_5_VLAdapter",
    "InternVLAdapter",
    "get_adapter",
    "register_adapter",
]


# substring of model class name -> adapter factory (zero-arg callable -> adapter)
_REGISTRY = {
    "Qwen2_5_VL": Qwen2_5_VLAdapter,
    "InternVL": InternVLAdapter,
}


def register_adapter(class_name_substring: str, factory: Callable[[], ModelAdapter]) -> None:
    """Register an adapter so `get_adapter` can auto-dispatch to it.

    Args:
        class_name_substring: a substring matched against `type(model).__name__`.
            The first registered substring contained in the model's class name
            wins, so prefer a specific token (e.g. ``"LlavaNext"``).
        factory: a zero-argument callable returning a fresh adapter instance.

    Example:
        >>> from actpatch import register_adapter
        >>> register_adapter("MyVLM", MyVLMAdapter)

    You do not need to register at all if you construct the adapter yourself
    and pass it directly to ``ActivationPatcher(model, MyVLMAdapter())``.
    """
    _REGISTRY[class_name_substring] = factory


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
        f"Known prefixes: {list(_REGISTRY)}. "
        f"Either pass an adapter explicitly to ActivationPatcher(model, adapter), "
        f"or call actpatch.register_adapter(<substring>, <factory>)."
    )
