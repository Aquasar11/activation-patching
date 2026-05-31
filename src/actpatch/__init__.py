"""actpatch — activation patching for VLMs."""
from ._logging import disable_debug_logging, enable_debug_logging, get_logger
from .adapters import (
    InternVLAdapter,
    ModelAdapter,
    Qwen2_5_VLAdapter,
    get_adapter,
)
from .image_utils import (
    image_token_positions,
    mask_to_token_indices,
    rect_mask,
)
from .patcher import ActivationPatcher
from .specs import CacheSpec, Component, PatchSpec, SourceCache

__all__ = [
    # core
    "ActivationPatcher",
    "Component",
    "PatchSpec",
    "CacheSpec",
    "SourceCache",
    # adapters
    "ModelAdapter",
    "Qwen2_5_VLAdapter",
    "InternVLAdapter",
    "get_adapter",
    # image helpers
    "image_token_positions",
    "mask_to_token_indices",
    "rect_mask",
    # debugging
    "enable_debug_logging",
    "disable_debug_logging",
    "get_logger",
]

__version__ = "0.1.0"
