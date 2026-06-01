# actpatch

Activation patching for Vision-Language Models.

`actpatch` is a small, hookable PyTorch library that copies activations from a
**source** forward pass into a **target** forward pass at chosen
`(layer, token)` positions, then lets you observe how the model's output
changes. It is the standard mechanistic-interpretability tool for *causal
localisation* — "which positions/components actually carry this information?" —
extended to multimodal decoders so you can patch **image tokens**.

Concretely, it can answer questions like: *if I overwrite the apple-image
tokens in one run with the cat-image activations from another run, does the
model now say "cat"?*

- Patch the **residual-stream input** of a decoder block and/or the **K / V
  projections** inside its attention.
- Two execution modes: **online** (recompute everything downstream live) and
  **offline** (reuse a prefilled KV cache and only recompute a suffix).
- Ships adapters for **Qwen2.5-VL** and **InternVL 3.5**. New models are a
  ~40-line adapter — see [Extending to a new model](#extending-to-a-new-model).

> **New here?** Start with the tutorial-style [`docs/GUIDE.md`](docs/GUIDE.md),
> run the [`examples/`](examples/), or open the visual walkthrough
> [`notebooks/apple_cat_demo.ipynb`](notebooks/apple_cat_demo.ipynb). This README
> is the reference.

---

## Contents

- [Install](#install)
- [Concepts](#concepts)
- [Quick start](#quick-start)
- [Patching only the foreground of an image](#patching-only-the-foreground-of-an-image)
- [Online vs offline](#online-vs-offline)
- [API reference](#api-reference)
- [Extending to a new model](#extending-to-a-new-model)
- [Debugging](#debugging)
- [Tests](#tests)
- [Project layout](#project-layout)

---

## Install

```bash
pip install -e .[test]
```

Requires `torch`, `torchvision` (the HF VLM processors pull it in), and
`transformers`. Exact pins live in `pyproject.toml`.

---

## Concepts

A patching experiment has three moving parts:

| Term | Meaning |
|------|---------|
| **source** | the donor forward pass — its activations are cached. |
| **target** | the run you modify — source activations are written into it. |
| **patch spec** | *where* to patch: a set of `(layer, token, component)` triples. |

The unit of patching is a **`Component`** at a `(layer, token)` coordinate:

- `Component.RESID_IN` — the residual-stream tensor fed *into* a decoder block.
- `Component.K` — the output of that block's `self_attn.k_proj` (pre-RoPE).
- `Component.V` — the output of that block's `self_attn.v_proj`.

You first **cache** the source (`cache_source`), then run a **patched forward**
on the target (`patched_forward`). Caching and patching use the same
coordinate shape, so the same positions you record are the ones you overwrite.

---

## Quick start

The canonical apple→cat experiment (full runnable versions live in
`tests/e2e/`):

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from actpatch import (
    ActivationPatcher, CacheSpec, PatchSpec, Component,
    get_adapter, image_token_positions,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

adapter = get_adapter(model)                  # auto-dispatch on model class
patcher = ActivationPatcher(model, adapter)

# Build inputs for both images (see tests/e2e for the prompt helper).
source_inputs = build_inputs(processor, "cat.jpeg")     # donor
target_inputs = build_inputs(processor, "red_apple.jpeg")  # run we modify

# Image-token positions in each prompt.
img_id = adapter.get_image_token_id(model)
src_img = image_token_positions(source_inputs["input_ids"][0], img_id)
tgt_img = image_token_positions(target_inputs["input_ids"][0], img_id)

layers = range(len(adapter.get_decoder_layers(model)))

# 1) Cache the cat activations at the image-token positions, all layers.
source_cache = patcher.cache_source(
    source_inputs,
    CacheSpec.for_layers_tokens(
        layers=layers,
        tokens=src_img.tolist(),
        components=[Component.RESID_IN, Component.K, Component.V],
    ),
)

# 2) Patch them into the apple run and read the next-token distribution.
out = patcher.patched_forward(
    target_inputs,
    source_cache,
    PatchSpec.for_layers_tokens(
        layers=layers,
        tokens=tgt_img.tolist(),
        components=[Component.RESID_IN, Component.K, Component.V],
    ),
    mode="online",
)
next_token = out.logits[0, -1].argmax()
print(processor.tokenizer.decode(next_token))   # -> "cat"-ish
```

> **Note on aligning source and target positions.** `cache_source` keys cached
> tensors by the *source* token index. If the image block sits at different
> absolute positions in the two prompts, remap the cache keys from source to
> target indices before patching (the e2e tests show the exact one-liner). When
> the prompt template is identical, the positions already line up.

---

## Patching only the foreground of an image

VLMs lay image tokens out as a flattened 2-D grid. `actpatch` gives you the
grid shape (post spatial-merge) and a helper to turn a 2-D boolean mask into
the flat sequence positions, so you can patch *the object* and leave the
background alone.

```python
import torch
from actpatch import image_token_positions, mask_to_token_indices, rect_mask

img_id = adapter.get_image_token_id(model)
positions = image_token_positions(target_inputs["input_ids"][0], img_id)
grid = adapter.image_grid_shape(target_inputs, model)   # e.g. (H, W)

# Keep the central region (a crude "foreground"); or hand-author any mask.
mask = rect_mask(grid, top=grid[0]//5, left=grid[1]//5,
                 bottom=4*grid[0]//5, right=4*grid[1]//5)

fg_positions = mask_to_token_indices(mask, positions, grid)   # list[int]
patch = PatchSpec.for_layers_tokens(layers, fg_positions, [Component.RESID_IN])
```

`mask_to_token_indices` validates that the mask shape matches the grid and that
the number of image tokens matches `H*W`, so a wrong grid fails loudly rather
than silently patching the wrong tokens.

> **Align the two images' grids first.** Both Qwen2.5-VL and InternVL tokenise
> images at *dynamic* resolution, so two differently-shaped photos yield
> different numbers of image tokens — and image patching needs a 1:1 map
> between source and target positions. Force an identical grid before
> processing:
>
> - **Qwen2.5-VL** preserves aspect ratio, so resize both images to the same
>   square whose side is a multiple of `patch_size * spatial_merge_size`
>   (28 by default): `Image.open(p).convert("RGB").resize((448, 448))` → a
>   16×16 = 256-token grid.
> - **InternVL** tiles dynamically; pass `crop_to_patches=False` to the
>   processor so every image is a single tile with a fixed token count.
>
> `tests/e2e/_e2e_helpers.py` shows both in `load_square_image` and the
> per-model build functions.

---

## Online vs offline

Both modes end by predicting the next token; they differ in how much is
recomputed.

**Online** (`mode="online"`) — the default. The whole target sequence is run
again with hooks live, so every position downstream of a patch is recomputed.
This is the faithful, no-shortcuts mode; use it for correctness-critical
analysis.

```python
out = patcher.patched_forward(target_inputs, source_cache, patch, mode="online")
```

**Offline** (`mode="offline"`, requires `start_index`) — prefill the target
once to build a KV cache, write K/V patches for positions *before*
`start_index` directly into that cache, then only run the contiguous suffix
`[start_index, T)` live. This mirrors real decoding: you patch the image region
into the cache, then continue generation from a later position.

```python
out = patcher.patched_forward(
    target_inputs, source_cache, patch,
    mode="offline", start_index=image_end + 1,
)
```

Semantics to keep in mind:

- Positions `< start_index` are served from the cache. **K/V** patches there are
  written into the cache; **residual** patches there are *skipped* (a residual
  stream value isn't reconstructible from a KV cache alone). Patch residuals in
  online mode, or at positions `>= start_index`.
- `forward_pass_indices` lets you control exactly which positions enter the live
  forward. In offline mode it must be the contiguous suffix `[start_index, T)`;
  in online mode it may be any subset (the patcher sets `position_ids`
  accordingly).

---

## API reference

Everything below is importable from the top-level `actpatch` package.

### `Component`
Enum: `RESID_IN`, `K`, `V` (see [Concepts](#concepts)).

### `CacheSpec` / `PatchSpec`
Frozen dataclasses describing `{layer: {token: frozenset(components)}}`.
Build them with the convenience constructor:

```python
CacheSpec.for_layers_tokens(layers, tokens, components)
PatchSpec.for_layers_tokens(layers, tokens, components)
PatchSpec.for_layers_tokens(...)   # same shape; what to overwrite
CacheSpec.from_patch_spec(patch)   # record exactly what a patch will need
```

You can also construct the nested dict directly for irregular specs:

```python
PatchSpec(patches={2: {6: frozenset({Component.K, Component.V})}})
```

### `SourceCache`
Returned by `cache_source`. Holds the captured tensors (`resid_in`, `k_proj`,
`v_proj`, keyed by `(layer, token)`), the source prefill `kv_cache`
(`DynamicCache`, used by offline mode), plus `seq_len` / `dtype` / `device`.

### `ActivationPatcher(model, adapter)`

| Method | Purpose |
|--------|---------|
| `cache_source(inputs, cache_spec, *, keep_on_device=False, keep_kv_cache=True)` | Run the source forward and capture activations into a `SourceCache`. Captured tensors move to CPU by default; pass `keep_on_device=True` to retain them on the model's device. |
| `patched_forward(target_inputs, source_cache, patch_spec, mode="online", start_index=None, forward_pass_indices=None)` | Run the target forward with source activations patched in. Returns the model's output object (has `.logits`, `.past_key_values`). |
| `patched_generate(target_inputs, source_cache, patch_spec, *, mode="online", start_index=None, max_new_tokens=1)` | Greedy-decode `max_new_tokens` tokens; patches apply on the first step and persist through the KV cache. Returns token ids `[B, max_new_tokens]`. |
| `patching(source_cache, patch_spec, *, forward_pass_indices=None)` | Context manager that keeps patch hooks active while *you* drive the model — e.g. `with patcher.patching(cache, patch): model.generate(**inputs, ...)`. Lets `generate` handle position-ids / KV cache natively, so you can watch the full generation with and without patching. |

### Image helpers
- `image_token_positions(input_ids, image_token_id) -> LongTensor`
- `mask_to_token_indices(mask_2d, positions, grid_shape, order="row_major") -> list[int]`
- `rect_mask(grid_shape, top, left, bottom, right) -> BoolTensor`

### Adapters
- `get_adapter(model) -> ModelAdapter` — auto-dispatch on class name.
- `register_adapter(class_name_substring, factory)` — register your own.
- `Qwen2_5_VLAdapter`, `InternVLAdapter`, and the `ModelAdapter` protocol.

### Debugging
- `enable_debug_logging(level=logging.DEBUG, stream=None, fmt=None)`
- `disable_debug_logging()`
- `get_logger(name)`

---

## Extending to a new model

Adding a model means writing **one adapter class** that tells the
model-agnostic core where things live. You never subclass or monkey-patch the
HF model — patching is done entirely through forward hooks.

### Step 1 — Find the layout

For the target model, locate four things (read the HF `modeling_*.py` or just
`print(model)`):

1. The list of LLM decoder layers, e.g. `model.language_model.layers`.
2. The K/V projections on a layer, e.g. `layer.self_attn.k_proj` / `.v_proj`.
3. The image-token id, usually `model.config.image_token_id`.
4. How to compute the post-merge image grid `(H, W)` from processor inputs.

### Step 2 — Implement the `ModelAdapter` protocol

The protocol (see `actpatch/adapters/base.py`) has six methods:

```python
class ModelAdapter(Protocol):
    def get_decoder_layers(self, model) -> list[nn.Module]: ...
    def get_attn_kv_projs(self, layer) -> tuple[nn.Module, nn.Module]: ...  # (k_proj, v_proj)
    def get_image_token_id(self, model) -> int: ...
    def num_kv_heads(self, model) -> int: ...
    def head_dim(self, model) -> int: ...
    def image_grid_shape(self, inputs, model) -> tuple[int, int]: ...
```

Use the `resolve_decoder_layers` helper for robustness: it tries several
candidate attribute paths and only accepts one whose first layer actually
exposes `self_attn.k_proj`/`v_proj` (this avoids accidentally grabbing the
*vision* encoder's layers).

### Step 3 — Example: a minimal LLaVA-style adapter

```python
# my_project/llava_adapter.py
from __future__ import annotations
from typing import List, Mapping, Tuple
from torch import nn
from actpatch.adapters.base import resolve_decoder_layers

# Ordered candidates; the resolver picks the first that has self_attn.k_proj/v_proj.
_CANDIDATE_LAYER_PATHS = (
    "language_model.model.layers",
    "model.language_model.layers",
    "language_model.layers",
)

class LlavaAdapter:
    name = "llava"

    def get_decoder_layers(self, model: nn.Module) -> List[nn.Module]:
        layers, _path = resolve_decoder_layers(model, _CANDIDATE_LAYER_PATHS)
        return layers

    def get_attn_kv_projs(self, layer: nn.Module) -> Tuple[nn.Module, nn.Module]:
        return layer.self_attn.k_proj, layer.self_attn.v_proj

    def get_image_token_id(self, model: nn.Module) -> int:
        return int(model.config.image_token_index)   # LLaVA names it *_index

    def _text_cfg(self, model):
        cfg = model.config
        return getattr(cfg, "text_config", cfg)

    def num_kv_heads(self, model: nn.Module) -> int:
        return int(self._text_cfg(model).num_key_value_heads)

    def head_dim(self, model: nn.Module) -> int:
        tc = self._text_cfg(model)
        return int(getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))

    def image_grid_shape(self, inputs: Mapping[str, object], model: nn.Module) -> Tuple[int, int]:
        # CLIP-ViT patch grid: (image_size / patch_size) per side, e.g. 336/14 = 24.
        vc = model.config.vision_config
        side = vc.image_size // vc.patch_size
        return side, side
```

### Step 4 — Use it

Either pass the adapter explicitly:

```python
from actpatch import ActivationPatcher
from my_project.llava_adapter import LlavaAdapter

patcher = ActivationPatcher(model, LlavaAdapter())
```

…or register it once so `get_adapter` finds it automatically:

```python
from actpatch import register_adapter, get_adapter
from my_project.llava_adapter import LlavaAdapter

register_adapter("Llava", LlavaAdapter)     # matched against type(model).__name__
patcher = ActivationPatcher(model, get_adapter(model))
```

### Step 5 — Sanity-check the adapter

Before a full experiment, verify the wiring (cheap, no patching):

```python
layers = patcher.adapter.get_decoder_layers(model)
assert len(layers) == model.config.text_config.num_hidden_layers
k_proj, v_proj = patcher.adapter.get_attn_kv_projs(layers[0])
assert hasattr(k_proj, "weight") and hasattr(v_proj, "weight")

# Image grid must match the number of image tokens in the prompt.
from actpatch import image_token_positions
pos = image_token_positions(inputs["input_ids"][0], patcher.adapter.get_image_token_id(model))
H, W = patcher.adapter.image_grid_shape(inputs, model)
assert len(pos) == H * W, (len(pos), (H, W))
```

> **Gotchas worth checking on a new model**
> - **GQA**: `k_proj`/`v_proj` output `num_kv_heads * head_dim`, *not*
>   `num_heads * head_dim`. `num_kv_heads`/`head_dim` must reflect the KV heads.
> - **K is patched pre-RoPE** — the standard convention. Post-RoPE patching is
>   out of scope for v1.
> - **Spatial merge / pixel-shuffle**: `image_grid_shape` must return the grid
>   *after* any token-merging, so it equals the actual image-token count.
> - **Image token count must match** between source and target when patching
>   image tokens; `actpatch` fails loudly if it doesn't.

---

## Debugging

The library is silent by default. Turn on verbose tracing of caching, hook
registration, per-position patch application, and the offline cache surgery:

```python
import actpatch
actpatch.enable_debug_logging()             # DEBUG -> stderr
# ... run cache_source / patched_forward ...
actpatch.disable_debug_logging()            # go quiet again
```

Traces are emitted on the `actpatch.*` logger hierarchy, so you can also wire
them into your own logging config without the helper:

```python
import logging
logging.getLogger("actpatch").setLevel(logging.DEBUG)
```

Typical output shows the resolved decoder-layer path, how many hooks were
registered, and each `(layer, token)` capture/patch — including the
target→local row mapping that offline mode uses, which is the easiest thing to
get wrong.

---

## Tests

```bash
pytest tests/ --ignore=tests/e2e            # unit suite (no GPU needed)
ACTPATCH_RUN_E2E=1 pytest tests/e2e -m slow -s   # end-to-end on a GPU box
```

The unit suite runs against a tiny synthetic transformer (`tests/tiny_model.py`)
that mimics the HF decoder-layer interface, so it exercises the real hook and
cache machinery without downloading weights. The e2e suite runs the apple→cat
experiment on real VLMs and is opt-in via `ACTPATCH_RUN_E2E=1`. It performs a
**full image-token swap** (every cat image token → the apple run), which is the
most robust form of the experiment and is independent of how a model orders its
image tokens.

Useful env vars:
- `ACTPATCH_QWEN_MODEL`, `ACTPATCH_INTERNVL_MODEL` — override the model ids.
- `ACTPATCH_DEBUG=1` — turn on debug tracing during the e2e run, printing every
  capture/patch so you can confirm the hooks fire and count the patched slots.

---

## Project layout

```
src/actpatch/
  specs.py          Component enum, PatchSpec, CacheSpec, SourceCache
  hooks.py          capture/patch forward hooks + HookHandle context manager
  cache_ops.py      DynamicCache access + mutation (offline mode)
  image_utils.py    image-token positions + 2D-mask -> token indices
  patcher.py        ActivationPatcher (the public class)
  _logging.py       opt-in debug tracing
  adapters/
    base.py         ModelAdapter protocol + resolve_decoder_layers helper
    qwen2_5_vl.py   Qwen2.5-VL adapter
    internvl.py     InternVL 3.5 adapter
    __init__.py     get_adapter / register_adapter registry
tests/              unit suite + tests/e2e (apple/cat, opt-in)
docs/GUIDE.md       tutorial-style user guide (concepts, recipes, troubleshooting)
examples/           runnable scripts (quickstart.py, offline_demo.py)
notebooks/          apple_cat_demo.ipynb — visual before/after walkthrough
```
