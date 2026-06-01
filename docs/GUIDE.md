# actpatch — User Guide

A hands-on guide for using `actpatch` in your own projects. If you just want the
API surface, see the [README](../README.md). This document is the "how do I
actually do X" companion.

---

## Contents

1. [What activation patching is (mental model)](#1-mental-model)
2. [Install](#2-install)
3. [Your first patch in five minutes](#3-your-first-patch-in-five-minutes)
4. [How the library is structured](#4-how-the-library-is-structured)
5. [Recipes](#5-recipes)
   - [Full image swap](#recipe-full-image-swap)
   - [Generate a full response under patching](#recipe-generate-under-patching)
   - [Patch only the foreground object](#recipe-patch-only-the-foreground)
   - [Layer scan: where does the information live?](#recipe-layer-scan)
   - [Offline mode (KV-cache reuse)](#recipe-offline-mode)
   - [Text-only patching](#recipe-text-only-patching)
6. [Troubleshooting](#6-troubleshooting)
7. [FAQ](#7-faq)

---

## 1. Mental model

Activation patching answers **causal** questions about a model: *"which internal
activations actually carry this piece of information?"* You answer it by running
the model twice and splicing one run into the other.

```
        SOURCE run (donor)                 TARGET run (the one you modify)
   ┌───────────────────────┐          ┌───────────────────────────────┐
   │  cat image + prompt    │          │  apple image + prompt          │
   │                        │          │                                │
   │  cache activations  ───┼────────► │  overwrite the same positions  │
   │  at chosen (layer,tok) │  patch   │  then finish the forward pass  │
   └───────────────────────┘          └───────────────────────────────┘
                                                  │
                                                  ▼
                                       next-token prediction changes
                                       (apple → cat) if those activations
                                       carried the "what object" signal
```

Three things define an experiment:

| Term | Meaning |
|------|---------|
| **source** | the donor run; its activations are recorded (`cache_source`). |
| **target** | the run you modify; source activations are written into it (`patched_forward`). |
| **patch spec** | *where* to splice: a set of `(layer, token, component)` coordinates. |

A **component** is the thing you copy at a `(layer, token)` location:

- `Component.RESID_IN` — the residual-stream vector entering a decoder block.
- `Component.K` — the key projection (`k_proj`) output inside that block's attention.
- `Component.V` — the value projection (`v_proj`) output.

Patching `RESID_IN` at *all* layers for a set of tokens effectively **transplants
those tokens wholesale** from source into target. That is the strongest
intervention and what the apple→cat demo uses.

---

## 2. Install

```bash
pip install -e .            # from the repo root
# or, for development (tests + linter):
pip install -e .[dev]
```

You need `torch`, `torchvision`, and `transformers` (pinned in `pyproject.toml`).
A GPU is required only for the real VLMs; the unit tests run on CPU.

---

## 3. Your first patch in five minutes

The complete apple→cat experiment with Qwen2.5-VL:

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from actpatch import (
    ActivationPatcher, CacheSpec, PatchSpec, Component,
    get_adapter, image_token_positions,
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
device = "cuda"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

adapter = get_adapter(model)                 # auto-detects the model class
patcher = ActivationPatcher(model, adapter)

PROMPT = "What is the main object in this image? Answer with one word:"

def build_inputs(image_path):
    # Resize to a fixed square so source and target share one image-token grid.
    image = Image.open(image_path).convert("RGB").resize((448, 448))
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt").to(device)

def top1(inputs_or_logits):
    logits = inputs_or_logits
    tok = logits[0, -1].argmax()
    return processor.tokenizer.decode([tok]).strip()

src = build_inputs("data/cat.jpeg")       # donor
tgt = build_inputs("data/red_apple.jpeg") # the run we modify

# Baseline: what does the model say about the apple?
with torch.no_grad():
    print("baseline:", top1(model(**tgt).logits))   # -> "Apple"

# 1) cache the cat's image-token activations across all layers
img_id = adapter.get_image_token_id(model)
src_pos = image_token_positions(src["input_ids"][0], img_id).tolist()
tgt_pos = image_token_positions(tgt["input_ids"][0], img_id).tolist()
layers = range(len(adapter.get_decoder_layers(model)))
comps = [Component.RESID_IN, Component.K, Component.V]

cache = patcher.cache_source(src, CacheSpec.for_layers_tokens(layers, src_pos, comps))

# 2) re-key the cache from source positions to the matching target positions
#    (identical here because both prompts share the same layout), then patch
src_to_tgt = dict(zip(src_pos, tgt_pos))
for store in (cache.resid_in, cache.k_proj, cache.v_proj):
    store_remapped = {(L, src_to_tgt[t]): v for (L, t), v in store.items()}
    store.clear(); store.update(store_remapped)

patch = PatchSpec.for_layers_tokens(layers, tgt_pos, comps)
out = patcher.patched_forward(tgt, cache, patch, mode="online")
print("patched: ", top1(out.logits))                # -> "Cat"
```

That's the whole loop: **build inputs → cache source → patch target → read the
output**. Everything else in the library is variations on this.

> **Why the re-key step?** `cache_source` stores tensors keyed by the *source*
> token index. When source and target put the image block at the same
> positions (same prompt template, same image-token count) the mapping is the
> identity and you can skip it — but doing it explicitly keeps the code correct
> if the layouts ever differ.

---

## 4. How the library is structured

```
actpatch
├── ActivationPatcher      the one class you use: cache_source / patched_forward / patched_generate
├── Component              RESID_IN | K | V
├── PatchSpec, CacheSpec   "which (layer, token, component) coordinates"
├── SourceCache            the recorded activations (returned by cache_source)
├── get_adapter / ModelAdapter / register_adapter   model-specific wiring
└── image_token_positions / mask_to_token_indices / rect_mask   image helpers
```

The **core is model-agnostic**: it operates on a list of decoder layers and
their `k_proj`/`v_proj` modules. A thin **adapter** tells it where those live for
a given model. Patching is done with PyTorch forward hooks — no subclassing, no
monkey-patching. See the README's "Extending to a new model" for adding one.

---

## 5. Recipes

All recipes assume `patcher`, `adapter`, `model`, and built `src`/`tgt` inputs
from the quick-start above.

<a name="recipe-full-image-swap"></a>
### Full image swap

The strongest, most robust intervention — replace the entire image
representation. This is the quick-start example. Use it when you want to be sure
the visual content is fully transplanted (it does not depend on how a model
orders its image tokens).

<a name="recipe-generate-under-patching"></a>
### Generate a full response under patching

`patched_forward` gives you the logits for one step. To watch the model produce
a **whole response** with the patch in effect, use the `patching()` context
manager and call the model's own `generate()` inside it — that way HF handles
position-ids and the KV cache natively (important for Qwen2.5-VL's M-RoPE):

```python
cache = patcher.cache_source(src, CacheSpec.for_layers_tokens(layers, src_pos, comps))
# (re-key cache from src_pos -> tgt_pos as in the quick-start)
patch = PatchSpec.for_layers_tokens(layers, tgt_pos, comps)

with patcher.patching(cache, patch):
    out = model.generate(**tgt, max_new_tokens=40, do_sample=False)
text = processor.tokenizer.decode(out[0, tgt["input_ids"].shape[1]:], skip_special_tokens=True)
print(text)   # describes the *cat*, even though tgt is the apple image
```

The patch applies during the prefill (where the image tokens are present) and
bakes into the KV cache, so the entire generated sentence reflects the swap.
Positions absent from later single-token steps are skipped automatically.

`notebooks/apple_cat_demo.ipynb` shows this end to end with before/after text.

<a name="recipe-patch-only-the-foreground"></a>
### Patch only the foreground object

To localize an effect to *the object* and leave the background alone, build a 2-D
mask over the image-token grid:

```python
from actpatch import mask_to_token_indices, rect_mask

grid = adapter.image_grid_shape(tgt, model)         # e.g. (16, 16)
mask = rect_mask(grid, top=3, left=3, bottom=13, right=13)   # central region
fg_src = mask_to_token_indices(mask, image_token_positions(src["input_ids"][0], img_id), grid)
fg_tgt = mask_to_token_indices(mask, image_token_positions(tgt["input_ids"][0], img_id), grid)
# cache at fg_src, re-key to fg_tgt, patch at fg_tgt (same pattern as quick-start)
```

> **Caveat:** a subset mask assumes **row-major** token order. That holds for
> Qwen2.5-VL but *not* for models that reorder tokens (e.g. InternVL's
> pixel-shuffle), where a "central" mask maps to scattered cells. For those,
> prefer the full swap.

<a name="recipe-layer-scan"></a>
### Layer scan: where does the information live?

Patch one layer at a time and watch the output. The layer where patching first
flips the prediction is where that information becomes decisive.

```python
cache = patcher.cache_source(src, CacheSpec.for_layers_tokens(layers, src_pos, comps))
src_to_tgt = dict(zip(src_pos, tgt_pos))
for store in (cache.resid_in, cache.k_proj, cache.v_proj):
    store.update({(L, src_to_tgt[t]): v for (L, t), v in dict(store).items()})

for L in range(len(adapter.get_decoder_layers(model))):
    patch = PatchSpec.for_layers_tokens([L], tgt_pos, [Component.RESID_IN])
    out = patcher.patched_forward(tgt, cache, patch, mode="online")
    print(L, processor.tokenizer.decode([out.logits[0, -1].argmax()]).strip())
```

<a name="recipe-offline-mode"></a>
### Offline mode (KV-cache reuse)

Online mode recomputes the whole sequence on every patch. Offline mode prefills
once, writes K/V patches into the cache for positions *before* `start_index`,
and only recomputes the suffix `[start_index, T)`. Use it when you patch image
tokens (early) and then decode from a later position — it mirrors real
generation and is cheaper across many runs.

```python
# patch image-token K/V into the cache, then decode from after the image block
out = patcher.patched_forward(
    tgt, cache, patch, mode="offline", start_index=last_image_pos + 1,
)
```

Rules: positions `< start_index` are served from the cache (only **K/V** patches
apply there; residual patches before `start_index` are skipped because a
residual can't be reconstructed from a KV cache). `start_index` must be in
`[0, T)`.

<a name="recipe-text-only-patching"></a>
### Text-only patching

Nothing about the library is image-specific — `image_token_positions` is just a
convenience. To patch text tokens, pass whatever positions you want:

```python
spec = PatchSpec.for_layers_tokens(layers=[5, 6, 7], tokens=[12], components=[Component.RESID_IN])
```

This is the classic "swap the subject name" mech-interp experiment.

---

## 6. Troubleshooting

| Symptom | Likely cause & fix |
|---|---|
| `Source/target image-token counts differ` | The two images produced different grids. Resize both to the same square (Qwen) and pass `crop_to_patches=False` (InternVL). See the README "Align the two images' grids first." |
| Patch has little/no effect | (a) You patched too few tokens/layers — try a full swap across all layers with `RESID_IN+K+V`. (b) A foreground mask mis-mapped due to token reordering — use a full swap. (c) Confirm hooks fire: `actpatch.enable_debug_logging()` prints every capture/patch. |
| `No adapter registered for model class ...` | Pass an adapter explicitly (`ActivationPatcher(model, MyAdapter())`) or `register_adapter("ClassNameSubstring", MyAdapter)`. |
| `Could not locate decoder layers ...` | The adapter's candidate attribute paths don't match this model. Add the right path; the resolver only accepts a layer list whose first element has `self_attn.k_proj`/`v_proj`. |
| `mask shape ... does not match grid_shape` / `positions has N entries but grid implies M` | Your 2-D mask or `image_grid_shape` is wrong for this model/input. Print `adapter.image_grid_shape(inputs, model)` and the number of image tokens. |
| `start_index must be in [0, T)` | Offline `start_index` left no live positions; pick a value strictly less than the sequence length. |

Turn on tracing any time:

```python
import actpatch
actpatch.enable_debug_logging()      # DEBUG -> stderr; logs every hook
# ...
actpatch.disable_debug_logging()
```

---

## 7. FAQ

**Does this modify or fine-tune the model?** No. It registers temporary forward
hooks for the duration of one forward pass and removes them afterward. Weights
are untouched.

**Is K patched before or after RoPE?** Before (it hooks `k_proj`'s output). This
is the standard mech-interp convention. Source and target are both pre-RoPE so
the splice is consistent.

**Can I reuse one `SourceCache` for many patches?** Yes — cache once, then call
`patched_forward` repeatedly with different `PatchSpec`s. The cache is read-only
during patching.

**Batched inputs?** The image helpers expect the same image-token positions
across the batch (they validate this). Patching applies to all batch rows.

**Which models are supported out of the box?** Qwen2.5-VL and InternVL 3.5. Any
LLM-decoder VLM can be added with a ~40-line adapter (README → "Extending to a
new model").

**Where's a full runnable demo?** `notebooks/apple_cat_demo.ipynb` (shows both
images and before/after predictions) and `examples/`.
