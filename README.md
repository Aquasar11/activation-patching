# actpatch

Activation patching for Vision-Language Models.

A small, hookable PyTorch module that lets you copy activations from a *source*
forward pass into a *target* forward pass at chosen (layer, token) positions,
then observe how the model output changes. Supports residual-stream and K/V
patches on the LLM decoder, with both **online** (live recompute) and
**offline** (KV-cache reuse) modes.

Adapters ship for **Qwen2.5-VL** and **InternVL 3.5**; new adapters are a
~30-line subclass.

## Install

```bash
pip install -e .[test]
```

## Quick start

```python
from actpatch import (
    ActivationPatcher, CacheSpec, PatchSpec, Component, get_adapter,
)

adapter = get_adapter(model)              # auto-dispatch on model class
patcher = ActivationPatcher(model, adapter)

source_cache = patcher.cache_source(source_inputs, CacheSpec.for_layers_tokens(
    layers=range(model.config.num_hidden_layers),
    tokens=image_token_positions,
    components=[Component.RESID_IN, Component.K, Component.V],
))

out = patcher.patched_forward(
    target_inputs,
    source_cache,
    PatchSpec.for_layers_tokens(
        layers=range(model.config.num_hidden_layers),
        tokens=image_token_positions,
        components=[Component.RESID_IN],
    ),
    mode="online",
)
```

See `tests/e2e/` for the apple/cat experiment.

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
them into your own logging config (e.g. `logging.getLogger("actpatch").setLevel(...)`)
without calling the helper.

## Tests

```bash
pytest tests/ --ignore=tests/e2e            # unit suite (no GPU needed)
ACTPATCH_RUN_E2E=1 pytest tests/e2e -m slow # end-to-end on a GPU box
```
