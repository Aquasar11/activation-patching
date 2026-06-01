# Examples

Runnable scripts demonstrating `actpatch`. They need a GPU and will download
model weights on first run.

| Script | What it shows |
|--------|---------------|
| `quickstart.py` | Online **full image swap** (apple → cat) with before/after predictions. |
| `offline_demo.py` | **Offline** mode: patch image K/V into the KV cache, decode from after the image block. |
| `common.py` | Shared helpers (model load, input building, top-k) used by both. |

```bash
python examples/quickstart.py                                   # Qwen2.5-VL
python examples/quickstart.py --model OpenGVLab/InternVL3_5-1B-hf
python examples/offline_demo.py
```

For a richer, visual walkthrough (renders both images + predictions inline) see
`notebooks/apple_cat_demo.ipynb`. For the concepts and more recipes, see
[`docs/GUIDE.md`](../docs/GUIDE.md).
