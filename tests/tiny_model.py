"""A small synthetic causal LM whose decoder-layer interface mimics HF.

Used by the unit suite so we can verify hook-based patching without needing a
multi-GB VLM. The shape of `forward` deliberately matches what HF Qwen2/Llama
decoder layers expose so the same hooks work unchanged.

* `model.layers` is an `nn.ModuleList[TinyDecoderLayer]`.
* `layer.self_attn.{q,k,v,o}_proj` are linears.
* Forward accepts `past_key_values`, `use_cache`, `cache_position`, returns an
  object with `.logits` and `.past_key_values`.
* Grouped-query attention (num_kv_heads can be < num_heads), matching GQA in
  Qwen2.5-VL.
* No positional encoding — keeps the test reference simple. Causal masking
  uses absolute positions derived from `cache_position`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn


@dataclass
class TinyOutputs:
    logits: torch.Tensor
    past_key_values: object | None = None


class TinyAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.layer_idx: int = -1  # set by the layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None and use_cache:
            # HF DynamicCache.update returns the concatenated K, V.
            k, v = past_key_values.update(k, v, self.layer_idx)
        T_kv = k.size(-2)

        # GQA expand to num_heads.
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Absolute positions of queries (this batch) and keys (full kv).
        if cache_position is not None:
            q_pos = cache_position.to(hidden_states.device)
        else:
            q_pos = torch.arange(T_kv - T, T_kv, device=hidden_states.device)
        k_pos = torch.arange(T_kv, device=hidden_states.device)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = q_pos[:, None] < k_pos[None, :]
        scores = scores.masked_fill(causal.view(1, 1, T, T_kv), float("-inf"))
        if attention_mask is not None:
            keep = attention_mask.bool()[:, None, None, :T_kv]
            scores = scores.masked_fill(~keep, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class TinyDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = TinyAttention(hidden, num_heads, num_kv_heads, head_dim)
        self.self_attn.layer_idx = layer_idx
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden, bias=False),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
    ):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(
            h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        hidden_states = residual + h
        return (hidden_states,)


class TinyVLM(nn.Module):
    def __init__(
        self,
        vocab: int = 64,
        hidden: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 8,
        image_token_id: int = 7,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [
                TinyDecoderLayer(i, hidden, num_heads, num_kv_heads, head_dim)
                for i in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden,
            image_token_id=image_token_id,
            vocab_size=vocab,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> TinyOutputs:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds is required.")
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache  # lazy
            past_key_values = DynamicCache()

        for layer in self.layers:
            h = layer(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )[0]

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return TinyOutputs(
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
        )


class TinyAdapter:
    """Minimal `ModelAdapter` implementation for `TinyVLM`."""

    def get_decoder_layers(self, model: TinyVLM) -> list[nn.Module]:
        return list(model.layers)

    def get_attn_kv_projs(self, layer: TinyDecoderLayer):
        return layer.self_attn.k_proj, layer.self_attn.v_proj

    def get_image_token_id(self, model: TinyVLM) -> int:
        return int(model.config.image_token_id)

    def num_kv_heads(self, model: TinyVLM) -> int:
        return int(model.config.num_key_value_heads)

    def head_dim(self, model: TinyVLM) -> int:
        return int(model.config.head_dim)

    def image_grid_shape(self, inputs, model):  # pragma: no cover — unused in unit tests
        raise NotImplementedError
