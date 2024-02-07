import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from typing import Tuple
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from attention import scaled_dot_product_attention_gqa
from rope import RoPE
from model_args import ModelArgs


def make_norm(args: ModelArgs) -> nn.Module:
    if args.llm_type in ["phi"]:
        return nn.LayerNorm(
            args.dim, eps=args.norm_eps, elementwise_affine=True, bias=True
        )
    return RMSNorm(args.dim, eps=args.norm_eps)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.self_attn = SelfAttention(args)
        self.mlp = FeedForward(args)
        self.input_layernorm = make_norm(args)
        if args.llm_type in ["phi"]:
            self.post_attention_layernorm = nn.Identity()
        else:  # llama and similars
            self.post_attention_layernorm = make_norm(args)

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        if self.args.llm_type in ["phi"]:
            residual = x
            x = self.input_layernorm(x)
            x1, x2 = self.self_attn(x, start_index), self.mlp(x)
            return residual + x1 + x2
        x = x + self.self_attn(self.input_layernorm(x), start_index)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads, self.n_kv_heads = args.n_heads, args.n_kv_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        self.d_head = args.dim // args.n_heads

        bias = True if args.llm_type in ["phi", "qwen"] else False
        self.q_proj = nn.Linear(args.dim, self.n_heads * self.d_head, bias=bias)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.d_head, bias=bias)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.d_head, bias=bias)
        bias = True if args.llm_type in ["phi"] else False
        self.o_proj = nn.Linear(self.n_heads * self.d_head, args.dim, bias=bias)

        self.kv_cache = KVCache(
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            n_kv_heads=self.n_kv_heads,
            d_head=self.d_head,
        )
        self.rope = RoPE(args)

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if start_index == 0:
            self.kv_cache.reset()

        # [B, L, D] --> [B, L, D]
        q: torch.Tensor = self.q_proj(x)
        # [B, L, D] --> [B, L, D_kv], D_kv may smaller than D
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        # [B, L, D] --> [B, L, n_heads, d_head]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_head)

        # apply rotary position embedding
        q = self.rope(q, start_index)
        k = self.rope(k, start_index)

        # write new key and new value into the kv cache
        self.kv_cache(k, v)
        # read out all cached key and value --> [B, L_kv, n_kv_heads, d_head]
        k, v = self.kv_cache()

        # [B, L, n_heads, d_head] --> [B, n_heads, L, d_head]
        q = q.permute(0, 2, 1, 3).contiguous()
        # [B, L_kv, n_kv_heads, d_head] --> [B, n_kv_heads, L_kv, d_head]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        # --> [B, n_heads, L, d_head], if query seq length == 1,
        # set is_causal to False to avoid attention mask construction to save computation
        output = scaled_dot_product_attention_gqa(q, k, v, is_causal=q.shape[-2] > 1)
        # [B, n_heads, L, d_head] --> [B, L, n_heads, d_head] --> [B, L, D]
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # [B, L, D] --> [B, L, D]
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.llm_type = args.llm_type
        self.dim = args.dim
        self.hidden_dim = args.ffn_hidden_dim

        if self.llm_type in ["phi"]:
            self.gate_proj = nn.Linear(self.dim, self.hidden_dim, bias=True)
            self.down_proj = nn.Linear(self.hidden_dim, self.dim, bias=True)
        else:
            self.gate_proj = nn.Linear(self.dim, self.hidden_dim, bias=False)
            self.down_proj = nn.Linear(self.hidden_dim, self.dim, bias=False)
            self.up_proj = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.llm_type in ["phi"]:
            # [B, L, D] --> [B, L, hD]
            x1 = F.gelu(self.gate_proj(x), approximate="tanh")
            # [B, L, hD] --> [B, L, D]
            return self.down_proj(x1)
        # [B, L, D] --> [B, L, hD]
        x1, x2 = F.silu(self.gate_proj(x)), self.up_proj(x)
        # [B, L, hD] --> [B, L, D]
        return self.down_proj(x1 * x2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim, self.eps = dim, eps

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x) -> torch.Tensor:
        # [B, L, D] --> [B, L, D]
        x = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # [D] * [B, L, D] --> [B, L, D], broadcasting
        return self.weight * x


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size: int, max_seq_len: int, n_kv_heads: int, d_head: int
    ):
        super().__init__()
        self.max_batch_size, self.max_seq_len = max_batch_size, max_seq_len
        self.n_kv_heads, self.d_head = n_kv_heads, d_head

        kv_cache_shape = torch.Size([max_batch_size, max_seq_len, n_kv_heads, d_head])
        k_cache = torch.zeros(kv_cache_shape)
        v_cache = torch.zeros(kv_cache_shape)
        self.register_buffer("k_cache", k_cache, persistent=False)
        self.register_buffer("v_cache", v_cache, persistent=False)

        self.batch_size = 0  # actual batch size
        self.seq_len = 0  # seq length have been cached

    def reset(self):
        self.batch_size = 0
        self.seq_len = 0

    def forward(
        self, k: torch.Tensor = None, v: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if k is not None and v is not None:
            return self._write(k, v)
        assert k is None and v is None
        assert self.batch_size > 0 and self.seq_len > 0, "nothing has been cached"
        return self._read()

    def _write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        assert k.ndim == 4 and k.shape == v.shape
        if self.batch_size == 0:
            self.batch_size = k.shape[0]
        assert k.shape[2] == self.n_kv_heads and k.shape[3] == self.d_head
        assert self.seq_len + k.shape[1] <= self.max_seq_len

        self.k_cache[: self.batch_size, self.seq_len : self.seq_len + k.shape[1]] = k
        self.v_cache[: self.batch_size, self.seq_len : self.seq_len + k.shape[1]] = v
        self.seq_len += k.shape[1]

    def _read(self) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.k_cache[: self.batch_size, : self.seq_len]
        v = self.v_cache[: self.batch_size, : self.seq_len]
        return k, v


if __name__ == "__main__":
    model_args = ModelArgs(
        n_vocab=32,
        dim=64,
        n_layers=4,
        n_heads=8,
        n_kv_heads=None,
        norm_eps=1e-5,
        ffn_hidden_dim=128,
        max_batch_size=1,
        max_seq_len=256,
    )
    rope = RoPE(model_args)
    rope(torch.randn(1, 4, 8, 64))

    kvcache = KVCache(max_batch_size=1, max_seq_len=128, n_kv_heads=8, d_head=64)
    kvcache(torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64))
    o1, o2 = kvcache()
    assert o1.shape == torch.Size([1, 4, 8, 64])
    assert o2.shape == torch.Size([1, 4, 8, 64])
    kvcache(torch.randn(1, 3, 8, 64), torch.randn(1, 3, 8, 64))
    o1, o2 = kvcache()
    assert o1.shape == torch.Size([1, 7, 8, 64])
    assert o2.shape == torch.Size([1, 7, 8, 64])
