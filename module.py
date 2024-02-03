import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from attention import scaled_dot_product_attention_gqa
from arg import ModelArgs


def make_norm(
    dim: int,
    norm_eps: float = 1e-5,
    norm_type: str = "rmsnorm",
    norm_with_affine: bool = True,
    **kwargs
) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=norm_eps)
    else:
        return nn.LayerNorm(
            dim, eps=norm_eps, elementwise_affine=norm_with_affine, bias=False
        )


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = (
            args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        )
        self.ffn_hidden_dim = args.ffn_hidden_dim

        self.attention = SelfAttention(
            self.dim,
            self.n_heads,
            self.n_kv_heads,
            args.max_batch_size,
            args.max_seq_len,
        )
        self.feed_forward = FeedForward(self.dim, self.ffn_hidden_dim)
        self.attention_norm = make_norm(**args.model_dump())
        self.feed_forward_norm = make_norm(**args.model_dump())

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        # [B, L, D] + [B, L, D] --> [B, L, D]
        x = x + self.attention(self.attention_norm(x), start_index)
        # [B, L, D] + [B, L, D] --> [B, L, D]
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()

        self.d, self.n_heads, self.n_kv_heads = dim, n_heads, n_kv_heads
        self.d_head = dim // n_heads

        self.wq = nn.Linear(self.d, self.n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(self.d, self.n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(self.d, self.n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(self.n_heads * self.d_head, self.d, bias=False)

        self.kv_cache = KVCache(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_kv_heads=self.n_kv_heads,
            d_head=self.d_head,
        )
        self.rope = RoPE(max_seq_len=max_seq_len, dim=self.d_head, theta=10000.0)

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if start_index == 0:
            self.kv_cache.reset()

        # [B, L, D] --> [B, L, D]
        q: torch.Tensor = self.wq(x)
        # [B, L, D] --> [B, L, D_kv], D_kv may smaller than D
        k: torch.Tensor = self.wk(x)
        v: torch.Tensor = self.wv(x)

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
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_hidden_dim: int):
        super().__init__()
        self.dim, self.hidden_dim = dim, ffn_hidden_dim

        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] --> [B, L, hD]
        x1, x2 = F.silu(self.w1(x)), self.w3(x)
        # [B, L, hD] --> [B, L, D]
        return self.w2(x1 * x2)


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


class RoPE(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        dim: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be divisible by 2"
        self.max_seq_len, self.dim = max_seq_len, dim
        self.inv_theta = 1.0 / theta

        # [max_seq_len, dim//2]
        freqs_complex: torch.Tensor = self._precompute_freqs()
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def forward(self, x: torch.Tensor, start_index: int = 0) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[3] == self.dim
        B, L, H, D = x.shape
        assert start_index + L <= self.max_seq_len
        # [B, L, H, D] --> [B, L, H, D/2, 2] --> complex of shape [B, L, H, D/2]
        x_complex = torch.view_as_complex(x.view(B, L, H, D // 2, 2).float())
        # [L, D/2] --> [1, L, 1, D/2]
        f_complex = self.freqs_complex[start_index : start_index + L].view(1, L, 1, -1)
        # [1, L, 1, D/2] x [B, L, H, D/2] --> [B, L, H, D/2]
        x_rotated = f_complex * x_complex
        # complex of shape [B, L, H, D/2] --> real of shape [B, L, H, D/2, 2] --> [B, L, H, D]
        output = torch.view_as_real(x_rotated).view(B, L, H, D)
        return output.type_as(x)

    def _precompute_freqs(self) -> torch.Tensor:
        dtype = torch.float32
        # theta_i = 10000^(-2(i-1)/dim), i = [1, 2, ... dim/2]
        i = torch.arange(0, self.dim, 2, dtype=dtype)
        inv_freq = self.inv_theta ** (i / self.dim)  # shape: [dim/2]
        t = torch.arange(0, self.max_seq_len, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # shape: [max_seq_len, dim/2]
        # --> exp(j * freqs) = cos(freqs) + j * sin(freqs), complex of shape: [max_seq_len, dim/2]
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex


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
    rope = RoPE(max_seq_len=128, dim=64)
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
