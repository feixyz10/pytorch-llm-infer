import torch
import torch.nn as nn

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from model_args import ModelArgs


class RoPE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.max_seq_len, self.inv_theta = args.max_seq_len, 1.0 / args.rope_theta
        self.dim = args.dim // args.n_heads
        self.rope_dim = self.dim
        if args.rope_partial_factor is not None:
            self.rope_dim = int(args.rope_partial_factor * self.dim)
        assert self.rope_dim % 2 == 0, f"Dim ({self.rope_dim}) must be divisible by 2"

        # complex of shape: [max_seq_len, rope_dim//2]
        freqs_complex: torch.Tensor = self._precompute_inv_freq()
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[-1] == self.dim
        _, L, _, _ = x.shape
        assert start_index + L <= self.max_seq_len
        if self.dim == self.rope_dim:
            return self._forward(x, start_index)
        x, x_pass = x[..., : self.rope_dim].contiguous(), x[..., self.rope_dim :]
        x = self._forward(x, start_index)
        return torch.cat([x, x_pass], dim=-1)

    def _forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        B, L, H, _ = x.shape
        # NONE: llama2 (and followings) implementation of RoPE is a bit different from the paper.
        # I wasted a lot of time trying to figure it out.
        # [B, L, H, rope_dim] --> [B, L, H, 2, rope_dim/2] --> [B, L, H, rope_dim/2, 2]
        x = x.reshape(B, L, H, 2, -1).transpose(-1, -2).contiguous().float()
        # --> complex of shape [B, L, H, rope_dim/2]
        x_complex = torch.view_as_complex(x)
        # [L, rope_dim/2] --> [1, L, 1, rope_dim/2]
        f_complex = self.freqs_complex[start_index : start_index + L].view(1, L, 1, -1)
        # [1, L, 1, rope_dim/2] x [B, L, H, rope_dim/2] --> [B, L, H, rope_dim/2]
        x_rotated = f_complex * x_complex
        # complex of shape [B, L, H, rope_dim/2] --> real of shape [B, L, H, rope_dim/2, 2] -> [B, L, H, 2, rope_dim/2]
        # NOTE: the following two line of codes give totally different results,
        # but since it will be applied for both query and key in self-attension,
        # so (surprisingly but can be proved) it will lead to same result after self-attention.
        # BTW, the second line is more efficient.
        x_rotated = torch.view_as_real(x_rotated).transpose(-1, -2)
        # x_rotated = torch.view_as_real(x_rotated)
        return x_rotated.reshape(B, L, H, -1).type_as(x)  # --> [B, L, H, rope_dim]

    def _precompute_inv_freq(self) -> torch.Tensor:
        dtype = torch.float32
        # theta_i = 10000^(-2(i-1)/rope_dim), i = [1, 2, ..., rope_dim/2]
        i = torch.arange(0, self.rope_dim, 2, dtype=dtype)
        inv_freq = self.inv_theta ** (i / self.rope_dim)  # shape: [rope_dim/2]
        t = torch.arange(0, self.max_seq_len, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # shape: [max_seq_len, dim/2]
        # --> exp(j * freqs) = cos(freqs) + j * sin(freqs), complex of shape: [max_seq_len, dim/2]
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex


class RoPEref(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.max_seq_len, self.inv_theta = args.max_seq_len, 1.0 / args.rope_theta
        self.dim = args.dim // args.n_heads
        self.rope_dim = self.dim
        if args.rope_partial_factor is not None:
            self.rope_dim = int(args.rope_partial_factor * self.dim)
        assert self.rope_dim % 2 == 0, f"Dim ({self.rope_dim}) must be divisible by 2"
        self.dtype = torch.float32

        i = torch.arange(0, self.rope_dim, 2, dtype=self.dtype)
        inv_freq = self.inv_theta ** (i / self.rope_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = 0
        self._set_cos_sin_cache(
            seq_len=self.max_seq_len,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )

    def forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[-1] == self.dim
        _, L, _, _ = x.shape

        if start_index + L > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=start_index + L, device=x.device, dtype=x.dtype
            )

        if self.dim == self.rope_dim:
            return self._forward(x, start_index)
        x, x_pass = x[..., : self.rope_dim].contiguous(), x[..., self.rope_dim :]
        x = self._forward(x, start_index)
        return torch.cat([x, x_pass], dim=-1)

    def _forward(self, x: torch.Tensor, start_index: int) -> torch.Tensor:

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        _, L, _, _ = x.shape  # [B, L, H, rope_dim]
        cos = self.cos_cache[start_index : start_index + L].view(1, L, 1, -1)
        sin = self.sin_cache[start_index : start_index + L].view(1, L, 1, -1)
        x_emb = (x * cos) + (rotate_half(x) * sin)
        return x_emb

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cache", emb.sin().to(dtype), persistent=False)

    def _precompute_inv_freq(self) -> torch.Tensor:
        dtype = torch.float32
        inv_theta = 1.0 / self.theta
        # theta_i = 10000^(-2(i-1)/rope_dim), i = [1, 2, ..., rope_dim/2]
        i = torch.arange(0, self.rope_dim, 2, dtype=dtype)
        inv_freq = inv_theta ** (i / self.rope_dim)  # shape: [rope_dim/2]
        t = torch.arange(0, self.max_seq_len, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # shape: [max_seq_len, dim/2]
        # --> exp(j * freqs) = cos(freqs) + j * sin(freqs), complex of shape: [max_seq_len, dim/2]
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex


if __name__ == "__main__":
    torch.random.manual_seed(0)

    args = ModelArgs(dim=128, n_heads=4, max_seq_len=15, rope_partial_factor=0.5)

    rope = RoPE(args)
    rope2 = RoPEref(args)

    x = torch.randn(1, 10, 4, 128 // 4)
    y1, y3 = rope(x, 0), rope(x, 4)
    y2, y4 = rope2(x, 0), rope2(x, 4)

    assert torch.allclose(y1, y2)
    assert torch.allclose(y3, y4)
