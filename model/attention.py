import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Scaled dot product attention with support for group queriy attention (GQA). if n_heads == n_kv_heads,
    this is fallback to multi-head attention (MHA).

    Args:
        query: [B, ..., n_heads, L, E]
        key: [B, ..., n_kv_heads, S, E]
        value: [B, ..., n_kv_heads, S, Ev]
        kwargs: extra arguments, e.g. attn_mask, dropout_p, is_causal, scale

        Above B is batch size, L is query sequence length, S is key&value sequence length,
        E is query&key embedding dimension, Ev is value embedding dimension,
        n_heads is number of query heads, n_kv_heads is number of key&value heads.

        NOTE: n_heads should be divisible by n_kv_heads, L must be <= S.

    Returns:
        Attention output of shape [B, ..., n_heads, L, Ev]
    """
    n_heads, L, _ = query.shape[-3:]
    n_kv_heads, S, _ = key.shape[-3:]
    assert n_heads % n_kv_heads == 0 and L <= S

    attn_mask = kwargs.get("attn_mask", None)
    dropout_p = kwargs.get("dropout_p", 0.0)
    is_causal = kwargs.get("is_causal", False)
    scale = kwargs.get("scale", None)

    if is_causal and attn_mask is None:
        mask_shape = (*query.shape[:-2], L, S)
        attn_mask = torch.tril(
            torch.ones(mask_shape, dtype=torch.bool, device=query.device),
            diagonal=S - L,
        )

    if n_heads == n_kv_heads:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )

    query = query.view(*query.shape[:-3], n_kv_heads, n_heads // n_kv_heads, L, -1)
    query = query.transpose(-3, -4)
    key = key.view(*key.shape[:-3], 1, n_kv_heads, S, -1)
    value = value.view(*value.shape[:-3], 1, n_kv_heads, S, -1)
    if is_causal:
        mask_shape = (*query.shape[:-2], L, S)
        attn_mask = attn_mask.view(*mask_shape)
    # # following lines are equivalent to the F.scaled_dot_product_attention, but may slower
    # weights = (query @ key.transpose(-2, -1)) / math.sqrt(key.size(-1))
    # if is_causal:
    #     weights = weights.masked_fill(~attn_mask, -float("inf"))
    # weights = F.softmax(weights, dim=-1)
    # output = weights @ v
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
    )
    return output.transpose(-3, -4).reshape(*output.shape[:-4], n_heads, L, -1)


def scaled_dot_product_attention_gqa2(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Same as scaled_dot_product_attention_gqa. Easier to understand but slower.
    Recommend to use scaled_dot_product_attention_gqa instead.
    """

    def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
        batch_size, n_kv_heads, seq_len, head_dim = x.shape
        if n_repeat == 1:
            return x
        x = x[:, :, None, :, :]
        x = x.expand(batch_size, n_kv_heads, n_repeat, seq_len, head_dim)
        return x.reshape(batch_size, n_kv_heads * n_repeat, seq_len, head_dim)

    n_heads, n_kv_heads = q.shape[-3], v.shape[-3]
    L, S = q.shape[-2], v.shape[-2]
    if n_heads != n_kv_heads:
        k = repeat_kv(k, n_heads // n_kv_heads)
        v = repeat_kv(v, n_heads // n_kv_heads)
    weights = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
    attn_mask = torch.tril(torch.ones_like(weights, dtype=torch.bool), diagonal=S - L)
    weights = weights.masked_fill(~attn_mask, -float("inf"))
    weights = F.softmax(weights, dim=-1)
    output = weights @ v
    return output


if __name__ == "__main__":

    def test1(q, k, v):
        """two methods give same result"""
        o1 = scaled_dot_product_attention_gqa(q, k, v, is_causal=True)
        o2 = scaled_dot_product_attention_gqa2(q, k, v)
        assert torch.allclose(o1, o2)

    ## multi-head attention
    q = torch.randn(1, 6, 4, 10)  # [B, n_heads, L, d_head]
    k = torch.randn(1, 6, 10, 10)  # [B, n_kv_heads, S, d_head]
    v = torch.eye(10).unsqueeze(0).unsqueeze(0).repeat(1, k.shape[1], 1, 1)
    test1(q, k, v)

    ## group query attention, n_heads != n_kv_heads
    k = torch.randn(1, 2, 10, 10)
    v = torch.eye(10).unsqueeze(0).unsqueeze(0).repeat(1, k.shape[1], 1, 1)
    test1(q, k, v)

    def test2(q, k, v):
        """is_causal has no effect on result if query seq length == 1"""
        assert q.shape[-2] == 1
        o1 = scaled_dot_product_attention_gqa(q, k, v, is_causal=True)
        o2 = scaled_dot_product_attention_gqa(q, k, v, is_casual=False)
        assert torch.allclose(o1, o2)

    q = torch.randn(1, 6, 1, 10)
    test2(q, k, v)
    q = torch.randn(1, 2, 1, 10)
    test2(q, k, v)
