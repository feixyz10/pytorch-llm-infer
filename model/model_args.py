from typing import Literal, Optional
from pydantic import BaseModel


class ModelArgs(BaseModel):
    llm_type: Literal["llama", "phi"] = "llama"
    # basic
    dim: int = -1
    n_vocab: int = -1
    n_layers: int = -1
    # attention
    n_heads: int = -1
    n_kv_heads: Optional[int] = None  # None means n_kv_heads = n_heads
    # mlp
    ffn_hidden_dim: int = -1
    # norm
    norm_eps: float = 1e-5
    # position embedding
    rope_theta: float = 10000.0
    rope_partial_factor: Optional[float] = None
    # other
    max_batch_size: int = 1
    max_seq_len: int = 2048
