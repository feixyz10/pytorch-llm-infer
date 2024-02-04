import json
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel


class ModelArgs(BaseModel):
    n_vocab: int
    dim: int

    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int] = None

    ffn_hidden_dim: int

    norm_type: Literal["rmsnorm", "default"] = "rmsnorm"
    norm_eps: float = 1e-5
    norm_with_affine: bool = True

    max_batch_size: int = 1
    max_seq_len: int = 2048
