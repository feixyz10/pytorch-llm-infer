from typing import Literal, Optional
from pydantic import BaseModel


class ModelArgs(BaseModel):
    n_vocab: int = 32000
    dim: int = 4096

    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None

    ffn_hidden_dim: int = 10732

    norm_type: Literal["rmsnorm", "default"] = "rmsnorm"
    norm_eps: float = 1e-5
    norm_with_affine: bool = True

    max_batch_size: int = 1
    max_seq_len: int = 2048


class GenerationConfig(BaseModel):
    max_prompt_length: int = 512  # max length of prompt, truncated if longer
    do_sample: bool = True  # sampling next token if True, otherwise greedy search
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
