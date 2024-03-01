import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
from typing import Callable, List
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent))

from model_args import ModelArgs
from module import EncoderBlock, make_norm
from helper import load_model_state_dict


class CausalLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_vocab = args.n_vocab
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.n_kv_heads = (
            args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        )
        self.d_head = self.dim // self.n_heads
        self.ffn_hidden_dim = args.ffn_hidden_dim
        self.norm_eps = args.norm_eps
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len

        self.model = nn.ModuleDict(
            OrderedDict(
                {
                    "embed_tokens": nn.Embedding(self.n_vocab, self.dim),
                    "layers": nn.ModuleList(
                        [EncoderBlock(args) for _ in range(self.n_layers)]
                    ),
                    "norm": make_norm(args),
                }
            )
        )
        lm_head_bias = True if args.llm_type in ["phi"] else False
        self.lm_head = nn.Linear(self.dim, self.n_vocab, bias=lm_head_bias)

        self.start_index = 0

    def reset(self):
        self.start_index = 0
        return self

    def set_start_index(self, start_index: int):
        self.start_index = start_index

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.ndim <= 2
        while tokens.ndim < 2:
            tokens = tokens.unsqueeze(0)
        _, L = tokens.shape

        h = self.model["embed_tokens"](tokens)  # [B, L] --> [B, L, D]
        if self.args.llm_type in ["gemma"]:
            h = h * self.dim**0.5
        for layer in self.model["layers"]:
            h = layer(h, self.start_index)  # [B, L, D] --> [B, L, D]
        h = self.model["norm"](h)  # [B, L, D] --> [B, L, D]

        logits = self.lm_head(h)  # [B, L, D] --> [B, L, n_vocab]

        self.start_index += L
        return logits

    @staticmethod
    def from_pretrained(
        model_dir: Path | List[Path],
        model_args: ModelArgs,
        strict=True,
        convert_state_dict_fun: Callable = None,
    ) -> "CausalLM":
        state_dict: OrderedDict = load_model_state_dict(model_dir)
        if convert_state_dict_fun is not None:
            state_dict = convert_state_dict_fun(state_dict)
        model = CausalLM(model_args)
        model.load_state_dict(state_dict, strict=strict)
        return model


if __name__ == "__main__":
    model_args = ModelArgs(
        n_vocab=345,
        dim=128,
        n_layers=6,
        n_heads=8,
        n_kv_heads=None,
        ffn_hidden_dim=256,
        norm_eps=1e-5,
        norm_type="rmsnorm",
        norm_with_affine=True,
        max_batch_size=1,
        max_seq_len=512,
    )

    model = CausalLM(model_args)
    with open("./temp/llama2-dummy.txt", "w") as f:
        for k, v in model.state_dict().items():
            f.write(f"{k}: {[*v.shape]}\n")
