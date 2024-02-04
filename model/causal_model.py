import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import json
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent))

from model_args import ModelArgs
from module import EncoderBlock, make_norm


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

        self.tok_embeddings = nn.Embedding(self.n_vocab, self.dim)
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(self.n_layers)])
        self.norm = make_norm(**args.model_dump())
        self.output = nn.Linear(self.dim, self.n_vocab, bias=False)

        self.start_index = 0

    def reset(self):
        self.start_index = 0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.ndim <= 2
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        _, L = tokens.shape

        h = self.tok_embeddings(tokens)  # [B, L] --> [B, L, D]
        for layer in self.layers:
            h = layer(h, self.start_index)  # [B, L, D] --> [B, L, D]
        h = self.norm(h)  # [B, L, D] --> [B, L, D]
        logits = self.output(h)  # [B, L, D] --> [B, L, n_vocab]

        self.start_index += L
        return logits

    @staticmethod
    def from_pretrained(
        model_fn: Path,
        model_args: Path | ModelArgs,
        strict=True,
        convert_state_dict_fun: Callable = None,
    ) -> "CausalLM":
        state_dict = torch.load(model_fn, map_location="cpu")
        if convert_state_dict_fun is not None:
            state_dict = convert_state_dict_fun(state_dict)
        try:
            args = ModelArgs(**json.load(model_args.open()))
        except:
            args = model_args
        model = CausalLM(args)
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
