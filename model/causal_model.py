import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import json
import safetensors.torch
from pathlib import Path
from typing import Callable, List
from collections import OrderedDict

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

        self.model = nn.ModuleDict(
            OrderedDict(
                {
                    "embed_tokens": nn.Embedding(self.n_vocab, self.dim),
                    "layers": nn.ModuleList(
                        [EncoderBlock(args) for _ in range(self.n_layers)]
                    ),
                    "norm": make_norm(**args.model_dump()),
                }
            )
        )
        self.lm_head = nn.Linear(self.dim, self.n_vocab, bias=False)

        self.start_index = 0

    def reset(self):
        self.start_index = 0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.ndim <= 2
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        _, L = tokens.shape

        h = self.model["embed_tokens"](tokens)  # [B, L] --> [B, L, D]
        for layer in self.model["layers"]:
            h = layer(h, self.start_index)  # [B, L, D] --> [B, L, D]
        h = self.model["norm"](h)  # [B, L, D] --> [B, L, D]

        logits = self.lm_head(h)

        self.start_index += L
        return logits

    @staticmethod
    def from_pretrained(
        model_fns: Path | List[Path],
        model_args: Path | ModelArgs,
        strict=True,
        convert_state_dict_fun: Callable = None,
    ) -> "CausalLM":
        if isinstance(model_fns, Path):
            model_fns = [model_fns]
        assert all([fn.is_file() for fn in model_fns])
        state_dict = OrderedDict()
        for fn in model_fns:
            state_dict.update(_load_state_dict(fn))
        if convert_state_dict_fun is not None:
            state_dict = convert_state_dict_fun(state_dict)
        try:
            args = ModelArgs(**json.load(model_args.open()))
        except:
            args = model_args
        model = CausalLM(args)
        model.load_state_dict(state_dict, strict=strict)
        return model


def _load_state_dict(fn: Path) -> OrderedDict:
    if fn.suffix == ".safetensors":
        state_dict = safetensors.torch.load_file(fn, device="cpu")
    elif fn.suffix in [".bin"]:
        state_dict = torch.load(fn, map_location="cpu")
    else:
        raise ValueError(f"Unknown file type: {fn.suffix}")
    return state_dict


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
