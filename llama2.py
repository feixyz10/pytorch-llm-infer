import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from arg import ModelArgs
from module import EncoderBlock, RMSNorm


class Llama2ForCausalLM(nn.Module):
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
        self.norm = RMSNorm(self.dim, eps=args.norm_eps)
        self.output = nn.Linear(self.dim, self.n_vocab, bias=False)

        self.start_index = 0

    def reset(self):
        self.start_index = 0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.ndim <= 2
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        _, L = tokens.shape

        # [B, L] --> [B, L, D]
        h = self.tok_embeddings(tokens)
        # [B, L, D] --> [B, L, D]
        for layer in self.layers:
            h = layer(h, self.start_index)
        # [B, L, D] --> [B, L, D]
        h = self.norm(h)
        # [B, L, D] --> [B, L, n_vocab]
        logits = self.output(h)

        self.start_index += L

        return logits

    @staticmethod
    def from_pretrained(model_fn: Path, model_args_fn: Path = None, strict=True):
        state_dict = torch.load(model_fn, map_location="cpu")
        args = ModelArgs()
        if model_args_fn is not None:
            args = ModelArgs(**json.load(model_args_fn.open()))
        model = Llama2ForCausalLM(args)
        model.load_state_dict(state_dict, strict=strict)
        return model


if __name__ == "__main__":

    model_args = ModelArgs(
        n_vocab=1234,
        dim=64,
        n_layers=4,
        n_heads=8,
        n_kv_heads=None,
        norm_eps=1e-5,
        ffn_hidden_dim=128,
        max_batch_size=1,
        max_seq_len=256,
    )

    model = Llama2ForCausalLM(model_args)
    o1 = model(torch.tensor([1, 2, 3]))
    o2 = model(torch.tensor([5, 6]))
    o3 = model(torch.tensor([10]))
    model.reset()
    o3 = model(torch.tensor([10]))
    with open("./temp/llama2-7B.txt", "w") as f:
        for k, v in model.state_dict().items():
            f.write(f"{k}: {[*v.shape]}\n")
