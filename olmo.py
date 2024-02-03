import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from collections import OrderedDict

from arg import ModelArgs
from module import EncoderBlock, make_norm


class OlmoModelForCausalLM(nn.Module):
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
    def from_pretrained(
        model_fn: Path, model_args: Path | ModelArgs = None, strict=True
    ):
        state_dict = torch.load(model_fn, map_location="cpu")
        args = ModelArgs()
        if model_args is not None:
            if isinstance(model_args, ModelArgs):
                args = model_args
            else:
                args = ModelArgs(**json.load(model_args.open()))
        model = __class__(args)
        model.load_state_dict(__class__.convert_state_dict(state_dict), strict=strict)
        return model

    @staticmethod
    def convert_state_dict(state_dict: OrderedDict):
        state_dict_new = OrderedDict()
        for k, v in state_dict.items():
            k: str = k
            v: torch.Tensor = v
            if "transformer.wte" in k:
                state_dict_new["tok_embeddings.weight"] = v
            elif "transformer.blocks" in k:
                segs = k.split(".")
                li = int(segs[3])
                weight_name = segs[4]
                if weight_name == "attn_out":
                    state_dict_new[f"layers.{li}.attention.wo.weight"] = v
                elif weight_name == "att_proj":
                    wq, wk, wv = v.chunk(3)
                    state_dict_new[f"layers.{li}.attention.wq.weight"] = wq
                    state_dict_new[f"layers.{li}.attention.wk.weight"] = wk
                    state_dict_new[f"layers.{li}.attention.wv.weight"] = wv
                elif weight_name == "ff_out":
                    state_dict_new[f"layers.{li}.feed_forward.w2.weight"] = v
                elif weight_name == "ff_proj":
                    # https://github.com/allenai/OLMo/blob/main/olmo/model.py#L373
                    w1, w2 = v.chunk(2)
                    state_dict_new[f"layers.{li}.feed_forward.w1.weight"] = w2
                    state_dict_new[f"layers.{li}.feed_forward.w3.weight"] = w1
                else:
                    raise ValueError(
                        f"Unknown weight name for layer {li}: {weight_name}"
                    )
            else:
                raise ValueError(f"Unknown key: {k}")

        return state_dict_new


if __name__ == "__main__":

    model_args = ModelArgs(
        n_vocab=50304,
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=None,
        norm_eps=1e-5,
        norm_type="default",
        norm_with_affine=False,
        ffn_hidden_dim=8192,
        max_batch_size=1,
        max_seq_len=256,
    )

    model_fn = (
        Path.home() / "Downloads/checkpoints/huggingface/OLMo-1B/pytorch_model.bin"
    )
    model = OlmoModelForCausalLM.from_pretrained(model_fn, model_args, strict=False)
    with open("./temp/olmo-1B-converted.txt", "w") as f:
        for k, v in model.state_dict().items():
            f.write(f"{k}: {[*v.shape]}\n")
