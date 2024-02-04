import torch

from pathlib import Path
from collections import OrderedDict
from typing import List

from model.model_args import ModelArgs


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def convert_state_dict_for_llama2(state_dict: OrderedDict) -> OrderedDict:
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        if "rotary_emb" in k:
            continue
        state_dict_new[k] = v
    return state_dict_new


def get_model_filenames(model_dir: Path) -> List[Path]:
    assert model_dir.is_dir()
    if len(list(model_dir.glob("*.safetensors"))) > 0:
        return list(model_dir.glob("*.safetensors"))
    if len(list(model_dir.glob("*.pth"))) > 0:
        return list(model_dir.glob("*.pth"))
    if len(list(model_dir.glob("*.bin"))) > 0:
        return list(model_dir.glob("*.bin"))
    raise ValueError(f"Cannot find model files in {model_dir}")


def convert_state_dict_for_olmo(state_dict: OrderedDict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        k: str = k
        v: torch.Tensor = v
        if "wte" in k:
            state_dict_new["model.embed_tokens.weight"] = v
        elif "blocks" in k:
            segs = k.split(".")
            bi = int(segs[3])
            weight_name = segs[4]
            if weight_name == "attn_out":
                state_dict_new[f"model.layers.{bi}.self_attn.o_proj.weight"] = v
            elif weight_name == "att_proj":
                wq, wk, wv = v.chunk(3)
                state_dict_new[f"model.layers.{bi}.self_attn.q_proj.weight"] = wq
                state_dict_new[f"model.layers.{bi}.self_attn.k_proj.weight"] = wk
                state_dict_new[f"model.layers.{bi}.self_attn.v_proj.weight"] = wv
            elif weight_name == "ff_out":
                state_dict_new[f"model.layers.{bi}.mlp.down_proj.weight"] = v
            elif weight_name == "ff_proj":
                # https://github.com/allenai/OLMo/blob/main/olmo/model.py#L373
                w1, w2 = v.chunk(2)
                state_dict_new[f"model.layers.{bi}.mlp.gate_proj.weight"] = w2
                state_dict_new[f"model.layers.{bi}.mlp.up_proj.weight"] = w1
            else:
                raise ValueError(f"Unknown weight name for layer {bi}: {weight_name}")
        else:
            raise ValueError(f"Unknown key: {k}")

    return state_dict_new


MODEL_ARGS_MAP = {
    "TinyLlama-1.1B-Chat-v1.0": ModelArgs(
        n_vocab=32000,
        dim=2048,
        n_layers=22,
        n_heads=32,
        n_kv_heads=4,
        ffn_hidden_dim=5632,
        norm_eps=1e-5,
        norm_type="rmsnorm",
        max_batch_size=1,
        max_seq_len=2048,
    ),
    "llama-7b": ModelArgs(
        n_vocab=50304,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,
        ffn_hidden_dim=11008,
        norm_eps=1e-5,
        norm_type="rmsnorm",
        max_batch_size=1,
        max_seq_len=4096,
    ),
    "chinese-alpaca-2-1.3b": ModelArgs(
        n_vocab=55296,
        dim=4096,
        n_layers=4,
        n_heads=32,
        n_kv_heads=None,
        ffn_hidden_dim=11008,
        norm_eps=1e-5,
        norm_type="rmsnorm",
        max_batch_size=1,
        max_seq_len=4096,
    ),
    "OLMo-1B": ModelArgs(
        n_vocab=50304,
        dim=2048,
        n_layers=16,
        n_heads=16,
        n_kv_heads=None,
        norm_eps=1e-5,
        norm_type="default",
        norm_with_affine=False,
        ffn_hidden_dim=8192,
        max_batch_size=1,
        max_seq_len=2048,
    ),
    "phi-2": ModelArgs(
        n_vocab=51200,
        dim=2560,
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,
        ffn_hidden_dim=10240,
        norm_eps=1e-5,
        norm_type="rmsnorm",
        max_batch_size=1,
        max_seq_len=2048,
    ),
}

MODEL_ARGS_MAP["chinese-llama-2-1.3b"] = MODEL_ARGS_MAP["chinese-alpaca-2-1.3b"]

CONVERT_STATE_DICT_FUN_MAP = {
    "TinyLlama-1.1B-Chat-v1.0": convert_state_dict_for_llama2,
    "llama-7b": convert_state_dict_for_llama2,
    "OLMo-1B": convert_state_dict_for_olmo,
    "chinese-alpaca-2-1.3b": convert_state_dict_for_llama2,
    "chinese-llama-2-1.3b": convert_state_dict_for_llama2,
}
