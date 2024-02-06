import torch

from pathlib import Path
from collections import OrderedDict
import safetensors.torch
from typing import Callable, List, Tuple

from model.model_args import ModelArgs


def get_device(device="auto") -> torch.device:
    assert device in ["auto", "cpu", "cuda", "mps"]
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def get_model_state_dict_filenames(model_dir: Path) -> List[Path]:
    assert model_dir.is_dir()
    if len(list(model_dir.glob("*.safetensors"))) > 0:
        return list(model_dir.glob("*.safetensors"))
    if len(list(model_dir.glob("*.pth"))) > 0:
        return list(model_dir.glob("*.pth"))
    if len(list(model_dir.glob("*.bin"))) > 0:
        return list(model_dir.glob("*.bin"))
    raise ValueError(f"Cannot find model files in {model_dir}")


def _load_model_state_dict(fn: Path | List[Path]) -> OrderedDict:
    state_dict = OrderedDict()
    if not isinstance(fn, (list, tuple)):
        fn = [fn]
    for f in fn:
        if f.suffix == ".safetensors":
            state_dict.update(safetensors.torch.load_file(f, device="cpu"))
        elif f.suffix in [".pth", ".bin"]:
            state_dict.update(torch.load(f, map_location="cpu"))
        else:
            raise ValueError(f"Unknown file type: {f.suffix}")
    return state_dict


def load_model_state_dict(model_path: Path | List[Path]) -> OrderedDict:
    try:  # model_path is a directory
        model_fns = get_model_state_dict_filenames(model_path)
        return _load_model_state_dict(model_fns)
    except:
        return _load_model_state_dict(model_path)


def print_model_state_dict(model_dir: Path, save_fn: Path = None) -> str:
    state_dict = load_model_state_dict(model_dir)
    state_dict_info = ""
    for k, v in state_dict.items():
        state_dict_info += f"{k}: {[*v.shape]}\n"
    if save_fn is not None:
        with open(save_fn, "w") as f:
            f.write(state_dict_info)
    return state_dict_info


def convert_state_dict_for_llama2(state_dict: OrderedDict) -> OrderedDict:
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        if "rotary_emb" in k:
            continue
        state_dict_new[k] = v
    return state_dict_new


def convert_state_dict_for_phi2(state_dict: OrderedDict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        k: str = k
        k = k.replace("fc1", "gate_proj")
        k = k.replace("fc2", "down_proj")
        k = k.replace("dense", "o_proj")
        k = k.replace("final_layernorm", "norm")
        state_dict_new[k] = v
    return state_dict_new


def convert_state_dict_for_olmo(state_dict: OrderedDict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        k: str = k
        v: torch.Tensor = v
        if "wte" in k:
            state_dict_new["model.embed_tokens.weight"] = v
            state_dict_new["lm_head.weight"] = v
        elif "blocks" in k:
            segs = k.split(".")
            bi = int(segs[3])  # block index
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
        llm_type="llama",
        n_vocab=32000,
        dim=2048,
        n_layers=22,
        n_heads=32,
        n_kv_heads=4,
        ffn_hidden_dim=5632,
        max_batch_size=1,
        max_seq_len=2048,
    ),
    "llama-7b": ModelArgs(
        llm_type="llama",
        n_vocab=50304,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,
        ffn_hidden_dim=11008,
        max_batch_size=1,
        max_seq_len=4096,
    ),
    "phi-2": ModelArgs(
        llm_type="phi",
        n_vocab=51200,
        dim=2560,
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,
        ffn_hidden_dim=10240,
        rope_partial_factor=0.4,
        max_batch_size=1,
        max_seq_len=2048,
    ),
    "phi-1_5": ModelArgs(
        llm_type="phi",
        n_vocab=51200,
        dim=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=None,
        ffn_hidden_dim=8192,
        rope_partial_factor=0.5,
        max_batch_size=1,
        max_seq_len=2048,
    ),
}

CONVERT_STATE_DICT_FUN_MAP = {
    "TinyLlama-1.1B-Chat-v1.0": convert_state_dict_for_llama2,
    "llama-7b": convert_state_dict_for_llama2,
    "OLMo-1B": convert_state_dict_for_olmo,
    "phi-2": convert_state_dict_for_phi2,
    "phi-1_5": convert_state_dict_for_phi2,
}


def preprocess_prompt_for_tinyllama_chat(
    user_prompt: str,
    history: List[Tuple[str, str]] = None,
    system_prompt: str = None,
) -> str:
    user_prompt = user_prompt.strip()
    sys_prompt = "You are a chatbot who can help answer questions."
    if system_prompt is not None:
        sys_prompt = system_prompt.strip()
    if history is None or len(history) == 0:
        return f"<|system|>\n{sys_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
    prompt = f"<|system|>\n{sys_prompt}</s>\n"
    for _, (u, r) in enumerate(history):
        prompt += f"<|user|>\n{u}</s>\n<|assistant|>\n{r}</s>\n"
    prompt += f"<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
    return prompt


def _default_fun(x, *args, **kwargs):
    return x


def postprocess_responce_for_tinyllama_chat(response: str) -> str:
    idx = response.find("</s>")
    idx = 0 if idx == -1 else idx + 4
    response = response[idx:].strip()
    response = response.replace("<|assistant|>", "assistant: ")
    response = response.replace("<|user|>", "user: ")
    response = response.replace("</s>", "")
    response = response.replace("<s>", "")
    return response


PROMPT_PREPROCESS_FUN_MAP = {
    "TinyLlama-1.1B-Chat-v1.0": preprocess_prompt_for_tinyllama_chat,
}

RESPONCE_POSTPROCESS_FUN_MAP = {
    "TinyLlama-1.1B-Chat-v1.0": postprocess_responce_for_tinyllama_chat,
}


def get_prompt_preprocess_fun(
    model_name: str,
) -> Callable[[str, List[Tuple[str, str]], str], str] | None:
    return PROMPT_PREPROCESS_FUN_MAP.get(model_name, _default_fun)


def get_responce_postprocess_fun(
    model_name: str,
) -> Callable[[str], str] | None:
    return RESPONCE_POSTPROCESS_FUN_MAP.get(model_name, _default_fun)


def get_state_dict_convert_fun(
    model_name: str,
) -> Callable[[OrderedDict], OrderedDict] | None:
    return CONVERT_STATE_DICT_FUN_MAP.get(model_name, _default_fun)


def get_model_args(model_name: str) -> ModelArgs:
    return MODEL_ARGS_MAP.get(model_name, None)
