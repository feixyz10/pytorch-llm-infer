import torch

from collections import OrderedDict

from model.model_args import ModelArgs


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def convert_state_dict_for_llama2(state_dict: OrderedDict) -> OrderedDict:
    del state_dict["rope.freqs"]
    return state_dict


def convert_state_dict_for_chinese_llama2(state_dict: OrderedDict) -> OrderedDict:
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        k: str = k
        if "rotary_emb" in k:
            continue
        k = k.replace("model.", "")
        k = k.replace("embed_tokens", "tok_embeddings")
        k = k.replace("self_attn", "attention")
        k = k.replace("mlp", "feed_forward")
        k = k.replace("q_proj", "wq")
        k = k.replace("k_proj", "wk")
        k = k.replace("v_proj", "wv")
        k = k.replace("o_proj", "wo")
        k = k.replace("gate_proj", "w1")
        k = k.replace("down_proj", "w2")
        k = k.replace("up_proj", "w3")
        k = k.replace("input_layernorm", "attention_norm")
        k = k.replace("post_attention_layernorm", "feed_forward_norm")
        k = k.replace("lm_head", "output")
        state_dict_new[k] = v
    return state_dict_new


def convert_state_dict_for_olmo(state_dict: OrderedDict):
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
                raise ValueError(f"Unknown weight name for layer {li}: {weight_name}")
        else:
            raise ValueError(f"Unknown key: {k}")

    return state_dict_new


MODEL_ARGS_MAP = {
    "llama-7b": ModelArgs(
        n_vocab=50304,
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
    "llama-7b": convert_state_dict_for_chinese_llama2,
    "chinese-alpaca-2-1.3b": convert_state_dict_for_chinese_llama2,
    "OLMo-1B": convert_state_dict_for_olmo,
    "chinese-llama-2-1.3b": convert_state_dict_for_chinese_llama2,
}
