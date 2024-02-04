from pathlib import Path

from model.model_args import ModelArgs
from model.causal_model import CausalLM


class Llama2ForCausalLM(CausalLM):
    def __init__(self, args: ModelArgs):
        super().__init__(args)


class OlmoModelForCausalLM(CausalLM):
    def __init__(self, args: ModelArgs):
        super().__init__(args)


if __name__ == "__main__":

    from helper import *

    # model_name = "chinese-alpaca-2-1.3b"
    model_name = "OLMo-1B"
    model_fn = Path() / f"checkpoints/{model_name}/pytorch_model.bin"
    model = CausalLM.from_pretrained(
        model_fn,
        MODEL_ARGS_MAP[model_name],
        strict=True,
        convert_state_dict_fun=CONVERT_STATE_DICT_FUN_MAP[model_name],
    )
    with open(f"./temp/{model_name}-converted.txt", "w") as f:
        for k, v in model.state_dict().items():
            f.write(f"{k}: {[*v.shape]}\n")
