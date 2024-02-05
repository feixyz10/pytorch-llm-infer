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

    from helper import (
        get_model_args,
        get_state_dict_convert_fun,
        get_model_state_dict_filenames,
    )

    model_name = "OLMo-1B"
    model_dir = Path() / f"checkpoints/{model_name}"
    model_fn = get_model_state_dict_filenames(model_dir)
    model = CausalLM.from_pretrained(
        model_fn,
        get_model_args(model_name),
        strict=True,
        convert_state_dict_fun=get_state_dict_convert_fun(model_name),
    )
    with open(f"./temp/{model_name}-converted.txt", "w") as f:
        for k, v in model.state_dict().items():
            f.write(f"{k}: {[*v.shape]}\n")
