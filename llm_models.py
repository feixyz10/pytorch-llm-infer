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
        print_model_state_dict,
    )

    model_name = "phi-1_5"  # "phi-2"  # "OLMo-1B"
    model_dir = Path() / f"checkpoints/{model_name}"
    print_model_state_dict(model_dir, f"./temp/{model_name}.txt")
    model = CausalLM.from_pretrained(
        model_dir,
        get_model_args(model_name),
        strict=True,
        convert_state_dict_fun=get_state_dict_convert_fun(model_name),
    )
    with open(f"./temp/{model_name}-converted.txt", "w") as f:
        for k, v in model.state_dict().items():
            f.write(f"{k}: {[*v.shape]}\n")
