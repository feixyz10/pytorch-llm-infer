import torch
import torch.nn as nn

from pathlib import Path
from pydantic import BaseModel
from typing import List

from model.causal_model import CausalLM

from transformers import AutoTokenizer
from helper import get_model_args, get_state_dict_convert_fun, get_prompt_preprocess_fun


class GenerationConfig(BaseModel):
    max_prompt_length: int = 512  # max length of prompt, truncated if longer
    max_length: int = 256
    do_sample: bool = True  # sampling next token if True, otherwise greedy search
    temperature: float = 1.0
    top_k: int = 3
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    verbose: bool = True
    stream: bool = True


class Pipeline(nn.Module):
    def __init__(
        self, model: CausalLM, tokenizer: AutoTokenizer, model_name: str = None
    ):
        super().__init__()
        self.model: CausalLM = model
        self.tokenizer: AutoTokenizer = tokenizer
        self.model_name: str = model_name

    def generate(
        self,
        prompts: str,
        config: GenerationConfig = None,
        device: torch.device = None,
    ) -> str:
        """Now only support batch_size == 1"""
        if config is None:
            config = GenerationConfig()

        if get_prompt_preprocess_fun(self.model_name) is not None:
            prompts = get_prompt_preprocess_fun(self.model_name)(prompts)
        input_token_ids = self.tokenizer.encode(prompts)
        if len(input_token_ids) > config.max_prompt_length:
            input_token_ids = input_token_ids[-config.max_prompt_length :]
        input_token_len = len(input_token_ids)
        input_token_ids = torch.tensor(input_token_ids, dtype=torch.int64).to(device)

        self.model.to(device)
        self.model.eval()
        self.model.reset()
        if config.verbose:
            print(f"{prompts}", end="")
        output_token_ids = []
        outputs = ""
        with torch.inference_mode():
            next_token_id = self._generate_next_token(input_token_ids, config)
            output_token_ids.append(next_token_id)
            while True:
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                outputs = self.tokenizer.decode(output_token_ids)
                if config.verbose and config.stream:
                    print(self.tokenizer.decode(output_token_ids), end="\r")
                if input_token_len + len(output_token_ids) >= config.max_length:
                    break
                token_id = torch.tensor([next_token_id], dtype=torch.int64).to(device)
                next_token_id = self._generate_next_token(token_id, config)
                output_token_ids.append(next_token_id)
        if config.verbose and config.stream:
            print("")
        return outputs

    def _generate_next_token(
        self, token_ids: torch.Tensor, config: GenerationConfig
    ) -> int:
        with torch.inference_mode():
            logits = self.model(token_ids)[:, -1, :]
            next_token_id = self._sample(logits, config)
            return next_token_id

    @staticmethod
    def from_pretrained(
        model_dir: Path,
        strict=True,
    ) -> "Pipeline":
        assert model_dir.is_dir()
        model_name = model_dir.name
        model_fn = get_model_state_dict_filenames(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        model_args = get_model_args(model_name)
        assert model_args.n_vocab == len(tokenizer)
        model = CausalLM.from_pretrained(
            model_fn,
            model_args,
            strict,
            get_state_dict_convert_fun(model_name),
        )
        model.eval()
        return Pipeline(model, tokenizer, model_name)

    def _sample(self, logits: torch.Tensor, config: GenerationConfig) -> int:
        """Only support batch_size == 1
        TODO: top_p sampling
        """
        logits = logits[0]

        if config.do_sample == False or config.temperature <= 0.01 or config.top_k <= 1:
            return logits.argmax(dim=-1).item()

        logits = logits / config.temperature
        probs = nn.functional.softmax(logits, dim=-1)
        probs, indices = torch.topk(probs, config.top_k)
        probs = probs / probs.sum()

        return indices.tolist()[torch.multinomial(probs, num_samples=1).item()]


if __name__ == "__main__":
    from helper import get_device

    gen_config = GenerationConfig(
        max_prompt_length=512,
        do_sample=True,
        temperature=1.0,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.0,
        verbose=True,
    )

    device = get_device()
    model_name = "TinyLlama-1.1B-Chat-v1.0"
    # model_name = "OLMo-1B"
    model_dir = Path() / f"checkpoints/{model_name}"
    tokenizer_fn = Path() / f"checkpoints/{model_name}/tokenizer.model"
    pipeline = Pipeline.from_pretrained(model_dir)
    prompts = [
        "Write me a function to calculate the first 10 digits of the fibonacci sequence in Python and print it out to the CLI.",
        "hello",
    ]
    output = pipeline.generate(prompts[0], gen_config, device=device)
