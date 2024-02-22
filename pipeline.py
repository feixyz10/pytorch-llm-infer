import torch
import torch.nn as nn

import os
import platform
from pathlib import Path
from pydantic import BaseModel
from typing import List, Tuple

from model.causal_model import CausalLM

from transformers import AutoTokenizer
from helper import (
    get_model_args,
    get_state_dict_convert_fun,
    get_prompt_preprocess_fun,
    get_responce_postprocess_fun,
)


class GenerationConfig(BaseModel):
    max_prompt_length: int = 512  # max length of prompt, truncated if longer
    max_length: int = 100
    do_sample: bool = True  # sampling next token if True, otherwise greedy search
    temperature: float = 1.0
    top_k: int = 3
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    verbose: bool = True


class Pipeline(nn.Module):
    def __init__(
        self, model: CausalLM, tokenizer: AutoTokenizer, model_name: str = None
    ):
        super().__init__()
        self.model: CausalLM = model
        self.tokenizer: AutoTokenizer = tokenizer
        self.model_name: str = model_name

    def chat(
        self,
        prompt: str,
        history: List[Tuple[str, str]] = None,
        config: GenerationConfig = None,
        device: torch.device = None,
        flush_every: int = 1,
        **kwargs,
    ):
        if config is None:
            config = GenerationConfig()
        if history is None:
            history = []

        count, clear_cmd = 0, get_clear_command()
        for out, response in self._generate(
            prompt, history=history, config=config, device=device
        ):
            count += 1
            if count % flush_every == 0:
                os.system(clear_cmd)
                print(out, flush=True)
        os.system(clear_cmd)
        print(out, flush=True)
        return out, history + [(prompt, response)]

    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None,
        device: torch.device = None,
        **kwargs,
    ) -> str:
        """Now only support batch_size == 1"""
        for response, _ in self._generate(prompt, config=config, device=device):
            pass
        print(response)
        return response

    def stream_generate(
        self,
        prompt: str,
        config: GenerationConfig = None,
        device: torch.device = None,
        flush_every: int = 1,
        **kwargs,
    ) -> str:
        count, clear_cmd = 0, get_clear_command()
        for out, _ in self._generate(prompt, config=config, device=device):
            count += 1
            if count % flush_every == 0:
                os.system(clear_cmd)
                print(out, flush=True)
        os.system(clear_cmd)
        print(out, flush=True)
        return out

    def _generate(
        self,
        prompt: str,
        history: List[Tuple[str, str]] = None,
        config: GenerationConfig = None,
        device: torch.device = None,
    ):
        if config is None:
            config = GenerationConfig()
        prompt = self._preprocess(prompt, history=history)

        input_token_ids_raw = self.tokenizer.encode(prompt)
        input_token_ids = input_token_ids_raw
        if len(input_token_ids_raw) > config.max_prompt_length:
            input_token_ids = input_token_ids[-config.max_prompt_length :]
        input_token_ids = torch.tensor(input_token_ids, dtype=torch.int64).to(device)

        def construct_output(output_token_ids: List[int]):
            out_all = self.tokenizer.decode([*input_token_ids_raw, *output_token_ids])
            out = self.tokenizer.decode(output_token_ids)
            return self._postprocess(out_all), self._postprocess(out)

        self.model = self.model.to(device).eval().reset()
        output_token_ids = []
        with torch.inference_mode():
            next_token_id = self._generate_next_token(input_token_ids, config)
            output_token_ids.append(next_token_id)
            while True:
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                if len(input_token_ids) + len(output_token_ids) > config.max_length:
                    break
                token_id = torch.tensor([next_token_id], dtype=torch.int64).to(device)
                next_token_id = self._generate_next_token(token_id, config)
                output_token_ids.append(next_token_id)
                yield construct_output(output_token_ids)
            yield construct_output(output_token_ids)

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

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        model_args = get_model_args(model_name)
        if model_args.n_vocab != len(tokenizer):
            print(
                f"WARNING: model_args.n_vocab ({model_args.n_vocab}) != len(tokenizer) "
                f"({len(tokenizer)})"
            )

        model = CausalLM.from_pretrained(
            model_dir,
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

    def _preprocess(self, prompt: str, history: List[Tuple[str, str]] = None):
        return get_prompt_preprocess_fun(self.model_name)(prompt, history)

    def _postprocess(self, response: str):
        return get_responce_postprocess_fun(self.model_name)(response)


def get_clear_command():
    os_name = platform.system()
    clear_command = "cls" if os_name == "Windows" else "clear"
    return clear_command


if __name__ == "__main__":
    from helper import get_device

    gen_config = GenerationConfig(
        max_prompt_length=50,
        max_length=70,
        do_sample=True,
        temperature=1.0,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.0,
        verbose=True,
    )

    device = get_device("cpu")
    model_names = [
        "phi-1_5",
        "phi-2",
        "TinyLlama-1.1B-Chat-v1.0",
        "Qwen1.5-0.5B-Chat",
        "Qwen1.5-1.8B-Chat",
        "gemma-2b-it",
    ]
    model_name = model_names[3]
    print("model name:", model_name)
    print("device:", device.type)
    model_dir = Path() / f"checkpoints/{model_name}"
    pipeline = Pipeline.from_pretrained(model_dir)
    prompts = [
        "# print all primes between 1 and n\ndef print_prime(n):\n",
        "Write a detailed analogy between mathematics and a lighthouse.\nAnswer: ",
        "Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?\nBob: ",
        "I don't know why, I'm struggling to maintain focus while studying. Any suggestions?",
    ]
    prompt = prompts[3]
    # output = pipeline.stream_generate(
    #     prompt, config=gen_config, device=device, flush_every=1
    # )
    # output = pipeline.generate(prompt, config=gen_config, device=device)

    history = None
    output, history = pipeline.chat(
        prompt, history=history, config=gen_config, device=device
    )
    output, history = pipeline.chat(
        "Great suggestions, thank you!",
        history=history,
        config=gen_config,
        device=device,
    )
