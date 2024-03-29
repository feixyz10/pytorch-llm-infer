import torch
import torch.nn as nn

import os
import platform
from pathlib import Path
from pydantic import BaseModel
from typing import List, Tuple

from model.causal_model import CausalLM

from transformers import AutoTokenizer
from model.helper import (
    get_model_args,
    get_state_dict_convert_fun,
    get_prompt_preprocess_fun,
    get_responce_postprocess_fun,
)
from model.samplers import SamplerBase
from model.decode import generate_yield


class GenerationConfig(BaseModel):
    max_prompt_length: int = 512  # max length of prompt, truncated if longer
    max_length: int = 100
    do_sample: bool = True  # sampling next token if True, otherwise greedy search
    temperature: float = 1.0
    top_k: int = 3
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    speculative_k: int = 5
    verbose: bool = True


class Pipeline(nn.Module):
    def __init__(
        self,
        model: CausalLM,
        tokenizer: AutoTokenizer,
        model_name: str = None,
        draft_model: CausalLM = None,
        draft_model_name: str = None,
    ):
        super().__init__()
        self.model: CausalLM = model
        self.tokenizer: AutoTokenizer = tokenizer
        self.model_name: str = model_name
        self.draft_model: CausalLM = draft_model
        self.draft_model_name: str = draft_model_name

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

    @torch.inference_mode()
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
            out_all = self.tokenizer.decode(
                [*input_token_ids_raw, *output_token_ids], skip_special_tokens=True
            )
            out = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
            return self._postprocess(out_all), self._postprocess(out)

        self.model = self.model.to(device).eval().reset()
        if self.draft_model is not None:
            self.draft_model = self.draft_model.to(device).eval().reset()
        sampler = SamplerBase(config.temperature, config.top_k)
        for out in generate_yield(
            self.model,
            input_token_ids,
            sampler,
            config.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            draft_model=self.draft_model,
            speculative_k=config.speculative_k,
            verbose=config.verbose,
        ):
            yield construct_output(out)

    @torch.inference_mode()
    def _generate_next_token(
        self, token_ids: torch.Tensor, config: GenerationConfig
    ) -> int:
        logits = self.model(token_ids)[:, -1, :]
        next_token_id = self._sample(logits, config)
        return next_token_id

    @staticmethod
    def from_pretrained(
        model_dir: Path,
        draft_model_dir: Path = None,
        strict=True,
    ) -> "Pipeline":
        assert model_dir.is_dir()
        assert draft_model_dir is None or draft_model_dir.is_dir()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )

        def _load_model(model_dir: Path):
            if model_dir is None:
                return None, None
            model_name = model_dir.name
            model_args = get_model_args(model_name)
            if model_args.n_vocab != len(tokenizer):
                print(
                    f"WARNING: {model_name}: model_args.n_vocab ({model_args.n_vocab}) != len(tokenizer) "
                    f"({len(tokenizer)})"
                )
            model = CausalLM.from_pretrained(
                model_dir,
                model_args,
                strict,
                get_state_dict_convert_fun(model_name),
            )
            model.eval()
            return model, model_name

        model, model_name = _load_model(model_dir)
        draft_model, draft_model_name = _load_model(draft_model_dir)

        return Pipeline(model, tokenizer, model_name, draft_model, draft_model_name)

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
    from model.helper import get_device

    gen_config = GenerationConfig(
        max_prompt_length=1024,
        max_length=2048,
        do_sample=True,
        temperature=1.0,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.0,
        verbose=True,
    )

    device = get_device("auto")
    model_names = [
        "phi-1_5",
        "phi-2",
        "TinyLlama-1.1B-Chat-v1.0",
        "Qwen1.5-0.5B-Chat",
        "Qwen1.5-1.8B-Chat",
        "gemma-2b-it",
    ]
    model_name = model_names[3]
    draft_model_name = model_names[3] if 1 == 2 else None
    print(f"model name: {model_name}, draft model name: {draft_model_name}")
    mode = "normal decoding" if draft_model_name is None else "speculative decoding"
    print(f"{mode}")
    print("device:", device.type)
    model_dir: Path = Path() / f"checkpoints/{model_name}"
    draft_model_dir = (
        model_dir.parent / draft_model_name if draft_model_name is not None else None
    )
    pipeline = Pipeline.from_pretrained(model_dir, draft_model_dir)
    prompts = [
        'def print_prime(n):\n    """\n    print all primes between 1 and n\n    """\n',
        "Instruct: Write a detailed analogy between mathematics and a lighthouse.\n\nOutput:",
        "Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?\n\nBob: ",
        "I don't know why, I'm struggling to maintain focus while studying. Any suggestions?",
        "Please tell me a joke about Python.",
    ]
    prompt = prompts[-2]
    # output = pipeline.stream_generate(
    #     prompt, config=gen_config, device=device, flush_every=1
    # )
    # output = pipeline.generate(prompt, config=gen_config, device=device)

    history = None
    output, history = pipeline.chat(
        prompt, history=history, config=gen_config, device=device
    )
    output, history = pipeline.chat(
        "Very nice answer, thank you!",
        history=history,
        config=gen_config,
        device=device,
    )
