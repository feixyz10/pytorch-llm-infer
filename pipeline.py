import torch
import torch.nn as nn

from pathlib import Path
from pydantic import BaseModel

from model.causal_model import CausalLM
from sentencepiece import SentencePieceProcessor
from helper import MODEL_ARGS_MAP, CONVERT_STATE_DICT_FUN_MAP


class GenerationConfig(BaseModel):
    max_prompt_length: int = 512  # max length of prompt, truncated if longer
    do_sample: bool = True  # sampling next token if True, otherwise greedy search
    temperature: float = 1.0
    top_k: int = 3
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    verbose: bool = True


class Pipeline(nn.Module):
    def __init__(self, model: CausalLM, tokenizer: SentencePieceProcessor):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self, prompts: str, config: GenerationConfig = None, device: torch.device = None
    ) -> torch.Tensor:
        """Now only support batch_size == 1"""
        if config is None:
            config = GenerationConfig()

        token_ids = self.tokenizer.Encode(prompts)
        if len(token_ids) > config.max_prompt_length:
            token_ids = token_ids[-config.max_prompt_length :]
        token_ids = torch.tensor(token_ids, dtype=torch.int64).to(device)

        self.model.to(device)
        self.model.eval()
        self.model.reset()
        outputs = "assistant: "
        print(f"user: {prompts}", end="\n")
        with torch.inference_mode():
            next_token_id = self._generate_next(token_ids, config)
            while next_token_id != self.tokenizer.eos_id():
                outputs += self.tokenizer.Decode([next_token_id])
                print(outputs, end="\r")
                token_ids = torch.tensor([next_token_id], dtype=torch.int64).to(device)
                next_token_id = self._generate_next(token_ids, config)

        return outputs

    def _generate_next(self, token_ids: torch.Tensor, config: GenerationConfig) -> int:
        with torch.inference_mode():
            logits = self.model(token_ids)[:, -1, :]
            next_token_id = self._sample(logits, config)
            return next_token_id

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

    @staticmethod
    def from_pretrained(
        model_fn: Path,
        tokenizer_fn: Path,
        model_name: str = None,
        strict=True,
    ) -> "Pipeline":
        assert model_fn.suffix in [".pth", ".bin", ".safetensors"]
        assert tokenizer_fn.suffix == ".model"

        if model_name is None:
            model_name = model_fn.parent.name

        tokenizer = SentencePieceProcessor()
        tokenizer.Load(str(tokenizer_fn))
        assert MODEL_ARGS_MAP[model_name].n_vocab == len(tokenizer)
        model = CausalLM.from_pretrained(
            model_fn,
            MODEL_ARGS_MAP[model_name],
            strict,
            CONVERT_STATE_DICT_FUN_MAP[model_name],
        )
        model.eval()

        return Pipeline(model, tokenizer)


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
    # model_name = "chinese-alpaca-2-1.3b"
    model_name = "chinese-llama-2-1.3b"
    model_fn = Path() / f"checkpoints/{model_name}/pytorch_model.bin"
    tokenizer_fn = Path() / f"checkpoints/{model_name}/tokenizer.model"
    pipeline = Pipeline.from_pretrained(model_fn, tokenizer_fn)
    prompts = [
        "很久很久以前有一个小男孩，他",
        "Long long ago, there was a little boy, he",
    ]
    output = pipeline.generate(prompts[0], gen_config, device=device)
    print(output)
