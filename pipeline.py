import torch
import torch.nn as nn

from typing import List
from pathlib import Path

from arg import GenerationConfig
from llama2 import Llama2ForCausalLM
from sentencepiece import SentencePieceProcessor


class Pipeline(nn.Module):
    def __init__(self, model: Llama2ForCausalLM, tokenizer: SentencePieceProcessor):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self, prompts: List[str], config: GenerationConfig = None
    ) -> torch.Tensor:
        pass

    @staticmethod
    def from_pretrained(
        model_fn: Path, tokenizer_fn: Path, model_args_fn: Path
    ) -> "Pipeline":
        assert model_fn.suffix == ".pth"
        assert tokenizer_fn.suffix == ".model"
        assert model_args_fn.suffix == ".json"

        model = Llama2ForCausalLM.from_pretrained(model_fn, model_args_fn)
        tokenizer = SentencePieceProcessor().Load(tokenizer_fn)

        return Pipeline(model, tokenizer)


if __name__ == "__main__":
    pass
