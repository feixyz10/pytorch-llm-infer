from ast import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from typing import Tuple
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from causal_model import CausalLM
from samplers import SamplerBase


@torch.inference_mode()
def prefill(
    model: CausalLM,
    x: torch.Tensor,
) -> torch.Tensor:
    model = model.reset()
    logits = model(x)
    return logits


@torch.inference_mode()
def autoregressive_decode(
    model: CausalLM,
    curr_token: torch.Tensor,  # [B, 1], currently only support B == 1
    num_new_tokens: int,
    sampler: SamplerBase,
    eos_token_id: int = None,
    return_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore

    def get_return(ts, ps):
        if return_probs:
            return (
                torch.cat(ts, dim=-1),
                torch.cat(ps, dim=-2),
            )
        else:
            return torch.cat(ts, dim=-1)

    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens):
        logits = model(curr_token)
        if return_probs:
            new_token, new_prob = sampler.sample_from_logits(logits[:, -1])
            new_tokens.append(new_token)
            new_probs.append(new_prob)
        else:
            new_token = sampler.sample_index_from_logits(logits[:, -1])
            new_tokens.append(new_token)
        if eos_token_id is not None and new_token.item() == eos_token_id:
            break
        curr_token = new_token
    return get_return(new_tokens, new_probs)


@torch.inference_mode()
def autoregressive_decode_yield(
    model: CausalLM,
    curr_token: torch.Tensor,  # [B, 1], currently only support B == 1
    num_new_tokens: int,
    sampler: SamplerBase,
    eos_token_id: int = None,
    return_probs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore

    def get_return(ts, ps):
        if return_probs:
            return (
                torch.cat(ts, dim=-1),
                torch.cat(ps, dim=-2),
            )
        else:
            return torch.cat(ts, dim=-1)

    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens):
        logits = model(curr_token)
        if return_probs:
            new_token, new_prob = sampler.sample_from_logits(logits[:, -1])
            new_tokens.append(new_token)
            new_probs.append(new_prob)
        else:
            new_token = sampler.sample_index_from_logits(logits[:, -1])
            new_tokens.append(new_token)
        if eos_token_id is not None and new_token.item() == eos_token_id:
            break
        curr_token = new_token
        yield get_return(new_tokens, new_probs)
    yield get_return(new_tokens, new_probs)


@torch.inference_mode()
def speculative_decode_yield(
    model: CausalLM,
    curr_token: torch.Tensor,  # [B, 1] or [1], currently only support B == 1
    sampler: SamplerBase,
    draft_model: CausalLM,
    speculative_k: int = 5,
) -> torch.Tensor:  # type: ignore
    curr_token = curr_token.view(-1)
    tokens_draft, probs_draft = autoregressive_decode(
        draft_model, curr_token, speculative_k, sampler, return_probs=True
    )
    tokens_to_verify = torch.cat([curr_token, tokens_draft], dim=-1)
    logits = model(tokens_to_verify)  # [B, K+1, n_vocab]
    probs = sampler.logits_to_probs(logits)[0]  # [K+1, n_vocab]
    # p: prob for target model; q: prob for draft model
    p = probs[:-1].gather(1, tokens_draft.unsqueeze(-1)).squeeze(-1)
    q = probs_draft.gather(1, tokens_draft.unsqueeze(-1)).squeeze(-1)
    accept_prob = torch.min(torch.ones_like(p), p / q)
    reject_loc = torch.nonzero(torch.rand_like(accept_prob) > accept_prob)
    if reject_loc.numel() == 0:  # all draft tokens will be accepted
        last_token = sampler.sample_index_from_logits(probs[-1], keepdim=True)
        draft_model(tokens_draft[-1])
        yield torch.cat([tokens_draft, last_token], dim=-1)
    else:
        accept_n = reject_loc[0].item()
        p, q = probs[accept_n], probs_draft[accept_n]
        p_prime = torch.clamp(p - q, 0)
        p_prime = p_prime / p_prime.sum()
        new_tokens = sampler.sample_index(p_prime, keepdim=True)
        draft_model.set_start_index(
            draft_model.start_index - speculative_k + accept_n + 1
        )
        model.set_start_index(model.start_index - speculative_k + accept_n)
        yield torch.cat([tokens_draft[:accept_n], new_tokens], dim=-1)


@torch.inference_mode()
def speculative_decode(
    model: CausalLM,
    curr_token: torch.Tensor,  # [B, 1] or [1], currently only support B == 1
    sampler: SamplerBase,
    draft_model: CausalLM,
    speculative_k: int = 5,
) -> torch.Tensor:
    for tokens in speculative_decode_yield(
        model, curr_token, sampler, draft_model, speculative_k
    ):
        pass
    return tokens


@torch.inference_mode()
def generate_yield(
    model: CausalLM,
    prompt: torch.Tensor,  # [B, L], currently only support B == 1
    sampler: SamplerBase,
    max_length: int,
    eos_token_id: int,
    draft_model: CausalLM = None,
    speculative_k: int = 5,
    verbose: bool = False,
):
    assert prompt.ndim <= 1 or (prompt.ndim == 2 and prompt.shape[0] == 1)
    L = prompt.shape[-1] if prompt.ndim >= 1 else 1
    outputs = []
    do_speculative = draft_model is not None
    logits = prefill(model, prompt)
    next_token = sampler.sample_index_from_logits(logits[:, -1])
    outputs.append(next_token.item())
    total_seq_len = L + len(outputs)
    eos_reached = False
    if not do_speculative:
        if verbose:
            print("normal decoding mode", flush=True)
        for next_tokens in autoregressive_decode_yield(
            model, next_token, max_length - total_seq_len, sampler, eos_token_id
        ):
            yield [*outputs, *[x.item() for x in next_tokens]]
    else:
        if verbose:
            print("speculative decoding mode", flush=True)
        prefill(draft_model, prompt)
        while total_seq_len < max_length - speculative_k and not eos_reached:
            seq_len_left = max_length - total_seq_len
            next_tokens = speculative_decode(
                model,
                next_token,
                sampler,
                draft_model,
                min(speculative_k, seq_len_left - 1),
            )
            for x in [x.item() for x in next_tokens]:
                outputs.append(x)
                if x == eos_token_id:
                    eos_reached = True
                    break
            total_seq_len = L + len(outputs)
            next_token = next_tokens[-1]
            yield outputs
        if not eos_reached:
            for next_tokens in autoregressive_decode_yield(
                model, next_token, max_length - total_seq_len, sampler, eos_token_id
            ):
                yield [*outputs, *[x.item() for x in next_tokens]]


@torch.inference_mode()
def generate(
    model: CausalLM,
    prompt: torch.Tensor,  # [B, L], currently only support B == 1
    sampler: SamplerBase,
    max_length: int,
    eos_token_id: int,
    draft_model: CausalLM = None,
    speculative_k: int = 5,
    verbose: bool = False,
):
    for tokens in generate_yield(
        model,
        prompt,
        sampler,
        max_length,
        eos_token_id,
        draft_model,
        speculative_k,
        verbose,
    ):
        pass
    return tokens


if __name__ == "__main__":

    import copy

    class DummyModel(nn.Module):
        def __init__(self, n_vocab: int = 100, max_seq_len=100):
            super().__init__()
            self.start_index = 0
            self.n_vocab = n_vocab
            self.V = torch.randn(max_seq_len, self.n_vocab) * 10

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            assert tokens.ndim <= 2
            while tokens.ndim < 2:
                tokens = tokens.unsqueeze(0)
            assert tokens.shape[0] == 1
            _, L = tokens.shape
            self.start_index = self.start_index + L
            # return torch.rand(*tokens.shape, self.n_vocab)
            return self.V[tokens.view(-1)].reshape(*tokens.shape, self.n_vocab)

        def reset(self):
            self.start_index = 0
            return self

        def set_start_index(self, start_index: int):
            self.start_index = start_index

    torch.manual_seed(16)
    model = DummyModel()
    # draft_model = copy.deepcopy(model)
    draft_model = DummyModel()
    sampler = SamplerBase(temperature=1.0, top_k=10)

    # prefill(model, torch.LongTensor([1, 2]))
    # prefill(draft_model, torch.LongTensor([1, 2]))
    # speculative_decode(model, torch.LongTensor([1]), sampler, draft_model, 3)
    prompt = torch.LongTensor([1, 2])
    outputs1 = generate(model, prompt, sampler, 10, 38, draft_model, 3)
    outputs2 = generate(model, prompt, sampler, 10, 38, None, 3)
    print(outputs1)
    print(outputs2)
