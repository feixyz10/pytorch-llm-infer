import torch
import torch.nn.functional as F


class SamplerBase:
    def __init__(self, temperature: float = 1.0, top_k: int = None):
        self.top_k = top_k
        # self.top_p = top_p
        self.temperature = max(temperature, 1e-4)

    @torch.inference_mode()
    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [..., n_vocab] --> [..., n_vocab]
        logits = logits / self.temperature
        if self.top_k is not None:
            k = min(self.top_k, logits.shape[-1])
            v_topk, _ = torch.topk(logits, k=k, dim=-1)
            thresh = v_topk[..., -1].unsqueeze(-1)
            logits = torch.where(logits < thresh, -float("inf"), logits)
        probs = F.softmax(logits, dim=-1)
        return probs

    @torch.inference_mode()
    def sample_from_logits(self, logits: torch.Tensor, keepdim=False) -> torch.Tensor:
        # [..., n_vocab] --> [...] if keepdim else [..., 1], [..., n_vocab]
        probs = self.logits_to_probs(logits)
        idx = self.sample_index(probs, keepdim=keepdim)
        return idx, probs

    @torch.inference_mode()
    def sample_index_from_logits(
        self, logits: torch.Tensor, keepdim=False
    ) -> torch.Tensor:
        # [..., n_vocab] --> [...] if keepdim else [..., 1]
        p = self.logits_to_probs(logits)
        return self.sample_index(p, keepdim=keepdim)

    @torch.inference_mode()
    def sample_index(self, p: torch.Tensor, keepdim=False) -> torch.Tensor:
        #### following two methods are equivalent, crazy!!!
        #### method 1, seems to be a bit quicker
        q = torch.empty_like(p).exponential_(1)
        return torch.argmax(p / q, dim=-1, keepdim=keepdim)
        # ### method 2
        # shape, n = p.shape[:-1], p.shape[-1]
        # p = p.reshape(-1, n)
        # return torch.multinomial(p, num_samples=1).reshape(*shape, 1)


if __name__ == "__main__":
    torch.manual_seed(16)
    x = torch.randn(1, 2, 10)
    sampler = SamplerBase(temperature=1.0, top_k=3)
    x, p = sampler.sample_from_logits(x, last=True)
    print(x)
    print(p)

    # def do_sampleing(p: torch.Tensor):
    #     shape, n = p.shape[:-1], p.shape[-1]
    #     p = p.reshape(-1, n)
    #     return torch.multinomial(p, num_samples=1).reshape(*shape, 1)

    # def do_sampleing2(p: torch.Tensor):
    #     q = torch.empty_like(p).exponential_(1)
    #     return torch.argmax(p / q, dim=-1, keepdim=True)

    # import time

    # x = torch.randn(1, 100, 1000000)
    # x = F.softmax(x, dim=-1)
    # torch.manual_seed(16)
    # start = time.time()
    # for _ in range(1):
    #     y1 = do_sampleing(x)
    # print(time.time() - start)

    # torch.manual_seed(16)
    # start = time.time()
    # for _ in range(1):
    #     y2 = do_sampleing2(x)
    # print(time.time() - start)

    # assert torch.allclose(y1, y2)
