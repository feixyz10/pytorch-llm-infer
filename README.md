# How to use

- download model file to *checkpoints* directory from huggingface
- modify *pipelines.py* to adapt to your test running
- run `python pipelines.py`


# Features
##  Future Already Added

- [x] rotary position embeddings / partial rotory position embeddings
- [x] multi-head / multi-query / group-query attention
- [x] kv cache
- [x] supported models: llama2 / tinyllama / phi / qwen

## TODO list
- [ ] speculative_decoding
    - https://github.com/jzhang38/TinyLlama/blob/main/speculative_decoding/instruct_hf_assisted_decoding.py
- [ ] quantization
- [ ] Linear and Dynamically Scaled RoPE
    - https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/
    - https://arxiv.org/pdf/2309.00071.pdf
- [ ] batch size > 1
- [ ] batch size > 1 and convert to batch_size=1
- [ ] support models: OLMo, etc.