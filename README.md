# pytorch-llm-infer

Oh my own pytorch implementation (from scratch) of some famous large language models (LLM) for inference purposes only. Just a project for practice and learning, DO NOT use it in serious applications.

# How to use

- download model file to *checkpoints* directory from huggingface
- modify *pipeline.py* to adapt to your test running
- run `python pipeline.py`


# Features

##  Features Already Added

- [x] rotary position embeddings / partial rotory position embeddings --> ./model/rope.py
- [x] multi-head / multi-query / group-query attention --> ./model/attention.py
- [x] kv cache --> ./model/module.py

## Supported Models
- [x] [llama2](https://huggingface.co/meta-llama)
- [x] [tinyllama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [x] [phi-2](https://huggingface.co/microsoft/phi-2), [phi-1_5](https://huggingface.co/microsoft/phi-1_5)
- [x] [qwen](https://huggingface.co/Qwen)
- [x] [Gemma](https://huggingface.co/google/gemma-2b-it)


## TODO list

- [ ] fancy sampling
    - [ ] top-k
    - [ ] top-p
    - [ ] repetition_penalty
    - [ ] beam search
- [ ] quantization
- [ ] [speculative_decoding](https://github.com/jzhang38/TinyLlama/blob/main/speculative_decoding/instruct_hf_assisted_decoding.py)
- [ ] Linear and Dynamically Scaled RoPE
    - https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/
    - https://arxiv.org/pdf/2309.00071.pdf
- [ ] batch size > 1
- [ ] batch size > 1 and concat to batch_size=1
- [ ] support more models: 
    - [ ] [OLMo](https://huggingface.co/allenai/OLMo-1B)

## Coffee(:coffee:):

Please buy me a [:coffee:](https://ko-fi.com/excitingme) if you find this project useful. 
