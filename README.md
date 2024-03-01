# pytorch-llm-infer

Oh my own pytorch implementation (from scratch) of some famous large language models (LLM) for inference purposes only. Just a project for practice and learning, DO NOT use it in serious applications.

# How to use

- download model file to *checkpoints* directory from huggingface
- modify *pipeline.py* to adapt to your test running
- run `python pipeline.py`

# Results

**Gemma**
![](./imgs/gemma-2b-it.png)

**Qwen**
![](./imgs/qwen-1_8b-chat.png)

# Features

##  Features Already Added

- [x] rotary position embeddings / partial rotory position embeddings --> ./model/rope.py
- [x] multi-head / multi-query / group-query attention --> ./model/attention.py
- [x] kv cache --> ./model/module.py
- parallel sampling
    - [x] [speculative_decoding](https://github.com/jzhang38/TinyLlama/blob/main/speculative_decoding/instruct_hf_assisted_decoding.py)

## Supported Models

- [x] [tinyllama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [x] [phi-2](https://huggingface.co/microsoft/phi-2), [phi-1_5](https://huggingface.co/microsoft/phi-1_5)
- [x] [qwen](https://huggingface.co/Qwen)
- [x] [Gemma](https://huggingface.co/google/gemma-2b-it) *20230222*
- [x] [llama2](https://huggingface.co/meta-llama)


## TODO list

- fancy sampling
    - [ ] top-k
    - [ ] top-p
    - [ ] repetition_penalty
    - [ ] beam search
- parallel sampling
    - [ ] [speculative_decoding](https://github.com/jzhang38/TinyLlama/blob/main/speculative_decoding/instruct_hf_assisted_decoding.py)
    - [ ] [Jacobi decoding]()
    - [ ] [Look ahead decoding]()
- [ ] quantization
- [ ] context window extension 
- [ ] continuous batching
- support more models: 
    - [ ] [OLMo](https://huggingface.co/allenai/OLMo-1B)
- [ ] Streamer like huggingface

## Coffee

Please buy me a [:coffee:](https://ko-fi.com/excitingme) if you find this project useful. 
