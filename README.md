# Continuous Depth Batching (CDB) for Looped Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2608.09444-b31b1b.svg)](https://arxiv.org/abs/2608.09444)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://kschwethelm.github.io/continuous-depth-batching/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains our [paper's](https://arxiv.org/abs/2608.09444) implementation of Continuous Depth Batching (CDB), a method for efficient inference with depth-adaptive looped language models.
It includes the core inference code, workload and evaluation utilities, and scripts for reproducing the experimental results.

> [!WARNING]
> The backend code will soon be replaced by our much faster implementation at [loop-sglang](https://github.com/kschwethelm/loop-sglang).

## Scope

The repository provides a continuous batching baseline and continuous depth batching for Ouro ([1.4B](https://huggingface.co/KristianS7/Ouro-1.4B), [2.6B](https://huggingface.co/KristianS7/Ouro-2.6B)) and [Huginn](https://huggingface.co/tomg-group-umd/huginn-0125), including scheduling with and without refill.
The engines currently support only greedy decoding.
The implementation adapts [Hugging Face Transformers' continuous batching](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/continuous_batching/continuous_api.py) and the Ouro and Huginn model releases, with selected designs informed by [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) and [SGLang](https://github.com/sgl-project/sglang).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and license details.

## Setup

1. **Install [uv](https://docs.astral.sh/uv/).**

2. **Clone or extract this repository and enter its root directory.**

3. **Create the machine and cluster configuration:**

   ```bash
   cp shells/_machine_config.sh.template shells/_machine_config.sh
   ```

   Edit `shells/_machine_config.sh` to configure the repository path, cache directories, SLURM settings, GPU type, and FlashAttention backend.

4. **Install the dependencies:**

   ```bash
   uv sync --frozen
   ```

   When using FlashAttention 2, run:

   ```bash
   uv sync --frozen --extra fa2
   ```

## SLURM Job Submission

All experiment jobs are submitted through `shells/_submit.sh`, which reads the machine configuration `shells/_machine_config.sh` and selects the appropriate SLURM resources.
For example, run the offline throughput sweep for Ouro on the ShareGPT workload with:

```bash
MODEL=ouro DATASET=sharegpt ./shells/_submit.sh \
  shells/paper/throughput_exit-sweep.sh -- --time=24:00:00
```

The workload must be generated before submitting the throughput sweep.
Arguments before `--` are passed to the experiment script, while arguments after `--` are passed to `sbatch`.
Scripts ending in `_cpu.sh` are submitted to the configured CPU partition, and all other scripts are submitted to the configured GPU partition.
Job logs are written to `logs/<script-name>/`.

## Reproducing the results

The shell scripts under [`shells/paper/`](shells/paper/) reproduce all experiments reported in the paper.
See [`shells/paper/README.md`](shells/paper/README.md) for details.

## Minimal CDB example

The following example loads Huginn with the shared KV cache and early-exit lookahead gate:

```python
from datasets import load_dataset

from looped_cdb.continuous_depth_batching import (
    ContinuousDepthBatchingConfig,
    ContinuousDepthBatchingEngine,
)
from looped_cdb.eval.model_loading import load_huginn_model, load_huginn_tokenizer

model_id = "tomg-group-umd/huginn-0125"
model = load_huginn_model(model_id)
tokenizer = load_huginn_tokenizer(model_id)

dataset = load_dataset("openai/gsm8k", "main", split="test[:8]")
prompts = [f"Q: {row['question']}\nA:" for row in dataset]

engine = ContinuousDepthBatchingEngine.from_model(
    model,
    ContinuousDepthBatchingConfig(
        max_recurrent_steps=16,
        exit_threshold=0.28,
        kv_policy="single",
    ),
)
outputs = engine.generate_batch(
    tokenizer(prompts, add_special_tokens=False)["input_ids"],
    max_new_tokens=128,
    eos_token_id=tokenizer.eos_token_id,
)
```

## Cite

If you find our code helpful, please cite our paper:

```bibtex
@misc{schwethelm2026cdb,
      title={Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching},
      author={Kristian Schwethelm and Daniel Rueckert and Georgios Kaissis},
      year={2026},
      eprint={2608.09444},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.09444},
}
```
