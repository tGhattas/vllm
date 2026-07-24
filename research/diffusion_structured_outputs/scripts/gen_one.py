# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal per-request structured-outputs example for DiffusionGemma.

Runs the real NVFP4 model with a regex structured-outputs spec passed through the
normal SamplingParams API (no precompiled constraint file). The DiffusionSampler
compiles the token-DFA per request and steers every denoising step to a
regex-accepted canvas. Requires a Blackwell GPU for the NVFP4 checkpoint.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python gen_one.py
"""

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams as SO

M = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
llm = LLM(
    model=M,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.7,
    max_model_len=1024,
    enforce_eager=True,
)
sp = SamplingParams(
    temperature=1.0,
    max_tokens=24,
    structured_outputs=SO(
        regex=r"\{\"active\":(true|false),\"age\":-?(0|[1-9][0-9]*)\}"
    ),
)
o = llm.generate(["Give me a JSON object describing a person. "], sp)
print(f"### PER-REQUEST REGEX OUTPUT: {o[0].outputs[0].text!r}")
