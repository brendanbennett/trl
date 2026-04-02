# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl[vllm]",
#     "math_verify",
#     "latex2sympy2_extended",
# ]
# ///

"""
Evaluate a model on the MATH-500 benchmark using greedy decoding and math_verify for answer checking.

Examples:

    # Evaluate a specific checkpoint from a timestamped training run:
    python scripts/eval_math500.py --model_name_or_path output_20260327_143022/checkpoint-1800

    # Evaluate the base model for comparison:
    python scripts/eval_math500.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

    # With custom generation settings:
    python scripts/eval_math500.py --model_name_or_path output_20260327_143022/checkpoint-1800 --max_tokens 4096 --temperature 0.0

    # Evaluate an intermediate checkpoint:
    python scripts/eval_math500.py --model_name_or_path output_20260327_143022/checkpoint-100
"""

from dataclasses import dataclass, field

import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams

from trl.rewards import accuracy_reward


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default="output",
        metadata={"help": "Model name or path to evaluate. Defaults to 'output' (the GRPO training output dir)."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum number of tokens to generate per problem."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature. Use 0.0 for greedy decoding."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs for tensor parallelism."},
    )
    dataset_name: str = field(
        default="HuggingFaceH4/MATH-500",
        metadata={"help": "MATH-500 dataset name on the Hub."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, split="test")

    # Fix tokenizer_config.json if extra_special_tokens was saved as a list (transformers compatibility issue)
    tokenizer_config_path = os.path.join(script_args.model_name_or_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        if isinstance(tokenizer_config.get("extra_special_tokens"), list):
            tokenizer_config["extra_special_tokens"] = {}
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)

    # Build chat-formatted prompts
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    prompts = []
    for example in dataset:
        messages = [{"role": "user", "content": example["problem"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Generate completions
    llm = LLM(model=script_args.model_name_or_path, tensor_parallel_size=script_args.tensor_parallel_size, enforce_eager=True)
    sampling_params = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=script_args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    completions = [[{"role": "assistant", "content": output.outputs[0].text}] for output in outputs]

    # Score with accuracy_reward
    solutions = dataset["answer"]
    rewards = accuracy_reward(completions, solutions)

    # Compute accuracy (skip examples where gold is unparseable, i.e. reward is None)
    valid = [(r, i) for i, r in enumerate(rewards) if r is not None]
    num_correct = sum(r for r, _ in valid)
    num_valid = len(valid)
    num_skipped = len(rewards) - num_valid
    accuracy = num_correct / num_valid * 100 if num_valid > 0 else 0.0

    print(f"\n{'=' * 50}")
    print(f"Model: {script_args.model_name_or_path}")
    print(f"Dataset: {script_args.dataset_name}")
    print(f"Total examples: {len(rewards)}")
    print(f"Evaluated: {num_valid} (skipped {num_skipped} unparseable)")
    print(f"Correct: {int(num_correct)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 50}")
