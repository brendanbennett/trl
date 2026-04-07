from dataclasses import dataclass, field
from datetime import datetime

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        metadata={"help": "Model name or path to train from."},
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Output directory. Defaults to 'output_YYYYMMDD_HHMMSS'."},
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per device during training."},
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of gradient accumulation steps."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.output_dir is None:
        script_args.output_dir = f"output/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    training_args = GRPOConfig(
        output_dir=script_args.output_dir,
        save_strategy="steps",
        save_steps=100,
        report_to="wandb",
        learning_rate=1e-6,
        beta=0.001,
        max_completion_length=4096,
        num_generations=8,
        temperature=0.6,
        max_steps=1800,
        max_grad_norm=5.0,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        use_vllm=False,  # restricted to 1 task on 1 GPU
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    trainer = GRPOTrainer(
        model=script_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()
