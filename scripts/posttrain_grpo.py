from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

training_args = GRPOConfig(
    output_dir="output",
    report_to="wandb",
    learning_rate=1e-6,
    beta=0.001,
    max_prompt_length=2048,
    max_completion_length=24576,
    num_generations=16,
    temperature=0.6,
    max_steps=1800,
    max_grad_norm=5.0,
    per_device_train_batch_size=1,
    use_vllm=False, # restricted to 1 task on 1 GPU
)

trainer = GRPOTrainer(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()