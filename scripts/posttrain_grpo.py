from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
    args=GRPOConfig(output_dir="output", report_to="wandb"),
)
trainer.train()