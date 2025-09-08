import os

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
import unsloth  # noqa: F401  # isort: skip
from unsloth import FastLanguageModel  # noqa: F401  # isort: skip
import logging

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from rich.logging import RichHandler
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from train.reward import attack_reward
from utils.tool import to_trl_format

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

dataset = load_dataset("parquet", data_files="./data/defense.parquet", split="train")

assert isinstance(dataset, Dataset)


trl_dataset = dataset.map(to_trl_format, remove_columns=dataset.column_names)
splits = trl_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]
lora_rank = 64
model_id = "Qwen/Qwen3-8B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=4096,
    fast_inference=True,
    dtype=torch.bfloat16,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
)
training_args = GRPOConfig(
    num_generations=16,
    max_prompt_length=1024,
    max_completion_length=4096,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    logging_steps=5,
    save_strategy="steps",
    save_steps=25,
    eval_steps=5,
    num_train_epochs=4,
    bf16=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    optim="adamw_8bit",
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[attack_reward],
    train_dataset=trl_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)


trainer.train()
