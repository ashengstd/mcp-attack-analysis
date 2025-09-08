import unsloth  # noqa: F401  # isort: skip
from unsloth import FastLanguageModel  # noqa: F401  # isort: skip
import logging

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from rich.logging import RichHandler
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from train.reward import attack_reward, format_reward
from utils.prompt import eval_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

dataset = load_dataset("parquet", data_files="./data/defense.parquet", split="train")

assert isinstance(dataset, Dataset)


def to_trl_format(example):
    user_query = f"Query: {example['query']}"
    tool = (
        f"{{'name': '{example['tool_name']}', 'description': 'A tool to {example['tool_content']}', "
        f"'parameters': {{'input': {{'type': 'string', 'description': 'The input to the tool'}}}}}}"
    )
    prompt = f"{eval_prompt}\n\nTool: {tool}\nQuery: {user_query}"
    return {
        "prompt": prompt,
        "is_attack": int(example["is_attack"]),
    }


trl_dataset = dataset.map(to_trl_format, remove_columns=dataset.column_names)
lora_rank = 32
model_id = "Qwen/Qwen3-8B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=4096,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.9,
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
    num_generations=8,
    per_device_train_batch_size=4,  # 按需调整
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=4,
    bf16=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    optim="adamw_8bit",
    use_vllm=True,
    vllm_mode="colocate",
)


trainer = GRPOTrainer(
    model=model,
    reward_funcs=[attack_reward, format_reward],
    train_dataset=trl_dataset,
    args=training_args,
)


trainer.train()
