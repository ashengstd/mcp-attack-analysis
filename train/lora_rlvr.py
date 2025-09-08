import logging

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from peft import LoraConfig
from rich.logging import RichHandler
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from train.reward import attack_reward
from utils.tool import to_trl_format

lora_rank = 32

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
rlvr_dataset = splits["train"]
sft_dataset = splits["test"]

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    load_in_4bit=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_args = SFTConfig(
    per_device_train_batch_size=4,
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=2,
    bf16=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
)

sft_trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset,
    peft_config=lora_config,
    args=sft_args,
)

sft_trainer.train()
sft_trainer.save_model("output/sft_bootstrap/")
tokenizer.save_pretrained("output/sft_bootstrap/")


del sft_trainer
del model
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    "output/sft_bootstrap/",
    torch_dtype=torch.bfloat16,
    load_in_4bit=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "output/sft_bootstrap/", trust_remote_code=True
)

training_args = GRPOConfig(
    max_completion_length=4096,
    num_generations=16,
    per_device_train_batch_size=4,
    logging_steps=10,
    save_strategy="no",
    num_train_epochs=4,
    bf16=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    optim="adamw_8bit",
    use_vllm=True,
    vllm_mode="colocate",
    vllm_tensor_parallel_size=4,
    vllm_gpu_memory_utilization=0.6,
    vllm_enable_sleep_mode=True,
)


trainer = GRPOTrainer(
    model=model,
    reward_funcs=[attack_reward],
    train_dataset=trl_dataset,
    peft_config=lora_config,
    args=training_args,
)


trainer.train()

trainer.save_model("output/rl/lora/")
tokenizer.save_pretrained("output/rl/lora/")
