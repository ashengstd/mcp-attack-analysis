import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl.trainer.sft_trainer import SFTTrainer

from utils.tool import remove_think_content

dataset = load_dataset("parquet", data_files="./data/defence.parquet", split="train")

assert isinstance(dataset, Dataset)


def to_trl_format(example):
    user_query = f"Query: {example['query']}"
    tool = (
        f"{{'name': '{example['tool_name']}', 'description': 'A tool to {example['tool_content']}', "
        f"'parameters': {{'input': {{'type': 'string', 'description': 'The input to the tool'}}}}}}"
    )
    prompt = f"Tool: {tool}\nQuery: {user_query}"
    completion = remove_think_content(example["defense"])
    return {
        "prompt": prompt,
        "completion": completion,
        "is_attack": int(example["is_attack"]),
    }


trl_dataset = dataset.map(to_trl_format, remove_columns=dataset.column_names)

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
sft_dataset = load_dataset("trl-lib/Capybara", split="train")
assert isinstance(sft_dataset, Dataset)

trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset,
    peft_config=lora_config,
)
trainer.train()
