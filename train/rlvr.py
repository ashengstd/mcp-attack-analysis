import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.callbacks import RichProgressCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from train.reward import attack_reward
from utils.tool import to_trl_format

dataset = load_from_disk("./data/attacks")
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

assert isinstance(train_dataset, Dataset)
assert isinstance(eval_dataset, Dataset)

train_dataset = train_dataset.map(
    to_trl_format, remove_columns=train_dataset.column_names
)
eval_dataset = eval_dataset.map(to_trl_format, remove_columns=eval_dataset.column_names)

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="kernels-community/flash-attn",
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

training_args = GRPOConfig(
    max_prompt_length=1024,
    max_completion_length=8196,
    num_generations=32,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,
    eval_steps=5,
    num_train_epochs=4,
    bf16=True,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    use_vllm=True,
    vllm_mode="colocate",
    vllm_tensor_parallel_size=8,
    vllm_gpu_memory_utilization=0.3,
    vllm_enable_sleep_mode=True,
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[attack_reward],
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    callbacks=[RichProgressCallback()],
)


trainer.train()
