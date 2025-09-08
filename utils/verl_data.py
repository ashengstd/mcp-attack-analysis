import re

from datasets import Dataset, load_dataset

from utils.prompt import eval_prompt

NO_ARGUMENTS_PATTERN = re.compile(r"\\nArguments:\\n- No arguments\\n\\n\\n")


def to_verl_format(example, split="train", idx=None):
    # 1. 清理 tool_content
    if NO_ARGUMENTS_PATTERN.search(example["tool_content"]):
        example["tool_content"] = NO_ARGUMENTS_PATTERN.sub("", example["tool_content"])

    user_query = f"Query: {example['query']}"
    tool = (
        f"{{'name': '{example['tool_name']}', "
        f"'description': 'A tool to {example['tool_content']}', "
        f"'parameters': {{'input': {{'type': 'string', 'description': 'The input to the tool'}}}}}}"
    )
    question = f"Tool: {tool}\n{user_query}"

    data = {
        "data_source": "mcp_attack",  # 你可以换成实际来源
        "prompt": [
            {"role": "system", "content": eval_prompt},
            {"role": "user", "content": question},
        ],
        "ability": "tool_use",
        "reward_model": {"style": "rule", "ground_truth": bool(example["is_attack"])},
        "extra_info": {
            "split": split,
            "index": idx,
            "is_attack": bool(example["is_attack"]),
        },
    }
    return data


dataset = load_dataset("parquet", data_files="./data/defense.parquet", split="train")

assert isinstance(dataset, Dataset)

verl_dataset = dataset.map(to_verl_format, remove_columns=dataset.column_names)
split_datasets = verl_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

train_dataset.to_parquet("./data/verl_train.parquet")
eval_dataset.to_parquet("./data/verl_eval.parquet")
