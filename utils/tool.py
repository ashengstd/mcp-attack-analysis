import json
import re

from utils.prompt import eval_prompt

NO_ARGUMENTS_PATTERN = re.compile(r"\\nArguments:\\n- No arguments\\n\\n\\n")


def extract_tool_call(text: str) -> bool:
    text = remove_think_content(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False

    if not isinstance(data, dict):
        return False

    if data.get("tool_call") is not None:
        tool = data["tool_call"]
        if isinstance(tool, dict) and "name" in tool and "parameters" in tool:
            return True
    return False


def is_valid_structured_output(text: str) -> bool:
    text = remove_think_content(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False

    if not isinstance(data, dict):
        return False
    if "tool_call" not in data or "answer" not in data:
        return False

    # 互斥检查
    if data["tool_call"] is not None:
        if not isinstance(data["tool_call"], dict):
            return False
        if "name" not in data["tool_call"] or "parameters" not in data["tool_call"]:
            return False
    else:
        if not data["answer"]:
            return False

    return True


def remove_think_content(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def to_trl_format(example):
    if NO_ARGUMENTS_PATTERN.search(example["tool_content"]):
        example["tool_content"] = NO_ARGUMENTS_PATTERN.sub("", example["tool_content"])
    user_query = f"Query: {example['query']}"
    tool = (
        f"{{'name': '{example['tool_name']}', 'description': 'A tool to {example['tool_content']}', "
        f"'parameters': {{'input': {{'type': 'string', 'description': 'The input to the tool'}}}}}}"
    )
    prompt = f"Tool: {tool}\nQuery: {user_query}"
    return {
        "prompt": [
            {"role": "system", "content": eval_prompt},
            {"role": "user", "content": prompt},
        ],
        "is_attack": int(example["is_attack"]),
    }
