import json
import logging
import re

logger = logging.getLogger(__name__)

TOOL_CALL_PATTERN = re.compile(r'(?P<json>\{"tool_call".*?\})', re.DOTALL)


def extract_tool_call(text: str) -> bool:
    text = remove_think_tags(text)
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return False
    snippet = match.group("json")
    try:
        _ = json.loads(snippet)
    except json.JSONDecodeError:
        return False
    return True


def valid_tool_call(text: str) -> bool:
    """Check if the text contains a valid tool call in JSON format."""
    text = remove_think_tags(text)
    start = text.find('{"tool_call"')
    if start == -1:
        return False
    else:
        match = TOOL_CALL_PATTERN.search(text)
        if not match:
            return False
        snippet = match.group("json")
        try:
            _ = json.loads(snippet)
            return True
        except json.JSONDecodeError:
            return False


def remove_think_tags(text: str) -> str:
    return re.sub(r"<think>(.*?)</think>", r"\1", text, flags=re.DOTALL)
