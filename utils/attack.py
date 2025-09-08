from pathlib import Path
from typing import Literal

from datasets import load_from_disk
from pydantic import BaseModel


class AttackScenario(BaseModel):
    attack_id: str
    server_name: str
    tool_name: str
    tool_content: str
    query: str
    is_attack: bool


class AttackResult(AttackScenario):
    response: str
    attack_success: bool


class AttackDefense(AttackScenario):
    defense: str


def load_attack_scenarios(
    file_path: Path, split: Literal["train", "test"] = "test"
) -> list[AttackScenario]:
    dataset = load_from_disk(str(file_path))
    assert split in dataset
    ds = dataset[split]
    scenarios = [AttackScenario(**item) for item in ds]  # type: ignore
    return scenarios
