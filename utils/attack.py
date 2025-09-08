import json
from pathlib import Path

from pydantic import BaseModel


class AttackScenario(BaseModel):
    attack_id: str
    server_name: str
    tool_name: str
    tool_content: str
    query: str
    security_risk: str
    paradigm: str
    defense: str = ""


def load_attack_scenarios(file_path: Path) -> list[AttackScenario]:
    assert file_path.exists(), f"File {file_path} does not exist."
    with open(file_path, "r") as file:
        data = json.load(file)
    scenarios = []
    for attack_type_dict in data:
        for attack_id, attack in attack_type_dict.items():
            scenarios.append(
                AttackScenario(
                    attack_id=attack_id,
                    server_name=attack["server_name"],
                    tool_name=attack["tool_name"],
                    tool_content=attack["tool_content"],
                    query=attack["query"],
                    security_risk=attack["security risk"],
                    paradigm=attack["paradigm"],
                )
            )
    return scenarios
