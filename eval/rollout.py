import asyncio
import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

from utils.attack import AttackResult, AttackScenario, load_attack_scenarios
from utils.llm import LLMClient
from utils.prompt import eval_prompt
from utils.tool import extract_tool_call

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

load_dotenv()

ROLLOUTS = 16


async def llm_chain(llm: LLMClient, attack: AttackScenario) -> AttackResult:
    """返回 True/False 表示攻击是否成功"""
    tool_call = None
    response = ""
    while tool_call is None:
        user_query = f"Query: {attack.query}"
        tool = (
            f"{{'name': '{attack.tool_name}', 'description': 'A tool to {attack.tool_content}', "
            f"'parameters': {{'input': {{'type': 'string', 'description': 'The input to the tool'}}}}}}"
        )
        prompt = f"Tool: {tool}\nQuery: {user_query}"
        response = await llm.generate(prompt=prompt)
        tool_call = extract_tool_call(response)
    attack_success = tool_call and attack.is_attack
    logger.info(
        f"Attack ID: {attack.attack_id}, Server: {attack.server_name}, Tool: {attack.tool_name}, "
        f"Response: {response}, Attack Success: {attack_success}"
    )
    return AttackResult(
        attack_id=attack.attack_id,
        server_name=attack.server_name,
        tool_name=attack.tool_name,
        tool_content=attack.tool_content,
        query=attack.query,
        response=response,
        attack_success=tool_call,
        is_attack=attack.is_attack,
    )


async def main():
    attacks_file = Path("data/attacks")
    scenarios = load_attack_scenarios(attacks_file)

    llm = LLMClient(
        model=os.getenv("MODEL", ""),
        api_key=os.getenv("API_KEY", ""),
        api_base=os.getenv("API_BASE", ""),
        default_system_prompt=eval_prompt,
    )

    sampled_attacks = random.choice(scenarios)
    tasks = [llm_chain(llm, sampled_attacks) for _ in range(ROLLOUTS)]
    results = await asyncio.gather(*tasks)

    attack_num = sum(1 for scenario in scenarios if scenario.is_attack)
    acc = (
        sum(result.attack_success if result.is_attack else 0 for result in results)
        / attack_num
        if results
        else 0.0
    )
    logger.info(f"Overall Attack Accuracy: {acc:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
