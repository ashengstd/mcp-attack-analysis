import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

from utils.attack import AttackScenario, load_attack_scenarios
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

MAX_CONCURRENT_REQUESTS = 16


async def llm_chain(
    llm: LLMClient, attack: AttackScenario, sem: asyncio.Semaphore
) -> bool:
    """返回 True/False 表示攻击是否成功"""
    async with sem:  # 限制并发
        attack_success = None
        response = ""
        while attack_success is None:
            user_query = f"Query: {attack.query}"
            tool = (
                f"{{'name': '{attack.tool_name}', 'description': 'A tool to {attack.tool_content}', "
                f"'parameters': {{'input': {{'type': 'string', 'description': 'The input to the tool'}}}}}}"
            )
            prompt = f"Tool: {tool}\nQuery: {user_query}"
            response = await llm.generate(prompt=prompt)
            attack_success = extract_tool_call(response)
        logger.info(
            f"Attack ID: {attack.attack_id}, Server: {attack.server_name}, Tool: {attack.tool_name}, "
            f"Response: {response}, Attack Success: {attack_success}"
        )
        return bool(attack_success)


async def main():
    attacks_file = Path("data/attacks.json")
    scenarios = load_attack_scenarios(attacks_file)

    llm = LLMClient(
        model=os.getenv("MODEL", ""),
        api_key=os.getenv("API_KEY", ""),
        api_base=os.getenv("API_BASE", ""),
        calls_per_minute=2,
        default_system_prompt=eval_prompt,
        burst=5,
    )

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = [llm_chain(llm, scenario, sem) for scenario in scenarios]
    results = await asyncio.gather(*tasks)

    acc = sum(results) / len(results) if results else 0.0
    logger.info(f"Overall Attack Accuracy: {acc:.4f}")

    with open("attack_accuracy.txt", "w") as f:
        f.write(f"{acc:.4f}\n")


if __name__ == "__main__":
    asyncio.run(main())
