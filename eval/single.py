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


async def llm_chain(llm: LLMClient, attack: AttackScenario):
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


async def main():
    attacks_file = Path("data/attacks.json")
    scenarios = load_attack_scenarios(attacks_file)
    llm = LLMClient(
        model=os.getenv("MODEL", "ERROR: No MODEL in env"),
        api_key=os.getenv("API_KEY", "ERROR: No API_KEY in env"),
        api_base=os.getenv("API_BASE", "ERROR: No API_BASE in env"),
        default_system_prompt=eval_prompt,
    )
    await llm_chain(
        llm=llm,
        attack=AttackScenario(
            attack_id="test",
            server_name="test_server",
            tool_name="test_tool",
            tool_content="Get the weather.",
            query="What's the weather like today?",
            security_risk="low",
            paradigm="test_paradigm",
        ),
    )
    for scenario in scenarios:
        await llm_chain(llm=llm, attack=scenario)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
