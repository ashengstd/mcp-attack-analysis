import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from rich.logging import RichHandler

from utils.attack import AttackScenario, load_attack_scenarios
from utils.llm import LLMClient
from utils.prompt import generate_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

load_dotenv()


async def llm_chain(llm: LLMClient, attack: AttackScenario) -> AttackScenario:
    user_query = f"Query: {attack.query}"
    tool = (
        f"{{'name': '{attack.tool_name}', 'description': 'A tool to {attack.tool_content}', "
        f"'parameters': {{'input': {{'type': 'string', 'description': 'The input to the tool'}}}}}}"
    )
    prompt = f"Tool: {tool}\nQuery: {user_query}"
    logger.info(f"Harmful prompt: {prompt}")
    defense = await llm.generate(prompt=prompt)
    logger.info(
        f"Attack ID: {attack.attack_id}, Server: {attack.server_name}, Tool: {attack.tool_name}, Defense Response: {defense}"
    )
    attack.defense = defense
    return attack


async def main():
    attacks_file = Path("data/attacks.json")
    scenarios = load_attack_scenarios(attacks_file)

    llm = LLMClient(
        model=os.getenv("MODEL", "ERROR: No MODEL in env"),
        api_key=os.getenv("API_KEY", "ERROR: No API_KEY in env"),
        api_base=os.getenv("API_BASE", "ERROR: No API_BASE in env"),
        default_system_prompt=generate_prompt,
    )

    # ⚠️ 并发执行
    sem = asyncio.Semaphore(32)  # 最多并发 32 个

    async def limited_llm_chain(llm, attack):
        async with sem:
            return await llm_chain(llm, attack)

    tasks = [limited_llm_chain(llm, s) for s in scenarios]
    defense = await asyncio.gather(*tasks)

    rows = [d.model_dump() for d in defense]
    table = pa.Table.from_pylist(rows)
    output_file = Path("data/defence.parquet")
    pq.write_table(table, output_file)

    print(f"Saved {len(defense)} records to {output_file}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
