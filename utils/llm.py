from typing import Optional, cast

from litellm import acompletion
from litellm.types.utils import (
    ModelResponse,
)
from pydantic.dataclasses import dataclass


@dataclass
class LLMClient:
    model: str
    api_key: str
    api_base: str = ""
    timeout: Optional[int] = 600
    additional_system_prompt: Optional[str] = None
    calls_per_minute: Optional[int] = None  # None 表示不限制
    burst: Optional[int] = None
    default_system_prompt: str = ""

    async def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.default_system_prompt
                + (self.additional_system_prompt or ""),
            },
            {"role": "user", "content": prompt},
        ]
        reponse = cast(
            ModelResponse,
            await acompletion(
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base,
                messages=messages,
                stream=False,
                timeout=self.timeout,
            ),
        )
        return reponse.choices[0].message.content  # pyright: ignore[reportReturnType, reportAttributeAccessIssue]
