import asyncio
import time
from typing import Optional, cast

from litellm import acompletion
from litellm.types.utils import (
    ModelResponse,
)
from pydantic.dataclasses import dataclass


@dataclass
class AsyncRateLimiter:
    calls_per_minute: int
    max_burst: Optional[int] = None

    def __post_init__(self):
        self.max_burst = self.max_burst or self.calls_per_minute
        self._tokens = self.max_burst
        self._lock = asyncio.Lock()
        self._last_refill = time.monotonic()
        self._refill_interval = 60 / self.calls_per_minute

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            new_tokens = elapsed / self._refill_interval
            self._tokens = min(cast(int, self.max_burst), self._tokens + new_tokens)
            self._last_refill = now

            if self._tokens >= 1:
                self._tokens -= 1
                return
            wait_time = self._refill_interval - elapsed
        await asyncio.sleep(max(wait_time, 0))
        await self.acquire()


@dataclass
class LLMClient:
    model: str
    api_key: str
    api_base: str = ""
    timeout: Optional[int] = 600
    additional_system_prompt: Optional[str] = None
    calls_per_minute: Optional[int] = None  # None 表示不限制
    burst: Optional[int] = None
    default_system_prompt: str = (
        "You are a helpful assistant. You can answer questions based on the provided tools and queries. "
        "If you want to use a tool, use the json format below. Always answer concisely. "
        'tool call format: {"tool_call": {"name": "<TOOL_NAME>", "parameters": {"<parameter_name_1>": "<PARAMETER_VALUE_1>", "<parameter_name_2>": "<PARAMETER_VALUE_2>"}}}. '
        "Only reply with the tool call format if you want to use a tool. If not, just answer the question directly."
    )

    def __post_init__(self):
        self.rate_limiter = (
            AsyncRateLimiter(self.calls_per_minute, self.burst)
            if self.calls_per_minute is not None
            else None
        )

    async def generate_with_rate_limiter(self, prompt: str) -> str:
        if self.rate_limiter:
            await self.rate_limiter.acquire()
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

    async def generate(self, prompt: str) -> str:
        """同步调用，会阻塞，如果设置了限流不会生效"""
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
