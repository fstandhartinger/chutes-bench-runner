"""Inference client interface shared by Chutes and Gremium providers."""
from __future__ import annotations

from typing import Any, Optional, Protocol


class InferenceClient(Protocol):
    """Minimal interface implemented by inference providers."""

    provider: str

    async def run_inference(
        self,
        model_slug: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        ...

    async def get_completion_text(
        self,
        model_slug: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        ...

    async def get_completion_messages(
        self,
        model_slug: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        ...

    async def get_model_max_output_length(self, model_identifier: str) -> Optional[int]:
        ...

    async def get_model_context_length(self, model_identifier: str) -> Optional[int]:
        ...

    def get_api_base_url(self) -> str:
        ...

    def get_api_key(self) -> Optional[str]:
        ...

    async def close(self) -> None:
        ...
