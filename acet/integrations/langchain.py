"""LangChain memory integration for ACET."""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, Dict, List, Optional, Protocol, Sequence, Type, cast

from acet.core.models import ContextDelta, DeltaStatus
from acet.engine import ACETEngine


class _BaseMemoryProtocol(Protocol):
    @property
    def memory_variables(self) -> List[str]:
        ...

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        ...

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        ...

    def clear(self) -> None:
        ...


try:  # pragma: no cover - import shim
    from langchain_core.memory import BaseMemory as _LangChainBaseMemory
except (ImportError, ModuleNotFoundError):  # pragma: no cover - compatibility shim
    try:
        from langchain.memory.base import BaseMemory as _LangChainBaseMemory
    except (ImportError, ModuleNotFoundError):
        _LangChainBaseMemory = object

if isinstance(_LangChainBaseMemory, type):  # pragma: no cover - runtime detection
    _BaseMemoryRuntime: Type[_BaseMemoryProtocol] = cast(
        Type[_BaseMemoryProtocol], _LangChainBaseMemory
    )
else:

    class _FallbackBaseMemory:
        """Fallback base class used when LangChain is unavailable."""

        @property
        def memory_variables(self) -> List[str]:
            raise NotImplementedError

        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            raise NotImplementedError

        async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            raise NotImplementedError

        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
            raise NotImplementedError

        async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
            raise NotImplementedError

        def clear(self) -> None:
            raise NotImplementedError

    _BaseMemoryRuntime = cast(Type[_BaseMemoryProtocol], _FallbackBaseMemory)

_BaseMemoryRuntimeType = cast(type, _BaseMemoryRuntime)


class ACETMemory(_BaseMemoryRuntimeType):  # type: ignore[misc, valid-type]
    """LangChain memory wrapper that sources context from the ACET engine."""

    def __init__(
        self,
        engine: ACETEngine,
        *,
        context_key: str = "ace_context",
        input_keys: Sequence[str] | None = None,
        output_key: str = "response",
        ground_truth_key: Optional[str] = None,
        top_k: int = 10,
        update_context: bool = True,
    ) -> None:
        self.engine = engine
        self.context_key = context_key
        self.input_keys = list(input_keys) if input_keys else ["input", "query", "prompt"]
        self.output_key = output_key
        self.ground_truth_key = ground_truth_key
        self.top_k = top_k
        self.update_context = update_context

        self._last_context_deltas: List[ContextDelta] = []
        self._last_context_bullets: List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        """Return the keys that will be added to LangChain prompts."""
        return [self.context_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper that fetches context for synchronous chains."""
        result = self._run_sync(self.aload_memory_variables(inputs))
        return cast(Dict[str, Any], result)

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async variant used by LangChain async chains."""
        query = self._extract_query(inputs)
        if not query:
            self._reset_context()
            return {self.context_key: ""}

        active = await self.engine.storage.query_deltas(status=DeltaStatus.ACTIVE)
        if not active:
            self._reset_context()
            return {self.context_key: ""}

        ranked = await self.engine.ranker.rank(query, active, top_k=self.top_k)
        deltas = [delta for delta, _ in ranked]
        bullets, _ = self.engine.budget_manager.pack_deltas(deltas)

        usable_count = len(bullets)
        self._last_context_deltas = deltas[:usable_count]
        self._last_context_bullets = bullets

        formatted = "\n".join(bullets)
        return {self.context_key: formatted}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Synchronous wrapper for LangChain compatibility."""
        self._run_sync(self.asave_context(inputs, outputs))

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Ingest interaction results back into ACET."""
        if not self.update_context:
            self._reset_context()
            return

        query = self._extract_query(inputs)
        answer = self._extract_output(outputs)
        if not query or not answer:
            self._reset_context()
            return

        evidence = self._extract_evidence(outputs)
        ground_truth = self._extract_ground_truth(inputs)

        await self.engine.ingest_interaction(
            query=query,
            answer=answer,
            evidence=evidence,
            context_deltas=self._last_context_deltas,
            ground_truth=ground_truth,
        )

        self._reset_context()

    def clear(self) -> None:
        """Reset tracked history. Part of the LangChain memory API."""
        self._reset_context()

    def _reset_context(self) -> None:
        self._last_context_deltas = []
        self._last_context_bullets = []

    def _extract_query(self, inputs: Dict[str, Any]) -> str:
        for key in self.input_keys:
            value = inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _extract_output(self, outputs: Dict[str, Any]) -> str:
        value = outputs.get(self.output_key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            text = value.get("content") or value.get("response") or value.get("text")
            if isinstance(text, str):
                return text
        return ""

    def _extract_evidence(self, outputs: Dict[str, Any]) -> List[str]:
        evidence = outputs.get("evidence")
        if isinstance(evidence, list) and all(isinstance(item, str) for item in evidence):
            return evidence
        return []

    def _extract_ground_truth(self, inputs: Dict[str, Any]) -> Optional[str]:
        if not self.ground_truth_key:
            return None
        value = inputs.get(self.ground_truth_key)
        if isinstance(value, str):
            return value
        return None

    def _run_sync(self, coro: Coroutine[Any, Any, Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "ACETMemory detected an active event loop; use the async `aload_memory_variables` "
            "or `asave_context` methods instead."
        )


