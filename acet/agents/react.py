"""ReAct-style agent built on top of ACET."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from acet.core.models import ContextDelta, DeltaStatus
from acet.engine import ACETEngine
from acet.llm.base import BaseLLMProvider, LLMResponse, Message


@dataclass(frozen=True)
class Tool:
    """Tool specification used by the ReAct agent."""

    name: str
    description: str
    coroutine: Callable[[str], Awaitable[Any]] | Callable[[str], Any]

    async def run(self, tool_input: str) -> str:
        """Execute the tool with a given input and return a stringified observation."""
        result = self.coroutine(tool_input)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)


@dataclass
class _ParsedStep:
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[str] = None
    final_answer: Optional[str] = None


class ReActAgent:
    """Async ReAct loop that leverages ACET-provided context and tooling."""

    THOUGHT_PREFIX = "Thought:"
    ACTION_PREFIX = "Action:"
    ACTION_INPUT_PREFIX = "Action Input:"
    FINAL_ANSWER_PREFIX = "Final Answer:"

    def __init__(
        self,
        engine: ACETEngine,
        llm: BaseLLMProvider,
        tools: Sequence[Tool],
        *,
        system_prompt: Optional[str] = None,
        max_steps: int = 6,
        context_top_k: int = 10,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.engine = engine
        self.llm = llm
        self.tools = list(tools)
        self.max_steps = max_steps
        self.context_top_k = context_top_k
        self.llm_kwargs = llm_kwargs or {}

        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools
        ) or "- (no tools available)"
        default_prompt = (
            "You are an autonomous ReAct agent. Reason step-by-step, decide whether to use a tool, "
            "and continue until you can provide a final answer.\n"
            "Format each response as:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool name or 'none'>\n"
            "Action Input: <input for the tool>\n"
            "If you have finished, respond with:\n"
            "Final Answer: <concise answer>\n"
            "Available tools:\n"
            f"{tool_descriptions}"
        )
        self.system_prompt = system_prompt or default_prompt
        self._tool_index = {tool.name: tool for tool in self.tools}

    async def run(
        self,
        query: str,
        *,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the ReAct loop for a single query."""
        active_deltas = await self.engine.storage.query_deltas(status=DeltaStatus.ACTIVE)
        ranked = await self.engine.ranker.rank(query, active_deltas, top_k=self.context_top_k)
        ordered_deltas = [delta for delta, _ in ranked]
        context_bullets, tokens_used = self.engine.budget_manager.pack_deltas(ordered_deltas)
        usable_deltas: List[ContextDelta] = ordered_deltas[: len(context_bullets)]

        context_section = "\n".join(context_bullets) if context_bullets else "None."
        system_content = (
            f"{self.system_prompt}\n\n### Context Playbook\n{context_section}\n"
            "Use the context when reasoning and deciding on tool usage."
        )

        messages: List[Message] = [
            Message(role="system", content=system_content),
            Message(role="user", content=query),
        ]

        steps: List[Dict[str, Any]] = []
        observations: List[str] = []
        llm_responses: List[LLMResponse] = []
        final_answer: Optional[str] = None

        for _ in range(self.max_steps):
            response = await self.llm.complete(messages, **self.llm_kwargs)
            llm_responses.append(response)

            content = response.content.strip()
            messages.append(Message(role="assistant", content=content))

            parsed = self._parse_response(content)
            step_record = {
                "thought": parsed.thought,
                "action": parsed.action,
                "action_input": parsed.action_input,
                "raw_response": content,
                "usage": response.usage,
            }
            steps.append(step_record)

            if parsed.final_answer:
                final_answer = parsed.final_answer
                break

            if parsed.action and parsed.action.lower() != "none":
                observation = await self._execute_tool(parsed.action, parsed.action_input or "")
                observations.append(observation)
                messages.append(
                    Message(role="tool", content=f"{parsed.action} observation: {observation}")
                )
                continue

            # When no action or final answer is provided, treat the whole content as the answer.
            final_answer = content
            break

        if final_answer is None:
            final_answer = (
                f"Unable to produce a final answer after {self.max_steps} steps. "
                "Review the intermediate reasoning for details."
            )

        ace_result = await self.engine.ingest_interaction(
            query=query,
            answer=final_answer,
            evidence=observations,
            context_deltas=usable_deltas,
            ground_truth=ground_truth,
        )

        result_metadata = {
            "context_tokens": ace_result.get("context_tokens", tokens_used),
            "steps": steps,
            "llm_responses": llm_responses,
            "observations": observations,
        }
        if metadata:
            result_metadata.update(metadata)

        return {
            "answer": final_answer,
            "context": context_bullets,
            "created_deltas": ace_result.get("created_deltas", []),
            "report": ace_result.get("report"),
            "metadata": result_metadata,
        }

    async def _execute_tool(self, action: str, tool_input: str) -> str:
        tool = self._tool_index.get(action)
        if not tool:
            available = ", ".join(self._tool_index.keys()) or "none"
            return f"Unknown tool '{action}'. Available tools: {available}"
        try:
            return await tool.run(tool_input)
        except Exception as exc:  # pragma: no cover - defensive guard
            return f"Tool '{action}' failed with error: {exc}"

    def _parse_response(self, text: str) -> _ParsedStep:
        parsed = _ParsedStep()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(self.FINAL_ANSWER_PREFIX):
                parsed.final_answer = stripped[len(self.FINAL_ANSWER_PREFIX) :].strip()
            elif stripped.startswith(self.ACTION_PREFIX):
                parsed.action = stripped[len(self.ACTION_PREFIX) :].strip()
            elif stripped.startswith(self.ACTION_INPUT_PREFIX):
                parsed.action_input = stripped[len(self.ACTION_INPUT_PREFIX) :].strip()
            elif stripped.startswith(self.THOUGHT_PREFIX):
                parsed.thought = stripped[len(self.THOUGHT_PREFIX) :].strip()
        return parsed

