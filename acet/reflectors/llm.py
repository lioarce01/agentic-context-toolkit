"""LLM-based reflector implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, cast

from acet.core.interfaces import Reflector
from acet.core.models import ReflectionReport
from acet.llm.base import BaseLLMProvider, Message


class LLMReflector(Reflector):
    """Reflector that leverages an injected LLM provider."""

    REFLECTION_PROMPT = """You are an expert analyst. Review the interaction below and provide JSON feedback.\n\nQuestion: {question}\nAnswer: {answer}\nEvidence: {evidence}\nContext Used: {context}\nGround Truth: {ground_truth}\n\nIdentify issues (omissions, contradictions, policy gaps, uncertainties, hallucinations)\nand propose new reusable insights. Respond with valid JSON using the schema:\n{{\n  "issues": [{{"type": "omission", "explanation": "...", "severity": 3}}],\n  "proposed_insights": [{{\n    "topic": "...",\n    "guideline": "...",\n    "conditions": ["..."],\n    "evidence": ["..."],\n    "tags": ["..."],\n    "confidence": 0.85\n  }}]\n}}"""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self.llm = llm_provider

    async def reflect(
        self,
        query: str,
        answer: str,
        evidence: List[str],
        context: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectionReport:
        payload = self.REFLECTION_PROMPT.format(
            question=query,
            answer=answer,
            evidence="\n".join(evidence) if evidence else "(none)",
            context="\n".join(context) if context else "(none)",
            ground_truth=ground_truth or "Not provided",
        )

        messages = [
            Message(role="system", content="You critique LLM outputs and capture reusable insights."),
            Message(role="user", content=payload),
        ]

        llm_kwargs = {**kwargs}
        llm_kwargs.setdefault("response_format", {"type": "json_object"})

        try:
            response = await self.llm.complete(messages, **llm_kwargs)
        except Exception:
            llm_kwargs.pop("response_format", None)
            response = await self.llm.complete(messages, **llm_kwargs)

        data = self._parse_json(response.content)
        return ReflectionReport(
            question=query,
            answer=answer,
            evidence_refs=evidence,
            issues=data.get("issues", []),
            proposed_insights=data.get("proposed_insights", []),
        )

    async def refine(
        self,
        report: ReflectionReport,
        iterations: int = 3,
    ) -> ReflectionReport:
        current = report
        for _ in range(max(iterations, 0)):
            prompt = (
                "Improve the reflection report JSON below. Ensure issues and proposed_insights "
                "are accurate and complete. Return valid JSON with the same structure.\n\n"
                f"{current.model_dump_json(indent=2)}"
            )
            messages = [
                Message(role="system", content="You refine JSON analysis reports."),
                Message(role="user", content=prompt),
            ]
            try:
                response = await self.llm.complete(messages, response_format={"type": "json_object"})
            except Exception:
                response = await self.llm.complete(messages)

            try:
                data = self._parse_json(response.content)
                current = ReflectionReport(
                    question=current.question,
                    answer=current.answer,
                    evidence_refs=current.evidence_refs,
                    issues=data.get("issues", []),
                    proposed_insights=data.get("proposed_insights", []),
                )
            except ValueError:
                break
        return current

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        def _load_json(candidate: str) -> Dict[str, Any]:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("Reflector response must be a JSON object.")
            return cast(Dict[str, Any], data)

        try:
            return _load_json(content)
        except json.JSONDecodeError:
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    try:
                        return _load_json(part)
                    except json.JSONDecodeError:
                        continue
            raise ValueError("Failed to parse reflector JSON response.") from None
