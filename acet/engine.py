"""ACET engine orchestration."""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from acet.core.budget import TokenBudgetManager
from acet.core.interfaces import Curator, Generator, Reflector, StorageBackend
from acet.core.models import ACETConfig, ContextDelta, DeltaStatus, ReflectionReport
from acet.retrieval import DeltaRanker

logger = structlog.get_logger(__name__)


class ACETEngine:
    """Coordinates generator, reflector, curator, and storage workflows."""

    def __init__(
        self,
        generator: Generator,
        reflector: Reflector,
        curator: Curator,
        storage: StorageBackend,
        ranker: DeltaRanker,
        config: Optional[ACETConfig] = None,
    ) -> None:
        self.generator = generator
        self.reflector = reflector
        self.curator = curator
        self.storage = storage
        self.ranker = ranker
        self.config = config or ACETConfig()
        self.budget_manager = TokenBudgetManager(budget=self.config.token_budget)

    async def run_offline_adaptation(
        self,
        training_data: List[Dict[str, Any]],
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Iterate over labelled data to build context deltas."""
        total_epochs = epochs or self.config.max_epochs
        stats = {
            "epochs": total_epochs,
            "total_samples": len(training_data),
            "deltas_created": 0,
            "deltas_activated": 0,
            "reflections_ran": 0,
        }

        for epoch in range(total_epochs):
            logger.info("offline_adaptation_epoch_start", epoch=epoch + 1, total=total_epochs)
            for index, sample in enumerate(training_data):
                query = sample.get("query") or sample.get("question")
                if not query:
                    logger.warning("offline_sample_missing_query", index=index)
                    continue

                ground_truth = sample.get("ground_truth")
                await self._process_sample(
                    query=query,
                    ground_truth=ground_truth,
                    stats=stats,
                    run_reflection=True,
                )

                if (index + 1) % 25 == 0:
                    logger.info(
                        "offline_adaptation_progress",
                        processed=index + 1,
                        total=len(training_data),
                        epoch=epoch + 1,
                    )

            activated = await self.storage.activate_staged()
            stats["deltas_activated"] += activated
            logger.info("offline_adaptation_epoch_complete", epoch=epoch + 1, activated=activated)

        logger.info("offline_adaptation_complete", stats=stats)
        return stats

    async def run_online_adaptation(
        self,
        query: str,
        ground_truth: Optional[str] = None,
        update_context: bool = True,
    ) -> Dict[str, Any]:
        """Handle a single live query with optional reflection and context updates."""
        logger.info("online_adaptation_start", query_preview=query[:80])

        active_deltas = await self._load_relevant_deltas(query)
        context_bullets, tokens_used = self.budget_manager.pack_deltas(active_deltas)

        generation = await self.generator.generate(query, context_bullets)

        created_deltas: List[ContextDelta] = []
        if update_context:
            report = await self.reflector.reflect(
                query=query,
                answer=generation.get("answer", ""),
                evidence=generation.get("evidence", []),
                context=context_bullets,
                ground_truth=ground_truth,
            )
            created_deltas = await self._curate_and_persist(report)

        result = {
            "answer": generation.get("answer", ""),
            "evidence": generation.get("evidence", []),
            "injected_context": context_bullets,
            "created_deltas": created_deltas,
            "metadata": {
                "active_deltas_count": len(active_deltas),
                "context_tokens": tokens_used,
                **generation.get("metadata", {}),
            },
        }

        logger.info("online_adaptation_complete", created=len(created_deltas))
        return result

    async def _process_sample(
        self,
        query: str,
        ground_truth: Optional[str],
        stats: Dict[str, Any],
        run_reflection: bool,
    ) -> None:
        active_deltas = await self._load_relevant_deltas(query)
        context_bullets, _ = self.budget_manager.pack_deltas(active_deltas)

        generation = await self.generator.generate(query, context_bullets)

        if run_reflection and self._should_reflect():
            stats["reflections_ran"] += 1
            report = await self.reflector.reflect(
                query=query,
                answer=generation.get("answer", ""),
                evidence=generation.get("evidence", []),
                context=context_bullets,
                ground_truth=ground_truth,
            )
            new_deltas = await self._curate_and_persist(report)
            stats["deltas_created"] += len(new_deltas)

    async def _curate_and_persist(
        self,
        report: ReflectionReport,
    ) -> List[ContextDelta]:
        existing_deltas = await self.storage.query_deltas()
        curated = await self.curator.curate(report, existing_deltas)
        if curated:
            await self.storage.save_deltas(curated)
            logger.info("curator_saved_deltas", count=len(curated))
        return curated

    async def _load_relevant_deltas(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[ContextDelta]:
        active = await self.storage.query_deltas(status=DeltaStatus.ACTIVE)
        if not active:
            return []

        ranked = await self.ranker.rank(query, active, top_k=top_k)
        return [delta for delta, _ in ranked]

    def _should_reflect(self) -> bool:
        if self.config.reflection_sample_rate <= 0.0:
            return False
        if self.config.reflection_sample_rate >= 1.0:
            return True
        return random.random() < self.config.reflection_sample_rate

    async def ingest_interaction(
        self,
        query: str,
        answer: str,
        *,
        evidence: Optional[List[str]] = None,
        context_deltas: Optional[List[ContextDelta]] = None,
        ground_truth: Optional[str] = None,
        update_usage: bool = True,
    ) -> Dict[str, Any]:
        """Ingest an externally generated interaction for reflection and curation."""
        evidence_list = evidence or []
        deltas = context_deltas or []
        context_bullets, tokens_used = self.budget_manager.pack_deltas(deltas)

        report = await self.reflector.reflect(
            query=query,
            answer=answer,
            evidence=evidence_list,
            context=context_bullets,
            ground_truth=ground_truth,
        )
        created_deltas = await self._curate_and_persist(report)

        if update_usage and deltas:
            for delta in deltas:
                delta.usage_count += 1
                delta.updated_at = datetime.utcnow()
                await self.storage.update_delta(delta)

        return {
            "report": report,
            "created_deltas": created_deltas,
            "context_tokens": tokens_used,
        }
