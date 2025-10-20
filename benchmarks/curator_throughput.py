"""Benchmark curator throughput with varying duplicate ratios."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import uuid
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acet.core.models import ContextDelta, ReflectionReport  # noqa: E402
from acet.curators.standard import StandardCurator  # noqa: E402


class StubEmbeddingProvider:
    """Inexpensive embedding provider used to keep runs deterministic."""

    async def embed(self, text: str) -> List[float]:
        return [float(ord(ch) % 101) / 50.0 for ch in text[:32]]

    async def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [await self.embed(text) for text in texts]

    def similarity(self, emb1: Sequence[float], emb2: Sequence[float]) -> float:
        if not emb1 or not emb2:
            return 0.0
        dot = sum(a * b for a, b in zip(emb1, emb2, strict=False))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)


def build_existing(count: int) -> List[ContextDelta]:
    return [
        ContextDelta(
            id=str(uuid.uuid4()),
            topic=f"existing-topic-{index % 5}",
            guideline=f"Existing guideline #{index}",
            evidence=[f"doc-{index}"],
            tags=[f"tag-{index % 3}"],
            confidence=0.8,
        )
        for index in range(count)
    ]


def build_report(proposals: int, duplicate_ratio: float) -> ReflectionReport:
    proposed = []
    duplicate_cutoff = int(proposals * duplicate_ratio)
    for index in range(proposals):
        if index < duplicate_cutoff:
            topic = f"existing-topic-{index % 5}"
            guideline = f"Existing guideline #{index % 10}"
        else:
            topic = f"new-topic-{index % 7}"
            guideline = f"New guideline #{index}"

        proposed.append(
            ReflectionReport.ProposedInsight(
                topic=topic,
                guideline=guideline,
                conditions=[f"condition-{index % 3}"],
                evidence=[f"evidence-{index % 4}"],
                tags=[f"tag-{index % 5}"],
                confidence=0.7,
            )
        )

    return ReflectionReport(
        question="benchmark",
        answer="benchmark",
        proposed_insights=proposed,
    )


async def run_iteration(
    existing_count: int,
    proposals: int,
    duplicate_ratio: float,
    curator: StandardCurator,
) -> None:
    existing = build_existing(existing_count)
    report = build_report(proposals, duplicate_ratio)
    await curator.curate(report, existing)


async def run_benchmark(
    iterations: int,
    existing_count: int,
    proposals: int,
    duplicate_ratio: float,
) -> List[float]:
    curator = StandardCurator(embedding_provider=StubEmbeddingProvider(), min_confidence=0.0)
    timings: List[float] = []
    for _ in range(iterations):
        start = perf_counter()
        await run_iteration(existing_count, proposals, duplicate_ratio, curator)
        timings.append(perf_counter() - start)
    return timings


def summarise(samples: Sequence[float]) -> str:
    mean = statistics.mean(samples) * 1000
    median = statistics.median(samples) * 1000
    stdev = statistics.pstdev(samples) * 1000
    p95 = statistics.quantiles(samples, n=100, method="inclusive")[94] * 1000 if len(samples) >= 5 else None
    parts = [
        f"mean={mean:.2f} ms",
        f"median={median:.2f} ms",
        f"stdev={stdev:.2f} ms",
    ]
    if p95 is not None:
        parts.append(f"p95={p95:.2f} ms")
    return ", ".join(parts)


def maybe_plot(
    samples: Sequence[float],
    output_path: Path,
    subtitle: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc

    times_ms = [value * 1000 for value in samples]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(times_ms) + 1), times_ms, marker="o", linewidth=1)
    plt.xlabel("Iteration")
    plt.ylabel("Latency (ms)")
    plt.title("Curator Curate() Latency per Iteration")
    if subtitle:
        plt.suptitle(subtitle, y=0.96, fontsize=9)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved curator throughput plot to {output_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark curator dedup throughput.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations.")
    parser.add_argument("--existing", type=int, default=150, help="Existing delta count supplied to the curator.")
    parser.add_argument("--proposals", type=int, default=300, help="Number of proposed insights per iteration.")
    parser.add_argument(
        "--duplicate-ratio",
        type=float,
        default=0.3,
        help="Fraction of proposals intentionally overlapping existing deltas.",
    )
    parser.add_argument("--plot", type=Path, help="Optional output path for latency plot.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    durations = asyncio.run(
        run_benchmark(
            iterations=args.iterations,
            existing_count=args.existing,
            proposals=args.proposals,
            duplicate_ratio=args.duplicate_ratio,
        )
    )

    print("Curator Throughput Benchmark")
    print("---------------------------")
    print(
        f"existing={args.existing}, proposals={args.proposals}, duplicate_ratio={args.duplicate_ratio:.2f}, "
        f"iterations={args.iterations}"
    )
    print(summarise(durations))

    if args.plot:
        subtitle = (
            f"existing={args.existing} | proposals={args.proposals} | duplicate_ratio={args.duplicate_ratio:.2f}"
        )
        maybe_plot(durations, args.plot.resolve(), subtitle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
