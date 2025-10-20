"""Standalone benchmark for delta retrieval and ranking throughput."""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

# Ensure the project root is importable when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acet.core.interfaces import EmbeddingProvider  # noqa: E402
from acet.core.models import ContextDelta, DeltaStatus  # noqa: E402
from acet.retrieval import DeltaRanker  # noqa: E402


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider to keep benchmark runs reproducible."""

    async def embed(self, text: str) -> List[float]:
        return self._vectorize(text)

    async def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vectorize(text) for text in texts]

    def similarity(self, emb1: Sequence[float], emb2: Sequence[float]) -> float:
        if not emb1 or not emb2:
            return 0.0
        dot = sum(a * b for a, b in zip(emb1, emb2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def _vectorize(text: str) -> List[float]:
        # Encode characters to small floats; keep length bounded for stability.
        return [float(ord(char) % 97) / 50.0 for char in text[:32]]


def build_deltas(count: int, seed: int) -> List[ContextDelta]:
    rng = random.Random(seed)
    deltas: List[ContextDelta] = []
    for index in range(count):
        rng.seed(seed + index)
        deltas.append(
            ContextDelta(
                topic=f"topic-{index % 5}",
                guideline=f"Guideline #{index}: respond to case {index}",
                conditions=[f"condition-{index % 3}"],
                evidence=[f"doc-{index}-{i}" for i in range(2)],
                tags=[f"tag-{index % 7}"],
                status=DeltaStatus.ACTIVE,
                usage_count=rng.randint(0, 50),
                recency=rng.random(),
                confidence=0.6 + (index % 4) * 0.05,
                risk_level=rng.choice(["low", "medium"]),
            )
        )
    return deltas


async def run_benchmark(
    deltas: List[ContextDelta],
    iterations: int,
    query: str,
    top_k: int,
) -> List[float]:
    embedding_provider = StubEmbeddingProvider()
    ranker = DeltaRanker(embedding_provider=embedding_provider)

    durations: List[float] = []
    for _ in range(iterations):
        # Reset embeddings to force a full scoring pass on each iteration.
        for delta in deltas:
            delta.embedding = None

        start = perf_counter()
        await ranker.rank(query, deltas, top_k=top_k)
        end = perf_counter()
        durations.append(end - start)
    return durations


def format_stats(samples: Sequence[float]) -> str:
    mean = statistics.mean(samples)
    median = statistics.median(samples)
    stdev = statistics.pstdev(samples)
    minimum = min(samples)
    maximum = max(samples)
    p95 = (
        statistics.quantiles(samples, n=100, method="inclusive")[94]
        if len(samples) >= 10
        else None
    )

    def ms(value: float) -> str:
        return f"{value * 1000:.3f} ms"

    lines = [
        f"iterations: {len(samples)}",
        f"mean:      {ms(mean)}",
        f"median:    {ms(median)}",
        f"stdev:     {ms(stdev)}",
        f"min:       {ms(minimum)}",
        f"max:       {ms(maximum)}",
    ]
    if p95 is not None:
        lines.append(f"p95:       {ms(p95)}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark delta retrieval latency.")
    parser.add_argument("--deltas", type=int, default=250, help="Number of active deltas to rank.")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations to run.")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K deltas to return.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used when constructing synthetic deltas.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="How should I handle refund escalations for enterprise customers?",
        help="Benchmark query string.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        metavar="PATH",
        help="Optional path to write a latency plot (requires matplotlib).",
    )
    return parser.parse_args(argv)


def _maybe_plot(
    durations: Sequence[float],
    output_path: Path,
    subtitle: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from exc

    plt.figure(figsize=(8, 4.5))
    times_ms = [d * 1000 for d in durations]
    plt.plot(range(1, len(times_ms) + 1), times_ms, marker="o", linestyle="-", linewidth=1)
    plt.title("Delta Retrieval Latency per Iteration")
    if subtitle:
        plt.suptitle(subtitle, y=0.96, fontsize=9)
    plt.xlabel("Iteration")
    plt.ylabel("Latency (ms)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved latency plot to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    deltas = build_deltas(args.deltas, seed=args.seed)

    durations = asyncio.run(
        run_benchmark(
            deltas=deltas,
            iterations=args.iterations,
            query=args.query,
            top_k=args.top_k,
        )
    )

    print("Delta Retrieval Benchmark")
    print("-------------------------")
    print(f"deltas:    {args.deltas}")
    print(f"top_k:     {args.top_k}")
    print(format_stats(durations))

    if args.plot:
        output = Path(args.plot).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        subtitle = (
            f"active deltas={args.deltas} | top-k={args.top_k} | iterations={args.iterations}"
        )
        _maybe_plot(durations, output, subtitle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
