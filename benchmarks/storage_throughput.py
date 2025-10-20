"""Benchmark save/query throughput for ACET storage backends."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import uuid
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acet.core.models import ContextDelta, DeltaStatus  # noqa: E402
from acet.storage.memory import MemoryBackend  # noqa: E402
from acet.storage.sqlite import SQLiteBackend  # noqa: E402


def build_deltas(count: int) -> List[ContextDelta]:
    return [
        ContextDelta(
            id=str(uuid.uuid4()),
            topic=f"topic-{index % 5}",
            guideline=f"Guideline #{index}",
            conditions=[f"condition-{index % 3}"],
            evidence=[f"doc-{index}-{i}" for i in range(2)],
            tags=[f"tag-{index % 7}"],
            status=DeltaStatus.STAGED,
            confidence=0.7,
        )
        for index in range(count)
    ]


def create_backend(name: str, workdir: Path) -> Tuple[object, Callable[[], None]]:
    if name == "memory":
        return MemoryBackend(), lambda: None

    db_path = workdir / f"storage-bench-{uuid.uuid4().hex}.db"
    backend = SQLiteBackend(db_path=str(db_path))

    def cleanup() -> None:
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass

    return backend, cleanup


async def exercise_backend(backend: object, deltas: List[ContextDelta]) -> None:
    await backend.save_deltas(deltas)
    saved = await backend.query_deltas()
    assert len(saved) == len(deltas)

    await backend.activate_staged()
    active = await backend.query_deltas(status=DeltaStatus.ACTIVE)
    assert len(active) == len(deltas)


async def run_iterations(
    backend_name: str,
    iterations: int,
    deltas: List[ContextDelta],
    workdir: Path,
) -> List[float]:
    timings: List[float] = []
    for _ in range(iterations):
        backend, cleanup = create_backend(backend_name, workdir)
        start = perf_counter()
        try:
            await exercise_backend(backend, deltas)
        finally:
            cleanup()
        timings.append(perf_counter() - start)
    return timings


def summarise(samples: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.pstdev(samples),
        "min": min(samples),
        "max": max(samples),
    }


def format_summary(label: str, stats: Dict[str, float]) -> str:
    ms = {key: value * 1000 for key, value in stats.items()}
    return (
        f"{label:<8} mean={ms['mean']:.3f} ms  median={ms['median']:.3f} ms  "
        f"stdev={ms['stdev']:.3f} ms  min={ms['min']:.3f} ms  max={ms['max']:.3f} ms"
    )


def maybe_plot(
    results: Dict[str, List[float]],
    output_path: Path,
    subtitle: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc

    labels = list(results.keys())
    means = [statistics.mean(vals) * 1000 for vals in results.values()]
    errors = [statistics.pstdev(vals) * 1000 for vals in results.values()]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, means, yerr=errors, capsize=6, color=["#5B8FF9", "#61DDAA"])
    plt.ylabel("Latency (ms)")
    plt.title("Storage Save+Query Latency by Backend")
    if subtitle:
        plt.suptitle(subtitle, y=0.96, fontsize=9)
    plt.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    for bar, mean in zip(bars, means, strict=False):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{mean:.2f} ms", ha="center", va="bottom")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved throughput plot to {output_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark storage backend throughput.")
    parser.add_argument(
        "--backend",
        choices=["memory", "sqlite", "all"],
        default="all",
        help="Which backend to benchmark.",
    )
    parser.add_argument("--iterations", type=int, default=30, help="Benchmark iterations per backend.")
    parser.add_argument("--batch-size", type=int, default=300, help="Number of deltas saved per iteration.")
    parser.add_argument("--workdir", type=Path, default=Path("./benchmarks/artifacts"), help="Working directory.")
    parser.add_argument("--plot", type=Path, help="Optional output path for latency plot.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    deltas = build_deltas(args.batch_size)
    backend_names = ["memory", "sqlite"] if args.backend == "all" else [args.backend]

    results: Dict[str, List[float]] = {}
    for name in backend_names:
        print(f"Running {name} backend benchmark...")
        durations = asyncio.run(run_iterations(name, args.iterations, deltas, workdir))
        results[name] = durations

    print("\nStorage Throughput Benchmark")
    print("---------------------------")
    for name in backend_names:
        stats = summarise(results[name])
        print(format_summary(name, stats))

    if args.plot:
        subtitle = (
            f"batch_size={args.batch_size} | iterations={args.iterations} | backend={args.backend}"
        )
        maybe_plot(results, args.plot.resolve(), subtitle)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
