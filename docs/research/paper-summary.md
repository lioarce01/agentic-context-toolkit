# Research Summary

ACET implements the **Agentic Context Engineering** approach described in the accompanying research paper. The high-level goals:

- Capture reusable context deltas (insights, heuristics, policies) from LLM interactions.
- Score and rank deltas based on similarity, recency, usage, and risk.
- Inject curated deltas back into future prompts to improve downstream responses.

## Methodology

1. **Generation** – Produce an answer with current context injected.
2. **Reflection** – Analyse the response, compare against ground truth (if available), and propose potential improvements.
3. **Curation** – Score/deduplicate the proposed insights to determine which become actionable context deltas.
4. **Activation** – Promote staged deltas to `ACTIVE`, making them eligible for retrieval in future requests.

The loop repeats continuously, allowing applications to adapt in near real time without retraining the underlying language model.

## Key Differentiators

- **Provider agnostic** – Works with any LLM that satisfies the `BaseLLMProvider` interface.
- **Storage agnostic** – Supports in-memory, SQLite, Postgres/pgvector, or custom backends.
- **Token-aware** – `TokenBudgetManager` ensures we never overrun prompt windows.
- **Observability-first** – Structured logging and configurable thresholds make it easy to monitor drift.

For continued progress tracking and future roadmap updates, follow the repository changelog and release notes.

## Performance Baseline

To track runtime guarantees we provide a reproducible delta-retrieval benchmark that exercises the core ranking loop. The benchmark suite lives in two places:

- `tests/benchmarks/test_delta_retrieval.py` &mdash; integrates with `pytest-benchmark` and runs in CI (skipped automatically unless the benchmark plugin is available). It asserts mean latency stays below the 100&nbsp;ms budget defined in the roadmap.
- `benchmarks/delta_retrieval.py` &mdash; a standalone CLI that emits detailed statistics (mean, median, stdev, p95) for manual profiling or CI artifacts.

Run the CLI benchmark from an activated virtual environment:

```bash
python benchmarks/delta_retrieval.py --deltas 250 --top-k 25 --iterations 30
```

On a typical development workstation we observe results similar to:

```
Delta Retrieval Benchmark
-------------------------
deltas:    250
top_k:     25
iterations: 30
mean:      1.732 ms
median:    1.739 ms
stdev:     0.050 ms
min:       1.641 ms
max:       1.842 ms
p95:       1.816 ms
```

Use these numbers as the baseline when optimizing alternative embedding providers, storage backends, or ranker configurations.

For visual analysis, pass a file path to `--plot` (requires `matplotlib`):

```bash
python benchmarks/delta_retrieval.py --iterations 50 --plot benchmarks/artifacts/delta_latency.png
```

The script will generate a line chart of per-iteration latency to help spot variance or regressions at a glance.

### Storage Throughput

To validate persistence overhead, use the storage throughput benchmark:

```bash
python benchmarks/storage_throughput.py --backend all --batch-size 300 --iterations 30 --plot benchmarks/artifacts/storage_latency.png
```

This command measures `MemoryBackend` and `SQLiteBackend` save/query cycles, prints per-backend latency summaries, and (optionally) writes a bar chart comparing mean latency with standard deviation error bars. Tune `--backend` to focus on a single implementation or adjust `--batch-size` to simulate larger workloads.

### Curator & Deduplication

Measure curator performance when processing large batches of proposed insights (with configurable duplicate ratios):

```bash
python benchmarks/curator_throughput.py --proposals 300 --duplicate-ratio 0.3 --iterations 20 --plot benchmarks/artifacts/curator_latency.png
```

The CLI prints aggregate statistics (mean/median/stdev/p95) and can emit a per-iteration latency plot. This helps track how deduplication thresholds or embedding strategies impact curation cost as reflection output scales.
