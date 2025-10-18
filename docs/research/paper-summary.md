# Research Summary

ACT implements the **Agentic Context Engineering** approach described in the accompanying research paper. The high-level goals:

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

For more detail on implementation phases and long-term roadmap, refer to `planning.md` in the repository root.
