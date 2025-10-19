# Offline & Online Adaptation

ACET supports two complementary workflows:

1. **Offline adaptation** – digest labelled transcripts or transcripts with ground-truth answers to bootstrap your context delta library.
2. **Online adaptation** – learn continuously from live user interactions.

This guide explains how the workflows operate and how to make the most of them.

## Offline Adaptation Pipeline

Offline adaptation runs batches of historical examples through the generator, reflector, and curator. It is ideal when you have synthetic or annotated datasets.

```python
stats = await engine.run_offline_adaptation(
    training_data=[
        {"query": "How do I reset my password?", "ground_truth": "Guide user to reset link."},
        {"query": "What is the refund policy?", "ground_truth": "30 days with receipt."},
    ],
    epochs=2,
)
```

Key behaviours:

- Each sample triggers a generation using the current active deltas.
- The reflection step compares the generation with the ground truth and proposes improvements.
- The curator scores, deduplicates, and stages accepted deltas.
- At the end of each epoch, `StorageBackend.activate_staged()` promotes staged deltas to `ACTIVE`.

This loop is intentionally deterministic once the random seed for reflection sampling is fixed. You can run it in CI to validate that new insight templates still pass curation thresholds.

## Online Adaptation Pipeline

Online adaptation is designed for real-time usage. It loads the most relevant deltas for the incoming query, injects them into the generator prompt, and then decides whether to reflect based on `ACETConfig.reflection_sample_rate`.

```python
result = await engine.run_online_adaptation(
    query="Customer says the shipment is delayed – what now?",
    ground_truth=None,                 # optional
    update_context=True,               # toggle reflection/curation
)

print(result["answer"])
print(result["created_deltas"])
```

Important outputs in the `result` dictionary:

- `answer` / `evidence` – the generator’s response.
- `injected_context` – the bullet list of deltas that made it into the prompt.
- `metadata["context_tokens"]` – tokens consumed by the context bullets.
- `created_deltas` – any new context deltas accepted by the curator.

If `update_context=False`, ACET still returns a generation but skips reflection+curation. This is useful when operating in evaluation mode.

## Controlling Reflection Frequency

Reflection is powerful but can be expensive. Tune `ACETConfig.reflection_sample_rate` to balance freshness and cost:

- Set to `0.0` to disable reflection entirely.
- Set to `1.0` to always reflect (recommended for offline fine-tuning data).
- Pick a value like `0.2` to reflect 20% of the time in production.

Because `_should_reflect()` draws from `random.random()`, you can seed Python’s RNG for deterministic output when testing.

## Monitoring

Track the following runtime metrics:

- **`reflections_ran`** – number of reflections executed.
- **`deltas_created` / `deltas_activated`** – how many insights enter the library.
- **`context_tokens`** – always watch prompt growth to avoid hitting model limits.

Structured logs from `structlog` include these counts, making it easy to ship them to your observability stack.

Continue to {doc}`integrations` to see how ACET plugs into LangChain and the bundled ReAct agent.
