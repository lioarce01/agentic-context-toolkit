# Integrations

ACT exposes integration points so you can drop the toolkit into existing agent stacks.

## LangChain Memory

`acet.integrations.ACTMemory` implements LangChain’s `BaseMemory` API. Attach it to any LCEL chain to inject ACT context and automatically feed interaction transcripts back into the engine.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from acet.integrations import ACTMemory

memory = ACTMemory(
    engine=act_engine,
    context_key="act_context",
    input_keys=["input"],
    output_key="response",
    top_k=8,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Use the provided ACT context when responding."),
        ("system", "{act_context}"),
        ("user", "{input}"),
    ]
)

chain = prompt | ChatOpenAI(model="gpt-4o-mini")  # any LangChain-compatible LLM
```

Whenever the chain is called, ACTMemory:

1. Pulls the top-`k` relevant context deltas from the engine.
2. Formats them as bullet points (`acet.core.budget.TokenBudgetManager`).
3. Returns the formatted context via `memory_variables`.
4. After the chain finishes, ingests the user/assistant exchange for reflection+curation.

## ReAct Agent

`acet.agents.react.ReActAgent` provides a fully async ReAct loop that:

- Loads context deltas via `ACETEngine`.
- Calls the configured LLM provider.
- Executes tool calls (any async callable).
- Reflects on the final answer and persists useful deltas.

```python
agent = ReActAgent(
    engine=act_engine,
    llm=OpenAIProvider(model="gpt-4o"),
    tools=[
        Tool(
            name="search",
            description="Query the knowledge base.",
            coroutine=search_tool,
        )
    ],
    max_steps=5,
)

result = await agent.run("Find the escalation policy for delayed shipments.")
print(result["answer"])
```

The agent returns the final answer, metadata describing each reasoning step, and any newly curated deltas.

## Custom Integrations

- Implement the abstract interfaces in `acet.core.interfaces` to integrate bespoke generators, reflectors, curators, or storage layers.  
- Wrap ACT inside other orchestration frameworks (Haystack, LlamaIndex, semantic kernel) by modelling either `ACETEngine.run_online_adaptation` or `ACETEngine.ingest_interaction` as a callback step.

Refer to {doc}`../examples/index` for runnable scripts that showcase these integrations end-to-end.
