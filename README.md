# Epistemic Conflict Engine v3.0

This workspace contains a production-oriented revision of the original ECE proposal. The goal is not only to keep the dialectical idea intact, but also to make the math, retrieval logic, and LangGraph orchestration internally consistent and runnable.

## What Was Corrected

- The original Dialectical Attention Mechanism used only the difference between query-to-thesis and query-to-antithesis similarities. That does not measure contradiction between the two documents. The revised score combines:
  - query relevance,
  - pairwise ideological distance,
  - thesis/antithesis asymmetry as a secondary signal,
  - temporal decay over a configurable time window.
- Temporal decay is now meaningful. In the original code, documents were already hard-filtered to the exact year, so `Delta t` collapsed to zero and the decay term had no effect.
- Neo4j retrieval now oversamples candidates before filtering by `author_class` and epoch. This avoids starving the candidate pool after ANN retrieval.
- The LangGraph HITL pause no longer relies on `interrupt_before`, which LangGraph documents position as a debugging breakpoint rather than the preferred human-in-the-loop mechanism. The workflow now uses `interrupt()` with a durable checkpointer.
- `MemorySaver` was replaced in the runnable setup with a durable checkpoint strategy:
  - SQLite for local workflows,
  - PostgreSQL when `LANGGRAPH_POSTGRES_URL` is configured.
- The state schema was fixed so fields are optional until produced. The original version required keys before they existed and used an additive reducer on `iteration` without a real loop.
- The synthesis step now uses thesis, antithesis, and structured material grounding together. The original snippet ignored thesis and antithesis during synthesis.
- The so-called "CoT" section has been reframed as an auditable pipeline. Production systems should not depend on hidden chain-of-thought; they should expose explicit intermediate artifacts and validation stages.

## Revised Dialectical Score

For a thesis-antithesis pair `(t, a)` and query `q`, the production score is:

```text
R(q, t, a) = (sim(q, t) + sim(q, a)) / 2
C(t, a)    = 1 - sim(t, a)
A(q, t, a) = abs(sim(q, t) - sim(q, a))
B(q, t, a) = 1 - A(q, t, a)
D(y, t, a) = exp(-lambda * ((delta_t + delta_a) / 2))

raw(t, a) = (alpha * R + beta * C + gamma * B) * D
S(t, a)   = softmax(raw(t, a) / tau)
```

Where:

- `R` keeps both documents relevant to the query.
- `C` measures ideological distance between thesis and antithesis directly.
- `A` is still computed for diagnostics.
- `B` rewards balanced query alignment instead of accidentally preferring one relevant and one weakly relevant document.
- `D` penalizes temporal drift within a configurable year window.
- `tau` controls how sharply the model prefers the top contradictions.

Recommended defaults:

- `alpha = 0.35`
- `beta = 0.50`
- `gamma = 0.15`
- `lambda = 0.08`
- `tau = 0.20`

## Revised Validation Pipeline

The original 5-stage protocol is now implemented as an auditable validation pipeline:

1. Candidate Retrieval
2. Temporal Gating
3. Ideological Partitioning and Pair Re-ranking
4. Human Material Grounding Interrupt
5. Constrained Synthesis

This avoids treating hidden reasoning as a system primitive. Each stage produces inspectable state.

## Workspace Layout

- `start-local-ece.ps1`: **single-click launcher** — sets up everything and opens an interactive menu.
- `FIXES.md`: explicit list of problems found in the original text/code and the corresponding fixes.
- `ece/retrieval.py`: time-aware contrastive Neo4j retrieval and pair scoring.
- `ece/workflow.py`: LangGraph workflow with dynamic HITL interrupt and durable checkpointing.
- `run_ece.py`: CLI entrypoint with `start` and `resume` commands.
- `ECE_v3_ALL_IN_ONE.py`: single-file version (retrieval + workflow + CLI in one script).
- `requirements.txt`: Python dependencies.
- `.env.example`: required environment variables.
- `examples/material_grounding.example.json`: sample grounding payload used at the HITL step.

## Single-Click Launcher

Right-click `start-local-ece.ps1` and select **Run with PowerShell**, or run from a terminal:

```powershell
.\start-local-ece.ps1
```

The script automatically:

1. Creates a Python virtual environment (`.venv`) and installs all pip dependencies
2. Checks Ollama, starts the service if needed, and pulls the required models (`llama3.1`, `nomic-embed-text`)
3. Starts Neo4j Community Edition from the local `var/` directory with the configured password
4. Loads the demo corpus (`bootstrap-demo --reset`)
5. Runs preflight checks
6. Opens an interactive menu with all ECE commands (start, resume, preflight, bootstrap, stop, exit)

Optional flags:

| Flag | Effect |
| --- | --- |
| `-SkipOllama` | Skip Ollama setup (if already running) |
| `-SkipNeo4j` | Skip Neo4j startup (if already running) |
| `-SkipBootstrap` | Skip demo corpus reload |
| `-NonInteractive` | Run setup only, exit without menu |

Example with flags:

```powershell
.\start-local-ece.ps1 -SkipOllama -SkipBootstrap
```

## Install

```powershell
pip install -r requirements.txt
```

Set environment variables from `.env.example`.

## Run

If you want a ready-to-test local demo corpus in Neo4j, bootstrap it first:

```powershell
python ECE_v3_ALL_IN_ONE.py bootstrap-demo --reset
```

Start a thread and pause for material grounding:

```powershell
python run_ece.py start --topic "Prison reform" --year 1975 --thread-id prison-1975
```

Resume the same thread with structured grounding:

```powershell
python run_ece.py resume --thread-id prison-1975 --grounding-file examples/material_grounding.example.json
```

Run service readiness checks:

```powershell
python run_ece.py preflight
```

Use the single-file package:

```powershell
python ECE_v3_ALL_IN_ONE.py preflight
python ECE_v3_ALL_IN_ONE.py start --topic "Prison reform" --year 1975 --thread-id prison-1975
python ECE_v3_ALL_IN_ONE.py resume --thread-id prison-1975 --grounding-file examples/material_grounding.example.json
```

Optional dialectical extension:

```powershell
python ECE_v3_ALL_IN_ONE.py start --topic "Prison reform" --year 1975 --thread-id prison-1975 --enable-rebuttal
```

## Production Notes

- SQLite checkpointing is suitable for local and single-node workflows.
- For multi-instance production deployments, set `LANGGRAPH_POSTGRES_URL` and use PostgreSQL-backed checkpoints.
- The retrieval layer uses Neo4j's `db.index.vector.queryNodes()` for backward compatibility with Neo4j 5.x. If you are on Neo4j 2026.01+, the newer `SEARCH` clause with filters can further improve pre-filtered retrieval.
- The workflow assumes discourse nodes store `text`, `embedding`, and `author_class`, and connect to an `Epoch` node through `(:Discourse)-[:EXISTED_IN]->(:Epoch)`.

## Implementation References

- LangGraph interrupts: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- Neo4j vector indexes: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes
- Neo4j vector functions: https://neo4j.com/docs/cypher-manual/current/functions/vector/
