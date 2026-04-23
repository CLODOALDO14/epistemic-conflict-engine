# Problems Found and How They Were Fixed

## Conceptual Issues

| Area | Original problem | Fix |
| --- | --- | --- |
| RLHF claim | The text treated consensus bias as an absolute incapacity. | Reframed as a tendency that matters for critical-social-science use cases, not an ontological impossibility. |
| "Chain of Thought" | The protocol was described as CoT, which suggests hidden reasoning as a controllable production artifact. | Reframed as an explicit validation pipeline with inspectable state transitions. |
| Bibliography | "Attention Is All You Need (Re-evaluated for contrastive weighting)" is not an accurate description of the paper. | Removed the misleading reinterpretation and grounded implementation claims in actual retrieval/orchestration docs. |

## Mathematical Issues

| Area | Original problem | Fix |
| --- | --- | --- |
| Contradiction Gap | `abs(Cos(q, vt) - Cos(q, va))` measures asymmetry relative to the query, not contradiction between thesis and antithesis. | Added direct pairwise ideological distance `1 - sim(t, a)` and kept asymmetry only as a secondary feature. |
| Temporal decay | The code filtered to the exact epoch, making `Delta t` effectively zero. | Added a configurable temporal window and computed year distance relative to the query year. |
| Pair set `P` | The formula referred to thesis-antithesis pairs without specifying how they were formed. | Implemented explicit cross-pairing between institutional and critical candidate pools. |
| Softmax stability | The formula was mathematically fine but implementation details were absent. | Added stable softmax in Python to avoid overflow. |

## Neo4j Retrieval Issues

| Area | Original problem | Fix |
| --- | --- | --- |
| Candidate starvation | Filtering by `author_class` and epoch after ANN retrieval can return too few usable documents. | Added oversampling before filtering and then trimmed to the requested `top_k`. |
| Missing pair features | The query returned only text and score, which is insufficient for pairwise contradiction scoring. | Returned `embedding`, `author_class`, epoch bounds, and temporal distance. |
| Temporal ambiguity | `year` was used as an exact gate without a production fallback window. | Added `temporal_window` support and year-distance calculation. |
| Durability | The snippet omitted connection cleanup and environment-based configuration. | Added `from_env()`, `close()`, and context-manager support. |

## LangGraph Issues

| Area | Original problem | Fix |
| --- | --- | --- |
| Model integration | `langchain_community.llms.Ollama` is not the best current choice for structured chat workflows. | Switched to `langchain_ollama.ChatOllama`. |
| HITL pattern | `interrupt_before=["Synthesis"]` is a static breakpoint, not the preferred production HITL design. | Implemented a dedicated node that calls `interrupt()` and resumes with `Command(resume=...)`. |
| Checkpointer | `MemorySaver` is not durable. | Added SQLite default and PostgreSQL option. |
| State schema | Required fields were absent at graph start; `iteration` reducer was unnecessary. | Made state partial and stage-driven. |
| Synthesis prompt | It ignored thesis and antithesis entirely. | The synthesis prompt now takes all three: thesis, antithesis, and material grounding. |
| Validation | Human grounding was free-form text. | Grounding is now a structured JSON payload with validation and reprompting. |

## Output Quality Issues

| Area | Original problem | Fix |
| --- | --- | --- |
| Auditability | The original flow did not expose a structured artifact trail. | The revised workflow keeps explicit retrieval pairs, stage labels, and grounding payloads in state. |
| Hallucination control | The synthesis agent could improvise beyond the material evidence. | The revised prompt explicitly marks unsupported claims as unresolved rather than fabricating closure. |
