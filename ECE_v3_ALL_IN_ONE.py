from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from neo4j import Driver, GraphDatabase


THESIS_CLASSES = ("state", "institutional")
ANTITHESIS_CLASSES = ("critical", "genealogical")
DEMO_DATASET_SLUG = "ece_demo_v1"
DEMO_CORPUS: list[dict[str, Any]] = [
    {
        "key": "prison_reform_state",
        "topic": "prison_reform",
        "author_class": "state",
        "epoch_start": 1968,
        "epoch_end": 1985,
        "text": (
            "Prison reform / Hapishane reformu: Interior ministry memoranda present prison centralization "
            "as rational administration. Standardized inspection, hygiene, labor training, and record-keeping "
            "are said to rehabilitate inmates and secure public order."
        ),
    },
    {
        "key": "prison_reform_genealogy",
        "topic": "prison_reform",
        "author_class": "genealogical",
        "epoch_start": 1968,
        "epoch_end": 1985,
        "text": (
            "Genealogical critique / Hapishane soybilimi: Prison reform expands surveillance and disciplinary "
            "normalization. Rehabilitation language masks new techniques for classifying, monitoring, and "
            "extracting compliant labor from poor and stigmatized populations."
        ),
    },
    {
        "key": "welfare_admin_institutional",
        "topic": "welfare_administration",
        "author_class": "institutional",
        "epoch_start": 1965,
        "epoch_end": 1982,
        "text": (
            "Welfare administration / Sosyal yardim idaresi: Central agencies describe targeted benefits, case "
            "files, and eligibility screening as efficient protection of families and labor-market stability."
        ),
    },
    {
        "key": "welfare_admin_critical",
        "topic": "welfare_administration",
        "author_class": "critical",
        "epoch_start": 1965,
        "epoch_end": 1982,
        "text": (
            "Critical social policy / Elestirel sosyal politika: Means-tested welfare links aid to behavioral "
            "monitoring and moral classification. The welfare office governs poverty through paperwork, sanctions, "
            "and norms of productive citizenship."
        ),
    },
    {
        "key": "public_health_state",
        "topic": "public_health",
        "author_class": "state",
        "epoch_start": 1890,
        "epoch_end": 1915,
        "text": (
            "Public health campaign / Halk sagligi kampanyasi: Municipal officials justify sanitation surveys, "
            "vaccination drives, and housing inspections as scientific prevention of epidemics and urban disorder."
        ),
    },
    {
        "key": "public_health_genealogy",
        "topic": "public_health",
        "author_class": "genealogical",
        "epoch_start": 1890,
        "epoch_end": 1915,
        "text": (
            "Genealogy of public health / Halk sagligi soybilimi: Sanitary expertise legitimizes intrusive "
            "inspection of working-class bodies and neighborhoods, translating inequality into medicalized "
            "administrative control."
        ),
    },
    {
        "key": "census_state",
        "topic": "census",
        "author_class": "institutional",
        "epoch_start": 1880,
        "epoch_end": 1910,
        "text": (
            "Census and legibility / Sayim ve okunabilirlik: State statisticians argue that standardized census "
            "categories make population management, taxation, and infrastructure planning more rational."
        ),
    },
    {
        "key": "census_critical",
        "topic": "census",
        "author_class": "critical",
        "epoch_start": 1880,
        "epoch_end": 1910,
        "text": (
            "Critique of census reason / Sayim aklinin elestirisi: Census categories simplify complex social "
            "identities so the state can render subjects governable, comparable, and allocable."
        ),
    },
    {
        "key": "factory_regime_institutional",
        "topic": "factory_discipline",
        "author_class": "institutional",
        "epoch_start": 1835,
        "epoch_end": 1855,
        "text": (
            "Factory discipline / Fabrika disiplini: Industrial reformers defend timekeeping, supervision, and "
            "workshop rules as necessary for productivity, skill formation, and orderly urban growth."
        ),
    },
    {
        "key": "factory_regime_critical",
        "topic": "factory_discipline",
        "author_class": "critical",
        "epoch_start": 1835,
        "epoch_end": 1855,
        "text": (
            "Labor discipline critique / Emek disiplini elestirisi: Factory regulation converts time into "
            "measurable obedience, subordinating bodies to industrial rhythms and capitalist command."
        ),
    },
]


def _load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class CandidateNode:
    node_id: str
    text: str
    author_class: str
    query_similarity: float
    epoch_start: int
    epoch_end: int
    year_distance: int
    embedding: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DialecticalPair:
    thesis: CandidateNode
    antithesis: CandidateNode
    relevance: float
    ideological_distance: float
    asymmetry: float
    temporal_decay: float
    raw_score: float
    normalized_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis": self.thesis.to_dict(),
            "antithesis": self.antithesis.to_dict(),
            "relevance": self.relevance,
            "ideological_distance": self.ideological_distance,
            "asymmetry": self.asymmetry,
            "temporal_decay": self.temporal_decay,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
        }


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator == 0:
        return 0.0
    raw = _dot(left, right) / denominator
    return max(-1.0, min(1.0, raw))


def _normalized_cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    normalized_similarity = (_cosine_similarity(left, right) + 1.0) / 2.0
    return 1.0 - normalized_similarity


def _stable_softmax(values: Sequence[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [value / temperature for value in values]
    anchor = max(scaled)
    exps = [math.exp(value - anchor) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def _require_env_vars(names: list[str]) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        vars_list = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {vars_list}")


class TemporalEpistemicDB:
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        *,
        database: str = "neo4j",
        vector_index: str = "discourse",
    ) -> None:
        self.database = database
        self.vector_index = vector_index
        self.driver: Driver = GraphDatabase.driver(uri, auth=(username, password))

    @classmethod
    def from_env(cls) -> "TemporalEpistemicDB":
        _require_env_vars(["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"])
        return cls(
            uri=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            vector_index=os.getenv("NEO4J_VECTOR_INDEX", "discourse"),
        )

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "TemporalEpistemicDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _fetch_candidates(
        self,
        *,
        query_embedding: Sequence[float],
        target_year: int,
        author_classes: Sequence[str],
        top_k: int,
        temporal_window: int,
        oversample_factor: int,
    ) -> list[CandidateNode]:
        candidate_k = max(top_k * oversample_factor, top_k)
        min_year = target_year - temporal_window
        max_year = target_year + temporal_window

        query = """
        CALL db.index.vector.queryNodes($index_name, $candidate_k, $embedding)
        YIELD node, score
        WHERE node.author_class IN $author_classes
        CALL (node) {
            MATCH (node)-[:EXISTED_IN]->(ep:Epoch)
            WHERE ep.start <= $max_year AND ep.end >= $min_year
            WITH ep,
                 CASE
                     WHEN $year < ep.start THEN ep.start - $year
                     WHEN $year > ep.end THEN $year - ep.end
                     ELSE 0
                 END AS year_distance
            ORDER BY year_distance ASC, ep.start ASC
            RETURN ep.start AS epoch_start, ep.end AS epoch_end, year_distance
            LIMIT 1
        }
        RETURN elementId(node) AS node_id,
               node.text AS text,
               node.author_class AS author_class,
               score AS query_similarity,
               epoch_start,
               epoch_end,
               year_distance,
               node.embedding AS embedding
        ORDER BY query_similarity DESC, year_distance ASC
        LIMIT $top_k
        """

        with self.driver.session(database=self.database) as session:
            rows = session.run(
                query,
                index_name=self.vector_index,
                candidate_k=candidate_k,
                embedding=list(query_embedding),
                author_classes=list(author_classes),
                year=target_year,
                min_year=min_year,
                max_year=max_year,
                top_k=top_k,
            ).data()

        candidates: list[CandidateNode] = []
        for row in rows:
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            candidates.append(
                CandidateNode(
                    node_id=row["node_id"],
                    text=row["text"],
                    author_class=row["author_class"],
                    query_similarity=float(row["query_similarity"]),
                    epoch_start=int(row["epoch_start"]),
                    epoch_end=int(row["epoch_end"]),
                    year_distance=int(row["year_distance"]),
                    embedding=[float(value) for value in embedding],
                )
            )
        return candidates

    def retrieve_dialectical_pairs(
        self,
        query_embedding: Sequence[float],
        target_year: int,
        *,
        top_k: int = 3,
        temporal_window: int = 15,
        oversample_factor: int = 8,
        alpha: float = 0.35,
        beta: float = 0.50,
        gamma: float = 0.15,
        lambda_decay: float = 0.08,
        temperature: float = 0.20,
    ) -> list[DialecticalPair]:
        thesis_candidates = self._fetch_candidates(
            query_embedding=query_embedding,
            target_year=target_year,
            author_classes=THESIS_CLASSES,
            top_k=top_k,
            temporal_window=temporal_window,
            oversample_factor=oversample_factor,
        )
        antithesis_candidates = self._fetch_candidates(
            query_embedding=query_embedding,
            target_year=target_year,
            author_classes=ANTITHESIS_CLASSES,
            top_k=top_k,
            temporal_window=temporal_window,
            oversample_factor=oversample_factor,
        )

        if not thesis_candidates:
            raise ValueError("No thesis candidates found for the configured time window and classes.")
        if not antithesis_candidates:
            raise ValueError("No antithesis candidates found for the configured time window and classes.")

        scored_pairs: list[dict[str, Any]] = []
        for thesis in thesis_candidates:
            for antithesis in antithesis_candidates:
                relevance = (thesis.query_similarity + antithesis.query_similarity) / 2.0
                ideological_distance = _normalized_cosine_distance(thesis.embedding, antithesis.embedding)
                asymmetry = abs(thesis.query_similarity - antithesis.query_similarity)
                alignment_balance = max(0.0, 1.0 - asymmetry)
                year_distance = (thesis.year_distance + antithesis.year_distance) / 2.0
                temporal_decay = math.exp(-lambda_decay * year_distance)
                raw_score = (
                    (alpha * relevance)
                    + (beta * ideological_distance)
                    + (gamma * alignment_balance)
                ) * temporal_decay
                scored_pairs.append(
                    {
                        "thesis": thesis,
                        "antithesis": antithesis,
                        "relevance": relevance,
                        "ideological_distance": ideological_distance,
                        "asymmetry": asymmetry,
                        "temporal_decay": temporal_decay,
                        "raw_score": raw_score,
                    }
                )

        normalized = _stable_softmax(
            [pair["raw_score"] for pair in scored_pairs],
            temperature=temperature,
        )

        pairs = [
            DialecticalPair(
                thesis=pair["thesis"],
                antithesis=pair["antithesis"],
                relevance=pair["relevance"],
                ideological_distance=pair["ideological_distance"],
                asymmetry=pair["asymmetry"],
                temporal_decay=pair["temporal_decay"],
                raw_score=pair["raw_score"],
                normalized_score=score,
            )
            for pair, score in zip(scored_pairs, normalized)
        ]
        pairs.sort(key=lambda item: item.normalized_score, reverse=True)
        return pairs[:top_k]


class EpistemicState(TypedDict, total=False):
    topic: str
    target_year: int
    enable_rebuttal: bool
    retrieved_pairs: list[dict[str, Any]]
    thesis: str
    antithesis: str
    rebuttal: str
    material_grounding: dict[str, Any]
    synthesis: str
    stage: str


def build_initial_state(
    *,
    topic: str,
    target_year: int,
    enable_rebuttal: bool = False,
    retrieved_pairs: list[DialecticalPair],
) -> EpistemicState:
    return {
        "topic": topic,
        "target_year": target_year,
        "enable_rebuttal": enable_rebuttal,
        "retrieved_pairs": [pair.to_dict() for pair in retrieved_pairs],
        "stage": "RETRIEVAL_COMPLETE",
    }


def _format_pairs(
    retrieved_pairs: list[dict[str, Any]],
    *,
    include_thesis: bool = True,
    include_antithesis: bool = True,
) -> str:
    lines: list[str] = []
    for index, pair in enumerate(retrieved_pairs, start=1):
        thesis = pair["thesis"]
        antithesis = pair["antithesis"]
        block = [
            f"Pair {index}",
            f"Dialectical score: {pair['normalized_score']:.4f}",
            f"Theory distance: {pair['ideological_distance']:.4f}",
        ]
        if include_thesis:
            block.append(
                f"Thesis source [{thesis['author_class']}, {thesis['epoch_start']}-{thesis['epoch_end']}]: {thesis['text']}"
            )
        if include_antithesis:
            block.append(
                f"Antithesis source [{antithesis['author_class']}, {antithesis['epoch_start']}-{antithesis['epoch_end']}]: {antithesis['text']}"
            )
        lines.append("\n".join(block))
    return "\n\n".join(lines)


def _invoke_text(model: ChatOllama, *, system_prompt: str, human_prompt: str) -> str:
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_chunks: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text_chunks.append(str(chunk.get("text", "")))
            else:
                text_chunks.append(str(chunk))
        return "\n".join(part for part in text_chunks if part).strip()
    return str(content).strip()


def _create_chat_model() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.0,
    )


def _create_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _quote_cypher_name(name: str) -> str:
    if not name or any(char in name for char in ("`", "\r", "\n")):
        raise ValueError(f"Invalid Cypher identifier: {name!r}")
    return f"`{name}`"


def _get_index_state(session, *, index_name: str) -> str | None:
    row = session.run(
        "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state LIMIT 1",
        name=index_name,
    ).single()
    if row is None:
        return None
    return str(row["state"])


def _count_ready_discourse_nodes(session) -> int:
    row = session.run(
        """
        MATCH (d:Discourse)-[:EXISTED_IN]->(:Epoch)
        WHERE d.author_class IN $author_classes
          AND d.embedding IS NOT NULL
        RETURN count(d) AS count
        """,
        author_classes=list(THESIS_CLASSES + ANTITHESIS_CLASSES),
    ).single()
    return int(row["count"])


def _ensure_vector_index(session, *, index_name: str, dimensions: int) -> None:
    quoted_index_name = _quote_cypher_name(index_name)
    session.run(
        f"""
        CREATE VECTOR INDEX {quoted_index_name} IF NOT EXISTS
        FOR (d:Discourse) ON (d.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: $dimensions,
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """,
        dimensions=dimensions,
    ).consume()

    deadline = time.time() + 60
    while time.time() < deadline:
        state = _get_index_state(session, index_name=index_name)
        if state == "ONLINE":
            return
        time.sleep(1.0)
    raise RuntimeError(f"Vector index {index_name!r} did not become ONLINE within 60 seconds.")


def _has_valid_grounding(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    for field in ("economic_data", "institutional_records", "citations"):
        value = payload.get(field)
        if value is None:
            return False
        if isinstance(value, list) and not value:
            return False
    return True


def compile_epistemic_graph(checkpointer):
    model = _create_chat_model()

    def thesis_agent(state: EpistemicState) -> EpistemicState:
        pair_context = _format_pairs(state["retrieved_pairs"], include_antithesis=False)
        system_prompt = (
            "You are an institutional discourse analyst. "
            "Reconstruct the governing thesis from the provided material. "
            "Do not critique it yet. Prioritize the evidence most semantically aligned with the user topic and treat weaker matches as secondary. "
            "Keep the answer concise, historically grounded, and explicit about evidential limits. "
            "Respond in Turkish."
        )
        human_prompt = (
            f"Topic: {state['topic']}\n"
            f"Target year: {state['target_year']}\n\n"
            "Use the thesis-side evidence below to reconstruct the strongest institutional narrative.\n\n"
            f"{pair_context}\n\n"
            "Focus on the highest-scoring topic-aligned evidence first. Output 2 short paragraphs. Avoid synthesis."
        )
        return {
            "thesis": _invoke_text(model, system_prompt=system_prompt, human_prompt=human_prompt),
            "stage": "THESIS_GENERATED",
        }

    def antithesis_agent(state: EpistemicState) -> EpistemicState:
        pair_context = _format_pairs(state["retrieved_pairs"], include_thesis=False)
        system_prompt = (
            "You are a genealogical critic in the style of critical social theory. "
            "Expose exclusions, disciplinary logics, and power effects. "
            "Prioritize the evidence most semantically aligned with the user topic and treat weaker matches as secondary. "
            "Keep the answer concise, historically grounded, and explicit about evidential limits. Respond in Turkish."
        )
        human_prompt = (
            f"Topic: {state['topic']}\n"
            f"Target year: {state['target_year']}\n\n"
            f"Thesis:\n{state['thesis']}\n\n"
            "Use the antithesis-side evidence below to generate a historically grounded critique.\n\n"
            f"{pair_context}\n\n"
            "Focus on the highest-scoring topic-aligned evidence first. Output 2 short paragraphs. Do not synthesize yet."
        )
        return {
            "antithesis": _invoke_text(model, system_prompt=system_prompt, human_prompt=human_prompt),
            "stage": "ANTITHESIS_GENERATED",
        }

    def rebuttal_agent(state: EpistemicState) -> EpistemicState:
        pair_context = _format_pairs(state["retrieved_pairs"], include_antithesis=False)
        system_prompt = (
            "You are an institutional discourse analyst defending the governing thesis. "
            "You must respond to the genealogical critique (antithesis) by either absorbing its points "
            "(hegemonic incorporation) or refuting its historical claims. "
            "Use only the retrieved thesis-side evidence and explicitly mark unsupported defenses as unresolved. "
            "Respond in Turkish."
        )
        human_prompt = (
            f"Topic: {state['topic']}\n"
            f"Target year: {state['target_year']}\n\n"
            f"Original Thesis:\n{state['thesis']}\n\n"
            f"Antithesis (Critique):\n{state['antithesis']}\n\n"
            "Use the thesis-side evidence below to support your defense if necessary.\n\n"
            f"{pair_context}\n\n"
            "Generate a robust rebuttal or theoretical absorption of the critique.\n"
            "Output 2 short paragraphs."
        )
        return {
            "rebuttal": _invoke_text(model, system_prompt=system_prompt, human_prompt=human_prompt),
            "stage": "REBUTTAL_GENERATED",
        }

    def material_grounding_agent(state: EpistemicState) -> EpistemicState:
        prompt: dict[str, Any] = {
            "instruction": "Inject structured material grounding before synthesis.",
            "topic": state["topic"],
            "target_year": state["target_year"],
            "required_fields": [
                "economic_data",
                "institutional_records",
                "citations",
            ],
            "guideline": "Unsupported claims must remain unresolved in the final synthesis.",
        }

        while True:
            grounding = interrupt(prompt)
            if _has_valid_grounding(grounding):
                return {
                    "material_grounding": grounding,
                    "stage": "MATERIAL_GROUNDED",
                }
            prompt = {
                "instruction": "Invalid grounding payload. Please provide a JSON object.",
                "required_fields": [
                    "economic_data",
                    "institutional_records",
                    "citations",
                ],
                "example_file": "examples/material_grounding.example.json",
            }

    def synthesis_agent(state: EpistemicState) -> EpistemicState:
        pair_context = _format_pairs(state["retrieved_pairs"])
        grounding = json.dumps(state["material_grounding"], ensure_ascii=False, indent=2)
        rebuttal_section = ""
        if state.get("rebuttal"):
            rebuttal_section = f"Rebuttal (Defense/Absorption):\n{state['rebuttal']}\n\n"
        system_prompt = (
            "You are a critical-historical synthesis engine. "
            "Produce a doctoral-level synthesis, but only when supported by the material grounding. "
            "If the evidence is insufficient, mark the point as unresolved. "
            "Retrieved discourse pairs define the argumentative field, but only the material grounding can support empirical claims. "
            "Do not reveal hidden reasoning; provide only the final analytical answer. Respond in Turkish."
        )
        human_prompt = (
            f"Topic: {state['topic']}\n"
            f"Target year: {state['target_year']}\n\n"
            f"Thesis:\n{state['thesis']}\n\n"
            f"Antithesis:\n{state['antithesis']}\n\n"
            f"{rebuttal_section}"
            f"Retrieved pair context:\n{pair_context}\n\n"
            f"Material grounding:\n{grounding}\n\n"
            "Focus on the highest-scoring topic-aligned pair or pairs. Use lower-scoring context only if it clearly supports the same topic.\n"
            "Do not refer to pair numbers in the final answer.\n\n"
            "Return three sections with headings:\n"
            "1. Maddi Dayanak\n"
            "2. Diyalektik Sentez\n"
            "3. Cozulemeyen Gerilimler\n\n"
            "Do not claim that a source proves more than it explicitly states. If support is indirect, say it is indirect."
        )
        return {
            "synthesis": _invoke_text(model, system_prompt=system_prompt, human_prompt=human_prompt),
            "stage": "FINAL_SYNTHESIS",
        }

    builder = StateGraph(EpistemicState)
    builder.add_node("generate_thesis", thesis_agent)
    builder.add_node("generate_antithesis", antithesis_agent)
    builder.add_node("generate_rebuttal", rebuttal_agent)
    builder.add_node("request_material_grounding", material_grounding_agent)
    builder.add_node("generate_synthesis", synthesis_agent)
    builder.add_edge(START, "generate_thesis")
    builder.add_edge("generate_thesis", "generate_antithesis")
    builder.add_conditional_edges(
        "generate_antithesis",
        lambda state: "generate_rebuttal" if state.get("enable_rebuttal") else "request_material_grounding",
        {
            "generate_rebuttal": "generate_rebuttal",
            "request_material_grounding": "request_material_grounding",
        },
    )
    builder.add_edge("generate_rebuttal", "request_material_grounding")
    builder.add_edge("request_material_grounding", "generate_synthesis")
    builder.add_edge("generate_synthesis", END)
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_checkpointer() -> Iterator[Any]:
    postgres_url = os.getenv("LANGGRAPH_POSTGRES_URL")
    if postgres_url:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(postgres_url) as checkpointer:
            checkpointer.setup()
            yield checkpointer
            return

    from langgraph.checkpoint.sqlite import SqliteSaver

    sqlite_path = Path(os.getenv("LANGGRAPH_SQLITE_PATH", "var/ece_checkpoints.db"))
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(sqlite_path)) as checkpointer:
        yield checkpointer


def _check_and_print_interrupt(result: dict[str, Any], graph, config, args) -> bool:
    python_cmd = f"& '{sys.executable}'"
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        print("Workflow paused for human material grounding.")
        for interrupt_obj in interrupts:
            payload = getattr(interrupt_obj, "value", interrupt_obj)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        thread_id = getattr(args, "thread_id", "<thread_id>")
        print(
            "\nResume this thread with: "
            f"{python_cmd} {os.path.basename(sys.argv[0])} resume --thread-id {thread_id} "
            "--grounding-file path/to/grounding.json"
        )
        return True

    state_snapshot = graph.get_state(config)
    if state_snapshot.next:
        print("Workflow paused for human material grounding.")
        for task in state_snapshot.tasks:
            for interrupt_obj in task.interrupts:
                payload = getattr(interrupt_obj, "value", interrupt_obj)
                print(json.dumps(payload, ensure_ascii=False, indent=2))
        thread_id = getattr(args, "thread_id", "<thread_id>")
        print(
            "\nResume this thread with: "
            f"{python_cmd} {os.path.basename(sys.argv[0])} resume --thread-id {thread_id} "
            "--grounding-file path/to/grounding.json"
        )
        return True
    return False


def _print_final_result(result: dict[str, Any]) -> None:
    if "synthesis" in result:
        print(result["synthesis"])
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def _smoke_compile_graph() -> None:
    thesis = CandidateNode("t1", "State thesis", "state", 0.9, 1970, 1980, 0, [0.1, 0.2, 0.3])
    antithesis = CandidateNode("a1", "Critical antithesis", "critical", 0.85, 1970, 1980, 0, [0.3, 0.2, 0.1])
    pair = DialecticalPair(
        thesis=thesis,
        antithesis=antithesis,
        relevance=0.875,
        ideological_distance=0.4,
        asymmetry=0.05,
        temporal_decay=1.0,
        raw_score=0.5,
        normalized_score=1.0,
    )
    initial_state = build_initial_state(
        topic="health check",
        target_year=1975,
        enable_rebuttal=False,
        retrieved_pairs=[pair],
    )
    with open_checkpointer() as checkpointer:
        graph = compile_epistemic_graph(checkpointer)
        if not graph or initial_state["stage"] != "RETRIEVAL_COMPLETE":
            raise RuntimeError("Graph compilation smoke check failed.")


def preflight() -> int:
    print("Preflight checks started.")
    checks: list[tuple[str, bool, str]] = []

    try:
        _require_env_vars(
            [
                "NEO4J_URI",
                "NEO4J_USERNAME",
                "NEO4J_PASSWORD",
                "OLLAMA_BASE_URL",
                "OLLAMA_MODEL",
                "OLLAMA_EMBEDDING_MODEL",
            ]
        )
        checks.append(("Environment variables", True, "OK"))
    except Exception as exc:
        checks.append(("Environment variables", False, str(exc)))

    try:
        _smoke_compile_graph()
        checks.append(("Graph/checkpointer compile", True, "OK"))
    except Exception as exc:
        checks.append(("Graph/checkpointer compile", False, str(exc)))

    try:
        _create_embedding_model().embed_query("ECE health check")
        checks.append(("Ollama embedding endpoint", True, "OK"))
    except Exception as exc:
        checks.append(("Ollama embedding endpoint", False, str(exc)))

    try:
        chat = _create_chat_model()
        chat.invoke("Reply with only: OK")
        checks.append(("Ollama chat endpoint", True, "OK"))
    except Exception as exc:
        checks.append(("Ollama chat endpoint", False, str(exc)))

    try:
        with TemporalEpistemicDB.from_env() as db:
            db.driver.verify_connectivity()
        checks.append(("Neo4j connectivity", True, "OK"))
    except Exception as exc:
        checks.append(("Neo4j connectivity", False, str(exc)))

    try:
        with TemporalEpistemicDB.from_env() as db:
            with db.driver.session(database=db.database) as session:
                state = _get_index_state(session, index_name=db.vector_index)
        if state == "ONLINE":
            checks.append(("Neo4j vector index", True, f"{db.vector_index} ONLINE"))
        else:
            checks.append(
                (
                    "Neo4j vector index",
                    False,
                    f"{db.vector_index} missing or not ONLINE. Run `python ECE_v3_ALL_IN_ONE.py bootstrap-demo`.",
                )
            )
    except Exception as exc:
        checks.append(("Neo4j vector index", False, str(exc)))

    try:
        with TemporalEpistemicDB.from_env() as db:
            with db.driver.session(database=db.database) as session:
                count = _count_ready_discourse_nodes(session)
        if count > 0:
            checks.append(("Discourse corpus", True, f"{count} retrievable discourse nodes"))
        else:
            checks.append(
                (
                    "Discourse corpus",
                    False,
                    "No retrievable discourse nodes found. Run `python ECE_v3_ALL_IN_ONE.py bootstrap-demo` or load your corpus.",
                )
            )
    except Exception as exc:
        checks.append(("Discourse corpus", False, str(exc)))

    failed = [item for item in checks if not item[1]]
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
    if failed:
        print(f"Preflight completed with {len(failed)} failure(s).")
        return 2
    print("Preflight completed successfully.")
    return 0


def bootstrap_demo_data(args: argparse.Namespace) -> None:
    _require_env_vars(
        [
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "OLLAMA_BASE_URL",
            "OLLAMA_EMBEDDING_MODEL",
        ]
    )

    embedding_model = _create_embedding_model()
    texts = [record["text"] for record in DEMO_CORPUS]
    embeddings = embedding_model.embed_documents(texts)
    if not embeddings or not embeddings[0]:
        raise RuntimeError("Embedding model returned an empty demo corpus embedding set.")
    dimensions = len(embeddings[0])

    demo_records: list[dict[str, Any]] = []
    for record, embedding in zip(DEMO_CORPUS, embeddings):
        enriched = dict(record)
        enriched["dataset"] = DEMO_DATASET_SLUG
        enriched["embedding"] = [float(value) for value in embedding]
        demo_records.append(enriched)

    with TemporalEpistemicDB.from_env() as db:
        with db.driver.session(database=db.database) as session:
            if args.reset:
                session.run(
                    "MATCH (d:Discourse {dataset: $dataset}) DETACH DELETE d",
                    dataset=DEMO_DATASET_SLUG,
                ).consume()
                session.run(
                    "MATCH (e:Epoch {dataset: $dataset}) DETACH DELETE e",
                    dataset=DEMO_DATASET_SLUG,
                ).consume()

            _ensure_vector_index(
                session,
                index_name=db.vector_index,
                dimensions=dimensions,
            )

            for record in demo_records:
                session.run(
                    """
                    MERGE (e:Epoch {
                        dataset: $dataset,
                        start: $epoch_start,
                        end: $epoch_end
                    })
                    MERGE (d:Discourse {
                        dataset: $dataset,
                        key: $key
                    })
                    SET d.text = $text,
                        d.topic = $topic,
                        d.author_class = $author_class,
                        d.embedding = $embedding
                    MERGE (d)-[:EXISTED_IN]->(e)
                    """,
                    **record,
                ).consume()

            discourse_count = session.run(
                "MATCH (d:Discourse {dataset: $dataset}) RETURN count(d) AS count",
                dataset=DEMO_DATASET_SLUG,
            ).single()["count"]
            epoch_count = session.run(
                "MATCH (e:Epoch {dataset: $dataset}) RETURN count(e) AS count",
                dataset=DEMO_DATASET_SLUG,
            ).single()["count"]

    print(f"Demo corpus ready: {discourse_count} discourse nodes, {epoch_count} epoch nodes.")
    print(f"Vector index: {os.getenv('NEO4J_VECTOR_INDEX', 'discourse')}")
    print("Suggested smoke test:")
    python_cmd = f"& '{sys.executable}'"
    print(
        f"{python_cmd} ECE_v3_ALL_IN_ONE.py start "
        "--topic \"prison reform\" --year 1975 --thread-id demo-prison-1975"
    )


def start_workflow(args: argparse.Namespace) -> None:
    _require_env_vars(
        [
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD",
            "OLLAMA_BASE_URL",
            "OLLAMA_EMBEDDING_MODEL",
        ]
    )
    query_text = args.query or args.topic
    query_embedding = _create_embedding_model().embed_query(query_text)

    with TemporalEpistemicDB.from_env() as db:
        try:
            pairs = db.retrieve_dialectical_pairs(
                query_embedding,
                args.year,
                top_k=args.top_k,
                temporal_window=args.temporal_window,
            )
        except Exception as exc:
            raise RuntimeError(
                "Dialectical retrieval failed. Ensure the vector index exists and the corpus contains "
                "matching Discourse/Epoch nodes for the requested year and author classes. "
                "For a sample corpus run `python ECE_v3_ALL_IN_ONE.py bootstrap-demo --reset`."
            ) from exc

    initial_state = build_initial_state(
        topic=args.topic,
        target_year=args.year,
        enable_rebuttal=args.enable_rebuttal,
        retrieved_pairs=pairs,
    )
    config = {"configurable": {"thread_id": args.thread_id}}
    with open_checkpointer() as checkpointer:
        graph = compile_epistemic_graph(checkpointer)
        result = graph.invoke(initial_state, config=config)

        if _check_and_print_interrupt(result, graph, config, args):
            return

    _print_final_result(result)


def resume_workflow(args: argparse.Namespace) -> None:
    grounding_file = Path(args.grounding_file)
    grounding_payload = json.loads(grounding_file.read_text(encoding="utf-8"))
    config = {"configurable": {"thread_id": args.thread_id}}
    with open_checkpointer() as checkpointer:
        graph = compile_epistemic_graph(checkpointer)
        result = graph.invoke(Command(resume=grounding_payload), config=config)

        if _check_and_print_interrupt(result, graph, config, args):
            return

    _print_final_result(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Epistemic Conflict Engine as a single file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a new research thread.")
    start.add_argument("--topic", required=True)
    start.add_argument("--year", type=int, required=True)
    start.add_argument("--thread-id", required=True)
    start.add_argument("--query", help="Override retrieval query text.")
    start.add_argument("--top-k", type=int, default=3)
    start.add_argument("--temporal-window", type=int, default=15)
    start.add_argument("--enable-rebuttal", action="store_true", help="Enable the optional institutional rebuttal stage.")

    resume = subparsers.add_parser("resume", help="Resume a paused thread with grounding data.")
    resume.add_argument("--thread-id", required=True)
    resume.add_argument("--grounding-file", required=True)

    bootstrap = subparsers.add_parser("bootstrap-demo", help="Load a small demo corpus and create the vector index.")
    bootstrap.add_argument(
        "--reset",
        action="store_true",
        help="Delete only the existing demo dataset before reloading it.",
    )

    subparsers.add_parser("preflight", help="Check local service readiness.")
    return parser.parse_args()


def main() -> None:
    _load_dotenv()
    args = _parse_args()
    try:
        if args.command == "start":
            start_workflow(args)
            return
        if args.command == "resume":
            resume_workflow(args)
            return
        if args.command == "bootstrap-demo":
            bootstrap_demo_data(args)
            return
        if args.command == "preflight":
            raise SystemExit(preflight())
        raise ValueError(f"Unsupported command: {args.command}")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
