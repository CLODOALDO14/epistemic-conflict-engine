from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langgraph.types import Command

from ece.retrieval import TemporalEpistemicDB
from ece.workflow import build_initial_state, compile_epistemic_graph, open_checkpointer


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Epistemic Conflict Engine workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a new research thread.")
    start.add_argument("--topic", required=True)
    start.add_argument("--year", type=int, required=True)
    start.add_argument("--thread-id", required=True)
    start.add_argument("--query", help="Override the retrieval query text. Defaults to topic.")
    start.add_argument("--top-k", type=int, default=3)
    start.add_argument("--temporal-window", type=int, default=15)

    resume = subparsers.add_parser("resume", help="Resume a paused thread with grounding data.")
    resume.add_argument("--thread-id", required=True)
    resume.add_argument("--grounding-file", required=True)

    subparsers.add_parser("preflight", help="Check local service readiness for real execution.")

    return parser.parse_args()


def _embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _print_interrupt_payload(result: dict[str, Any]) -> None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return
    print("Workflow paused for human material grounding.")
    for item in interrupts:
        payload = getattr(item, "value", item)
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_final_result(result: dict[str, Any]) -> None:
    if "synthesis" in result:
        print(result["synthesis"])
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def _require_env_vars(names: list[str]) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        vars_list = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {vars_list}")


def _smoke_compile_graph() -> None:
    from ece.retrieval import CandidateNode, DialecticalPair

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
    initial_state = build_initial_state(topic="health check", target_year=1975, retrieved_pairs=[pair])
    with open_checkpointer() as checkpointer:
        graph = compile_epistemic_graph(checkpointer)
        if not graph or initial_state["stage"] != "RETRIEVAL_COMPLETE":
            raise RuntimeError("Graph compilation smoke check failed.")


def _preflight() -> int:
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
        _embedding_model().embed_query("ECE health check")
        checks.append(("Ollama embedding endpoint", True, "OK"))
    except Exception as exc:
        checks.append(("Ollama embedding endpoint", False, str(exc)))

    try:
        chat = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,
        )
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

    failed = [item for item in checks if not item[1]]
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
    if failed:
        print(f"Preflight completed with {len(failed)} failure(s).")
        return 2
    print("Preflight completed successfully.")
    return 0


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
    query_embedding = _embedding_model().embed_query(query_text)

    with TemporalEpistemicDB.from_env() as db:
        pairs = db.retrieve_dialectical_pairs(
            query_embedding,
            args.year,
            top_k=args.top_k,
            temporal_window=args.temporal_window,
        )

    initial_state = build_initial_state(
        topic=args.topic,
        target_year=args.year,
        retrieved_pairs=pairs,
    )

    config = {"configurable": {"thread_id": args.thread_id}}
    with open_checkpointer() as checkpointer:
        graph = compile_epistemic_graph(checkpointer)
        result = graph.invoke(initial_state, config=config)

    if "__interrupt__" in result:
        _print_interrupt_payload(result)
        python_cmd = f"& '{sys.executable}'"
        print(
            "Resume this thread with: "
            f"{python_cmd} run_ece.py resume --thread-id {args.thread_id} "
            "--grounding-file path/to/grounding.json"
        )
        return

    _print_final_result(result)


def resume_workflow(args: argparse.Namespace) -> None:
    grounding_file = Path(args.grounding_file)
    grounding_payload = json.loads(grounding_file.read_text(encoding="utf-8"))
    config = {"configurable": {"thread_id": args.thread_id}}

    with open_checkpointer() as checkpointer:
        graph = compile_epistemic_graph(checkpointer)
        result = graph.invoke(Command(resume=grounding_payload), config=config)

    if "__interrupt__" in result:
        _print_interrupt_payload(result)
        return

    _print_final_result(result)


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
        if args.command == "preflight":
            raise SystemExit(_preflight())
        raise ValueError(f"Unsupported command: {args.command}")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
