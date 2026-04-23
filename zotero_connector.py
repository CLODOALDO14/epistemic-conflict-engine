"""
Zotero → ECE Connector
======================
Fetches bibliographic items from a user's Zotero library (via the Web API
or a local Zotero 7+ instance) and converts them into Discourse nodes that
the Epistemic Conflict Engine can ingest into Neo4j.

Authentication
--------------
Each user supplies their **own** credentials via environment variables:

    ZOTERO_LIBRARY_ID   – numeric user or group ID (from zotero.org/settings/keys)
    ZOTERO_API_KEY      – personal API key          (same page)
    ZOTERO_LIBRARY_TYPE – "user" (default) or "group"

For local-only read access (Zotero 7+ desktop running):

    ZOTERO_LOCAL=true

No credentials are stored in this repository.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------------
# Lazy import helper – pyzotero is optional until actually used
# ---------------------------------------------------------------------------

def _get_pyzotero():
    """Import pyzotero lazily so the rest of ECE works without it."""
    try:
        from pyzotero import zotero
        return zotero
    except ImportError:
        print(
            "ERROR: pyzotero is not installed.\n"
            "  Install it with:  pip install pyzotero\n"
            "  Or add it to requirements.txt.",
            file=sys.stderr,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ZoteroSource:
    """A single bibliographic source extracted from Zotero."""
    key: str
    title: str
    abstract: str
    authors: list[str]
    date: str
    item_type: str
    tags: list[str]
    collections: list[str]
    url: str
    full_text: str  # note body or abstract fallback
    raw_data: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def combined_text(self) -> str:
        """Return the richest text representation available."""
        if self.full_text:
            return self.full_text
        if self.abstract:
            return self.abstract
        return self.title

    def to_discourse_record(
        self,
        *,
        topic: str,
        author_class: str,
        epoch_start: int,
        epoch_end: int,
        dataset: str = "zotero_import",
    ) -> dict[str, Any]:
        """Convert to a dict compatible with ECE's Neo4j Discourse node schema."""
        return {
            "key": f"zotero_{self.key}",
            "topic": topic,
            "author_class": author_class,
            "epoch_start": epoch_start,
            "epoch_end": epoch_end,
            "text": self.combined_text,
            "dataset": dataset,
            "zotero_item_key": self.key,
            "zotero_title": self.title,
            "zotero_authors": ", ".join(self.authors),
            "zotero_date": self.date,
            "zotero_url": self.url,
        }


# ---------------------------------------------------------------------------
# Year extraction helper
# ---------------------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"\b(1[4-9]\d{2}|20[0-2]\d)\b")


def _extract_year(date_str: str) -> int | None:
    """Try to pull a 4-digit year from a Zotero date string."""
    match = _YEAR_PATTERN.search(date_str or "")
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Zotero client wrapper
# ---------------------------------------------------------------------------

class ZoteroClient:
    """
    Thin wrapper around pyzotero that reads credentials exclusively from
    environment variables.  Never stores or logs API keys.
    """

    def __init__(self) -> None:
        zotero = _get_pyzotero()

        self.local_mode = os.getenv("ZOTERO_LOCAL", "").lower() in ("true", "1", "yes")

        if self.local_mode:
            # Local Zotero 7+ – no API key required
            library_id = os.getenv("ZOTERO_LIBRARY_ID", "0")
            self._client = zotero.Zotero(library_id, "user", local=True)
            return

        # Web API mode – require credentials from env
        library_id = os.getenv("ZOTERO_LIBRARY_ID")
        api_key = os.getenv("ZOTERO_API_KEY")
        library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")

        if not library_id or not api_key:
            print(
                "ERROR: Zotero Web API credentials missing.\n"
                "  Set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY in your .env file.\n"
                "  Get them from: https://www.zotero.org/settings/keys\n"
                "\n"
                "  Alternatively, set ZOTERO_LOCAL=true to use a running\n"
                "  Zotero 7+ desktop instance (read-only, no key needed).",
                file=sys.stderr,
            )
            raise SystemExit(1)

        if library_type not in ("user", "group"):
            print(
                f"ERROR: ZOTERO_LIBRARY_TYPE must be 'user' or 'group', got '{library_type}'.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        self._client = zotero.Zotero(library_id, library_type, api_key)

    # ── Fetch helpers ─────────────────────────────────────────────

    def fetch_items(
        self,
        *,
        collection: str | None = None,
        tag: str | None = None,
        item_type: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> list[ZoteroSource]:
        """
        Retrieve items from the Zotero library with optional filters.

        Parameters
        ----------
        collection : str, optional
            Zotero collection key to restrict results.
        tag : str, optional
            Only items with this tag.
        item_type : str, optional
            Zotero item type filter (e.g. "journalArticle", "book").
        query : str, optional
            Free-text search query.
        limit : int
            Maximum items to return (default 50).
        """
        kwargs: dict[str, Any] = {"limit": limit}
        if tag:
            kwargs["tag"] = tag
        if item_type:
            kwargs["itemType"] = item_type
        if query:
            kwargs["q"] = query

        if collection:
            raw_items = self._client.collection_items(collection, **kwargs)
        else:
            raw_items = self._client.top(**kwargs)

        sources: list[ZoteroSource] = []
        for item in raw_items:
            data = item.get("data", {})
            if data.get("itemType") in ("attachment", "note"):
                continue

            source = self._parse_item(data)
            sources.append(source)

        return sources

    def fetch_collection_list(self) -> list[dict[str, str]]:
        """Return a list of {key, name} for all collections."""
        collections = self._client.collections()
        return [
            {"key": c["data"]["key"], "name": c["data"]["name"]}
            for c in collections
        ]

    def fetch_item_notes(self, item_key: str) -> list[str]:
        """Fetch child notes of an item (for full-text extraction)."""
        children = self._client.children(item_key)
        notes: list[str] = []
        for child in children:
            data = child.get("data", {})
            if data.get("itemType") == "note":
                note_text = data.get("note", "")
                # Strip HTML tags from Zotero note markup
                cleaned = re.sub(r"<[^>]+>", "", note_text).strip()
                if cleaned:
                    notes.append(cleaned)
        return notes

    # ── Internal parsing ──────────────────────────────────────────

    def _parse_item(self, data: dict[str, Any]) -> ZoteroSource:
        """Parse a raw Zotero item dict into a ZoteroSource."""
        creators = data.get("creators", [])
        authors = []
        for creator in creators:
            first = creator.get("firstName", "")
            last = creator.get("lastName", "")
            name = creator.get("name", "")
            if name:
                authors.append(name)
            elif last:
                authors.append(f"{first} {last}".strip())

        return ZoteroSource(
            key=data.get("key", ""),
            title=data.get("title", "Untitled"),
            abstract=data.get("abstractNote", ""),
            authors=authors,
            date=data.get("date", ""),
            item_type=data.get("itemType", ""),
            tags=[t.get("tag", "") for t in data.get("tags", [])],
            collections=data.get("collections", []),
            url=data.get("url", ""),
            full_text="",  # populated separately via notes
            raw_data=data,
        )


# ---------------------------------------------------------------------------
# ECE Integration: import Zotero items into Neo4j as Discourse nodes
# ---------------------------------------------------------------------------

def import_zotero_to_neo4j(
    *,
    topic: str,
    author_class_map: dict[str, str] | None = None,
    default_author_class: str = "institutional",
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    collection: str | None = None,
    tag: str | None = None,
    item_type: str | None = None,
    query: str | None = None,
    limit: int = 50,
    dataset: str = "zotero_import",
    include_notes: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Fetch items from Zotero and (optionally) load them into Neo4j as
    Discourse nodes for the ECE pipeline.

    Parameters
    ----------
    topic : str
        ECE topic label for all imported items.
    author_class_map : dict, optional
        Map Zotero tags to ECE author classes.
        Example: {"thesis": "state", "kritik": "critical"}
    default_author_class : str
        Fallback author class when no tag matches.
    epoch_start / epoch_end : int, optional
        Override epoch bounds.  If None, extracted from item date.
    collection : str, optional
        Zotero collection key filter.
    tag : str, optional
        Zotero tag filter.
    item_type : str, optional
        Zotero item type filter.
    query : str, optional
        Free-text search.
    limit : int
        Max items.
    dataset : str
        Dataset slug for Neo4j.
    include_notes : bool
        If True, fetch child notes and use them as full_text.
    dry_run : bool
        If True, return records without writing to Neo4j.

    Returns
    -------
    list[dict]
        The discourse records that were (or would be) imported.
    """
    if author_class_map is None:
        author_class_map = {
            "thesis": "state",
            "tez": "state",
            "state": "state",
            "devlet": "state",
            "institutional": "institutional",
            "kurumsal": "institutional",
            "antithesis": "critical",
            "antitez": "critical",
            "critical": "critical",
            "elestirel": "critical",
            "genealogical": "genealogical",
            "soybilimsel": "genealogical",
        }

    client = ZoteroClient()
    items = client.fetch_items(
        collection=collection,
        tag=tag,
        item_type=item_type,
        query=query,
        limit=limit,
    )

    if not items:
        print("No items found in Zotero with the given filters.")
        return []

    # Enrich with notes
    if include_notes:
        for item in items:
            notes = client.fetch_item_notes(item.key)
            if notes:
                item.full_text = "\n\n".join(notes)

    # Convert to discourse records
    records: list[dict[str, Any]] = []
    for item in items:
        # Determine author_class from tags
        ac = default_author_class
        for item_tag in item.tags:
            tag_lower = item_tag.lower().strip()
            if tag_lower in author_class_map:
                ac = author_class_map[tag_lower]
                break

        # Determine epoch from date
        year = _extract_year(item.date)
        es = epoch_start if epoch_start is not None else (year - 5 if year else 1900)
        ee = epoch_end if epoch_end is not None else (year + 5 if year else 2000)

        record = item.to_discourse_record(
            topic=topic,
            author_class=ac,
            epoch_start=es,
            epoch_end=ee,
            dataset=dataset,
        )
        records.append(record)

    print(f"Prepared {len(records)} discourse records from Zotero.")

    if dry_run:
        print("Dry run – no data written to Neo4j.")
        for r in records:
            print(f"  [{r['author_class']}] {r['zotero_title'][:60]}...")
        return records

    # Write to Neo4j
    _write_records_to_neo4j(records, dataset=dataset)
    return records


def _write_records_to_neo4j(
    records: list[dict[str, Any]],
    *,
    dataset: str,
) -> None:
    """Embed texts and write Discourse+Epoch nodes to Neo4j."""
    # Import ECE dependencies here to keep this module loosely coupled
    import time

    from langchain_ollama import OllamaEmbeddings
    from neo4j import GraphDatabase

    def _require_env(name: str) -> str:
        val = os.getenv(name)
        if not val:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return val

    # Load .env if present
    _load_dotenv_if_exists()

    neo4j_uri = _require_env("NEO4J_URI")
    neo4j_user = _require_env("NEO4J_USERNAME")
    neo4j_pass = _require_env("NEO4J_PASSWORD")
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")
    vector_index = os.getenv("NEO4J_VECTOR_INDEX", "discourse")

    embedding_model = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    # Compute embeddings
    texts = [r["text"] for r in records]
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = embedding_model.embed_documents(texts)
    if not embeddings or not embeddings[0]:
        raise RuntimeError("Embedding model returned empty results.")

    dimensions = len(embeddings[0])

    for record, embedding in zip(records, embeddings):
        record["embedding"] = [float(v) for v in embedding]

    # Write to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
    try:
        with driver.session(database=neo4j_db) as session:
            # Ensure vector index
            quoted = f"`{vector_index}`"
            session.run(
                f"""
                CREATE VECTOR INDEX {quoted} IF NOT EXISTS
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

            # Wait for index
            deadline = time.time() + 60
            while time.time() < deadline:
                row = session.run(
                    "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state LIMIT 1",
                    name=vector_index,
                ).single()
                if row and str(row["state"]) == "ONLINE":
                    break
                time.sleep(1.0)

            # Upsert nodes
            for record in records:
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
                        d.embedding = $embedding,
                        d.zotero_item_key = $zotero_item_key,
                        d.zotero_title = $zotero_title,
                        d.zotero_authors = $zotero_authors,
                        d.zotero_date = $zotero_date,
                        d.zotero_url = $zotero_url
                    MERGE (d)-[:EXISTED_IN]->(e)
                    """,
                    **record,
                ).consume()

            count = session.run(
                "MATCH (d:Discourse {dataset: $dataset}) RETURN count(d) AS count",
                dataset=dataset,
            ).single()["count"]

        print(f"Zotero import complete: {count} discourse nodes in dataset '{dataset}'.")
    finally:
        driver.close()


def _load_dotenv_if_exists(dotenv_path: str = ".env") -> None:
    """Minimal .env loader (same logic as ECE main)."""
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


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Standalone CLI for Zotero import."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Import Zotero library items into ECE's Neo4j corpus."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── list-collections ──────────────────────────────────────────
    sub.add_parser("list-collections", help="List all Zotero collections.")

    # ── import ────────────────────────────────────────────────────
    imp = sub.add_parser("import", help="Import items into Neo4j.")
    imp.add_argument("--topic", required=True, help="ECE topic label.")
    imp.add_argument("--collection", help="Zotero collection key.")
    imp.add_argument("--tag", help="Filter by Zotero tag.")
    imp.add_argument("--item-type", help="Filter by item type (e.g. journalArticle).")
    imp.add_argument("--query", "-q", help="Free-text search.")
    imp.add_argument("--limit", type=int, default=50)
    imp.add_argument("--default-class", default="institutional",
                     choices=["state", "institutional", "critical", "genealogical"])
    imp.add_argument("--epoch-start", type=int, help="Override epoch start year.")
    imp.add_argument("--epoch-end", type=int, help="Override epoch end year.")
    imp.add_argument("--dataset", default="zotero_import", help="Neo4j dataset slug.")
    imp.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    imp.add_argument("--no-notes", action="store_true", help="Skip fetching child notes.")

    # ── preview ───────────────────────────────────────────────────
    prev = sub.add_parser("preview", help="Preview items without importing.")
    prev.add_argument("--collection", help="Zotero collection key.")
    prev.add_argument("--tag", help="Filter by Zotero tag.")
    prev.add_argument("--query", "-q", help="Free-text search.")
    prev.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    _load_dotenv_if_exists()

    if args.command == "list-collections":
        client = ZoteroClient()
        collections = client.fetch_collection_list()
        if not collections:
            print("No collections found.")
            return
        print(f"{'Key':<12} Name")
        print("-" * 50)
        for c in collections:
            print(f"{c['key']:<12} {c['name']}")
        return

    if args.command == "preview":
        client = ZoteroClient()
        items = client.fetch_items(
            collection=args.collection,
            tag=args.tag,
            query=args.query,
            limit=args.limit,
        )
        if not items:
            print("No items found.")
            return
        for i, item in enumerate(items, 1):
            year = _extract_year(item.date) or "?"
            authors = ", ".join(item.authors[:2]) or "Unknown"
            print(f"  {i}. [{year}] {item.title[:70]}")
            print(f"     Authors: {authors}")
            print(f"     Type: {item.item_type} | Tags: {', '.join(item.tags[:5])}")
            if item.abstract:
                print(f"     Abstract: {item.abstract[:120]}...")
            print()
        return

    if args.command == "import":
        import_zotero_to_neo4j(
            topic=args.topic,
            default_author_class=args.default_class,
            epoch_start=args.epoch_start,
            epoch_end=args.epoch_end,
            collection=args.collection,
            tag=args.tag,
            item_type=args.item_type,
            query=args.query,
            limit=args.limit,
            dataset=args.dataset,
            include_notes=not args.no_notes,
            dry_run=args.dry_run,
        )
        return


if __name__ == "__main__":
    _cli_main()
