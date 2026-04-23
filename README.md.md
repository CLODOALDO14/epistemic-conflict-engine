# Epistemic Conflict Engine v3

**Philosophical, dialectical reasoning with LLaMA 3.1 — built for researchers, not chatbots.**

ECE v3 retrieves *contradictory* sources (thesis vs. antithesis), measures their ideological distance, and forces LLaMA 3.1 to synthesize a grounded answer instead of averaging opinions. It works locally with Ollama, stores knowledge in Neo4j, and connects directly to **your own Zotero library**.

> One-click Windows launcher included. No API keys are stored in the repo.

---

## Why this is different

- **Dialectical Attention**: scores pairs by `(relevance + contradiction + balance) × temporal decay`, not just cosine similarity
- **Human-in-the-loop**: pauses for structured material grounding before synthesis
- **Private by design**: your Zotero API key lives only in your local `.env`
- **Reproducible**: LangGraph checkpointing (SQLite/Postgres), full audit trail

---

## Quick Start (Windows, 3 minutes)

1. **Download** the latest release and unzip
2. Copy `.env.example` → `.env`
3. Double-click `start-local-ece.ps1` → "Run with PowerShell"

The script will:
- create a Python venv
- install dependencies
- pull `llama3.1` and `nomic-embed-text` via Ollama
- start Neo4j locally
- load demo corpus
- open interactive menu

Then choose:
```
1 → start → topic: "Prison reform" → year: 1975 → thread-id: demo1
```

---

## First-Time Setup

### Requirements
- Windows 10/11, PowerShell 5.1+
- Python 3.11+ (from python.org)
- 8GB RAM minimum

### Connect your Zotero (optional but recommended)

Each user connects their *own* library. Nothing is uploaded.

```bash
python setup_zotero.py
```

You’ll be asked for:
- `ZOTERO_LIBRARY_ID` – find at https://www.zotero.org/settings/keys (numeric User ID)
- `ZOTERO_API_KEY` – create a private key on the same page
- Or set `ZOTERO_LOCAL=true` to use Zotero 7 desktop (no key needed)

Test:
```bash
python zotero_connector.py list-collections
python zotero_connector.py preview --tag philosophy --limit 5
```

Import to Neo4j:
```bash
python zotero_connector.py import --topic "Heidegger" --tag philosophy --default-class critical --epoch-start 1920 --epoch-end 1976
```

---

## Usage

### Single-file mode (recommended)
```bash
python ECE_v3_ALL_IN_ONE.py start --topic "Welfare administration" --year 1975 --thread-id welfare-1975
# ... system pauses for grounding ...
python ECE_v3_ALL_IN_ONE.py resume --thread-id welfare-1975 --grounding-file examples/material_grounding.example.json
```

### With your Zotero data
1. Import: `python zotero_connector.py import --topic "YourTopic" --collection ABC123`
2. Run ECE as above – it will retrieve from your imported nodes

---

## Privacy & Security

- **No credentials in code.** `zotero_connector.py` reads only from environment variables
- `.env` is in `.gitignore` – never commit it
- `ZOTERO_LOCAL` mode uses local Zotero database, zero cloud traffic
- All inference runs locally via Ollama

---

## Project Structure
```
ECE_v3_ALL_IN_ONE.py      # full engine (retrieval + workflow)
zotero_connector.py       # Zotero → Neo4j importer
setup_zotero.py           # interactive credential setup
start-local-ece.ps1       # one-click launcher
requirements.txt
.env.example
examples/material_grounding.example.json
```

---

## Publishing to GitHub

This repo is ready to share. Your personal keys are not included.

---

## License

MIT – use freely, cite if you publish research.

Built for critical social science, philosophy, and anyone tired of consensus-biased LLMs.
