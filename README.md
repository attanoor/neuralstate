# NeuralState

A persistent memory layer for autonomous agents that **reorganizes itself from experience**.

Agents write raw *episodes* (what they tried, what happened). In the background, a
consolidation worker "sleeps" on them: it extracts a knowledge graph, settles
contradictions, forgets what is stale, and â€” the core idea â€” turns repeated failures into
**heuristic rules** the agent recalls before it acts. Rules that stop helping retire on
their own.

```
agent â”€â”€record()â”€â”€â–º episodes â”€â”€â–º consolidation cycle (background worker)
                                   1. extract     facts, entities, relations + provenance
                                   2. reconcile   merge duplicates, settle contradictions
                                   3. synthesize  â‰¥5 similar failures â†’ validated rule
                                   4. decay       confidence decay, archive cold memories
agent â—„â”€â”€recall()â”€â”€ rules â†’ facts â†’ episodes, packed into a token budget
```

## Quick start

You need Postgres 16+ with the [pgvector](https://github.com/pgvector/pgvector) extension:
a role `neuralstate` (password `neuralstate`) owning databases `neuralstate` and
`neuralstate_test` (the latter for integration tests), each with `CREATE EXTENSION vector;`.
The quickest route is Docker:

```sh
docker run -d --name neuralstate-db -p 5432:5432 -e POSTGRES_USER=neuralstate \
  -e POSTGRES_PASSWORD=neuralstate -e POSTGRES_DB=neuralstate pgvector/pgvector:pg16
docker exec neuralstate-db createdb -U neuralstate neuralstate_test
```

On Ubuntu/WSL without Docker: `apt install postgresql-16 postgresql-16-pgvector`, then create
the role and databases above. The migration creates the extension if the role is allowed to.

```powershell
# 1. Python env (shown for Windows; use .venv/bin/ on Linux/macOS)
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env          # pick providers; see below

# 2. Schema, API + dashboard, worker
.\.venv\Scripts\neuralstate migrate
.\.venv\Scripts\neuralstate api        # http://127.0.0.1:8000/ui , API docs at /docs
.\.venv\Scripts\neuralstate worker     # separate terminal; run more for throughput

# 3. Watch an agent learn
.\.venv\Scripts\neuralstate demo
```

No API keys? Set `NS_LLM_PROVIDER=fake` (deterministic heuristics) and keep
`NS_EMBED_PROVIDER=fastembed` (local) to run everything offline.

> **WSL note:** connect to `127.0.0.1`, not `localhost` (which can resolve to `::1`, not
> forwarded). WSL shuts its VM down when no WSL process is running, which stops
> Postgres; keep a WSL terminal open or set `vmIdleTimeout=-1` under `[wsl2]` in
> `%UserProfile%\.wslconfig`.

## Using it from an agent

```python
from neuralstate.sdk import NeuralStateClient

ns = NeuralStateClient("http://127.0.0.1:8000", agent="support-bot")

memory = ns.recall("Refund a duplicated charge for acme", task_type="billing", budget_tokens=1500)
system_prompt = BASE_PROMPT + "\n<memory>\n" + memory["context"] + "\n</memory>"
# ... run the agent ...
ns.record(summary="Refund duplicated charge for acme", task_type="billing",
          outcome="failure", outcome_detail="payment provider rejected partial refund",
          content={"messages": transcript, "approach": "partial refund via API"},
          applied_rule_ids=memory["applied_rule_ids"])   # credit/blame the rules it used
```

* **Outcomes are the learning signal.** Record `success` / `failure` / `partial`, or record
  `unknown` and attach the outcome later with `ns.set_outcome(episode_id, ...)`.
* **`applied_rule_ids`** closes the loop: rules gain confidence when they help and are
  retired when they keep failing (`NS_RULE_ARCHIVE_*`).
* `content` is free-form JSON. LLM reasoners read all of it. The offline `fake` reasoner
  only reads structured hints: `facts: [{subject, predicate, object}]`, `entities`, `approach`.

See [examples/claude_agent.py](examples/claude_agent.py) for a full Claude-based agent.

## REST API (`/v1`, OpenAPI at `/docs`)

| | |
|---|---|
| `POST /agents` | create or fetch an agent by name |
| `POST /episodes` Â· `POST /episodes/{id}/outcome` | record work; attach outcomes later |
| `POST /recall` | rules â†’ facts â†’ episodes for a task, packed to `budget_tokens`, plus a ready `context` string |
| `GET /graph` Â· `GET /nodes/{id}` | knowledge graph; a node with its evidence episodes and change history |
| `GET /rules` Â· `PATCH /rules/{id}` | list rules; disable, re-enable or edit one |
| `GET /review` Â· `POST /review/{id}/resolve` | contradictions that were too close to call |
| `GET /events` Â· `POST /events/{id}/revert` | audit log of every memory mutation; undo any of them |
| `POST /consolidate` | queue a cycle, or `wait: true` to run it now and get stats |

Agent references accept an id or a name. Set `NS_API_KEY` to require `X-API-Key`.

## How the learning works

**Rules from failures.** Failures not yet covered by a rule are clustered by embedding
(average linkage, per task type). A cluster of `NS_MIN_RULE_EVIDENCE` (default 5) becomes
a draft rule â€” *when X, avoid Y, instead Z, because â€¦* â€” written from the failures and the
most similar successes. The draft must pass a second check (does it explain the failures
without contradicting a known success?) before it is stored. A draft close to an existing
rule reinforces it instead. One close to a retired or disabled rule is suppressed, since
memory already knows that rule didn't help or wasn't wanted. Rules are embedded by the
situations they came from, so a new task that resembles past failures recalls them.

**Contradictions.** New facts that conflict with existing ones (same subject and predicate,
or simply very similar embeddings) are judged by the reasoner: duplicate, contradiction or
consistent. Each side of a contradiction is scored on recency, evidence count and
confidence. A clear winner (margin â‰¥ `NS_CONTRADICTION_MARGIN`) supersedes the loser
automatically. Close calls are marked *disputed* (still recalled, flagged) and re-scored
every cycle, so they settle as evidence accumulates. An operator can also decide early from
the review queue. Nothing ever waits on a human.

**Forgetting.** Facts that aren't used or re-observed lose confidence after a grace period.
Cold, low-confidence ones are archived. Nothing is hard-deleted, and every change is in
the audit log with a before/after snapshot, so any of it can be reverted.

## Providers

| | LLM (`NS_LLM_PROVIDER`) | Embeddings (`NS_EMBED_PROVIDER`) |
|---|---|---|
| default | `anthropic`: Claude Opus 5.5, structured outputs, server-side refusal fallbacks | `fastembed`: local bge-small, 384-dim |
| also | `openai` (any `NS_OPENAI_BASE_URL`-compatible server), `ollama` | `openai`, `ollama` |
| offline | `fake`: deterministic heuristics | `fake`: feature hashing |

`NS_EMBED_DIM` is baked into the schema at `neuralstate migrate`. Switching embedding
models later means re-embedding, because vectors from different models aren't comparable.

## Development

```powershell
.\.venv\Scripts\python -m pytest                       # unit tests (no DB)
.\.venv\Scripts\python -m pytest -m integration        # needs Postgres; uses neuralstate_test, one schema per test
```

Layout: `consolidation/` (stages, reasoners, prompts, scoring), `core/` (ingest, recall,
queries, revert), `worker/` (Postgres job queue with `SKIP LOCKED`, per-agent advisory
locks), `api/`, `dashboard/` (Jinja + HTMX + Cytoscape), `sdk/`, `migrations/`.
