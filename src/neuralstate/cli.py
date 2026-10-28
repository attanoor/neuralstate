"""`neuralstate` command line: migrate, api, worker, consolidate, recall, demo."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path

import httpx
import typer

from neuralstate.config import get_settings

app = typer.Typer(help="NeuralState: a self-evolving memory layer for autonomous agents.", no_args_is_help=True)


def _logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _windows_loop() -> None:
    if sys.platform == "win32":  # asyncpg is happiest on the selector loop
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@app.command()
def migrate(revision: str = "head") -> None:
    """Create or upgrade the database schema."""
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    _windows_loop()
    command.upgrade(cfg, revision)
    typer.echo(f"database at {revision}")


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, log_level: str = "info") -> None:
    """Serve the REST API and dashboard."""
    import uvicorn

    _windows_loop()
    uvicorn.run("neuralstate.api.app:create_app", factory=True, host=host, port=port, reload=reload,
                log_level=log_level)


def _runtime():
    from neuralstate.db.session import make_engine, make_sessionmaker
    from neuralstate.providers.registry import build_providers

    settings = get_settings()
    engine = make_engine(settings)
    return settings, engine, make_sessionmaker(engine), build_providers(settings)


@app.command()
def worker(once: bool = typer.Option(False, help="Drain due jobs, then exit"), log_level: str = "info") -> None:
    """Run the background consolidation worker (run several for throughput)."""
    from neuralstate.worker.runner import run_worker

    _logging(log_level)

    async def main():
        settings, engine, sessions, providers = _runtime()
        try:
            await run_worker(engine, sessions, providers, settings, once=once)
        finally:
            await engine.dispose()

    _windows_loop()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        typer.echo("worker stopped")


@app.command()
def consolidate(agent: str, kind: str = "cycle", log_level: str = "warning") -> None:
    """Run one consolidation cycle for AGENT (id or name) now, and print its stats."""
    from neuralstate.core.queries import resolve_agent
    from neuralstate.db.session import session_scope
    from neuralstate.worker.runner import run_now

    _logging(log_level)

    async def main():
        settings, engine, sessions, providers = _runtime()
        try:
            async with session_scope(sessions) as s:
                agent_id = (await resolve_agent(s, agent)).id
            return await run_now(engine, sessions, providers, settings, agent_id, kind)
        finally:
            await engine.dispose()

    _windows_loop()
    typer.echo(json.dumps(asyncio.run(main()), indent=2, default=str))


@app.command()
def recall(agent: str, query: str, task_type: str = typer.Option(None), budget: int = 2000) -> None:
    """Print what AGENT would recall for QUERY."""
    from neuralstate.core.queries import resolve_agent
    from neuralstate.core.recall import recall as do_recall
    from neuralstate.db.session import session_scope

    async def main():
        settings, engine, sessions, providers = _runtime()
        try:
            async with session_scope(sessions) as s:
                agent_id = (await resolve_agent(s, agent)).id
                return await do_recall(s, providers.embedder, settings, agent_id=agent_id, query=query,
                                       task_type=task_type, budget_tokens=budget)
        finally:
            await engine.dispose()

    _windows_loop()
    result = asyncio.run(main())
    typer.echo(result["context"] or "(nothing relevant in memory)")


def _reachable(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


@app.command()
def demo(url: str = "http://127.0.0.1:8000", agent: str = typer.Option(None, help="Agent name (default: new)"),
         api_key: str = typer.Option(None, envvar="NS_API_KEY")) -> None:
    """Watch an agent learn a rule from repeated failures. Starts an API server if none is running."""
    from neuralstate import demo as scenario

    if not _reachable(url):
        import uvicorn

        from neuralstate.api.app import create_app

        port = httpx.URL(url).port or 8000
        typer.echo(f"no API at {url}; starting one in-process on port {port}")
        _windows_loop()
        server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(120):
            if _reachable(url):
                break
            time.sleep(0.5)
        else:
            raise typer.Exit("API did not start; is the database migrated? (neuralstate migrate)")
    scenario.run(url, api_key=api_key, agent_name=agent, say=typer.echo)


if __name__ == "__main__":
    app()
