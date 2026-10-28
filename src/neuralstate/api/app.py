from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from neuralstate import __version__
from neuralstate.api.state import AppState
from neuralstate.config import Settings, get_settings
from neuralstate.db.session import make_engine, make_sessionmaker
from neuralstate.providers.registry import Providers, build_providers


def create_app(settings: Settings | None = None, providers: Providers | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings)
        app.state.ns = AppState(settings=settings, engine=engine, sessions=make_sessionmaker(engine),
                                providers=providers or build_providers(settings))
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(title="NeuralState", version=__version__, lifespan=lifespan,
                  description="A self-evolving memory layer for autonomous agents.")

    from neuralstate.api.v1 import router as v1
    from neuralstate.dashboard.routes import HERE as UI_DIR, LoginRequired, router as ui

    app.include_router(v1)
    app.include_router(ui)
    app.mount("/ui/static", StaticFiles(directory=UI_DIR / "static"), name="ui-static")

    @app.exception_handler(LoginRequired)
    async def _login(request: Request, _exc: LoginRequired):
        return RedirectResponse("/ui/login", status_code=303)

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        state: AppState = app.state.ns
        async with state.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok", "version": __version__, "reasoner": state.providers.reasoner.name,
                "embedder": state.providers.embedder.name}

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/ui")

    return app
