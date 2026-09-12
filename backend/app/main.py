from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings

from app.services.retrieval import (
    get_retrieval_service,
)

from app.services.generation import (
    get_generation_service,
)

from app.utils.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting NeuroAtlas backend..."
    )

    logger.info(
        "Loading retrieval service..."
    )

    get_retrieval_service()

    logger.info(
        "Loading generation service..."
    )

    get_generation_service()

    logger.info(
        "Startup complete."
    )

    yield

    logger.info(
        "Shutting down NeuroAtlas backend."
    )


app = FastAPI(
    title="NeuroAtlas RAG API",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(query_router)