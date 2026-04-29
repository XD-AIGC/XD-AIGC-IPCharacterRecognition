from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import MEDIA_DIR, STATIC_DIR
from app.db import Base, engine
from app.routers.api import router as api_router
from app.routers.ui import router as ui_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Album Backend MVP - Face Review")
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(api_router)
app.include_router(ui_router)
