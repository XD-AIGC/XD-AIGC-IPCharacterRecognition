from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "request": request,
            "image_id": None,
        },
    )


@router.get("/review/{image_id}", response_class=HTMLResponse)
def review_page(request: Request, image_id: int):
    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "request": request,
            "image_id": image_id,
        },
    )




@router.get("/classes", response_class=HTMLResponse)
def classes_page(request: Request):
    return templates.TemplateResponse(
        request,
        "classes.html",
        {},
    )


@router.get("/classes/{cluster_id}", response_class=HTMLResponse)
def class_detail_page(request: Request, cluster_id: int):
    return templates.TemplateResponse(
        request,
        "class_detail.html",
        {
            "cluster_id": cluster_id,
        },
    )
