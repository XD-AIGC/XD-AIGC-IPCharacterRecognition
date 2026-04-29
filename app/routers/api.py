from __future__ import annotations

import json
from typing import Annotated, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import DetectedFace, FaceCluster, FaceClusterMember, FaceEmbedding, Image
from app.services.face_index import FaceFaissIndex
from app.services.ingest import ingest_image

router = APIRouter(prefix="/api", tags=["album-face"])


def to_media_url(path: str) -> str:
    path = path.replace("\\", "/")
    if "/media/" in path:
        return "/media/" + path.split("/media/")[-1]
    if "data/media/" in path:
        return "/media/" + path.split("data/media/")[-1]
    return "/media/" + path.split("/")[-1]


@router.get("/health")
def health():
    return {"ok": True}


@router.post("/images/upload")
async def upload_image(
    file: Annotated[UploadFile, File(description="image file")],
    db: Session = Depends(get_db),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    return ingest_image(db, content, file.filename or "upload.png")


@router.get("/images/{image_id}/review-data")
def review_data(image_id: int, db: Session = Depends(get_db)):
    image_row = db.scalar(select(Image).where(Image.id == image_id))
    if not image_row:
        raise HTTPException(status_code=404, detail="image not found")

    face_rows = db.execute(
        select(DetectedFace)
        .where(DetectedFace.image_id == image_id)
        .order_by(DetectedFace.face_index.asc())
    ).scalars().all()

    faces = []
    for face in face_rows:
        suggested_clusters_raw = json.loads(face.suggested_clusters_json) if face.suggested_clusters_json else []

        suggested_clusters = []
        for c in suggested_clusters_raw:
            item = dict(c)
            if item.get("sample_crop_path") and not item.get("sample_crop_url"):
                item["sample_crop_url"] = to_media_url(item["sample_crop_path"])
            suggested_clusters.append(item)
        # suggested_clusters = json.loads(face.suggested_clusters_json) if face.suggested_clusters_json else []

        faces.append({
            "face_id": face.id,
            "face_index": face.face_index,
            "crop_url": to_media_url(face.crop_path),
            "detector_score": face.detector_score,
            "suggested_score": face.suggested_score,
            "bbox": [face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2],
            "raw_bbox": [face.raw_x1, face.raw_y1, face.raw_x2, face.raw_y2] if face.raw_x1 is not None else None,
            "expanded_bbox": [face.exp_x1, face.exp_y1, face.exp_x2, face.exp_y2] if face.exp_x1 is not None else None,
            "suggested_clusters": suggested_clusters,
            "review_status": face.review_status,
        })

    return {
        "ok": True,
        "image_id": image_row.id,
        "image_url": to_media_url(image_row.file_path),
        "width": image_row.width,
        "height": image_row.height,
        "review_status": image_row.review_status,
        "face_count": len(faces),
        "faces": faces,
        "image_candidates": json.loads(image_row.suggested_clusters_json) if image_row.suggested_clusters_json else [],
    }


class FaceLabelItem(BaseModel):
    face_id: int
    class_name: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None)
    skip: bool = Field(default=False)


class ReviewSubmitRequest(BaseModel):
    items: List[FaceLabelItem]


@router.post("/images/{image_id}/labels")
def submit_labels(image_id: int, body: ReviewSubmitRequest, db: Session = Depends(get_db)):
    image_row = db.scalar(select(Image).where(Image.id == image_id))
    if not image_row:
        raise HTTPException(status_code=404, detail="image not found")

    face_ids = [item.face_id for item in body.items]
    face_rows = db.execute(
        select(DetectedFace).where(DetectedFace.id.in_(face_ids))
    ).scalars().all()
    face_map = {f.id: f for f in face_rows}

    vectors_to_add = []

    for item in body.items:
        face = face_map.get(item.face_id)
        if not face or face.image_id != image_id:
            continue

        if item.skip:
            face.review_status = "skipped"
            continue

        target_cluster = None
        if item.cluster_id is not None:
            target_cluster = db.scalar(
                select(FaceCluster).where(FaceCluster.id == item.cluster_id)
            )

        class_name = (item.class_name or "").strip()
        if target_cluster is None and class_name:
            target_cluster = db.scalar(
                select(FaceCluster).where(FaceCluster.name == class_name)
            )
            if target_cluster is None:
                target_cluster = FaceCluster(name=class_name)
                db.add(target_cluster)
                db.commit()
                db.refresh(target_cluster)

        if target_cluster is None:
            raise HTTPException(
                status_code=400,
                detail=f"face_id={item.face_id} 缺少 class_name 或 cluster_id"
            )

        member = db.scalar(
            select(FaceClusterMember).where(
                FaceClusterMember.cluster_id == target_cluster.id,
                FaceClusterMember.face_id == face.id,
            )
        )
        if member is None:
            db.add(
                FaceClusterMember(
                    cluster_id=target_cluster.id,
                    face_id=face.id,
                    similarity_score=face.suggested_score,
                    source="user",
                    is_representative=False,
                )
            )

        face.final_cluster_id = target_cluster.id
        face.review_status = "done"

        emb = db.scalar(select(FaceEmbedding).where(FaceEmbedding.face_id == face.id))
        if emb and not emb.indexed:
            vectors_to_add.append((face.id, emb.vector_path))
            emb.indexed = True

    if vectors_to_add:
        first_path = vectors_to_add[0][1]
        first_vec = np.load(first_path).astype("float32")
        index = FaceFaissIndex(dim=int(first_vec.shape[0]))
        for face_id, vector_path in vectors_to_add:
            vec = np.load(vector_path).astype("float32")
            index.add(vec, face_id)
        index.save()

    image_row.review_status = "done"
    db.commit()
    return {"ok": True, "image_id": image_id, "saved_faces": len(vectors_to_add)}


@router.get("/clusters/search")
def search_clusters(q: str = "", db: Session = Depends(get_db)):
    q = q.strip()
    stmt = select(FaceCluster)
    if q:
        stmt = stmt.where(FaceCluster.name.like(f"%{q}%"))
    rows = db.execute(stmt.order_by(FaceCluster.name.asc()).limit(30)).scalars().all()
    return [{"id": r.id, "name": r.name or f"类_{r.id}"} for r in rows]


@router.get("/clusters")
def list_clusters(db: Session = Depends(get_db)):
    cluster_rows = db.execute(
        select(FaceCluster).order_by(FaceCluster.updated_at.desc(), FaceCluster.id.asc())
    ).scalars().all()

    results = []
    for cluster in cluster_rows:
        member_rows = db.execute(
            select(DetectedFace)
            .join(FaceClusterMember, FaceClusterMember.face_id == DetectedFace.id)
            .where(FaceClusterMember.cluster_id == cluster.id)
            .order_by(FaceClusterMember.created_at.asc())
        ).scalars().all()

        sample_faces = member_rows[:8]
        results.append({
            "cluster_id": cluster.id,
            "name": cluster.name or f"类_{cluster.id}",
            "face_count": len(member_rows),
            "samples": [
                {
                    "face_id": f.id,
                    "crop_url": to_media_url(f.crop_path),
                    "image_id": f.image_id,
                }
                for f in sample_faces
            ],
            "updated_at": cluster.updated_at.isoformat() if cluster.updated_at else None,
        })

    return {
        "ok": True,
        "count": len(results),
        "items": results,
    }


@router.get("/clusters/{cluster_id}")
def cluster_detail(cluster_id: int, db: Session = Depends(get_db)):
    cluster = db.scalar(select(FaceCluster).where(FaceCluster.id == cluster_id))
    if not cluster:
        raise HTTPException(status_code=404, detail="cluster not found")

    rows = db.execute(
        select(DetectedFace, Image, FaceClusterMember)
        .join(FaceClusterMember, FaceClusterMember.face_id == DetectedFace.id)
        .join(Image, Image.id == DetectedFace.image_id)
        .where(FaceClusterMember.cluster_id == cluster_id)
        .order_by(FaceClusterMember.created_at.asc())
    ).all()

    faces = []
    for face, image_row, member in rows:
        faces.append({
            "face_id": face.id,
            "image_id": image_row.id,
            "crop_url": to_media_url(face.crop_path),
            "image_url": to_media_url(image_row.file_path),
            "detector_score": face.detector_score,
            "similarity_score": member.similarity_score,
            "raw_bbox": [face.raw_x1, face.raw_y1, face.raw_x2, face.raw_y2] if face.raw_x1 is not None else None,
            "expanded_bbox": [face.exp_x1, face.exp_y1, face.exp_x2, face.exp_y2] if face.exp_x1 is not None else None,
            "review_status": face.review_status,
        })

    return {
        "ok": True,
        "cluster": {
            "cluster_id": cluster.id,
            "name": cluster.name or f"类_{cluster.id}",
            "face_count": len(faces),
            "updated_at": cluster.updated_at.isoformat() if cluster.updated_at else None,
        },
        "faces": faces,
    }
