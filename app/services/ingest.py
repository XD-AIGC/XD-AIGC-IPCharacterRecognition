from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import EMBED_MODEL_PATH, FACE_TOPK, FACE_SIM_THRESHOLD, IMAGE_DIR, VECTOR_DIR
from app.models import DetectedFace, FaceEmbedding, Image
from app.services.face_detector import detect_and_crop_faces
from app.services.face_embedder import LocalClipFaceEmbedder
from app.services.face_index import FaceFaissIndex
from app.services.face_matcher import (
    aggregate_image_candidates,
    face_candidates_from_neighbors,
    filter_clusters_above_threshold,
)

_embedder: LocalClipFaceEmbedder | None = None


def get_embedder() -> LocalClipFaceEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = LocalClipFaceEmbedder()
    return _embedder


def get_search_index(dim: int) -> FaceFaissIndex:
    # 不缓存，保证每次读取最新索引
    return FaceFaissIndex(dim=dim)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_image_file(content: bytes, original_name: str) -> str:
    suffix = Path(original_name).suffix.lower() or ".png"
    file_hash = sha256_bytes(content)
    save_path = IMAGE_DIR / f"{file_hash}{suffix}"
    save_path.write_bytes(content)
    return str(save_path)


def save_face_vector(face_id: int, vec: np.ndarray) -> str:
    path = VECTOR_DIR / f"face_{face_id}.npy"
    np.save(path, vec)
    return str(path)


def to_media_url(path: str) -> str:
    path = path.replace("\\", "/")
    if "/media/" in path:
        return "/media/" + path.split("/media/")[-1]
    if "data/media/" in path:
        return "/media/" + path.split("data/media/")[-1]
    return "/media/" + path.split("/")[-1]


def recompute_face_suggestions(db: Session, face_row: DetectedFace) -> dict:
    emb = db.scalar(select(FaceEmbedding).where(FaceEmbedding.face_id == face_row.id))
    if not emb:
        face_row.suggested_clusters_json = json.dumps([], ensure_ascii=False)
        face_row.suggested_cluster_id = None
        face_row.suggested_score = None
        db.flush()
        return {
            "face_id": face_row.id,
            "face_index": face_row.face_index,
            "bbox": [face_row.bbox_x1, face_row.bbox_y1, face_row.bbox_x2, face_row.bbox_y2],
            "raw_bbox": [face_row.raw_x1, face_row.raw_y1, face_row.raw_x2, face_row.raw_y2],
            "expanded_bbox": [face_row.exp_x1, face_row.exp_y1, face_row.exp_x2, face_row.exp_y2],
            "crop_path": face_row.crop_path,
            "detector_score": face_row.detector_score,
            "suggested_score": None,
            "suggested_clusters": [],
            "selected_cluster_id": None,
        }

    vec = np.load(emb.vector_path).astype("float32")
    index = get_search_index(dim=int(emb.dim))
    neighbors = index.search(vec, k=FACE_TOPK + 1)

    # 去掉自己，避免重复图/已入索引样本直接命中自身
    neighbors = [(fid, score) for fid, score in neighbors if fid != face_row.id][:FACE_TOPK]

    candidates = face_candidates_from_neighbors(db, neighbors)
    matched_clusters = filter_clusters_above_threshold(candidates, FACE_SIM_THRESHOLD)

    print("[DEBUG][RECOMPUTE] face_id =", face_row.id)
    print("[DEBUG][RECOMPUTE] FACE_SIM_THRESHOLD =", FACE_SIM_THRESHOLD)
    print("[DEBUG][RECOMPUTE] neighbors =", neighbors)
    print("[DEBUG][RECOMPUTE] candidates =", candidates)
    print("[DEBUG][RECOMPUTE] matched_clusters =", matched_clusters)

    best = matched_clusters[0] if matched_clusters else None

    face_row.suggested_clusters_json = json.dumps(matched_clusters, ensure_ascii=False)
    if best:
        face_row.suggested_cluster_id = best["cluster_id"]
        face_row.suggested_score = best["raw_best_score"]
    else:
        face_row.suggested_cluster_id = None
        face_row.suggested_score = None

    db.flush()

    return {
        "face_id": face_row.id,
        "face_index": face_row.face_index,
        "bbox": [face_row.bbox_x1, face_row.bbox_y1, face_row.bbox_x2, face_row.bbox_y2],
        "raw_bbox": [face_row.raw_x1, face_row.raw_y1, face_row.raw_x2, face_row.raw_y2],
        "expanded_bbox": [face_row.exp_x1, face_row.exp_y1, face_row.exp_x2, face_row.exp_y2],
        "crop_path": face_row.crop_path,
        "detector_score": face_row.detector_score,
        "suggested_score": face_row.suggested_score,
        "suggested_clusters": [
            {
                **c,
                "sample_crop_url": to_media_url(c["sample_crop_path"]) if c.get("sample_crop_path") else None,
            }
            for c in matched_clusters
        ],
        "selected_cluster_id": best["cluster_id"] if best else None,
    }


def recompute_image_suggestions(db: Session, image_row: Image) -> dict:
    face_rows = db.execute(
        select(DetectedFace)
        .where(DetectedFace.image_id == image_row.id)
        .order_by(DetectedFace.face_index.asc())
    ).scalars().all()

    if not face_rows:
        image_row.review_status = "no_faces"
        image_row.suggested_clusters_json = json.dumps([], ensure_ascii=False)
        db.commit()
        return {
            "ok": True,
            "image_id": image_row.id,
            "review_url": f"/review/{image_row.id}",
            "face_count": 0,
            "faces": [],
            "image_candidates": [],
        }

    face_payloads = [recompute_face_suggestions(db, face_row) for face_row in face_rows]
    image_candidates = aggregate_image_candidates(face_payloads)

    image_row.suggested_clusters_json = json.dumps(image_candidates, ensure_ascii=False)
    db.commit()

    return {
        "ok": True,
        "image_id": image_row.id,
        "review_url": f"/review/{image_row.id}",
        "face_count": len(face_payloads),
        "faces": face_payloads,
        "image_candidates": image_candidates,
    }


def ingest_image(db: Session, file_bytes: bytes, filename: str) -> dict:
    file_hash = sha256_bytes(file_bytes)
    existed = db.scalar(select(Image).where(Image.sha256 == file_hash))
    if existed:
        refreshed = recompute_image_suggestions(db, existed)
        refreshed["is_duplicate"] = True
        return refreshed

    image_path = save_image_file(file_bytes, filename)
    pil = PILImage.open(image_path)
    width, height = pil.size

    image_row = Image(
        sha256=file_hash,
        file_path=image_path,
        width=width,
        height=height,
        review_status="pending",
    )
    db.add(image_row)
    db.commit()
    db.refresh(image_row)

    crops = detect_and_crop_faces(image_path=image_path, image_id=image_row.id)

    if not crops:
        image_row.review_status = "no_faces"
        image_row.suggested_clusters_json = json.dumps([], ensure_ascii=False)
        db.commit()
        return {
            "ok": True,
            "image_id": image_row.id,
            "review_url": f"/review/{image_row.id}",
            "face_count": 0,
            "image_candidates": [],
        }

    embedder = get_embedder()
    face_payloads: List[dict] = []

    try:
        for idx, crop in enumerate(crops):
            face_row = DetectedFace(
                image_id=image_row.id,
                face_index=idx,
                bbox_x1=crop.expanded_bbox[0],
                bbox_y1=crop.expanded_bbox[1],
                bbox_x2=crop.expanded_bbox[2],
                bbox_y2=crop.expanded_bbox[3],
                raw_x1=crop.raw_bbox[0],
                raw_y1=crop.raw_bbox[1],
                raw_x2=crop.raw_bbox[2],
                raw_y2=crop.raw_bbox[3],
                exp_x1=crop.expanded_bbox[0],
                exp_y1=crop.expanded_bbox[1],
                exp_x2=crop.expanded_bbox[2],
                exp_y2=crop.expanded_bbox[3],
                detector_score=crop.score,
                crop_path=crop.crop_path,
                review_status="pending",
            )
            db.add(face_row)
            db.flush()
            db.refresh(face_row)

            vec = embedder.encode_path(crop.crop_path)
            vector_path = save_face_vector(face_row.id, vec)

            db.add(
                FaceEmbedding(
                    face_id=face_row.id,
                    dim=int(vec.shape[0]),
                    vector_path=vector_path,
                    model_path=EMBED_MODEL_PATH,
                    faiss_id=face_row.id,
                    indexed=False,
                )
            )
            db.flush()

            index = get_search_index(dim=int(vec.shape[0]))
            neighbors = index.search(vec, k=FACE_TOPK)

            candidates = face_candidates_from_neighbors(db, neighbors)
            matched_clusters = filter_clusters_above_threshold(candidates, FACE_SIM_THRESHOLD)

            print("[DEBUG] face_id =", face_row.id)
            print("[DEBUG] FACE_SIM_THRESHOLD =", FACE_SIM_THRESHOLD)
            print("[DEBUG] neighbors =", neighbors)
            print("[DEBUG] candidates =", candidates)
            print("[DEBUG] matched_clusters =", matched_clusters)

            best = matched_clusters[0] if matched_clusters else None

            face_row.suggested_clusters_json = json.dumps(matched_clusters, ensure_ascii=False)
            if best:
                face_row.suggested_cluster_id = best["cluster_id"]
                face_row.suggested_score = best["raw_best_score"]
            else:
                face_row.suggested_cluster_id = None
                face_row.suggested_score = None

            face_payloads.append(
                {
                    "face_id": face_row.id,
                    "face_index": face_row.face_index,
                    "bbox": [face_row.bbox_x1, face_row.bbox_y1, face_row.bbox_x2, face_row.bbox_y2],
                    "raw_bbox": [face_row.raw_x1, face_row.raw_y1, face_row.raw_x2, face_row.raw_y2],
                    "expanded_bbox": [face_row.exp_x1, face_row.exp_y1, face_row.exp_x2, face_row.exp_y2],
                    "crop_path": face_row.crop_path,
                    "detector_score": face_row.detector_score,
                    "suggested_score": face_row.suggested_score,
                    "suggested_clusters": [
                        {
                            **c,
                            "sample_crop_url": to_media_url(c["sample_crop_path"]) if c.get("sample_crop_path") else None,
                        }
                        for c in matched_clusters
                    ],
                    "selected_cluster_id": best["cluster_id"] if best else None,
                }
            )

        image_candidates = aggregate_image_candidates(face_payloads)
        image_row.suggested_clusters_json = json.dumps(image_candidates, ensure_ascii=False)
        db.commit()

    except Exception:
        db.rollback()
        image_row.review_status = "failed"
        db.commit()
        raise

    return {
        "ok": True,
        "is_duplicate": False,
        "image_id": image_row.id,
        "review_url": f"/review/{image_row.id}",
        "face_count": len(face_payloads),
        "faces": face_payloads,
        "image_candidates": image_candidates,
    }