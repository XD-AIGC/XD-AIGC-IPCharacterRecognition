from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import FACE_MARGIN, FACE_MIN_SCORE, FACE_SIM_THRESHOLD
from app.models import DetectedFace, FaceCluster, FaceClusterMember


def face_candidates_from_neighbors(db: Session, neighbors: List[tuple[int, float]]) -> List[dict]:
    if not neighbors:
        return []

    face_ids = [face_id for face_id, _ in neighbors]

    rows = db.execute(
        select(
            FaceClusterMember.face_id,
            FaceClusterMember.cluster_id,
            FaceCluster.name,
            DetectedFace.crop_path,
        )
        .join(FaceCluster, FaceCluster.id == FaceClusterMember.cluster_id)
        .join(DetectedFace, DetectedFace.id == FaceClusterMember.face_id)
        .where(FaceClusterMember.face_id.in_(face_ids))
    ).all()

    face_to_cluster = {}
    for face_id, cluster_id, cluster_name, crop_path in rows:
        face_to_cluster[face_id] = (cluster_id, cluster_name, crop_path)

    buckets = defaultdict(
        lambda: {
            "scores": [],
            "raw_scores": [],
            "cluster_name": None,
            "sample_paths": [],
        }
    )

    for rank, (neighbor_face_id, score) in enumerate(neighbors):
        if score < FACE_MIN_SCORE:
            continue

        cluster_info = face_to_cluster.get(neighbor_face_id)
        if not cluster_info:
            continue

        cluster_id, cluster_name, crop_path = cluster_info
        weight = 1.0 / (1.0 + 0.15 * rank)

        item = buckets[cluster_id]
        item["scores"].append(score * weight)
        item["raw_scores"].append(score)
        item["cluster_name"] = cluster_name
        if crop_path:
            item["sample_paths"].append(crop_path)

    out = []
    for cluster_id, item in buckets.items():
        top3 = sorted(item["scores"], reverse=True)[:3]
        raw_top3 = sorted(item["raw_scores"], reverse=True)[:3]

        score = sum(top3) / max(len(top3), 1)
        raw_score = max(raw_top3) if raw_top3 else 0.0
        avg_top3 = sum(raw_top3) / max(len(raw_top3), 1)

        sample_path = random.choice(item["sample_paths"]) if item["sample_paths"] else None

        out.append(
            {
                "cluster_id": cluster_id,
                "cluster_name": item["cluster_name"] or f"类_{cluster_id}",
                "score": float(score),
                "raw_best_score": float(raw_score),
                "avg_top3_score": float(avg_top3),
                "hit_count": len(item["scores"]),
                "sample_crop_path": sample_path,
            }
        )

    out.sort(key=lambda x: (x["raw_best_score"], x["score"], x["hit_count"]), reverse=True)
    return out


def choose_best_face_cluster(candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    if best["raw_best_score"] < FACE_SIM_THRESHOLD:
        return None
    if second and (best["raw_best_score"] - second["raw_best_score"]) < FACE_MARGIN:
        return None
    return best


def filter_clusters_above_threshold(candidates: List[dict], threshold: float = FACE_SIM_THRESHOLD) -> List[dict]:
    matched = []
    for c in candidates:
        if c["raw_best_score"] >= threshold:
            matched.append(c)
    return matched


def aggregate_image_candidates(face_rows: List[dict]) -> List[dict]:
    buckets = defaultdict(lambda: {"cluster_name": None, "score": 0.0, "face_ids": [], "sample_crop_url": None})

    for face in face_rows:
        for cand in face.get("suggested_clusters", []):
            cid = cand["cluster_id"]
            buckets[cid]["cluster_name"] = cand["cluster_name"]
            buckets[cid]["score"] += cand["raw_best_score"]
            buckets[cid]["face_ids"].append(face["face_id"])
            if not buckets[cid]["sample_crop_url"] and cand.get("sample_crop_url"):
                buckets[cid]["sample_crop_url"] = cand["sample_crop_url"]

    out = []
    for cid, item in buckets.items():
        out.append(
            {
                "cluster_id": cid,
                "cluster_name": item["cluster_name"] or f"类_{cid}",
                "score": round(item["score"], 4),
                "matched_face_ids": sorted(set(item["face_ids"])),
                "sample_crop_url": item["sample_crop_url"],
            }
        )

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def dump_json(data) -> str:
    return json.dumps(data, ensure_ascii=False)