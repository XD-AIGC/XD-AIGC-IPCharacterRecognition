from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    file_path: Mapped[str] = mapped_column(String(512))
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    suggested_clusters_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DetectedFace(Base):
    __tablename__ = "detected_faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    face_index: Mapped[int] = mapped_column(Integer)

    # 原来的 bbox 字段保留，作为“兼容字段”，默认存扩展框（黄框）
    bbox_x1: Mapped[int] = mapped_column(Integer)
    bbox_y1: Mapped[int] = mapped_column(Integer)
    bbox_x2: Mapped[int] = mapped_column(Integer)
    bbox_y2: Mapped[int] = mapped_column(Integer)

    # 新增：原始检测脸框（红框）
    raw_x1: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    raw_y1: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    raw_x2: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    raw_y2: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # 新增：扩展后的 embedding 框（黄框）
    exp_x1: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    exp_y1: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    exp_x2: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    exp_y2: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    detector_score: Mapped[float] = mapped_column(Float)

    # 这里保存的是“黄框 crop”
    crop_path: Mapped[str] = mapped_column(String(512))

    suggested_cluster_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("face_clusters.id", ondelete="SET NULL"),
        nullable=True,
    )
    suggested_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    suggested_clusters_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    final_cluster_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("face_clusters.id", ondelete="SET NULL"),
        nullable=True,
    )

    review_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("image_id", "face_index", name="uq_image_face_index"),
    )


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    face_id: Mapped[int] = mapped_column(ForeignKey("detected_faces.id", ondelete="CASCADE"), primary_key=True)
    dim: Mapped[int] = mapped_column(Integer)
    vector_path: Mapped[str] = mapped_column(String(512))
    model_path: Mapped[str] = mapped_column(String(512))
    faiss_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FaceCluster(Base):
    __tablename__ = "face_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(String(128), unique=True, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FaceClusterMember(Base):
    __tablename__ = "face_cluster_members"

    cluster_id: Mapped[int] = mapped_column(ForeignKey("face_clusters.id", ondelete="CASCADE"), primary_key=True)
    face_id: Mapped[int] = mapped_column(ForeignKey("detected_faces.id", ondelete="CASCADE"), primary_key=True)
    similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="user")
    is_representative: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)