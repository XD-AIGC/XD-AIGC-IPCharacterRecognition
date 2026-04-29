from __future__ import annotations

from pathlib import Path
import numpy as np
from sqlalchemy import select
# from ...app.db import SessionLocal
# from ...app.models import FaceEmbedding, FaceClusterMember
# from ...app.services.face_index import FaceFaissIndex
# from ...app.config import FAISS_INDEX_PATH
from app.db import SessionLocal
from app.models import FaceEmbedding, FaceClusterMember
from app.services.face_index import FaceFaissIndex
from app.config import FAISS_INDEX_PATH


def main():
    db = SessionLocal()

    rows = db.execute(
        select(FaceEmbedding.face_id, FaceEmbedding.vector_path, FaceEmbedding.dim)
        .join(FaceClusterMember, FaceClusterMember.face_id == FaceEmbedding.face_id)
        .order_by(FaceEmbedding.face_id.asc())
    ).all()

    if not rows:
        print("没有任何已入类的人脸 embedding，无法重建索引。")
        return

    # 删除旧索引文件
    if Path(FAISS_INDEX_PATH).exists():
        Path(FAISS_INDEX_PATH).unlink()
        print(f"已删除旧索引: {FAISS_INDEX_PATH}")

    dim = rows[0][2]
    index = FaceFaissIndex(dim=dim)

    face_ids_added = []

    for face_id, vector_path, dim in rows:
        vp = Path(vector_path)
        if not vp.exists():
            print(f"[WARN] 向量文件不存在，跳过 face_id={face_id}: {vector_path}")
            continue

        vec = np.load(vp).astype("float32")
        index.add(vec, face_id)
        face_ids_added.append(face_id)

    index.save()
    print(f"重建完成，共写入 {len(face_ids_added)} 张已入类人脸到索引。")

    # 顺手修正 indexed 标记
    emb_rows = db.execute(select(FaceEmbedding)).scalars().all()
    added_set = set(face_ids_added)
    for emb in emb_rows:
        emb.indexed = emb.face_id in added_set
    db.commit()

    db.close()


if __name__ == "__main__":
    main()