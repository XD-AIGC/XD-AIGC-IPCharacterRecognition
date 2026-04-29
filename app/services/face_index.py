from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH


class FaceFaissIndex:
    def __init__(self, dim: int, index_path: str | None = None):
        self.dim = dim
        self.index_path = str(index_path or FAISS_INDEX_PATH)
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
        else:
            base = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap2(base)
# faiss.IndexIDMap2 是一个包装器，允许我们为每个向量指定一个唯一的 ID，这样我们就可以在搜索结果中直接得到对应的 face_id，而不需要额外的映射表。
    def add(self, vec: np.ndarray, face_id: int):
        x = np.asarray(vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(x)
        ids = np.asarray([face_id], dtype="int64")
        self.index.add_with_ids(x, ids)
# add_with_ids 方法允许我们将向量和对应的 ID 一起添加到索引中，这样在搜索时就可以直接返回这些 ID。
    def search(self, vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        x = np.asarray(vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(x)
        scores, ids = self.index.search(x, k)
        out = []
        for face_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if face_id != -1:
                out.append((int(face_id), float(score)))
        return out
# 
    def save(self):
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
