from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MEDIA_DIR = DATA_DIR / "media"
IMAGE_DIR = MEDIA_DIR / "images"
FACE_DIR = MEDIA_DIR / "faces"
VECTOR_DIR = DATA_DIR / "vectors"
FAISS_DIR = DATA_DIR / "faiss"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

for p in [DATA_DIR, MEDIA_DIR, IMAGE_DIR, FACE_DIR, VECTOR_DIR, FAISS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'album_face.db'}")

EMBED_MODEL_PATH = os.getenv(
    "EMBED_MODEL_PATH",
    "/AIGC_Group/models/hy-motion/clip-vit-large-patch14",
)
DEVICE = os.getenv("DEVICE", "cuda")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1") == "1"
# 阈值要大于这个值才可以认为是匹配成功的，值越大越严格
FACE_SIM_THRESHOLD = float(os.getenv("FACE_SIM_THRESHOLD", "0.80"))
# 第一名和第二名的差距至少要大于这个值，才认为是有效匹配
FACE_MARGIN = float(os.getenv("FACE_MARGIN", "0.03"))
FACE_TOPK = int(os.getenv("FACE_TOPK", "40"))
# 人脸检测模型的置信度阈值，只有大于这个值的检测结果才会被保留
FACE_MIN_SCORE = float(os.getenv("FACE_MIN_SCORE", "0.65"))
# 黄框扩展
FACE_CROP_EXPAND = float(os.getenv("FACE_CROP_EXPAND", "0.18"))
ANIME_FACE_LEVEL = os.getenv("ANIME_FACE_LEVEL", "s")
ANIME_FACE_VERSION = os.getenv("ANIME_FACE_VERSION", "v1.4")
ANIME_FACE_MODEL_NAME = os.getenv("ANIME_FACE_MODEL_NAME", "").strip() or None
# 人脸置信度
ANIME_FACE_CONF_THRESHOLD = float(os.getenv("ANIME_FACE_CONF_THRESHOLD", "0.25"))
# 人脸去重
ANIME_FACE_IOU_THRESHOLD = float(os.getenv("ANIME_FACE_IOU_THRESHOLD", "0.50"))

FAISS_INDEX_PATH = FAISS_DIR / "faces.index"
