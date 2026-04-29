from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.config import DEVICE, EMBED_MODEL_PATH, LOCAL_FILES_ONLY


class LocalClipFaceEmbedder:
    def __init__(self, model_path: str = EMBED_MODEL_PATH, device: str = DEVICE):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=LOCAL_FILES_ONLY,
            trust_remote_code=True,
        )
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def encode_pil(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        feats = self.model.get_image_features(**inputs)

        # 兼容返回 ModelOutput 的情况
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            feats = feats.last_hidden_state[:, 0, :]
        elif not isinstance(feats, torch.Tensor):
            raise RuntimeError(f"get_image_features 返回了非 Tensor 类型: {type(feats)}")

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].detach().cpu().numpy().astype("float32")

    @torch.inference_mode()
    def encode_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return self.encode_pil(image)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        print(type(feats), getattr(feats, "keys", lambda: [])())
        # feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats[0].detach().cpu().numpy().astype("float32")

    @torch.inference_mode()
    def encode_path(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return self.encode_pil(image)


