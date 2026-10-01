import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger("cctv.face_embedder")

class BaseFaceEmbedder:
    """Abstract interface for Face Embedding Models."""
    def get_embedding(self, face_crop: np.ndarray) -> Optional[List[float]]:
        raise NotImplementedError

    @staticmethod
    def compute_similarity(emb1: List[float], emb2: List[float]) -> float:
        """Computes Cosine Similarity between two L2-normalized embeddings in [-1.0, 1.0]."""
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

class TorchVisionDeepEmbedder(BaseFaceEmbedder):
    """
    Deep Convolutional Face Embedder utilizing pre-trained deep visual feature backbone
    with spatial pooling and L2-normalized 128-dimensional unit hypersphere projection.
    """
    def __init__(self, embedding_dim: int = 128, device: str = "cpu"):
        self.embedding_dim = embedding_dim
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_version = "torchvision_deep_mobilenet_v1"

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            import torchvision.models as models
            # Load pre-trained MobileNetV2 feature backbone
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            # Use convolutional feature extraction layers
            self.backbone = mobilenet.features.to(self.device)
            self.backbone.eval()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.projection = nn.Linear(1280, self.embedding_dim).to(self.device)
            # Seed projection weights deterministically
            torch.manual_seed(42)
            nn.init.orthogonal_(self.projection.weight)
            self.projection.eval()
            logger.info("Initialized Pre-trained TorchVision MobileNetV2 Deep Face Embedder.")
        except Exception as e:
            logger.warning(f"Fallback to custom CNN embedder ({e})")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            ).to(self.device)
            self.pool = nn.Identity()
            self.projection = nn.Linear(128, self.embedding_dim).to(self.device)

    def get_embedding(self, face_crop: np.ndarray) -> Optional[List[float]]:
        if face_crop is None or face_crop.size == 0:
            return None

        try:
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.backbone(tensor)
                pooled = self.pool(features).flatten(1)
                emb = self.projection(pooled)
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
                vec = emb_norm.squeeze(0).cpu().numpy().tolist()
                return [round(float(v), 5) for v in vec]
        except Exception as e:
            logger.error(f"Error in deep face embedding extraction: {e}")
            return None

class SpatialTextureFaceEmbedder(BaseFaceEmbedder):
    """
    Lightweight, fast CPU face texture and spatial gradient histogram embedder.
    Extracts multi-block spatial HOG and intensity histograms normalized to 128 dimensions.
    Useful for environments where PyTorch weights download is constrained.
    """
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.model_version = "spatial_texture_embedder_v1"

    def get_embedding(self, face_crop: np.ndarray) -> Optional[List[float]]:
        if face_crop is None or face_crop.size == 0:
            return None
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            resized = cv2.equalizeHist(resized)
            
            # 4x4 spatial blocks of 8-bin gradient histograms (4*4*8 = 128 features)
            gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=1)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            
            bins = 8
            hist_vec = []
            bh, bw = 16, 16 # 64 / 4
            for r in range(4):
                for c in range(4):
                    block_mag = mag[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                    block_ang = ang[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                    hist, _ = np.histogram(block_ang, bins=bins, range=(0, 360), weights=block_mag)
                    hist_vec.extend(hist)

            vec = np.array(hist_vec[:self.embedding_dim], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            return [round(float(v), 5) for v in vec.tolist()]
        except Exception as e:
            logger.error(f"Error in texture face embedding extraction: {e}")
            return None

# Factory function with pluggable backend selection
def get_face_embedder(backend: str = "deep") -> BaseFaceEmbedder:
    if backend == "deep":
        return TorchVisionDeepEmbedder()
    elif backend == "texture":
        return SpatialTextureFaceEmbedder()
    else:
        return TorchVisionDeepEmbedder()

FaceEmbedder = TorchVisionDeepEmbedder
