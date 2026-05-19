"""
DINOv2-based Feature Extractor for ID Card Re-ID.

Replaces: MobileNetV3 ImageNet features (category-level, not instance-level)

DINOv2 (arxiv:2304.07193) advantages:
- Self-supervised → learns instance-level features (not just categories)
- +34% mAP on Oxford-Hard instance retrieval vs OpenCLIP/CLIP
- KoLeo regularizer → uniformly distributed embedding space
- CLS token → global instance descriptor (768-dim for base)

Available models (all Apache-2.0):
- facebook/dinov2-small: 21M params, 384-dim, ~8ms  → mobile/edge
- facebook/dinov2-base:  86M params, 768-dim, ~15ms → production (RECOMMENDED)
- facebook/dinov2-large: 300M params, 1024-dim, ~40ms → research-grade
"""

import numpy as np
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class DINOv2Extractor:
    """
    Extract instance-level embeddings from card crops using DINOv2.
    
    Usage:
        extractor = DINOv2Extractor(config)
        embedding = extractor.extract(card_crop)  # np.ndarray [dim]
        similarity = extractor.compare(emb1, emb2)  # float [-1, 1]
    """
    
    def __init__(self, config):
        """
        Args:
            config: EmbeddingConfig from enhanced/config.py
        """
        self.config = config
        self.model = None
        self.processor = None
        self._device = None
        self._load_model()
    
    def _load_model(self):
        """Load DINOv2 model from HuggingFace Hub."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            
            model_name = self.config.backend.value
            logger.info(f"Loading DINOv2 model: {model_name}")
            
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            
            # Device selection
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(self.config.device)
            
            self.model = self.model.to(self._device)
            logger.info(f"DINOv2 loaded on {self._device} ({self.config.embedding_dim}-dim)")
            
        except ImportError:
            logger.warning("transformers/torch not installed. Run: pip install transformers torch. Using mock embeddings.")
            self.model = None
        except Exception as e:
            logger.warning(f"Could not load DINOv2: {e}. Using random embeddings.")
            self.model = None
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a normalized embedding from a card crop.
        
        Args:
            image: BGR numpy array (H, W, 3) — card crop
        
        Returns:
            L2-normalized embedding vector, shape [embedding_dim]
        """
        if self.model is None:
            return self._mock_extract(image)
        
        import torch
        from PIL import Image
        
        # Convert BGR to RGB PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(image[:, :, ::-1])
        else:
            pil_image = Image.fromarray(image)
        
        # Process and extract
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # CLS token = global instance embedding
        cls_token = outputs.last_hidden_state[:, 0, :]  # [1, dim]
        
        # L2-normalize for cosine similarity
        if self.config.normalize:
            cls_token = torch.nn.functional.normalize(cls_token, dim=-1)
        
        return cls_token.cpu().numpy().squeeze()  # [dim]
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple card crops.
        
        Args:
            images: List of BGR numpy arrays
        
        Returns:
            Embedding matrix, shape [N, embedding_dim]
        """
        if not images:
            return np.empty((0, self.config.embedding_dim))
        
        if self.model is None:
            return np.array([self._mock_extract(img) for img in images])
        
        import torch
        from PIL import Image
        
        pil_images = []
        for img in images:
            if len(img.shape) == 3 and img.shape[2] == 3:
                pil_images.append(Image.fromarray(img[:, :, ::-1]))
            else:
                pil_images.append(Image.fromarray(img))
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(pil_images), self.config.batch_size):
            batch = pil_images[i:i + self.config.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            if self.config.normalize:
                cls_tokens = torch.nn.functional.normalize(cls_tokens, dim=-1)
            
            all_embeddings.append(cls_tokens.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)
    
    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """
        Extract patch-level features for more fine-grained matching.
        Returns all patch tokens (not just CLS).
        
        Useful for: detailed texture comparison, partial card matching
        """
        if self.model is None:
            return np.random.randn(196, self.config.embedding_dim).astype(np.float32)
        
        import torch
        from PIL import Image
        
        pil_image = Image.fromarray(image[:, :, ::-1]) if len(image.shape) == 3 else Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # All patch tokens (exclude CLS at position 0)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [1, N_patches, dim]
        return patch_tokens.cpu().numpy().squeeze()  # [N_patches, dim]
    
    @staticmethod
    def compare(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding_a, embedding_b: L2-normalized embedding vectors
        
        Returns:
            Cosine similarity in [-1, 1] (higher = more similar)
        """
        return float(np.dot(embedding_a, embedding_b))
    
    @staticmethod
    def compare_batch(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between query and gallery embeddings.
        
        Args:
            query: Single embedding [dim]
            gallery: Gallery matrix [N, dim]
        
        Returns:
            Similarity scores [N]
        """
        return gallery @ query  # Assumes L2-normalized
    
    def _mock_extract(self, image: np.ndarray) -> np.ndarray:
        """Generate a deterministic mock embedding based on image content."""
        # Use image statistics as a simple fingerprint
        if image.size == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)
        
        np.random.seed(int(image.mean() * 1000) % (2**31))
        emb = np.random.randn(self.config.embedding_dim).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        return emb
