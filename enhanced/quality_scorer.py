"""
Card Quality Scorer.

NEW component (not in original system).

Scores card crop quality for:
1. Deciding whether to update Re-ID gallery (skip blurry/glaring crops)
2. User feedback ("hold card steady", "move to better lighting")
3. Selecting the best frame for reference matching

Quality dimensions:
- Blur (Laplacian variance) — most important for readability
- Glare/specular reflection — common with laminated cards
- Resolution (card pixel size) — too small = unusable
- Aspect ratio conformance — deviations suggest perspective distortion
"""

import numpy as np
import cv2
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Card image quality scoring module.
    
    Usage:
        scorer = QualityScorer(config)
        score, details = scorer.score(card_crop)
        # score: float [0, 1], higher = better
        # details: dict with per-dimension scores
    """
    
    def __init__(self, config):
        """
        Args:
            config: QualityConfig from enhanced/config.py
        """
        self.config = config
    
    def score(self, card_crop: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Compute quality score for a card crop.
        
        Args:
            card_crop: BGR image of the card crop
        
        Returns:
            (overall_score, detail_scores)
            - overall_score: weighted average [0, 1]
            - detail_scores: per-dimension scores [0, 1]
        """
        if card_crop.size == 0:
            return 0.0, {"blur": 0, "glare": 0, "size": 0, "aspect": 0}
        
        blur_score = self._compute_blur_score(card_crop)
        glare_score = self._compute_glare_score(card_crop)
        size_score = self._compute_size_score(card_crop)
        aspect_score = self._compute_aspect_score(card_crop)
        
        # Weighted average
        overall = (
            self.config.blur_weight * blur_score +
            self.config.glare_weight * glare_score +
            self.config.size_weight * size_score +
            self.config.aspect_weight * aspect_score
        )
        
        details = {
            "blur": round(blur_score, 3),
            "glare": round(glare_score, 3),
            "size": round(size_score, 3),
            "aspect": round(aspect_score, 3),
            "overall": round(overall, 3),
        }
        
        return overall, details
    
    def _compute_blur_score(self, image: np.ndarray) -> float:
        """
        Blur detection using Laplacian variance.
        Higher variance = sharper image = better quality.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize: 0 (very blurry) to 1 (sharp)
        score = min(1.0, variance / self.config.laplacian_threshold)
        return score
    
    def _compute_glare_score(self, image: np.ndarray) -> float:
        """
        Glare/specular reflection detection.
        Checks for over-exposed regions (hot spots from flash/laminate reflection).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Count near-white pixels (glare)
        threshold = 240  # Near-white
        bright_pixels = (gray >= threshold).sum()
        total_pixels = gray.size
        bright_ratio = bright_pixels / total_pixels
        
        # Also check for large contiguous bright regions (specular highlights)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour_area = 0
        if contours:
            max_contour_area = max(cv2.contourArea(c) for c in contours) / total_pixels
        
        # Score: 1 (no glare) to 0 (severe glare)
        ratio_score = 1.0 - min(1.0, bright_ratio / self.config.glare_max_brightness)
        contour_score = 1.0 - min(1.0, max_contour_area * 10)  # Penalize large bright spots
        
        return min(ratio_score, contour_score)
    
    def _compute_size_score(self, image: np.ndarray) -> float:
        """
        Resolution adequacy check.
        Card needs minimum pixel size for text readability and feature extraction.
        """
        h, w = image.shape[:2]
        min_side = min(h, w)
        
        # Score: 0 (too small) to 1 (adequate)
        score = min(1.0, min_side / self.config.min_card_pixels)
        return score
    
    def _compute_aspect_score(self, image: np.ndarray) -> float:
        """
        Aspect ratio conformance.
        ISO/IEC 7810 ID-1: 85.6mm × 53.98mm ≈ 1.585:1
        Deviations suggest perspective distortion or wrong detection.
        """
        h, w = image.shape[:2]
        aspect = max(w, h) / max(min(w, h), 1)
        
        ideal = self.config.ideal_aspect_ratio
        deviation = abs(aspect - ideal) / ideal
        
        # Score: 1 (perfect ratio) to 0 (very distorted)
        score = max(0.0, 1.0 - deviation)
        return score
    
    def get_quality_label(self, overall_score: float) -> str:
        """Convert quality score to human-readable label."""
        if overall_score >= 0.85:
            return "excellent"
        elif overall_score >= 0.70:
            return "good"
        elif overall_score >= 0.50:
            return "acceptable"
        elif overall_score >= 0.30:
            return "poor"
        else:
            return "unusable"
    
    def get_feedback(self, details: Dict[str, float]) -> str:
        """Generate user feedback based on quality scores."""
        issues = []
        if details["blur"] < 0.5:
            issues.append("Hold the card steady (image is blurry)")
        if details["glare"] < 0.5:
            issues.append("Reduce glare (move away from light source)")
        if details["size"] < 0.5:
            issues.append("Move closer to the card (too small)")
        if details["aspect"] < 0.5:
            issues.append("Hold the card flat (too much perspective distortion)")
        
        if not issues:
            return "Card quality is good"
        return "; ".join(issues)
