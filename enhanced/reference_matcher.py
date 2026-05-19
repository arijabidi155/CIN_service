"""
Reference Image Matcher using SuperPoint + LightGlue.

Replaces: SIFT/AKAZE/ORB + structural features

SuperPoint + LightGlue (arxiv:2306.13643) advantages:
- 88.9% precision on HPatches vs ~60% for SIFT
- Adaptive early-exit: sub-10ms on planar (card) pairs
- Learned features robust to blur, glare, compression
- RANSAC homography from matched keypoints

HF Model: ETH-CVG/lightglue_superpoint (44K downloads, transformers-native)

Fallback chain:
1. SuperPoint + LightGlue (primary - best accuracy)
2. DINOv2 embedding similarity (fallback for degraded images)
3. AKAZE/ORB (legacy fallback if no GPU)
"""

import numpy as np
import cv2
from typing import Dict, Optional, List, Any
import logging

logger = logging.getLogger(__name__)


class LightGlueMatcher:
    """
    Reference image matcher using SuperPoint + LightGlue.
    
    Workflow:
    1. User provides a reference image of their ID card
    2. System detects and crops cards from video
    3. This matcher compares each crop against the reference
    4. Returns match score, inlier count, and homography
    """
    
    def __init__(self, config):
        """
        Args:
            config: MatcherConfig from enhanced/config.py
        """
        self.config = config
        self.pipeline = None
        self._fallback_mode = False
        self._load_model()
    
    def _load_model(self):
        """Load SuperPoint + LightGlue matching pipeline."""
        if self.config.backend.value == "lightglue":
            self._load_lightglue()
        elif self.config.backend.value == "sift":
            self._load_sift_fallback()
        elif self.config.backend.value == "orb":
            self._load_orb_fallback()
    
    def _load_lightglue(self):
        """Load LightGlue via HuggingFace transformers pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            self.pipeline = hf_pipeline(
                "keypoint-matching",
                model=self.config.model_name,
            )
            logger.info(f"Loaded LightGlue matcher: {self.config.model_name}")
        except Exception as e:
            logger.warning(f"Could not load LightGlue: {e}. Using AKAZE fallback.")
            self._load_akaze_fallback()
    
    def _load_sift_fallback(self):
        """SIFT + BFMatcher fallback."""
        self.detector = cv2.SIFT_create(nfeatures=self.config.max_keypoints)
        self.matcher_cv = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self._fallback_mode = True
        logger.info("Using SIFT fallback matcher")
    
    def _load_orb_fallback(self):
        """ORB + BFMatcher fallback."""
        self.detector = cv2.ORB_create(nfeatures=self.config.max_keypoints)
        self.matcher_cv = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._fallback_mode = True
        logger.info("Using ORB fallback matcher")
    
    def _load_akaze_fallback(self):
        """AKAZE fallback when LightGlue unavailable."""
        self.detector = cv2.AKAZE_create()
        self.matcher_cv = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._fallback_mode = True
        logger.info("Using AKAZE fallback matcher")
    
    def match(self, reference_image: np.ndarray, card_crop: np.ndarray) -> Dict[str, Any]:
        """
        Match a card crop against a reference image.
        
        Args:
            reference_image: BGR reference image (clean scan)
            card_crop: BGR card crop from video frame
        
        Returns:
            Dict with:
                - match: bool (whether it's a match)
                - score: float (0-1, match confidence)
                - num_matches: int (total keypoint matches)
                - num_inliers: int (RANSAC inliers)
                - inlier_ratio: float (inliers/matches)
                - homography: list (3x3 matrix) or None
                - method: str ("lightglue" or "akaze_fallback")
        """
        if self._fallback_mode:
            return self._match_opencv(reference_image, card_crop)
        else:
            return self._match_lightglue(reference_image, card_crop)
    
    def _match_lightglue(self, reference: np.ndarray, crop: np.ndarray) -> Dict[str, Any]:
        """Match using SuperPoint + LightGlue."""
        from PIL import Image
        
        # Convert to PIL
        ref_pil = Image.fromarray(reference[:, :, ::-1])
        crop_pil = Image.fromarray(crop[:, :, ::-1])
        
        try:
            results = self.pipeline(
                [ref_pil, crop_pil],
                threshold=self.config.match_threshold,
            )
        except Exception as e:
            logger.warning(f"LightGlue matching failed: {e}. Using fallback.")
            return self._match_opencv(reference, crop)
        
        if not results or len(results) == 0:
            return self._empty_result("lightglue")
        
        matches = results[0] if isinstance(results, list) else results
        
        # Extract matched keypoint coordinates
        pts_ref = []
        pts_crop = []
        scores = []
        
        if isinstance(matches, list):
            for m in matches:
                if isinstance(m, dict):
                    if "keypoints0" in m and "keypoints1" in m:
                        pts_ref.append([m["keypoints0"]["x"], m["keypoints0"]["y"]])
                        pts_crop.append([m["keypoints1"]["x"], m["keypoints1"]["y"]])
                        scores.append(m.get("score", 0.5))
        elif isinstance(matches, dict):
            if "keypoints0" in matches and "keypoints1" in matches:
                kp0 = matches["keypoints0"]
                kp1 = matches["keypoints1"]
                if isinstance(kp0, list):
                    for k0, k1 in zip(kp0, kp1):
                        pts_ref.append([k0["x"], k0["y"]])
                        pts_crop.append([k1["x"], k1["y"]])
        
        num_matches = len(pts_ref)
        
        if num_matches < 4:
            return self._empty_result("lightglue", num_matches=num_matches)
        
        pts_ref = np.array(pts_ref, dtype=np.float64)
        pts_crop = np.array(pts_crop, dtype=np.float64)
        
        # RANSAC homography
        H, mask = cv2.findHomography(pts_ref, pts_crop, cv2.RANSAC, self.config.ransac_threshold)
        num_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
        
        is_match = (num_inliers >= self.config.min_inliers and
                    inlier_ratio >= self.config.min_inlier_ratio)
        
        return {
            "match": is_match,
            "score": float(inlier_ratio),
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": float(inlier_ratio),
            "homography": H.tolist() if H is not None else None,
            "method": "lightglue",
        }
    
    def _match_opencv(self, reference: np.ndarray, crop: np.ndarray) -> Dict[str, Any]:
        """Fallback matching using OpenCV feature detectors."""
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        
        kp1, des1 = self.detector.detectAndCompute(ref_gray, None)
        kp2, des2 = self.detector.detectAndCompute(crop_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return self._empty_result("opencv_fallback")
        
        # KNN matching with ratio test (Lowe's)
        raw_matches = self.matcher_cv.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        num_matches = len(good_matches)
        
        if num_matches < 4:
            return self._empty_result("opencv_fallback", num_matches=num_matches)
        
        pts_ref = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float64)
        pts_crop = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float64)
        
        H, mask = cv2.findHomography(pts_ref, pts_crop, cv2.RANSAC, self.config.ransac_threshold)
        num_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
        
        is_match = (num_inliers >= self.config.min_inliers and
                    inlier_ratio >= self.config.min_inlier_ratio)
        
        return {
            "match": is_match,
            "score": float(inlier_ratio),
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": float(inlier_ratio),
            "homography": H.tolist() if H is not None else None,
            "method": "opencv_fallback",
        }
    
    def _empty_result(self, method: str, num_matches: int = 0) -> Dict[str, Any]:
        """Return empty match result."""
        return {
            "match": False,
            "score": 0.0,
            "num_matches": num_matches,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "homography": None,
            "method": method,
        }
    
    def match_multi_reference(self, references: List[np.ndarray],
                               card_crop: np.ndarray) -> Dict[str, Any]:
        """
        Match a card crop against multiple reference images.
        Returns the best match.
        """
        best_result = self._empty_result("none")
        best_ref_idx = -1
        
        for idx, ref in enumerate(references):
            result = self.match(ref, card_crop)
            if result["score"] > best_result["score"]:
                best_result = result
                best_ref_idx = idx
        
        best_result["reference_index"] = best_ref_idx
        return best_result
