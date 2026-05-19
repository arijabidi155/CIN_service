"""
Main Pipeline: ID Card Detection, Tracking, Re-ID, and Reference Matching.

Orchestrates all components:
1. Detector → find cards in frame
2. Tracker → assign track IDs, handle multi-card
3. Feature Extractor → DINOv2 embeddings per card
4. Re-ID Manager → EMA gallery for re-entry detection
5. Quality Scorer → card quality assessment
6. Reference Matcher → SuperPoint + LightGlue for reference comparison

Architecture: Option B (Balanced Production) from R&D report.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .config import PipelineConfig
from .detector import IDCardDetector, Detection
from .tracker import IDCardTracker, Track
from .feature_extractor import DINOv2Extractor
from .reid_manager import ReIDManager
from .reference_matcher import LightGlueMatcher
from .quality_scorer import QualityScorer

logger = logging.getLogger(__name__)


@dataclass
class CardResult:
    """Result for a single detected card in a frame."""
    track_id: int
    bbox: list                    # [x1, y1, x2, y2]
    confidence: float
    track_state: str              # "tentative", "confirmed", "lost"
    quality_score: float          # [0, 1]
    quality_details: Dict[str, float]
    quality_label: str            # "excellent", "good", "acceptable", "poor", "unusable"
    quality_feedback: str         # Human-readable feedback
    is_reidentified: bool         # True if this card was re-identified from gallery
    reid_similarity: Optional[float]  # Cosine similarity if re-identified
    reference_match: Optional[Dict]   # Reference matching result if available
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 3),
            "track_state": self.track_state,
            "quality": {
                "score": round(self.quality_score, 3),
                "label": self.quality_label,
                "feedback": self.quality_feedback,
                "details": self.quality_details,
            },
            "reid": {
                "is_reidentified": self.is_reidentified,
                "similarity": round(self.reid_similarity, 3) if self.reid_similarity else None,
            },
            "reference_match": self.reference_match,
        }


@dataclass
class FrameResult:
    """Result for a complete frame processing."""
    frame_number: int
    cards: List[CardResult]
    processing_time_ms: float
    gallery_stats: Dict
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "num_cards": len(self.cards),
            "cards": [c.to_dict() for c in self.cards],
            "processing_time_ms": round(self.processing_time_ms, 1),
            "gallery": self.gallery_stats,
        }


class IDCardPipeline:
    """
    End-to-end ID card detection, tracking, re-identification pipeline.
    
    Usage:
        config = PipelineConfig.production_config()
        pipeline = IDCardPipeline(config)
        
        # Process frames
        for frame in video_stream:
            result = pipeline.process_frame(frame)
            for card in result.cards:
                print(f"Card {card.track_id}: {card.quality_label}")
        
        # Set reference image for matching
        pipeline.set_reference(reference_image)
        
        # Process with reference matching
        result = pipeline.process_frame(frame)
        for card in result.cards:
            if card.reference_match and card.reference_match["match"]:
                print(f"Card {card.track_id} matches reference!")
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline with all components.
        
        Args:
            config: Pipeline configuration (default: production config)
        """
        self.config = config or PipelineConfig.production_config()
        self.frame_count = 0
        self.reference_images: List[np.ndarray] = []
        
        # Initialize components
        logger.info("Initializing ID Card Pipeline (Option B: Balanced Production)")
        
        self.detector = IDCardDetector(self.config.detector)
        self.tracker = IDCardTracker(self.config.tracker)
        self.extractor = DINOv2Extractor(self.config.embedding)
        self.reid_manager = ReIDManager(self.config.gallery, self.extractor)
        self.quality_scorer = QualityScorer(self.config.quality)
        
        if self.config.enable_reference_matching:
            self.matcher = LightGlueMatcher(self.config.matcher)
        else:
            self.matcher = None
        
        logger.info("Pipeline initialized successfully")
    
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single video frame.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
        
        Returns:
            FrameResult with detected/tracked cards
        """
        t_start = time.time()
        self.frame_count += 1
        
        # 1. Detect cards
        detections = self.detector.detect(frame)
        
        # 2. Track cards
        tracks = self.tracker.update(detections, frame)
        
        # 3. Process each tracked card
        card_results = []
        active_track_ids = set()
        
        for track in tracks:
            active_track_ids.add(track.track_id)
            
            # Extract card crop
            det = self._find_detection_for_track(track, detections)
            if det is not None:
                crop = det.crop_from(frame)
            else:
                # Use predicted bbox
                crop = self._crop_bbox(frame, track.bbox)
            
            if crop.size == 0:
                continue
            
            # Quality scoring
            quality_score, quality_details = (0.0, {}) 
            quality_label, quality_feedback = "unknown", ""
            if self.config.enable_quality_scoring:
                quality_score, quality_details = self.quality_scorer.score(crop)
                quality_label = self.quality_scorer.get_quality_label(quality_score)
                quality_feedback = self.quality_scorer.get_feedback(quality_details)
            
            # Re-ID: update gallery for confirmed tracks
            is_reidentified = False
            reid_similarity = None
            
            if self.config.enable_reid:
                if track.is_confirmed and quality_score >= 0.3:
                    self.reid_manager.update_track(
                        track.track_id, crop,
                        track.confidence, self.frame_count
                    )
            
            # Reference matching
            reference_match = None
            if (self.config.enable_reference_matching and
                self.matcher is not None and
                self.reference_images and
                track.is_confirmed and
                quality_score >= 0.5):
                
                reference_match = self.matcher.match_multi_reference(
                    self.reference_images, crop
                )
            
            card_results.append(CardResult(
                track_id=track.track_id,
                bbox=track.bbox.tolist(),
                confidence=track.confidence,
                track_state=track.state,
                quality_score=quality_score,
                quality_details=quality_details,
                quality_label=quality_label,
                quality_feedback=quality_feedback,
                is_reidentified=is_reidentified,
                reid_similarity=reid_similarity,
                reference_match=reference_match,
            ))
        
        # 4. Handle Re-ID for unmatched detections (potential re-entries)
        if self.config.enable_reid:
            # Check if any new detections could be re-entering cards
            for det in detections:
                matched_to_track = any(
                    self._is_same_detection(det, track)
                    for track in tracks
                )
                if not matched_to_track and det.confidence >= self.config.detector.confidence_threshold:
                    crop = det.crop_from(frame)
                    if crop.size > 0:
                        reid_result = self.reid_manager.try_reidentify(crop, self.frame_count)
                        if reid_result is not None:
                            recovered_id, similarity = reid_result
                            logger.info(f"Re-entry detected: card {recovered_id} returned (sim={similarity:.3f})")
            
            # Mark tracks that disappeared as lost in gallery
            all_tracked = self.tracker.get_all_tracks()
            for tid, track in all_tracked.items():
                if track.time_since_update == 1 and tid not in active_track_ids:
                    self.reid_manager.mark_lost(tid, self.frame_count)
            
            # Cleanup expired gallery entries
            self.reid_manager.cleanup(self.frame_count)
        
        processing_time = (time.time() - t_start) * 1000  # ms
        
        return FrameResult(
            frame_number=self.frame_count,
            cards=card_results,
            processing_time_ms=processing_time,
            gallery_stats=self.reid_manager.get_gallery_stats() if self.config.enable_reid else {},
        )
    
    def set_reference(self, reference_image: np.ndarray, append: bool = False):
        """
        Set reference image(s) for matching.
        
        Args:
            reference_image: BGR image of the reference card
            append: If True, add to existing references; otherwise replace
        """
        if append:
            self.reference_images.append(reference_image)
        else:
            self.reference_images = [reference_image]
        
        logger.info(f"Reference images set: {len(self.reference_images)} total")
    
    def clear_references(self):
        """Remove all reference images."""
        self.reference_images = []
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            "frames_processed": self.frame_count,
            "active_tracks": len(self.tracker.tracks),
            "gallery_stats": self.reid_manager.get_gallery_stats() if self.config.enable_reid else {},
            "num_references": len(self.reference_images),
        }
    
    def _find_detection_for_track(self, track: Track, detections: List[Detection]) -> Optional[Detection]:
        """Find the detection that matches a track (by IoU)."""
        best_iou = 0
        best_det = None
        for det in detections:
            iou = self.tracker._compute_iou(track.bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = det
        return best_det if best_iou > 0.3 else None
    
    def _crop_bbox(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract crop from bbox."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2].copy()
    
    def _is_same_detection(self, det: Detection, track: Track) -> bool:
        """Check if a detection corresponds to a track."""
        return self.tracker._compute_iou(det.bbox, track.bbox) > 0.3
