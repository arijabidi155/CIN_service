"""
Re-ID Manager with EMA Gallery for card re-entry detection.

Replaces: Simple feature comparison with static thresholds

Architecture (from BoT-SORT / StrongSORT research):
- EMA-updated per-track embeddings (α=0.9)
- Gallery of lost tracks persists for configurable duration
- Cosine similarity matching with adaptive thresholds
- Privacy-safe: stores only embeddings, never raw images

Key improvement: When a card leaves and re-enters the frame,
the gallery matches it against stored embeddings to recover
the original track ID instead of assigning a new one.

References:
- StrongSORT (arxiv:2202.13514): EMA α=0.9, gallery-based Re-ID
- BoT-SORT (arxiv:2206.14651): min(IoU, cos_sim) fusion
- HAT-ReID (arxiv:2503.12562): FLD projection for similar-appearance objects
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class GalleryEntry:
    """Single entry in the Re-ID gallery."""
    track_id: int
    embedding: np.ndarray       # DINOv2 embedding (768-dim float32)
    first_seen: float           # Timestamp when track was created
    last_seen: float            # Timestamp of last update
    last_frame: int             # Frame number of last update
    update_count: int = 0       # Number of embedding updates
    confidence_history: list = None  # Detection confidence history
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = []
    
    @property
    def avg_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history[-10:]) / len(self.confidence_history[-10:])


class ReIDManager:
    """
    Re-ID gallery manager for ID card re-entry detection.
    
    Workflow:
    1. When a new confirmed track appears → store its DINOv2 embedding
    2. Every frame the track is active → EMA-update its embedding
    3. When a track is lost → keep embedding in gallery with age counter
    4. When a new unmatched detection appears → match against gallery
    5. If match found → re-assign the old track ID (recovered re-entry)
    6. Gallery entries expire after max_gallery_age frames
    
    Privacy:
    - Only DINOv2 embeddings stored (768-dim, 1.5KB each in float16)
    - Embeddings are NOT invertible to original images
    - Auto-delete after configurable time
    """
    
    def __init__(self, config, feature_extractor):
        """
        Args:
            config: GalleryConfig + PrivacyConfig from enhanced/config.py
            feature_extractor: DINOv2Extractor instance
        """
        self.config = config
        self.extractor = feature_extractor
        
        # Active gallery: tracks currently being tracked
        self.active_gallery: Dict[int, GalleryEntry] = {}
        # Lost gallery: tracks that have been lost but may return
        self.lost_gallery: Dict[int, GalleryEntry] = {}
        
        # Statistics
        self.total_reids = 0
        self.false_reids = 0  # Requires ground truth to compute
    
    def update_track(self, track_id: int, card_crop: np.ndarray,
                     confidence: float, frame_number: int):
        """
        Update gallery with a new observation of an active track.
        
        Args:
            track_id: Current track ID
            card_crop: BGR image of the card crop
            confidence: Detection confidence
            frame_number: Current frame number
        """
        if confidence < self.config.min_confidence_for_gallery:
            return
        
        # Extract embedding
        embedding = self.extractor.extract(card_crop)
        
        if track_id in self.active_gallery:
            # EMA update (StrongSORT: α=0.9)
            entry = self.active_gallery[track_id]
            entry.embedding = (
                self.config.ema_alpha * entry.embedding + 
                (1 - self.config.ema_alpha) * embedding
            )
            # Re-normalize after EMA
            norm = np.linalg.norm(entry.embedding)
            if norm > 0:
                entry.embedding /= norm
            
            entry.last_seen = time.time()
            entry.last_frame = frame_number
            entry.update_count += 1
            entry.confidence_history.append(confidence)
        else:
            # New track
            self.active_gallery[track_id] = GalleryEntry(
                track_id=track_id,
                embedding=embedding,
                first_seen=time.time(),
                last_seen=time.time(),
                last_frame=frame_number,
                update_count=1,
                confidence_history=[confidence],
            )
            logger.debug(f"Added track {track_id} to active gallery")
    
    def mark_lost(self, track_id: int, frame_number: int):
        """
        Move a track from active to lost gallery.
        Called when tracker marks a track as lost.
        """
        if track_id in self.active_gallery:
            entry = self.active_gallery.pop(track_id)
            entry.last_frame = frame_number
            self.lost_gallery[track_id] = entry
            logger.debug(f"Track {track_id} moved to lost gallery "
                        f"(updated {entry.update_count} times, "
                        f"avg conf {entry.avg_confidence:.2f})")
    
    def try_reidentify(self, card_crop: np.ndarray, frame_number: int) -> Optional[Tuple[int, float]]:
        """
        Try to re-identify a new unmatched detection against the lost gallery.
        
        Args:
            card_crop: BGR image of the unmatched card detection
            frame_number: Current frame number
        
        Returns:
            (track_id, similarity) if match found, None otherwise
        """
        if not self.lost_gallery:
            return None
        
        # Extract query embedding
        query_embedding = self.extractor.extract(card_crop)
        
        # Compare against all lost gallery entries
        best_match_id = None
        best_similarity = -1.0
        
        for track_id, entry in self.lost_gallery.items():
            # Compute cosine similarity
            similarity = float(np.dot(query_embedding, entry.embedding))
            
            # Adaptive threshold: lower threshold for recently-lost tracks
            # (they're more likely to be genuine re-entries)
            threshold = self._adaptive_threshold(entry, frame_number)
            
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = track_id
        
        if best_match_id is not None:
            self.total_reids += 1
            logger.info(f"Re-identified track {best_match_id} with similarity {best_similarity:.3f}")
            
            # Move back to active gallery
            entry = self.lost_gallery.pop(best_match_id)
            entry.last_seen = time.time()
            entry.last_frame = frame_number
            
            # Update with new observation
            entry.embedding = (
                self.config.ema_alpha * entry.embedding +
                (1 - self.config.ema_alpha) * query_embedding
            )
            norm = np.linalg.norm(entry.embedding)
            if norm > 0:
                entry.embedding /= norm
            
            self.active_gallery[best_match_id] = entry
            
            return best_match_id, best_similarity
        
        return None
    
    def _adaptive_threshold(self, entry: GalleryEntry, current_frame: int) -> float:
        """
        Adaptive threshold based on how long the track has been lost.
        
        Rationale: Recently-lost tracks are more likely to be genuine re-entries,
        so we can use a slightly lower threshold. Older entries need higher
        confidence to avoid false matches.
        """
        if not self.config.adaptive_threshold:
            return self.config.match_threshold
        
        frames_lost = current_frame - entry.last_frame
        base = self.config.match_threshold
        
        if frames_lost < 30:
            # Recently lost (< 1 second): slightly more permissive
            return base - 0.05
        elif frames_lost < 120:
            # Moderately lost (1-4 seconds): standard threshold
            return base
        else:
            # Long lost (> 4 seconds): stricter threshold
            return base + 0.05
    
    def cleanup(self, frame_number: int):
        """
        Remove expired entries from lost gallery.
        Called every frame.
        """
        expired = [
            tid for tid, entry in self.lost_gallery.items()
            if (frame_number - entry.last_frame) > self.config.max_gallery_age
        ]
        for tid in expired:
            del self.lost_gallery[tid]
            logger.debug(f"Expired lost track {tid} from gallery")
    
    def get_gallery_stats(self) -> Dict:
        """Return gallery statistics for monitoring."""
        return {
            "active_tracks": len(self.active_gallery),
            "lost_tracks": len(self.lost_gallery),
            "total_reids": self.total_reids,
            "gallery_memory_bytes": sum(
                entry.embedding.nbytes 
                for entry in list(self.active_gallery.values()) + list(self.lost_gallery.values())
            ),
        }
    
    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get the current embedding for a track (for external use)."""
        if track_id in self.active_gallery:
            return self.active_gallery[track_id].embedding.copy()
        if track_id in self.lost_gallery:
            return self.lost_gallery[track_id].embedding.copy()
        return None
