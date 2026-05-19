"""
Configuration for ID Card Detection, Tracking & Re-ID System.
Research-backed architecture (Option B: Balanced Production).

References:
- Detection: D-FINE-S (arxiv:2410.13842) / YOLOv11n (Ultralytics)
- Tracking: OC-SORT (arxiv:2203.14360) → BoT-SORT-ReID (arxiv:2206.14651)
- Re-ID: DINOv2 (arxiv:2304.07193)
- Matching: SuperPoint + LightGlue (arxiv:2306.13643)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class DetectorBackend(Enum):
    YOLO11 = "yolo11"
    DFINE = "dfine"
    RTDETR = "rtdetr"


class TrackerBackend(Enum):
    OCSORT = "ocsort"
    BOTSORT = "botsort"
    BYTETRACK = "bytetrack"


class EmbeddingBackend(Enum):
    DINOV2_SMALL = "facebook/dinov2-small"
    DINOV2_BASE = "facebook/dinov2-base"
    DINOV2_LARGE = "facebook/dinov2-large"


class MatcherBackend(Enum):
    LIGHTGLUE = "lightglue"
    SIFT = "sift"
    ORB = "orb"


@dataclass
class DetectorConfig:
    """ID card detector configuration."""
    backend: DetectorBackend = DetectorBackend.YOLO11
    model_path: str = "yolo11n.pt"  # Ultralytics YOLO11n or fine-tuned checkpoint
    confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.45
    input_size: Tuple[int, int] = (640, 640)
    # ID card aspect ratio range (ISO/IEC 7810 ID-1: 1.585:1, allowing perspective distortion)
    min_aspect_ratio: float = 0.8   # More permissive than old 1.2 (handles tilted cards)
    max_aspect_ratio: float = 3.0   # More permissive than old 2.0
    min_area_ratio: float = 0.005   # Minimum card area as fraction of frame
    max_area_ratio: float = 0.80    # Maximum card area as fraction of frame
    multi_scale: bool = False       # Enable multi-scale inference for small cards
    device: str = "auto"            # "auto", "cpu", "cuda", "mps"


@dataclass
class TrackerConfig:
    """Multi-object tracker configuration."""
    backend: TrackerBackend = TrackerBackend.OCSORT
    # OC-SORT specific (arxiv:2203.14360)
    det_thresh: float = 0.5         # High-confidence detection threshold
    max_age: int = 120              # Frames to keep lost track (4s at 30fps)
    min_hits: int = 3               # Min detections before track is confirmed
    iou_threshold: float = 0.3      # IoU matching threshold
    delta_t: int = 3                # OC-SORT: observation momentum window
    inertia: float = 0.2            # OC-SORT: momentum weight
    use_byte: bool = True           # Enable BYTE low-score recovery
    # BoT-SORT specific (Phase 5 upgrade)
    enable_cmc: bool = False        # Camera motion compensation (ECC)
    enable_reid: bool = True        # Use appearance features in association
    reid_weight: float = 0.5        # Weight for appearance vs motion


@dataclass
class EmbeddingConfig:
    """Re-ID embedding model configuration."""
    backend: EmbeddingBackend = EmbeddingBackend.DINOV2_BASE
    embedding_dim: int = 768        # DINOv2-base output dimension
    normalize: bool = True          # L2-normalize embeddings
    batch_size: int = 8             # Max batch for embedding extraction
    device: str = "auto"


@dataclass
class GalleryConfig:
    """Re-ID gallery (EMA-updated embedding store)."""
    ema_alpha: float = 0.9          # Exponential moving average for embedding update
    match_threshold: float = 0.82   # Cosine similarity threshold for re-entry matching
    max_gallery_age: int = 300      # Max frames to keep a lost track in gallery (10s at 30fps)
    min_confidence_for_gallery: float = 0.6  # Min detection confidence to update gallery
    adaptive_threshold: bool = True  # Adjust threshold based on track age


@dataclass
class MatcherConfig:
    """Reference image matching configuration."""
    backend: MatcherBackend = MatcherBackend.LIGHTGLUE
    model_name: str = "ETH-CVG/lightglue_superpoint"
    match_threshold: float = 0.2    # LightGlue keypoint match confidence
    min_inliers: int = 15           # Minimum RANSAC inliers for valid match
    min_inlier_ratio: float = 0.40  # Minimum inlier/total ratio
    ransac_threshold: float = 3.0   # RANSAC reprojection error (pixels)
    max_keypoints: int = 1024       # Max keypoints per image
    # Fallback: if too few keypoints, use DINOv2 embedding similarity
    fallback_to_embedding: bool = True
    fallback_min_keypoints: int = 50
    fallback_similarity_threshold: float = 0.75
    device: str = "auto"


@dataclass
class QualityConfig:
    """Card image quality scoring."""
    blur_weight: float = 0.35
    glare_weight: float = 0.25
    size_weight: float = 0.25
    aspect_weight: float = 0.15
    # Thresholds
    laplacian_threshold: float = 500.0  # Below this = blurry
    min_card_pixels: int = 200          # Minimum short side in pixels
    ideal_aspect_ratio: float = 1.585   # ISO/IEC 7810 ID-1
    glare_max_brightness: float = 0.95  # Fraction of pixels above threshold


@dataclass
class PrivacyConfig:
    """Privacy and data handling configuration."""
    store_raw_images: bool = False       # Never store raw card images
    store_embeddings_only: bool = True   # Store only DINOv2 embeddings (not invertible)
    embedding_dtype: str = "float16"     # 768 × 2 bytes = 1.5KB per card
    auto_delete_gallery_hours: int = 24  # Auto-delete gallery entries after N hours
    log_detections: bool = True          # Log detection events (without images)
    log_matches: bool = True             # Log match events (without images)
    encrypt_embeddings: bool = False     # Optional: encrypt stored embeddings at rest


@dataclass
class TemporalConfig:
    """Temporal smoothing and track management."""
    bbox_smooth_alpha: float = 0.7       # EMA for bbox coordinates
    confidence_smooth_alpha: float = 0.8  # EMA for track confidence
    tentative_frames: int = 3            # Frames before track is confirmed
    interpolation_max_gap: int = 5       # Max gap for linear interpolation (GSI-style)


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    
    # Global settings
    target_fps: int = 30
    max_cards_per_frame: int = 10
    enable_quality_scoring: bool = True
    enable_reid: bool = True
    enable_reference_matching: bool = True
    verbose: bool = False

    @classmethod
    def edge_config(cls) -> "PipelineConfig":
        """Lightweight config for edge/mobile deployment (Option A)."""
        return cls(
            detector=DetectorConfig(
                backend=DetectorBackend.YOLO11,
                model_path="yolo11n.pt",
                confidence_threshold=0.4,
                input_size=(416, 416),
            ),
            tracker=TrackerConfig(
                backend=TrackerBackend.OCSORT,
                max_age=60,
                enable_reid=False,
            ),
            embedding=EmbeddingConfig(
                backend=EmbeddingBackend.DINOV2_SMALL,
                embedding_dim=384,
            ),
            matcher=MatcherConfig(
                backend=MatcherBackend.ORB,
                min_inliers=10,
            ),
        )

    @classmethod
    def production_config(cls) -> "PipelineConfig":
        """Balanced production config (Option B - RECOMMENDED)."""
        return cls()  # defaults are Option B

    @classmethod
    def research_config(cls) -> "PipelineConfig":
        """Maximum quality research config (Option C)."""
        return cls(
            detector=DetectorConfig(
                backend=DetectorBackend.DFINE,
                model_path="dfine_l_objects365.pth",
                confidence_threshold=0.3,
                input_size=(800, 800),
                multi_scale=True,
            ),
            tracker=TrackerConfig(
                backend=TrackerBackend.BOTSORT,
                max_age=300,
                enable_cmc=True,
                enable_reid=True,
            ),
            embedding=EmbeddingConfig(
                backend=EmbeddingBackend.DINOV2_LARGE,
                embedding_dim=1024,
            ),
            gallery=GalleryConfig(
                match_threshold=0.78,
                max_gallery_age=600,
            ),
        )
