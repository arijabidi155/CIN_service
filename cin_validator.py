import cv2
import requests
import numpy as np
import os
import sys

from enhanced.config import PipelineConfig
from enhanced.detector import IDCardDetector
from enhanced.quality_scorer import QualityScorer
from enhanced.reference_matcher import LightGlueMatcher

class CINValidator:
    def __init__(self):
        # Load production config (Option B)
        self.config = PipelineConfig.production_config()
        # Force CPU device for Hugging Face free spaces / general compatibility
        self.config.detector.device = "cpu"
        # Use YOLOv11 nano model
        self.config.detector.model_path = "yolo11n.pt"
        
        self.detector = IDCardDetector(self.config.detector)
        self.quality_scorer = QualityScorer(self.config.quality)
        
        # Initialize LightGlue Matcher for Tunisian features verification
        self.matcher = LightGlueMatcher(self.config.matcher)
        
        # Path to reference images
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        
        # Recto reference paths
        self.ref_flag_path = os.path.join(self.assets_dir, "ref_flag.jpg")
        self.ref_emblem_path = os.path.join(self.assets_dir, "ref_emblem.jpg")
        
        # Verso reference paths
        self.ref_verso_seal_path = os.path.join(self.assets_dir, "ref_verso_seal.jpg")
        self.ref_verso_fingerprint_path = os.path.join(self.assets_dir, "ref_verso_fingerprint.jpg")
        
        # Safe loading of reference crops
        self.ref_flag = cv2.imread(self.ref_flag_path) if os.path.exists(self.ref_flag_path) else None
        self.ref_emblem = cv2.imread(self.ref_emblem_path) if os.path.exists(self.ref_emblem_path) else None
        
        self.ref_verso_seal = cv2.imread(self.ref_verso_seal_path) if os.path.exists(self.ref_verso_seal_path) else None
        self.ref_verso_fingerprint = cv2.imread(self.ref_verso_fingerprint_path) if os.path.exists(self.ref_verso_fingerprint_path) else None
        
        # Status logging
        if self.ref_flag is None or self.ref_emblem is None:
            print("WARNING: Tunisian Recto reference crops not found. Recto specific validation will be bypassed.")
        else:
            print("Successfully loaded Tunisian Recto reference images for LightGlue matching!")
            
        if self.ref_verso_seal is None or self.ref_verso_fingerprint is None:
            print("WARNING: Tunisian Verso reference crops not found. Verso specific validation will be bypassed.")
        else:
            print("Successfully loaded Tunisian Verso reference images for LightGlue matching!")

    def download_image(self, url: str) -> np.ndarray:
        response = requests.get(url, timeout=15)
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from URL")
        return img

    def validate(self, image_url: str, side: str = "recto") -> dict:
        try:
            img = self.download_image(image_url)
        except Exception as e:
            return {"status": "error", "message": f"Failed to download image: {str(e)}"}

        # 1. Run YOLO Detector
        detections = self.detector.detect(img)
        
        # If no card detected
        if not detections:
            return {"status": "no_card"}

        # Take the best detection (highest confidence)
        best_detection = max(detections, key=lambda d: d.confidence)
        
        # Crop the card from frame
        card_crop = best_detection.crop_from(img)
        
        # 2. Check quality (Blur/Netteté)
        overall_score, details = self.quality_scorer.score(card_crop)
        
        # Reject if blur score is less than 0.5 (equivalent to variance < 250 on a 500 threshold)
        if details["blur"] < 0.5:
            return {
                "status": "blurry",
                "score": overall_score,
                "details": details,
                "feedback": self.quality_scorer.get_feedback(details)
            }
            
        # 3. Tunisian Invariants Verification (SuperPoint + LightGlue)
        if side == "recto":
            # Only run if reference images are available
            if self.ref_flag is not None and self.ref_emblem is not None:
                # Match against the Tunisian Flag (top-left)
                match_flag = self.matcher.match(self.ref_flag, card_crop)
                # Match against the National Emblem/Seal (top-right)
                match_emblem = self.matcher.match(self.ref_emblem, card_crop)
                
                # We require at least one robust visual anchor match to validate it is a Tunisian CIN Recto
                is_tunisian = match_flag["match"] or match_emblem["match"]

                if not is_tunisian:
                    return {
                        "status": "no_card",
                        "score": overall_score,
                        "details": details,
                        "feedback": "Le document n'est pas identifié comme le Recto d'une CIN tunisienne valide (ancres visuelles introuvables)."
                    }
        elif side == "verso":
            if self.ref_verso_seal is not None and self.ref_verso_fingerprint is not None:
                # Match against the Ministry Seal (bottom center)
                match_seal = self.matcher.match(self.ref_verso_seal, card_crop)
                # Match against the Fingerprint box (right)
                match_fingerprint = self.matcher.match(self.ref_verso_fingerprint, card_crop)
                
                is_tunisian_verso = match_seal["match"] or match_fingerprint["match"]
                
                if not is_tunisian_verso:
                    return {
                        "status": "no_card",
                        "score": overall_score,
                        "details": details,
                        "feedback": "Le document n'est pas identifié comme le Verso d'une CIN tunisienne valide."
                    }
            
        return {
            "status": "valid",
            "score": overall_score,
            "details": details,
            "feedback": "Card validation successful"
        }
