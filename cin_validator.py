import cv2
import requests
import numpy as np
import os
import sys
import easyocr
import re
import torch

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
        
        # Initialize EasyOCR reader for bilingual Arabic/French validation
        use_gpu = torch.cuda.is_available()
        print(f"EasyOCR initializing (GPU={use_gpu})...")
        self.ocr_reader = easyocr.Reader(['ar', 'en'], gpu=use_gpu)
        
        # Initialize barcode detector with fallback warning
        try:
            self.barcode_detector = cv2.barcode.BarcodeDetector()
            print("Successfully initialized cv2.barcode.BarcodeDetector!")
        except AttributeError:
            print("WARNING: cv2.barcode.BarcodeDetector is not available in this OpenCV build. Barcode detection will be bypassed.")
            self.barcode_detector = None
            
        # Initialize LightGlue Matcher for Tunisian features verification (Legacy/Fallback)
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

    def has_barcode(self, img_crop) -> bool:
        """
        Checks the cropped ID card image to see if a barcode or QR code is present.
        Returns True if found, False otherwise.
        """
        if not hasattr(self, 'barcode_detector') or self.barcode_detector is None:
            print("--> [BARCODE] Detector not initialized or unavailable.")
            return False
            
        try:
            # Convert to grayscale for clear line contrast detection
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            res = self.barcode_detector.detectAndDecode(gray)
            ok = res[0]
            decoded_info = res[1]
            
            # If 'ok' is True AND we actually have a non-empty string item in the list
            if ok and len(decoded_info) > 0 and decoded_info[0].strip():
                print(f"--> [BARCODE DETECTED]: Found valid code containing data: {decoded_info[0]}")
                return True
        except Exception as e:
            print(f"--> [BARCODE ERROR]: Failed to execute barcode detector: {e}")
            
        print("--> [NO BARCODE]: No readable codes found on this document surface.")
        return False

    def preprocess_for_ocr(self, card_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses the cropped ID card image to enhance text visibility.
        Applies grayscale conversion, CLAHE contrast enhancement, blurring, and Otsu thresholding.
        """
        # 1. Passage en niveaux de gris
        gray = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY) if len(card_crop.shape) == 3 else card_crop
        
        # 2. Augmentation du contraste (CLAHE) pour faire ressortir le texte
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 3. Flou léger pour enlever le bruit des motifs de fond
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 4. Seuillage adaptatif pour obtenir un texte noir pur sur fond blanc
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    def validate(self, image_url: str, side: str = "recto") -> dict:
        try:
            img = self.download_image(image_url)
            print(f"--- DEBUG SAHL EXPRESS ---")
            print(f"Image téléchargée avec succès. Résolution d'origine : {img.shape}")
            print(f"Demande de validation pour le côté : {side}")
        except Exception as e:
            return {"status": "error", "message": f"Failed to download image: {str(e)}"}

        # 1. Run YOLO Detector
        detections = self.detector.detect(img)
        print(f"Nombre de cartes détectées par YOLO : {len(detections)}")
        
        # Take the best detection or fallback to using the full image
        if detections:
            best_detection = max(detections, key=lambda d: d.confidence)
            card_crop = best_detection.crop_from(img)
            using_crop = True
            print(f"--> [YOLO] Card detected. Using cropped region of size {card_crop.shape}.")
        else:
            print("--> [YOLO] No card detected. Falling back to the full image.")
            card_crop = img
            using_crop = False

        # 2. Check quality (Blur/Netteté)
        overall_score, details = self.quality_scorer.score(card_crop)
        print(f"--- [DEBUG QUALITY] Score de flou calculé : {details.get('blur', 0)}")
        
        # 3. OCR Text Extraction (Arabic & French)
        print("--> Preprocessing image for OCR...")
        try:
            ocr_input = self.preprocess_for_ocr(card_crop)
            print("--> Running EasyOCR on preprocessed card crop/image...")
            ocr_results = self.ocr_reader.readtext(ocr_input, detail=0)
            combined_text = " ".join(ocr_results)
            print(f"--> [OCR EXTRACTED TEXT]: {combined_text}")
        except Exception as e:
            print(f"Error during OCR extraction: {e}")
            combined_text = ""

        # 4. Apply Validation Rules
        if side == "recto":
            has_name = "الاسم" in combined_text
            has_last_name = "اللقب" in combined_text
            has_dob = "الولادة" in combined_text or "ولادة" in combined_text
            
            # Find CIN number (8 digits, starting with 0 or 1)
            cin_match = re.search(r'\b[01]\d{7}\b', combined_text)
            cin_number = cin_match.group(0) if cin_match else None
            
            print(f"--> [RECTO CHECK] Name: {has_name} | Last Name: {has_last_name} | DOB: {has_dob} | CIN: {cin_number}")
            
            is_valid = has_name and has_last_name and has_dob and (cin_number is not None)
            
            details["cin_number"] = cin_number
            details["barcode_present"] = False
            
            if is_valid:
                return {
                    "status": "valid",
                    "score": overall_score,
                    "details": details,
                    "feedback": f"Card validation successful (CIN: {cin_number})"
                }
            else:
                missing = []
                if not has_name: missing.append("الاسم")
                if not has_last_name: missing.append("اللقب")
                if not has_dob: missing.append("تاريخ الولادة")
                if not cin_number: missing.append("numéro de CIN (8 chiffres)")
                
                return {
                    "status": "no_card",
                    "score": overall_score,
                    "details": details,
                    "feedback": f"Recto validation failed. Missing: {', '.join(missing)}"
                }
                
        elif side == "verso":
            has_address = "العنوان" in combined_text
            has_issue_place = "تونس في" in combined_text
            has_barcode_flag = self.has_barcode(card_crop)
            
            print(f"--> [VERSO CHECK] Address: {has_address} | Issue Place (Tunis on): {has_issue_place} | Barcode: {has_barcode_flag}")
            
            is_valid = has_address and has_issue_place
            
            details["barcode_present"] = has_barcode_flag
            
            if is_valid:
                if has_barcode_flag:
                    feedback = "Verso tracking pass: Barcode structural anchor verified."
                else:
                    feedback = "Verso validation successful (Note: Barcode anchor missing or unreadable)."
                
                return {
                    "status": "valid",
                    "score": overall_score,
                    "details": details,
                    "feedback": feedback
                }
            else:
                missing = []
                if not has_address: missing.append("العنوان")
                if not has_issue_place: missing.append("تونس في")
                
                return {
                    "status": "no_card",
                    "score": overall_score,
                    "details": details,
                    "feedback": f"Verso validation failed. Missing: {', '.join(missing)}"
                }
                
        return {
            "status": "no_card",
            "score": overall_score,
            "details": details,
            "feedback": "Unknown verification side specified."
        }
