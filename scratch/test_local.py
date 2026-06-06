import sys
import os
import cv2

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cin_validator import CINValidator

print("Instantiating CINValidator...")
validator = CINValidator()

# Use assets/ref_flag.jpg for testing
assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
image_path = os.path.join(assets_dir, "ref_flag.jpg")

if os.path.exists(image_path):
    print(f"\nLoading local test image: {image_path}")
    img = cv2.imread(image_path)
    
    # Mock download_image to return our local image
    validator.download_image = lambda url: img
    
    # Test Recto
    print("\n--- Testing Recto Validation ---")
    res_recto = validator.validate("dummy_url", "recto")
    print("Recto Result:")
    for k, v in res_recto.items():
        print(f"  {k}: {v}")
        
    # Test Verso
    print("\n--- Testing Verso Validation ---")
    res_verso = validator.validate("dummy_url", "verso")
    print("Verso Result:")
    for k, v in res_verso.items():
        print(f"  {k}: {v}")
else:
    print(f"Test image not found at: {image_path}")
