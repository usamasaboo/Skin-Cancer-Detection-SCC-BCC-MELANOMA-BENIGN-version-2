import cv2
import torch
import numpy as np
from app import apply_clinical_refinement, CLASSES
from clinical_features import extract_clinical_features

def test_blank_override(img_path):
    print(f"\nTesting {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Image not found")
        return
        
    clinical = extract_clinical_features(img_bgr)
    print(f"Blank detected: {clinical.get('is_blank')}")
    
    # Simulate a bad model prediction (e.g. Model says SCC on a blank arm)
    dummy_ai = torch.tensor([0.05, 0.05, 0.05, 0.85], dtype=torch.float32)
    
    refined = apply_clinical_refinement(dummy_ai, clinical)
    for i, p in enumerate(refined):
        print(f"  {CLASSES[i]}: {p.item():.3f}")

if __name__ == "__main__":
    test_blank_override("dummy_blank.jpg")
    test_blank_override("dummy_hair.jpg")
    test_blank_override("dummy_lesion.jpg")
