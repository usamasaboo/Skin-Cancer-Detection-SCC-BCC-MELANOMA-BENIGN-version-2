import cv2
import torch
import numpy as np
from app import apply_clinical_refinement, CLASSES
from clinical_features import extract_clinical_features

def test_glare():
    # Simulate a pale arm with a strong reflection but no structure
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200
    cv2.circle(img, (112, 112), 30, (255, 255, 255), -1) # glare
    cv2.imwrite("dummy_white_glare.jpg", img)
    
    clinical = extract_clinical_features(img)
    print("Glare Test - Shiny Score:", clinical['shiny'], "Blank:", clinical['is_blank'])

def test_dark_skin():
    # Simulate dark skin
    img = np.ones((224, 224, 3), dtype=np.uint8) * 30
    cv2.imwrite("dummy_dark_skin.jpg", img)
    
    clinical = extract_clinical_features(img)
    print("Dark Skin Test - Blank:", clinical['is_blank'])
    
def test_ai_confidence():
    dummy_ai = torch.tensor([0.02, 0.90, 0.05, 0.03], dtype=torch.float32) # highly confident benign
    # Give it fake scary clinicals
    clinical = {'asymmetry': 0.9, 'border': 0.9, 'shiny': 0.1, 'roughness': 0.1, 'redness': 0.1, 'ulcer': 0.1, 'multicolor': 0.9, 'is_blank': False}
    refined = apply_clinical_refinement(dummy_ai, clinical)
    print("Confident AI Override Test:")
    for i, p in enumerate(refined):
        print(f"  {CLASSES[i]}: {p.item():.3f}")

if __name__ == "__main__":
    test_glare()
    test_dark_skin()
    test_ai_confidence()
