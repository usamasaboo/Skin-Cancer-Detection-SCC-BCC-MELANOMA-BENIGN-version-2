import torch
from app import apply_clinical_refinement, CLASSES

def test_melanoma_boost():
    # Simulate a confused model where Benign and Melanoma are tied
    ai_probs = torch.tensor([0.05, 0.45, 0.45, 0.05], dtype=torch.float32)
    
    print("--- SCENARIO 1: Classic Melanoma (Shape + Color) ---")
    clinical_1 = {'asymmetry': 0.8, 'border': 0.8, 'shiny': 0.1, 'roughness': 0.1, 'redness': 0.1, 'ulcer': 0.1, 'multicolor': 0.7, 'is_blank': False}
    ref_1 = apply_clinical_refinement(ai_probs, clinical_1)
    for i, p in enumerate(ref_1): print(f"  {CLASSES[i]}: {p.item():.3f}")

    print("\n--- SCENARIO 2: Nodular Melanoma (Perfect Shape, Crazy Color) ---")
    clinical_2 = {'asymmetry': 0.2, 'border': 0.2, 'shiny': 0.1, 'roughness': 0.1, 'redness': 0.1, 'ulcer': 0.1, 'multicolor': 0.9, 'is_blank': False}
    ref_2 = apply_clinical_refinement(ai_probs, clinical_2)
    for i, p in enumerate(ref_2): print(f"  {CLASSES[i]}: {p.item():.3f}")
    
    print("\n--- SCENARIO 3: Amelanotic Melanoma (Crazy Shape, No Color) ---")
    clinical_3 = {'asymmetry': 0.9, 'border': 0.9, 'shiny': 0.1, 'roughness': 0.1, 'redness': 0.1, 'ulcer': 0.1, 'multicolor': 0.2, 'is_blank': False}
    ref_3 = apply_clinical_refinement(ai_probs, clinical_3)
    for i, p in enumerate(ref_3): print(f"  {CLASSES[i]}: {p.item():.3f}")

if __name__ == "__main__":
    test_melanoma_boost()
