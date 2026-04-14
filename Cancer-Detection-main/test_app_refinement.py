import os
import torch
import cv2
import numpy as np
import json
from PIL import Image
from app import model, transform, device, apply_clinical_refinement, extract_clinical_features, CLASSES
from image_preprocessing import preprocess_image
import glob

test_dirs = {
    'bcc': r'C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\bcc',
    'scc': r'C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\scc',
    'melanoma': r'C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\melanoma',
    'benign': r'C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\benign'
}

results = []

def test_on_dir(expected_class, dir_path, num_samples=5):
    files = glob.glob(os.path.join(dir_path, '*.jpg'))[:num_samples]
    
    for f in files:
        try:
            pil_image = Image.open(f).convert("RGB")
            img_np_rgb = np.array(pil_image)
            img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

            processed_img_bgr = preprocess_image(img_np_bgr, 224)
            processed_img_rgb = cv2.cvtColor(processed_img_bgr.astype('float32'), cv2.COLOR_BGR2RGB)
            input_tensor = transform(processed_img_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                ai_probs = torch.nn.functional.softmax(outputs[0], dim=0)

            clinical_metrics = extract_clinical_features(img_np_bgr)
            refined_probs = apply_clinical_refinement(ai_probs, clinical_metrics)

            ai_pred = CLASSES[torch.argmax(ai_probs).item()]
            ref_pred = CLASSES[torch.argmax(refined_probs).item()]
            
            results.append({
                'expected': expected_class,
                'file': os.path.basename(f),
                'ai_pred': ai_pred,
                'ref_pred': ref_pred,
                'ai_probs': {CLASSES[i]: round(p.item(), 3) for i, p in enumerate(ai_probs)},
                'ref_probs': {CLASSES[i]: round(p.item(), 3) for i, p in enumerate(refined_probs)},
                'clinical_metrics': {k: round(v, 3) for k, v in clinical_metrics.items()}
            })
        except Exception as e:
            results.append({'file': os.path.basename(f), 'error': str(e)})

if __name__ == "__main__":
    for cls, path in test_dirs.items():
        if os.path.exists(path):
            test_on_dir(cls, path)
        else:
            results.append({'error': f"Path not found: {path}"})
            
    with open('refinement_results.json', 'w') as f:
        json.dump(results, f, indent=2)
