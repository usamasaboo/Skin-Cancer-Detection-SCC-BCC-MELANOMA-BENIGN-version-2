import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
import os
import pandas as pd

# Config
MODEL_PATH = "saved_models/efficientnet_b0_skin_cancer_4class_v2.pth"
CLASSES = ['bcc', 'benign', 'melanoma', 'scc']
DATASET_PATH = r"c:\Users\Usama\Downloads\ISIC_2019_Training_Input\ISIC_2019_Training_Input"
BALANCED_CSV = r"c:\Users\Usama\Downloads\balanced_3.csv"

def verify():
    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found yet. Training might still be in progress.")
        return

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Sample images for testing (different from training if possible, but for verification just some IDs)
    df = pd.read_csv(BALANCED_CSV)
    test_samples = []
    for cls_name in ['BCC', 'MEL', 'SCC']:
        sample_id = df[df['label'] == cls_name].iloc[10]['image_id'] # Use index 10 as a simple "non-first" sample
        test_samples.append((sample_id, cls_name.lower() if cls_name != 'MEL' else 'melanoma'))
    
    # Add a benign sample from HAM (known ID)
    test_samples.append(('ISIC_0024306', 'benign'))

    print("\n--- Model V2 Verification ---")
    for img_id, expected in test_samples:
        img_path = os.path.join(DATASET_PATH, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            print(f"Image {img_id} not found.")
            continue
            
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_class_idx = torch.max(probabilities, 0)
            pred_class = CLASSES[top_class_idx]

        status = "✅ PASS" if pred_class == expected else "❌ FAIL"
        print(f"ID: {img_id} | Expected: {expected:8} | Pred: {pred_class:8} ({top_prob.item()*100:5.2f}%) | {status}")

if __name__ == "__main__":
    verify()
