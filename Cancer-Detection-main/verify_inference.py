import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import io
import numpy as np
import cv2
from image_preprocessing import preprocess_image
import matplotlib.pyplot as plt

# Config
MODEL_PATH = "saved_models/efficientnet_b0_skin_cancer.pth"
IMG_PATH = r"c:\Users\Usama\Downloads\Cancer-Detection-main\Cancer-Detection-main\balamced_dataset\images\train\bcc\ISIC_0024331.jpg" # Raw Sample
CLASSES = ['bcc', 'melanoma', 'scc']

def predict_single_image(img_path, model_path):
    print(f"Loading model from {model_path}...")
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pil_image = Image.open(img_path).convert("RGB")
    
    # Convert to BGR for cv2 preprocessing
    img_np_rgb = np.array(pil_image)
    img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

    # Preprocess
    processed_img_bgr = preprocess_image(img_np_bgr, 224)
    
    # Convert back to RGB for model
    processed_img_rgb = cv2.cvtColor(processed_img_bgr.astype('float32'), cv2.COLOR_BGR2RGB)

    input_tensor = transform(processed_img_rgb).unsqueeze(0)

    print(f"Running inference on {img_path}...")
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {probabilities[i].item()*100:.2f}%")

    top_prob, top_catid = torch.topk(probabilities, 1)
    print(f"\nPredicted Class: {CLASSES[top_catid]}")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(IMG_PATH):
        predict_single_image(IMG_PATH, MODEL_PATH)
    else:
        print("Model or Image not found.")
