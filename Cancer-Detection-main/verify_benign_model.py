import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import timm
import os

# Config
MODEL_PATH = "saved_models/efficientnet_b0_skin_cancer_4class.pth"
CLASSES = ['bcc', 'benign', 'melanoma', 'scc']
IMAGE_PATH = r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\benign"

# Get first benign image
images = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.jpg')]
if not images:
    print("No images found in benign directory.")
    exit()

test_image = os.path.join(IMAGE_PATH, images[0])

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

# Prediction
image = Image.open(test_image).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_class_idx = torch.max(probabilities, 0)

print(f"Test Image: {test_image}")
print(f"Top Prediction: {CLASSES[top_class_idx]} ({top_prob.item()*100:.2f}%)")
print("\nAll probabilities:")
for i, prob in enumerate(probabilities):
    print(f" - {CLASSES[i]:10}: {prob.item()*100:.2f}%")

if top_class_idx == 1: # Benign
    print("\n✅ Verification Successful: Benign image correctly classified!")
else:
    print("\n⚠️ Verification Warning: Image was not classified as benign, but the model has 4 outputs.")
