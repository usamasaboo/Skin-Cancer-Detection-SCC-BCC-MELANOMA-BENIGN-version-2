import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from pprint import pprint

# ===============================
# IMPORT CUSTOM MODULES
# ===============================
from image_preprocessing import preprocess_dataset
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
# ===============================
# DATA PREPROCESSING
# ===============================
INPUT_DIRS = {
    "bcc": r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\bcc",
    "scc": r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\scc",
    "melanoma": r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\melanoma",
    "benign": r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\benign"
}

OUTPUT_BASE = "cleaned_images/train"

# Note: We are re-running preprocessing to ensure it's consistent, but since we already did it, 
# it will just overwrite or re-process. In a real scenario we might check if it exists.
print("Starting Preprocessing...")
preprocess_dataset(
    input_dirs=INPUT_DIRS,
    output_base=OUTPUT_BASE,
    img_size=384,
    augment_per_image=2
)
print("Preprocessing Finished.")


# ===============================
# MODEL CONFIG (CLASSIFICATION)
# ===============================
# We are overriding the complex detection config with a clean Classification config
selected_model = {
    "model_name": "efficientnet_b4",
    "num_classes": 4,  # bcc, scc, melanoma, benign
    "input_size": 384,
    "epochs": 10,
    "lr": 0.0001,
    "train_batch_size": 8,  # Reduced for B4 memory
    "save_path": "./saved_models/"
}

print("\nSelected Model Config:")
pprint(selected_model)

# ===============================
# HYPERPARAMETERS
# ===============================
hyper_param = {
    "seed": 42,
    "image_size": selected_model["input_size"],
    "num_classes": selected_model["num_classes"],
    "epochs": selected_model["epochs"],
    "lr": selected_model["lr"],
    "train_batch_size": selected_model["train_batch_size"],
    "model_name": selected_model["model_name"],
    "save_path": selected_model["save_path"]
}

os.makedirs(hyper_param["save_path"], exist_ok=True)
torch.manual_seed(hyper_param["seed"])

# ===============================
# DATASET CLASS
# ===============================
class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        # Ensure consistent label ordering
        self.classes = sorted(os.listdir(root_dir))
        print(f"Detected classes: {self.classes}")

        for label_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
                
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                # Check for valid image extensions roughly
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((img_path, label_idx))
        
        print(f"Total samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ===============================
# TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((hyper_param["image_size"], hyper_param["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = SkinCancerDataset(OUTPUT_BASE, transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=hyper_param["train_batch_size"],
    shuffle=True
)

# ===============================
# MODEL (BACKBONE-BASED)
# ===============================
print(f"Creating model: {hyper_param['model_name']}")
model = timm.create_model(
    hyper_param["model_name"],
    pretrained=True,
    num_classes=hyper_param["num_classes"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_param["lr"])

# ===============================
# TRAINING LOOP
# ===============================
print("Starting training...")
for epoch in range(hyper_param["epochs"]):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate accuracy for this batch
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{hyper_param['epochs']}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    # SAVE MODEL after each epoch
    save_file = os.path.join(
        hyper_param["save_path"],
        f"{hyper_param['model_name']}_skin_cancer_4class.pth"
    )
    torch.save(model.state_dict(), save_file)
    print(f"Model saved after epoch {epoch+1}")

# ===============================
# SAVE MODEL
# ===============================
save_file = os.path.join(
    hyper_param["save_path"],
    f"{hyper_param['model_name']}_skin_cancer_4class.pth"
)

torch.save(model.state_dict(), save_file)
print(f"\n✅ Model saved at {save_file}")

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
