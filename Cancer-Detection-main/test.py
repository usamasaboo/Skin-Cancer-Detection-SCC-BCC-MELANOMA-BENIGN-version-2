# ==============================
# IMPORTS
# ==============================
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import albumentations as A
import timm
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================
# CONFIG
# ==============================
INPUT_DIRS = {
    "bcc": r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\bcc",
    "scc": r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\scc",
    "melanoma": r"C:\Users\Usama\Downloads\Cancer-Detection-main\balanced_dataset_1\balanced_dataset_1\images\train\melanoma"
}

OUTPUT_BASE = r"cleaned_images/train"
IMG_SIZE = 384
AUGMENT_PER_IMAGE = 2
BATCH_SIZE = 32
EPOCHS = 10
CSV_PATH = "balanced_3_trainval.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# IMAGE PREPROCESSING
# ==============================
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_image(img_path, augment=False, aug_pipeline=None, image_size=224):
    image = cv2.imread(img_path)
    if image is None:
        return None

    image = cv2.resize(image, (image_size, image_size))
    image = remove_hair(image)
    image = enhance_contrast(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    if augment and aug_pipeline:
        image = aug_pipeline(image=image)["image"]

    return image


# ==============================
# AUGMENTATION
# ==============================
def get_augmentation(image_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussianBlur(blur_limit=5, p=0.3),
        A.HueSaturationValue(10, 20, 10, p=0.4),
        A.CoarseDropout(
            max_holes=1,
            max_height=int(image_size * 0.3),
            max_width=int(image_size * 0.3),
            p=0.4
        )
    ])

# ==============================
# DATASET PREPROCESSING
# ==============================
def preprocess_dataset(input_dirs, output_dir, image_size=224, augment_per_image=2):
    os.makedirs(output_dir, exist_ok=True)
    aug_pipeline = get_augmentation(image_size)

    for label, folder in input_dirs.items():
        out_class_dir = os.path.join(output_dir, label)
        os.makedirs(out_class_dir, exist_ok=True)

        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"{label}: {len(images)} images found")

        for img_name in tqdm(images, desc=f"Cleaning {label}"):
            img_path = os.path.join(folder, img_name)
            base, ext = os.path.splitext(img_name)

            clean_img = preprocess_image(img_path, image_size=image_size)
            if clean_img is None:
                continue

            cv2.imwrite(
                os.path.join(out_class_dir, img_name),
                clean_img
            )

            for i in range(augment_per_image):
                aug_img = preprocess_image(
                    img_path,
                    augment=True,
                    aug_pipeline=aug_pipeline,
                    image_size=image_size
                )
                cv2.imwrite(
                    os.path.join(out_class_dir, f"{base}_aug{i}{ext}"),
                    aug_img
                )

    print("✅ Preprocessing complete!")

# ==============================
# PYTORCH DATASET
# ==============================
class SkinCancerDataset(Dataset):
    def __init__(self, csv_path, split):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.label_map = {"bcc": 0, "benign": 1, "melanoma": 2, "scc": 3}

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path"]
        label = self.label_map[self.df.loc[idx, "label"]]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label

# ==============================
# TRAINING FUNCTIONS
# ==============================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            targets.extend(labels.numpy())

    return accuracy_score(targets, preds), confusion_matrix(targets, preds)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    preprocess_dataset(
        input_dirs=INPUT_DIRS,
        output_dir=OUTPUT_BASE,
        image_size=IMG_SIZE,
        augment_per_image=AUGMENT_PER_IMAGE
    )

    train_ds = SkinCancerDataset(CSV_PATH, "train")
    val_ds   = SkinCancerDataset(CSV_PATH, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=4)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        acc, cm = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {loss:.4f}")
        print(f"Val Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)

    torch.save(model.state_dict(), "skin_cancer_classifier.pth")
    print("✅ Model saved!")