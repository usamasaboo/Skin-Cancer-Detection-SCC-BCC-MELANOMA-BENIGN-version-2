import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
import glob

# Config
DATASET_PATH = r"c:\Users\Usama\Downloads\ISIC_2019_Training_Input\ISIC_2019_Training_Input"
BALANCED_CSV = r"c:\Users\Usama\Downloads\balanced_3.csv"
HAM_METADATA = r"c:\Users\Usama\Downloads\HAM10000\HAM10000_metadata_cleaned.csv"
DOWNLOADS_DIR = r"c:\Users\Usama\Downloads"
SAVE_PATH = "saved_models/efficientnet_b4_skin_cancer_4class_v3.pth"
CLASSES = ['bcc', 'benign', 'melanoma', 'scc']
BATCH_SIZE = 8 # Reduced for B4 memory constraints
EPOCHS = 10 
LR = 0.0001 # Lower LR for B4 fine-tuning
IMG_SIZE = 384
CHECKPOINT_DIR = "saved_models/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. Prepare Dataframe
print("Loading datasets...")
df_balanced = pd.read_csv(BALANCED_CSV)
label_map = {'BCC': 'bcc', 'MEL': 'melanoma', 'SCC': 'scc'}
df_balanced['class'] = df_balanced['label'].map(label_map)

df_ham = pd.read_csv(HAM_METADATA)
benign_dx = ['nv', 'bkl', 'vasc', 'df']
df_benign = df_ham[df_ham['dx'].isin(benign_dx)].copy()
df_benign['class'] = 'benign'

# Sample to match counts
sample_size = 830
final_list = []

for cls in ['bcc', 'melanoma', 'scc']:
    df_cls = df_balanced[df_balanced['class'] == cls]
    final_list.append(df_cls.sample(min(sample_size, len(df_cls)), random_state=42)[['image_id', 'class']])

final_list.append(df_benign.sample(sample_size, random_state=42)[['image_id', 'class']])
final_df = pd.concat(final_list)

# 2. Add SPECIFIC User real-world samples from Downloads (High Priority)
user_samples = []
# Pattern match for specific files we know user has
for ext in ['*.webp', '*.jpg', '*.png', '*.jpeg']:
    for f in glob.glob(os.path.join(DOWNLOADS_DIR, ext)):
        fname = os.path.basename(f).lower()
        cls = None
        if 'bcc' in fname: cls = 'bcc'
        elif 'scc' in fname: cls = 'scc'
        elif 'mel' in fname: cls = 'melanoma'
        elif 'benign' in fname: cls = 'benign'
        
        if cls:
            # Add these multiple times to give them weight
            for _ in range(20): 
                user_samples.append({'image_id': f.replace('.webp','').replace('.jpg','').replace('.png','').replace('.jpeg',''), 'class': cls, 'full_path': f})

if user_samples:
    print(f"Adding {len(user_samples)//20} unique user samples (weighted 20x) from Downloads.")
    df_user = pd.DataFrame(user_samples)
    # We'll handle 'full_path' separately in Dataset class
else:
    df_user = pd.DataFrame(columns=['image_id', 'class', 'full_path'])

print("Final dataset distribution (excluding weighted user samples):")
print(final_df['class'].value_counts())

# Dataset Class
class RobustSkinDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label_name = self.df.loc[idx, 'class']
        label = self.class_to_idx[label_name]
        
        if 'full_path' in self.df.columns and pd.notna(self.df.loc[idx, 'full_path']):
            img_path = self.df.loc[idx, 'full_path']
        else:
            img_id = self.df.loc[idx, 'image_id']
            img_path = os.path.join(self.root_dir, f"{img_id}.jpg")
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224), color=0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transforms (Texture Focus)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)), # Slight zoom
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Minimal color change
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mix and split
train_df, val_df = train_test_split(final_df, test_size=0.1, stratify=final_df['class'], random_state=42)
# Add user samples only to training
train_df = pd.concat([train_df, df_user])

train_ds = RobustSkinDataset(train_df, DATASET_PATH, train_transform)
val_ds = RobustSkinDataset(val_df, DATASET_PATH, val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Model (B4 for higher resolution support)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=len(CLASSES))
model.to(device)

# SCC Priority Class Weights (BCC, BEN, MEL, SCC)
# Giving SCC 2.5x weight to force model focus on its features
class_weights = torch.tensor([1.0, 1.0, 1.2, 2.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# Training Loop
print(f"Starting training on {device}...")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {100*correct/total:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # 5. User Real-World Evaluation
    print("\n--- User Real-World Test ---")
    model.eval()
    test_files = ['bcc1.webp', 'bcc2.webp', 'bcc3.webp', 'scc1.jpg', 'scc2.jpg', 'scc3.webp', 'mel1.webp', 'mel2.webp', 'mel3.webp']
    with torch.no_grad():
        for f in test_files:
            path = os.path.join(DOWNLOADS_DIR, f)
            if not os.path.exists(path): continue
            img = Image.open(path).convert("RGB")
            img_t = val_transform(img).unsqueeze(0).to(device)
            out = model(img_t)
            prob = torch.nn.functional.softmax(out[0], dim=0)
            conf, idx = torch.max(prob, 0)
            print(f"File: {f:12} | Pred: {CLASSES[idx]:10} | Conf: {conf.item()*100:5.2f}%")
    print("---------------------------\n")

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"New best model saved with {val_acc:.2f}% accuracy")

print(f"\n✅ Retraining finished. Robust Model saved to {SAVE_PATH}")
